from __future__ import annotations

from difflib import SequenceMatcher
import logging
import os
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

from ..vad import SpeechSegment, VadConfig, detect_voice_segments, resolve_vad_config
from .models import TranscriptionResult, TranscriptionSegment
from .options import TranscribeOptions
from .pipeline import _build_silence_result, _is_waveform_silent, _transcribe_single, transcribe_waveform

# Whisper の標準サンプルレート（16kHz）
_SR = 16000

logger = logging.getLogger(__name__)

ChunkWindow = Tuple[int, int, int, int]
_DEFAULT_VAD_CONFIG = resolve_vad_config()
_DEFAULT_DUPLICATE_GAP_SECONDS = 0.5
_CONTAINED_MATCH_EDGE_MARGIN_SECONDS = 0.35


def _build_chunks(length: int, chunk_samples: int, overlap_samples: int) -> List[ChunkWindow]:
    """チャンクとオーバーラップを考慮した領域リストを返す。

    戻り値は (raw_start, raw_end, main_start, main_end) のタプル。
    raw_* はオーバーラップを含む範囲、main_* は重複を除いた本体部分。
    """

    if chunk_samples <= 0 or length <= 0:
        return [(0, length, 0, length)]

    overlap_samples = max(0, min(overlap_samples, chunk_samples // 2))
    starts = list(range(0, length, chunk_samples))
    chunks: List[Tuple[int, int, int, int]] = []
    for idx, main_start in enumerate(starts):
        main_end = min(main_start + chunk_samples, length)
        raw_start = max(0, main_start - overlap_samples if idx > 0 else main_start)
        raw_end = min(length, main_end + (overlap_samples if main_end < length else 0))
        if raw_end <= raw_start:
            continue
        chunks.append((raw_start, raw_end, main_start, main_end))
        if main_end >= length:
            break
    if not chunks:
        chunks.append((0, length, 0, length))
    return chunks


def _clip_segment(start: float, end: float, window_start: float, window_end: float) -> Optional[Tuple[float, float]]:
    if end <= window_start or start >= window_end:
        return None
    new_start = max(start, window_start)
    new_end = min(end, window_end)
    if new_end <= new_start:
        return None
    return new_start, new_end


def _seg_get(seg: Any, key: str) -> Any:
    if isinstance(seg, dict):
        return seg.get(key)
    return getattr(seg, key, None)


def _to_float(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)):
        as_float = float(value)
        if np.isfinite(as_float):
            return as_float
    return None


def _normalize_text(text: str) -> str:
    return "".join((text or "").strip().split())


def _text_similarity(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    if left == right:
        return 1.0
    return float(SequenceMatcher(a=left, b=right).ratio())


def _segment_edge_margin(start: float, end: float, *, window_start: float, window_end: float) -> float:
    left = max(0.0, float(start) - float(window_start))
    right = max(0.0, float(window_end) - float(end))
    return min(left, right)


def _segment_relation(
    prev: TranscriptionSegment,
    new: TranscriptionSegment,
    *,
    allow_contained_match: bool,
) -> tuple[bool, float]:
    prev_start = float(getattr(prev, "start", 0.0))
    prev_end = float(getattr(prev, "end", prev_start))
    new_start = float(getattr(new, "start", 0.0))
    new_end = float(getattr(new, "end", new_start))

    gap = new_start - prev_end
    overlap = min(prev_end, new_end) - max(prev_start, new_start)
    prev_dur = max(prev_end - prev_start, 1e-6)
    new_dur = max(new_end - new_start, 1e-6)
    overlap_ratio = max(0.0, overlap) / max(min(prev_dur, new_dur), 1e-6)

    prev_text = _normalize_text(getattr(prev, "text", "") or "")
    new_text = _normalize_text(getattr(new, "text", "") or "")
    sim = _text_similarity(prev_text, new_text)
    contained = bool(prev_text and new_text and (prev_text in new_text or new_text in prev_text))

    duplicate_like = False
    if prev_text and new_text:
        if prev_text == new_text and gap <= _DEFAULT_DUPLICATE_GAP_SECONDS:
            duplicate_like = True
        elif sim >= 0.72 and gap <= 0.35:
            duplicate_like = True
        elif allow_contained_match and contained and gap <= 0.35:
            duplicate_like = True
        elif overlap_ratio >= 0.8 and gap <= 0.2:
            duplicate_like = True
    return duplicate_like, sim


def _segment_quality_score(segment: TranscriptionSegment, *, edge_margin: float) -> tuple[float, int, float, float, float, float]:
    avg_logprob = _to_float(getattr(segment, "avg_logprob", None))
    no_speech = _to_float(getattr(segment, "no_speech_prob", None))
    comp_ratio = _to_float(getattr(segment, "compression_ratio", None))
    start = float(getattr(segment, "start", 0.0))
    end = float(getattr(segment, "end", start))
    duration = max(0.0, end - start)
    metric_count = int(avg_logprob is not None) + int(no_speech is not None) + int(comp_ratio is not None)
    logprob_score = avg_logprob if avg_logprob is not None else -99.0
    no_speech_score = -no_speech if no_speech is not None else -99.0
    comp_score = -abs(comp_ratio - 1.0) if comp_ratio is not None else -99.0
    return (
        float(edge_margin),
        metric_count,
        logprob_score,
        no_speech_score,
        comp_score,
        duration,
    )


def _prefer_new_segment(
    prev: TranscriptionSegment,
    prev_edge_margin: float,
    new: TranscriptionSegment,
    new_edge_margin: float,
) -> bool:
    prev_score = _segment_quality_score(prev, edge_margin=prev_edge_margin)
    new_score = _segment_quality_score(new, edge_margin=new_edge_margin)
    if new_score != prev_score:
        return new_score > prev_score
    return float(getattr(new, "end", 0.0)) >= float(getattr(prev, "end", 0.0))


def _merge_results(
    partials: Sequence[TranscriptionResult],
    *,
    chunk_windows: Sequence[ChunkWindow],
    filename: str,
    language: Optional[str],
) -> TranscriptionResult:
    segments_with_margin: List[tuple[TranscriptionSegment, float]] = []
    total_input = 0
    total_clipped = 0
    total_kept = 0
    total_dedup_skipped = 0
    total_dedup_replaced = 0
    for idx, (res, (raw_start, raw_end, main_start, main_end)) in enumerate(
        zip(partials, chunk_windows),
        start=1,
    ):
        offset_sec = raw_start / float(_SR)
        raw_end_sec = raw_end / float(_SR)
        window_start = main_start / float(_SR)
        window_end = main_end / float(_SR)
        window_input = 0
        window_clipped = 0
        window_kept = 0
        window_dedup_skipped = 0
        window_dedup_replaced = 0
        for seg in getattr(res, "segments", []) or []:
            total_input += 1
            window_input += 1
            seg_start = float(_seg_get(seg, "start") or 0.0) + offset_sec
            seg_end = float(_seg_get(seg, "end") or seg_start) + offset_sec
            clipped = _clip_segment(seg_start, seg_end, window_start, window_end)
            if clipped is None:
                total_clipped += 1
                window_clipped += 1
                continue
            new_start, new_end = clipped
            seg_text = str(_seg_get(seg, "text") or "")
            new_segment = TranscriptionSegment.model_validate(
                {
                    "start": new_start,
                    "end": new_end,
                    "text": seg_text,
                    "avg_logprob": _seg_get(seg, "avg_logprob"),
                    "compression_ratio": _seg_get(seg, "compression_ratio"),
                    "no_speech_prob": _seg_get(seg, "no_speech_prob"),
                    "temperature": _seg_get(seg, "temperature"),
                }
            )
            edge_margin = _segment_edge_margin(
                new_start,
                new_end,
                window_start=window_start,
                window_end=window_end,
            )
            if segments_with_margin:
                last, last_edge_margin = segments_with_margin[-1]
                allow_contained_match = (
                    last_edge_margin <= _CONTAINED_MATCH_EDGE_MARGIN_SECONDS
                    or edge_margin <= _CONTAINED_MATCH_EDGE_MARGIN_SECONDS
                )
                duplicate_like, similarity = _segment_relation(
                    last,
                    new_segment,
                    allow_contained_match=allow_contained_match,
                )
                if duplicate_like:
                    replaced = _prefer_new_segment(last, last_edge_margin, new_segment, edge_margin)
                    if replaced:
                        segments_with_margin[-1] = (new_segment, edge_margin)
                        total_dedup_replaced += 1
                        window_dedup_replaced += 1
                    else:
                        total_dedup_skipped += 1
                        window_dedup_skipped += 1
                    logger.debug(
                        "chunk_merge_dedup window=%d similarity=%.3f replaced=%s",
                        idx,
                        similarity,
                        "1" if replaced else "0",
                    )
                    if not replaced:
                        continue
                    # 置換済みのため append は不要
                    continue
            segments_with_margin.append((new_segment, edge_margin))
            total_kept += 1
            window_kept += 1
        logger.debug(
            "chunk_merge_window index=%d raw_start=%.3f raw_end=%.3f main_start=%.3f main_end=%.3f input=%d clipped=%d kept=%d dedup_skipped=%d dedup_replaced=%d",
            idx,
            offset_sec,
            raw_end_sec,
            window_start,
            window_end,
            window_input,
            window_clipped,
            window_kept,
            window_dedup_skipped,
            window_dedup_replaced,
        )

    logger.debug(
        "chunk_merge_summary windows=%d input=%d clipped=%d kept=%d dedup_skipped=%d dedup_replaced=%d",
        len(chunk_windows),
        total_input,
        total_clipped,
        total_kept,
        total_dedup_skipped,
        total_dedup_replaced,
    )

    segments: List[TranscriptionSegment] = [seg for seg, _ in segments_with_margin]

    segments.sort(key=lambda s: (float(getattr(s, "start", 0.0)), float(getattr(s, "end", 0.0))))
    text_parts: List[str] = []
    for seg in segments:
        seg_text = getattr(seg, "text", "")
        if not seg_text:
            continue
        text_parts.append(seg_text)
    combined_text = "".join(text_parts).strip()
    duration = segments[-1].end if segments else None

    if not segments:
        all_silent = bool(partials) and all(
            not (getattr(res, "segments", None) or [])
            and not (getattr(res, "text", "") or "").strip()
            and getattr(res, "duration", None) == 0.0
            for res in partials
        )
        if all_silent:
            duration = 0.0
        else:
            fallback_text = "".join(res.text or "" for res in partials).strip()
            if fallback_text:
                combined_text = fallback_text
            if chunk_windows:
                duration = chunk_windows[-1][3] / float(_SR)

    return TranscriptionResult(
        filename=filename,
        text=combined_text,
        language=language,
        duration=duration,
        segments=segments,
    )


def _phase_detect_segments(
    waveform: np.ndarray,
    sample_rate: int,
    *,
    vad_config: VadConfig,
) -> List[SpeechSegment]:
    """Phase 1 (VAD): 波形から音声セグメントを抽出する。"""

    try:
        segments = detect_voice_segments(waveform, sample_rate, config=vad_config)
    except Exception:  # noqa: BLE001 - VAD 失敗時は従来チャンクへフォールバックする
        logger.debug("vad_detection_failed", exc_info=True)
        return []
    if not segments:
        logger.debug("vad_no_segments_detected")
    return segments


def _phase_plan_chunking(
    total_samples: int,
    chunk_samples: int,
    overlap_samples: int,
    segments: Sequence[SpeechSegment],
    *,
    vad_margin_samples: int = 0,
) -> List[ChunkWindow]:
    """Phase 2 (Chunking): VAD 区間を優先してチャンク計画を立てる。"""

    if not segments:
        return _build_chunks(total_samples, chunk_samples, overlap_samples)

    margin = max(int(vad_margin_samples), 0)
    expanded_ranges: List[Tuple[int, int]] = []
    for segment in sorted(segments, key=lambda x: int(x.start_sample)):
        seg_start = max(0, int(segment.start_sample) - margin)
        seg_end = min(total_samples, int(segment.end_sample) + margin)
        if seg_end <= seg_start:
            continue
        if expanded_ranges and seg_start <= expanded_ranges[-1][1]:
            prev_start, prev_end = expanded_ranges[-1]
            expanded_ranges[-1] = (prev_start, max(prev_end, seg_end))
            continue
        expanded_ranges.append((seg_start, seg_end))

    planned: List[ChunkWindow] = []
    for seg_start, seg_end in expanded_ranges:
        seg_length = seg_end - seg_start
        local_chunks = _build_chunks(seg_length, chunk_samples, overlap_samples)
        for raw_start, raw_end, main_start, main_end in local_chunks:
            offset = seg_start
            planned.append(
                (
                    raw_start + offset,
                    raw_end + offset,
                    main_start + offset,
                    main_end + offset,
                )
            )
    if planned:
        return planned
    return _build_chunks(total_samples, chunk_samples, overlap_samples)




def _phase_run_asr_waveform(
    chunk_windows: Sequence[ChunkWindow],
    waveform: np.ndarray,
    *,
    filename: str,
    options: TranscribeOptions,
) -> List[TranscriptionResult]:
    """Phase 3 (ASR): 波形チャンクを直接 ASR へ渡す。"""

    results: List[TranscriptionResult] = []
    transcribe_kwargs = options.build_transcribe_kwargs()
    for idx, (raw_start, raw_end, _, _) in enumerate(chunk_windows):
        chunk_wave = waveform[raw_start:raw_end]
        chunk_name = f"{filename}#chunk{idx+1}"
        if _is_waveform_silent(chunk_wave):
            results.append(
                _build_silence_result(
                    display_name=chunk_name,
                    language=options.language,
                )
            )
            continue
        results.append(
            _transcribe_single(
                audio_input=chunk_wave,
                display_name=chunk_name,
                model_name=options.model_name,
                transcribe_kwargs=transcribe_kwargs,
                language_hint=options.language,
            )
        )
    return results


def transcribe_waveform_chunked(
    waveform: np.ndarray,
    *,
    options: TranscribeOptions,
    name: str,
    chunk_seconds: float = 25.0,
    overlap_seconds: float = 1.0,
    vad_config: VadConfig | None = None,
) -> TranscriptionResult:
    """Waveform 入力をチャンク化し、オーバーラップを除去しながら結合する。"""

    vad_cfg = vad_config or _DEFAULT_VAD_CONFIG

    wave = waveform
    if wave.ndim > 1:
        wave = np.mean(wave, axis=-1)
    total = int(wave.shape[-1])

    chunk_seconds = max(float(chunk_seconds or 0.0), 0.0)
    overlap_seconds = max(float(overlap_seconds or 0.0), 0.0)
    if chunk_seconds <= 0.0 or total <= int(_SR * chunk_seconds):
        return transcribe_waveform(
            wave,
            options=options,
            name=name,
        )

    chunk_samples = max(int(_SR * chunk_seconds), 1)
    overlap_samples = int(_SR * overlap_seconds)
    raw_margin = os.getenv("ASR_VAD_BOUNDARY_MARGIN_SECONDS")
    if raw_margin is None:
        vad_margin_samples = max(overlap_samples, 0)
    else:
        try:
            margin_seconds = float(raw_margin)
        except ValueError:
            margin_seconds = overlap_seconds
        if not np.isfinite(margin_seconds) or margin_seconds < 0:
            margin_seconds = overlap_seconds
        vad_margin_samples = max(int(_SR * margin_seconds), 0)
    segments = _phase_detect_segments(wave, _SR, vad_config=vad_cfg)
    chunk_windows = _phase_plan_chunking(
        total,
        chunk_samples,
        overlap_samples,
        segments,
        vad_margin_samples=vad_margin_samples,
    )
    logger.debug(
        "chunk_plan_summary filename=%s total_samples=%d vad_segments=%d chunk_windows=%d chunk_samples=%d overlap_samples=%d vad_margin_samples=%d",
        name,
        total,
        len(segments),
        len(chunk_windows),
        chunk_samples,
        overlap_samples,
        vad_margin_samples,
    )
    partials = _phase_run_asr_waveform(
        chunk_windows,
        wave,
        filename=name,
        options=options,
    )
    return _merge_results(partials, chunk_windows=chunk_windows, filename=name, language=options.language)


__all__ = ["transcribe_waveform_chunked"]
