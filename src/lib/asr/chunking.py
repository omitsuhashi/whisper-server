from __future__ import annotations

import logging
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


def _merge_results(
    partials: Sequence[TranscriptionResult],
    *,
    chunk_windows: Sequence[ChunkWindow],
    filename: str,
    language: Optional[str],
) -> TranscriptionResult:
    segments: List[TranscriptionSegment] = []
    for res, (raw_start, _, main_start, main_end) in zip(partials, chunk_windows):
        offset_sec = raw_start / float(_SR)
        window_start = main_start / float(_SR)
        window_end = main_end / float(_SR)
        for seg in getattr(res, "segments", []) or []:
            seg_start = float(_seg_get(seg, "start") or 0.0) + offset_sec
            seg_end = float(_seg_get(seg, "end") or seg_start) + offset_sec
            clipped = _clip_segment(seg_start, seg_end, window_start, window_end)
            if clipped is None:
                continue
            new_start, new_end = clipped
            seg_text = str(_seg_get(seg, "text") or "")
            if segments:
                last = segments[-1]
                last_text = getattr(last, "text", "").strip()
                if last_text and last_text == seg_text.strip():
                    gap = float(new_start) - float(getattr(last, "end", new_start))
                    if gap <= 0.5:
                        continue
            segments.append(
                TranscriptionSegment.model_validate(
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
            )

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
) -> List[ChunkWindow]:
    """Phase 2 (Chunking): VAD 区間を優先してチャンク計画を立てる。"""

    if not segments:
        return _build_chunks(total_samples, chunk_samples, overlap_samples)

    planned: List[ChunkWindow] = []
    for segment in segments:
        seg_start = int(segment.start_sample)
        seg_end = int(segment.end_sample)
        if seg_end <= seg_start:
            continue
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
    segments = _phase_detect_segments(wave, _SR, vad_config=vad_cfg)
    chunk_windows = _phase_plan_chunking(total, chunk_samples, overlap_samples, segments)
    partials = _phase_run_asr_waveform(
        chunk_windows,
        wave,
        filename=name,
        options=options,
    )
    return _merge_results(partials, chunk_windows=chunk_windows, filename=name, language=options.language)


__all__ = ["transcribe_waveform_chunked"]
