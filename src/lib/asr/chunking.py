from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ..audio import decode_audio_bytes, encode_waveform_to_wav_bytes
from .models import TranscriptionResult, TranscriptionSegment
from .main import transcribe_all_bytes as _default_transcribe_all_bytes

# Whisper の標準サンプルレート（16kHz）
_SR = 16000

TranscribeBytesFn = Callable[
    [Iterable[bytes | bytearray | memoryview], str, Optional[str], Optional[str], Optional[Sequence[str]]],
    List[TranscriptionResult],
]


def _build_chunks(length: int, chunk_samples: int, overlap_samples: int) -> List[Tuple[int, int, int, int]]:
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


def _merge_results(
    partials: Sequence[TranscriptionResult],
    *,
    chunk_windows: Sequence[Tuple[int, int, int, int]],
    filename: str,
    language: Optional[str],
) -> TranscriptionResult:
    segments: List[TranscriptionSegment] = []
    for res, (raw_start, _, main_start, main_end) in zip(partials, chunk_windows):
        offset_sec = raw_start / float(_SR)
        window_start = main_start / float(_SR)
        window_end = main_end / float(_SR)
        for seg in getattr(res, "segments", []) or []:
            seg_start = float(getattr(seg, "start", 0.0) or 0.0) + offset_sec
            seg_end = float(getattr(seg, "end", seg_start) or seg_start) + offset_sec
            clipped = _clip_segment(seg_start, seg_end, window_start, window_end)
            if clipped is None:
                continue
            new_start, new_end = clipped
            seg_text = getattr(seg, "text", "")
            if segments:
                last = segments[-1]
                last_text = getattr(last, "text", "").strip()
                if last_text and last_text == seg_text.strip():
                    gap = float(new_start) - float(getattr(last, "end", new_start))
                    if gap <= 0.5:
                        continue
            segments.append(
                TranscriptionSegment.model_validate(
                    {"start": new_start, "end": new_end, "text": seg_text}
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


def transcribe_paths_chunked(
    audio_paths: Iterable[str | Path],
    *,
    model_name: str,
    language: Optional[str],
    task: Optional[str],
    chunk_seconds: float = 25.0,
    overlap_seconds: float = 1.0,
    transcribe_all_bytes_fn: Optional[TranscribeBytesFn] = None,
) -> List[TranscriptionResult]:
    """ファイル入力をチャンク化し、オーバーラップを除去しながら結合する。"""

    fn = transcribe_all_bytes_fn or _default_transcribe_all_bytes
    results: List[TranscriptionResult] = []

    chunk_seconds = max(float(chunk_seconds or 0.0), 0.0)
    overlap_seconds = max(float(overlap_seconds or 0.0), 0.0)

    for path_like in audio_paths:
        path = Path(path_like)
        raw = path.read_bytes()
        waveform = decode_audio_bytes(raw, sample_rate=_SR)
        if waveform.ndim > 1:
            waveform = np.mean(waveform, axis=-1)
        total = int(waveform.shape[-1])

        if chunk_seconds <= 0.0 or total <= int(_SR * chunk_seconds):
            partials = fn([raw], model_name=model_name, language=language, task=task, names=[path.name])
            if partials:
                res = partials[0]
                if hasattr(res, "model_copy"):
                    res = res.model_copy(update={"filename": path.name})
                else:
                    setattr(res, "filename", path.name)
                results.append(res)
            continue

        chunk_samples = max(int(_SR * chunk_seconds), 1)
        overlap_samples = int(_SR * overlap_seconds)
        chunk_windows = _build_chunks(total, chunk_samples, overlap_samples)

        blobs: List[bytes] = []
        for raw_start, raw_end, _, _ in chunk_windows:
            chunk_wave = waveform[raw_start:raw_end]
            blobs.append(encode_waveform_to_wav_bytes(chunk_wave, sample_rate=_SR))

        chunk_names = [f"{path.name}#chunk{idx+1}" for idx in range(len(chunk_windows))]
        partials = fn(blobs, model_name=model_name, language=language, task=task, names=chunk_names)
        merged = _merge_results(partials, chunk_windows=chunk_windows, filename=path.name, language=language)
        results.append(merged)

    return results


__all__ = ["transcribe_paths_chunked"]
