from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np

from ..audio import AudioDecodeError, coerce_to_bytes, decode_audio_bytes
from .formats import save_json_diarization, save_rttm, save_srt_with_speaker
from .models import (
    DiarizationResult,
    SpeakerAnnotatedTranscript,
    SpeakerSegment,
    SpeakerTurn,
)
from .options import DiarizeOptions
from .pipeline import annotation_to_turns, build_diarization_kwargs, load_pipeline
from .postprocess import attach_speaker_labels


def diarize_file(audio_path: str | Path, *, options: Optional[DiarizeOptions] = None) -> DiarizationResult:
    """音声ファイルから話者分離。"""

    opt = options or DiarizeOptions()
    pipeline = load_pipeline(opt)
    path = str(Path(audio_path))

    annotation = pipeline(path, **build_diarization_kwargs(opt))
    turns = annotation_to_turns(annotation)

    return DiarizationResult(filename=Path(path).name, duration=None, turns=turns)


def diarize_waveform(
    waveform: np.ndarray,
    *,
    sample_rate: int,
    display_name: str = "stream",
    options: Optional[DiarizeOptions] = None,
) -> DiarizationResult:
    """波形（-1..1 の float32, mono）から話者分離。"""

    opt = options or DiarizeOptions()
    pipeline = load_pipeline(opt)

    if waveform.ndim > 1:
        waveform = np.mean(waveform, axis=-1)

    annotation = pipeline({"waveform": waveform, "sample_rate": sample_rate}, **build_diarization_kwargs(opt))
    turns = annotation_to_turns(annotation)
    duration = float(len(waveform) / sample_rate)
    return DiarizationResult(filename=str(display_name), duration=duration, turns=turns)


def diarize_all(
    audio_paths: Iterable[str | Path],
    *,
    options: Optional[DiarizeOptions] = None,
) -> List[DiarizationResult]:
    """複数ファイルをまとめて話者分離。"""

    opt = options or DiarizeOptions()
    pipeline = load_pipeline(opt)
    kwargs = build_diarization_kwargs(opt)

    results: List[DiarizationResult] = []
    for audio_path in audio_paths:
        resolved = str(Path(audio_path))
        annotation = pipeline(resolved, **kwargs)
        turns = annotation_to_turns(annotation)
        results.append(DiarizationResult(filename=Path(resolved).name, turns=turns))
    return results


def diarize_all_bytes(
    audio_blobs: Iterable[bytes | bytearray | memoryview],
    *,
    names: Sequence[str] | None = None,
    options: Optional[DiarizeOptions] = None,
    sample_rate: int = 16000,
) -> List[DiarizationResult]:
    """メモリ上の音声データをまとめて話者分離（簡易デコード付）。"""

    opt = options or DiarizeOptions()
    pipeline = load_pipeline(opt)
    kwargs = build_diarization_kwargs(opt)
    name_overrides = list(names or [])

    results: List[DiarizationResult] = []
    for index, blob in enumerate(audio_blobs):
        audio_bytes = coerce_to_bytes(blob)
        display_name = name_overrides[index] if index < len(name_overrides) else f"stream_{index + 1}"
        try:
            waveform = decode_audio_bytes(audio_bytes, sample_rate=sample_rate)
        except AudioDecodeError as exc:
            raise _translate_decode_error("diarize_bytes", exc) from exc

        annotation = pipeline({"waveform": waveform, "sample_rate": sample_rate}, **kwargs)
        turns = annotation_to_turns(annotation)
        duration = float(len(waveform) / sample_rate)
        results.append(DiarizationResult(filename=display_name, duration=duration, turns=turns))
    return results


def _translate_decode_error(context: str, exc: AudioDecodeError) -> RuntimeError:
    if exc.kind == "ffmpeg-not-found":
        return RuntimeError(f"ffmpeg が見つかりません（{context}）")
    if exc.kind == "ffmpeg-error":
        detail = exc.detail or "unknown error"
        return RuntimeError(f"音声デコードに失敗しました（{context}）: {detail}")
    if exc.kind == "empty-output":
        return RuntimeError(f"音声デコード結果が空でした（{context}）。")
    if exc.kind == "empty-input":
        return RuntimeError(f"音声データが空でした（{context}）。")
    return RuntimeError(f"音声デコードに失敗しました（{context}）。")


__all__ = [
    "DiarizeOptions",
    "SpeakerTurn",
    "DiarizationResult",
    "SpeakerSegment",
    "SpeakerAnnotatedTranscript",
    "attach_speaker_labels",
    "save_rttm",
    "save_json_diarization",
    "save_srt_with_speaker",
    "diarize_file",
    "diarize_waveform",
    "diarize_all",
    "diarize_all_bytes",
]
