from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Sequence

from mlx_whisper.audio import SAMPLE_RATE

from .models import TranscriptionResult, TranscriptionSegment
from .options import TranscribeOptions
from .pipeline import transcribe_paths, transcribe_streams


def transcribe_all(
    audio_paths: Iterable[str | Path],
    *,
    model_name: str,
    language: str | None = None,
    task: str | None = None,
    **decode_options,
) -> List[TranscriptionResult]:
    """複数ファイルをまとめて書き起こすヘルパー関数。"""

    options = TranscribeOptions(
        model_name=model_name,
        language=language,
        task=task,
        decode_options=dict(decode_options),
    )
    return transcribe_paths(audio_paths, options=options)


def transcribe_all_bytes(
    audio_blobs: Iterable[bytes | bytearray | memoryview],
    *,
    model_name: str,
    language: str | None = None,
    task: str | None = None,
    names: Sequence[str] | None = None,
    **decode_options,
) -> List[TranscriptionResult]:
    """メモリ上の音声データを書き起こすヘルパー関数。"""

    options = TranscribeOptions(
        model_name=model_name,
        language=language,
        task=task,
        decode_options=dict(decode_options),
    )
    return transcribe_streams(
        audio_blobs,
        options=options,
        names=names,
        sample_rate=SAMPLE_RATE,
    )


__all__ = [
    "TranscriptionResult",
    "TranscriptionSegment",
    "TranscribeOptions",
    "transcribe_all",
    "transcribe_all_bytes",
]
