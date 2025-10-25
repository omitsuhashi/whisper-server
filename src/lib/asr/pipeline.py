from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterable, List, Sequence

from mlx_whisper import transcribe

from ..audio import AudioDecodeError, coerce_to_bytes, decode_audio_bytes
from .converters import build_transcription_result
from .models import TranscriptionResult
from .options import TranscribeOptions

logger = logging.getLogger(__name__)


def transcribe_paths(audio_paths: Iterable[str | Path], *, options: TranscribeOptions) -> List[TranscriptionResult]:
    """複数ファイルをまとめて書き起こす。"""

    resolved = [Path(path) for path in audio_paths]
    if not resolved:
        return []

    missing = [str(path) for path in resolved if not path.exists()]
    if missing:
        raise FileNotFoundError(f"存在しない音声ファイルがあります: {', '.join(missing)}")

    if not options.model_name:
        raise ValueError("model_name を指定してください。")

    transcribe_kwargs = options.build_transcribe_kwargs()
    results: List[TranscriptionResult] = []
    for path in resolved:
        logger.debug(
            "音声ファイルを書き起こし中: %s (model=%s language=%s task=%s)",
            path.name,
            options.model_name,
            options.language,
            options.task,
        )
        results.append(
            _transcribe_single(
                audio_input=str(path),
                display_name=path,
                model_name=options.model_name,
                transcribe_kwargs=transcribe_kwargs,
            )
        )
    return results


def transcribe_streams(
    audio_blobs: Iterable[bytes | bytearray | memoryview],
    *,
    options: TranscribeOptions,
    names: Sequence[str] | None = None,
    sample_rate: int,
) -> List[TranscriptionResult]:
    """メモリ上の音声データを書き起こす。"""

    transcribe_kwargs = options.build_transcribe_kwargs()
    name_overrides = list(names or [])

    results: List[TranscriptionResult] = []
    for index, blob in enumerate(audio_blobs):
        audio_bytes = coerce_to_bytes(blob)
        display_name = name_overrides[index] if index < len(name_overrides) else f"stream_{index + 1}"
        logger.debug(
            "音声ストリームを書き起こし中: %s (model=%s language=%s task=%s)",
            display_name,
            options.model_name,
            options.language,
            options.task,
        )
        try:
            waveform = decode_audio_bytes(audio_bytes, sample_rate=sample_rate)
        except AudioDecodeError as exc:
            raise _translate_decode_error(exc) from exc

        results.append(
            _transcribe_single(
                audio_input=waveform,
                display_name=display_name,
                model_name=options.model_name,
                transcribe_kwargs=transcribe_kwargs,
            )
        )
    return results


def _transcribe_single(
    *,
    audio_input: Any,
    display_name: Path | str,
    model_name: str,
    transcribe_kwargs: dict[str, Any],
) -> TranscriptionResult:
    """transcribeを呼び出しTranscriptionResultへ変換する。"""

    raw_result = transcribe(audio_input, path_or_hf_repo=model_name, **transcribe_kwargs)
    return build_transcription_result(display_name, raw_result)


def _translate_decode_error(exc: AudioDecodeError) -> RuntimeError:
    if exc.kind == "ffmpeg-not-found":
        return RuntimeError("ffmpeg が見つかりません。音声データをデコードできませんでした。")
    if exc.kind == "ffmpeg-error":
        detail = exc.detail or "unknown error"
        return RuntimeError(f"音声データのデコードに失敗しました: {detail}")
    if exc.kind == "empty-output":
        return RuntimeError("音声データのデコード結果が空でした。")
    if exc.kind == "empty-input":
        return RuntimeError("音声データが空でした。")
    return RuntimeError("音声データのデコードに失敗しました。")


__all__ = [
    "transcribe_paths",
    "transcribe_streams",
]
