from __future__ import annotations

import logging
from pathlib import Path
from subprocess import CalledProcessError, run
from typing import Any, Iterable, List, Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from mlx_whisper import transcribe
from mlx_whisper.audio import SAMPLE_RATE

logger = logging.getLogger(__name__)


class TranscriptionSegment(BaseModel):
    """Whisperが返す1区間分の書き起こし結果。"""

    model_config = ConfigDict(extra="ignore")

    id: int = 0
    seek: int | None = None
    start: float = 0.0
    end: float = 0.0
    text: str = ""
    tokens: List[int] | None = None
    temperature: float | None = None
    avg_logprob: float | None = None
    compression_ratio: float | None = None
    no_speech_prob: float | None = None


class TranscriptionResult(BaseModel):
    """書き起こし結果を扱いやすい形に正規化したモデル。"""

    model_config = ConfigDict(extra="ignore")

    filename: str
    text: str = ""
    language: str | None = None
    duration: float | None = None
    segments: List[TranscriptionSegment] = Field(default_factory=list)


def transcribe_all(
    audio_paths: Iterable[str | Path],
    *,
    model_name: str,
    language: str | None = None,
    task: str | None = None,
    **decode_options: Any,
) -> List[TranscriptionResult]:
    """複数ファイルをまとめて書き起こすヘルパー関数。"""

    resolved = [Path(path) for path in audio_paths]
    if not resolved:
        return []

    missing = [str(path) for path in resolved if not path.exists()]
    if missing:
        raise FileNotFoundError(f"存在しない音声ファイルがあります: {', '.join(missing)}")

    if not model_name:
        raise ValueError("model_name を指定してください。")

    transcribe_kwargs = _build_transcribe_kwargs(language, task, decode_options)
    results: List[TranscriptionResult] = []
    for path in resolved:
        logger.info(
            "音声ファイルを書き起こし中: %s (model=%s language=%s task=%s)",
            path.name,
            model_name,
            language,
            task,
        )
        results.append(
            _transcribe_single(
                audio_input=str(path),
                display_name=path,
                model_name=model_name,
                transcribe_kwargs=transcribe_kwargs,
            )
        )

    return results


def transcribe_all_bytes(
    audio_blobs: Iterable[bytes | bytearray | memoryview],
    *,
    model_name: str,
    language: str | None = None,
    task: str | None = None,
    names: Sequence[str] | None = None,
    **decode_options: Any,
) -> List[TranscriptionResult]:
    """メモリ上の音声データを書き起こすヘルパー関数。"""

    transcribe_kwargs = _build_transcribe_kwargs(language, task, decode_options)
    name_overrides = list(names or [])

    results: List[TranscriptionResult] = []
    for index, blob in enumerate(audio_blobs):
        audio_bytes = _coerce_to_bytes(blob)
        display_name = (
            name_overrides[index] if index < len(name_overrides) else f"stream_{index + 1}"
        )
        logger.info(
            "音声ストリームを書き起こし中: %s (model=%s language=%s task=%s)",
            display_name,
            model_name,
            language,
            task,
        )
        waveform = _decode_audio_bytes(audio_bytes)
        results.append(
            _transcribe_single(
                audio_input=waveform,
                display_name=display_name,
                model_name=model_name,
                transcribe_kwargs=transcribe_kwargs,
            )
        )

    return results


def _build_transcription_result(source: Path | str, payload: dict[str, Any]) -> TranscriptionResult:
    """mlx_whisperの戻り値をTranscriptionResultへ変換する。"""

    segments_raw = payload.get("segments", []) or []
    segments = [TranscriptionSegment.model_validate(segment) for segment in segments_raw]
    filename = source.name if isinstance(source, Path) else Path(str(source)).name
    return TranscriptionResult(
        filename=filename,
        text=payload.get("text", ""),
        language=payload.get("language"),
        duration=payload.get("duration"),
        segments=segments,
    )


def _build_transcribe_kwargs(
    language: str | None,
    task: str | None,
    decode_options: dict[str, Any],
) -> dict[str, Any]:
    """transcribe関数へ渡すオプション辞書を組み立てる。"""

    transcribe_kwargs: dict[str, Any] = dict(decode_options)
    if language:
        transcribe_kwargs["language"] = language
    if task:
        transcribe_kwargs["task"] = task
    return transcribe_kwargs


def _transcribe_single(
    *,
    audio_input: Any,
    display_name: Path | str,
    model_name: str,
    transcribe_kwargs: dict[str, Any],
) -> TranscriptionResult:
    """transcribeを呼び出しTranscriptionResultへ変換する。"""

    raw_result = transcribe(audio_input, path_or_hf_repo=model_name, **transcribe_kwargs)
    return _build_transcription_result(display_name, raw_result)


def _coerce_to_bytes(blob: bytes | bytearray | memoryview) -> bytes:
    """各種バイト列を生のbytesに統一する。"""

    if isinstance(blob, bytes):
        return blob
    if isinstance(blob, bytearray):
        return bytes(blob)
    return bytes(memoryview(blob))


def _decode_audio_bytes(audio_bytes: bytes) -> np.ndarray:
    """音声ファイルのバイト列を波形へデコードする。"""

    if not audio_bytes:
        raise ValueError("音声データが空です。")

    cmd = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        "pipe:0",
        "-threads",
        "0",
        "-f",
        "s16le",
        "-ac",
        "1",
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(SAMPLE_RATE),
        "-",
    ]
    try:
        completed = run(cmd, input=audio_bytes, capture_output=True, check=True)
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg が見つかりません。音声データをデコードできませんでした。") from exc
    except CalledProcessError as exc:
        stderr = exc.stderr.decode(errors="ignore") if exc.stderr else "unknown error"
        raise RuntimeError(f"音声データのデコードに失敗しました: {stderr}") from exc

    waveform = np.frombuffer(completed.stdout, dtype=np.int16).astype(np.float32)
    if waveform.size == 0:
        raise RuntimeError("音声データのデコード結果が空でした。")
    return waveform / 32768.0


__all__ = [
    "TranscriptionResult",
    "TranscriptionSegment",
    "transcribe_all",
    "transcribe_all_bytes",
]
