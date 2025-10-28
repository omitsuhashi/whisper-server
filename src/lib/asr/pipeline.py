from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterable, List, Sequence

import numpy as np

from mlx_whisper import transcribe

from ..audio import AudioDecodeError, coerce_to_bytes, decode_audio_bytes, is_silent_audio
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
        try:
            if is_silent_audio(path):
                logger.info("silence_detected: %s", path.name)
                results.append(
                    _build_silence_result(
                        display_name=path,
                        language=options.language,
                    )
                )
                continue
        except Exception:  # noqa: BLE001 - 検出失敗時は通常フローへ
            logger.debug("silence_detection_failed", exc_info=True)

        results.append(
            _transcribe_single(
                audio_input=str(path),
                display_name=path,
                model_name=options.model_name,
                transcribe_kwargs=transcribe_kwargs,
                language_hint=options.language,
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

        if _is_waveform_silent(waveform):
            logger.info("silence_detected: %s", display_name)
            results.append(
                _build_silence_result(
                    display_name=display_name,
                    language=options.language,
                )
            )
            continue

        results.append(
            _transcribe_single(
                audio_input=waveform,
                display_name=display_name,
                model_name=options.model_name,
                transcribe_kwargs=transcribe_kwargs,
                language_hint=options.language,
            )
        )
    return results


def _transcribe_single(
    *,
    audio_input: Any,
    display_name: Path | str,
    model_name: str,
    transcribe_kwargs: dict[str, Any],
    language_hint: str | None,
) -> TranscriptionResult:
    """transcribeを呼び出しTranscriptionResultへ変換する。"""

    raw_result = transcribe(audio_input, path_or_hf_repo=model_name, **transcribe_kwargs)
    if _should_force_silence(raw_result):
        logger.info("silence_by_model: %s", display_name)
        return _build_silence_result(
            display_name=display_name,
            language=raw_result.get("language") or language_hint,
        )
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


def _is_waveform_silent(waveform: np.ndarray, *, threshold: float = 2e-4) -> bool:
    if waveform.size == 0:
        return True
    energy = float(np.mean(np.abs(waveform)))
    peak = float(np.max(np.abs(waveform))) if waveform.size else 0.0
    return energy < threshold and peak < threshold * 5


def _build_silence_result(*, display_name: Path | str, language: str | None) -> TranscriptionResult:
    filename = display_name.name if isinstance(display_name, Path) else Path(str(display_name)).name
    return TranscriptionResult(
        filename=filename,
        text="",
        language=language,
        duration=0.0,
        segments=[],
    )


def _should_force_silence(raw_result: dict[str, Any]) -> bool:
    text = (raw_result.get("text") or "").strip()
    if not text:
        return False

    segments = raw_result.get("segments") or []
    if not isinstance(segments, list) or not segments:
        return False

    scores: list[float] = []
    for segment in segments:
        if isinstance(segment, dict):
            prob = segment.get("no_speech_prob")
            if isinstance(prob, (int, float)):
                scores.append(float(prob))
    if not scores:
        return False

    max_score = max(scores)
    avg_score = sum(scores) / len(scores)
    return max_score >= 0.6 or avg_score >= 0.5


__all__ = [
    "transcribe_paths",
    "transcribe_streams",
]
