from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from mlx_whisper import transcribe

from ..diagnostics.memlog import snapshot as _memsnap
from .converters import build_transcription_result
from .models import TranscriptionResult
from .options import TranscribeOptions

logger = logging.getLogger(__name__)


def transcribe_waveform(
    waveform: np.ndarray,
    *,
    options: TranscribeOptions,
    name: str,
) -> TranscriptionResult:
    """Waveform を直接書き起こす。"""

    display_name = name or "waveform"
    logger.debug(
        "waveformを書き起こし中: %s (model=%s language=%s task=%s)",
        display_name,
        options.model_name,
        options.language,
        options.task,
    )

    transcribe_kwargs = options.build_transcribe_kwargs()
    if _is_waveform_silent(waveform):
        logger.info("silence_detected: %s", display_name)
        return _build_silence_result(
            display_name=display_name,
            language=options.language,
        )
    return _transcribe_single(
        audio_input=waveform,
        display_name=display_name,
        model_name=options.model_name,
        transcribe_kwargs=transcribe_kwargs,
        language_hint=options.language,
    )


def _transcribe_single(
    *,
    audio_input: Any,
    display_name: Path | str,
    model_name: str,
    transcribe_kwargs: dict[str, Any],
    language_hint: str | None,
) -> TranscriptionResult:
    """transcribeを呼び出しTranscriptionResultへ変換する。"""

    effective_kwargs = dict(transcribe_kwargs or {})
    if "condition_on_previous_text" not in effective_kwargs:
        effective_kwargs["condition_on_previous_text"] = True

    _memsnap("asr_pre_transcribe", extra={"model": model_name, "input_type": type(audio_input).__name__})
    raw_result = transcribe(audio_input, path_or_hf_repo=model_name, **effective_kwargs)
    _memsnap("asr_post_transcribe", extra={"segments": len(raw_result.get("segments") or [])})

    repeat_detected = _should_retry_without_condition(raw_result)
    if repeat_detected and _is_garble_text(raw_result.get("text", "")):
        logger.warning("garble_detected_skip_retry: %s", display_name)
        return _build_silence_result(
            display_name=display_name,
            language=raw_result.get("language") or language_hint,
        )

    if effective_kwargs.get("condition_on_previous_text", True) and repeat_detected:
        fallback_kwargs = dict(effective_kwargs)
        fallback_kwargs["condition_on_previous_text"] = False
        logger.warning("condition_retry_without_previous_text: %s", display_name)
        raw_result = transcribe(audio_input, path_or_hf_repo=model_name, **fallback_kwargs)
        _memsnap(
            "asr_post_transcribe_retry",
            extra={
                "segments": len(raw_result.get("segments") or []),
                "condition_on_previous_text": False,
            },
        )
        repeat_detected = _should_retry_without_condition(raw_result)
        if repeat_detected:
            if _is_garble_text(raw_result.get("text", "")):
                logger.warning("garble_detected_after_retry: %s", display_name)
            logger.warning("repeat_detected_after_retry_force_silence: %s", display_name)
            return _build_silence_result(
                display_name=display_name,
                language=raw_result.get("language") or language_hint,
            )

    if _should_force_silence(raw_result):
        logger.info("silence_by_model: %s", display_name)
        return _build_silence_result(
            display_name=display_name,
            language=raw_result.get("language") or language_hint,
        )
    return build_transcription_result(display_name, raw_result)


def _resolve_silence_threshold(fallback: float) -> float:
    raw = os.getenv("ASR_SILENCE_THRESHOLD")
    if raw is None:
        return fallback
    try:
        value = float(raw)
    except ValueError:
        return fallback
    if not np.isfinite(value) or value <= 0:
        return fallback
    return value


def _is_waveform_silent(waveform: np.ndarray, *, threshold: float = 1e-3) -> bool:
    threshold = _resolve_silence_threshold(threshold)
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


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_csv(name: str, default: list[str]) -> list[str]:
    raw = os.getenv(name)
    if raw is None:
        return default
    items = [x.strip() for x in raw.split(",")]
    return [x for x in items if x]


def _iter_float(values: Iterable[float | None]) -> list[float]:
    out: list[float] = []
    for value in values:
        if isinstance(value, (int, float)):
            out.append(float(value))
    return out


def _should_force_silence(raw_result: dict[str, Any]) -> bool:
    text = (raw_result.get("text") or "").strip()
    if not text:
        return False

    segments = raw_result.get("segments") or []
    if not isinstance(segments, list) or not segments:
        return False

    phrases = _env_csv("ASR_HALLUCINATION_PHRASES", ["ご視聴ありがとうございました"])
    if phrases and any(phrase in text for phrase in phrases):
        no_speech_vals = _iter_float(
            [seg.get("no_speech_prob") for seg in segments if isinstance(seg, dict)]
        )
        comp_vals = _iter_float(
            [seg.get("compression_ratio") for seg in segments if isinstance(seg, dict)]
        )
        logp_vals = _iter_float(
            [seg.get("avg_logprob") for seg in segments if isinstance(seg, dict)]
        )
        ns_min = _env_float("ASR_HALLUCINATION_PHRASE_NO_SPEECH_MIN", 0.6)
        comp_th = _env_float("ASR_HALLUCINATION_COMPRESSION_RATIO", 2.4)
        logp_th = _env_float("ASR_HALLUCINATION_AVG_LOGPROB", -1.0)
        if (no_speech_vals and max(no_speech_vals) >= ns_min) or (
            comp_vals and logp_vals and max(comp_vals) >= comp_th and min(logp_vals) <= logp_th
        ):
            return True

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
    # Whisper の閾値は環境によって感度が高く出ることがあるため、より広い音量帯を許容するように上げておく。
    if max_score >= 0.85 or avg_score >= 0.75:
        return True

    comp_vals = _iter_float(
        [seg.get("compression_ratio") for seg in segments if isinstance(seg, dict)]
    )
    logp_vals = _iter_float([seg.get("avg_logprob") for seg in segments if isinstance(seg, dict)])
    comp_th = _env_float("ASR_FORCE_SILENCE_COMPRESSION_RATIO", 2.4)
    logp_th = _env_float("ASR_FORCE_SILENCE_AVG_LOGPROB", -1.0)
    if comp_vals and logp_vals and max(comp_vals) >= comp_th and min(logp_vals) <= logp_th:
        return True
    return False


def _should_retry_without_condition(raw_result: dict[str, Any]) -> bool:
    text = (raw_result.get("text") or "").strip()
    segments = raw_result.get("segments") or []
    if _has_repeated_tail(text):
        return True
    return _has_repeated_segments(segments)


def _has_repeated_tail(text: str, *, min_chunk: int = 6, max_chunk: int = 30) -> bool:
    if not text:
        return False
    normalized = text.replace("\n", "").replace("\r", "").replace(" ", "")
    if len(normalized) < min_chunk * 3:
        return _has_short_repeat_tail(normalized)
    tail = normalized[-max_chunk * 3 :]
    max_size = min(max_chunk, len(tail) // 3)
    for size in range(min_chunk, max_size + 1):
        if len(tail) < size * 3:
            break
        last = tail[-size:]
        mid = tail[-2 * size : -size]
        prev = tail[-3 * size : -2 * size]
        if last and last == mid == prev:
            return True
    return _has_short_repeat_tail(tail)


def _is_garble_text(text: str, *, min_repeats: int = 5, max_unit: int = 3) -> bool:
    if not text:
        return False
    normalized = text.replace("\n", "").replace("\r", "").replace(" ", "")
    if len(normalized) < min_repeats:
        return False
    unique_chars = set(normalized)
    if len(unique_chars) > 2:
        return False
    return _has_short_repeat_tail(normalized, min_repeats=min_repeats, max_unit=max_unit)


def _has_short_repeat_tail(text: str, *, min_repeats: int = 5, max_unit: int = 3) -> bool:
    if not text:
        return False
    if min_repeats <= 1:
        return False
    tail = text[-(max_unit * min_repeats) :]
    for unit in range(1, max_unit + 1):
        if len(tail) < unit * min_repeats:
            continue
        chunk = tail[-unit:]
        if chunk and chunk * min_repeats == tail[-unit * min_repeats :]:
            return True
    return False


def _has_repeated_segments(segments: Sequence[Any], *, min_repeats: int = 3) -> bool:
    texts: list[str] = []
    for segment in segments:
        if isinstance(segment, dict):
            content = (segment.get("text") or "").strip()
        elif hasattr(segment, "text"):
            content = str(getattr(segment, "text", "") or "").strip()
        else:
            content = ""
        if content:
            texts.append(content)
    if len(texts) < min_repeats:
        return False
    repeat = 1
    for idx in range(1, len(texts)):
        if texts[idx] == texts[idx - 1]:
            repeat += 1
            if repeat >= min_repeats:
                return True
        else:
            repeat = 1
    unique_ratio = len(set(texts)) / len(texts) if texts else 1.0
    return unique_ratio < 0.5 and len(texts) >= 6


__all__ = [
    "transcribe_waveform",
]
