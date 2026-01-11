from __future__ import annotations

import os
from typing import Iterable

from .models import (
    FlaggedSegment,
    TranscriptionDiagnostics,
    TranscriptionResult,
    TranscriptionSegment,
)
from .pipeline import _has_repeated_segments, _has_repeated_tail


_DEFAULT_AVG_LOGPROB_TH = -1.0
_DEFAULT_COMP_RATIO_TH = 2.4
_DEFAULT_NO_SPEECH_TH = 0.85
_DEFAULT_HALLUCINATION_PHRASES = ["ご視聴ありがとうございました"]


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


def _iter_non_none(values: Iterable[float | None]) -> list[float]:
    out: list[float] = []
    for value in values:
        if isinstance(value, (int, float)):
            out.append(float(value))
    return out


def analyze_transcription_quality(result: TranscriptionResult) -> TranscriptionDiagnostics:
    segments: list[TranscriptionSegment] = list(result.segments or [])
    text = (result.text or "").strip()

    avg_logprob_th = _env_float("ASR_FLAG_AVG_LOGPROB", _DEFAULT_AVG_LOGPROB_TH)
    comp_ratio_th = _env_float("ASR_FLAG_COMPRESSION_RATIO", _DEFAULT_COMP_RATIO_TH)
    no_speech_th = _env_float("ASR_FLAG_NO_SPEECH_PROB", _DEFAULT_NO_SPEECH_TH)

    phrases = _env_csv("ASR_HALLUCINATION_PHRASES", _DEFAULT_HALLUCINATION_PHRASES)

    flags: list[str] = []
    flagged: list[FlaggedSegment] = []

    no_speech_vals = _iter_non_none([seg.no_speech_prob for seg in segments])
    comp_vals = _iter_non_none([seg.compression_ratio for seg in segments])
    logp_vals = _iter_non_none([seg.avg_logprob for seg in segments])

    repeated_tail = _has_repeated_tail(text) if text else False
    repeated_segs = _has_repeated_segments(segments) if segments else False
    if repeated_tail:
        flags.append("repeat_tail")
    if repeated_segs:
        flags.append("repeat_segments")

    for seg in segments:
        reasons: list[str] = []
        if seg.avg_logprob is not None and seg.avg_logprob <= avg_logprob_th:
            reasons.append("avg_logprob_low")
        if seg.compression_ratio is not None and seg.compression_ratio >= comp_ratio_th:
            reasons.append("compression_ratio_high")
        if seg.no_speech_prob is not None and seg.no_speech_prob >= no_speech_th and (seg.text or "").strip():
            reasons.append("no_speech_prob_high")

        if reasons:
            flagged.append(
                FlaggedSegment(
                    start=float(seg.start),
                    end=float(seg.end),
                    text=seg.text or "",
                    reasons=reasons,
                    avg_logprob=seg.avg_logprob,
                    compression_ratio=seg.compression_ratio,
                    no_speech_prob=seg.no_speech_prob,
                )
            )

    if text and phrases and any(phrase in text for phrase in phrases):
        flags.append("hallucination_phrase_match")

    metrics = {
        "segments": float(len(segments)),
        "max_no_speech_prob": max(no_speech_vals) if no_speech_vals else 0.0,
        "min_avg_logprob": min(logp_vals) if logp_vals else 0.0,
        "max_compression_ratio": max(comp_vals) if comp_vals else 0.0,
        "flagged_segments": float(len(flagged)),
    }

    if flagged:
        flags.append("low_confidence_segments")

    return TranscriptionDiagnostics(flags=flags, flagged_segments=flagged, metrics=metrics)


__all__ = ["analyze_transcription_quality"]
