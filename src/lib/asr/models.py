from __future__ import annotations

from typing import Dict, List

from pydantic import BaseModel, ConfigDict, Field


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


class FlaggedSegment(BaseModel):
    model_config = ConfigDict(extra="ignore")

    start: float
    end: float
    text: str
    reasons: List[str] = Field(default_factory=list)
    avg_logprob: float | None = None
    compression_ratio: float | None = None
    no_speech_prob: float | None = None


class TranscriptionDiagnostics(BaseModel):
    model_config = ConfigDict(extra="ignore")

    flags: List[str] = Field(default_factory=list)
    flagged_segments: List[FlaggedSegment] = Field(default_factory=list)
    metrics: Dict[str, float] = Field(default_factory=dict)


class TranscriptionResult(BaseModel):
    """書き起こし結果を扱いやすい形に正規化したモデル。"""

    model_config = ConfigDict(extra="ignore")

    filename: str
    text: str = ""
    language: str | None = None
    duration: float | None = None
    segments: List[TranscriptionSegment] = Field(default_factory=list)
    diagnostics: TranscriptionDiagnostics | None = None
    window_start_seconds: float | None = None
    window_end_seconds: float | None = None


__all__ = [
    "FlaggedSegment",
    "TranscriptionDiagnostics",
    "TranscriptionResult",
    "TranscriptionSegment",
]
