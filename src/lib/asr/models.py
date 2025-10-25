from __future__ import annotations

from typing import List

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


class TranscriptionResult(BaseModel):
    """書き起こし結果を扱いやすい形に正規化したモデル。"""

    model_config = ConfigDict(extra="ignore")

    filename: str
    text: str = ""
    language: str | None = None
    duration: float | None = None
    segments: List[TranscriptionSegment] = Field(default_factory=list)


__all__ = ["TranscriptionResult", "TranscriptionSegment"]
