from __future__ import annotations

from typing import List

from pydantic import BaseModel, ConfigDict, Field


class SpeakerTurn(BaseModel):
    """話者区間（start/end は秒）。"""

    model_config = ConfigDict(extra="ignore")

    start: float
    end: float
    speaker: str


class DiarizationResult(BaseModel):
    """話者分離の結果（ファイル単位）。"""

    model_config = ConfigDict(extra="ignore")

    filename: str
    duration: float | None = None
    turns: List[SpeakerTurn] = Field(default_factory=list)

    @property
    def speakers(self) -> List[str]:
        return sorted({t.speaker for t in self.turns})


class SpeakerSegment(BaseModel):
    """話者ラベル付き ASR セグメント。"""

    model_config = ConfigDict(extra="ignore")

    id: int | None = None
    start: float
    end: float
    text: str
    speaker: str = "UNK"


class SpeakerAnnotatedTranscript(BaseModel):
    """話者ラベルを付与済みの書き起こし全体。"""

    model_config = ConfigDict(extra="ignore")

    filename: str
    segments: List[SpeakerSegment] = Field(default_factory=list)

    @property
    def speakers(self) -> List[str]:
        return sorted({s.speaker for s in self.segments})


__all__ = [
    "SpeakerTurn",
    "DiarizationResult",
    "SpeakerSegment",
    "SpeakerAnnotatedTranscript",
]
