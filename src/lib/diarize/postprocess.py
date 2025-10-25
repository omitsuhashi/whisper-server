from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field

from .models import DiarizationResult, SpeakerSegment

try:
    from ..asr import TranscriptionResult, TranscriptionSegment  # type: ignore
except Exception:  # pragma: no cover

    class TranscriptionSegment(BaseModel):  # type: ignore
        start: float = 0.0
        end: float = 0.0
        text: str = ""

    class TranscriptionResult(BaseModel):  # type: ignore
        filename: str = "unknown"
        text: str = ""
        language: str | None = None
        duration: float | None = None
        segments: List[TranscriptionSegment] = Field(default_factory=list)


def attach_speaker_labels(
    asr: TranscriptionResult,
    diar: DiarizationResult,
    *,
    default_label: str = "UNK",
) -> "SpeakerAnnotatedTranscript":
    """
    ASR セグメントに、diarization の話者ラベルを重ねる。
    ルール：時間重なりの合計が最大の話者を採用。重なり無しは“最も近い区間”の話者。
    """

    turns = diar.turns
    labeled: List[SpeakerSegment] = []
    for seg in asr.segments:
        best_label = default_label
        best_overlap = 0.0
        seg_s, seg_e = float(seg.start or 0.0), float(seg.end or 0.0)
        for turn in turns:
            overlap = _overlap(seg_s, seg_e, turn.start, turn.end)
            if overlap > best_overlap:
                best_overlap = overlap
                best_label = turn.speaker
        if best_overlap <= 0.0 and turns:
            mid = 0.5 * (seg_s + seg_e)
            best_label = min(turns, key=lambda tt: _dist(mid, 0.5 * (tt.start + tt.end))).speaker
        labeled.append(
            SpeakerSegment(
                id=getattr(seg, "id", None),
                start=seg_s,
                end=seg_e,
                text=seg.text or "",
                speaker=best_label,
            )
        )

    from .models import SpeakerAnnotatedTranscript  # local import to avoid cycle

    return SpeakerAnnotatedTranscript(filename=asr.filename, segments=labeled)


def _overlap(a_s: float, a_e: float, b_s: float, b_e: float) -> float:
    start = max(a_s, b_s)
    end = min(a_e, b_e)
    return max(0.0, end - start)


def _dist(x: float, y: float) -> float:
    return abs(x - y)


__all__ = ["attach_speaker_labels"]
