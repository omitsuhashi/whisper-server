from __future__ import annotations

import re
from typing import Iterable, List

from .models import PolishedSentence

try:
    from ..asr import TranscriptionSegment  # type: ignore
except Exception:  # pragma: no cover - fallback for import issues
    from pydantic import BaseModel, Field

    class TranscriptionSegment(BaseModel):  # type: ignore
        start: float = 0.0
        end: float = 0.0
        text: str = ""


def reassign_timestamps(
    sentences: Iterable[str],
    segments: List[TranscriptionSegment],
) -> List[PolishedSentence]:
    sentences = list(sentences)
    out: List[PolishedSentence] = []
    if not sentences:
        return out
    if not segments:
        step = 2.0
        t = 0.0
        for sentence in sentences:
            out.append(PolishedSentence(start=t, end=t + step, text=sentence))
            t += step
        return out

    i = 0
    for sentence in sentences:
        sent_len = max(1, len(re.sub(r"\s+", "", sentence)))
        start = segments[i].start if i < len(segments) else (out[-1].end if out else 0.0)
        acc = ""
        j = i
        while j < len(segments) and len(re.sub(r"\s+", "", acc)) < sent_len:
            acc = (acc + " " + segments[j].text).strip()
            j += 1
        if j > i:
            end = segments[min(j - 1, len(segments) - 1)].end
        else:
            end = segments[-1].end if segments else start + 1.5
        out.append(
            PolishedSentence(
                start=float(start),
                end=float(max(end, start + 0.01)),
                text=sentence,
            )
        )
        i = max(j, i + 1)

    for k in range(1, len(out)):
        if out[k].start < out[k - 1].end:
            out[k].start = out[k - 1].end
    return out


__all__ = ["reassign_timestamps"]
