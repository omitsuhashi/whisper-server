from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from .models import TranscriptionResult, TranscriptionSegment


@dataclass
class SegmentCommitter:
    lookback_seconds: float = 1.0
    committed_until: float = 0.0
    committed_segments: List[TranscriptionSegment] = field(default_factory=list)
    committed_text: str = ""

    def update(
        self,
        result: TranscriptionResult,
        *,
        window_start_seconds: float,
        now_total_seconds: float,
        final: bool = False,
    ) -> str:
        cutoff = float(now_total_seconds) if final else max(
            0.0, float(now_total_seconds) - float(self.lookback_seconds)
        )

        abs_segments: List[TranscriptionSegment] = []
        for seg in (result.segments or []):
            start = float(getattr(seg, "start", 0.0)) + float(window_start_seconds)
            end = float(getattr(seg, "end", start)) + float(window_start_seconds)
            text = str(getattr(seg, "text", "") or "")
            abs_segments.append(seg.model_copy(update={"start": start, "end": end, "text": text}))

        abs_segments.sort(key=lambda s: (float(s.end), float(s.start)))
        newly_committed: List[TranscriptionSegment] = []
        for seg in abs_segments:
            if float(seg.end) > cutoff:
                continue
            if float(seg.end) <= float(self.committed_until) + 1e-6:
                continue
            if self.committed_segments:
                last = self.committed_segments[-1]
                if (last.text or "").strip() == (seg.text or "").strip():
                    gap = float(seg.start) - float(last.end)
                    if gap <= 0.5:
                        self.committed_until = max(self.committed_until, float(seg.end))
                        continue

            self.committed_segments.append(seg)
            self.committed_until = max(self.committed_until, float(seg.end))
            newly_committed.append(seg)

        new_text = "".join(seg.text for seg in newly_committed if seg.text)
        if new_text:
            self.committed_text += new_text
        return new_text

    def build_result(self, *, filename: str, language: Optional[str]) -> TranscriptionResult:
        duration = self.committed_segments[-1].end if self.committed_segments else 0.0
        return TranscriptionResult(
            filename=filename,
            text=self.committed_text.strip(),
            language=language,
            duration=float(duration),
            segments=list(self.committed_segments),
        )


__all__ = ["SegmentCommitter"]
