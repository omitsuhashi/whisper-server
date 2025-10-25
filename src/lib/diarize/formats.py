from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List

from .models import DiarizationResult, SpeakerSegment


def save_rttm(path: str | Path, diar: DiarizationResult, *, uri: str | None = None) -> None:
    """RTTM 形式で保存。"""

    uri = uri or diar.filename
    lines: List[str] = []
    for turn in diar.turns:
        duration = max(0.0, turn.end - turn.start)
        lines.append(f"SPEAKER {uri} 1 {turn.start:.3f} {duration:.3f} <NA> <NA> {turn.speaker} <NA> <NA>")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_json_diarization(path: str | Path, diar: DiarizationResult) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(diar.model_dump(), handle, ensure_ascii=False, indent=2)


def save_srt_with_speaker(path: str | Path, segments: Iterable[SpeakerSegment]) -> None:
    def fmt_time(time_seconds: float) -> str:
        time_seconds = max(0.0, time_seconds)
        hours = int(time_seconds // 3600)
        minutes = int((time_seconds % 3600) // 60)
        seconds = int(time_seconds % 60)
        milliseconds = int(round((time_seconds - int(time_seconds)) * 1000))
        return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

    lines: List[str] = []
    for index, segment in enumerate(segments, 1):
        start = max(0.0, segment.start)
        end = max(start + 0.01, segment.end)
        lines.append(str(index))
        lines.append(f"{fmt_time(start)} --> {fmt_time(end)}")
        lines.append(f"{segment.speaker}: {segment.text}")
        lines.append("")
    Path(path).write_text("\n".join(lines), encoding="utf-8")


__all__ = [
    "save_json_diarization",
    "save_rttm",
    "save_srt_with_speaker",
]
