from __future__ import annotations

from pathlib import Path
from typing import Any

from .models import TranscriptionResult, TranscriptionSegment


def build_transcription_result(source: Path | str, payload: dict[str, Any]) -> TranscriptionResult:
    """mlx_whisperの戻り値をTranscriptionResultへ変換する。"""

    segments_raw = payload.get("segments", []) or []
    segments = [TranscriptionSegment.model_validate(segment) for segment in segments_raw]
    filename = source.name if isinstance(source, Path) else Path(str(source)).name
    return TranscriptionResult(
        filename=filename,
        text=payload.get("text", ""),
        language=payload.get("language"),
        duration=payload.get("duration"),
        segments=segments,
    )


__all__ = ["build_transcription_result"]
