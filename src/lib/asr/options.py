from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class TranscribeOptions:
    """Whisper 書き起こしの基本設定。"""

    model_name: str
    language: str | None = None
    task: str | None = None
    decode_options: Dict[str, Any] = field(default_factory=dict)

    def build_transcribe_kwargs(self) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = dict(self.decode_options)
        if self.language:
            kwargs["language"] = self.language
        if self.task:
            kwargs["task"] = self.task
        return kwargs


__all__ = ["TranscribeOptions"]
