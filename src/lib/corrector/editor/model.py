from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

from ..types import CorrectionRequest


@dataclass(frozen=True)
class EditorConfig:
    glossary: Dict[str, str] = field(default_factory=dict)
    enabled: bool = True
    default_confidence: float = 0.7


class BaseEditor:
    def edit(self, request: CorrectionRequest) -> str:
        raise NotImplementedError


class GlossaryEditor(BaseEditor):
    def __init__(self, config: EditorConfig | None = None) -> None:
        self.config = config or EditorConfig()

    def edit(self, request: CorrectionRequest) -> str:
        if not self.config.enabled:
            return request.text

        text = request.text
        for src, dst in self.config.glossary.items():
            if not src:
                continue
            text = text.replace(src, dst)
        return text
