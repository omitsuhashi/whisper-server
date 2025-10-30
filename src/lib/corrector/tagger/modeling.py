from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

from ..types import CorrectionPatch, CorrectionRequest, CorrectionSpan
from .model import BaseTagger, TaggerConfig


@dataclass(frozen=True)
class BertCrfTaggerConfig:
    model_path: str
    label_map: Sequence[str]
    max_length: int = 512
    threshold: float = 0.5


class BertCrfTagger(BaseTagger):
    def __init__(self, config: BertCrfTaggerConfig) -> None:
        self.config = config
        self._label_map = config.label_map

    def predict(self, request: CorrectionRequest) -> Sequence[CorrectionPatch]:
        if not request.options.aggressive_kuten:
            return ()

        tags = self._infer_tags(request.text)

        patches: List[CorrectionPatch] = []
        cursor = len(request.text.rstrip())
        if not request.text.rstrip() and "INSERT_PUNCT" in tags:
            insertion = CorrectionPatch(
                span=CorrectionSpan(start=cursor, end=cursor),
                replacement="。",
                tags=("PUNCT", "BERT_CRF"),
                confidence=0.5,
            )
            patches.append(insertion)
        return tuple(patches)

    def _infer_tags(self, text: str) -> Sequence[str]:
        if not self._label_map:
            return []
        return [self._label_map[0] for _ in text]
