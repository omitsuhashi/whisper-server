from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

from ..types import CorrectionPatch, CorrectionRequest, CorrectionSpan

_DEFAULT_SENTENCE_TERMINATORS: Tuple[str, ...] = ("。", "．", "!", "！", "?", "？", "…", "⋯")


@dataclass(frozen=True)
class TaggerConfig:
    """設定値を保持するデータクラス。"""

    sentence_terminators: Tuple[str, ...] = _DEFAULT_SENTENCE_TERMINATORS
    insertion_char: str = "。"
    insertion_confidence: float = 0.55
    add_rule_based_tag: bool = True


class BaseTagger:
    """補正タグ付け器の基本インターフェース。"""

    def predict(self, request: CorrectionRequest) -> Sequence[CorrectionPatch]:
        raise NotImplementedError


class RuleBasedPunctTagger(BaseTagger):
    """簡易な句点補完のルールベース実装。"""

    def __init__(self, config: TaggerConfig | None = None) -> None:
        self.config = config or TaggerConfig()

    def predict(self, request: CorrectionRequest) -> Tuple[CorrectionPatch, ...]:
        if not request.options.aggressive_kuten:
            return ()

        stripped = request.text.rstrip()
        if not stripped:
            return ()

        if stripped.endswith(self.config.sentence_terminators):
            return ()

        insertion_index = len(stripped)
        patch = CorrectionPatch(
            span=CorrectionSpan(start=insertion_index, end=insertion_index),
            replacement=self.config.insertion_char,
            tags=self._build_tags(),
            confidence=self.config.insertion_confidence,
        )
        return (patch,)

    def _build_tags(self) -> Tuple[str, ...]:
        if self.config.add_rule_based_tag:
            return ("PUNCT", "RULE_BASED")
        return ("PUNCT",)
