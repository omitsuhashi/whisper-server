from __future__ import annotations

import logging
import re
from typing import Any, List, Sequence, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from spacy.language import Language
else:  # pragma: no cover
    Language = Any

from .options import PolishOptions

logger = logging.getLogger(__name__)

_GINZA_MODEL_NAME = "ja_ginza_electra"
_GINZA_NLP: Language | None = None


def load_ginza(model_name: str, *, fallbacks: Sequence[str] = ()) -> Language:
    """GiNZA モデルをロードし、グローバルにキャッシュする。"""

    global _GINZA_MODEL_NAME, _GINZA_NLP

    candidates = [model_name, *fallbacks]
    last_exc: Exception | None = None

    for candidate in candidates:
        if _GINZA_NLP is not None and _GINZA_MODEL_NAME == candidate:
            return _GINZA_NLP

        try:
            import spacy

            nlp = spacy.load(candidate)
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            logger.debug(
                "GiNZAモデル '%s' の読み込みに失敗しました: %s",
                candidate,
                exc,
                exc_info=True,
            )
            continue

        _GINZA_MODEL_NAME = candidate
        _GINZA_NLP = nlp

        if candidate != model_name:
            logger.warning(
                "GiNZAモデル '%s' が利用できなかったため '%s' にフォールバックします。",
                model_name,
                candidate,
            )
        return nlp

    tried = "、".join(candidates)
    raise RuntimeError(
        f"GiNZA モデル '{model_name}' の読み込みに失敗しました"
        f"（試行済み: {tried}）。"
        " pip install ginza ja-ginza-electra 等で依存関係を満たしてください。"
    ) from last_exc


def split_with_ginza(text: str, opt: PolishOptions) -> List[str]:
    nlp = load_ginza(opt.ginza_model, fallbacks=opt.ginza_model_fallbacks)
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]


def split_with_heuristics(text: str, opt: PolishOptions) -> List[str]:
    pieces = re.split(r"(?<=[。！？!?])\s*|\n+", text)
    pieces = [p.strip() for p in pieces if p and p.strip()]
    out: List[str] = []
    buf = ""
    for chunk in pieces:
        buf = (buf + " " + chunk).strip() if buf else chunk
        too_long = len(buf) >= opt.max_sentence_len
        matched_tail = any(re.search(pattern, buf) for pattern in opt.period_heuristics)
        if too_long or matched_tail:
            if not re.search(r"[。！？!?]\s*$", buf):
                buf += "。"
            out.append(buf)
            buf = ""
    if buf:
        if not re.search(r"[。！？!?]\s*$", buf):
            buf += "。"
        out.append(buf)

    merged: List[str] = []
    for sentence in out:
        if merged and len(sentence) < opt.min_sentence_len:
            merged[-1] = (merged[-1].rstrip("。") + " " + sentence).strip()
            if not merged[-1].endswith(("。", "！", "？", "!", "?")):
                merged[-1] += "。"
        else:
            merged.append(sentence)
    return merged


def split_into_sentences(text: str, opt: PolishOptions) -> List[str]:
    if opt.use_ginza:
        try:
            sentences = split_with_ginza(text, opt)
            if sentences:
                return sentences
        except Exception as exc:  # noqa: BLE001
            if not opt.ginza_fallback_to_heuristics:
                raise
            logger.warning(
                "GiNZAによる文分割に失敗したためヒューリスティクスへフォールバックします: %s",
                exc,
            )
    return split_with_heuristics(text, opt)


__all__ = [
    "load_ginza",
    "split_into_sentences",
    "split_with_ginza",
    "split_with_heuristics",
]
