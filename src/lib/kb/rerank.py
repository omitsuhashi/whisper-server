from __future__ import annotations

import logging
import os
from dataclasses import replace
from typing import Sequence

import numpy as np
from sentence_transformers import CrossEncoder

from src.lib.kb.retrieve import RetrievedUnit

logger = logging.getLogger(__name__)

DEFAULT_RERANK_MODEL = os.getenv(
    "KB_RERANK_MODEL",
    "cross-encoder/ms-marco-MiniLM-L6-v2",
)
_rerank_model: CrossEncoder | None = None


def _ensure_model() -> CrossEncoder:
    global _rerank_model
    if _rerank_model is None:
        logger.info("CrossEncoder をロード中: %s", DEFAULT_RERANK_MODEL)
        _rerank_model = CrossEncoder(DEFAULT_RERANK_MODEL)
    return _rerank_model


def rerank(query: str, hits: Sequence[RetrievedUnit], topk: int = 5) -> list[RetrievedUnit]:
    """Cross-Encoder で再ランキングし、score を付与した新しいリストを返す。"""

    if not hits:
        return []
    try:
        model = _ensure_model()
    except Exception as exc:  # pragma: no cover - モデルDL失敗など
        logger.warning("CrossEncoder の初期化に失敗しました: %s", exc)
        return list(hits)[:topk]

    pairs = [(query, hit.text) for hit in hits]
    try:
        scores = model.predict(pairs, convert_to_numpy=True)
    except Exception as exc:  # pragma: no cover - 推論時例外
        logger.warning("CrossEncoder 推論に失敗しました: %s", exc)
        return list(hits)[:topk]

    reranked = []
    for hit, score in zip(hits, scores):
        reranked.append(replace(hit, rerank_score=float(score)))
    reranked.sort(
        key=lambda item: item.rerank_score if item.rerank_score is not None else item.similarity,
        reverse=True,
    )
    return reranked[: min(topk, len(reranked))]
