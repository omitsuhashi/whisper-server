from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List

from src.lib.kb.embed import encode_passages
from src.lib.kb.hyde import generate_hypothesis
from src.lib.kb.rerank import rerank
from src.lib.kb.retrieve import RetrievedUnit, hybrid_search

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class QueryResult:
    query: str
    hits: list[RetrievedUnit]
    hyde_text: str | None = None


def run_query(
    query: str,
    *,
    topn_dense: int = 50,
    topk: int = 10,
    lam_mmr: float = 0.5,
    alpha_time: float = 0.2,
    use_hyde: bool = True,
    use_rerank: bool = True,
    rerank_topk: int = 5,
) -> QueryResult:
    query_embedding = None
    hyde_text = None
    if use_hyde:
        hyde_text = generate_hypothesis(query)
        if hyde_text:
            try:
                query_embedding = encode_passages([hyde_text])[0]
            except Exception as exc:  # pragma: no cover - モデル依存
                logger.warning("HyDE エンコードに失敗しました: %s", exc)
                query_embedding = None

    hits = hybrid_search(
        query,
        topn_dense=topn_dense,
        topk=topk,
        lam_mmr=lam_mmr,
        alpha_time=alpha_time,
        query_embedding=query_embedding,
    )
    if use_rerank and hits:
        try:
            hits = rerank(query, hits, topk=rerank_topk)
        except Exception as exc:  # pragma: no cover - モデル依存
            logger.warning("CrossEncoder rerank に失敗しました: %s", exc)
    return QueryResult(query=query, hits=hits, hyde_text=hyde_text)
