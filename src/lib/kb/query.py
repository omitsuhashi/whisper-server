from __future__ import annotations

from dataclasses import dataclass
from typing import List

from src.lib.kb.retrieve import RetrievedUnit, hybrid_search


@dataclass(frozen=True)
class QueryResult:
    query: str
    hits: list[RetrievedUnit]


def run_query(
    query: str,
    *,
    topn_dense: int = 50,
    topk: int = 10,
    lam_mmr: float = 0.5,
    alpha_time: float = 0.2,
) -> QueryResult:
    hits = hybrid_search(
        query,
        topn_dense=topn_dense,
        topk=topk,
        lam_mmr=lam_mmr,
        alpha_time=alpha_time,
    )
    return QueryResult(query=query, hits=hits)
