from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, List, Sequence

import numpy as np
from sqlalchemy import text as sql_text

from src.lib.kb.db import session_scope
from src.lib.kb.embed import encode_queries


@dataclass(frozen=True)
class RetrievedUnit:
    unit_id: int
    note_id: int
    title: str | None
    source: str | None
    text: str
    created_at: datetime
    similarity: float
    freshness: float
    rerank_score: float | None = None


def _mmr(doc_vecs: np.ndarray, query_vec: np.ndarray, lam: float, topk: int) -> list[int]:
    if not len(doc_vecs):
        return []
    selected: list[int] = []
    candidates = list(range(len(doc_vecs)))
    similarities = doc_vecs @ query_vec
    while candidates and len(selected) < topk:
        if not selected:
            pivot = int(candidates[np.argmax(similarities[candidates])])
        else:
            selected_vecs = doc_vecs[selected]
            candidate_vecs = doc_vecs[candidates]
            diversity = np.max(selected_vecs @ candidate_vecs.T, axis=0)
            score = lam * similarities[candidates] - (1 - lam) * diversity
            pivot = int(candidates[np.argmax(score)])
        selected.append(pivot)
        candidates.remove(pivot)
    return selected


def _time_decay(ts: datetime, now: datetime, lam: float = 0.01) -> float:
    delta = max((now - ts).days, 0)
    return float(np.exp(-lam * delta))


def hybrid_search(
    query: str,
    *,
    topn_dense: int = 50,
    topk: int = 10,
    lam_mmr: float = 0.5,
    alpha_time: float = 0.2,
    query_embedding: np.ndarray | None = None,
) -> list[RetrievedUnit]:
    """pgvector で候補取得→MMR→時間重みで再スコアリングする。"""

    if topn_dense <= 0 or topk <= 0:
        return []

    query_vec = query_embedding if query_embedding is not None else encode_queries([query])[0]
    with session_scope() as session:
        rows = session.execute(
            sql_text(
                """
                SELECT
                    u.id AS unit_id,
                    u.note_id,
                    u.text,
                    u.embed,
                    u.created_at,
                    n.title,
                    n.source,
                    (1 - (u.embed <=> :q)) AS dense_score
                FROM units AS u
                JOIN notes AS n ON n.id = u.note_id
                WHERE u.embed IS NOT NULL
                ORDER BY u.embed <-> :q
                LIMIT :limit
                """,
            ),
            {"q": query_vec.tolist(), "limit": topn_dense},
        ).mappings().all()

    if not rows:
        return []

    doc_vecs = np.vstack([np.array(row["embed"], dtype=np.float32) for row in rows])
    selected_idx = _mmr(doc_vecs, query_vec, lam=lam_mmr, topk=min(topk, len(rows)))
    now = datetime.now(timezone.utc)
    results: list[RetrievedUnit] = []
    for idx in selected_idx:
        row = rows[idx]
        created_at = row["created_at"]
        freshness = _time_decay(created_at, now)
        score = float(row["dense_score"])
        final_score = (1 - alpha_time) * score + alpha_time * freshness
        results.append(
            RetrievedUnit(
                unit_id=row["unit_id"],
                note_id=row["note_id"],
                title=row["title"],
                source=row["source"],
                text=row["text"],
                created_at=created_at,
                similarity=final_score,
                freshness=freshness,
            ),
        )
    results.sort(key=lambda item: item.similarity, reverse=True)
    return results
