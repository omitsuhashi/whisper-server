from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

_WS = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Apply the minimum normalization needed for KTA."""
    normalized = (text or "").strip()
    normalized = _WS.sub(" ", normalized)
    return normalized


@dataclass(frozen=True)
class KeyTermSample:
    id: str
    transcript: str
    key_terms: list[str]


def key_term_accuracy(transcript: str, key_terms: Iterable[str]) -> float:
    transcript_normalized = normalize(transcript)
    terms: list[str] = []
    for term in key_terms:
        normalized = normalize(term)
        if normalized:
            terms.append(normalized)

    if not terms:
        return 1.0

    hits = sum(1 for term in terms if term in transcript_normalized)
    return hits / len(terms)
