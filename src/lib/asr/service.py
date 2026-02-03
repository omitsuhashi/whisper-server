from __future__ import annotations

from typing import Optional


def resolve_model_and_language(
    model: Optional[str],
    language: Optional[str],
    *,
    default_model: str,
    default_language: str,
) -> tuple[str, str]:
    return (model or default_model, language or default_language)


__all__ = [
    "resolve_model_and_language",
]
