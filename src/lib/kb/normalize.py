from __future__ import annotations

from functools import lru_cache
from typing import Sequence

from semantic_text_splitter import TextSplitter


@lru_cache(maxsize=8)
def _get_splitter(max_chars: int) -> TextSplitter:
    safe_chars = max(1, int(max_chars))
    return TextSplitter(safe_chars)


def semantic_chunks(text: str, language: str = "ja", max_chars: int = 1200) -> Sequence[str]:
    """意味を壊さない粒度でテキストを分割する。language はメタ情報用のラベルとして扱う。"""

    splitter = _get_splitter(max_chars)
    return list(splitter.chunks(text))
