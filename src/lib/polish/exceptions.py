from __future__ import annotations


class LLMPolishError(RuntimeError):
    """LLM を使った校正処理の失敗を表す例外。"""


__all__ = ["LLMPolishError"]
