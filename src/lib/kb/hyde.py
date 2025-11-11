from __future__ import annotations

import logging
import os
from typing import Optional

from litellm import completion

logger = logging.getLogger(__name__)

DEFAULT_HYDE_MODEL = os.getenv("KB_HYDE_MODEL", "openai/gpt-4o-mini")
PROMPT_TEMPLATE = """以下の質問に対して、もっともらしい背景説明（300字以内、箇条書き可）を日本語で作成してください。根拠が不明な断定は避け、一般的な背景や既知の事実にとどめてください。

質問: {query}

仮説文: """


def generate_hypothesis(query: str, model: Optional[str] = None) -> Optional[str]:
    """HyDE: クエリから仮説文書を生成する。失敗時は None を返す。"""

    model_name = model or DEFAULT_HYDE_MODEL
    if not model_name:
        return None
    prompt = PROMPT_TEMPLATE.format(query=query.strip())
    try:
        response = completion(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
        )
    except Exception as exc:  # pragma: no cover - API依存
        logger.warning("HyDE 生成に失敗しました: %s", exc)
        return None
    content = response.choices[0].message.get("content") if response.choices else None
    if not content:
        return None
    return content.strip()
