from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Sequence

PROMPT_TOKEN_LIMIT = 224
"""Whisper の initial_prompt が許容する概算トークン数。"""

_APPROX_CHARS_PER_TOKEN = 4
_DEFAULT_STYLE_GUIDANCE = (
    "英数字は半角、技術用語は英語綴り、句点は「。」読点は「、」。"
    "要約せず、話した内容をできるだけそのまま書き起こしてください。"
)


@dataclass(frozen=True)
class PromptContext:
    """会議単位で初期プロンプトを生成するためのメタデータ。"""

    agenda_items: Sequence[str] = ()
    participants: Sequence[str] = ()
    products: Sequence[str] = ()
    terms: Sequence[str] = ()
    dictionary: Sequence[str] = ()
    style_guidance: str | None = None


def normalize_prompt_items(raw: str | Sequence[str] | None) -> list[str]:
    """カンマ/改行区切りの文字列をトリム済みのリストへ変換する。"""

    if raw is None:
        return []
    values: list[str]
    if isinstance(raw, str):
        values = [raw]
    else:
        values = [value for value in raw if value]

    items: list[str] = []
    for value in values:
        parts = re.split(r"[,/\n]+", value)
        for part in parts:
            cleaned = part.strip()
            if cleaned:
                items.append(cleaned)
    return items


def build_initial_prompt(context: PromptContext, *, token_limit: int = PROMPT_TOKEN_LIMIT) -> str | None:
    """メタデータから 224 トークン以内を目安とした initial_prompt を生成する。"""

    agenda = ", ".join(context.agenda_items)
    participants = ", ".join(context.participants)
    products = ", ".join(context.products)
    terms = ", ".join(context.terms)
    dictionary = ", ".join(context.dictionary)

    lines: list[str] = []
    style = (context.style_guidance or _DEFAULT_STYLE_GUIDANCE).strip()
    if style:
        lines.append(style)
    if products:
        lines.append(f"製品・サービス: {products}")
    if agenda:
        lines.append(f"議題: {agenda}")
    if participants:
        lines.append(f"参加者: {participants}")
    if terms:
        lines.append(f"重要語彙: {terms}")
    if dictionary:
        lines.append(f"表記固定: {dictionary}")
    if not lines:
        return None

    prompt = " ".join(lines).strip()
    return _truncate_prompt(prompt, token_limit)


def _truncate_prompt(prompt: str, token_limit: int) -> str:
    max_chars = token_limit * _APPROX_CHARS_PER_TOKEN
    if len(prompt) <= max_chars:
        return prompt
    # 単語/読点単位で落としてから最終的に安全な長さへ丸める。
    clauses = re.split(r"([。、,])", prompt)
    rebuilt: list[str] = []
    length = 0
    for clause in clauses:
        if not clause:
            continue
        prospective = length + len(clause)
        if prospective > max_chars:
            break
        rebuilt.append(clause)
        length = prospective
    trimmed = "".join(rebuilt).strip()
    if len(trimmed) > max_chars:
        trimmed = trimmed[:max_chars].rstrip()
    return trimmed


def build_prompt_from_metadata(
    *,
    agenda: str | Sequence[str] | None = None,
    participants: str | Sequence[str] | None = None,
    products: str | Sequence[str] | None = None,
    terms: str | Sequence[str] | None = None,
    dictionary: str | Sequence[str] | None = None,
    style: str | None = None,
    token_limit: int = PROMPT_TOKEN_LIMIT,
) -> str | None:
    """CLI/HTTP から渡された文字列群を正規化して initial_prompt を返す。"""

    agenda_items = normalize_prompt_items(agenda)
    participant_items = normalize_prompt_items(participants)
    product_items = normalize_prompt_items(products)
    terms_items = normalize_prompt_items(terms)
    dictionary_items = normalize_prompt_items(dictionary)

    style_value = (style or "").strip()

    context = PromptContext(
        agenda_items=agenda_items,
        participants=participant_items,
        products=product_items,
        terms=terms_items,
        dictionary=dictionary_items,
        style_guidance=style_value or None,
    )
    return build_initial_prompt(context, token_limit=token_limit)


__all__ = [
    "PROMPT_TOKEN_LIMIT",
    "PromptContext",
    "build_initial_prompt",
    "build_prompt_from_metadata",
    "normalize_prompt_items",
]
