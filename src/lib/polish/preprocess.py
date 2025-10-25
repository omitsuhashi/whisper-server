from __future__ import annotations

import re
import unicodedata
from typing import Dict, Iterable, List, Tuple

from .options import PolishOptions


def normalize_text(s: str, opt: PolishOptions) -> str:
    if opt.normalize_width:
        s = unicodedata.normalize("NFKC", s)
    if opt.space_collapse:
        s = re.sub(r"[ \t\u3000]+", " ", s).strip()
    s = s.replace("，", "、").replace("．", "。")
    return s


def remove_fillers(s: str, opt: PolishOptions) -> str:
    if not opt.remove_fillers:
        return s
    for pattern in opt.filler_patterns:
        s = re.sub(pattern, "", s, flags=re.IGNORECASE)
    return re.sub(r"\s{2,}", " ", s).strip()


def limit_repeats(s: str, max_repeat: int) -> str:
    def repl(match: re.Match[str]) -> str:
        ch = match.group(1)
        return ch * max_repeat

    return re.sub(r"(.)\1{%d,}" % max_repeat, repl, s)


def compile_protected(patterns: Iterable[str]) -> List[re.Pattern[str]]:
    out: List[re.Pattern[str]] = []
    for pattern in patterns:
        pattern = pattern.strip()
        if not pattern:
            continue
        out.append(re.compile(pattern))
    return out


def protect_tokens(s: str, prot_res: List[re.Pattern[str]]) -> Tuple[str, Dict[str, str]]:
    mapping: Dict[str, str] = {}
    for idx, regex in enumerate(prot_res):
        def repl(match: re.Match[str]) -> str:
            token = match.group(0)
            key = f"__PROT_{idx}_{len(mapping)}__"
            mapping[key] = token
            return key

        s = regex.sub(repl, s)
    return s, mapping


def unprotect_tokens(s: str, mapping: Dict[str, str]) -> str:
    for key, token in mapping.items():
        s = s.replace(key, token)
    return s


def apply_terms(s: str, pairs: Tuple[Tuple[str, str], ...], regex: bool) -> str:
    if not pairs:
        return s
    for src, dst in pairs:
        pattern = src if regex else re.escape(src)
        s = re.sub(pattern, dst)
    return s


__all__ = [
    "apply_terms",
    "compile_protected",
    "limit_repeats",
    "normalize_text",
    "protect_tokens",
    "remove_fillers",
    "unprotect_tokens",
]
