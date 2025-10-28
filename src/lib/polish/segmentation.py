from __future__ import annotations

import re
from typing import List

from .options import PolishOptions

def split_with_heuristics(text: str, opt: PolishOptions) -> List[str]:
    pieces = re.split(r"(?<=[。！？!?])\s*|\n+", text)
    pieces = [p.strip() for p in pieces if p and p.strip()]
    out: List[str] = []
    buf = ""
    for chunk in pieces:
        buf = (buf + " " + chunk).strip() if buf else chunk
        too_long = len(buf) >= opt.max_sentence_len
        ends_with_punct = bool(re.search(r"[。！？!?]\s*$", buf))
        matched_tail = ends_with_punct or any(re.search(pattern, buf) for pattern in opt.period_heuristics)
        if too_long or matched_tail:
            if not re.search(r"[。！？!?]\s*$", buf):
                buf += "。"
            out.append(buf)
            buf = ""
    if buf:
        if not re.search(r"[。！？!?]\s*$", buf):
            buf += "。"
        out.append(buf)

    merged: List[str] = []
    for sentence in out:
        if merged and len(sentence) < opt.min_sentence_len:
            merged[-1] = (merged[-1].rstrip("。") + " " + sentence).strip()
            if not merged[-1].endswith(("。", "！", "？", "!", "?")):
                merged[-1] += "。"
        else:
            merged.append(sentence)
    return merged


def split_into_sentences(text: str, opt: PolishOptions) -> List[str]:
    return split_with_heuristics(text, opt)


__all__ = [
    "split_into_sentences",
    "split_with_heuristics",
]
