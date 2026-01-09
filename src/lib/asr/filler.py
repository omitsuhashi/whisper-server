from __future__ import annotations

import re

from .models import TranscriptionResult

# できるだけ誤爆しない「最小」セット
_JA_FILLERS = [
    r"え[ー〜]*っと",
    r"え[ー〜]*と",
    r"うーん",
    r"あの",
    r"[んン][ー〜]+",
    r"[あア][ー〜]+",
    r"[えエ][ー〜]+",
]

_PREFIX_RE = re.compile(
    rf"^(?:{'|'.join(_JA_FILLERS)})(?:(?:[、,。\s…ー〜]+)|$)(?P<rest>.*)$"
)
_INFIX_RE = re.compile(
    rf"(?P<pre>^|[、。,\s])(?:{'|'.join(_JA_FILLERS)})(?P<post>$|[、。,\s])"
)
_ONLY_PUNCT_RE = re.compile(r"^[\s、。,.!?…ー〜]*$")


def _strip_fillers_ja(text: str) -> str:
    s = text or ""
    if not s:
        return ""

    leading_ws_match = re.match(r"\s*", s)
    leading_ws = leading_ws_match.group(0) if leading_ws_match else ""
    s = s[len(leading_ws) :]

    while True:
        matched = _PREFIX_RE.match(s)
        if not matched:
            break
        s = matched.group("rest")

    s = _INFIX_RE.sub(lambda matched: f"{matched.group('pre')}{matched.group('post')}", s)
    s = re.sub(r"[、,]{2,}", "、", s)
    s = re.sub(r"\s{2,}", " ", s)

    if _ONLY_PUNCT_RE.fullmatch(s):
        return ""
    return f"{leading_ws}{s}"


def apply_filler_removal(result: TranscriptionResult, *, enabled: bool) -> TranscriptionResult:
    if not enabled:
        return result

    if result.segments:
        segments = []
        for seg in result.segments:
            cleaned = _strip_fillers_ja(seg.text or "")
            segments.append(seg.model_copy(update={"text": cleaned}))
        parts = [(seg.text or "") for seg in segments if (seg.text or "").strip()]
        combined_text = "".join(parts).strip()
        return result.model_copy(update={"text": combined_text, "segments": segments})

    cleaned_text = _strip_fillers_ja(result.text or "")
    return result.model_copy(update={"text": cleaned_text})


__all__ = ["apply_filler_removal"]
