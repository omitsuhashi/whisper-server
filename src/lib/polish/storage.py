from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .models import PolishedDocument, PolishedSentence


def save_txt(path: str | Path, text: str) -> None:
    Path(path).write_text(text + "\n", encoding="utf-8")


def save_json_doc(path: str | Path, doc: PolishedDocument) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc.model_dump(), f, ensure_ascii=False, indent=2)


def save_srt(path: str | Path, sentences: Iterable[PolishedSentence]) -> None:
    def fmt_time(t: float) -> str:
        t = max(0.0, t)
        h = int(t // 3600)
        m = int((t % 3600) // 60)
        s = int(t % 60)
        ms = int(round((t - int(t)) * 1000))
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    lines: list[str] = []
    for i, sentence in enumerate(sentences, 1):
        start = max(0.0, sentence.start)
        end = max(start + 0.01, sentence.end)
        lines.append(str(i))
        lines.append(f"{fmt_time(start)} --> {fmt_time(end)}")
        lines.append(sentence.text)
        lines.append("")
    Path(path).write_text("\n".join(lines), encoding="utf-8")


__all__ = ["save_json_doc", "save_srt", "save_txt"]
