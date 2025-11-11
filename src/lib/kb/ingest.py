from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from bs4 import BeautifulSoup
from pypdf import PdfReader
from sqlalchemy import select

from .db import init_database, session_scope
from .dedup import simhash_hex
from .models import Note, Unit
from .normalize import semantic_chunks

SUPPORTED_SUFFIXES = {
    ".txt",
    ".md",
    ".markdown",
    ".pdf",
    ".html",
    ".htm",
}


@dataclass(frozen=True)
class IngestOptions:
    root: Path
    pattern: str | None = None
    language: str = "ja"
    max_chars: int = 1200


@dataclass
class IngestStats:
    files_discovered: int = 0
    notes_created: int = 0
    units_created: int = 0
    skipped: int = 0
    errors: list[str] = field(default_factory=list)


def ingest(options: IngestOptions) -> IngestStats:
    """指定パス配下のファイルを読み込み、notes / units テーブルに登録する。"""

    init_database()
    stats = IngestStats()
    files = list(_discover_files(options.root, options.pattern))
    stats.files_discovered = len(files)
    if not files:
        return stats

    with session_scope() as session:
        for file_path in files:
            try:
                body = load_text(file_path)
            except Exception as exc:  # pragma: no cover - ファイル依存
                stats.errors.append(f"{file_path}: {exc}")
                continue

            note_hash = simhash_hex(body)
            existing = session.execute(select(Note.id).where(Note.hash == note_hash)).scalar_one_or_none()
            if existing:
                stats.skipped += 1
                continue

            note = Note(
                title=file_path.stem,
                body=body,
                source=str(file_path),
                lang=options.language,
                hash=note_hash,
            )
            session.add(note)
            session.flush()

            chunks = semantic_chunks(body, language=options.language, max_chars=options.max_chars)
            for chunk in chunks:
                session.add(
                    Unit(
                        note_id=note.id,
                        utype="semantic",
                        text=chunk,
                    ),
                )
            stats.notes_created += 1
            stats.units_created += len(chunks)

    return stats


def load_text(path: Path) -> str:
    """ファイル拡張子に応じてテキスト抽出を行う。"""

    suffix = path.suffix.lower()
    if suffix == ".pdf":
        reader = PdfReader(str(path))
        contents = []
        for page in reader.pages:
            contents.append(page.extract_text() or "")
        return "\n".join(contents)
    if suffix in {".md", ".markdown", ".txt"}:
        return path.read_text(encoding="utf-8", errors="ignore")
    if suffix in {".html", ".htm"}:
        html_text = path.read_text(encoding="utf-8", errors="ignore")
        return BeautifulSoup(html_text, "html.parser").get_text(" ", strip=True)
    raise ValueError(f"未対応のファイル形式です: {path}")


def _discover_files(root: Path, pattern: str | None) -> Iterable[Path]:
    if root.is_file():
        if root.suffix.lower() in SUPPORTED_SUFFIXES:
            yield root
        return

    if not root.exists():
        return

    if pattern:
        candidates = root.glob(pattern)
    else:
        candidates = root.rglob("*")

    for candidate in candidates:
        if candidate.is_file() and candidate.suffix.lower() in SUPPORTED_SUFFIXES:
            yield candidate
