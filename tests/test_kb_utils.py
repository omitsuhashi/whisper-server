from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.lib.kb.dedup import simhash_hex
from src.lib.kb.ingest import SUPPORTED_SUFFIXES, _discover_files, load_text
from src.lib.kb.normalize import semantic_chunks


class TestKbUtils(unittest.TestCase):
    def test_load_text_plain(self) -> None:
        with TemporaryDirectory() as tmpdir:
            text_path = Path(tmpdir) / "sample.md"
            text_path.write_text("# heading\n本文です。", encoding="utf-8")
            content = load_text(text_path)
            self.assertIn("heading", content)
            self.assertIn("本文", content)

    def test_load_text_html(self) -> None:
        with TemporaryDirectory() as tmpdir:
            html_path = Path(tmpdir) / "sample.html"
            html_path.write_text("<html><body><p>hello</p></body></html>", encoding="utf-8")
            content = load_text(html_path)
            self.assertEqual("hello", content)

    def test_discover_files_filters_suffix(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            valid = root / "doc.txt"
            valid.write_text("ok", encoding="utf-8")
            (root / "ignore.jpg").write_text("x", encoding="utf-8")
            discovered = list(_discover_files(root, None))
            self.assertEqual([valid], discovered)

    def test_simhash_consistency(self) -> None:
        text = "A quick test string"
        self.assertEqual(simhash_hex(text), simhash_hex(text))

    def test_semantic_chunks_returns_text(self) -> None:
        chunks = semantic_chunks("foo bar baz", language="en", max_chars=10)
        self.assertGreaterEqual(len(chunks), 1)
        self.assertTrue(all(isinstance(chunk, str) for chunk in chunks))


if __name__ == "__main__":
    unittest.main()
