from __future__ import annotations

import unittest

from src.lib.corrector import EditorConfig, GlossaryEditor
from src.lib.corrector.types import CorrectionOptions, CorrectionRequest


class TestGlossaryEditor(unittest.TestCase):
    def test_edit_applies_glossary(self) -> None:
        editor = GlossaryEditor(EditorConfig(glossary={"かな": "仮名"}))
        request = CorrectionRequest.from_raw(
            "これはかなのテストです。",
            options=CorrectionOptions(enable_editor=True),
        )
        result = editor.edit(request)
        self.assertEqual(result, "これは仮名のテストです。")

    def test_disabled_editor_returns_original(self) -> None:
        editor = GlossaryEditor(EditorConfig(glossary={"かな": "仮名"}, enabled=False))
        request = CorrectionRequest.from_raw(
            "かな",
            options=CorrectionOptions(enable_editor=True),
        )
        result = editor.edit(request)
        self.assertEqual(result, "かな")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
