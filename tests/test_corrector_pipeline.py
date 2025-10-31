from __future__ import annotations

import unittest

from src.lib.asr.models import TranscriptionResult
from src.lib.corrector import (
    CorrectionOptions,
    CorrectionPatch,
    CorrectionPipeline,
    CorrectionRequest,
    CorrectionSpan,
    EditorConfig,
    GlossaryEditor,
    apply_patches,
)
from src.lib.corrector.integration import apply_corrections_to_results


class TestCorrectorPipeline(unittest.TestCase):
    def test_apply_patches_inserts_before_trailing_whitespace(self) -> None:
        patch = CorrectionPatch(
            span=CorrectionSpan(start=3, end=3),
            replacement="。",
            tags=("PUNCT",),
            confidence=0.6,
        )
        original = "テスト  "
        corrected = apply_patches(original, [patch])
        self.assertEqual(corrected, "テスト。  ")

    def test_pipeline_inserts_sentence_terminator_when_aggressive(self) -> None:
        pipeline = CorrectionPipeline()
        request = CorrectionRequest.from_raw(
            "これはテスト",
            context_prev=None,
            language="ja",
            options=CorrectionOptions(aggressive_kuten=True),
        )

        result = pipeline.run(request)

        self.assertTrue(result.corrected_text.endswith("。"))
        self.assertEqual(len(result.patches), 1)
        patch = result.patches[0]
        self.assertEqual(patch.replacement, "。")
        self.assertEqual(patch.span.start, len("これはテスト"))
        self.assertEqual(patch.span.end, len("これはテスト"))

    def test_apply_corrector_updates_transcription_result_text(self) -> None:
        transcription = TranscriptionResult(
            filename="sample.wav",
            text="こんにちは",
            language="ja",
            segments=[],
        )

        updated, correction_map = apply_corrections_to_results(
            [transcription],
            language_hint="ja",
            options=CorrectionOptions(aggressive_kuten=True),
        )

        self.assertEqual(updated[0].text, "こんにちは。")
        correction = correction_map["sample.wav"]
        self.assertEqual(correction.corrected_text, "こんにちは。")
        self.assertTrue(correction.patches)

    def test_apply_corrector_keeps_existing_terminator(self) -> None:
        transcription = TranscriptionResult(
            filename="sample.wav",
            text="了解。",
            language="ja",
            segments=[],
        )

        updated, correction_map = apply_corrections_to_results(
            [transcription],
            language_hint="ja",
            options=CorrectionOptions(aggressive_kuten=True),
        )

        self.assertEqual(updated[0].text, "了解。")
        self.assertFalse(correction_map["sample.wav"].patches)

    def test_pipeline_applies_editor_when_enabled(self) -> None:
        editor = GlossaryEditor(EditorConfig(glossary={"かな": "仮名"}))
        pipeline = CorrectionPipeline(editor=editor)
        request = CorrectionRequest.from_raw(
            "ひらがなかな",
            options=CorrectionOptions(enable_editor=True),
        )
        result = pipeline.run(request)
        self.assertEqual(result.corrected_text, "ひらがな仮名")
        self.assertTrue(result.is_modified())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
