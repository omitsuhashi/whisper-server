from __future__ import annotations

import unittest

from src.lib.asr.models import TranscriptionSegment
from src.lib.polish.main import polish_text_from_segments
from src.lib.polish.options import PolishOptions
from src.lib.polish.segmentation import split_with_heuristics


class TestPolishSegmentation(unittest.TestCase):
    def setUp(self) -> None:
        self.options = PolishOptions()

    def test_split_with_punctuation_without_tail_pattern(self) -> None:
        text = (
            "今朝の会議まとめ:検討進捗ほぼみたいな感じで進めることにってその後スケジュール再確認しておくべきって"
            "言ってました。音声認識した文章の構成がうまくできていないのが特に大きな課題のようです。"
        )

        sentences = split_with_heuristics(text, self.options)

        self.assertEqual(len(sentences), 2)
        self.assertTrue(sentences[0].endswith("。"))
        self.assertTrue(sentences[1].endswith("。"))

    def test_polish_text_from_segments_preserves_sentence_count(self) -> None:
        segments = [
            TranscriptionSegment(start=0.0, end=5.1, text="今朝の会議まとめ:検討進捗ほぼみたいな感じで進めることにってその後スケジュール再確認しておくべきって言ってました。"),
            TranscriptionSegment(start=5.1, end=10.2, text="音声認識した文章の構成がうまくできていないのが特に大きな課題のようです。"),
        ]

        sentences = polish_text_from_segments(segments, options=self.options)

        self.assertEqual(len(sentences), 2)
        self.assertEqual(sentences[0].text.strip().endswith("。"), True)
        self.assertEqual(sentences[1].text.strip().endswith("。"), True)


if __name__ == "__main__":
    unittest.main()
