import unittest

from src.lib.asr.models import TranscriptionResult, TranscriptionSegment
from src.lib.asr.quality import analyze_transcription_quality


class TestASRQuality(unittest.TestCase):
    def test_flags_low_confidence_segment(self) -> None:
        result = TranscriptionResult(
            filename="x",
            text="hello",
            language="ja",
            segments=[
                TranscriptionSegment(
                    start=0.0,
                    end=0.5,
                    text="hello",
                    avg_logprob=-1.5,
                    compression_ratio=3.0,
                    no_speech_prob=0.2,
                )
            ],
        )
        diagnostics = analyze_transcription_quality(result)
        self.assertIn("low_confidence_segments", diagnostics.flags)
        self.assertEqual(len(diagnostics.flagged_segments), 1)
        self.assertIn("avg_logprob_low", diagnostics.flagged_segments[0].reasons)
        self.assertIn("compression_ratio_high", diagnostics.flagged_segments[0].reasons)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
