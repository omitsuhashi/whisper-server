import unittest
from unittest import mock

import numpy as np

from src.lib.asr.chunking import transcribe_waveform_chunked
from src.lib.asr.models import TranscriptionResult, TranscriptionSegment
from src.lib.asr.options import TranscribeOptions


class TestWaveformChunking(unittest.TestCase):
    @mock.patch("src.lib.asr.chunking.detect_voice_segments", return_value=[])
    def test_decode_options_forwarded_when_chunked(self, mock_detect: mock.Mock) -> None:
        captured: dict[str, object] = {}

        def fake_transcribe(blobs, *, model_name, language, task, names, **decode_options):
            captured["decode"] = dict(decode_options)
            captured["names"] = list(names)
            return [
                TranscriptionResult(
                    filename=name,
                    text="ok",
                    language=language,
                    segments=[TranscriptionSegment(start=0.0, end=0.2, text="ok")],
                )
                for name in names
            ]

        wave = np.zeros(16000 * 2, dtype=np.float32)
        options = TranscribeOptions(
            model_name="demo",
            language="ja",
            task=None,
            decode_options={"initial_prompt": "foo"},
        )
        result = transcribe_waveform_chunked(
            wave,
            options=options,
            name="pcm",
            chunk_seconds=1.0,
            overlap_seconds=0.0,
            transcribe_all_bytes_fn=fake_transcribe,
        )

        decode = captured.get("decode")
        names = captured.get("names")
        self.assertIsInstance(decode, dict)
        self.assertIsInstance(names, list)
        self.assertEqual(decode.get("initial_prompt"), "foo")
        self.assertEqual(len(names), 2)
        self.assertEqual(result.text, "okok")
        mock_detect.assert_called_once()

    @mock.patch("src.lib.asr.chunking.detect_voice_segments", return_value=[])
    def test_chunk_merge_combines_segments(self, mock_detect: mock.Mock) -> None:
        def fake_transcribe(blobs, *, model_name, language, task, names, **decode_options):
            results = []
            for idx, name in enumerate(names):
                text = "A" if idx == 0 else "B"
                results.append(
                    TranscriptionResult(
                        filename=name,
                        text=text,
                        language=language,
                        segments=[TranscriptionSegment(start=0.0, end=0.4, text=text)],
                    )
                )
            return results

        wave = np.zeros(16000 * 2, dtype=np.float32)
        options = TranscribeOptions(model_name="demo", language="ja", task=None)
        result = transcribe_waveform_chunked(
            wave,
            options=options,
            name="pcm",
            chunk_seconds=1.0,
            overlap_seconds=0.0,
            transcribe_all_bytes_fn=fake_transcribe,
        )

        self.assertEqual(result.text, "AB")
        self.assertEqual(len(result.segments), 2)
        self.assertAlmostEqual(result.segments[0].start, 0.0, places=3)
        self.assertAlmostEqual(result.segments[1].start, 1.0, places=3)
        mock_detect.assert_called_once()


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
