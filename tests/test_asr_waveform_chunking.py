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

        wave = np.full(16000 * 2, 0.01, dtype=np.float32)
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

        wave = np.full(16000 * 2, 0.01, dtype=np.float32)
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

    @mock.patch("src.lib.asr.chunking.detect_voice_segments", return_value=[])
    def test_chunk_merge_preserves_confidence_metrics(self, mock_detect: mock.Mock) -> None:
        def fake_transcribe(blobs, *, model_name, language, task, names, **decode_options):
            results = []
            for idx, name in enumerate(names):
                text = "A" if idx == 0 else "B"
                results.append(
                    TranscriptionResult(
                        filename=name,
                        text=text,
                        language=language,
                        segments=[
                            TranscriptionSegment(
                                start=0.0,
                                end=0.4,
                                text=text,
                                avg_logprob=-0.5,
                                compression_ratio=1.2,
                                no_speech_prob=0.1,
                            )
                        ],
                    )
                )
            return results

        wave = np.full(16000 * 2, 0.01, dtype=np.float32)
        options = TranscribeOptions(model_name="demo", language="ja", task=None)
        result = transcribe_waveform_chunked(
            wave,
            options=options,
            name="pcm",
            chunk_seconds=1.0,
            overlap_seconds=0.0,
            transcribe_all_bytes_fn=fake_transcribe,
        )

        self.assertEqual(result.segments[0].avg_logprob, -0.5)
        self.assertEqual(result.segments[0].compression_ratio, 1.2)
        self.assertEqual(result.segments[0].no_speech_prob, 0.1)
        mock_detect.assert_called_once()

    @mock.patch("src.lib.asr.chunking.detect_voice_segments", return_value=[])
    def test_chunk_merge_keeps_zero_duration_for_all_silent_partials(self, mock_detect: mock.Mock) -> None:
        def fake_transcribe(blobs, *, model_name, language, task, names, **decode_options):
            return [
                TranscriptionResult(
                    filename=name,
                    text="",
                    language=language,
                    duration=0.0,
                    segments=[],
                )
                for name in names
            ]

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

        self.assertEqual(result.duration, 0.0)
        self.assertEqual(result.text, "")
        self.assertEqual(result.segments, [])
        mock_detect.assert_called_once()

    @mock.patch("src.lib.asr.chunking.detect_voice_segments", return_value=[])
    def test_bytes_path_skips_silent_chunks(self, mock_detect: mock.Mock) -> None:
        captured: dict[str, object] = {}

        def fake_transcribe(blobs, *, model_name, language, task, names, **decode_options):
            captured["blobs"] = list(blobs)
            captured["names"] = list(names)
            return [
                TranscriptionResult(
                    filename=names[0],
                    text="B",
                    language=language,
                    segments=[TranscriptionSegment(start=0.0, end=0.4, text="B")],
                )
            ]

        wave = np.concatenate(
            [
                np.zeros(16000, dtype=np.float32),
                np.full(16000, 0.01, dtype=np.float32),
            ]
        )
        options = TranscribeOptions(model_name="demo", language="ja", task=None)
        result = transcribe_waveform_chunked(
            wave,
            options=options,
            name="pcm",
            chunk_seconds=1.0,
            overlap_seconds=0.0,
            transcribe_all_bytes_fn=fake_transcribe,
        )

        self.assertEqual(result.text, "B")
        self.assertEqual(captured.get("names"), ["pcm#chunk2"])
        self.assertEqual(len(captured.get("blobs", [])), 1)
        mock_detect.assert_called_once()

    @mock.patch("src.lib.asr.chunking.detect_voice_segments", return_value=[])
    @mock.patch("src.lib.asr.chunking._is_waveform_silent", return_value=False)
    @mock.patch("src.lib.asr.chunking._transcribe_single")
    @mock.patch("src.lib.asr.chunking.encode_waveform_to_wav_bytes")
    def test_chunked_waveform_avoids_encode_when_no_bytes_fn(
        self,
        mock_encode: mock.Mock,
        mock_transcribe_single: mock.Mock,
        mock_silent: mock.Mock,
        mock_detect: mock.Mock,
    ) -> None:
        mock_transcribe_single.side_effect = [
            TranscriptionResult(
                filename="pcm#chunk1",
                text="A",
                language="ja",
                segments=[TranscriptionSegment(start=0.0, end=0.4, text="A")],
            ),
            TranscriptionResult(
                filename="pcm#chunk2",
                text="B",
                language="ja",
                segments=[TranscriptionSegment(start=0.0, end=0.4, text="B")],
            ),
        ]

        wave = np.zeros(16000 * 2, dtype=np.float32)
        options = TranscribeOptions(model_name="demo", language="ja", task=None)
        result = transcribe_waveform_chunked(
            wave,
            options=options,
            name="pcm",
            chunk_seconds=1.0,
            overlap_seconds=0.0,
        )

        self.assertEqual(result.text, "AB")
        self.assertEqual(len(result.segments), 2)
        mock_encode.assert_not_called()
        mock_transcribe_single.assert_called()
        mock_silent.assert_called()
        mock_detect.assert_called_once()


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
