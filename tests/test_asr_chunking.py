import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tests.test_asr_stream import _generate_wav_bytes
from src.lib.asr.chunking import transcribe_paths_chunked
from src.lib.asr.models import TranscriptionResult
from src.lib.vad import SpeechSegment


class ChunkingDecodeOptionsTests(unittest.TestCase):
    def test_decode_options_forwarded_to_transcribe_all_bytes(self) -> None:
        captured: dict | None = None

        def fake_transcribe(blobs, *, model_name, language, task, names, **decode_options):
            nonlocal captured
            captured = decode_options
            return [
                TranscriptionResult(
                    filename=names[0],
                    text="ok",
                    language=language,
                    segments=[],
                )
            ]

        with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
            tmp.write(_generate_wav_bytes())
            tmp.flush()
            results = transcribe_paths_chunked(
                [tmp.name],
                model_name="demo",
                language="ja",
                task=None,
                chunk_seconds=0,
                overlap_seconds=0,
                transcribe_all_bytes_fn=fake_transcribe,
                initial_prompt="foo",
            )

        self.assertIsNotNone(captured)
        assert captured is not None
        self.assertEqual(captured.get("initial_prompt"), "foo")
        self.assertEqual(results[0].text, "ok")


class ChunkingVadPhaseTests(unittest.TestCase):
    def setUp(self) -> None:
        if shutil.which("ffmpeg") is None:
            self.skipTest("ffmpeg が見つかりません")

    @mock.patch("src.lib.asr.chunking.detect_voice_segments")
    def test_vad_segments_limit_chunk_dispatch(self, mock_detect: mock.Mock) -> None:
        sample_rate = 16000
        half_sec = int(sample_rate * 0.5)
        mock_detect.return_value = [
            SpeechSegment(start=0.0, end=0.5, start_sample=0, end_sample=half_sec),
            SpeechSegment(start=1.0, end=1.5, start_sample=sample_rate, end_sample=sample_rate + half_sec),
        ]

        chunk_names: list[list[str]] = []

        def fake_transcribe(blobs, *, model_name, language, task, names, **decode_options):
            chunk_names.append(list(names))
            return [
                TranscriptionResult(
                    filename=name,
                    text=name,
                    language=language,
                    segments=[],
                )
                for name in names
            ]

        audio_name = ""
        with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
            tmp.write(_generate_wav_bytes(duration_sec=2.0))
            tmp.flush()
            audio_name = Path(tmp.name).name
            transcribe_paths_chunked(
                [tmp.name],
                model_name="demo",
                language="ja",
                task=None,
                chunk_seconds=0.5,
                overlap_seconds=0.1,
                transcribe_all_bytes_fn=fake_transcribe,
            )

        self.assertEqual(len(chunk_names), 1)
        self.assertEqual(len(chunk_names[0]), 2)
        if chunk_names:
            self.assertEqual(chunk_names[0][0], f"{audio_name}#chunk1")
            self.assertEqual(chunk_names[0][1], f"{audio_name}#chunk2")
        mock_detect.assert_called_once()

    @mock.patch("src.lib.asr.chunking.detect_voice_segments", return_value=[])
    def test_vad_absence_falls_back_to_linear_chunking(self, mock_detect: mock.Mock) -> None:
        chunk_counts: list[int] = []

        def fake_transcribe(blobs, *, model_name, language, task, names, **decode_options):
            chunk_counts.append(len(blobs))
            return [
                TranscriptionResult(
                    filename=name,
                    text=name,
                    language=language,
                    segments=[],
                )
                for name in names
            ]

        with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
            tmp.write(_generate_wav_bytes(duration_sec=2.0))
            tmp.flush()
            transcribe_paths_chunked(
                [tmp.name],
                model_name="demo",
                language="ja",
                task=None,
                chunk_seconds=0.5,
                overlap_seconds=0.1,
                transcribe_all_bytes_fn=fake_transcribe,
            )

        self.assertEqual(chunk_counts, [4])
        mock_detect.assert_called_once()

    @mock.patch("src.lib.asr.chunking.detect_voice_segments")
    def test_vad_segments_overflow_falls_back_to_linear_chunking(self, mock_detect: mock.Mock) -> None:
        sample_rate = 16000
        segments = []
        for idx in range(20):
            start = idx * 0.1
            end = start + 0.05
            segments.append(
                SpeechSegment(
                    start=start,
                    end=end,
                    start_sample=int(start * sample_rate),
                    end_sample=int(end * sample_rate),
                )
            )
        mock_detect.return_value = segments

        chunk_counts: list[int] = []

        def fake_transcribe(blobs, *, model_name, language, task, names, **decode_options):
            chunk_counts.append(len(blobs))
            return [
                TranscriptionResult(
                    filename=name,
                    text=name,
                    language=language,
                    segments=[],
                )
                for name in names
            ]

        with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
            tmp.write(_generate_wav_bytes(duration_sec=2.0))
            tmp.flush()
            transcribe_paths_chunked(
                [tmp.name],
                model_name="demo",
                language="ja",
                task=None,
                chunk_seconds=0.5,
                overlap_seconds=0.1,
                transcribe_all_bytes_fn=fake_transcribe,
            )

        self.assertEqual(chunk_counts, [4])
        mock_detect.assert_called_once()


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
