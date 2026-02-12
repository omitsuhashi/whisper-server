import unittest
from unittest import mock

import numpy as np

from src.lib.asr.chunking import _merge_results, transcribe_waveform_chunked
from src.lib.asr.models import TranscriptionResult, TranscriptionSegment
from src.lib.asr.options import TranscribeOptions
from src.lib.vad import SpeechSegment


class TestAsrChunking(unittest.TestCase):
    @mock.patch("src.lib.asr.chunking.detect_voice_segments")
    @mock.patch("src.lib.asr.chunking._phase_run_asr_waveform")
    def test_vad_boundaries_have_margin(
        self,
        mock_run_asr: mock.Mock,
        mock_detect: mock.Mock,
    ) -> None:
        sample_rate = 16000
        waveform = np.zeros(sample_rate * 30, dtype=np.float32)
        mock_detect.return_value = [
            SpeechSegment(
                start=10.0,
                end=11.0,
                start_sample=sample_rate * 10,
                end_sample=sample_rate * 11,
            )
        ]

        def _fake_run(chunk_windows, waveform_arg, *, filename, options):
            self.assertEqual(len(chunk_windows), 1)
            raw_start, raw_end, main_start, main_end = chunk_windows[0]
            self.assertEqual(raw_start, sample_rate * 9)
            self.assertEqual(raw_end, sample_rate * 12)
            self.assertEqual(main_start, sample_rate * 9)
            self.assertEqual(main_end, sample_rate * 12)
            self.assertEqual(len(waveform_arg), len(waveform))
            return [
                TranscriptionResult(
                    filename=f"{filename}#chunk1",
                    text="",
                    language="ja",
                    duration=0.0,
                    segments=[],
                )
            ]

        mock_run_asr.side_effect = _fake_run
        result = transcribe_waveform_chunked(
            waveform,
            options=TranscribeOptions(model_name="mock-model", language="ja"),
            name="sample.pcm",
            chunk_seconds=25.0,
            overlap_seconds=1.0,
        )
        self.assertEqual(result.filename, "sample.pcm")

    def test_merge_prefers_better_boundary_candidate(self) -> None:
        sample_rate = 16000
        windows = [
            (0 * sample_rate, 26 * sample_rate, 0 * sample_rate, 25 * sample_rate),
            (24 * sample_rate, 51 * sample_rate, 25 * sample_rate, 50 * sample_rate),
        ]
        partials = [
            TranscriptionResult(
                filename="chunk1",
                text="",
                language="ja",
                segments=[
                    TranscriptionSegment(
                        start=24.6,
                        end=25.2,
                        text="きょうわ天気です",
                        avg_logprob=-1.4,
                        no_speech_prob=0.62,
                        compression_ratio=2.8,
                    )
                ],
            ),
            TranscriptionResult(
                filename="chunk2",
                text="",
                language="ja",
                segments=[
                    TranscriptionSegment(
                        start=0.6,
                        end=1.2,
                        text="きょうは天気です",
                        avg_logprob=-0.2,
                        no_speech_prob=0.05,
                        compression_ratio=1.2,
                    )
                ],
            ),
        ]

        merged = _merge_results(partials, chunk_windows=windows, filename="sample.pcm", language="ja")
        self.assertEqual(merged.text, "きょうは天気です")
        self.assertEqual(len(merged.segments), 1)
        self.assertEqual(merged.segments[0].text, "きょうは天気です")

    def test_merge_keeps_contained_phrases_inside_same_window(self) -> None:
        sample_rate = 16000
        windows = [(0, sample_rate * 30, 0, sample_rate * 30)]
        partials = [
            TranscriptionResult(
                filename="chunk1",
                text="",
                language="ja",
                segments=[
                    TranscriptionSegment(start=10.0, end=10.2, text="はい"),
                    TranscriptionSegment(start=10.25, end=10.9, text="はい、お願いします"),
                ],
            )
        ]

        merged = _merge_results(partials, chunk_windows=windows, filename="sample.pcm", language="ja")
        self.assertEqual(len(merged.segments), 2)
        self.assertEqual(merged.text, "はいはい、お願いします")

    def test_merge_keeps_contained_phrases_near_non_overlap_window_edge(self) -> None:
        sample_rate = 16000
        windows = [(0, sample_rate * 30, 0, sample_rate * 30)]
        partials = [
            TranscriptionResult(
                filename="chunk1",
                text="",
                language="ja",
                segments=[
                    TranscriptionSegment(start=0.05, end=0.2, text="はい"),
                    TranscriptionSegment(start=0.25, end=0.9, text="はい、お願いします"),
                ],
            )
        ]

        merged = _merge_results(partials, chunk_windows=windows, filename="sample.pcm", language="ja")
        self.assertEqual(len(merged.segments), 2)
        self.assertEqual(merged.text, "はいはい、お願いします")

    def test_merge_does_not_dedup_similar_text_in_window_center(self) -> None:
        sample_rate = 16000
        windows = [(0, sample_rate * 30, 0, sample_rate * 30)]
        partials = [
            TranscriptionResult(
                filename="chunk1",
                text="",
                language="ja",
                segments=[
                    TranscriptionSegment(start=10.0, end=10.5, text="きょうは天気です"),
                    TranscriptionSegment(start=10.55, end=11.1, text="きょうも天気です"),
                ],
            )
        ]

        merged = _merge_results(partials, chunk_windows=windows, filename="sample.pcm", language="ja")
        self.assertEqual(len(merged.segments), 2)
        self.assertEqual(merged.text, "きょうは天気ですきょうも天気です")

    def test_merge_contained_threshold_can_be_controlled_by_env(self) -> None:
        sample_rate = 16000
        windows = [(0, sample_rate * 30, 0, sample_rate * 30)]
        partials = [
            TranscriptionResult(
                filename="chunk1",
                text="",
                language="ja",
                segments=[
                    TranscriptionSegment(start=0.2, end=0.4, text="abc"),
                    TranscriptionSegment(start=0.45, end=1.0, text="abcxyz"),
                ],
            )
        ]

        with mock.patch.dict("os.environ", {"ASR_CONTAINED_MATCH_EDGE_MARGIN_SECONDS": "0.1"}, clear=False):
            merged = _merge_results(partials, chunk_windows=windows, filename="sample.pcm", language="ja")

        self.assertEqual(len(merged.segments), 2)
        self.assertEqual(merged.text, "abcabcxyz")

    def test_merge_logs_summary(self) -> None:
        sample_rate = 16000
        windows = [(0, sample_rate * 10, 0, sample_rate * 10)]
        partials = [
            TranscriptionResult(
                filename="chunk1",
                text="",
                language="ja",
                segments=[TranscriptionSegment(start=0.0, end=1.0, text="abc")],
            )
        ]
        with self.assertLogs("src.lib.asr.chunking", level="DEBUG") as captured:
            _merge_results(partials, chunk_windows=windows, filename="sample.pcm", language="ja")
        self.assertTrue(any("chunk_merge_summary" in line for line in captured.output))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
