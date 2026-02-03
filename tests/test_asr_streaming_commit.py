import unittest

from src.lib.asr.models import TranscriptionResult, TranscriptionSegment
from src.lib.asr.streaming_commit import SegmentCommitter


def make_result(segments: list[TranscriptionSegment]) -> TranscriptionResult:
    return TranscriptionResult(filename="pcm", text="", language="ja", segments=segments)


class TestStreamingCommitter(unittest.TestCase):
    def test_commit_respects_lookback(self) -> None:
        committer = SegmentCommitter(lookback_seconds=1.0)
        result = make_result(
            [
                TranscriptionSegment(start=8.0, end=9.0, text="A"),
                TranscriptionSegment(start=9.0, end=10.0, text="B"),
            ]
        )
        new_text = committer.update(
            result,
            window_start_seconds=0.0,
            now_total_seconds=10.0,
            final=False,
        )
        self.assertEqual(new_text, "A")
        self.assertEqual(len(committer.committed_segments), 1)

        new_text_final = committer.update(
            result,
            window_start_seconds=0.0,
            now_total_seconds=10.0,
            final=True,
        )
        self.assertEqual(new_text_final, "B")
        self.assertEqual(len(committer.committed_segments), 2)

    def test_commit_converts_to_absolute_time(self) -> None:
        committer = SegmentCommitter(lookback_seconds=0.0)
        result = make_result([TranscriptionSegment(start=0.0, end=1.0, text="X")])
        committer.update(
            result,
            window_start_seconds=5.0,
            now_total_seconds=6.0,
            final=True,
        )
        self.assertEqual(committer.committed_segments[0].start, 5.0)
        self.assertEqual(committer.committed_segments[0].end, 6.0)

    def test_commit_skips_duplicate_text_with_small_gap(self) -> None:
        committer = SegmentCommitter(lookback_seconds=0.0)
        first = make_result([TranscriptionSegment(start=0.0, end=1.0, text="Hi")])
        committer.update(
            first,
            window_start_seconds=0.0,
            now_total_seconds=1.0,
            final=True,
        )
        second = make_result([TranscriptionSegment(start=0.1, end=0.9, text="Hi")])
        new_text = committer.update(
            second,
            window_start_seconds=1.1,
            now_total_seconds=2.0,
            final=True,
        )
        self.assertEqual(new_text, "")
        self.assertEqual(len(committer.committed_segments), 1)
        self.assertAlmostEqual(committer.committed_until, 2.0, places=6)

    def test_build_result_uses_committed_until_after_skip(self) -> None:
        committer = SegmentCommitter(lookback_seconds=0.0)
        first = make_result([TranscriptionSegment(start=0.0, end=1.0, text="Hi")])
        committer.update(
            first,
            window_start_seconds=0.0,
            now_total_seconds=1.0,
            final=True,
        )
        second = make_result([TranscriptionSegment(start=0.1, end=0.9, text="Hi")])
        committer.update(
            second,
            window_start_seconds=1.1,
            now_total_seconds=2.0,
            final=True,
        )
        built = committer.build_result(filename="pcm", language="ja")
        self.assertEqual(built.duration, 2.0)
        self.assertEqual(len(built.segments), 1)

    def test_build_result_uses_committed_text_and_duration(self) -> None:
        committer = SegmentCommitter(lookback_seconds=0.0)
        result = make_result([TranscriptionSegment(start=0.0, end=1.0, text="A")])
        committer.update(
            result,
            window_start_seconds=0.0,
            now_total_seconds=1.0,
            final=True,
        )
        built = committer.build_result(filename="pcm", language="ja")
        self.assertEqual(built.text, "A")
        self.assertEqual(built.duration, 1.0)
        self.assertEqual(len(built.segments), 1)
