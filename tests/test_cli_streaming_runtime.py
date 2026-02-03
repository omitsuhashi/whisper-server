import sys
import types
import unittest

import numpy as np

# mlx_whisper の初期化を避けるためのスタブ
original_mlx = sys.modules.get("mlx_whisper")
original_mlx_audio = sys.modules.get("mlx_whisper.audio")

mlx_stub = types.ModuleType("mlx_whisper")
audio_stub = types.ModuleType("mlx_whisper.audio")
audio_stub.SAMPLE_RATE = 16000
mlx_stub.audio = audio_stub
sys.modules["mlx_whisper.audio"] = audio_stub
sys.modules["mlx_whisper"] = mlx_stub

from src.lib.asr.models import TranscriptionResult, TranscriptionSegment
from src.lib.asr.streaming_commit import SegmentCommitter
from src.lib.asr.streaming_input import PcmRingBuffer
from src.lib.asr.streaming_runtime import run_streaming_loop


def _make_reader(chunks):
    iterator = iter(chunks)

    def _reader(_chunk_size: int):
        try:
            return next(iterator), False
        except StopIteration:
            return np.zeros(0, dtype=np.float32), True

    return _reader


class TestStreamingRuntime(unittest.TestCase):
    def test_run_streaming_loop_limits_window(self) -> None:
        ring = PcmRingBuffer(max_samples=3)
        committer = SegmentCommitter(lookback_seconds=0.0)
        chunks = [
            np.array([0.1, 0.2], dtype=np.float32),
            np.array([0.3, 0.4], dtype=np.float32),
        ]
        sizes: list[int] = []

        def fake_transcribe(waveform: np.ndarray) -> TranscriptionResult:
            sizes.append(int(waveform.size))
            segment = TranscriptionSegment(start=0.0, end=0.5, text="a")
            return TranscriptionResult(filename="stdin", text="a", segments=[segment], language="ja")

        final = run_streaming_loop(
            read_waveform=_make_reader(chunks),
            chunk_size=2,
            interval=0.0,
            ring=ring,
            committer=committer,
            transcribe_fn=fake_transcribe,
            emit_fn=None,
            finalize_fn=lambda: committer.build_result(filename="stdin", language="ja"),
            target_sample_rate=1,
        )

        self.assertIsNotNone(final)
        self.assertTrue(all(size <= 3 for size in sizes))

    def test_run_streaming_loop_emits_incremental_text(self) -> None:
        ring = PcmRingBuffer(max_samples=10)
        committer = SegmentCommitter(lookback_seconds=0.0)
        chunks = [
            np.array([0.1, 0.2], dtype=np.float32),
            np.array([0.3, 0.4], dtype=np.float32),
        ]
        emitted: list[str] = []
        call_count = {"count": 0}

        def fake_transcribe(waveform: np.ndarray) -> TranscriptionResult:
            call_count["count"] += 1
            text = "A" if call_count["count"] == 1 else "B"
            end = 0.5 + (call_count["count"] - 1) * 0.5
            segment = TranscriptionSegment(start=0.0, end=end, text=text)
            return TranscriptionResult(filename="stdin", text=text, segments=[segment], language="ja")

        final = run_streaming_loop(
            read_waveform=_make_reader(chunks),
            chunk_size=2,
            interval=0.0,
            ring=ring,
            committer=committer,
            transcribe_fn=fake_transcribe,
            emit_fn=emitted.append,
            finalize_fn=lambda: committer.build_result(filename="stdin", language="ja"),
            target_sample_rate=1,
        )

        self.assertEqual(emitted, ["A", "B"])
        self.assertIsNotNone(final)
        self.assertEqual(final.text, "AB")


if __name__ == "__main__":
    if original_mlx is not None:
        sys.modules["mlx_whisper"] = original_mlx
    else:
        sys.modules.pop("mlx_whisper", None)

    if original_mlx_audio is not None:
        sys.modules["mlx_whisper.audio"] = original_mlx_audio
    else:
        sys.modules.pop("mlx_whisper.audio", None)

    unittest.main()
