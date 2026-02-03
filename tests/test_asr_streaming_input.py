import io
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

from src.lib.asr.streaming_input import PcmRingBuffer, PcmStreamReader


class TestPcmStreamReader(unittest.TestCase):
    def test_reader_handles_odd_bytes_across_reads(self) -> None:
        samples = np.array([0, 1000], dtype=np.int16)
        stream = io.BytesIO(samples.tobytes())
        reader = PcmStreamReader(stream, sample_rate=16000, target_sample_rate=16000)

        first, eof1 = reader.read_waveform(3)
        second, eof2 = reader.read_waveform(3)

        self.assertFalse(eof1)
        self.assertFalse(eof2)
        self.assertEqual(first.size + second.size, 2)


class TestPcmRingBuffer(unittest.TestCase):
    def test_ring_buffer_keeps_last_samples(self) -> None:
        ring = PcmRingBuffer(max_samples=3)
        ring.append(np.array([1.0, 2.0], dtype=np.float32))
        ring.append(np.array([3.0, 4.0], dtype=np.float32))

        self.assertEqual(ring.total_samples, 4)
        self.assertEqual(ring.samples.tolist(), [2.0, 3.0, 4.0])


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
