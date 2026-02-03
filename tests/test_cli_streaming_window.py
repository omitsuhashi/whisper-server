import io
import sys
import types
import unittest
import wave
from contextlib import closing
from unittest import mock

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
sys.modules.setdefault("cv2", types.ModuleType("cv2"))
sys.modules.setdefault("torch", types.ModuleType("torch"))

from src.cmd import cli
from src.lib.asr import TranscriptionResult


class FakeStdin(types.SimpleNamespace):
    def __init__(self, data: bytes):
        super().__init__(buffer=_FakeBuffer(data))


class _FakeBuffer(io.BytesIO):
    def __init__(self, data: bytes):
        super().__init__(data)

    def isatty(self) -> bool:
        return False


class CliStreamingWindowTests(unittest.TestCase):
    def test_streaming_transcription_uses_fixed_window_for_pcm(self) -> None:
        sr = 16000
        window_seconds = 0.01
        window_samples = int(sr * window_seconds)
        samples = np.arange(window_samples * 3, dtype=np.int16)
        fake_stdin = FakeStdin(samples.tobytes())
        captured: dict[str, bytes] = {}

        def fake_transcribe_all_bytes(payloads, **kwargs):
            captured["payload"] = payloads[0]
            return [TranscriptionResult(filename="stdin", text="ok")]

        with mock.patch.object(sys, "stdin", fake_stdin):
            cli._streaming_transcription(
                model_name="fake",
                language="ja",
                task=None,
                name="stdin",
                chunk_size=window_samples * 2,
                interval=0.0,
                transcribe_all_bytes_fn=fake_transcribe_all_bytes,
                decode_options={},
                emit_stdout=False,
                window_seconds=window_seconds,
                stream_input="pcm",
                stream_sample_rate=sr,
            )

        with closing(wave.open(io.BytesIO(captured["payload"]), "rb")) as wav:
            self.assertLessEqual(wav.getnframes(), window_samples)


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
