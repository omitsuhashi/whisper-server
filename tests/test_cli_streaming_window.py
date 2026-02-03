import importlib
import io
import sys
import types
import unittest
from unittest import mock

import numpy as np

# mlx_whisper の初期化を避けるためのスタブ
original_mlx = sys.modules.get("mlx_whisper")
original_mlx_audio = sys.modules.get("mlx_whisper.audio")

mlx_stub = types.ModuleType("mlx_whisper")
audio_stub = types.ModuleType("mlx_whisper.audio")
audio_stub.SAMPLE_RATE = 16000
mlx_stub.audio = audio_stub
mlx_stub.transcribe = lambda *args, **kwargs: {}
sys.modules["mlx_whisper.audio"] = audio_stub
sys.modules["mlx_whisper"] = mlx_stub
sys.modules.setdefault("cv2", types.ModuleType("cv2"))
sys.modules.setdefault("torch", types.ModuleType("torch"))

from src.cmd import cli
from src.lib.asr import TranscriptionResult, TranscriptionSegment


class FakeStdin(types.SimpleNamespace):
    def __init__(self, data: bytes):
        super().__init__(buffer=_FakeBuffer(data))


class _FakeBuffer(io.BytesIO):
    def __init__(self, data: bytes):
        super().__init__(data)

    def isatty(self) -> bool:
        return False


class CliStreamingWindowTests(unittest.TestCase):
    def test_streaming_transcription_commits_segments(self) -> None:
        sr = 16000
        window_seconds = 0.01
        window_samples = int(sr * window_seconds)
        samples = np.arange(window_samples * 2, dtype=np.int16)
        fake_stdin = FakeStdin(samples.tobytes())

        call_count = {"count": 0}

        def fake_transcribe_waveform(waveform, *, options, name):
            call_count["count"] += 1
            text = "A" if call_count["count"] == 1 else "B"
            segment = TranscriptionSegment(start=0.0, end=0.005, text=text)
            return TranscriptionResult(filename=name, text=text, segments=[segment], language="ja")

        pipeline = importlib.import_module("src.lib.asr.pipeline")
        with mock.patch.object(pipeline, "transcribe_waveform", side_effect=fake_transcribe_waveform):
            with mock.patch.object(sys, "stdin", fake_stdin):
                results = cli._streaming_transcription(
                    model_name="fake",
                    language="ja",
                    task=None,
                    name="stdin",
                    chunk_size=window_samples * 2,
                    interval=0.0,
                    transcribe_all_bytes_fn=lambda *args, **kwargs: [],
                    decode_options={},
                    emit_stdout=False,
                    window_seconds=window_seconds,
                    lookback_seconds=0.0,
                    stream_input="pcm",
                    stream_sample_rate=sr,
                )

        self.assertEqual(results[0].text, "AB")

    def test_streaming_transcription_sets_condition_default(self) -> None:
        sr = 16000
        window_seconds = 0.01
        window_samples = int(sr * window_seconds)
        samples = np.zeros(window_samples, dtype=np.int16)
        fake_stdin = FakeStdin(samples.tobytes())
        captured: dict[str, bool] = {}

        def fake_transcribe_waveform(waveform, *, options, name):
            captured["condition"] = options.decode_options.get("condition_on_previous_text")
            return TranscriptionResult(filename=name, text="", segments=[], language="ja")

        pipeline = importlib.import_module("src.lib.asr.pipeline")
        with mock.patch.object(pipeline, "transcribe_waveform", side_effect=fake_transcribe_waveform):
            with mock.patch.object(sys, "stdin", fake_stdin):
                cli._streaming_transcription(
                    model_name="fake",
                    language="ja",
                    task=None,
                    name="stdin",
                    chunk_size=window_samples * 2,
                    interval=0.0,
                    transcribe_all_bytes_fn=lambda *args, **kwargs: [],
                    decode_options={},
                    emit_stdout=False,
                    window_seconds=window_seconds,
                    lookback_seconds=0.0,
                    stream_input="pcm",
                    stream_sample_rate=sr,
                )

        self.assertFalse(captured["condition"])


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
