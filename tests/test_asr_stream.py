import atexit
import io
import shutil
import sys
import types
import unittest
import wave
from contextlib import closing
from unittest import mock

import numpy as np
from fastapi.testclient import TestClient

# mlx_whisper の初期化を避けるためのスタブ
original_mlx = sys.modules.get("mlx_whisper")
original_mlx_audio = sys.modules.get("mlx_whisper.audio")

mlx_stub = types.ModuleType("mlx_whisper")
audio_stub = types.ModuleType("mlx_whisper.audio")
audio_stub.SAMPLE_RATE = 16000
mlx_stub.transcribe = lambda *args, **kwargs: None  # 上書き前提のダミー
mlx_stub.audio = audio_stub
sys.modules["mlx_whisper.audio"] = audio_stub
sys.modules["mlx_whisper"] = mlx_stub


def _restore_original_modules() -> None:
    if original_mlx is not None:
        sys.modules["mlx_whisper"] = original_mlx
    else:
        sys.modules.pop("mlx_whisper", None)

    if original_mlx_audio is not None:
        sys.modules["mlx_whisper.audio"] = original_mlx_audio
    else:
        sys.modules.pop("mlx_whisper.audio", None)


atexit.register(_restore_original_modules)

from src.cmd import cli
from src.cmd import http as http_cmd
from src.lib.asr import TranscriptionResult
from src.lib.asr import main as asr_main


def _generate_wav_bytes(duration_sec: float = 0.1) -> bytes:
    sample_rate = 16000
    total_samples = int(sample_rate * duration_sec)
    payload = (np.sin(np.linspace(0, np.pi, total_samples)) * 1000).astype(np.int16)

    buffer = io.BytesIO()
    with closing(wave.open(buffer, "wb")) as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(payload.tobytes())
    return buffer.getvalue()


class FakeStdin(types.SimpleNamespace):
    def __init__(self, data: bytes):
        super().__init__(buffer=_FakeBuffer(data))


class _FakeBuffer(io.BytesIO):
    def __init__(self, data: bytes):
        super().__init__(data)

    def isatty(self) -> bool:
        return False


class TranscribeAllBytesTests(unittest.TestCase):
    def setUp(self) -> None:
        if shutil.which("ffmpeg") is None:
            self.skipTest("ffmpeg が見つかりません")
        self.audio_bytes = _generate_wav_bytes()

    @mock.patch("src.lib.asr.pipeline.transcribe")
    def test_transcribe_all_bytes_decodes_and_dispatches(self, mock_transcribe: mock.Mock) -> None:
        mock_transcribe.return_value = {
            "text": "ok",
            "segments": [],
            "language": "ja",
            "duration": 0.1,
        }

        results = asr_main.transcribe_all_bytes(
            [self.audio_bytes],
            model_name="fake-model",
            language="ja",
        )

        self.assertEqual(len(results), 1)
        result = results[0]
        self.assertIsInstance(result, TranscriptionResult)
        self.assertEqual(result.text, "ok")
        self.assertEqual(result.filename, "stream_1")

        mock_transcribe.assert_called_once()
        call_args, call_kwargs = mock_transcribe.call_args
        self.assertIsInstance(call_args[0], np.ndarray)
        self.assertGreater(call_args[0].size, 0)
        self.assertEqual(call_kwargs["path_or_hf_repo"], "fake-model")
        self.assertEqual(call_kwargs["language"], "ja")


class CliStreamTests(unittest.TestCase):
    def setUp(self) -> None:
        if shutil.which("ffmpeg") is None:
            self.skipTest("ffmpeg が見つかりません")
        self.audio_bytes = _generate_wav_bytes()

    @mock.patch("src.cmd.cli._load_asr_components")
    def test_stream_subcommand_uses_stdin_payload(self, mock_load_asr: mock.Mock) -> None:
        fake_result = TranscriptionResult(filename="stdin", text="hello")
        mock_transcribe_all = mock.Mock()
        mock_transcribe_bytes = mock.Mock(return_value=[fake_result])
        mock_load_asr.return_value = (
            TranscriptionResult,
            mock_transcribe_all,
            mock_transcribe_bytes,
        )

        fake_stdin = FakeStdin(self.audio_bytes)
        original_stdin = cli.sys.stdin
        cli.sys.stdin = fake_stdin
        try:
            args = cli.parse_args(["stream", "--model", "fake-model"])
            results = cli.run_cli(args)
        finally:
            cli.sys.stdin = original_stdin

        mock_transcribe_bytes.assert_called_once()
        (payloads,), kwargs = mock_transcribe_bytes.call_args
        self.assertEqual(len(payloads), 1)
        self.assertEqual(payloads[0], self.audio_bytes)
        self.assertEqual(kwargs["model_name"], "fake-model")
        self.assertEqual(results, [fake_result])
        self.assertFalse(getattr(args, "_stream_output", False))

    @mock.patch("src.cmd.cli._load_asr_components")
    def test_stream_subcommand_with_interval_streams_output(
        self, mock_load_asr: mock.Mock
    ) -> None:
        first = TranscriptionResult(filename="stdin", text="こ")
        second = TranscriptionResult(filename="stdin", text="こんにちは")
        mock_transcribe_all = mock.Mock()
        mock_transcribe_bytes = mock.Mock(side_effect=[[first], [second], [second]])
        mock_load_asr.return_value = (
            TranscriptionResult,
            mock_transcribe_all,
            mock_transcribe_bytes,
        )

        fake_audio = b"abcdwxyz"
        fake_stdin = FakeStdin(fake_audio)

        original_stdin = cli.sys.stdin
        original_stdout = sys.stdout
        cli.sys.stdin = fake_stdin
        sys.stdout = io.StringIO()

        times = [0.0, 1.0, 2.0, 3.0]

        def fake_monotonic() -> float:
            return times.pop(0)

        with mock.patch.object(cli.time, "monotonic", side_effect=fake_monotonic):
            try:
                args = cli.parse_args(
                    [
                        "stream",
                        "--model",
                        "fake",
                        "--stream-interval",
                        "0.5",
                        "--stream-chunk-size",
                        "4",
                    ]
                )
                results = cli.run_cli(args)
            finally:
                cli.sys.stdin = original_stdin
                captured = sys.stdout.getvalue()
                sys.stdout = original_stdout

        self.assertEqual(captured.strip(), "こんにちは")
        self.assertTrue(getattr(args, "_stream_output", False))
        self.assertEqual(results, [second])
        self.assertEqual(mock_transcribe_bytes.call_count, 3)


class HttpRestTests(unittest.TestCase):
    def setUp(self) -> None:
        if shutil.which("ffmpeg") is None:
            self.skipTest("ffmpeg が見つかりません")

    @mock.patch("src.cmd.http.transcribe_all")
    def test_transcribe_endpoint_handles_multiple_files(self, mock_transcribe_all: mock.Mock) -> None:
        mock_transcribe_all.return_value = [
            TranscriptionResult(filename="tmp1.wav", text="hello", language="en"),
            TranscriptionResult(filename="tmp2.wav", text="こんにちは", language="ja"),
        ]

        app = http_cmd.create_app()
        client = TestClient(app)
        files = [
            ("files", ("sample1.wav", _generate_wav_bytes(), "audio/wav")),
            ("files", ("sample2.wav", _generate_wav_bytes(), "audio/wav")),
        ]

        response = client.post(
            "/transcribe",
            files=files,
            data={"model": "fake-model", "language": "ja"},
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(len(payload), 2)
        self.assertEqual(payload[0]["filename"], "sample1.wav")
        self.assertEqual(payload[1]["filename"], "sample2.wav")
        self.assertEqual(payload[1]["text"], "こんにちは")

        mock_transcribe_all.assert_called_once()
        args, kwargs = mock_transcribe_all.call_args
        self.assertEqual(len(args[0]), 2)
        self.assertEqual(kwargs["model_name"], "fake-model")
        self.assertEqual(kwargs["language"], "ja")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
