import asyncio
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
from src.config.defaults import DEFAULT_MODEL_NAME
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

    @mock.patch.object(asr_main, "transcribe")
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

    @mock.patch.object(cli, "transcribe_all_bytes")
    def test_stream_subcommand_uses_stdin_payload(self, mock_transcribe_bytes: mock.Mock) -> None:
        fake_result = TranscriptionResult(filename="stdin", text="hello")
        mock_transcribe_bytes.return_value = [fake_result]

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

    @mock.patch.object(cli, "transcribe_all_bytes")
    def test_stream_subcommand_with_interval_streams_output(
        self, mock_transcribe_bytes: mock.Mock
    ) -> None:
        first = TranscriptionResult(filename="stdin", text="こ")
        second = TranscriptionResult(filename="stdin", text="こんにちは")
        mock_transcribe_bytes.side_effect = [[first], [second], [second]]

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


class HttpWebSocketTests(unittest.TestCase):
    def setUp(self) -> None:
        if shutil.which("ffmpeg") is None:
            self.skipTest("ffmpeg が見つかりません")
        self.audio_bytes = _generate_wav_bytes()

    @staticmethod
    async def _invoke_ws(actions, **handler_kwargs):
        app = http_cmd.create_app()
        route = next(r for r in app.routes if getattr(r, "path", None) == "/ws/transcribe")
        handler = route.endpoint

        websocket = _MockWebSocket(actions)
        await handler(websocket, **handler_kwargs)
        return websocket

    @mock.patch("src.cmd.http.transcribe_all_bytes")
    def test_ws_transcribe_success(self, mock_transcribe_bytes: mock.Mock) -> None:
        fake_result = TranscriptionResult(filename="ws", text="ok", language="ja")
        mock_transcribe_bytes.return_value = [fake_result]

        actions = [
            {"bytes": self.audio_bytes},
            {"text": "done"},
        ]

        websocket = asyncio.run(
            self._invoke_ws(
                actions,
                model="fake-model",
                language="ja",
                task=None,
            )
        )

        self.assertTrue(websocket.accepted)
        self.assertTrue(websocket.closed)
        self.assertEqual(websocket.close_code, 1000)
        self.assertEqual(websocket.sent[0]["text"], "ok")
        mock_transcribe_bytes.assert_called_once()
        _, kwargs = mock_transcribe_bytes.call_args
        self.assertEqual(kwargs["model_name"], "fake-model")
        self.assertEqual(kwargs["language"], "ja")

    def test_ws_transcribe_empty_audio(self) -> None:
        actions = [{"text": "done"}]

        websocket = asyncio.run(
            self._invoke_ws(
                actions,
                model=DEFAULT_MODEL_NAME,
                language=None,
                task=None,
            )
        )

        self.assertTrue(websocket.closed)
        self.assertEqual(websocket.close_code, 1007)
        self.assertIn("音声データが送信されていません", websocket.sent[0]["error"])


class _MockWebSocket:
    def __init__(self, actions):
        self._actions = iter(actions)
        self.sent = []
        self.accepted = False
        self.closed = False
        self.close_code = None

    async def accept(self) -> None:
        self.accepted = True

    async def receive(self):
        return next(self._actions)

    async def send_json(self, message) -> None:
        self.sent.append(message)

    async def close(self, code: int) -> None:
        self.closed = True
        self.close_code = code


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
