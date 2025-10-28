import argparse
import atexit
import io
from pathlib import Path
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
from src.lib.audio import PreparedAudio
from src.lib.polish import PolishedSentence


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


def _generate_silent_wav_bytes(duration_sec: float = 0.1) -> bytes:
    sample_rate = 16000
    total_samples = int(sample_rate * duration_sec)
    payload = np.zeros(total_samples, dtype=np.int16)

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


class _InterruptBuffer:
    def __init__(self, chunks):
        self._chunks = list(chunks)

    def read(self, size: int = -1):
        if not self._chunks:
            return b""
        item = self._chunks.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item

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

    @mock.patch("src.lib.asr.pipeline.transcribe")
    def test_transcribe_all_bytes_skips_silence(self, mock_transcribe: mock.Mock) -> None:
        mock_transcribe.return_value = {}

        results = asr_main.transcribe_all_bytes(
            [_generate_silent_wav_bytes()],
            model_name="fake-model",
            language="ja",
        )

        self.assertEqual(len(results), 1)
        result = results[0]
        self.assertEqual(result.text, "")
        self.assertEqual(len(result.segments), 0)
        mock_transcribe.assert_not_called()

    @mock.patch("src.lib.asr.pipeline.transcribe")
    def test_transcribe_all_bytes_forces_silence_when_model_reports_no_speech(
        self, mock_transcribe: mock.Mock
    ) -> None:
        mock_transcribe.return_value = {
            "text": "ご視聴ありがとうございました",
            "language": "ja",
            "segments": [
                {
                    "id": 0,
                    "start": 0.0,
                    "end": 1.0,
                    "text": "ご視聴ありがとうございました",
                    "no_speech_prob": 0.92,
                }
            ],
        }

        results = asr_main.transcribe_all_bytes(
            [self.audio_bytes],
            model_name="fake-model",
            language="ja",
        )

        self.assertEqual(results[0].text, "")
        mock_transcribe.assert_called_once()


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

    @mock.patch("src.cmd.cli.polish_text_from_segments")
    @mock.patch("src.cmd.cli._load_asr_components")
    def test_stream_subcommand_applies_polish_when_requested(
        self,
        mock_load_asr: mock.Mock,
        mock_polish: mock.Mock,
    ) -> None:
        raw_result = TranscriptionResult(
            filename="stdin",
            text="raw",
            segments=[{"start": 0.0, "end": 1.0, "text": "raw"}],
        )
        mock_transcribe_all = mock.Mock()
        mock_transcribe_bytes = mock.Mock(return_value=[raw_result])
        mock_load_asr.return_value = (
            TranscriptionResult,
            mock_transcribe_all,
            mock_transcribe_bytes,
        )

        polished_sentence = PolishedSentence(start=0.0, end=1.0, text="rawでした。")
        mock_polish.return_value = [polished_sentence]

        fake_stdin = FakeStdin(self.audio_bytes)
        original_stdin = cli.sys.stdin
        cli.sys.stdin = fake_stdin

        try:
            args = cli.parse_args(["stream", "--model", "fake", "--polish"])
            results = cli.run_cli(args)
        finally:
            cli.sys.stdin = original_stdin

        mock_transcribe_bytes.assert_called_once()
        mock_polish.assert_called_once()
        self.assertFalse(getattr(args, "_stream_output", False))
        self.assertTrue(getattr(args, "polish", False))
        self.assertEqual(results[0].text, "rawでした。")
        polished_map = getattr(args, "_polished_sentences", {})
        self.assertIn("stdin", polished_map)
        self.assertEqual(polished_map["stdin"][0].text, "rawでした。")

    @mock.patch("src.cmd.cli.polish_text_from_segments")
    @mock.patch("src.cmd.cli._load_asr_components")
    def test_stream_subcommand_handles_keyboard_interrupt(
        self,
        mock_load_asr: mock.Mock,
        mock_polish: mock.Mock,
    ) -> None:
        raw_result = TranscriptionResult(
            filename="stdin",
            text="raw",
            segments=[{"start": 0.0, "end": 1.0, "text": "raw"}],
        )
        mock_transcribe_all = mock.Mock()
        mock_transcribe_bytes = mock.Mock(return_value=[raw_result])
        mock_load_asr.return_value = (
            TranscriptionResult,
            mock_transcribe_all,
            mock_transcribe_bytes,
        )

        polished_sentence = PolishedSentence(start=0.0, end=1.0, text="仕上げ")
        mock_polish.return_value = [polished_sentence]

        interrupt_buffer = _InterruptBuffer([b"audio", KeyboardInterrupt()])
        fake_stdin = types.SimpleNamespace(buffer=interrupt_buffer)
        original_stdin = cli.sys.stdin
        cli.sys.stdin = fake_stdin

        try:
            args = cli.parse_args(["stream", "--model", "fake", "--polish"])
            results = cli.run_cli(args)
        finally:
            cli.sys.stdin = original_stdin

        self.assertEqual(results[0].text, "仕上げ")
        mock_polish.assert_called_once()

    @mock.patch("src.cmd.cli.transcribe_prepared_audios")
    @mock.patch("src.cmd.cli.prepare_audio")
    @mock.patch("src.cmd.cli._load_asr_components")
    def test_files_subcommand_applies_polish(
        self,
        mock_load_asr: mock.Mock,
        mock_prepare_audio: mock.Mock,
        mock_transcribe_prepared: mock.Mock,
    ) -> None:
        mock_load_asr.return_value = (
            TranscriptionResult,
            mock.Mock(),
            mock.Mock(),
        )
        prepared = PreparedAudio(path=Path("dummy.wav"), display_name="dummy.wav", silent=False)
        mock_prepare_audio.return_value = prepared

        raw_result = TranscriptionResult(
            filename="dummy.wav",
            text="raw",
            segments=[{"start": 0.0, "end": 1.0, "text": "raw"}],
        )
        mock_transcribe_prepared.return_value = [raw_result]

        with mock.patch("src.cmd.cli.polish_text_from_segments") as mock_polish:
            mock_polish.return_value = [PolishedSentence(start=0.0, end=1.0, text="polished")]
            args = cli.parse_args(["files", "--model", "fake", "--polish", "sample.wav"])
            results = cli.run_cli(args)

        mock_prepare_audio.assert_called_once()
        mock_transcribe_prepared.assert_called_once()
        self.assertEqual(results[0].text, "polished")
        polished_map = getattr(args, "_polished_sentences", {})
        self.assertEqual(polished_map["dummy.wav"][0].text, "polished")

    @mock.patch("src.cmd.cli.run_cli")
    @mock.patch("src.cmd.cli.parse_args")
    def test_main_plain_text_outputs_text_only(
        self,
        mock_parse_args: mock.Mock,
        mock_run_cli: mock.Mock,
    ) -> None:
        result = TranscriptionResult(filename="sample.wav", text="こんにちは", segments=[])
        args = argparse.Namespace(
            command="files",
            plain_text=True,
            polish=False,
            show_segments=False,
            log_level="INFO",
            _stream_output=False,
        )

        mock_parse_args.return_value = args
        mock_run_cli.return_value = [result]

        original_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            cli.main([])
            output = sys.stdout.getvalue().strip()
        finally:
            sys.stdout = original_stdout

        self.assertEqual(output, "こんにちは")


class HttpRestTests(unittest.TestCase):
    def setUp(self) -> None:
        if shutil.which("ffmpeg") is None:
            self.skipTest("ffmpeg が見つかりません")

    @mock.patch("src.cmd.http.transcribe_prepared_audios")
    def test_transcribe_endpoint_handles_multiple_files(self, mock_transcribe_prepared: mock.Mock) -> None:
        texts = iter(["hello", "こんにちは"])

        def fake_transcribe(prepared, **kwargs):
            entries = list(prepared)
            results = []
            for entry in entries:
                text = next(texts)
                results.append(
                    TranscriptionResult(
                        filename=entry.display_name,
                        text=text,
                        language=kwargs.get("language"),
                    )
                )
            return results

        mock_transcribe_prepared.side_effect = fake_transcribe

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

        mock_transcribe_prepared.assert_called_once()
        args, kwargs = mock_transcribe_prepared.call_args
        self.assertEqual(len(list(args[0])), 2)
        self.assertEqual(kwargs["model_name"], "fake-model")
        self.assertEqual(kwargs["language"], "ja")

    @mock.patch("src.cmd.http.polish_text_from_segments")
    def test_polish_endpoint_returns_polished_sentences(self, mock_polish: mock.Mock) -> None:
        mock_polish.return_value = [
            PolishedSentence(start=0.0, end=1.5, text="こんにちは。"),
            PolishedSentence(start=1.5, end=3.0, text="よろしくお願いします。"),
        ]

        app = http_cmd.create_app()
        client = TestClient(app)

        payload = {
            "segments": [
                {"start": 0.0, "end": 1.5, "text": "こんにちは"},
                {"start": 1.5, "end": 3.0, "text": "よろしくお願いします"},
            ],
            "options": {"use_ginza": False, "style": "ですます"},
        }

        response = client.post("/polish", json=payload)

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["sentence_count"], 2)
        self.assertEqual(body["text"], "こんにちは。\nよろしくお願いします。")
        self.assertEqual(len(body["sentences"]), 2)
        self.assertEqual(body["sentences"][0]["start"], 0.0)
        self.assertEqual(body["sentences"][1]["text"], "よろしくお願いします。")

        args, kwargs = mock_polish.call_args
        segments = list(args[0])
        self.assertEqual(len(segments), 2)
        self.assertEqual(segments[0].text, "こんにちは")
        options = kwargs["options"]
        self.assertFalse(options.use_ginza)
        self.assertEqual(options.style, "ですます")

    def test_polish_endpoint_rejects_empty_segments(self) -> None:
        app = http_cmd.create_app()
        client = TestClient(app)

        response = client.post("/polish", json={"segments": []})

        self.assertEqual(response.status_code, 422)
        self.assertIn("segments", response.text)

    @mock.patch("src.cmd.http.transcribe_prepared_audios")
    def test_transcribe_endpoint_skips_silent_audio(self, mock_transcribe_prepared: mock.Mock) -> None:
        speech_result = TranscriptionResult(filename="speech.wav", text="ありがとう")
        def fake_transcribe(prepared, **kwargs):
            entries = list(prepared)
            results = []
            for entry in entries:
                if entry.silent:
                    results.append(
                        TranscriptionResult(
                            filename=entry.display_name,
                            text="",
                            language=kwargs.get("language"),
                        )
                    )
                else:
                    results.append(speech_result.model_copy(update={"filename": entry.display_name}))
            return results

        mock_transcribe_prepared.side_effect = fake_transcribe

        app = http_cmd.create_app()
        client = TestClient(app)

        files = [
            ("files", ("silence.wav", _generate_silent_wav_bytes(), "audio/wav")),
            ("files", ("speech.wav", _generate_wav_bytes(), "audio/wav")),
        ]

        response = client.post(
            "/transcribe",
            files=files,
            data={"model": "fake-model", "language": "ja"},
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(len(payload), 2)
        self.assertEqual(payload[0]["filename"], "silence.wav")
        self.assertEqual(payload[0]["text"], "")
        self.assertEqual(payload[1]["text"], "ありがとう")

        mock_transcribe_prepared.assert_called_once()
        args, kwargs = mock_transcribe_prepared.call_args
        self.assertEqual(len(list(args[0])), 2)
        self.assertEqual(kwargs["language"], "ja")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
