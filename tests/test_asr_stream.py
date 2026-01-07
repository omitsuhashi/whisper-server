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


def _generate_pcm_bytes(samples: int = 200) -> bytes:
    payload = np.zeros(samples, dtype=np.int16)
    return payload.tobytes()


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

    @mock.patch("src.cmd.http.transcribe_paths_via_worker")
    @mock.patch("src.cmd.http.transcribe_prepared_audios")
    def test_transcribe_endpoint_handles_multiple_files(
        self,
        mock_transcribe_prepared: mock.Mock,
        mock_transcribe_worker: mock.Mock,
    ) -> None:
        texts = iter(["hello", "こんにちは"])
        mock_transcribe_worker.return_value = []

        def fake_transcribe(prepared, **kwargs):
            entries = list(prepared)
            results = []
            transcribe_all_fn = kwargs.get("transcribe_all_fn")
            decode_options = kwargs.get("decode_options")
            if transcribe_all_fn is not None:
                transcribe_all_fn(
                    [],
                    model_name=kwargs.get("model_name"),
                    language=kwargs.get("language"),
                    task=kwargs.get("task"),
                    **(decode_options or {}),
                )
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
            data={
                "model": "fake-model",
                "language": "ja",
                "prompt_agenda": "品質レビュー",
            },
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
        self.assertIsNotNone(kwargs.get("decode_options"))
        self.assertIn("議題: 品質レビュー", kwargs["decode_options"]["initial_prompt"])

    @mock.patch("src.cmd.http.transcribe_paths_via_worker")
    @mock.patch("src.cmd.http.transcribe_prepared_audios")
    def test_transcribe_endpoint_skips_silent_audio(
        self,
        mock_transcribe_prepared: mock.Mock,
        mock_transcribe_worker: mock.Mock,
    ) -> None:
        speech_result = TranscriptionResult(filename="speech.wav", text="ありがとう")
        mock_transcribe_worker.return_value = []
        def fake_transcribe(prepared, **kwargs):
            entries = list(prepared)
            results = []
            transcribe_all_fn = kwargs.get("transcribe_all_fn")
            decode_options = kwargs.get("decode_options")
            if transcribe_all_fn is not None:
                transcribe_all_fn(
                    [],
                    model_name=kwargs.get("model_name"),
                    language=kwargs.get("language"),
                    task=kwargs.get("task"),
                    **(decode_options or {}),
                )
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
        self.assertIsNone(kwargs.get("decode_options"))


class HttpRestPcmTests(unittest.TestCase):
    @mock.patch.dict("os.environ", {"ASR_OVERLAP_SECONDS": "1.0"})
    @mock.patch("src.cmd.http.transcribe_waveform_chunked")
    def test_transcribe_pcm_default_overlap_when_missing(
        self,
        mock_transcribe_chunked: mock.Mock,
    ) -> None:
        mock_transcribe_chunked.return_value = TranscriptionResult(filename="audio.pcm", text="ok", segments=[])

        app = http_cmd.create_app()
        client = TestClient(app)

        response = client.post(
            "/transcribe_pcm",
            files={"file": ("audio.pcm", _generate_pcm_bytes(), "application/octet-stream")},
            data={"chunk_seconds": "5"},
        )

        self.assertEqual(response.status_code, 200)
        mock_transcribe_chunked.assert_called_once()
        _, kwargs = mock_transcribe_chunked.call_args
        self.assertEqual(kwargs["chunk_seconds"], 5.0)
        self.assertEqual(kwargs["overlap_seconds"], 1.0)

    @mock.patch.dict(
        "os.environ",
        {"ASR_CHUNK_SECONDS": "4.0", "ASR_OVERLAP_SECONDS": "1.0"},
    )
    @mock.patch("src.cmd.http.transcribe_waveform_chunked")
    def test_transcribe_pcm_uses_env_chunk_when_missing(
        self,
        mock_transcribe_chunked: mock.Mock,
    ) -> None:
        mock_transcribe_chunked.return_value = TranscriptionResult(filename="audio.pcm", text="ok", segments=[])

        app = http_cmd.create_app()
        client = TestClient(app)

        response = client.post(
            "/transcribe_pcm",
            files={"file": ("audio.pcm", _generate_pcm_bytes(), "application/octet-stream")},
        )

        self.assertEqual(response.status_code, 200)
        mock_transcribe_chunked.assert_called_once()
        _, kwargs = mock_transcribe_chunked.call_args
        self.assertEqual(kwargs["chunk_seconds"], 4.0)
        self.assertEqual(kwargs["overlap_seconds"], 1.0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
