from __future__ import annotations

import atexit
import os
import sys
import types
import unittest
from unittest import mock

from fastapi.testclient import TestClient

os.environ.setdefault("MLX_USE_CPU", "1")

original_mlx = sys.modules.get("mlx_whisper")
original_mlx_audio = sys.modules.get("mlx_whisper.audio")

mlx_stub = types.ModuleType("mlx_whisper")
audio_stub = types.ModuleType("mlx_whisper.audio")
audio_stub.SAMPLE_RATE = 16000
mlx_stub.transcribe = lambda *args, **kwargs: None
mlx_stub.audio = audio_stub
sys.modules["mlx_whisper.audio"] = audio_stub
sys.modules["mlx_whisper"] = mlx_stub


def _restore_modules() -> None:
    if original_mlx is not None:
        sys.modules["mlx_whisper"] = original_mlx
    else:
        sys.modules.pop("mlx_whisper", None)

    if original_mlx_audio is not None:
        sys.modules["mlx_whisper.audio"] = original_mlx_audio
    else:
        sys.modules.pop("mlx_whisper.audio", None)


atexit.register(_restore_modules)

from src.cmd import http as http_cmd  # noqa: E402
from src.lib.polish import PolishedSentence  # noqa: E402
from src.lib.polish.llm_client import LLMPolishError  # noqa: E402


class PolishLLMEndpointTests(unittest.TestCase):
    @mock.patch("src.cmd.http.LLMPolisher")
    @mock.patch("src.cmd.http.polish_text_from_segments")
    def test_polish_llm_endpoint_returns_polished_sentences(
        self,
        mock_polish_text: mock.Mock,
        mock_polisher_cls: mock.Mock,
    ) -> None:
        base_sentences = [
            PolishedSentence(start=0.0, end=1.0, text="原文です。"),
            PolishedSentence(start=1.0, end=2.0, text="続きます。"),
        ]
        mock_polish_text.return_value = base_sentences

        polished_sentences = [
            PolishedSentence(start=0.0, end=1.0, text="校正済みの文です。"),
            PolishedSentence(start=1.0, end=2.0, text="文章が続きます。"),
        ]
        polisher_instance = mock.Mock()
        polisher_instance.polish.return_value = polished_sentences
        mock_polisher_cls.return_value = polisher_instance

        app = http_cmd.create_app()
        client = TestClient(app)

        payload = {
            "segments": [
                {"start": 0.0, "end": 1.0, "text": "原文です。"},
                {"start": 1.0, "end": 2.0, "text": "続きます。"},
            ],
            "options": {"style": "ですます"},
            "style": "常体",
            "extra_instructions": "語尾を柔らかくしてください。",
            "model_id": "demo/model",
            "temperature": 0.5,
            "top_p": 0.8,
            "max_tokens": 600,
        }

        response = client.post("/polish/llm", json=payload)

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["sentence_count"], 2)
        self.assertEqual(body["text"], "校正済みの文です。\n文章が続きます。")
        self.assertEqual(body["sentences"][0]["text"], "校正済みの文です。")

        mock_polisher_cls.assert_called_once()
        _, kwargs = mock_polisher_cls.call_args
        self.assertEqual(kwargs["model_id"], "demo/model")
        self.assertEqual(kwargs["temperature"], 0.5)
        self.assertEqual(kwargs["top_p"], 0.8)
        self.assertEqual(kwargs["max_tokens"], 600)

        polisher_instance.polish.assert_called_once()
        args, kwargs = polisher_instance.polish.call_args
        self.assertEqual(len(args[0]), 2)
        self.assertEqual(kwargs["style"], "常体")
        self.assertEqual(kwargs["extra_instructions"], "語尾を柔らかくしてください。")

    @mock.patch("src.cmd.http.LLMPolisher")
    def test_polish_llm_endpoint_returns_503_when_polisher_not_configured(
        self,
        mock_polisher_cls: mock.Mock,
    ) -> None:
        mock_polisher_cls.side_effect = ValueError("model missing")

        app = http_cmd.create_app()
        client = TestClient(app)

        payload = {
            "segments": [{"start": 0.0, "end": 1.0, "text": "原文です。"}],
        }

        response = client.post("/polish/llm", json=payload)

        self.assertEqual(response.status_code, 503)
        self.assertIn("model missing", response.text)

    @mock.patch("src.cmd.http.LLMPolisher")
    @mock.patch("src.cmd.http.polish_text_from_segments")
    def test_polish_llm_endpoint_returns_502_on_model_failure(
        self,
        mock_polish_text: mock.Mock,
        mock_polisher_cls: mock.Mock,
    ) -> None:
        mock_polish_text.return_value = [PolishedSentence(start=0.0, end=1.0, text="原文です。")]

        polisher_instance = mock.Mock()
        polisher_instance.polish.side_effect = LLMPolishError("llm error")
        mock_polisher_cls.return_value = polisher_instance

        app = http_cmd.create_app()
        client = TestClient(app)

        payload = {
            "segments": [{"start": 0.0, "end": 1.0, "text": "原文です。"}],
        }

        response = client.post("/polish/llm", json=payload)

        self.assertEqual(response.status_code, 502)
        self.assertIn("llm error", response.text)


if __name__ == "__main__":
    unittest.main()
