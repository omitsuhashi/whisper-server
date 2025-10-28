from __future__ import annotations

import os
import unittest
from unittest import mock

import httpx

from src.lib.polish.abeja_qwen import AbejaQwenError, AbejaQwenPolisher
from src.lib.polish.models import PolishedSentence


class TestAbejaQwenPolisher(unittest.TestCase):
    @mock.patch("src.lib.polish.abeja_qwen.httpx.post")
    def test_polish_returns_updated_sentences(self, mock_post: mock.Mock) -> None:
        mock_response = mock.Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = [
            {"generated_text": '{"sentences": [{"index": 1, "text": "修正済みの文です。"}]}'}
        ]
        mock_post.return_value = mock_response

        with mock.patch.dict(os.environ, {"HF_TOKEN": "dummy-token"}, clear=True):
            polisher = AbejaQwenPolisher()
            sentences = [PolishedSentence(start=0.0, end=1.0, text="原文です。")]
            result = polisher.polish(sentences)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].text, "修正済みの文です。")
        mock_post.assert_called_once()

    @mock.patch("src.lib.polish.abeja_qwen.httpx.post")
    def test_polish_falls_back_when_json_parse_fails(self, mock_post: mock.Mock) -> None:
        mock_response = mock.Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = [{"generated_text": "これはJSONではありません。"}]
        mock_post.return_value = mock_response

        with mock.patch.dict(os.environ, {"HF_TOKEN": "dummy-token"}, clear=True):
            polisher = AbejaQwenPolisher()
            sentences = [PolishedSentence(start=0.0, end=1.0, text="原文です。")]
            result = polisher.polish(sentences)

        self.assertEqual(result[0].text, "原文です。")

    def test_missing_token_raises_value_error(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                AbejaQwenPolisher()

    @mock.patch("src.lib.polish.abeja_qwen.httpx.post")
    def test_http_error_raises_custom_exception(self, mock_post: mock.Mock) -> None:
        httpx_response = httpx.Response(status_code=400, request=httpx.Request("POST", "https://example.com"))
        mock_response = mock.Mock()
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "error", request=httpx_response.request, response=httpx_response
        )
        mock_post.return_value = mock_response

        with mock.patch.dict(os.environ, {"HF_TOKEN": "dummy-token"}, clear=True):
            polisher = AbejaQwenPolisher()
            sentences = [PolishedSentence(start=0.0, end=1.0, text="原文です。")]
            with self.assertRaises(AbejaQwenError):
                polisher.polish(sentences)


if __name__ == "__main__":
    unittest.main()
