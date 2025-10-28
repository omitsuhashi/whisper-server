from __future__ import annotations

import os
import unittest
from unittest import mock

from src.lib.polish import llm_client
from src.lib.polish.llm_client import LLMPolishError, LLMPolisher
from src.lib.polish.models import PolishedSentence


class TestLLMPolisher(unittest.TestCase):
    class _DummyTokenizer:
        def __init__(self) -> None:
            self.messages = None

        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
            self.messages = messages
            return "PROMPT"

    @mock.patch("src.lib.polish.llm_client.mlx_make_sampler")
    @mock.patch("src.lib.polish.llm_client.mlx_generate")
    @mock.patch("src.lib.polish.llm_client.mlx_load")
    def test_polish_returns_updated_sentences(
        self,
        mock_load: mock.Mock,
        mock_generate: mock.Mock,
        mock_make_sampler: mock.Mock,
    ) -> None:
        tokenizer = mock.Mock()
        tokenizer.apply_chat_template.return_value = "PROMPT"
        mock_load.return_value = ("model", tokenizer)
        mock_generate.return_value = '{"sentences": [{"index": 1, "text": "修正済みです。"}]}'
        sampler = mock.Mock()
        mock_make_sampler.return_value = sampler

        with mock.patch.dict(os.environ, {"LLM_POLISH_MODEL": "demo/model"}, clear=True):
            llm_client._MODEL_CACHE.clear()
            polisher = LLMPolisher()
            sentences = [PolishedSentence(start=0.0, end=1.0, text="原文です。")]
            result = polisher.polish(sentences)

        self.assertEqual(result[0].text, "修正済みです。")
        tokenizer.apply_chat_template.assert_called_once()
        mock_generate.assert_called_once_with(
            "model",
            tokenizer,
            prompt="PROMPT",
            max_tokens=800,
            sampler=sampler,
        )
        mock_make_sampler.assert_called_once_with(temp=0.2, top_p=0.9)

    @mock.patch("src.lib.polish.llm_client.mlx_make_sampler")
    @mock.patch("src.lib.polish.llm_client.mlx_generate")
    @mock.patch("src.lib.polish.llm_client.mlx_load")
    def test_polish_falls_back_when_json_parse_fails(
        self,
        mock_load: mock.Mock,
        mock_generate: mock.Mock,
        mock_make_sampler: mock.Mock,
    ) -> None:
        tokenizer = mock.Mock()
        tokenizer.apply_chat_template.return_value = "PROMPT"
        mock_load.return_value = ("model", tokenizer)
        mock_generate.return_value = "これはJSONではありません。"
        mock_make_sampler.return_value = mock.Mock()

        with mock.patch.dict(os.environ, {"LLM_POLISH_MODEL": "demo/model"}, clear=True):
            llm_client._MODEL_CACHE.clear()
            polisher = LLMPolisher()
            sentences = [PolishedSentence(start=0.0, end=1.0, text="原文です。")]
            result = polisher.polish(sentences)

        self.assertEqual(result[0].text, "原文です。")

    def test_missing_model_id_raises_value_error(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                LLMPolisher(load_fn=mock.Mock(), generate_fn=mock.Mock())

    @mock.patch("src.lib.polish.llm_client.mlx_make_sampler", None)
    @mock.patch("src.lib.polish.llm_client.mlx_generate", None)
    @mock.patch("src.lib.polish.llm_client.mlx_load", None)
    @mock.patch("src.lib.polish.llm_client.import_module", side_effect=ImportError("missing"))
    def test_missing_mlx_dependency_raises_value_error(
        self,
        _mock_import: mock.Mock,
    ) -> None:
        with mock.patch.dict(os.environ, {"LLM_POLISH_MODEL": "demo/model"}, clear=True):
            with self.assertRaises(ValueError):
                LLMPolisher()

    @mock.patch("src.lib.polish.llm_client.mlx_make_sampler")
    @mock.patch("src.lib.polish.llm_client.mlx_generate")
    @mock.patch("src.lib.polish.llm_client.mlx_load")
    def test_generate_failure_raises_custom_exception(
        self,
        mock_load: mock.Mock,
        mock_generate: mock.Mock,
        mock_make_sampler: mock.Mock,
    ) -> None:
        tokenizer = mock.Mock()
        tokenizer.apply_chat_template.return_value = "PROMPT"
        mock_load.return_value = ("model", tokenizer)
        mock_generate.side_effect = RuntimeError("fail")
        mock_make_sampler.return_value = mock.Mock()

        with mock.patch.dict(os.environ, {"LLM_POLISH_MODEL": "demo/model"}, clear=True):
            llm_client._MODEL_CACHE.clear()
            polisher = LLMPolisher()
            with self.assertRaises(LLMPolishError):
                polisher.polish([PolishedSentence(start=0.0, end=1.0, text="原文です。")])

    def test_build_prompt_contains_kousei_keyword(self) -> None:
        tokenizer = self._DummyTokenizer()

        def fake_load(model_id: str):
            return ("model", tokenizer)

        llm_client._MODEL_CACHE.clear()
        polisher = LLMPolisher(
            model_id="demo/model",
            load_fn=fake_load,
            generate_fn=lambda *args, **kwargs: "",
            sampler_factory=lambda **kwargs: mock.Mock(),
        )

        sentences = [
            PolishedSentence(start=0.0, end=1.0, text="これはテスト用の文です。"),
            PolishedSentence(start=1.0, end=2.0, text="もう一文を追加します。"),
        ]

        prompt = polisher._build_prompt(
            sentences,
            style="ですます",
            extra_instructions="丁寧に校正してください。",
        )

        self.assertEqual(prompt, "PROMPT")
        self.assertIsNotNone(tokenizer.messages)
        system_message = tokenizer.messages[0]["content"]
        user_message = tokenizer.messages[1]["content"]
        self.assertIn("校正", system_message)
        self.assertIn("校正", user_message)


if __name__ == "__main__":
    unittest.main()
