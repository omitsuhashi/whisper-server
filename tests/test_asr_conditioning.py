import sys
import types
import unittest
from unittest import mock


if "mlx_whisper" not in sys.modules:
    fake_module = types.ModuleType("mlx_whisper")
    fake_audio_module = types.ModuleType("mlx_whisper.audio")
    fake_audio_module.SAMPLE_RATE = 16000
    sys.modules["mlx_whisper.audio"] = fake_audio_module

    def _placeholder_transcribe(*args, **kwargs):
        raise RuntimeError("placeholder")

    fake_module.transcribe = _placeholder_transcribe
    fake_module.audio = fake_audio_module
    sys.modules["mlx_whisper"] = fake_module

from src.lib.asr.pipeline import _transcribe_single


class ConditionFallbackTests(unittest.TestCase):
    @mock.patch("src.lib.asr.pipeline._memsnap")
    @mock.patch("src.lib.asr.pipeline.transcribe")
    def test_transcribe_defaults_condition_flag(self, mock_transcribe: mock.Mock, mock_memsnap: mock.Mock) -> None:
        mock_transcribe.return_value = {"text": "こんにちは", "segments": [{"text": "こんにちは"}]}

        result = _transcribe_single(
            audio_input=b"",
            display_name="sample.wav",
            model_name="demo",
            transcribe_kwargs={},
            language_hint="ja",
        )

        self.assertEqual(result.text, "こんにちは")
        self.assertEqual(mock_transcribe.call_count, 1)
        self.assertTrue(mock_transcribe.call_args.kwargs["condition_on_previous_text"])
        self.assertGreaterEqual(mock_memsnap.call_count, 2)

    @mock.patch("src.lib.asr.pipeline._memsnap")
    @mock.patch("src.lib.asr.pipeline.transcribe")
    def test_transcribe_falls_back_without_condition(self, mock_transcribe: mock.Mock, mock_memsnap: mock.Mock) -> None:
        loop_payload = {
            "text": "foo bar foo bar foo bar foo bar",
            "segments": [{"text": "foo bar"} for _ in range(6)],
            "language": "ja",
        }
        clean_payload = {
            "text": "clean result",
            "segments": [{"text": "clean result"}],
            "language": "ja",
        }
        mock_transcribe.side_effect = [loop_payload, clean_payload]

        result = _transcribe_single(
            audio_input=b"",
            display_name="loop.wav",
            model_name="demo",
            transcribe_kwargs={},
            language_hint="ja",
        )

        self.assertEqual(result.text, "clean result")
        self.assertEqual(mock_transcribe.call_count, 2)
        first_kwargs = mock_transcribe.call_args_list[0].kwargs
        second_kwargs = mock_transcribe.call_args_list[1].kwargs
        self.assertTrue(first_kwargs["condition_on_previous_text"])
        self.assertFalse(second_kwargs["condition_on_previous_text"])
        self.assertGreaterEqual(mock_memsnap.call_count, 3)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
