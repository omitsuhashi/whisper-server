import unittest

from src.lib.asr import TranscribeOptions


class TranscribeOptionsTests(unittest.TestCase):
    def test_build_transcribe_kwargs_includes_language_and_task(self) -> None:
        options = TranscribeOptions(
            model_name="demo",
            language="ja",
            task="translate",
            decode_options={"temperature": 0.2},
        )

        kwargs = options.build_transcribe_kwargs()

        self.assertEqual(kwargs["language"], "ja")
        self.assertEqual(kwargs["task"], "translate")
        self.assertEqual(kwargs["temperature"], 0.2)

    def test_build_transcribe_kwargs_skips_empty_values(self) -> None:
        options = TranscribeOptions(model_name="demo")

        kwargs = options.build_transcribe_kwargs()

        self.assertEqual(kwargs, {})


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
