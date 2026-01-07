import unittest

from src.lib.asr.prompting import (
    PromptContext,
    build_initial_prompt,
    build_prompt_from_metadata,
    normalize_prompt_items,
)


class PromptingTests(unittest.TestCase):
    def test_normalize_prompt_items_splits_multiple_delimiters(self) -> None:
        raw = "議題A,議題B\n議題C"
        self.assertEqual(
            normalize_prompt_items(raw),
            ["議題A", "議題B", "議題C"],
        )

    def test_build_initial_prompt_includes_metadata(self) -> None:
        ctx = PromptContext(
            agenda_items=["品質レビュー"],
            participants=["田中太郎"],
            products=["Awesome API"],
            style_guidance="英数字は半角。",
        )
        prompt = build_initial_prompt(ctx)
        self.assertIn("議題: 品質レビュー", prompt)
        self.assertIn("参加者: 田中太郎", prompt)
        self.assertIn("製品・サービス: Awesome API", prompt)
        self.assertTrue(prompt.startswith("英数字は半角。"))

    def test_build_initial_prompt_truncates_long_text(self) -> None:
        ctx = PromptContext(
            agenda_items=["議題" + str(i) for i in range(50)],
        )
        prompt = build_initial_prompt(ctx, token_limit=10)
        self.assertLessEqual(len(prompt), 10 * 4)

    def test_build_prompt_from_metadata_skips_empty_payload(self) -> None:
        self.assertIsNone(build_prompt_from_metadata())

    def test_build_prompt_from_metadata_accepts_strings(self) -> None:
        prompt = build_prompt_from_metadata(
            agenda="改善,品質",
            participants="田中,佐藤",
            products="Project X",
            terms="用語A,用語B",
            dictionary="辞書A=AA,辞書B=BB",
            style="常に敬体。",
            token_limit=50,
        )
        self.assertIn("議題: 改善, 品質", prompt)
        self.assertIn("参加者: 田中, 佐藤", prompt)
        self.assertIn("用語: 用語A, 用語B", prompt)
        self.assertIn("辞書: 辞書A=AA, 辞書B=BB", prompt)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
