from __future__ import annotations

import unittest
import json
import tempfile
from pathlib import Path

from src.lib.corrector.tagger.model import RuleBasedPunctTagger, TaggerConfig
from src.lib.corrector.tagger.modeling import BertCrfTagger, BertCrfTaggerConfig
from src.lib.corrector.tagger import trainer as tagger_trainer
from src.lib.corrector.types import CorrectionOptions, CorrectionRequest
from tools.data_prep import make_punct_data


class TestRuleBasedPunctTagger(unittest.TestCase):
    def setUp(self) -> None:
        self.tagger = RuleBasedPunctTagger()

    def test_inserts_punctuation_when_missing(self) -> None:
        request = CorrectionRequest.from_raw(
            "これはテスト",
            options=CorrectionOptions(aggressive_kuten=True),
        )
        patches = self.tagger.predict(request)
        self.assertEqual(len(patches), 1)
        patch = patches[0]
        self.assertEqual(patch.replacement, "。")
        self.assertIn("RULE_BASED", patch.tags)

    def test_keeps_existing_punctuation(self) -> None:
        request = CorrectionRequest.from_raw(
            "了解。",
            options=CorrectionOptions(aggressive_kuten=True),
        )
        patches = self.tagger.predict(request)
        self.assertEqual(patches, ())

    def test_respects_aggressive_flag(self) -> None:
        request = CorrectionRequest.from_raw(
            "これはテスト",
            options=CorrectionOptions(aggressive_kuten=False),
        )
        self.assertFalse(self.tagger.predict(request))


class TestPunctDataPrep(unittest.TestCase):
    def test_strip_punctuation(self) -> None:
        self.assertEqual(make_punct_data.strip_punctuation("テスト、です。"), "テストです")

    def test_collect_and_generate_examples(self) -> None:
        lines = ["これはテストです。", "", "  改行もテスト。  "]
        sentences = make_punct_data.collect_sentences(lines)
        self.assertEqual(sentences, ["これはテストです。", "改行もテスト。"])
        examples = make_punct_data.generate_examples(sentences)
        self.assertEqual(len(examples), 2)
        self.assertEqual(examples[0]["source"], "これはテストです")
        self.assertEqual(examples[0]["target"], "これはテストです。")

    def test_write_jsonl(self) -> None:
        examples = [{"source": "ABC", "target": "A,B,C。"}]
        with tempfile.TemporaryDirectory() as tmp_dir:
            output = Path(tmp_dir) / "sample.jsonl"
            make_punct_data.write_jsonl(examples, output)
            with output.open("r", encoding="utf-8") as infile:
                record = json.loads(infile.readline())
        self.assertEqual(record["source"], "ABC")
        self.assertEqual(record["target"], "A,B,C。")

    def test_split_examples_respects_ratios(self) -> None:
        sentences = [f"テスト{i}。" for i in range(10)]
        examples = make_punct_data.generate_examples(sentences)
        splits = make_punct_data.split_examples(examples, (0.5, 0.3, 0.2), seed=1)
        counts = [len(split.examples) for split in splits]
        self.assertEqual(sum(counts), 10)
        self.assertEqual([split.name for split in splits], ["train", "valid", "test"])


class TestTaggerTrainer(unittest.TestCase):
    def test_evaluate_split_returns_metrics(self) -> None:
        tagger = RuleBasedPunctTagger()
        samples = [
            tagger_trainer.PunctSample(source="これはテスト", target="これはテスト。"),
            tagger_trainer.PunctSample(source="了解。", target="了解。"),
        ]
        metrics = tagger_trainer.evaluate_split(tagger, samples)
        self.assertIn("punct_f1", metrics)

    def test_load_label_map(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "labels.json"
            path.write_text(json.dumps(["O", "INSERT_PUNCT"]), encoding="utf-8")
            labels = tagger_trainer._load_label_map(path)
        self.assertEqual(labels, ["O", "INSERT_PUNCT"])

    def test_load_samples_reads_jsonl(self) -> None:
        payload = {"source": "ABC", "target": "ABC。"}
        with tempfile.TemporaryDirectory() as tmp_dir:
            jsonl_path = Path(tmp_dir) / "train.jsonl"
            with jsonl_path.open("w", encoding="utf-8") as outfile:
                outfile.write(json.dumps(payload, ensure_ascii=False) + "\n")
            samples = tagger_trainer.load_samples(jsonl_path)
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0].target, "ABC。")

    def test_bert_crf_tagger_stub(self) -> None:
        config = BertCrfTaggerConfig(model_path="dummy", label_map=["O", "INSERT_PUNCT"])
        tagger = BertCrfTagger(config=config)
        request = CorrectionRequest.from_raw("テスト", options=CorrectionOptions(aggressive_kuten=True))
        patches = tagger.predict(request)
        self.assertEqual(patches, ())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
