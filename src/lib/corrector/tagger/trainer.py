from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from ..types import CorrectionOptions, CorrectionRequest, apply_patches
from .model import BaseTagger, RuleBasedPunctTagger, TaggerConfig
from .modeling import BertCrfTagger, BertCrfTaggerConfig


@dataclass(frozen=True)
class PunctSample:
    """句読点復元用の単純な教師データ。"""

    source: str
    target: str


def load_samples(path: Path) -> list[PunctSample]:
    """JSONL 形式のサンプルを読み込み、`PunctSample` のリストへ変換する。"""
    samples: list[PunctSample] = []
    with path.open("r", encoding="utf-8") as infile:
        for line in infile:
            record = json.loads(line)
            samples.append(PunctSample(source=record["source"], target=record["target"]))
    return samples


def evaluate_split(
    tagger: BaseTagger,
    samples: Sequence[PunctSample],
) -> Mapping[str, float]:
    total = len(samples)
    if total == 0:
        return {"punct_f1": 0.0}

    true_positive = 0
    predicted = 0
    relevant = 0

    for sample in samples:
        request = CorrectionRequest.from_raw(
            sample.source,
            options=CorrectionOptions(aggressive_kuten=True),
        )
        patches = tagger.predict(request)
        predicted_text = apply_patches(sample.source, patches)

        predicted_punct = _count_sentence_terminators(predicted_text)
        target_punct = _count_sentence_terminators(sample.target)
        predicted += predicted_punct
        relevant += target_punct
        true_positive += min(predicted_punct, target_punct)

    precision = true_positive / predicted if predicted else 0.0
    recall = true_positive / relevant if relevant else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"punct_f1": f1}


def build_rule_based(config: TaggerConfig | None = None) -> RuleBasedPunctTagger:
    return RuleBasedPunctTagger(config=config)


def build_bert_crf(model_path: Path, label_map_path: Path) -> BertCrfTagger:
    labels = _load_label_map(label_map_path)
    config = BertCrfTaggerConfig(model_path=str(model_path), label_map=labels)
    return BertCrfTagger(config=config)


def _count_sentence_terminators(text: str) -> int:
    return sum(char in TaggerConfig().sentence_terminators for char in text)


def _load_splits(dataset_dir: Path) -> dict[str, list[PunctSample]]:
    splits: dict[str, list[PunctSample]] = {}
    for name in ("train", "valid", "test"):
        file_path = dataset_dir / f"{name}.jsonl"
        if file_path.exists():
            splits[name] = load_samples(file_path)
    if not splits:
        raise FileNotFoundError(f"no dataset files found under {dataset_dir}")
    return splits


def _load_label_map(path: Path) -> Sequence[str]:
    with path.open("r", encoding="utf-8") as infile:
        data = json.load(infile)
    if not isinstance(data, list) or not all(isinstance(item, str) for item in data):
        raise ValueError("label map must be a JSON array of strings")
    return list(data)


def _build_tagger(args: argparse.Namespace) -> BaseTagger:
    if args.backend == "rule-based":
        return build_rule_based()
    if args.backend == "bert-crf":
        if not args.model_path or not args.label_map:
            raise SystemExit("--model-path と --label-map は bert-crf バックエンドで必須です")
        return build_bert_crf(args.model_path, args.label_map)
    raise SystemExit(f"unsupported backend: {args.backend}")


def main() -> None:
    parser = argparse.ArgumentParser(description="句読点タグ付け器の簡易トレーナー")
    parser.add_argument("--dataset", type=Path, help="単一の JSONL")
    parser.add_argument("--dataset-dir", type=Path, help="train/valid/test JSONL を含むディレクトリ")
    parser.add_argument("--backend", choices=("rule-based", "bert-crf"), default="rule-based")
    parser.add_argument("--model-path", type=Path, help="BERT/CRF モデルの重みパス")
    parser.add_argument("--label-map", type=Path, help="ラベル ID → タグ名の JSON")
    args = parser.parse_args()

    if not args.dataset and not args.dataset_dir:
        raise SystemExit("--dataset か --dataset-dir を指定してください")
    if args.dataset and args.dataset_dir:
        raise SystemExit("--dataset と --dataset-dir は同時指定できません")

    tagger = _build_tagger(args)
    results: dict[str, Mapping[str, float]]

    if args.dataset:
        samples = load_samples(args.dataset)
        results = {"dataset": evaluate_split(tagger, samples)}
    else:
        splits = _load_splits(args.dataset_dir)
        results = {name: evaluate_split(tagger, split) for name, split in splits.items()}

    print(json.dumps(results, ensure_ascii=False))


if __name__ == "__main__":  # pragma: no cover
    main()
