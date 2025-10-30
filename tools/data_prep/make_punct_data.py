from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Sequence, Tuple

PUNCT_PATTERN = re.compile(r"[。、．，,！？!？…⋯]")
DEFAULT_SPLIT = (0.8, 0.1, 0.1)
SPLIT_NAMES = ("train", "valid", "test")


@dataclass(frozen=True)
class DatasetSplit:
    name: str
    examples: List[dict[str, str]]


def strip_punctuation(text: str) -> str:
    return PUNCT_PATTERN.sub("", text)


def collect_sentences(lines: Iterable[str]) -> List[str]:
    sentences: List[str] = []
    for line in lines:
        trimmed = line.strip()
        if trimmed:
            sentences.append(trimmed)
    return sentences


def generate_examples(sentences: Sequence[str]) -> List[dict[str, str]]:
    return [{"source": strip_punctuation(sentence), "target": sentence} for sentence in sentences]


def split_examples(
    examples: Sequence[dict[str, str]],
    ratios: Tuple[float, float, float] = DEFAULT_SPLIT,
    *,
    seed: int = 13,
) -> List[DatasetSplit]:
    if len(examples) == 0:
        return [DatasetSplit(name, []) for name in SPLIT_NAMES]
    if len(ratios) != 3:
        raise ValueError("ratios must contain three values (train, valid, test)")
    if not 0.999 <= sum(ratios) <= 1.001:
        raise ValueError("ratios must sum to approximately 1.0")

    total = len(examples)
    rng = random.Random(seed)
    indices = list(range(total))
    rng.shuffle(indices)

    counts = [int(total * r) for r in ratios]
    remainder = total - sum(counts)
    for idx in range(remainder):
        counts[idx % 3] += 1

    splits: List[DatasetSplit] = []
    cursor = 0
    for name, count in zip(SPLIT_NAMES, counts):
        selected = [examples[indices[i]] for i in range(cursor, cursor + count)]
        splits.append(DatasetSplit(name=name, examples=selected))
        cursor += count
    return splits


def write_jsonl(examples: Iterable[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as outfile:
        for example in examples:
            outfile.write(json.dumps(example, ensure_ascii=False))
            outfile.write("\n")


def write_splits(splits: Sequence[DatasetSplit], output_dir: Path) -> None:
    for split in splits:
        write_jsonl(split.examples, output_dir / f"{split.name}.jsonl")


def parse_ratios(args: argparse.Namespace) -> Tuple[float, float, float]:
    ratios = (args.train_ratio, args.valid_ratio, args.test_ratio)
    total = sum(ratios)
    if not ratios or total == 0:
        return DEFAULT_SPLIT
    normalised = tuple(r / total for r in ratios)
    return normalised  # type: ignore[return-value]


def main() -> None:
    parser = argparse.ArgumentParser(description="句読点復元用の教師データを生成するユーティリティ")
    parser.add_argument("input", type=Path, help="句読点付きのテキスト（1行1文）")
    parser.add_argument("--output", type=Path, help="単一の JSONL 出力先パス")
    parser.add_argument("--output-dir", type=Path, help="train/valid/test を出力するディレクトリ")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--train-ratio", type=float, default=DEFAULT_SPLIT[0])
    parser.add_argument("--valid-ratio", type=float, default=DEFAULT_SPLIT[1])
    parser.add_argument("--test-ratio", type=float, default=DEFAULT_SPLIT[2])
    args = parser.parse_args()

    if not args.output and not args.output_dir:
        raise SystemExit("--output または --output-dir のいずれかを指定してください")
    if args.output and args.output_dir:
        raise SystemExit("--output と --output-dir は同時に指定できません")

    with args.input.open("r", encoding="utf-8") as infile:
        sentences = collect_sentences(infile)

    examples = generate_examples(sentences)

    if args.output:
        splits = split_examples(examples, DEFAULT_SPLIT, seed=args.seed)
        combined = splits[0].examples + splits[1].examples + splits[2].examples
        write_jsonl(combined, args.output)
        stats = {
            "total": len(examples),
            "output": str(args.output),
        }
    else:
        ratios = parse_ratios(args)
        splits = split_examples(examples, ratios, seed=args.seed)
        write_splits(splits, args.output_dir)
        stats = {
            "total": len(examples),
            "output_dir": str(args.output_dir),
            "counts": {split.name: len(split.examples) for split in splits},
        }

    print(json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":  # pragma: no cover
    main()
