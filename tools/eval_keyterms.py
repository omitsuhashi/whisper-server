#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import requests

from src.lib.eval.keyterms import key_term_accuracy


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", default="http://localhost:8000")
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--prompt_terms", default="")
    parser.add_argument("--prompt_dictionary", default="")
    args = parser.parse_args()

    total = 0
    acc_sum = 0.0
    for line in Path(args.jsonl).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        audio_path = Path(row["audio_path"])
        key_terms = row.get("key_terms") or []

        files = {
            "files": (audio_path.name, audio_path.read_bytes(), "application/octet-stream"),
        }
        data: dict[str, str] = {}
        if args.prompt_terms:
            data["prompt_terms"] = args.prompt_terms
        if args.prompt_dictionary:
            data["prompt_dictionary"] = args.prompt_dictionary

        response = requests.post(
            f"{args.server}/transcribe",
            files=files,
            data=data,
            timeout=600,
        )
        response.raise_for_status()
        payload = response.json()
        result = payload[0] if payload else {}
        text = result.get("text") or ""

        kta = key_term_accuracy(text, key_terms)
        total += 1
        acc_sum += kta
        print(f'{row.get("id","?")}\tkta={kta:.3f}\ttext={text[:80]}')

    if total:
        print(f"\nAVG_KTA={acc_sum/total:.4f} samples={total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
