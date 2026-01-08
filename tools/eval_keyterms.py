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
    parser.add_argument("--prompt_agenda", default="")
    parser.add_argument("--prompt_participants", default="")
    parser.add_argument("--prompt_products", default="")
    parser.add_argument("--prompt_style", default="")
    parser.add_argument("--chunk_seconds", type=float, default=None)
    parser.add_argument("--overlap_seconds", type=float, default=None)
    parser.add_argument("--mode", default="")
    args = parser.parse_args()

    total = 0
    acc_sum = 0.0
    jsonl_path = Path(args.jsonl)
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        audio_path = Path(row["audio_path"]).expanduser()
        if not audio_path.is_absolute():
            audio_path = (jsonl_path.parent / audio_path).resolve()
        key_terms = row.get("key_terms") or []

        files = {
            "files": (audio_path.name, audio_path.read_bytes(), "application/octet-stream"),
        }
        data: dict[str, str] = {}
        if args.prompt_terms:
            data["prompt_terms"] = args.prompt_terms
        if args.prompt_dictionary:
            data["prompt_dictionary"] = args.prompt_dictionary
        if args.prompt_agenda:
            data["prompt_agenda"] = args.prompt_agenda
        if args.prompt_participants:
            data["prompt_participants"] = args.prompt_participants
        if args.prompt_products:
            data["prompt_products"] = args.prompt_products
        if args.prompt_style:
            data["prompt_style"] = args.prompt_style
        if args.chunk_seconds is not None:
            data["chunk_seconds"] = str(args.chunk_seconds)
        if args.overlap_seconds is not None:
            data["overlap_seconds"] = str(args.overlap_seconds)
        headers = {}
        if args.mode:
            headers["X-Whisper-Mode"] = args.mode

        response = requests.post(
            f"{args.server}/transcribe",
            files=files,
            data=data,
            headers=headers,
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
