from __future__ import annotations

import argparse
import logging
from typing import Sequence

from src.config.defaults import DEFAULT_LANGUAGE, DEFAULT_MODEL_NAME
from src.lib.asr import TranscriptionResult, transcribe_all


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """CLI引数を定義して解析する。"""

    parser = argparse.ArgumentParser(description="mlx Whisper を用いた書き起こしCLI")
    parser.add_argument("audio", nargs="+", help="書き起こし対象の音声ファイルパス")
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME, help="使用するモデル名")
    parser.add_argument("--language", default=DEFAULT_LANGUAGE, help="言語ヒント（例: ja, en）")
    parser.add_argument("--task", default=None, help="Whisperタスク（例: translate）")
    parser.add_argument(
        "--show-segments",
        action="store_true",
        help="区間ごとの詳細結果も表示する場合は指定する",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="ログレベル (DEBUG/INFO/WARNING/ERROR)",
    )
    return parser.parse_args(argv)


def run_cli(args: argparse.Namespace) -> list[TranscriptionResult]:
    """コマンド引数を受け取り、書き起こし処理を実行する。"""

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    return transcribe_all(
        args.audio,
        model_name=args.model or DEFAULT_MODEL_NAME,
        language=args.language or DEFAULT_LANGUAGE,
        task=args.task,
    )


def main(argv: Sequence[str] | None = None) -> None:
    """エントリーポイント。実行結果を標準出力へ流す。"""

    args = parse_args(argv)

    try:
        results = run_cli(args)
    except Exception as exc:  # noqa: BLE001 - CLIからはエラーをそのまま通知する
        raise SystemExit(f"書き起こしに失敗しました: {exc}") from exc

    for result in results:
        print("=== ファイル:", result.filename)
        print("言語:", result.language or "不明")
        print("テキスト:\n", result.text)
        if args.show_segments:
            print("--- セグメント一覧 ---")
            for segment in result.segments:
                print(
                    f"[{segment.start:.2f}s - {segment.end:.2f}s] "
                    f"{segment.text.strip()}",
                )
        print()


if __name__ == "__main__":
    main()
