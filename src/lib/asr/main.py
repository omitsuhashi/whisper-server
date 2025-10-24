from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Iterable, List

from pydantic import BaseModel, ConfigDict, Field

from mlx_whisper import transcribe

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "mlx-community/whisper-large-v3-mlx"
"""デフォルトで使用するWhisper Largeモデル名。"""


class TranscriptionSegment(BaseModel):
    """Whisperが返す1区間分の書き起こし結果。"""

    model_config = ConfigDict(extra="ignore")

    id: int = 0
    seek: int | None = None
    start: float = 0.0
    end: float = 0.0
    text: str = ""
    tokens: List[int] | None = None
    temperature: float | None = None
    avg_logprob: float | None = None
    compression_ratio: float | None = None
    no_speech_prob: float | None = None


class TranscriptionResult(BaseModel):
    """書き起こし結果を扱いやすい形に正規化したモデル。"""

    model_config = ConfigDict(extra="ignore")

    filename: str
    text: str = ""
    language: str | None = None
    duration: float | None = None
    segments: List[TranscriptionSegment] = Field(default_factory=list)


def transcribe_all(
    audio_paths: Iterable[str | Path],
    *,
    model_name: str = DEFAULT_MODEL_NAME,
    language: str | None = None,
    task: str | None = None,
    **decode_options: Any,
) -> List[TranscriptionResult]:
    """複数ファイルをまとめて書き起こすヘルパー関数。"""

    resolved = [Path(path) for path in audio_paths]
    if not resolved:
        return []

    missing = [str(path) for path in resolved if not path.exists()]
    if missing:
        raise FileNotFoundError(f"存在しない音声ファイルがあります: {', '.join(missing)}")

    transcribe_kwargs: dict[str, Any] = dict(decode_options)
    if language:
        transcribe_kwargs["language"] = language
    if task:
        transcribe_kwargs["task"] = task

    results: List[TranscriptionResult] = []
    for path in resolved:
        logger.info(
            "音声ファイルを書き起こし中: %s (model=%s language=%s task=%s)",
            path.name,
            model_name,
            language,
            task,
        )
        raw_result = transcribe(str(path), path_or_hf_repo=model_name, **transcribe_kwargs)
        results.append(_build_transcription_result(path, raw_result))

    return results


def _build_transcription_result(path: Path, payload: dict[str, Any]) -> TranscriptionResult:
    """mlx_whisperの戻り値をTranscriptionResultへ変換する。"""

    segments_raw = payload.get("segments", []) or []
    segments = [TranscriptionSegment.model_validate(segment) for segment in segments_raw]
    return TranscriptionResult(
        filename=path.name,
        text=payload.get("text", ""),
        language=payload.get("language"),
        duration=payload.get("duration"),
        segments=segments,
    )


def main() -> None:
    """簡易テスト用のCLI。音声ファイルを書き起こして標準出力へ結果を表示する。"""

    parser = argparse.ArgumentParser(description="mlx Whisper を用いた書き起こしテスト")
    parser.add_argument("audio", nargs="+", help="書き起こし対象の音声ファイルパス")
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME, help="使用するモデル名")
    parser.add_argument("--language", default=None, help="言語ヒント（例: ja, en）")
    parser.add_argument("--task", default=None, help="Whisperタスク（例: translate）")
    parser.add_argument(
        "--show-segments",
        action="store_true",
        help="区間ごとの詳細結果も表示する場合は指定する",
    )
    args = parser.parse_args()

    try:
        results = transcribe_all(
            args.audio,
            model_name=args.model,
            language=args.language,
            task=args.task,
        )
    except Exception as exc:  # noqa: BLE001 - テスト実行時の例外は明示的に表示する
        parser.error(f"書き起こしに失敗しました: {exc}")
        return

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


__all__ = [
    "DEFAULT_MODEL_NAME",
    "TranscriptionResult",
    "TranscriptionSegment",
    "main",
    "transcribe_all",
]


if __name__ == "__main__":
    main()
