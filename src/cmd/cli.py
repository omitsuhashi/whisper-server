from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Iterable, Sequence

# 直接スクリプトとして実行された場合でも src パッケージを解決できるようにする
if __package__ in {None, ""}:  # python src/cmd/cli.py 等の実行形態に対応
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))

from src.config.defaults import DEFAULT_LANGUAGE, DEFAULT_MODEL_NAME
from src.lib.asr import TranscriptionResult, transcribe_all, transcribe_all_bytes
from src.lib.diarize import (
    DiarizeOptions,
    SpeakerAnnotatedTranscript,
    attach_speaker_labels,
    diarize_all,
)

STREAM_DEFAULT_CHUNK = 16_384  # 16 KB


def _read_stream(buffer: Iterable[bytes]) -> Iterable[bytes]:
    for chunk in buffer:
        if chunk:
            yield chunk


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """CLI引数を定義して解析する。"""

    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(description="mlx Whisper を用いた書き起こしCLI")
    subparsers = parser.add_subparsers(dest="command")

    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("--model", default=DEFAULT_MODEL_NAME, help="使用するモデル名")
    shared.add_argument("--language", default=DEFAULT_LANGUAGE, help="言語ヒント（例: ja, en）")
    shared.add_argument("--task", default=None, help="Whisperタスク（例: translate）")
    shared.add_argument(
        "--show-segments",
        action="store_true",
        help="区間ごとの詳細結果も表示する場合は指定する",
    )
    shared.add_argument(
        "--log-level",
        default="INFO",
        help="ログレベル (DEBUG/INFO/WARNING/ERROR)",
    )

    file_parser = subparsers.add_parser(
        "files",
        parents=[shared],
        help="ファイルパスを指定して書き起こす",
    )
    file_parser.add_argument("audio", nargs="+", help="書き起こし対象の音声ファイルパス")
    file_parser.add_argument(
        "--diarize",
        action="store_true",
        help="書き起こしに話者分離の結果を付加する",
    )
    file_parser.add_argument(
        "--diarize-num-speakers",
        type=int,
        default=None,
        help="想定話者数（指定すると精度向上が見込めます）",
    )
    file_parser.add_argument(
        "--diarize-min-speakers",
        type=int,
        default=None,
        help="想定最小話者数",
    )
    file_parser.add_argument(
        "--diarize-max-speakers",
        type=int,
        default=None,
        help="想定最大話者数",
    )
    file_parser.add_argument(
        "--diarize-device",
        default=None,
        help="話者分離を実行するデバイス（例: mps / cpu / cuda）",
    )
    file_parser.add_argument(
        "--diarize-token",
        default=None,
        help="Hugging Face のアクセストークン（未指定時は環境変数から探索）",
    )
    file_parser.add_argument(
        "--diarize-require-mps",
        action="store_true",
        help="MPS (Metal) が利用できない場合は失敗させる",
    )

    stream_parser = subparsers.add_parser(
        "stream",
        parents=[shared],
        help="標準入力から受け取った音声ストリームを書き起こす",
    )
    stream_parser.add_argument(
        "--name",
        default="stdin",
        help="結果に表示する仮想ファイル名",
    )
    stream_parser.add_argument(
        "--stream-interval",
        type=float,
        default=0.0,
        help="指定秒数ごとに書き起こしを更新して標準出力へ追記する (0 の場合は最後にまとめて出力)",
    )
    stream_parser.add_argument(
        "--stream-chunk-size",
        type=int,
        default=0,
        help=argparse.SUPPRESS,
    )

    subparsers.required = False
    parser.set_defaults(command="files")

    if argv and argv[0] not in {"files", "stream"}:
        argv = ["files", *argv]

    return parser.parse_args(argv)


def run_cli(args: argparse.Namespace) -> list[TranscriptionResult]:
    """コマンド引数を受け取り、書き起こし処理を実行する。"""

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    model_name = args.model or DEFAULT_MODEL_NAME
    language = args.language or DEFAULT_LANGUAGE

    args._stream_output = False  # type: ignore[attr-defined]

    if args.command == "stream":
        if sys.stdin.buffer.isatty():
            raise ValueError("ストリームモードでは標準入力へ音声データをパイプしてください。")

        chunk_size = args.stream_chunk_size if args.stream_chunk_size > 0 else STREAM_DEFAULT_CHUNK
        interval = max(float(args.stream_interval or 0.0), 0.0)

        if interval > 0:
            args._stream_output = True  # type: ignore[attr-defined]
            return _streaming_transcription(
                model_name=model_name,
                language=language,
                task=args.task,
                name=args.name,
                chunk_size=chunk_size,
                interval=interval,
            )

        audio_bytes = sys.stdin.buffer.read()
        if not audio_bytes:
            raise ValueError("標準入力から音声データを読み取れませんでした。")
        return transcribe_all_bytes(
            [audio_bytes],
            model_name=model_name,
            language=language,
            task=args.task,
            names=[args.name],
        )

    results = transcribe_all(
        args.audio,
        model_name=model_name,
        language=language,
        task=args.task,
    )

    if getattr(args, "diarize", False):
        diarize_options = DiarizeOptions(
            token=args.diarize_token,
            num_speakers=args.diarize_num_speakers,
            min_speakers=args.diarize_min_speakers,
            max_speakers=args.diarize_max_speakers,
            device=args.diarize_device,
            require_mps=args.diarize_require_mps,
        )
        diarization_results = diarize_all(args.audio, options=diarize_options)
        diarization_map = {dr.filename: dr for dr in diarization_results}
        speaker_transcripts: dict[str, SpeakerAnnotatedTranscript] = {}
        for result in results:
            diar = diarization_map.get(result.filename)
            if diar is None:
                raise RuntimeError(f"話者分離結果が見つかりませんでした: {result.filename}")
            speaker_transcripts[result.filename] = attach_speaker_labels(result, diar)
        args._speaker_transcripts = speaker_transcripts  # type: ignore[attr-defined]
        args._diarization_results = diarization_map  # type: ignore[attr-defined]

    return results


def _streaming_transcription(
    *,
    model_name: str,
    language: str | None,
    task: str | None,
    name: str,
    chunk_size: int,
    interval: float,
) -> list[TranscriptionResult]:
    """標準入力からのストリームを一定間隔で書き起こして標準出力へ追記する。"""

    stdin_buffer = sys.stdin.buffer
    audio_buffer = bytearray()
    last_emit_text = ""
    last_flush = time.monotonic()
    results: list[TranscriptionResult] = []

    def flush(force: bool = False) -> None:
        nonlocal last_emit_text, results
        if not audio_buffer:
            return
        if not force and (time.monotonic() - last_flush) < interval:
            return

        current_results = transcribe_all_bytes(
            [bytes(audio_buffer)],
            model_name=model_name,
            language=language,
            task=task,
            names=[name],
        )
        if not current_results:
            return
        results = current_results
        current_text = current_results[0].text

        if len(current_text) > len(last_emit_text):
            new_text = current_text[len(last_emit_text) :]
            if new_text:
                sys.stdout.write(new_text)
                sys.stdout.flush()
            last_emit_text = current_text

    while True:
        chunk = stdin_buffer.read(chunk_size)
        if not chunk:
            break
        audio_buffer.extend(chunk)
        now = time.monotonic()
        if (now - last_flush) >= interval:
            flush(force=True)
            last_flush = now

    flush(force=True)
    if results:
        sys.stdout.write("\n")
        sys.stdout.flush()
        return results
    raise ValueError("標準入力から音声データを読み取れませんでした。")


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
        speaker_map: dict[str, SpeakerAnnotatedTranscript] = getattr(args, "_speaker_transcripts", {})  # type: ignore[assignment]
        diar_map = getattr(args, "_diarization_results", {})
        speaker_transcript = speaker_map.get(result.filename) if speaker_map else None
        diar_result = diar_map.get(result.filename) if diar_map else None
        if speaker_transcript:
            print("話者:", ", ".join(speaker_transcript.speakers) or "-")
        elif diar_result:
            print("話者:", ", ".join(diar_result.speakers) or "-")
        if not getattr(args, "_stream_output", False):
            print("テキスト:\n", result.text)
        if args.show_segments:
            if speaker_transcript:
                print("--- 話者付きセグメント ---")
                for segment in speaker_transcript.segments:
                    print(
                        f"[{segment.start:.2f}s - {segment.end:.2f}s] "
                        f"{segment.speaker}: {segment.text.strip()}",
                    )
            else:
                print("--- セグメント一覧 ---")
                for segment in result.segments:
                    print(
                        f"[{segment.start:.2f}s - {segment.end:.2f}s] "
                        f"{segment.text.strip()}",
                    )
        print()


if __name__ == "__main__":
    main()
