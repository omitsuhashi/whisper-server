from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, List, Sequence, TYPE_CHECKING, cast

# 直接スクリプトとして実行された場合でも src パッケージを解決できるようにする
if __package__ in {None, ""}:  # python src/cmd/cli.py 等の実行形態に対応
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))

import cv2

from src.config.defaults import DEFAULT_LANGUAGE, DEFAULT_MODEL_NAME
from src.config.logging import setup_logging
from src.lib.audio import InvalidAudioError, PreparedAudio, prepare_audio
from src.lib.asr.service import resolve_model_and_language, transcribe_prepared_audios
from src.lib.asr.prompting import build_prompt_from_metadata
from src.lib.diagnostics.memwatch import ensure_memory_watchdog
from src.lib.video import FrameSamplingError, SampledFrame, sample_key_frames
from src.lib.context import (
    IntegratedMeetingContext,
    build_image_contexts_from_video,
    integrate_meeting_data,
    render_mermaid_mindmap,
)
from src.lib.youtube import YouTubeDescriptionFetcher

if TYPE_CHECKING:  # pragma: no cover
    from src.lib.asr import TranscriptionResult
    from src.lib.diarize import DiarizationResult, SpeakerAnnotatedTranscript

STREAM_DEFAULT_CHUNK = 16_384  # 16 KB
YOUTUBE_DEFAULT_OUTPUT = "youtube_descriptions.json"
YOUTUBE_ENV_API_KEY = "YOUTUBE_API_KEY"

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SavedFrameInfo:
    path: Path
    frame_index: int
    timestamp: float


@dataclass(frozen=True)
class YouTubeFetchResult:
    output_path: Path


@dataclass(frozen=True)
class FrameExtractionResult:
    video: Path
    frames: List[SavedFrameInfo]


CommandResult = (
    list["TranscriptionResult"]
    | list[FrameExtractionResult]
    | list["DiarizationResult"]
    | YouTubeFetchResult
)
CommandHandler = Callable[[argparse.Namespace], CommandResult]
Validator = Callable[[argparse.Namespace], None]


@dataclass(frozen=True)
class SubcommandSpec:
    name: str
    help: str
    configure: Callable[[argparse.ArgumentParser], None]
    handler: CommandHandler
    use_shared_parent: bool = False
    description: str | None = None
    aliases: tuple[str, ...] = ()
    validators: tuple[Validator, ...] = ()


def _read_stream(buffer: Iterable[bytes]) -> Iterable[bytes]:
    for chunk in buffer:
        if chunk:
            yield chunk


def _build_shared_parent_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
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
    parser.add_argument(
        "--plain-text",
        action="store_true",
        help="メタ情報を省きテキストのみ標準出力へ表示する",
    )
    parser.add_argument(
        "--prompt-agenda",
        default=None,
        help="initial_prompt 用の議題（カンマ区切り）",
    )
    parser.add_argument(
        "--prompt-participants",
        default=None,
        help="initial_prompt 用の参加者名（カンマ区切り）",
    )
    parser.add_argument(
        "--prompt-products",
        default=None,
        help="initial_prompt 用の製品名・ドメイン用語（カンマ区切り）",
    )
    parser.add_argument(
        "--prompt-style",
        default=None,
        help="initial_prompt へ追加するスタイル指示（未指定時は既定文）",
    )
    return parser


def _build_decode_options(args: argparse.Namespace) -> dict[str, Any] | None:
    prompt = build_prompt_from_metadata(
        agenda=getattr(args, "prompt_agenda", None),
        participants=getattr(args, "prompt_participants", None),
        products=getattr(args, "prompt_products", None),
        style=getattr(args, "prompt_style", None),
    )
    if not prompt:
        return None
    return {"initial_prompt": prompt}


def _configure_files_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("audio", nargs="+", help="書き起こし対象の音声ファイルパス")
    parser.add_argument(
        "--diarize",
        action="store_true",
        help="書き起こしに話者分離の結果を付加する",
    )
    parser.add_argument(
        "--diarize-num-speakers",
        type=int,
        default=None,
        help="想定話者数（指定すると精度向上が見込めます）",
    )
    parser.add_argument(
        "--diarize-min-speakers",
        type=int,
        default=None,
        help="想定最小話者数",
    )
    parser.add_argument(
        "--diarize-max-speakers",
        type=int,
        default=None,
        help="想定最大話者数",
    )
    parser.add_argument(
        "--diarize-device",
        default=None,
        help="話者分離を実行するデバイス（mps のみサポート）",
    )
    parser.add_argument(
        "--diarize-token",
        default=None,
        help="Hugging Face のアクセストークン（未指定時は環境変数から探索）",
    )
    parser.add_argument(
        "--video",
        nargs="*",
        default=None,
        help="音声と同順の画面録画動画（任意）。指定した場合はキーフレームから画面コンテキストを解析して統合します。",
    )
    parser.add_argument(
        "--context-min-scene-span",
        type=float,
        default=1.0,
        help="画面解析: 同一シーンとみなす最小間隔（秒）",
    )
    parser.add_argument(
        "--context-diff-threshold",
        type=float,
        default=0.3,
        help="画面解析: シーン切替のヒストグラム距離の閾値（0〜1）",
    )
    parser.add_argument(
        "--max-context-frames",
        type=int,
        default=50,
        help="画面解析: 最大解析フレーム数（肥大化防止のための上限）",
    )
    parser.add_argument(
        "--output",
        choices=["text", "json", "mermaid"],
        default=None,
        help="統合出力の形式（未指定: 既存の標準出力）",
    )


def _configure_stream_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--name",
        default="stdin",
        help="結果に表示する仮想ファイル名",
    )
    parser.add_argument(
        "--stream-interval",
        type=float,
        default=0.0,
        help="指定秒数ごとに書き起こしを更新して標準出力へ追記する (0 の場合は最後にまとめて出力)",
    )
    parser.add_argument(
        "--stream-chunk-size",
        type=int,
        default=0,
        help=argparse.SUPPRESS,
    )


def _configure_frames_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("video", nargs="+", help="分析対象の動画ファイルパス")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="抽出フレームを書き出すディレクトリ（既定: 各動画の <名前>_frames）",
    )
    parser.add_argument(
        "--min-scene-span",
        type=float,
        default=1.0,
        help="同一シーンとみなす最小間隔（秒）。値を小さくすると細かい切替も検出します。",
    )
    parser.add_argument(
        "--diff-threshold",
        type=float,
        default=0.3,
        help="ヒストグラム距離の閾値（0〜1）。小さいほど厳密、値を大きくすると変化を拾いやすくなります。",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="最大抽出枚数。指定しない場合はすべての変化を保存します。",
    )
    parser.add_argument(
        "--image-format",
        default="png",
        choices=["png", "jpg", "jpeg", "bmp"],
        help="保存する画像形式。",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="既存の出力先が存在する場合に上書きします。",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="ログレベル (DEBUG/INFO/WARNING/ERROR)",
    )


def _configure_diarize_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("audio", nargs="+", help="話者分離を行う音声ファイルパス")
    parser.add_argument(
        "--token",
        default=None,
        help="Hugging Face のアクセストークン（未指定時は環境変数から探索）",
    )
    parser.add_argument(
        "--num-speakers",
        type=int,
        default=None,
        help="想定話者数（指定すると安定する場合があります）",
    )
    parser.add_argument(
        "--min-speakers",
        type=int,
        default=None,
        help="想定最小話者数",
    )
    parser.add_argument(
        "--max-speakers",
        type=int,
        default=None,
        help="想定最大話者数",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="話者分離を実行するデバイス（mps のみサポート）",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="デコード時に使用するサンプルレート（既定: 16000 Hz）",
    )
    parser.add_argument(
        "--show-turns",
        action="store_true",
        help="話者区間を詳細表示する",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="ログレベル (DEBUG/INFO/WARNING/ERROR)",
    )


def _configure_youtube_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--api-key",
        "-k",
        dest="youtube_api_key",
        help=f"YouTube Data API キー。未指定時は環境変数 ${YOUTUBE_ENV_API_KEY} を参照します。",
    )
    parser.add_argument(
        "--handle",
        dest="youtube_handle",
        help="チャンネルの @ハンドル。先頭の @ はあってもなくても構いません。",
    )
    parser.add_argument(
        "--channel-id",
        dest="youtube_channel_id",
        help="チャンネル ID。指定した場合はハンドルのルックアップをスキップします。",
    )
    parser.add_argument(
        "--output",
        "-o",
        dest="youtube_output",
        default=YOUTUBE_DEFAULT_OUTPUT,
        help=f"書き出す JSON パス（既定: {YOUTUBE_DEFAULT_OUTPUT}）。",
    )
    parser.add_argument(
        "--ensure-ascii",
        dest="youtube_ensure_ascii",
        action="store_true",
        help="出力 JSON の非 ASCII 文字を全てエスケープします。",
    )
    parser.add_argument(
        "--pause",
        dest="youtube_pause",
        type=float,
        default=0.1,
        help="API 呼び出し間のスリープ秒数（既定: 0.1）。",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="ログレベル (DEBUG/INFO/WARNING/ERROR)",
    )


def _prepare_asr_runtime(
    args: argparse.Namespace,
) -> tuple[str, str | None, Any, Any]:
    args._stream_output = False  # type: ignore[attr-defined]
    args._decode_options = _build_decode_options(args)  # type: ignore[attr-defined]
    _, transcribe_all_fn, transcribe_all_bytes_fn = _load_asr_components()
    model_name, language = resolve_model_and_language(
        args.model,
        args.language,
        default_model=DEFAULT_MODEL_NAME,
        default_language=DEFAULT_LANGUAGE,
    )
    return model_name, language, transcribe_all_fn, transcribe_all_bytes_fn


def _handle_stream_command(args: argparse.Namespace) -> CommandResult:
    model_name, language, _, transcribe_all_bytes_fn = _prepare_asr_runtime(args)
    decode_options = dict(getattr(args, "_decode_options", {}) or {})
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
            transcribe_all_bytes_fn=transcribe_all_bytes_fn,
            decode_options=decode_options,
            emit_stdout=True,
        )

    audio_buffer = bytearray()
    try:
        while True:
            chunk = sys.stdin.buffer.read(8192)
            if not chunk:
                break
            audio_buffer.extend(chunk)
    except KeyboardInterrupt:
        if not audio_buffer:
            raise
        logger.debug("stream_interrupt: received_bytes=%d", len(audio_buffer))
    audio_bytes = bytes(audio_buffer)
    if not audio_bytes:
        raise ValueError("標準入力から音声データを読み取れませんでした。")
    return transcribe_all_bytes_fn(
        [audio_bytes],
        model_name=model_name,
        language=language,
        task=args.task,
        names=[args.name],
        **decode_options,
    )


def _handle_files_command(args: argparse.Namespace) -> CommandResult:
    model_name, language, transcribe_all_fn, _ = _prepare_asr_runtime(args)

    if getattr(args, "diarize", False) and args.diarize_device not in (None, "mps"):
        raise RuntimeError("話者分離は MPS 専用です。--diarize-device には mps 以外を指定できません。")

    prepared: list[PreparedAudio] = []
    for audio_path in args.audio:
        path = Path(audio_path)
        try:
            prepared.append(prepare_audio(path, path.name))
        except InvalidAudioError as exc:
            raise RuntimeError(f"音声ファイルの検証に失敗しました: {audio_path}: {exc}") from exc

    results = transcribe_prepared_audios(
        prepared,
        model_name=model_name,
        language=language,
        task=args.task,
        transcribe_all_fn=transcribe_all_fn,
        decode_options=getattr(args, "_decode_options", None),
    )

    if getattr(args, "diarize", False):
        if args.diarize_device not in (None, "mps"):
            raise RuntimeError("話者分離は MPS 専用です。--diarize-device には mps 以外を指定できません。")
        (DiarizeOptionsCls, attach_speaker_labels_fn, diarize_all_fn) = _load_diarize_components()
        diarize_options = DiarizeOptionsCls(
            token=args.diarize_token,
            num_speakers=args.diarize_num_speakers,
            min_speakers=args.diarize_min_speakers,
            max_speakers=args.diarize_max_speakers,
            device=args.diarize_device,
            require_mps=True,
        )
        diarization_results = diarize_all_fn(args.audio, options=diarize_options)
        diarization_map = {dr.filename: dr for dr in diarization_results}
        speaker_transcripts: dict[str, Any] = {}
        for result in results:
            diar = diarization_map.get(result.filename)
            if diar is None:
                raise RuntimeError(f"話者分離結果が見つかりませんでした: {result.filename}")
            speaker_transcripts[result.filename] = attach_speaker_labels_fn(result, diar)
        args._speaker_transcripts = speaker_transcripts  # type: ignore[attr-defined]
        args._diarization_results = diarization_map  # type: ignore[attr-defined]

    return results


def _handle_frames_command(args: argparse.Namespace) -> CommandResult:
    return _extract_frames(args)


def _handle_diarize_command(args: argparse.Namespace) -> CommandResult:
    return _run_diarize(args)


def _handle_youtube_command(args: argparse.Namespace) -> CommandResult:
    return _run_youtube_fetch(args)


def _validate_youtube_args(args: argparse.Namespace) -> None:
    if not (getattr(args, "youtube_handle", None) or getattr(args, "youtube_channel_id", None)):
        raise ValueError("--handle もしくは --channel-id のいずれかを指定してください。")


_SUBCOMMAND_SPECS: tuple[SubcommandSpec, ...] = (
    SubcommandSpec(
        name="files",
        help="ファイルパスを指定して書き起こす",
        configure=_configure_files_parser,
        handler=_handle_files_command,
        use_shared_parent=True,
    ),
    SubcommandSpec(
        name="stream",
        help="標準入力から受け取った音声ストリームを書き起こす",
        configure=_configure_stream_parser,
        handler=_handle_stream_command,
        use_shared_parent=True,
    ),
    SubcommandSpec(
        name="frames",
        help="動画から代表フレームを抽出する",
        configure=_configure_frames_parser,
        handler=_handle_frames_command,
    ),
    SubcommandSpec(
        name="diarize",
        help="音声ファイルに話者分離のみを実行する",
        configure=_configure_diarize_parser,
        handler=_handle_diarize_command,
    ),
    SubcommandSpec(
        name="youtube-fetch-descriptions",
        help="YouTube チャンネルの動画タイトル/説明文を JSON に保存する",
        configure=_configure_youtube_parser,
        handler=_handle_youtube_command,
        validators=(_validate_youtube_args,),
    ),
)

_SUBCOMMAND_MAP: dict[str, SubcommandSpec] = {}
for spec in _SUBCOMMAND_SPECS:
    _SUBCOMMAND_MAP[spec.name] = spec
    for alias in spec.aliases:
        _SUBCOMMAND_MAP[alias] = spec


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """CLI引数を定義して解析する。"""

    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(description="mlx Whisper を用いた書き起こしCLI")
    subparsers = parser.add_subparsers(dest="command")
    shared_parent = _build_shared_parent_parser()

    for spec in _SUBCOMMAND_SPECS:
        parents = [shared_parent] if spec.use_shared_parent else []
        subparser = subparsers.add_parser(
            spec.name,
            parents=parents,
            help=spec.help,
            description=spec.description or spec.help,
            aliases=list(spec.aliases),
        )
        spec.configure(subparser)

    subparsers.required = False
    parser.set_defaults(command="files")

    known_commands = set(_SUBCOMMAND_MAP.keys())
    if argv and argv[0] not in known_commands:
        argv = ["files", *argv]

    return parser.parse_args(argv)


def run_cli(
    args: argparse.Namespace,
) -> (
    list["TranscriptionResult"]
    | list[FrameExtractionResult]
    | list["DiarizationResult"]
    | YouTubeFetchResult
):
    """コマンド引数を受け取り、書き起こし処理を実行する。"""

    setup_logging(args.log_level)
    ensure_memory_watchdog()

    command = args.command or "files"
    spec = _SUBCOMMAND_MAP.get(command)
    if spec is None:
        raise RuntimeError(f"未対応のコマンドです: {command}")
    for validator in spec.validators:
        validator(args)
    return spec.handler(args)


def _load_diarize_components():
    from src.lib.diarize import (
        DiarizeOptions,
        attach_speaker_labels,
        diarize_all,
    )

    return DiarizeOptions, attach_speaker_labels, diarize_all


def _run_diarize(args: argparse.Namespace) -> list["DiarizationResult"]:
    if args.device not in (None, "mps"):
        raise RuntimeError("話者分離は MPS 専用です。--device には mps 以外を指定できません。")

    DiarizeOptionsCls, _, diarize_all_fn = _load_diarize_components()
    diarize_options = DiarizeOptionsCls(
        token=args.token,
        num_speakers=args.num_speakers,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
        device=args.device,
        require_mps=True,
        sample_rate=args.sample_rate,
    )
    return diarize_all_fn(args.audio, options=diarize_options)


def _load_asr_components():
    from src.lib.asr import TranscriptionResult, transcribe_all, transcribe_all_bytes

    return TranscriptionResult, transcribe_all, transcribe_all_bytes


def _run_youtube_fetch(args: argparse.Namespace) -> YouTubeFetchResult:
    api_key = getattr(args, "youtube_api_key", None) or os.getenv(YOUTUBE_ENV_API_KEY)
    if not api_key:
        raise RuntimeError(
            f"YouTube Data API キーを --api-key もしくは環境変数 {YOUTUBE_ENV_API_KEY} で指定してください。",
        )
    handle = getattr(args, "youtube_handle", None)
    channel_id = getattr(args, "youtube_channel_id", None)
    if not (handle or channel_id):
        raise RuntimeError("--handle もしくは --channel-id のいずれかを指定してください。")
    ensure_ascii = bool(getattr(args, "youtube_ensure_ascii", False))
    pause = max(float(getattr(args, "youtube_pause", 0.1)), 0.0)
    output = Path(getattr(args, "youtube_output"))

    fetcher = YouTubeDescriptionFetcher(api_key=api_key, pause_seconds=pause)
    output_path = fetcher.fetch_and_save(
        output_path=output,
        handle=handle,
        channel_id=channel_id,
        ensure_ascii=ensure_ascii,
    )
    return YouTubeFetchResult(output_path=output_path)


def _extract_frames(args: argparse.Namespace) -> list[FrameExtractionResult]:
    base_output = Path(args.output_dir) if args.output_dir else None
    if base_output:
        base_output.mkdir(parents=True, exist_ok=True)

    video_paths = [Path(p) for p in args.video]
    outputs: list[FrameExtractionResult] = []

    for video_path in video_paths:
        frames = sample_key_frames(
            video_path,
            min_scene_span=float(args.min_scene_span),
            diff_threshold=float(args.diff_threshold),
            max_frames=int(args.max_frames) if args.max_frames is not None else None,
        )
        target_dir = _prepare_output_dir(video_path, base_output, args.overwrite)

        saved_frames: list[SavedFrameInfo] = []
        extension = _normalize_extension(args.image_format)
        for sequence_index, frame in enumerate(frames):
            file_name = _build_frame_filename(video_path.stem, frame, sequence_index, extension)
            output_path = target_dir / file_name
            if not cv2.imwrite(str(output_path), frame.image):
                raise FrameSamplingError(f"フレームを書き込めませんでした: {output_path}")
            saved_frames.append(
                SavedFrameInfo(
                    path=output_path,
                    frame_index=frame.index,
                    timestamp=frame.timestamp,
                )
            )
        outputs.append(FrameExtractionResult(video=video_path, frames=saved_frames))

    return outputs


def _prepare_output_dir(video_path: Path, base_output: Path | None, overwrite: bool) -> Path:
    if base_output is None:
        base = video_path.parent
        prefix = f"{video_path.stem}_frames"
    else:
        base = base_output
        prefix = video_path.stem

    base.mkdir(parents=True, exist_ok=True)

    candidate = base / prefix
    if not candidate.exists():
        candidate.mkdir(parents=True, exist_ok=True)
        return candidate

    if overwrite:
        shutil.rmtree(candidate)
        candidate.mkdir(parents=True, exist_ok=True)
        return candidate

    suffix = 1
    while True:
        alternative = base / f"{prefix}_{suffix:02d}"
        if not alternative.exists():
            alternative.mkdir(parents=True, exist_ok=True)
            return alternative
        suffix += 1


def _normalize_extension(fmt: str) -> str:
    fmt_lower = fmt.lower()
    if fmt_lower in {"jpg", "jpeg"}:
        return "jpg"
    if fmt_lower == "bmp":
        return "bmp"
    return "png"


def _build_frame_filename(prefix: str, frame: SampledFrame, sequence_index: int, extension: str) -> str:
    timestamp_ms = int(round(frame.timestamp * 1000))
    frame_suffix = f"{frame.index:06d}_{sequence_index:04d}"
    return f"{prefix}_{timestamp_ms:08d}_{frame_suffix}.{extension}"


def _streaming_transcription(
    *,
    model_name: str,
    language: str | None,
    task: str | None,
    name: str,
    chunk_size: int,
    interval: float,
    transcribe_all_bytes_fn,
    decode_options: dict[str, Any] | None = None,
    emit_stdout: bool = False,
) -> list["TranscriptionResult"]:
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

        current_results = transcribe_all_bytes_fn(
            [bytes(audio_buffer)],
            model_name=model_name,
            language=language,
            task=task,
            names=[name],
            **(decode_options or {}),
        )
        if not current_results:
            return
        results = current_results
        current_text = current_results[0].text

        if emit_stdout and len(current_text) > len(last_emit_text):
            new_text = current_text[len(last_emit_text) :]
            if new_text:
                sys.stdout.write(new_text)
                sys.stdout.flush()
        last_emit_text = current_text

    try:
        while True:
            chunk = stdin_buffer.read(chunk_size)
            if not chunk:
                break
            audio_buffer.extend(chunk)
            now = time.monotonic()
            if (now - last_flush) >= interval:
                flush(force=True)
                last_flush = now
    except KeyboardInterrupt:
        if audio_buffer:
            logger.debug("stream_interrupt: buffered_bytes=%d", len(audio_buffer))
        else:
            raise

    flush(force=True)
    if results:
        if emit_stdout:
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
        if args.command == "frames":
            raise SystemExit(f"フレーム抽出に失敗しました: {exc}") from exc
        if args.command == "diarize":
            raise SystemExit(f"話者分離に失敗しました: {exc}") from exc
        raise SystemExit(f"書き起こしに失敗しました: {exc}") from exc

    if args.command == "youtube-fetch-descriptions":
        yt_result = cast(YouTubeFetchResult, results)
        print(f"YouTube の説明文を書き出しました: {yt_result.output_path}")
        return
    if args.command == "frames":
        for extraction in results:  # type: ignore[assignment]
            print("=== 動画:", extraction.video)
            if not extraction.frames:
                print("フレームが抽出されませんでした。")
                continue
            for frame in extraction.frames:
                print(f"[{frame.timestamp:.2f}s / #{frame.frame_index}] -> {frame.path}")
            print()
        return
    if args.command == "diarize":
        for result in results:  # type: ignore[assignment]
            print("=== ファイル:", result.filename)
            duration = getattr(result, "duration", None)
            if duration is not None:
                print(f"継続時間: {duration:.2f} 秒")
            print("話者:", ", ".join(result.speakers) or "-")
            if args.show_turns and hasattr(result, "turns"):
                print("--- セグメント一覧 ---")
                for turn in result.turns:
                    speaker = getattr(turn, "speaker", "-")
                    print(f"[{turn.start:.2f}s - {turn.end:.2f}s] {speaker}")
            print()
        return

    # 統合出力フロー（--output 指定時）
    selected_output = getattr(args, "output", None)
    if selected_output in {"json", "mermaid"}:  # type: ignore[truthy-bool]
        # 準備: 話者/ダイアライズ/画面コンテキスト
        speaker_map: dict[str, Any] = getattr(args, "_speaker_transcripts", {})  # type: ignore[assignment]
        diar_map = getattr(args, "_diarization_results", {})
        videos: list[str] = []
        if getattr(args, "video", None):
            videos = list(args.video)

        def video_for_index(i: int) -> str | None:
            if not videos:
                return None
            if i < len(videos):
                return videos[i]
            return videos[-1]

        integrated: list[IntegratedMeetingContext] = []
        for idx, result in enumerate(results):  # type: ignore[assignment]
            video_path = video_for_index(idx)
            image_contexts = None
            if video_path:
                try:
                    image_contexts = build_image_contexts_from_video(
                        video_path,
                        min_scene_span=float(getattr(args, "context_min_scene_span", 1.0)),
                        diff_threshold=float(getattr(args, "context_diff_threshold", 0.3)),
                        max_frames=int(getattr(args, "max_context_frames", 50)),
                    )
                except Exception as exc:  # 解析失敗は致命的でない
                    logger.warning("画面コンテキスト解析に失敗しました: %s", exc)

            st = speaker_map.get(result.filename) if speaker_map else None
            dr = diar_map.get(result.filename) if diar_map else None
            integrated.append(
                integrate_meeting_data(
                    transcript=result,
                    speaker_transcript=st,
                    diarization=dr,
                    image_contexts=image_contexts,
                )
            )

        if selected_output == "json":
            import json

            print(json.dumps([ctx.model_dump() for ctx in integrated], ensure_ascii=False, indent=2))
            return

        if selected_output == "mermaid":
            # 複数ファイル時は区切る
            for i, ctx in enumerate(integrated):
                if i > 0:
                    print()
                print(render_mermaid_mindmap(ctx))
            return

    # 既存の出力（互換維持）
    plain_text = getattr(args, "plain_text", False)
    stream_output = getattr(args, "_stream_output", False)
    for index, result in enumerate(results):  # type: ignore[assignment]
        if plain_text:
            if stream_output:
                continue
            if index > 0:
                print()
            print(result.text)
            continue

        print("=== ファイル:", result.filename)
        print("言語:", result.language or "不明")
        speaker_map: dict[str, Any] = getattr(args, "_speaker_transcripts", {})  # type: ignore[assignment]
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
