from __future__ import annotations

import asyncio
import json
import logging
import shutil
import tempfile
from pathlib import Path
from typing import Optional
from uuid import uuid4
from subprocess import CalledProcessError, run

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from src.config.defaults import DEFAULT_LANGUAGE, DEFAULT_MODEL_NAME
from src.config.logging import setup_logging
from mlx_whisper.audio import SAMPLE_RATE
from src.lib.asr import TranscriptionResult, transcribe_all
from src.lib.audio import AudioDecodeError, decode_audio_bytes

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    """FastAPIアプリケーションを構築して返す。"""

    setup_logging()
    app = FastAPI(title="mlx Whisper ASR")

    @app.get("/healthz")
    async def health_check() -> dict[str, str]:
        """死活監視用エンドポイント。"""

        return {"status": "ok"}

    @app.post("/transcribe", response_model=list[TranscriptionResult])
    async def transcribe_endpoint(  # noqa: PLR0912 - 例外処理で分岐が増える
        files: list[UploadFile] = File(...),
        model: str = Form(DEFAULT_MODEL_NAME),
        language: Optional[str] = Form(None),
        task: Optional[str] = Form(None),
    ) -> list[TranscriptionResult]:
        """アップロードされた音声群を書き起こして返す。"""

        if not files:
            raise HTTPException(status_code=400, detail="音声ファイルが指定されていません")

        logger.debug(
            "transcribe_request: files=%s model=%s language=%s task=%s",
            [upload.filename for upload in files],
            model or DEFAULT_MODEL_NAME,
            language or DEFAULT_LANGUAGE,
            task,
        )

        tmp_paths: list[Path] = []
        try:
            for upload in files:
                suffix = _infer_suffix(upload.filename)
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    shutil.copyfileobj(upload.file, tmp)
                tmp_path = Path(tmp.name)
                # debug用
                if False:
                    try:
                        _validate_audio_file(tmp_path, upload.filename)
                    except InvalidAudioError as exc:
                        _dump_audio_for_debug(tmp_path, upload.filename)
                        raise HTTPException(status_code=400, detail=str(exc)) from exc

                    _dump_audio_for_debug(tmp_path, upload.filename)
                tmp_paths.append(tmp_path)

            results = await asyncio.to_thread(
                transcribe_all,
                tmp_paths,
                model_name=model or DEFAULT_MODEL_NAME,
                language=language or DEFAULT_LANGUAGE,
                task=task,
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001 - 予期せぬ障害は500で返す
            logger.exception("書き起こしに失敗しました: %s", [f.filename for f in files])
            raise HTTPException(status_code=500, detail="書き起こしに失敗しました") from exc
        finally:
            for path in tmp_paths:
                path.unlink(missing_ok=True)

        if not results or len(results) != len(files):
            raise HTTPException(status_code=500, detail="書き起こし結果が不正です")

        updated: list[TranscriptionResult] = []
        for upload, result in zip(files, results, strict=True):
            display_name = upload.filename or result.filename
            updated.append(result.model_copy(update={"filename": display_name}))

        logger.debug(
            "transcribe_response: %s",
            [
                {
                    "filename": res.filename,
                    "segments": len(res.segments),
                    "language": res.language,
                    "duration": res.duration,
                }
                for res in updated
            ],
        )

        return updated

    return app


def _infer_suffix(filename: Optional[str]) -> str:
    """アップロードされたファイル名から拡張子を推測する。"""

    if not filename:
        return ".tmp"
    suffix = Path(filename).suffix
    return suffix if suffix else ".tmp"


def _dump_audio_for_debug(path: Path, original_name: Optional[str]) -> None:
    """デバッグ用に受信音声をカレントディレクトリへ保存する。"""

    if not logger.isEnabledFor(logging.DEBUG):
        return

    stem = Path(original_name).stem if original_name else path.stem
    suffix = Path(original_name).suffix if original_name else path.suffix
    if not suffix:
        suffix = path.suffix or ".wav"

    safe_stem = stem.replace("/", "_").replace("\\", "_") or "audio"
    dest = Path.cwd() / f"debug_{safe_stem}_{uuid4().hex}{suffix}"

    try:
        shutil.copy2(path, dest)
        logger.debug("transcribe_debug_dump: %s", dest)
    except OSError:
        logger.debug("transcribe_debug_dump_failed: %s", dest, exc_info=True)


class InvalidAudioError(Exception):
    """音声バリデーション失敗時に送出される例外。"""

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


def _validate_audio_file(path: Path, original_name: Optional[str]) -> None:
    """ffprobe/ffmpeg を用いて入力音声の妥当性を確認する。"""

    if not path.exists():
        raise InvalidAudioError("音声ファイルを一時ディスクへ保存できませんでした")
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise InvalidAudioError("音声ファイルを読み取れませんでした") from exc

    if size <= 0:
        raise InvalidAudioError("音声ファイルが空です")

    display_name = original_name or path.name

    try:
        metadata = _probe_audio_metadata(path)
    except FileNotFoundError:
        metadata = None
        logger.warning("ffprobe が見つかりません。音声検証を簡易チェックにフォールバックします。")
    if metadata is None:
        try:
            audio_bytes = path.read_bytes()
        except OSError as exc:  # pragma: no cover - 読み込み失敗は稀
            raise InvalidAudioError("音声ファイルの読み込みに失敗しました") from exc
        try:
            decode_audio_bytes(audio_bytes, sample_rate=SAMPLE_RATE)
        except AudioDecodeError as exc:
            raise InvalidAudioError("音声ファイルをデコードできませんでした") from exc
        logger.debug("transcribe_validation: %s (fallback decode) bytes=%d", display_name, len(audio_bytes))
        return

    duration = metadata.get("duration")
    if duration is not None and duration <= 0:
        raise InvalidAudioError("音声の長さが0秒と判定されました")

    logger.debug(
        "transcribe_validation: %s codec=%s channels=%s sample_rate=%s duration=%.3f",
        display_name,
        metadata.get("codec"),
        metadata.get("channels"),
        metadata.get("sample_rate"),
        duration or -1.0,
    )


def _probe_audio_metadata(path: Path) -> dict[str, float | int | str | None]:
    """ffprobe で音声トラック情報を取得する。"""

    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_streams",
        "-show_format",
        "-select_streams",
        "a:0",
        str(path),
    ]

    try:
        completed = run(cmd, capture_output=True, text=True, check=True)
    except FileNotFoundError as exc:
        raise FileNotFoundError from exc
    except CalledProcessError as exc:
        stderr = exc.stderr.strip() if exc.stderr else "unknown"
        raise InvalidAudioError(f"音声ファイルの解析に失敗しました: {stderr}") from exc

    if not completed.stdout:
        raise InvalidAudioError("音声ストリーム情報を取得できませんでした")

    try:
        probe = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise InvalidAudioError("音声メタデータの解析に失敗しました") from exc

    streams = probe.get("streams") or []
    if not streams:
        raise InvalidAudioError("音声トラックが含まれていません")

    stream = streams[0]

    def _to_float(value: object) -> float | None:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _to_int(value: object) -> int | None:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    duration = _to_float(stream.get("duration")) or _to_float(probe.get("format", {}).get("duration"))
    sample_rate = _to_int(stream.get("sample_rate"))
    channels = _to_int(stream.get("channels"))

    return {
        "codec": stream.get("codec_name"),
        "channels": channels,
        "sample_rate": sample_rate,
        "duration": duration,
    }


app = create_app()
