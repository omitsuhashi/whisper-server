from __future__ import annotations

import asyncio
import json
import logging
import shutil
import tempfile
from pathlib import Path
from typing import Optional, Sequence
from uuid import uuid4
from subprocess import CalledProcessError, run

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field, model_validator

from src.config.defaults import DEFAULT_LANGUAGE, DEFAULT_MODEL_NAME
from src.config.logging import setup_logging
from mlx_whisper.audio import SAMPLE_RATE
from src.lib.asr import TranscriptionResult, transcribe_all
from src.lib.asr.models import TranscriptionSegment
from src.lib.audio import AudioDecodeError, decode_audio_bytes
from src.lib.polish import PolishOptions, polish_text_from_segments

logger = logging.getLogger(__name__)


class PolishTermPairPayload(BaseModel):
    src: str = Field(..., min_length=1, description="置換対象の文字列")
    dst: str = Field(..., description="置換後の文字列")


class PolishOptionsPayload(BaseModel):
    style: Optional[str] = Field(None, description="文体指定（例: ですます, 常体）")
    use_ginza: Optional[bool] = Field(None, description="GiNZA による文分割を利用するか")
    ginza_model: Optional[str] = Field(None, description="利用する GiNZA モデル名")
    ginza_fallback_to_heuristics: Optional[bool] = Field(
        None, description="GiNZA 失敗時にヒューリスティクスへフォールバックするか"
    )
    remove_fillers: Optional[bool] = Field(None, description="フィラーワードを除去するか")
    filler_patterns: Optional[Sequence[str]] = Field(None, description="フィラーワードとして扱う正規表現の配列")
    normalize_width: Optional[bool] = Field(None, description="NFKC 正規化を適用するか")
    space_collapse: Optional[bool] = Field(None, description="余分なスペースを圧縮するか")
    remove_repeated_chars: Optional[bool] = Field(None, description="連続文字の上限を設けるか")
    max_char_repeat: Optional[int] = Field(None, ge=1, description="連続文字の最大回数")
    term_pairs: Optional[Sequence[PolishTermPairPayload]] = Field(None, description="用語置換の対応表")
    term_pairs_regex: Optional[bool] = Field(None, description="用語置換の左辺を正規表現として扱うか")
    protected_token_patterns: Optional[Sequence[str]] = Field(None, description="保護対象トークンの正規表現集合")
    period_heuristics: Optional[Sequence[str]] = Field(None, description="ヒューリスティクスで終端とみなすパターン")
    min_sentence_len: Optional[int] = Field(None, ge=1, description="ヒューリスティクスで結合する最小文長")
    max_sentence_len: Optional[int] = Field(None, ge=1, description="ヒューリスティクスで強制分割する最大文長")


class PolishSegmentPayload(BaseModel):
    start: float = Field(..., ge=0.0, description="区間開始秒")
    end: float = Field(..., ge=0.0, description="区間終了秒")
    text: str = Field(..., min_length=1, description="書き起こしテキスト")

    @model_validator(mode="after")
    def validate_range(self) -> "PolishSegmentPayload":
        if self.end < self.start:
            raise ValueError("end must be greater than or equal to start")
        return self

    def to_segment(self) -> TranscriptionSegment:
        return TranscriptionSegment.model_validate({"start": self.start, "end": self.end, "text": self.text})


class PolishRequestPayload(BaseModel):
    segments: Sequence[PolishSegmentPayload] = Field(..., description="校正対象のセグメント列")
    options: Optional[PolishOptionsPayload] = Field(None, description="校正オプション")

    @model_validator(mode="after")
    def validate_segments(self) -> "PolishRequestPayload":
        if not self.segments:
            raise ValueError("segments must not be empty")
        return self


class PolishedSentencePayload(BaseModel):
    start: float
    end: float
    text: str


class PolishResponsePayload(BaseModel):
    sentences: list[PolishedSentencePayload]
    text: str
    sentence_count: int


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

        entries: list[dict[str, object]] = []
        try:
            for upload in files:
                suffix = _infer_suffix(upload.filename)
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    shutil.copyfileobj(upload.file, tmp)
                tmp_path = Path(tmp.name)
                display_name = upload.filename or tmp_path.name

                try:
                    _validate_audio_file(tmp_path, upload.filename)
                except InvalidAudioError as exc:
                    _dump_audio_for_debug(tmp_path, upload.filename)
                    raise HTTPException(status_code=400, detail=str(exc)) from exc

                _dump_audio_for_debug(tmp_path, upload.filename)

                is_silent = _is_silent_audio(tmp_path)
                if is_silent:
                    logger.info("transcribe_silence_detected: %s", display_name)

                entries.append(
                    {
                        "path": tmp_path,
                        "display_name": display_name,
                        "silent": is_silent,
                    }
                )

            non_silent_paths = [entry["path"] for entry in entries if not entry["silent"]]
            results: list[TranscriptionResult] = []
            if non_silent_paths:
                results = await asyncio.to_thread(
                    transcribe_all,
                    non_silent_paths,
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
            for entry in entries:
                path = entry["path"]
                assert isinstance(path, Path)
                path.unlink(missing_ok=True)

        non_silent_count = sum(1 for entry in entries if not entry["silent"])
        if non_silent_count != len(results):
            raise HTTPException(status_code=500, detail="書き起こし結果が不正です")

        language_hint = language or DEFAULT_LANGUAGE
        updated: list[TranscriptionResult] = []
        result_iter = iter(results)
        for entry in entries:
            display_name = entry["display_name"]
            assert isinstance(display_name, str)
            if entry["silent"]:
                updated.append(
                    TranscriptionResult(
                        filename=display_name,
                        text="",
                        language=language_hint,
                        duration=0.0,
                        segments=[],
                    )
                )
                continue

            result = next(result_iter)
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

    @app.post("/polish", response_model=PolishResponsePayload)
    async def polish_endpoint(payload: PolishRequestPayload) -> PolishResponsePayload:
        """書き起こしセグメントを校正して返す。"""

        try:
            options = _build_polish_options(payload.options)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        segments = [item.to_segment() for item in payload.segments]

        logger.debug(
            "polish_request: segments=%d options=%s",
            len(segments),
            payload.options.model_dump(exclude_unset=True) if payload.options else {},
        )

        try:
            sentences = polish_text_from_segments(segments, options=options)
        except RuntimeError as exc:
            logger.exception("文章構成に失敗しました")
            raise HTTPException(status_code=500, detail="文章構成に失敗しました") from exc

        response_sentences = [PolishedSentencePayload(**sentence.model_dump()) for sentence in sentences]
        combined_text = "\n".join(sentence.text for sentence in sentences).strip()

        logger.debug("polish_response: sentences=%d", len(response_sentences))

        return PolishResponsePayload(
            sentences=response_sentences,
            text=combined_text,
            sentence_count=len(response_sentences),
        )

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


def _is_silent_audio(path: Path, *, threshold: float = 5e-4) -> bool:
    """音声ファイルが無音に近いかを簡易判定する。"""

    try:
        audio_bytes = path.read_bytes()
    except OSError:
        logger.debug("transcribe_silence_check_failed: read_error path=%s", path, exc_info=True)
        return False

    try:
        waveform = decode_audio_bytes(audio_bytes, sample_rate=SAMPLE_RATE)
    except AudioDecodeError as exc:
        logger.debug("transcribe_silence_check_failed: decode_error=%s", exc)
        return False

    if waveform.size == 0:
        return True

    energy = float(np.mean(np.abs(waveform)))
    peak = float(np.max(np.abs(waveform))) if waveform.size else 0.0
    logger.debug(
        "transcribe_silence_metrics: samples=%d energy=%.6f peak=%.6f",
        waveform.size,
        energy,
        peak,
    )
    return energy < threshold and peak < threshold * 5


def _build_polish_options(payload: PolishOptionsPayload | None) -> PolishOptions:
    if payload is None:
        return PolishOptions()

    data = payload.model_dump(exclude_unset=True)

    term_pairs = data.get("term_pairs")
    if term_pairs is not None:
        data["term_pairs"] = tuple((pair["src"], pair["dst"]) for pair in term_pairs)

    for key in ("filler_patterns", "protected_token_patterns", "period_heuristics"):
        value = data.get(key)
        if value is not None:
            data[key] = tuple(value)

    return PolishOptions(**data)


app = create_app()
