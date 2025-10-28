from __future__ import annotations

import asyncio
import logging
import shutil
import tempfile
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from src.config.defaults import DEFAULT_LANGUAGE, DEFAULT_MODEL_NAME
from src.config.logging import setup_logging
from src.lib.asr import TranscriptionResult, transcribe_all
from src.lib.polish import polish_text_from_segments
from src.lib.audio import (
    InvalidAudioError,
    PreparedAudio,
    dump_audio_for_debug,
    infer_suffix,
    prepare_audio,
)
from src.cmd.schemas.polish import (
    PolishOptionsPayload,
    PolishRequestPayload,
    PolishResponsePayload,
    PolishedSentencePayload,
    build_polish_options,
)

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

        entries: list[PreparedAudio] = []
        try:
            for upload in files:
                suffix = infer_suffix(upload.filename)
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    shutil.copyfileobj(upload.file, tmp)
                tmp_path = Path(tmp.name)
                try:
                    prepared = prepare_audio(tmp_path, upload.filename)
                except InvalidAudioError as exc:
                    dump_audio_for_debug(tmp_path, upload.filename)
                    raise HTTPException(status_code=400, detail=str(exc)) from exc

                dump_audio_for_debug(tmp_path, upload.filename)

                if prepared.silent:
                    logger.info("transcribe_silence_detected: %s", prepared.display_name)

                entries.append(prepared)

            non_silent_paths = [entry.path for entry in entries if not entry.silent]
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
                entry.path.unlink(missing_ok=True)

        non_silent_count = sum(1 for entry in entries if not entry.silent)
        if non_silent_count != len(results):
            raise HTTPException(status_code=500, detail="書き起こし結果が不正です")

        language_hint = language or DEFAULT_LANGUAGE
        updated: list[TranscriptionResult] = []
        result_iter = iter(results)
        for entry in entries:
            display_name = entry.display_name
            if entry.silent:
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
            options = build_polish_options(payload.options)
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
app = create_app()
