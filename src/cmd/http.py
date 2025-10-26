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
                    tmp_paths.append(Path(tmp.name))

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


app = create_app()
