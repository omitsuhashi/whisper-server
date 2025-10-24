from __future__ import annotations

import asyncio
import logging
import shutil
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from ..config.defaults import DEFAULT_LANGUAGE, DEFAULT_MODEL_NAME
from ..lib.asr import TranscriptionResult, transcribe_all

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    """FastAPIアプリケーションを構築して返す。"""

    app = FastAPI(title="mlx Whisper ASR")

    @app.get("/healthz")
    async def health_check() -> dict[str, str]:
        """死活監視用エンドポイント。"""

        return {"status": "ok"}

    @app.post("/transcribe", response_model=TranscriptionResult)
    async def transcribe_endpoint(
        file: UploadFile = File(...),
        model: str = Form(DEFAULT_MODEL_NAME),
        language: Optional[str] = Form(None),
        task: Optional[str] = Form(None),
    ) -> TranscriptionResult:
        """アップロードされた音声を書き起こして返す。"""

        tmp_path: Path | None = None
        try:
            suffix = _infer_suffix(file.filename)
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                shutil.copyfileobj(file.file, tmp)
                tmp_path = Path(tmp.name)

            results = await asyncio.to_thread(
                transcribe_all,
                [tmp_path],
                model_name=model or DEFAULT_MODEL_NAME,
                language=language or DEFAULT_LANGUAGE,
                task=task,
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001 - 予期せぬ障害は500で返す
            logger.exception("書き起こしに失敗しました: %s", file.filename or "upload")
            raise HTTPException(status_code=500, detail="書き起こしに失敗しました") from exc
        finally:
            if tmp_path is not None:
                tmp_path.unlink(missing_ok=True)

        if not results:
            raise HTTPException(status_code=500, detail="書き起こし結果が空でした")

        result = results[0]
        display_name = file.filename or result.filename
        return result.model_copy(update={"filename": display_name})

    return app


def _infer_suffix(filename: Optional[str]) -> str:
    """アップロードされたファイル名から拡張子を推測する。"""

    if not filename:
        return ".tmp"
    suffix = Path(filename).suffix
    return suffix if suffix else ".tmp"


app = create_app()
