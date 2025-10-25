from __future__ import annotations

import asyncio
import logging
import shutil
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket
from fastapi import WebSocketDisconnect

from src.config.defaults import DEFAULT_LANGUAGE, DEFAULT_MODEL_NAME
from src.lib.asr import TranscriptionResult, transcribe_all, transcribe_all_bytes

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

    @app.websocket("/ws/transcribe")
    async def transcribe_stream(
        websocket: WebSocket,
        model: str = DEFAULT_MODEL_NAME,
        language: Optional[str] = None,
        task: Optional[str] = None,
    ) -> None:
        """WebSocket経由で音声バイト列を受け取り、書き起こし結果を返す。"""

        await websocket.accept()
        audio_buffer = bytearray()
        last_text = ""
        closed = False

        async def emit(final: bool) -> None:
            nonlocal last_text, closed

            if not audio_buffer:
                await websocket.send_json({"error": "音声データが送信されていません。", "final": final})
                if final:
                    await websocket.close(code=1007)
                    closed = True
                return

            try:
                results = await asyncio.to_thread(
                    transcribe_all_bytes,
                    [bytes(audio_buffer)],
                    model_name=model or DEFAULT_MODEL_NAME,
                    language=language or DEFAULT_LANGUAGE,
                    task=task,
                )
            except Exception as exc:  # noqa: BLE001
                logger.exception("WebSocket書き起こし中にエラーが発生しました")
                await websocket.send_json({"error": "書き起こしに失敗しました。", "final": final})
                await websocket.close(code=1011)
                closed = True
                return

            if not results:
                await websocket.send_json({"error": "書き起こし結果が空でした。", "final": final})
                await websocket.close(code=1011)
                closed = True
                return

            result = results[0]
            text = result.text or ""
            delta = text[len(last_text) :] if text.startswith(last_text) else text
            last_text = text

            payload = result.model_dump()
            payload.update({"delta": delta, "final": final})
            await websocket.send_json(payload)

            if final:
                await websocket.close(code=1000)
                closed = True

        try:
            while True:
                message = await websocket.receive()
                data = message.get("bytes")
                if data:
                    audio_buffer.extend(data)
                    continue

                text = message.get("text")
                if text is None:
                    continue

                command = text.strip().lower()
                if command in {"flush", "partial"}:
                    await emit(final=False)
                elif command in {"done", "finish", "close"}:
                    await emit(final=True)
                    break
        except WebSocketDisconnect:
            return
        except Exception as exc:  # noqa: BLE001 - 予期せぬ障害はクライアントへ通知する
            logger.exception("WebSocket受信中にエラーが発生しました")
            await websocket.send_json({"error": f"受信中にエラーが発生しました: {exc}", "final": False})
            await websocket.close(code=1011)
            return

        if not closed and audio_buffer:
            await emit(final=True)

    return app


def _infer_suffix(filename: Optional[str]) -> str:
    """アップロードされたファイル名から拡張子を推測する。"""

    if not filename:
        return ".tmp"
    suffix = Path(filename).suffix
    return suffix if suffix else ".tmp"


app = create_app()
