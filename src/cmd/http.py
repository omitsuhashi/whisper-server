from __future__ import annotations

import asyncio
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from src.config.defaults import DEFAULT_LANGUAGE, DEFAULT_MODEL_NAME
from src.config.logging import setup_logging
from src.lib.asr import TranscriptionResult, transcribe_all
from src.lib.asr.chunking import transcribe_paths_chunked
from src.lib.asr.service import resolve_model_and_language, transcribe_prepared_audios
from src.lib.audio import (
    InvalidAudioError,
    PreparedAudio,
    dump_audio_for_debug,
    infer_suffix,
    prepare_audio,
)
from src.lib.corrector import CorrectionError, run_correction
from src.cmd.schemas.corrector import (
    CorrectionRequestPayload,
    CorrectionResponsePayload,
    CorrectionPatchPayload,
    build_correction_options,
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

    @app.post("/correct", response_model=CorrectionResponsePayload)
    async def correct_endpoint(payload: CorrectionRequestPayload) -> CorrectionResponsePayload:
        """文脈校正パイプラインを適用して差分パッチを返す。"""

        try:
            options = build_correction_options(payload.options)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        context_prev = payload.context_tuple()
        language = payload.language_or_default()

        logger.debug(
            "correct_request: text_len=%d context=%d language=%s options=%s",
            len(payload.text),
            len(context_prev),
            language,
            options.as_dict(),
        )

        try:
            result = await asyncio.to_thread(
                run_correction,
                payload.text,
                context_prev=context_prev,
                language=language,
                options=options,
            )
        except CorrectionError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        patches = [CorrectionPatchPayload.from_patch(patch) for patch in result.patches]

        logger.debug(
            "correct_response: patch_count=%d text_len=%d",
            len(patches),
            len(result.corrected_text),
        )

        return CorrectionResponsePayload(
            source_text=result.source_text,
            text=result.corrected_text,
            patches=patches,
            patch_count=len(patches),
            options=options.as_dict(),
        )

    @app.post("/transcribe", response_model=list[TranscriptionResult])
    async def transcribe_endpoint(  # noqa: PLR0912 - 例外処理で分岐が増える
        files: list[UploadFile] = File(...),
        model: str = Form(DEFAULT_MODEL_NAME),
        language: Optional[str] = Form(None),
        task: Optional[str] = Form(None),
        chunk_seconds: Optional[float] = Form(None),
        overlap_seconds: Optional[float] = Form(None),
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
            model_name_resolved, language_resolved = resolve_model_and_language(
                model,
                language,
                default_model=DEFAULT_MODEL_NAME,
                default_language=DEFAULT_LANGUAGE,
            )

            # チャンクとオーバーラップ秒の解決: フォーム値 > 環境変数 > 既定
            import os as _os

            default_chunk = 25.0
            default_overlap = 1.0

            def _resolve_float(raw: Optional[str], fallback: float) -> float:
                if raw is None:
                    return fallback
                try:
                    return float(raw)
                except ValueError:
                    return fallback

            env_chunk = _resolve_float(_os.getenv("ASR_CHUNK_SECONDS"), default_chunk)
            env_overlap = _resolve_float(_os.getenv("ASR_OVERLAP_SECONDS"), default_overlap)

            effective_chunk = float(chunk_seconds) if chunk_seconds is not None else env_chunk
            effective_overlap = float(overlap_seconds) if overlap_seconds is not None else env_overlap
            if effective_chunk <= 0:
                effective_overlap = 0.0
            else:
                effective_overlap = max(0.0, min(effective_overlap, effective_chunk / 2))

            logger.debug(
                "transcribe_chunking: chunk_seconds=%s overlap_seconds=%s",
                effective_chunk,
                effective_overlap,
            )

            transcribe_all_fn = None

            if _os.getenv("ASR_HTTP_SUBPROCESS", "0").lower() in {"1", "true", "on", "yes"}:
                from src.lib.asr.subproc import transcribe_bytes_subprocess as _subproc

                def _transcribe_subprocess(paths, *, model_name, language, task):
                    results: list[TranscriptionResult] = []
                    for path_like in paths:
                        path_obj = Path(path_like)
                        data = path_obj.read_bytes()
                        res = _subproc(
                            data,
                            model_name=model_name,
                            language=language,
                            task=task,
                            name=path_obj.name,
                        )
                        if not res:
                            continue
                        first = res[0]
                        if hasattr(first, "model_copy"):
                            first = first.model_copy(update={"filename": path_obj.name})
                        else:
                            setattr(first, "filename", path_obj.name)
                        results.append(first)
                    return results

                transcribe_all_fn = _transcribe_subprocess
            elif effective_chunk > 0:

                def _transcribe_chunked(paths, *, model_name, language, task):
                    return transcribe_paths_chunked(
                        paths,
                        model_name=model_name,
                        language=language,
                        task=task,
                        chunk_seconds=float(effective_chunk),
                        overlap_seconds=float(effective_overlap),
                    )

                transcribe_all_fn = _transcribe_chunked

            updated = await asyncio.to_thread(
                transcribe_prepared_audios,
                entries,
                model_name=model_name_resolved,
                language=language_resolved,
                task=task,
                transcribe_all_fn=transcribe_all_fn,
            )

        except (FileNotFoundError, InvalidAudioError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001 - 予期せぬ障害は500で返す
            logger.exception("書き起こしに失敗しました: %s", [f.filename for f in files])
            raise HTTPException(status_code=500, detail="書き起こしに失敗しました") from exc
        finally:
            for entry in entries:
                entry.path.unlink(missing_ok=True)

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
app = create_app()
