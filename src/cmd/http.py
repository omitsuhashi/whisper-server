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
from src.lib.asr.chunking import transcribe_paths_chunked
from src.lib.polish import LLMPolishError, LLMPolisher, polish_text_from_segments, unload_llm_models
from src.lib.asr.service import transcribe_prepared_audios, resolve_model_and_language
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
    LLMPolishRequestPayload,
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
            # サブプロセス優先モード
            if _os.getenv("ASR_HTTP_SUBPROCESS", "0").lower() in {"1", "true", "on", "yes"}:
                from src.lib.asr.subproc import transcribe_bytes_subprocess as _subproc

                updated: list[TranscriptionResult] = []
                for entry in entries:
                    if entry.silent:
                        updated.append(
                            TranscriptionResult(
                                filename=entry.display_name,
                                text="",
                                language=language_resolved,
                                duration=0.0,
                                segments=[],
                            )
                        )
                        continue
                    data = entry.path.read_bytes()
                    res = await asyncio.to_thread(
                        _subproc,
                        data,
                        model_name=model_name_resolved,
                        language=language_resolved,
                        task=task,
                        name=entry.display_name,
                    )
                    if res:
                        updated.append(res[0])
            else:
                # チャンク処理（非サブプロセス）
                non_silent_paths = [str(e.path) for e in entries if not e.silent]
                if non_silent_paths:
                    chunked = await asyncio.to_thread(
                        transcribe_paths_chunked,
                        non_silent_paths,
                        model_name=model_name_resolved,
                        language=language_resolved,
                        task=task,
                        chunk_seconds=float(effective_chunk),
                        overlap_seconds=float(effective_overlap),
                    )
                else:
                    chunked = []

                # 元の順序に合わせて filename などを反映
                it = iter(chunked)
                updated = []
                for entry in entries:
                    if entry.silent:
                        updated.append(
                            TranscriptionResult(
                                filename=entry.display_name,
                                text="",
                                language=language_resolved,
                                duration=0.0,
                                segments=[],
                            )
                        )
                    else:
                        res = next(it)
                        # display_name を上書き
                        if hasattr(res, "model_copy"):
                            res = res.model_copy(update={"filename": entry.display_name})
                        else:
                            setattr(res, "filename", entry.display_name)
                        updated.append(res)

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

    @app.post("/polish", response_model=PolishResponsePayload)
    async def polish_endpoint(payload: PolishRequestPayload) -> PolishResponsePayload:
        """書き起こしセグメントを校正して返す。"""

        try:
            options = build_polish_options(payload.options)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        segments = list(payload.to_segments())

        logger.debug(
            "polish_request: source=%s segments=%d text_len=%d options=%s",
            "segments" if payload.segments else "text",
            len(segments),
            len(payload.text or ""),
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

    @app.post("/polish/llm", response_model=PolishResponsePayload)
    async def polish_llm_endpoint(payload: LLMPolishRequestPayload) -> PolishResponsePayload:
        """外部 LLM を用いた校正を行う。"""

        try:
            options = build_polish_options(payload.options)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        segments = list(payload.to_segments())

        logger.debug(
            "polish_llm_request: source=%s segments=%d text_len=%d options=%s style=%s",
            "segments" if payload.segments else "text",
            len(segments),
            len(payload.text or ""),
            payload.options.model_dump(exclude_unset=True) if payload.options else {},
            payload.style or options.style,
        )

        base_sentences = polish_text_from_segments(segments, options=options)

        try:
            polisher = LLMPolisher(
                model_id=payload.model_id,
                temperature=payload.temperature if payload.temperature is not None else 0.2,
                top_p=payload.top_p if payload.top_p is not None else 0.9,
                max_tokens=payload.max_tokens if payload.max_tokens is not None else 800,
            )
        except ValueError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

        try:
            polished_sentences = await asyncio.to_thread(
                polisher.polish,
                base_sentences,
                style=payload.style or options.style,
                extra_instructions=payload.extra_instructions,
                parameters=payload.parameters,
            )
        except LLMPolishError as exc:
            logger.exception("LLM 校正に失敗しました")
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        finally:
            # 非キャッシュ or 明示解放要求時は早期に参照を外す
            try:
                polisher.close()
            except Exception:
                pass

        # 環境変数で即時アンロードを選べるようにする（既定: 無効）
        if os.getenv("LLM_POLISH_EAGER_UNLOAD", "0").lower() in {"1", "true", "on", "yes"}:
            target_model_id = payload.model_id or os.getenv("LLM_POLISH_MODEL")
            try:
                removed = unload_llm_models(target_model_id)
                logger.debug("llm_model_unloaded: model=%s removed=%d", target_model_id, removed)
            except Exception:
                logger.debug("llm_model_unload_failed", exc_info=True)

        response_sentences = [PolishedSentencePayload(**sentence.model_dump()) for sentence in polished_sentences]
        combined_text = "\n".join(sentence.text for sentence in polished_sentences).strip()

        logger.debug("polish_llm_response: sentences=%d", len(response_sentences))

        return PolishResponsePayload(
            sentences=response_sentences,
            text=combined_text,
            sentence_count=len(response_sentences),
        )

    return app
app = create_app()
