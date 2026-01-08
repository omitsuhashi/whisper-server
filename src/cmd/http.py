from __future__ import annotations

import asyncio
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional, Any
from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from mlx_whisper.audio import SAMPLE_RATE

from src.config.defaults import DEFAULT_LANGUAGE, DEFAULT_MODEL_NAME
from src.config.logging import setup_logging
from src.lib.asr import TranscriptionResult, transcribe_all
from src.lib.asr.chunking import transcribe_paths_chunked, transcribe_waveform_chunked
from src.lib.asr.options import TranscribeOptions
from src.lib.asr.pipeline import transcribe_waveform
from src.lib.asr.service import resolve_model_and_language, transcribe_prepared_audios
from src.lib.asr.prompting import build_prompt_from_metadata
from src.lib.audio import (
    AudioDecodeError,
    InvalidAudioError,
    PreparedAudio,
    decode_pcm_s16le_bytes,
    dump_audio_for_debug,
    infer_suffix,
    prepare_audio,
)
from src.lib.diagnostics.memwatch import ensure_memory_watchdog
from src.lib.asr.subproc import transcribe_paths_via_worker

logger = logging.getLogger(__name__)

DEFAULT_CHUNK_SECONDS = 25.0
DEFAULT_OVERLAP_SECONDS = 1.0


def _resolve_float(raw: Optional[str], fallback: float) -> float:
    if raw is None:
        return fallback
    try:
        return float(raw)
    except ValueError:
        return fallback


def _resolve_overlap_seconds(overlap_seconds: Optional[float], *, chunk_seconds: float) -> float:
    env_overlap = _resolve_float(os.getenv("ASR_OVERLAP_SECONDS"), DEFAULT_OVERLAP_SECONDS)
    effective_overlap = float(overlap_seconds) if overlap_seconds is not None else env_overlap
    if chunk_seconds <= 0:
        return 0.0
    return max(0.0, min(effective_overlap, chunk_seconds / 2))


def create_app() -> FastAPI:
    """FastAPIアプリケーションを構築して返す。"""

    setup_logging()
    ensure_memory_watchdog()
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
        prompt_agenda: Optional[str] = Form(None),
        prompt_participants: Optional[str] = Form(None),
        prompt_products: Optional[str] = Form(None),
        prompt_style: Optional[str] = Form(None),
        prompt_terms: Optional[str] = Form(None),
        prompt_dictionary: Optional[str] = Form(None),
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

                if prepared.silent:
                    logger.info("transcribe_silence_detected: %s", prepared.display_name)

                entries.append(prepared)
            model_name_resolved, language_resolved = resolve_model_and_language(
                model,
                language,
                default_model=DEFAULT_MODEL_NAME,
                default_language=DEFAULT_LANGUAGE,
            )

            decode_options: dict[str, Any] = {}
            prompt_value = build_prompt_from_metadata(
                agenda=prompt_agenda,
                participants=prompt_participants,
                products=prompt_products,
                style=prompt_style,
                terms=prompt_terms,
                dictionary=prompt_dictionary,
            )
            if prompt_value:
                decode_options["initial_prompt"] = prompt_value

            # チャンクとオーバーラップ秒の解決: フォーム値 > 環境変数 > 既定
            env_chunk = _resolve_float(os.getenv("ASR_CHUNK_SECONDS"), DEFAULT_CHUNK_SECONDS)
            effective_chunk = float(chunk_seconds) if chunk_seconds is not None else env_chunk
            effective_overlap = _resolve_overlap_seconds(overlap_seconds, chunk_seconds=effective_chunk)

            logger.debug(
                "transcribe_chunking: chunk_seconds=%s overlap_seconds=%s",
                effective_chunk,
                effective_overlap,
            )

            transcribe_all_fn = None

            if os.getenv("ASR_HTTP_SUBPROCESS", "1").lower() in {"1", "true", "on", "yes"}:

                def _transcribe_subprocess(paths, *, model_name, language, task, **decode_kwargs):
                    return transcribe_paths_via_worker(
                        paths,
                        model_name=model_name,
                        language=language,
                        task=task,
                        chunk_seconds=float(effective_chunk),
                        overlap_seconds=float(effective_overlap),
                        **decode_kwargs,
                    )

                transcribe_all_fn = _transcribe_subprocess
            elif effective_chunk > 0:

                def _transcribe_chunked(paths, *, model_name, language, task, **decode_kwargs):
                    return transcribe_paths_chunked(
                        paths,
                        model_name=model_name,
                        language=language,
                        task=task,
                        chunk_seconds=float(effective_chunk),
                        overlap_seconds=float(effective_overlap),
                        **decode_kwargs,
                    )

                transcribe_all_fn = _transcribe_chunked

            updated = await asyncio.to_thread(
                transcribe_prepared_audios,
                entries,
                model_name=model_name_resolved,
                language=language_resolved,
                task=task,
                transcribe_all_fn=transcribe_all_fn,
                decode_options=decode_options or None,
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

    @app.post("/transcribe_pcm", response_model=list[TranscriptionResult])
    async def transcribe_pcm_endpoint(
        file: UploadFile = File(...),
        sample_rate: int = Form(16000),
        model: str = Form(DEFAULT_MODEL_NAME),
        language: Optional[str] = Form(None),
        task: Optional[str] = Form(None),
        chunk_seconds: Optional[float] = Form(None),
        overlap_seconds: Optional[float] = Form(None),
        prompt_agenda: Optional[str] = Form(None),
        prompt_participants: Optional[str] = Form(None),
        prompt_products: Optional[str] = Form(None),
        prompt_style: Optional[str] = Form(None),
        prompt_terms: Optional[str] = Form(None),
        prompt_dictionary: Optional[str] = Form(None),
    ) -> list[TranscriptionResult]:
        """PCM(s16le, mono) をそのまま書き起こして返す。"""

        filename = file.filename or "pcm"
        logger.debug(
            "transcribe_pcm_request: file=%s model=%s language=%s task=%s sample_rate=%s",
            filename,
            model or DEFAULT_MODEL_NAME,
            language or DEFAULT_LANGUAGE,
            task,
            sample_rate,
        )

        if sample_rate <= 0:
            raise HTTPException(status_code=400, detail="sample_rate が不正です")

        pcm_bytes = await file.read()
        try:
            waveform = decode_pcm_s16le_bytes(
                pcm_bytes,
                sample_rate=int(sample_rate),
                target_sample_rate=SAMPLE_RATE,
            )
        except AudioDecodeError as exc:
            if exc.kind == "empty-input":
                detail = "音声データが空でした。"
            elif exc.kind == "empty-output":
                detail = "音声データのデコード結果が空でした。"
            elif exc.kind == "invalid-length":
                detail = "PCM データの長さが不正です。"
            else:
                detail = "音声データのデコードに失敗しました。"
            raise HTTPException(status_code=400, detail=detail) from exc

        model_name_resolved, language_resolved = resolve_model_and_language(
            model,
            language,
            default_model=DEFAULT_MODEL_NAME,
            default_language=DEFAULT_LANGUAGE,
        )
        decode_options: dict[str, Any] = {}
        prompt_value = build_prompt_from_metadata(
            agenda=prompt_agenda,
            participants=prompt_participants,
            products=prompt_products,
            style=prompt_style,
            terms=prompt_terms,
            dictionary=prompt_dictionary,
        )
        if prompt_value:
            decode_options["initial_prompt"] = prompt_value

        options = TranscribeOptions(
            model_name=model_name_resolved,
            language=language_resolved,
            task=task,
            decode_options=decode_options,
        )
        try:
            env_chunk = _resolve_float(os.getenv("ASR_CHUNK_SECONDS"), DEFAULT_CHUNK_SECONDS)
            effective_chunk = float(chunk_seconds) if chunk_seconds is not None else env_chunk
            effective_overlap = _resolve_overlap_seconds(overlap_seconds, chunk_seconds=effective_chunk)
            if effective_chunk > 0:
                result = await asyncio.to_thread(
                    transcribe_waveform_chunked,
                    waveform,
                    options=options,
                    name=filename,
                    chunk_seconds=effective_chunk,
                    overlap_seconds=effective_overlap,
                )
            else:
                result = await asyncio.to_thread(
                    transcribe_waveform,
                    waveform,
                    options=options,
                    name=filename,
                )
        except Exception as exc:  # noqa: BLE001 - 予期せぬ障害は500で返す
            logger.exception("書き起こしに失敗しました: %s", filename)
            raise HTTPException(status_code=500, detail="書き起こしに失敗しました") from exc

        logger.debug(
            "transcribe_pcm_response: %s",
            {
                "filename": result.filename,
                "segments": len(result.segments),
                "language": result.language,
                "duration": result.duration,
            },
        )
        return [result]

    return app
app = create_app()
