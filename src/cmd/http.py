from __future__ import annotations

import asyncio
from dataclasses import dataclass
import logging
import os
import shutil
import tempfile
import time
import uuid
from pathlib import Path
from typing import Optional, Any
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Request

from mlx_whisper.audio import SAMPLE_RATE

from src.config.defaults import DEFAULT_LANGUAGE, DEFAULT_MODEL_NAME
from src.config.logging import setup_logging
from src.lib.asr import TranscriptionResult, transcribe_all
from src.lib.asr.chunking import transcribe_paths_chunked, transcribe_waveform_chunked
from src.lib.asr.filler import apply_filler_removal
from src.lib.asr.options import TranscribeOptions
from src.lib.asr.pipeline import transcribe_waveform
from src.lib.asr.service import resolve_model_and_language, transcribe_prepared_audios
from src.lib.asr.prompting import build_prompt_from_metadata, normalize_prompt_items
from src.lib.asr.quality import analyze_transcription_quality
from src.lib.audio import (
    AudioDecodeError,
    InvalidAudioError,
    PreparedAudio,
    decode_pcm_s16le_bytes,
    dump_audio_for_debug,
    infer_suffix,
    prepare_audio,
)
from src.lib.diagnostics.request_context import set_request_context, reset_request_context
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


def _coerce_duration(duration: float | None) -> float:
    if duration is None:
        return 0.0
    return float(duration)


def _resolve_overlap_seconds(overlap_seconds: Optional[float], *, chunk_seconds: float) -> float:
    env_overlap = _resolve_float(os.getenv("ASR_OVERLAP_SECONDS"), DEFAULT_OVERLAP_SECONDS)
    effective_overlap = float(overlap_seconds) if overlap_seconds is not None else env_overlap
    if chunk_seconds <= 0:
        return 0.0
    return max(0.0, min(effective_overlap, chunk_seconds / 2))


def _use_default_style_prompt() -> bool:
    raw = (os.getenv("ASR_DEFAULT_STYLE_PROMPT", "1") or "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _filler_enabled(mode: str) -> bool:
    cleaned = (mode or "").strip().lower()
    return cleaned in {"final", "clean"}


@dataclass(frozen=True)
class ResolvedTranscribeInputs:
    model_name: str
    language: str
    task: Optional[str]
    chunk_seconds: float
    overlap_seconds: float
    decode_options: dict[str, Any] | None
    prompt_terms_count: int
    prompt_dict_count: int
    prompt_chars: int


def _resolve_transcribe_inputs(
    *,
    model: str,
    language: Optional[str],
    task: Optional[str],
    chunk_seconds: Optional[float],
    overlap_seconds: Optional[float],
    prompt_agenda: Optional[str],
    prompt_participants: Optional[str],
    prompt_products: Optional[str],
    prompt_style: Optional[str],
    prompt_terms: Optional[str],
    prompt_dictionary: Optional[str],
) -> ResolvedTranscribeInputs:
    def _has_prompt_value(value: Optional[str]) -> bool:
        return bool(value and value.strip())

    model_name_resolved, language_resolved = resolve_model_and_language(
        model,
        language,
        default_model=DEFAULT_MODEL_NAME,
        default_language=DEFAULT_LANGUAGE,
    )
    terms_items = normalize_prompt_items(prompt_terms)
    dict_items = normalize_prompt_items(prompt_dictionary)
    has_prompt_metadata = any(
        (
            _has_prompt_value(prompt_agenda),
            _has_prompt_value(prompt_participants),
            _has_prompt_value(prompt_products),
            _has_prompt_value(prompt_style),
            bool(terms_items),
            bool(dict_items),
        )
    )

    if has_prompt_metadata:
        prompt_value = build_prompt_from_metadata(
            agenda=prompt_agenda,
            participants=prompt_participants,
            products=prompt_products,
            style=prompt_style,
            terms=prompt_terms,
            dictionary=prompt_dictionary,
            language=language_resolved,
        )
    elif _use_default_style_prompt():
        prompt_value = build_prompt_from_metadata(language=language_resolved)
    else:
        prompt_value = None

    decode_options = {"initial_prompt": prompt_value} if prompt_value else None

    env_chunk = _resolve_float(os.getenv("ASR_CHUNK_SECONDS"), DEFAULT_CHUNK_SECONDS)
    effective_chunk = float(chunk_seconds) if chunk_seconds is not None else env_chunk
    effective_overlap = _resolve_overlap_seconds(overlap_seconds, chunk_seconds=effective_chunk)

    return ResolvedTranscribeInputs(
        model_name=model_name_resolved,
        language=language_resolved,
        task=task,
        chunk_seconds=effective_chunk,
        overlap_seconds=effective_overlap,
        decode_options=decode_options,
        prompt_terms_count=len(terms_items),
        prompt_dict_count=len(dict_items),
        prompt_chars=len(prompt_value or ""),
    )


def create_app() -> FastAPI:
    """FastAPIアプリケーションを構築して返す。"""

    setup_logging()
    ensure_memory_watchdog()
    app = FastAPI(title="mlx Whisper ASR")

    @app.middleware("http")
    async def _attach_request_context(request: Request, call_next):
        req_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
        sess_id = request.headers.get("X-Whisper-Session")
        mode = request.headers.get("X-Whisper-Mode") or "-"
        request.state.whisper_mode = mode
        tokens = set_request_context(req_id, sess_id)
        started = time.monotonic()
        status_code: int | None = None
        try:
            response = await call_next(request)
            status_code = response.status_code
            response.headers["X-Request-ID"] = req_id
            return response
        except HTTPException as exc:
            status_code = exc.status_code
            raise
        except Exception:
            status_code = 500
            raise
        finally:
            elapsed_ms = int((time.monotonic() - started) * 1000)
            logger.info(
                "http_request_end method=%s path=%s status=%s mode=%s duration_ms=%d",
                request.method,
                request.url.path,
                status_code if status_code is not None else 500,
                mode,
                elapsed_ms,
            )
            reset_request_context(tokens)

    @app.get("/healthz")
    async def health_check() -> dict[str, str]:
        """死活監視用エンドポイント。"""

        return {"status": "ok"}

    @app.post("/transcribe", response_model=list[TranscriptionResult])
    async def transcribe_endpoint(  # noqa: PLR0912 - 例外処理で分岐が増える
        request: Request,
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

        mode = getattr(request.state, "whisper_mode", "-")
        started = time.monotonic()

        if not files:
            raise HTTPException(status_code=400, detail="音声ファイルが指定されていません")

        inputs = _resolve_transcribe_inputs(
            model=model,
            language=language,
            task=task,
            chunk_seconds=chunk_seconds,
            overlap_seconds=overlap_seconds,
            prompt_agenda=prompt_agenda,
            prompt_participants=prompt_participants,
            prompt_products=prompt_products,
            prompt_style=prompt_style,
            prompt_terms=prompt_terms,
            prompt_dictionary=prompt_dictionary,
        )

        logger.debug(
            "transcribe_request: files=%s model=%s language=%s task=%s",
            [upload.filename for upload in files],
            model or DEFAULT_MODEL_NAME,
            language or DEFAULT_LANGUAGE,
            task,
        )

        logger.info(
            "transcribe_start mode=%s files=%d model=%s language=%s task=%s chunk=%s overlap=%s terms=%d dict=%d prompt_chars=%d",
            mode,
            len(files),
            inputs.model_name,
            inputs.language,
            inputs.task,
            inputs.chunk_seconds,
            inputs.overlap_seconds,
            inputs.prompt_terms_count,
            inputs.prompt_dict_count,
            inputs.prompt_chars,
        )

        entries: list[PreparedAudio] = []
        asr_start = None
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
                    logger.warning(
                        "transcribe_failed kind=invalid_audio file=%s",
                        upload.filename,
                    )
                    raise HTTPException(status_code=400, detail=str(exc)) from exc

                if prepared.silent:
                    logger.info("transcribe_silence_detected: %s", prepared.display_name)

                entries.append(prepared)

            logger.debug(
                "transcribe_chunking: chunk_seconds=%s overlap_seconds=%s",
                inputs.chunk_seconds,
                inputs.overlap_seconds,
            )

            transcribe_all_fn = None

            if os.getenv("ASR_HTTP_SUBPROCESS", "1").lower() in {"1", "true", "on", "yes"}:

                def _transcribe_subprocess(paths, *, model_name, language, task, **decode_kwargs):
                    return transcribe_paths_via_worker(
                        paths,
                        model_name=model_name,
                        language=language,
                        task=task,
                        chunk_seconds=float(inputs.chunk_seconds),
                        overlap_seconds=float(inputs.overlap_seconds),
                        **decode_kwargs,
                    )

                transcribe_all_fn = _transcribe_subprocess
            elif inputs.chunk_seconds > 0:

                def _transcribe_chunked(paths, *, model_name, language, task, **decode_kwargs):
                    return transcribe_paths_chunked(
                        paths,
                        model_name=model_name,
                        language=language,
                        task=task,
                        chunk_seconds=float(inputs.chunk_seconds),
                        overlap_seconds=float(inputs.overlap_seconds),
                        **decode_kwargs,
                    )

                transcribe_all_fn = _transcribe_chunked

            asr_start = time.monotonic()
            updated = await asyncio.to_thread(
                transcribe_prepared_audios,
                entries,
                model_name=inputs.model_name,
                language=inputs.language,
                task=inputs.task,
                transcribe_all_fn=transcribe_all_fn,
                decode_options=inputs.decode_options,
            )

        except (FileNotFoundError, InvalidAudioError) as exc:
            logger.warning("transcribe_failed kind=invalid_audio detail=%s", str(exc))
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001 - 予期せぬ障害は500で返す
            logger.exception("transcribe_failed kind=server_error files=%s", [f.filename for f in files])
            raise HTTPException(status_code=500, detail="書き起こしに失敗しました") from exc
        finally:
            for entry in entries:
                entry.path.unlink(missing_ok=True)

        asr_end = time.monotonic()
        asr_started = asr_start or started
        prep_ms = int((asr_started - started) * 1000)
        asr_ms = int((asr_end - asr_started) * 1000)
        total_ms = int((asr_end - started) * 1000)
        segments_total = sum(len(res.segments) for res in updated)
        duration_total = sum(_coerce_duration(res.duration) for res in updated)

        logger.info(
            "transcribe_done mode=%s files=%d results=%d segments=%d duration_sec=%.3f prep_ms=%d asr_ms=%d total_ms=%d",
            mode,
            len(files),
            len(updated),
            segments_total,
            duration_total,
            prep_ms,
            asr_ms,
            total_ms,
        )

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

        if _filler_enabled(mode):
            updated = [apply_filler_removal(res, enabled=True) for res in updated]
        updated = [
            res.model_copy(update={"diagnostics": analyze_transcription_quality(res)})
            for res in updated
        ]

        return updated

    @app.post("/transcribe_pcm", response_model=list[TranscriptionResult])
    async def transcribe_pcm_endpoint(
        request: Request,
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
        mode = getattr(request.state, "whisper_mode", "-")
        started = time.monotonic()

        inputs = _resolve_transcribe_inputs(
            model=model,
            language=language,
            task=task,
            chunk_seconds=chunk_seconds,
            overlap_seconds=overlap_seconds,
            prompt_agenda=prompt_agenda,
            prompt_participants=prompt_participants,
            prompt_products=prompt_products,
            prompt_style=prompt_style,
            prompt_terms=prompt_terms,
            prompt_dictionary=prompt_dictionary,
        )

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
            logger.warning(
                "transcribe_pcm_failed kind=decode_error file=%s reason=%s",
                filename,
                exc.kind,
            )
            raise HTTPException(status_code=400, detail=detail) from exc

        logger.info(
            "transcribe_pcm_start mode=%s file=%s bytes=%d sample_rate=%d model=%s language=%s task=%s chunk=%s overlap=%s terms=%d dict=%d prompt_chars=%d",
            mode,
            filename,
            len(pcm_bytes),
            int(sample_rate),
            inputs.model_name,
            inputs.language,
            inputs.task,
            inputs.chunk_seconds,
            inputs.overlap_seconds,
            inputs.prompt_terms_count,
            inputs.prompt_dict_count,
            inputs.prompt_chars,
        )

        options = TranscribeOptions(
            model_name=inputs.model_name,
            language=inputs.language,
            task=inputs.task,
            decode_options=inputs.decode_options or {},
        )
        try:
            asr_start = time.monotonic()
            if inputs.chunk_seconds > 0:
                result = await asyncio.to_thread(
                    transcribe_waveform_chunked,
                    waveform,
                    options=options,
                    name=filename,
                    chunk_seconds=inputs.chunk_seconds,
                    overlap_seconds=inputs.overlap_seconds,
                )
            else:
                result = await asyncio.to_thread(
                    transcribe_waveform,
                    waveform,
                    options=options,
                    name=filename,
                )
        except Exception as exc:  # noqa: BLE001 - 予期せぬ障害は500で返す
            logger.exception("transcribe_pcm_failed kind=server_error file=%s", filename)
            raise HTTPException(status_code=500, detail="書き起こしに失敗しました") from exc

        asr_end = time.monotonic()
        prep_ms = int((asr_start - started) * 1000)
        asr_ms = int((asr_end - asr_start) * 1000)
        total_ms = int((asr_end - started) * 1000)

        logger.info(
            "transcribe_pcm_done mode=%s file=%s segments=%d duration_sec=%.3f prep_ms=%d asr_ms=%d total_ms=%d",
            mode,
            result.filename,
            len(result.segments),
            _coerce_duration(result.duration),
            prep_ms,
            asr_ms,
            total_ms,
        )

        logger.debug(
            "transcribe_pcm_response: %s",
            {
                "filename": result.filename,
                "segments": len(result.segments),
                "language": result.language,
                "duration": result.duration,
            },
        )
        enabled = _filler_enabled(mode)
        result = apply_filler_removal(result, enabled=enabled)
        result = result.model_copy(update={"diagnostics": analyze_transcription_quality(result)})
        return [result]

    return app
app = create_app()
