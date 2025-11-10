from __future__ import annotations

import logging
import math
import multiprocessing as mp
import os
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Iterable

from src.lib.asr.models import TranscriptionResult

logger = logging.getLogger(__name__)

_IDLE_TIMEOUT = max(float(os.getenv("ASR_SUBPROC_IDLE_SECONDS", "600") or 0.0), 0.0)
_REQUEST_TIMEOUT = max(float(os.getenv("ASR_SUBPROC_REQUEST_TIMEOUT", "300") or 0.0), 0.0) or None
_TEST_MODE = os.getenv("ASR_SUBPROC_TEST_MODE", "0").lower() in {"1", "true", "on", "yes"}

_WORKER_LOCK = threading.Lock()
_WORKER_HANDLE: "_WorkerHandle | None" = None
_SHUTDOWN_TIMER: threading.Timer | None = None
_CHUNK_TRANSCRIBE: Callable[..., list[TranscriptionResult]] | None = None


class _WorkerHandle:
    def __init__(self) -> None:
        ctx = mp.get_context("spawn")
        parent_conn, child_conn = ctx.Pipe()
        self._conn = parent_conn
        self._process = ctx.Process(target=_worker_entrypoint, args=(child_conn,), daemon=True)
        self._process.start()
        child_conn.close()
        logger.debug("ASRサブプロセスを起動: pid=%s", self._process.pid)

    @property
    def pid(self) -> int | None:
        return self._process.pid if self._process.is_alive() else None

    def request(self, payload: dict, timeout: float | None) -> dict:
        self._conn.send(payload)
        if timeout is not None:
            if not self._conn.poll(timeout):
                raise TimeoutError("ASR worker response timed out")
            return self._conn.recv()
        return self._conn.recv()

    def is_alive(self) -> bool:
        return self._process.is_alive()

    def shutdown(self) -> None:
        try:
            self._conn.send({"cmd": "shutdown"})
        except Exception:
            pass
        try:
            self._conn.close()
        except Exception:
            pass
        self._process.join(timeout=1.0)
        if self._process.is_alive():
            self._process.kill()


def _worker_entrypoint(conn: mp.connection.Connection) -> None:
    while True:
        try:
            message = conn.recv()
        except EOFError:
            break
        command = message.get("cmd")
        if command == "shutdown":
            break
        if command != "transcribe_paths":
            conn.send({"ok": False, "error": f"unknown_command:{command}"})
            continue
        try:
            payload = _handle_transcribe(message)
            conn.send({"ok": True, "results": payload})
        except Exception as exc:  # pragma: no cover - 例外は親側で扱う
            conn.send({"ok": False, "error": str(exc), "traceback": traceback.format_exc()})
    conn.close()


def _handle_transcribe(message: dict) -> list[dict]:
    paths = [Path(p) for p in message.get("paths") or []]
    model_name = message.get("model_name")
    language = message.get("language")
    task = message.get("task")
    decode_options = message.get("decode_options") or {}
    chunk_seconds = message.get("chunk_seconds")
    overlap_seconds = message.get("overlap_seconds")

    if _TEST_MODE:
        return [
            {
                "filename": path.name,
                "text": f"{path.name}:{os.getpid()}",
                "language": language,
                "duration": 0.0,
                "segments": [],
            }
            for path in paths
        ]

    chunk_value = _clamp_non_negative(chunk_seconds)
    overlap_value = _clamp_non_negative(overlap_seconds)

    if chunk_value <= 0.0:
        results = _transcribe_full_paths(
            paths,
            model_name=model_name,
            language=language,
            task=task,
            decode_options=decode_options,
        )
        return [res.model_dump() for res in results]

    chunk_fn = _resolve_chunk_transcriber()
    effective_overlap = min(overlap_value, chunk_value / 2.0)
    results = chunk_fn(
        paths,
        model_name=model_name,
        language=language,
        task=task,
        chunk_seconds=chunk_value,
        overlap_seconds=effective_overlap,
        **decode_options,
    )
    return [res.model_dump() for res in results]


def _ensure_worker() -> _WorkerHandle:
    global _WORKER_HANDLE
    with _WORKER_LOCK:
        if _WORKER_HANDLE is None or not _WORKER_HANDLE.is_alive():
            _shutdown_worker_locked(reason="restart")
            _WORKER_HANDLE = _WorkerHandle()
        return _WORKER_HANDLE


def _shutdown_worker_locked(*, reason: str = "unspecified") -> None:
    global _WORKER_HANDLE
    global _SHUTDOWN_TIMER
    if _SHUTDOWN_TIMER is not None:
        _SHUTDOWN_TIMER.cancel()
        _SHUTDOWN_TIMER = None
    if _WORKER_HANDLE is not None:
        pid = _WORKER_HANDLE.pid
        try:
            _WORKER_HANDLE.shutdown()
        except Exception:
            logger.warning("ASRサブプロセスの停止処理で例外: pid=%s 理由=%s", pid or "unknown", reason, exc_info=True)
        logger.debug("ASRサブプロセスを停止: pid=%s 理由=%s", pid or "unknown", reason)
        _WORKER_HANDLE = None


def _schedule_idle_shutdown() -> None:
    if _IDLE_TIMEOUT <= 0:
        return

    def _timeout_shutdown() -> None:
        with _WORKER_LOCK:
            _shutdown_worker_locked(reason="idle_timeout")

    global _SHUTDOWN_TIMER
    if _SHUTDOWN_TIMER is not None:
        _SHUTDOWN_TIMER.cancel()
    timer = threading.Timer(_IDLE_TIMEOUT, _timeout_shutdown)
    timer.daemon = True
    _SHUTDOWN_TIMER = timer
    timer.start()


def transcribe_paths_via_worker(
    audio_paths: Iterable[str | Path],
    *,
    model_name: str,
    language: str | None,
    task: str | None,
    chunk_seconds: float | None = None,
    overlap_seconds: float | None = None,
    **decode_options: Any,
) -> list[TranscriptionResult]:
    payload = {
        "cmd": "transcribe_paths",
        "paths": [str(Path(p)) for p in audio_paths],
        "model_name": model_name,
        "language": language,
        "task": task,
        "chunk_seconds": chunk_seconds,
        "overlap_seconds": overlap_seconds,
        "decode_options": dict(decode_options or {}),
    }
    attempts = 0
    while attempts < 2:
        worker = _ensure_worker()
        try:
            response = worker.request(payload, timeout=_REQUEST_TIMEOUT)
        except (TimeoutError, BrokenPipeError, EOFError, OSError):
            attempts += 1
            with _WORKER_LOCK:
                _shutdown_worker_locked(reason="request_failure")
            if attempts >= 2:
                raise
            time.sleep(0.1)
            continue

        if not response.get("ok"):
            raise RuntimeError(response.get("error") or "ASR worker error")

        _schedule_idle_shutdown()
        return [TranscriptionResult.model_validate(entry) for entry in response.get("results") or []]

    raise RuntimeError("ASR worker unavailable")


def _resolve_chunk_transcriber() -> Callable[..., list[TranscriptionResult]]:
    global _CHUNK_TRANSCRIBE
    if _CHUNK_TRANSCRIBE is None:
        from src.lib.asr.chunking import transcribe_paths_chunked as _chunk_fn  # noqa: WPS433

        _CHUNK_TRANSCRIBE = _chunk_fn
    return _CHUNK_TRANSCRIBE


def _set_chunk_transcriber(fn: Callable[..., list[TranscriptionResult]] | None) -> None:
    global _CHUNK_TRANSCRIBE
    _CHUNK_TRANSCRIBE = fn


def _transcribe_full_paths(
    paths: Iterable[Path],
    *,
    model_name: str,
    language: str | None,
    task: str | None,
    decode_options: dict[str, Any],
) -> list[TranscriptionResult]:
    from src.lib.asr.options import TranscribeOptions  # noqa: WPS433 - worker内のみ import
    from src.lib.asr.pipeline import transcribe_paths  # noqa: WPS433

    options = TranscribeOptions(
        model_name=model_name,
        language=language,
        task=task,
        decode_options=dict(decode_options),
    )
    return transcribe_paths(paths, options=options)


def _clamp_non_negative(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(numeric):
        return 0.0
    return max(0.0, numeric)


__all__ = ["transcribe_paths_via_worker"]
