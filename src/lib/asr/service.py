from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from src.lib.audio import PreparedAudio

from . import TranscriptionResult

TranscribeAllFn = Callable[[Iterable[str | Path]], list[TranscriptionResult]]


def resolve_model_and_language(
    model: Optional[str],
    language: Optional[str],
    *,
    default_model: str,
    default_language: str,
) -> tuple[str, str]:
    return (model or default_model, language or default_language)


def transcribe_prepared_audios(
    prepared: Iterable[PreparedAudio],
    *,
    model_name: str,
    language: str,
    task: Optional[str] = None,
    transcribe_all_fn: Optional[TranscribeAllFn] = None,
    decode_options: Optional[dict[str, Any]] = None,
) -> list[TranscriptionResult]:
    entries = list(prepared)
    non_silent = [str(entry.path) for entry in entries if not entry.silent]
    results: list[TranscriptionResult] = []
    decode_kwargs = dict(decode_options or {})
    if non_silent:
        if transcribe_all_fn is None:
            from src.lib import asr as asr_module  # local import for patchability

            transcribe_all_fn = asr_module.transcribe_all
        results = transcribe_all_fn(
            non_silent,
            model_name=model_name,
            language=language,
            task=task,
            **decode_kwargs,
        )

    result_iter = iter(results)
    updated: list[TranscriptionResult] = []
    for entry in entries:
        if entry.silent:
            updated.append(
                TranscriptionResult(
                    filename=entry.display_name,
                    text="",
                    language=language,
                    duration=0.0,
                    segments=[],
                )
            )
            continue
        result = next(result_iter)
        if hasattr(result, "model_copy"):
            updated.append(result.model_copy(update={"filename": entry.display_name}))
        else:  # テスト用のダミーオブジェクト等
            setattr(result, "filename", entry.display_name)
            updated.append(result)
    return updated


__all__ = [
    "resolve_model_and_language",
    "transcribe_prepared_audios",
]
