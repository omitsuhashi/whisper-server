from __future__ import annotations

from typing import Sequence, TYPE_CHECKING

from .pipeline import CorrectionPipeline
from .types import CorrectionOptions, CorrectionRequest, CorrectionResult

if TYPE_CHECKING:  # pragma: no cover
    from src.lib.asr import TranscriptionResult


def apply_corrections_to_results(
    results: Sequence["TranscriptionResult"],
    *,
    language_hint: str | None,
    options: CorrectionOptions,
    context_size: int = 2,
    pipeline: CorrectionPipeline | None = None,
) -> tuple[list["TranscriptionResult"], dict[str, CorrectionResult]]:
    if not results:
        return [], {}

    executor = pipeline or CorrectionPipeline(default_options=options)
    correction_map: dict[str, CorrectionResult] = {}
    updated: list["TranscriptionResult"] = []
    context_buffer: list[str] = []

    for result in results:
        context_prev = tuple(context_buffer[-context_size:]) if context_size > 0 else ()
        request = CorrectionRequest.from_raw(
            result.text,
            context_prev=context_prev,
            language=result.language or language_hint or "ja",
            options=options,
        )
        correction = executor.run(request)
        correction_map[result.filename] = correction
        if correction.is_modified():
            if hasattr(result, "model_copy"):
                updated_result = result.model_copy(update={"text": correction.corrected_text})
            else:
                setattr(result, "text", correction.corrected_text)
                updated_result = result
        else:
            updated_result = result

        updated.append(updated_result)
        context_buffer.append(correction.corrected_text)
        if context_size > 0 and len(context_buffer) > context_size:
            del context_buffer[:-context_size]

    return updated, correction_map


__all__ = ["apply_corrections_to_results"]
