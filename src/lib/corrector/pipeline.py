from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Iterator, Sequence

from .types import (
    CorrectionOptions,
    CorrectionPatch,
    CorrectionRequest,
    CorrectionResult,
    CorrectionSpan,
    apply_patches,
)

_SENTENCE_TERMINATORS = {"。", "．", "!", "！", "?", "？", "…", "⋯"}


@dataclass
class CorrectionPipeline:
    default_options: CorrectionOptions = field(default_factory=CorrectionOptions)

    def run(self, request: CorrectionRequest) -> CorrectionResult:
        options = CorrectionOptions(
            aggressive_kuten=request.options.aggressive_kuten
            or self.default_options.aggressive_kuten,
            normalize_numbers=request.options.normalize_numbers
            or self.default_options.normalize_numbers,
        )
        patches: list[CorrectionPatch] = []
        patches.extend(self._maybe_insert_sentence_terminator(request.text, options))

        corrected = apply_patches(request.text, patches)
        return CorrectionResult(
            source_text=request.text,
            corrected_text=corrected,
            patches=tuple(patches),
        )

    def run_streaming(
        self,
        chunks: Iterable[str],
        *,
        context_prev: Sequence[str] | None = None,
        language: str = "ja",
        options: CorrectionOptions | None = None,
    ) -> Iterator[CorrectionResult]:
        buffer: list[str] = []
        resolved_context = tuple(context_prev or ())
        for chunk in chunks:
            if not chunk:
                continue
            buffer.append(chunk)
            text = "".join(buffer)
            request = CorrectionRequest.from_raw(
                text,
                context_prev=resolved_context,
                language=language,
                options=options or self.default_options,
            )
            yield self.run(request)

    def _maybe_insert_sentence_terminator(
        self,
        text: str,
        options: CorrectionOptions,
    ) -> list[CorrectionPatch]:
        if not options.aggressive_kuten:
            return []

        stripped = text.rstrip()
        if not stripped:
            return []
        last_char = stripped[-1]
        if last_char in _SENTENCE_TERMINATORS:
            return []

        insertion_index = len(stripped)
        patch = CorrectionPatch(
            span=CorrectionSpan(start=insertion_index, end=insertion_index),
            replacement="。",
            tags=("PUNCT",),
            confidence=0.55,
        )
        return [patch]


def run_correction(
    text: str,
    *,
    context_prev: Sequence[str] | None = None,
    language: str = "ja",
    options: CorrectionOptions | None = None,
    pipeline: CorrectionPipeline | None = None,
) -> CorrectionResult:
    executor = pipeline or CorrectionPipeline()
    request = CorrectionRequest.from_raw(
        text,
        context_prev=context_prev,
        language=language,
        options=options or executor.default_options,
    )
    return executor.run(request)


__all__ = [
    "CorrectionPipeline",
    "run_correction",
]
