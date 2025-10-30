from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Iterator, Sequence

from .tagger.model import BaseTagger, RuleBasedPunctTagger
from .types import CorrectionOptions, CorrectionPatch, CorrectionRequest, CorrectionResult, apply_patches


@dataclass
class CorrectionPipeline:
    default_options: CorrectionOptions = field(default_factory=CorrectionOptions)
    tagger: BaseTagger = field(default_factory=RuleBasedPunctTagger)

    def run(self, request: CorrectionRequest) -> CorrectionResult:
        options = CorrectionOptions(
            aggressive_kuten=request.options.aggressive_kuten
            or self.default_options.aggressive_kuten,
            normalize_numbers=request.options.normalize_numbers
            or self.default_options.normalize_numbers,
        )
        resolved_request = CorrectionRequest(
            text=request.text,
            context_prev=request.context_prev,
            language=request.language,
            options=options,
        )
        patches: list[CorrectionPatch] = []
        patches.extend(self.tagger.predict(resolved_request))

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
