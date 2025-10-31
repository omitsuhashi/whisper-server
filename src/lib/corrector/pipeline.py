from __future__ import annotations

from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Iterable, Iterator, Sequence

from .editor import BaseEditor, GlossaryEditor
from .tagger.model import BaseTagger, RuleBasedPunctTagger
from .types import (
    CorrectionOptions,
    CorrectionPatch,
    CorrectionRequest,
    CorrectionResult,
    CorrectionSpan,
    apply_patches,
)

_PUNCTUATION_CHARS = {"。", "．", "!", "！", "?", "？", "…", "⋯", "、", "，"}


@dataclass
class CorrectionPipeline:
    default_options: CorrectionOptions = field(default_factory=CorrectionOptions)
    tagger: BaseTagger = field(default_factory=RuleBasedPunctTagger)
    editor: BaseEditor | None = field(default_factory=GlossaryEditor)

    def run(self, request: CorrectionRequest) -> CorrectionResult:
        options = CorrectionOptions(
            aggressive_kuten=request.options.aggressive_kuten
            or self.default_options.aggressive_kuten,
            normalize_numbers=request.options.normalize_numbers
            or self.default_options.normalize_numbers,
            enable_editor=request.options.enable_editor or self.default_options.enable_editor,
        )
        resolved_request = CorrectionRequest(
            text=request.text,
            context_prev=request.context_prev,
            language=request.language,
            options=options,
        )
        original_text = request.text
        tagger_patches = tuple(self.tagger.predict(resolved_request))
        intermediate = apply_patches(original_text, tagger_patches)

        final_text = intermediate
        if self.editor and options.enable_editor:
            editor_request = CorrectionRequest(
                text=intermediate,
                context_prev=request.context_prev,
                language=request.language,
                options=options,
            )
            final_text = self.editor.edit(editor_request)

        corrected = final_text
        patches = self._build_patches(original_text, corrected)
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

    def _build_patches(self, source: str, target: str) -> list[CorrectionPatch]:
        if source == target:
            return []
        matcher = SequenceMatcher(a=source, b=target)
        patches: list[CorrectionPatch] = []
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                continue
            if tag == "replace":
                start, end = i1, i2
                replacement = target[j1:j2]
            elif tag == "delete":
                start, end = i1, i2
                replacement = ""
            elif tag == "insert":
                start = end = i1
                replacement = target[j1:j2]
            else:  # pragma: no cover
                continue
            patch_tags = self._infer_tags(replacement)
            patches.append(
                CorrectionPatch(
                    span=CorrectionSpan(start=start, end=end),
                    replacement=replacement,
                    tags=patch_tags,
                    confidence=self._confidence_for_tags(patch_tags),
                )
            )
        return patches

    def _infer_tags(self, replacement: str) -> tuple[str, ...]:
        if not replacement:
            return ("DELETE",)
        if all(ch in _PUNCTUATION_CHARS for ch in replacement):
            return ("PUNCT",)
        return ("KANJI",)

    def _confidence_for_tags(self, tags: tuple[str, ...]) -> float:
        if "PUNCT" in tags:
            return 0.55
        if "KANJI" in tags:
            return 0.65
        return 0.5

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
