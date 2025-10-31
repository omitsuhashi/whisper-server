from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence, Tuple


class CorrectionError(ValueError):
    """補正処理に関する汎用例外。"""


class PatchValidationError(CorrectionError):
    """パッチが適用できない場合に発生する例外。"""


@dataclass(frozen=True)
class CorrectionSpan:
    start: int
    end: int

    def validate(self, text_length: int) -> None:
        if not (0 <= self.start <= self.end <= text_length):
            raise PatchValidationError(
                f"span out of range: start={self.start}, end={self.end}, length={text_length}"
            )


@dataclass(frozen=True)
class CorrectionOptions:
    aggressive_kuten: bool = False
    normalize_numbers: bool = False
    enable_editor: bool = False

    @classmethod
    def from_mapping(cls, data: Mapping[str, object] | None) -> "CorrectionOptions":
        if data is None:
            return cls()
        return cls(
            aggressive_kuten=bool(data.get("aggressive_kuten", False)),
            normalize_numbers=bool(data.get("normalize_numbers", False)),
            enable_editor=bool(data.get("enable_editor", False)),
        )

    def update(self, **overrides: object) -> "CorrectionOptions":
        return CorrectionOptions(
            aggressive_kuten=bool(overrides.get("aggressive_kuten", self.aggressive_kuten)),
            normalize_numbers=bool(overrides.get("normalize_numbers", self.normalize_numbers)),
            enable_editor=bool(overrides.get("enable_editor", self.enable_editor)),
        )

    def as_dict(self) -> dict[str, bool]:
        return {
            "aggressive_kuten": self.aggressive_kuten,
            "normalize_numbers": self.normalize_numbers,
            "enable_editor": self.enable_editor,
        }


@dataclass(frozen=True)
class CorrectionPatch:
    span: CorrectionSpan
    replacement: str
    tags: Tuple[str, ...] = field(default_factory=tuple)
    confidence: float = 0.0

    def with_span(self, *, start: int | None = None, end: int | None = None) -> "CorrectionPatch":
        span = CorrectionSpan(
            start=self.span.start if start is None else start,
            end=self.span.end if end is None else end,
        )
        return CorrectionPatch(
            span=span,
            replacement=self.replacement,
            tags=self.tags,
            confidence=self.confidence,
        )

    def add_tags(self, *extra: str) -> "CorrectionPatch":
        if not extra:
            return self
        merged = tuple(dict.fromkeys((*self.tags, *extra)))
        return CorrectionPatch(span=self.span, replacement=self.replacement, tags=merged, confidence=self.confidence)


@dataclass(frozen=True)
class CorrectionRequest:
    text: str
    context_prev: Tuple[str, ...] = field(default_factory=tuple)
    language: str = "ja"
    options: CorrectionOptions = field(default_factory=CorrectionOptions)

    @classmethod
    def from_raw(
        cls,
        text: str,
        *,
        context_prev: Sequence[str] | None = None,
        language: str = "ja",
        options: CorrectionOptions | Mapping[str, object] | None = None,
    ) -> "CorrectionRequest":
        contexts = tuple(context_prev or ())
        if isinstance(options, CorrectionOptions):
            opts = options
        else:
            opts = CorrectionOptions.from_mapping(options)
        return cls(text=text, context_prev=contexts, language=language, options=opts)

    @property
    def latest_context(self) -> str:
        if not self.context_prev:
            return ""
        return self.context_prev[-1]


@dataclass(frozen=True)
class CorrectionResult:
    source_text: str
    corrected_text: str
    patches: Tuple[CorrectionPatch, ...] = field(default_factory=tuple)

    def is_modified(self) -> bool:
        return bool(self.patches)


def apply_patches(text: str, patches: Sequence[CorrectionPatch]) -> str:
    if not patches:
        return text

    sorted_patches = sorted(patches, key=lambda patch: (patch.span.start, patch.span.end))
    fragments: list[str] = []
    cursor = 0
    text_length = len(text)

    for patch in sorted_patches:
        patch.span.validate(text_length)
        if patch.span.start < cursor:
            raise PatchValidationError(
                "patch spans overlap or are out of order: "
                f"prev_cursor={cursor}, start={patch.span.start}"
            )
        fragments.append(text[cursor : patch.span.start])
        fragments.append(patch.replacement)
        cursor = patch.span.end

    fragments.append(text[cursor:])
    return "".join(fragments)


__all__ = [
    "CorrectionError",
    "PatchValidationError",
    "CorrectionSpan",
    "CorrectionOptions",
    "CorrectionPatch",
    "CorrectionRequest",
    "CorrectionResult",
    "apply_patches",
]
