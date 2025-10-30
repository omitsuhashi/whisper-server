from __future__ import annotations

from typing import Optional, Sequence

from pydantic import BaseModel, Field, model_validator

from src.lib.corrector import CorrectionOptions, CorrectionPatch, CorrectionSpan


class CorrectionOptionsPayload(BaseModel):
    aggressive_kuten: Optional[bool] = Field(
        None,
        description="末尾句点補完ルールを有効化するかどうか",
    )
    normalize_numbers: Optional[bool] = Field(
        None,
        description="数値表記の簡易正規化を有効化するか（MVP占位）",
    )


class CorrectionSpanPayload(BaseModel):
    start: int = Field(..., ge=0, description="置換対象の開始インデックス（0-based）")
    end: int = Field(..., ge=0, description="置換対象の終了インデックス（0-based, open）")

    @model_validator(mode="after")
    def validate_range(self) -> "CorrectionSpanPayload":
        if self.end < self.start:
            raise ValueError("end must be greater than or equal to start")
        return self

    @classmethod
    def from_span(cls, span: CorrectionSpan) -> "CorrectionSpanPayload":
        return cls(start=span.start, end=span.end)


class CorrectionPatchPayload(BaseModel):
    span: CorrectionSpanPayload
    replacement: str = Field(..., description="差し替える文字列")
    tags: list[str] = Field(default_factory=list, description="付与タグの配列")
    confidence: float = Field(..., ge=0.0, le=1.0, description="確信度 (0-1)")

    @classmethod
    def from_patch(cls, patch: CorrectionPatch) -> "CorrectionPatchPayload":
        return cls(
            span=CorrectionSpanPayload.from_span(patch.span),
            replacement=patch.replacement,
            tags=list(patch.tags),
            confidence=float(patch.confidence),
        )


class CorrectionRequestPayload(BaseModel):
    text: str = Field(..., min_length=1, description="補正対象のテキスト")
    context_prev: Optional[Sequence[str]] = Field(
        None,
        description="直前の確定文コンテキスト。最新を末尾とする",
    )
    language: Optional[str] = Field(None, description="言語コード。未指定時は ja")
    options: Optional[CorrectionOptionsPayload] = Field(None, description="補正オプション")

    @model_validator(mode="after")
    def normalize_context(self) -> "CorrectionRequestPayload":
        if self.context_prev is not None:
            self.context_prev = tuple(item for item in self.context_prev if item is not None)
        return self

    def context_tuple(self) -> tuple[str, ...]:
        if not self.context_prev:
            return ()
        return tuple(str(item) for item in self.context_prev if item)

    def language_or_default(self) -> str:
        return (self.language or "ja").strip() or "ja"


class CorrectionResponsePayload(BaseModel):
    source_text: str = Field(..., description="補正処理前のテキスト")
    text: str = Field(..., description="補正後のテキスト")
    patches: list[CorrectionPatchPayload] = Field(..., description="適用された差分パッチ")
    patch_count: int = Field(..., ge=0, description="パッチ数")
    options: dict[str, bool] = Field(..., description="適用したオプションの値")


def build_correction_options(payload: CorrectionOptionsPayload | None) -> CorrectionOptions:
    if payload is None:
        return CorrectionOptions()
    data = payload.model_dump(exclude_unset=True)
    return CorrectionOptions(**data)


__all__ = [
    "CorrectionOptionsPayload",
    "CorrectionSpanPayload",
    "CorrectionPatchPayload",
    "CorrectionRequestPayload",
    "CorrectionResponsePayload",
    "build_correction_options",
]
