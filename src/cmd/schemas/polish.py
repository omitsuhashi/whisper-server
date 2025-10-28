from __future__ import annotations

from typing import Optional, Sequence

from pydantic import BaseModel, Field, model_validator

from src.lib.asr.models import TranscriptionSegment
from src.lib.polish import PolishOptions


class PolishTermPairPayload(BaseModel):
    src: str = Field(..., min_length=1, description="置換対象の文字列")
    dst: str = Field(..., description="置換後の文字列")


class PolishOptionsPayload(BaseModel):
    style: Optional[str] = Field(None, description="文体指定（例: ですます, 常体）")
    use_ginza: Optional[bool] = Field(None, description="GiNZA による文分割を利用するか")
    ginza_model: Optional[str] = Field(None, description="利用する GiNZA モデル名")
    ginza_fallback_to_heuristics: Optional[bool] = Field(
        None, description="GiNZA 失敗時にヒューリスティクスへフォールバックするか"
    )
    remove_fillers: Optional[bool] = Field(None, description="フィラーワードを除去するか")
    filler_patterns: Optional[Sequence[str]] = Field(None, description="フィラーワードとして扱う正規表現の配列")
    normalize_width: Optional[bool] = Field(None, description="NFKC 正規化を適用するか")
    space_collapse: Optional[bool] = Field(None, description="余分なスペースを圧縮するか")
    remove_repeated_chars: Optional[bool] = Field(None, description="連続文字の上限を設けるか")
    max_char_repeat: Optional[int] = Field(None, ge=1, description="連続文字の最大回数")
    term_pairs: Optional[Sequence[PolishTermPairPayload]] = Field(None, description="用語置換の対応表")
    term_pairs_regex: Optional[bool] = Field(None, description="用語置換の左辺を正規表現として扱うか")
    protected_token_patterns: Optional[Sequence[str]] = Field(None, description="保護対象トークンの正規表現集合")
    period_heuristics: Optional[Sequence[str]] = Field(None, description="ヒューリスティクスで終端とみなすパターン")
    min_sentence_len: Optional[int] = Field(None, ge=1, description="ヒューリスティクスで結合する最小文長")
    max_sentence_len: Optional[int] = Field(None, ge=1, description="ヒューリスティクスで強制分割する最大文長")


class PolishSegmentPayload(BaseModel):
    start: float = Field(..., ge=0.0, description="区間開始秒")
    end: float = Field(..., ge=0.0, description="区間終了秒")
    text: str = Field(..., min_length=1, description="書き起こしテキスト")

    @model_validator(mode="after")
    def validate_range(self) -> "PolishSegmentPayload":
        if self.end < self.start:
            raise ValueError("end must be greater than or equal to start")
        return self

    def to_segment(self) -> TranscriptionSegment:
        return TranscriptionSegment.model_validate({"start": self.start, "end": self.end, "text": self.text})


class PolishRequestPayload(BaseModel):
    segments: Sequence[PolishSegmentPayload] = Field(..., description="校正対象のセグメント列")
    options: Optional[PolishOptionsPayload] = Field(None, description="校正オプション")

    @model_validator(mode="after")
    def validate_segments(self) -> "PolishRequestPayload":
        if not self.segments:
            raise ValueError("segments must not be empty")
        return self


class PolishedSentencePayload(BaseModel):
    start: float
    end: float
    text: str


class PolishResponsePayload(BaseModel):
    sentences: list[PolishedSentencePayload]
    text: str
    sentence_count: int


def build_polish_options(payload: PolishOptionsPayload | None) -> PolishOptions:
    if payload is None:
        return PolishOptions()

    data = payload.model_dump(exclude_unset=True)

    term_pairs = data.get("term_pairs")
    if term_pairs is not None:
        data["term_pairs"] = tuple((pair["src"], pair["dst"]) for pair in term_pairs)

    for key in ("filler_patterns", "protected_token_patterns", "period_heuristics"):
        value = data.get(key)
        if value is not None:
            data[key] = tuple(value)

    return PolishOptions(**data)


__all__ = [
    "PolishOptionsPayload",
    "PolishRequestPayload",
    "PolishResponsePayload",
    "PolishedSentencePayload",
    "PolishTermPairPayload",
    "PolishOptions",
    "build_polish_options",
]
