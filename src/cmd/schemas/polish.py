from __future__ import annotations

from typing import Any, Optional, Sequence

from pydantic import BaseModel, Field, model_validator

from src.lib.asr.models import TranscriptionSegment
from src.lib.polish import PolishOptions


class PolishTermPairPayload(BaseModel):
    src: str = Field(..., min_length=1, description="置換対象の文字列")
    dst: str = Field(..., description="置換後の文字列")


class PolishOptionsPayload(BaseModel):
    style: Optional[str] = Field(None, description="文体指定（例: ですます, 常体）")
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
    text: Optional[str] = Field(None, description="校正対象の全文。segments を省略した場合に利用")
    segments: Optional[Sequence[PolishSegmentPayload]] = Field(
        None,
        description="校正対象のセグメント列。指定が無い場合は text を使用",
    )
    options: Optional[PolishOptionsPayload] = Field(None, description="校正オプション")

    @model_validator(mode="after")
    def validate_source(self) -> "PolishRequestPayload":
        segments = list(self.segments) if self.segments else []
        text_value = (self.text or "").strip()

        if segments:
            self.segments = tuple(segments)
            return self

        if text_value:
            self.text = text_value
            self.segments = None
            return self

        raise ValueError("text または segments のいずれかを指定してください")

    def to_segments(self) -> Sequence[TranscriptionSegment]:
        if self.segments:
            return [item.to_segment() for item in self.segments]

        assert self.text is not None  # validate_source で保証
        return [
            TranscriptionSegment.model_validate(
                {
                    "start": 0.0,
                    "end": 0.0,
                    "text": self.text,
                }
            )
        ]


class PolishedSentencePayload(BaseModel):
    start: float
    end: float
    text: str


class PolishResponsePayload(BaseModel):
    sentences: list[PolishedSentencePayload]
    text: str
    sentence_count: int


class LLMPolishRequestPayload(PolishRequestPayload):
    style: Optional[str] = Field(None, description="LLM 側へ指示する文体。未指定時は options.style を利用")
    extra_instructions: Optional[str] = Field(None, description="モデルへ追加する日本語指示文")
    parameters: Optional[dict[str, Any]] = Field(None, description="mlx_lm.generate に渡す追加パラメーター")
    model_id: Optional[str] = Field(None, description="利用する mlx-lm モデル ID")
    temperature: Optional[float] = Field(None, ge=0.0, description="サンプリング温度")
    top_p: Optional[float] = Field(None, ge=0.0, le=1.0, description="トークン選択の top-p")
    max_tokens: Optional[int] = Field(None, ge=1, description="生成する最大トークン数")


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
    "LLMPolishRequestPayload",
]
