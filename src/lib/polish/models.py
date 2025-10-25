from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class PolishedSentence(BaseModel):
    """校正後の“文”。SRT 等に使えるよう区間を保持。"""

    model_config = ConfigDict(extra="ignore")
    start: float
    end: float
    text: str


class PolishedDocument(BaseModel):
    """校正済みドキュメント全体。"""

    model_config = ConfigDict(extra="ignore")
    filename: str
    sentences: list[PolishedSentence] = Field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n".join(s.text for s in self.sentences)


__all__ = ["PolishedSentence", "PolishedDocument"]
