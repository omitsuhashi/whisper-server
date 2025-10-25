from __future__ import annotations

from typing import Iterable, List, Optional, Tuple

from pydantic import BaseModel, Field

from .models import PolishedDocument, PolishedSentence
from .options import PolishOptions
from .preprocess import (
    apply_terms,
    compile_protected,
    limit_repeats,
    normalize_text,
    protect_tokens,
    remove_fillers,
    unprotect_tokens,
)
from .segmentation import split_into_sentences
from .storage import save_json_doc, save_srt, save_txt
from .timestamps import reassign_timestamps

try:
    from ..asr import TranscriptionResult, TranscriptionSegment  # type: ignore
except Exception:  # pragma: no cover - fallback for import不備時
    class TranscriptionSegment(BaseModel):  # type: ignore
        start: float = 0.0
        end: float = 0.0
        text: str = ""

    class TranscriptionResult(BaseModel):  # type: ignore
        filename: str = "unknown"
        text: str = ""
        language: str | None = None
        duration: float | None = None
        segments: List[TranscriptionSegment] = Field(default_factory=list)


def polish_text_from_segments(
    segments: Iterable[TranscriptionSegment],
    *,
    options: Optional[PolishOptions] = None,
) -> List[PolishedSentence]:
    """ASRセグメント列から校正済みの文列を作成。"""

    opt = options or PolishOptions()

    cleaned: List[TranscriptionSegment] = []
    for segment in segments:
        text = normalize_text(segment.text or "", opt)
        text = remove_fillers(text, opt)
        if opt.remove_repeated_chars:
            text = limit_repeats(text, opt.max_char_repeat)
        cleaned.append(
            TranscriptionSegment.model_validate(
                {
                    "start": float(segment.start or 0.0),
                    "end": float(segment.end or 0.0),
                    "text": text,
                }
            )
        )

    joined = " ".join(seg.text for seg in cleaned).strip()
    protected_regexes = compile_protected(opt.protected_token_patterns)
    masked, mapping = protect_tokens(joined, protected_regexes)
    masked = apply_terms(masked, opt.term_pairs, regex=opt.term_pairs_regex)

    sentences: List[str] = split_into_sentences(masked, opt)
    sentences = [unprotect_tokens(sentence, mapping) for sentence in sentences]

    if opt.style == "常体":
        sentences = [s.replace("です。", "だ。").replace("ます。", "る。") for s in sentences]

    return reassign_timestamps(sentences, cleaned)


def polish_result(
    result: TranscriptionResult,
    *,
    options: Optional[PolishOptions] = None,
) -> PolishedDocument:
    sentences = polish_text_from_segments(result.segments, options=options)
    return PolishedDocument(filename=result.filename, sentences=sentences)


__all__ = [
    "PolishOptions",
    "PolishedDocument",
    "PolishedSentence",
    "polish_result",
    "polish_text_from_segments",
    "save_json_doc",
    "save_srt",
    "save_txt",
]
