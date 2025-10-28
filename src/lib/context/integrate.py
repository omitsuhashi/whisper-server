from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

from pydantic import BaseModel, Field

from src.lib.asr import TranscriptionResult, TranscriptionSegment
from src.lib.diarize import DiarizationResult, SpeakerAnnotatedTranscript
from src.lib.video import sample_key_frames
from .image_analysis import ImageContext, analyze_frame_bgr


class IntegratedUtterance(BaseModel):
    start: float
    end: float
    text: str
    speaker: str | None = None
    image_context_index: int | None = None


class IntegratedMeetingContext(BaseModel):
    filename: str
    speakers: List[str] = Field(default_factory=list)
    image_contexts: List[ImageContext] = Field(default_factory=list)
    utterances: List[IntegratedUtterance] = Field(default_factory=list)


def build_image_contexts_from_video(
    video_path: str,
    *,
    min_scene_span: float = 1.0,
    diff_threshold: float = 0.3,
    max_frames: Optional[int] = None,
) -> List[ImageContext]:
    """動画から抽出した代表フレームを解析して ImageContext に変換する。"""

    frames = sample_key_frames(
        video_path,
        min_scene_span=float(min_scene_span),
        diff_threshold=float(diff_threshold),
        max_frames=max_frames if max_frames is not None else None,
    )
    contexts: List[ImageContext] = []
    for f in frames:
        contexts.append(analyze_frame_bgr(f.image, frame_index=f.index, timestamp=f.timestamp))
    return contexts


def _match_context_index(image_contexts: Sequence[ImageContext], t: float) -> int | None:
    if not image_contexts:
        return None
    # 近傍の timestamp を線形探索（件数は通常少数の想定）。
    best_i = None
    best_dist = 10**9
    for i, ctx in enumerate(image_contexts):
        d = abs(ctx.timestamp - t)
        if d < best_dist:
            best_dist = d
            best_i = i
    return best_i


def integrate_meeting_data(
    *,
    transcript: TranscriptionResult,
    speaker_transcript: SpeakerAnnotatedTranscript | None,
    diarization: DiarizationResult | None,
    image_contexts: Sequence[ImageContext] | None,
) -> IntegratedMeetingContext:
    """ASR/話者/画面の3要素を統合してタイムラインを作る。"""

    image_contexts = list(image_contexts or [])
    speakers: List[str] = []
    utterances: List[IntegratedUtterance] = []

    if speaker_transcript is not None:
        speakers = speaker_transcript.speakers
        for seg in speaker_transcript.segments:
            mid = (seg.start + seg.end) / 2.0
            idx = _match_context_index(image_contexts, mid)
            utterances.append(
                IntegratedUtterance(
                    start=seg.start,
                    end=seg.end,
                    text=seg.text,
                    speaker=seg.speaker,
                    image_context_index=idx,
                )
            )
    else:
        speakers = diarization.speakers if diarization is not None else []
        for seg in transcript.segments or []:
            mid = (seg.start + seg.end) / 2.0
            idx = _match_context_index(image_contexts, mid)
            utterances.append(
                IntegratedUtterance(
                    start=seg.start,
                    end=seg.end,
                    text=seg.text,
                    speaker=None,
                    image_context_index=idx,
                )
            )

    return IntegratedMeetingContext(
        filename=transcript.filename,
        speakers=speakers,
        image_contexts=list(image_contexts),
        utterances=utterances,
    )


def _fmt_time(t: float) -> str:
    m = int(t // 60)
    s = int(t % 60)
    return f"{m:02d}:{s:02d}"


def render_mermaid_mindmap(ctx: IntegratedMeetingContext) -> str:
    """Mermaid mindmap を組み立てる（シンプル版）。"""

    lines: List[str] = []
    lines.append("mindmap")
    lines.append(f"  {ctx.filename}")

    # Speakers
    if ctx.speakers:
        lines.append("    Speakers")
        for sp in ctx.speakers:
            lines.append(f"      {sp}")

    # Screens (image contexts)
    if ctx.image_contexts:
        lines.append("    Screens")
        for i, ic in enumerate(ctx.image_contexts):
            title = ic.label or f"frame#{i}"
            lines.append(f"      {i:02d}: {title}")

    # Transcript timeline (grouped in 2-minute buckets forコンパクトさ)
    if ctx.utterances:
        lines.append("    Transcript")
        for utt in ctx.utterances[:100]:  # 出力肥大化を防ぐ
            stamp = f"[{_fmt_time(utt.start)}-{_fmt_time(utt.end)}]"
            sp = f"{utt.speaker}: " if utt.speaker else ""
            text = (utt.text or "").strip().replace("\n", " ")
            text = text[:80]
            # 可能ならスクリーンをぶら下げ
            if utt.image_context_index is not None and 0 <= utt.image_context_index < len(ctx.image_contexts):
                ic = ctx.image_contexts[utt.image_context_index]
                title = ic.label
                lines.append(f"      {stamp} {sp}{text}")
                lines.append(f"        screen: {title}")
            else:
                lines.append(f"      {stamp} {sp}{text}")

    return "\n".join(lines) + "\n"

