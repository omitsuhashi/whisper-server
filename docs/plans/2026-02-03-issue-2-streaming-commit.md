# Issue-2 Streaming Segment Committer Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a stateful committer that converts windowed ASR segments into appended, de-duplicated output with lookback handling.

**Architecture:** Implement `SegmentCommitter` in `src/lib/asr/streaming_commit.py`, keeping state for committed segments/text and returning only newly committed text per update. Tests validate cutoff behavior, absolute time conversion, minimal de-duplication, and `build_result` output.

**Tech Stack:** Python 3.13, pydantic models, unittest.

### Task 1: Add failing tests for SegmentCommitter

**Files:**
- Create: `tests/test_asr_streaming_commit.py`

**Step 1: Write the failing test**

```python
import unittest

from src.lib.asr.models import TranscriptionResult, TranscriptionSegment
from src.lib.asr.streaming_commit import SegmentCommitter


def make_result(segments: list[TranscriptionSegment]) -> TranscriptionResult:
    return TranscriptionResult(filename="pcm", text="", language="ja", segments=segments)


class TestStreamingCommitter(unittest.TestCase):
    def test_commit_respects_lookback(self) -> None:
        committer = SegmentCommitter(lookback_seconds=1.0)
        result = make_result(
            [
                TranscriptionSegment(start=8.0, end=9.0, text="A"),
                TranscriptionSegment(start=9.0, end=10.0, text="B"),
            ]
        )
        new_text = committer.update(
            result,
            window_start_seconds=0.0,
            now_total_seconds=10.0,
            final=False,
        )
        self.assertEqual(new_text, "A")
        self.assertEqual(len(committer.committed_segments), 1)

        new_text_final = committer.update(
            result,
            window_start_seconds=0.0,
            now_total_seconds=10.0,
            final=True,
        )
        self.assertEqual(new_text_final, "B")
        self.assertEqual(len(committer.committed_segments), 2)

    def test_commit_converts_to_absolute_time(self) -> None:
        committer = SegmentCommitter(lookback_seconds=0.0)
        result = make_result([TranscriptionSegment(start=0.0, end=1.0, text="X")])
        committer.update(
            result,
            window_start_seconds=5.0,
            now_total_seconds=6.0,
            final=True,
        )
        self.assertEqual(committer.committed_segments[0].start, 5.0)
        self.assertEqual(committer.committed_segments[0].end, 6.0)

    def test_commit_skips_duplicate_text_with_small_gap(self) -> None:
        committer = SegmentCommitter(lookback_seconds=0.0)
        first = make_result([TranscriptionSegment(start=0.0, end=1.0, text="Hi")])
        committer.update(
            first,
            window_start_seconds=0.0,
            now_total_seconds=1.0,
            final=True,
        )
        second = make_result([TranscriptionSegment(start=0.1, end=0.9, text="Hi")])
        new_text = committer.update(
            second,
            window_start_seconds=1.1,
            now_total_seconds=2.0,
            final=True,
        )
        self.assertEqual(new_text, "")
        self.assertEqual(len(committer.committed_segments), 1)
        self.assertAlmostEqual(committer.committed_until, 2.0, places=6)

    def test_build_result_uses_committed_text_and_duration(self) -> None:
        committer = SegmentCommitter(lookback_seconds=0.0)
        result = make_result([TranscriptionSegment(start=0.0, end=1.0, text="A")])
        committer.update(
            result,
            window_start_seconds=0.0,
            now_total_seconds=1.0,
            final=True,
        )
        built = committer.build_result(filename="pcm", language="ja")
        self.assertEqual(built.text, "A")
        self.assertEqual(built.duration, 1.0)
        self.assertEqual(len(built.segments), 1)
```

**Step 2: Run test to verify it fails**

Run: `python3 -m unittest tests.test_asr_streaming_commit -v`

Expected: FAIL with `ModuleNotFoundError` for `src.lib.asr.streaming_commit`.

### Task 2: Implement SegmentCommitter

**Files:**
- Create: `src/lib/asr/streaming_commit.py`
- Test: `tests/test_asr_streaming_commit.py`

**Step 1: Write minimal implementation**

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from .models import TranscriptionResult, TranscriptionSegment


@dataclass
class SegmentCommitter:
    lookback_seconds: float = 1.0
    committed_until: float = 0.0
    committed_segments: List[TranscriptionSegment] = field(default_factory=list)
    committed_text: str = ""

    def update(
        self,
        result: TranscriptionResult,
        *,
        window_start_seconds: float,
        now_total_seconds: float,
        final: bool = False,
    ) -> str:
        cutoff = float(now_total_seconds) if final else max(
            0.0, float(now_total_seconds) - float(self.lookback_seconds)
        )

        abs_segments: List[TranscriptionSegment] = []
        for seg in (result.segments or []):
            start = float(getattr(seg, "start", 0.0)) + float(window_start_seconds)
            end = float(getattr(seg, "end", start)) + float(window_start_seconds)
            text = str(getattr(seg, "text", "") or "")
            abs_segments.append(seg.model_copy(update={"start": start, "end": end, "text": text}))

        abs_segments.sort(key=lambda s: (float(s.end), float(s.start)))
        newly_committed: List[TranscriptionSegment] = []
        for seg in abs_segments:
            if float(seg.end) > cutoff:
                continue
            if float(seg.end) <= float(self.committed_until) + 1e-6:
                continue
            if self.committed_segments:
                last = self.committed_segments[-1]
                if (last.text or "").strip() == (seg.text or "").strip():
                    gap = float(seg.start) - float(last.end)
                    if gap <= 0.5:
                        self.committed_until = max(self.committed_until, float(seg.end))
                        continue

            self.committed_segments.append(seg)
            self.committed_until = max(self.committed_until, float(seg.end))
            newly_committed.append(seg)

        new_text = "".join(seg.text for seg in newly_committed if seg.text)
        if new_text:
            self.committed_text += new_text
        return new_text

    def build_result(self, *, filename: str, language: Optional[str]) -> TranscriptionResult:
        duration = self.committed_segments[-1].end if self.committed_segments else 0.0
        return TranscriptionResult(
            filename=filename,
            text=self.committed_text.strip(),
            language=language,
            duration=float(duration),
            segments=list(self.committed_segments),
        )


__all__ = ["SegmentCommitter"]
```

**Step 2: Run test to verify it passes**

Run: `python3 -m unittest tests.test_asr_streaming_commit -v`

Expected: PASS.

**Step 3: Commit**

```bash
git add tests/test_asr_streaming_commit.py src/lib/asr/streaming_commit.py
git commit -m "✨ add streaming segment committer"
```
