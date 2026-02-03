# CLI Streaming Windowed ASR Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** stream-interval で固定 window の waveform を ASR し、SegmentCommitter で確定領域のみ追記する。

**Architecture:** PCM ring buffer を読み取り、`transcribe_waveform` で window を書き起こす。`SegmentCommitter` が window 内の segment を絶対時刻へ補正して確定し、stdout には増分のみを append する。EOF では `build_result()` で全文を返す。

**Tech Stack:** Python 3.13, numpy, unittest

### Task 1: streaming の挙動テスト追加

**Files:**
- Modify: `tests/test_cli_streaming_window.py`

**Step 1: Write the failing test**

```python
class CliStreamingWindowTests(unittest.TestCase):
    def test_streaming_transcription_commits_segments(self):
        # SegmentCommitter が append-only で確定すること
        ...

    def test_streaming_transcription_sets_condition_default(self):
        # condition_on_previous_text=False が既定で入ること
        ...
```

**Step 2: Run test to verify it fails**

Run: `python3 -m unittest tests/test_cli_streaming_window.py -q`
Expected: FAIL (windowed ASR 未対応)

**Step 3: Write minimal implementation**

```python
# _streaming_transcription 内で
# - TranscribeOptions.decode_options に condition_on_previous_text=False を setdefault
# - transcribe_waveform + SegmentCommitter.update を使用
# - EOF で committer.build_result を返す
```

**Step 4: Run test to verify it passes**

Run: `python3 -m unittest tests/test_cli_streaming_window.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add src/cmd/cli.py tests/test_cli_streaming_window.py
git commit -m "✨ stream: add windowed asr committer"
```
