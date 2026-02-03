# CLI Streaming Runtime Tests Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** streaming runtime を純関数化し、ffmpeg/実 ASR に依存しないユニットテストを追加する。

**Architecture:** `_streaming_transcription` の読取ループを `run_streaming_loop` に抽出し、PCM reader と transcribe 関数を注入する。テストでは fake PCM/ fake transcribe を使い、window サイズと append-only を検証する。

**Tech Stack:** Python 3.13, numpy, unittest

### Task 1: runtime テスト追加

**Files:**
- Create: `tests/test_cli_streaming_runtime.py`

**Step 1: Write the failing test**

```python
class TestStreamingRuntime(unittest.TestCase):
    def test_run_streaming_loop_limits_window(self):
        ...

    def test_run_streaming_loop_emits_incremental_text(self):
        ...
```

**Step 2: Run test to verify it fails**

Run: `python3 -m unittest tests/test_cli_streaming_runtime.py -q`
Expected: FAIL (module not found)

**Step 3: Write minimal implementation**

```python
def run_streaming_loop(...):
    # read -> ring.append -> interval flush
```

**Step 4: Run test to verify it passes**

Run: `python3 -m unittest tests/test_cli_streaming_runtime.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add src/lib/asr/streaming_runtime.py tests/test_cli_streaming_runtime.py src/cmd/cli.py
git commit -m "✨ stream: add runtime loop for tests"
```
