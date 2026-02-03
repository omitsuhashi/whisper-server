# Issue-3 Condition Retry Guard Test Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a regression test ensuring no retry occurs when `condition_on_previous_text=False`.

**Architecture:** Extend `tests/test_asr_conditioning.py` with a new unit test under `ConditionFallbackTests` that asserts single-call behavior.

**Tech Stack:** Python 3.13, unittest, mock.

### Task 1: Add failing test for retry guard

**Files:**
- Modify: `tests/test_asr_conditioning.py`

**Step 1: Write the failing test**

```python
    @mock.patch("src.lib.asr.pipeline._memsnap")
    @mock.patch("src.lib.asr.pipeline.transcribe")
    def test_no_retry_when_condition_flag_false(self, mock_transcribe: mock.Mock, mock_memsnap: mock.Mock) -> None:
        loop_payload = {
            "text": "foo bar foo bar foo bar foo bar",
            "segments": [{"text": "foo bar"} for _ in range(6)],
            "language": "ja",
        }
        mock_transcribe.return_value = loop_payload

        result = _transcribe_single(
            audio_input=b"",
            display_name="loop.wav",
            model_name="demo",
            transcribe_kwargs={"condition_on_previous_text": False},
            language_hint="ja",
        )

        self.assertEqual(mock_transcribe.call_count, 1)
        self.assertEqual(result.text, loop_payload["text"])
```

**Step 2: Run test to verify it fails**

Run: `python3 -m unittest tests.test_asr_conditioning.ConditionFallbackTests.test_no_retry_when_condition_flag_false -v`

Expected: FAIL if retry still occurs (call_count > 1).

### Task 2: Confirm behavior passes (no code change expected)

**Files:**
- Test: `tests/test_asr_conditioning.py`

**Step 1: Re-run test to verify it passes**

Run: `python3 -m unittest tests.test_asr_conditioning.ConditionFallbackTests.test_no_retry_when_condition_flag_false -v`

Expected: PASS (call_count == 1).

**Step 2: Commit**

```bash
git add tests/test_asr_conditioning.py
git commit -m "✨ add test for no-retry when condition flag false"
```
