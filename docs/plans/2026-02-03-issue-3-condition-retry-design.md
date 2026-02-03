# Issue-3 Condition Retry Guard Test Design

## Design
This change adds a single regression test to lock in the expected behavior: when `condition_on_previous_text` is explicitly set to `False`, `_transcribe_single()` must not perform a retry. The test uses the existing mocking pattern in `tests/test_asr_conditioning.py` to return a repeated-text payload, calls `_transcribe_single()` with `transcribe_kwargs={"condition_on_previous_text": False}`, and asserts that `transcribe` is invoked exactly once and the returned text matches the first payload. No production code changes are required; the test simply documents the intended retry guard for streaming usage.
