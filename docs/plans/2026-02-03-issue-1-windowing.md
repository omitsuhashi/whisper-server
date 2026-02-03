# Issue-1 Windowing Helper Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a shared waveform windowing helper with stable input/output semantics and unit tests.

**Architecture:** Introduce `src/lib/asr/windowing.py` with a small dataclass and a pure slicing function. Tests validate default/full-window behavior, clamping, and error handling without touching CLI/HTTP code yet.

**Tech Stack:** Python 3.13, numpy, unittest.

### Task 1: Add failing tests for window slicing

**Files:**
- Create: `tests/test_asr_windowing.py`

**Step 1: Write the failing test**

```python
import unittest

import numpy as np

from src.lib.asr.windowing import slice_waveform_by_seconds


class TestWindowing(unittest.TestCase):
    def test_slice_defaults_to_full_waveform(self) -> None:
        wave = np.arange(16000 * 3, dtype=np.float32)
        result = slice_waveform_by_seconds(
            wave,
            sample_rate=16000,
            start_seconds=None,
            end_seconds=None,
        )
        self.assertEqual(result.start_seconds, 0.0)
        self.assertAlmostEqual(result.end_seconds, 3.0, places=6)
        self.assertAlmostEqual(result.total_seconds, 3.0, places=6)
        self.assertEqual(len(result.waveform), len(wave))
        np.testing.assert_array_equal(result.waveform, wave)

    def test_slice_clamps_out_of_range(self) -> None:
        wave = np.arange(30, dtype=np.float32)
        result = slice_waveform_by_seconds(
            wave,
            sample_rate=10,
            start_seconds=-1.0,
            end_seconds=10.0,
        )
        self.assertEqual(result.start_seconds, 0.0)
        self.assertEqual(result.end_seconds, 3.0)
        self.assertEqual(len(result.waveform), len(wave))

    def test_slice_empty_when_end_before_start(self) -> None:
        wave = np.arange(30, dtype=np.float32)
        result = slice_waveform_by_seconds(
            wave,
            sample_rate=10,
            start_seconds=2.5,
            end_seconds=1.0,
        )
        self.assertEqual(result.start_seconds, 2.5)
        self.assertEqual(result.end_seconds, 2.5)
        self.assertEqual(len(result.waveform), 0)

    def test_invalid_sample_rate_raises(self) -> None:
        wave = np.arange(10, dtype=np.float32)
        with self.assertRaises(ValueError):
            slice_waveform_by_seconds(
                wave,
                sample_rate=0,
                start_seconds=None,
                end_seconds=None,
            )
```

**Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_asr_windowing -v`

Expected: FAIL with `ModuleNotFoundError` for `src.lib.asr.windowing`.

### Task 2: Implement windowing helper

**Files:**
- Create: `src/lib/asr/windowing.py`
- Test: `tests/test_asr_windowing.py`

**Step 1: Write minimal implementation**

```python
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class WindowedWaveform:
    waveform: np.ndarray
    start_seconds: float
    end_seconds: float
    total_seconds: float


def _finite_or(value: Optional[float], fallback: float) -> float:
    if value is None:
        return fallback
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return fallback
    if not math.isfinite(parsed):
        return fallback
    return parsed


def slice_waveform_by_seconds(
    waveform: np.ndarray,
    *,
    sample_rate: int,
    start_seconds: Optional[float],
    end_seconds: Optional[float],
) -> WindowedWaveform:
    if sample_rate <= 0:
        raise ValueError("sample_rate must be > 0")

    total_samples = int(waveform.shape[-1]) if hasattr(waveform, "shape") else 0
    total_seconds = float(total_samples) / float(sample_rate) if total_samples > 0 else 0.0

    raw_start = _finite_or(start_seconds, 0.0)
    raw_end = _finite_or(end_seconds, total_seconds)

    start = max(0.0, min(raw_start, total_seconds))
    end = max(0.0, min(raw_end, total_seconds))
    if end < start:
        end = start

    start_sample = int(round(start * float(sample_rate)))
    end_sample = int(round(end * float(sample_rate)))
    start_sample = max(0, min(start_sample, total_samples))
    end_sample = max(start_sample, min(end_sample, total_samples))

    sliced = waveform[start_sample:end_sample]
    return WindowedWaveform(
        waveform=sliced,
        start_seconds=start,
        end_seconds=end,
        total_seconds=total_seconds,
    )


__all__ = ["WindowedWaveform", "slice_waveform_by_seconds"]
```

**Step 2: Run test to verify it passes**

Run: `python -m unittest tests.test_asr_windowing -v`

Expected: PASS.

**Step 3: Commit**

```bash
git add tests/test_asr_windowing.py src/lib/asr/windowing.py
git commit -m "✨ add windowing helper for waveform slicing"
```
