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
