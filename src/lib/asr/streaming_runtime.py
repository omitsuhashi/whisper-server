from __future__ import annotations

from typing import Callable
import time

import numpy as np

from .models import TranscriptionResult
from .streaming_commit import SegmentCommitter
from .streaming_input import PcmRingBuffer

ReadWaveformFn = Callable[[int], tuple[np.ndarray, bool]]
TranscribeFn = Callable[[np.ndarray], TranscriptionResult]
EmitFn = Callable[[str], None]
FinalizeFn = Callable[[], TranscriptionResult]
TimeSourceFn = Callable[[], float]


def run_streaming_loop(
    *,
    read_waveform: ReadWaveformFn,
    chunk_size: int,
    interval: float,
    ring: PcmRingBuffer,
    committer: SegmentCommitter,
    transcribe_fn: TranscribeFn,
    emit_fn: EmitFn | None,
    finalize_fn: FinalizeFn,
    target_sample_rate: int,
    time_source: TimeSourceFn = time.monotonic,
) -> TranscriptionResult | None:
    last_flush = time_source()
    final_result: TranscriptionResult | None = None

    def flush(*, final: bool) -> None:
        nonlocal final_result
        if ring.samples.size == 0:
            return
        now_total_seconds = float(ring.total_samples) / float(target_sample_rate)
        window_start_seconds = max(
            0.0,
            now_total_seconds - (float(ring.samples.size) / float(target_sample_rate)),
        )
        result = transcribe_fn(ring.samples)
        new_text = committer.update(
            result,
            window_start_seconds=window_start_seconds,
            now_total_seconds=now_total_seconds,
            final=final,
        )
        if emit_fn and new_text:
            emit_fn(new_text)
        if final:
            final_result = finalize_fn()

    while True:
        waveform, eof = read_waveform(chunk_size)
        if eof:
            break
        if waveform.size > 0:
            ring.append(waveform)
        now = time_source()
        if interval <= 0 or (now - last_flush) >= interval:
            flush(final=False)
            last_flush = now

    flush(final=True)
    return final_result


__all__ = ["run_streaming_loop"]
