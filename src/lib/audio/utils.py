from __future__ import annotations

import logging

import numpy as np


class AudioDecodeError(Exception):
    """Raised when audio bytes cannot be decoded into a waveform."""

    def __init__(self, kind: str, detail: str | None = None):
        super().__init__(kind)
        self.kind = kind
        self.detail = detail


logger = logging.getLogger(__name__)


def decode_pcm_s16le_bytes(
    pcm_bytes: bytes,
    *,
    sample_rate: int,
    target_sample_rate: int | None = None,
) -> np.ndarray:
    """Decode raw PCM s16le mono into a normalized float32 waveform (-1..1).

    Resamples when target_sample_rate is provided.
    """

    if not pcm_bytes:
        raise AudioDecodeError("empty-input")
    if len(pcm_bytes) % 2 != 0:
        raise AudioDecodeError("invalid-length", f"received_bytes={len(pcm_bytes)}")

    logger.debug(
        "decode_pcm_s16le_bytes: received_bytes=%d sample_rate=%d",
        len(pcm_bytes),
        sample_rate,
    )

    waveform = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
    if waveform.size == 0:
        raise AudioDecodeError("empty-output")
    waveform = waveform / 32768.0

    if target_sample_rate is not None and target_sample_rate != sample_rate:
        waveform = _resample_waveform(
            waveform,
            source_rate=sample_rate,
            target_rate=target_sample_rate,
        )

    return waveform


def _resample_waveform(waveform: np.ndarray, *, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate <= 0 or target_rate <= 0:
        raise AudioDecodeError("invalid-sample-rate")
    if waveform.size == 0 or source_rate == target_rate:
        return waveform

    target_length = max(1, int(round(waveform.size * target_rate / source_rate)))
    logger.debug(
        "resample_waveform: source_rate=%d target_rate=%d samples=%d->%d",
        source_rate,
        target_rate,
        waveform.size,
        target_length,
    )
    old_indices = np.arange(waveform.size, dtype=np.float32)
    new_indices = np.linspace(0, waveform.size - 1, num=target_length, dtype=np.float32)
    return np.interp(new_indices, old_indices, waveform).astype(np.float32)


__all__ = [
    "AudioDecodeError",
    "decode_pcm_s16le_bytes",
]
