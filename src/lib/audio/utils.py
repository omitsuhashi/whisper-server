from __future__ import annotations

from subprocess import CalledProcessError, run

import numpy as np


class AudioDecodeError(Exception):
    """Raised when audio bytes cannot be decoded into a waveform."""

    def __init__(self, kind: str, detail: str | None = None):
        super().__init__(kind)
        self.kind = kind
        self.detail = detail


def coerce_to_bytes(blob: bytes | bytearray | memoryview) -> bytes:
    """Normalize supported buffer-like inputs into raw bytes."""

    if isinstance(blob, bytes):
        return blob
    if isinstance(blob, bytearray):
        return bytes(blob)
    return bytes(memoryview(blob))


def decode_audio_bytes(audio_bytes: bytes, *, sample_rate: int) -> np.ndarray:
    """Decode arbitrary audio bytes into a normalized float32 mono waveform."""

    if not audio_bytes:
        raise AudioDecodeError("empty-input")

    cmd = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        "pipe:0",
        "-threads",
        "0",
        "-f",
        "s16le",
        "-ac",
        "1",
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(sample_rate),
        "-",
    ]
    try:
        completed = run(cmd, input=audio_bytes, capture_output=True, check=True)
    except FileNotFoundError as exc:  # pragma: no cover
        raise AudioDecodeError("ffmpeg-not-found") from exc
    except CalledProcessError as exc:
        stderr = exc.stderr.decode(errors="ignore") if exc.stderr else "unknown error"
        raise AudioDecodeError("ffmpeg-error", stderr) from exc

    waveform = np.frombuffer(completed.stdout, dtype=np.int16).astype(np.float32)
    if waveform.size == 0:
        raise AudioDecodeError("empty-output")
    return waveform / 32768.0


__all__ = [
    "AudioDecodeError",
    "coerce_to_bytes",
    "decode_audio_bytes",
]
