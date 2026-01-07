from __future__ import annotations

import logging
from subprocess import CalledProcessError, run
from io import BytesIO
import wave

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


logger = logging.getLogger(__name__)


def decode_audio_bytes(audio_bytes: bytes, *, sample_rate: int) -> np.ndarray:
    """Decode arbitrary audio bytes into a normalized float32 mono waveform."""

    if not audio_bytes:
        raise AudioDecodeError("empty-input")

    logger.info(
        "decode_audio_bytes: received_bytes=%d sample_rate=%d",
        len(audio_bytes),
        sample_rate,
    )

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
        logger.warning("decode_audio_bytes: ffmpeg failed: %s", stderr.strip())
        raise AudioDecodeError("ffmpeg-error", stderr) from exc

    waveform = np.frombuffer(completed.stdout, dtype=np.int16).astype(np.float32)
    if waveform.size == 0:
        raise AudioDecodeError("empty-output")
    return waveform / 32768.0


def decode_pcm_s16le_bytes(pcm_bytes: bytes, *, sample_rate: int) -> np.ndarray:
    """Decode raw PCM s16le mono into a normalized float32 waveform (-1..1)."""

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
    return waveform / 32768.0


def encode_waveform_to_wav_bytes(waveform: np.ndarray, *, sample_rate: int) -> bytes:
    """Encode mono float waveform (-1..1) to 16-bit PCM WAV bytes."""

    if waveform.ndim != 1:
        raise ValueError(f"encode_waveform_to_wav_bytes expects 1-D mono waveform, got {waveform.shape}")
    wf = waveform.astype(np.float32)
    wf = np.clip(wf, -1.0, 1.0)
    pcm16 = (wf * 32767.0).astype(np.int16)

    buf = BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(sample_rate))
        w.writeframes(pcm16.tobytes())
    return buf.getvalue()


__all__ = [
    "AudioDecodeError",
    "coerce_to_bytes",
    "decode_audio_bytes",
    "decode_pcm_s16le_bytes",
    "encode_waveform_to_wav_bytes",
]
