from __future__ import annotations

from dataclasses import dataclass, field
from subprocess import PIPE, Popen
from typing import BinaryIO

import numpy as np

from src.lib.audio.utils import decode_pcm_s16le_bytes


@dataclass
class PcmRingBuffer:
    max_samples: int
    samples: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    total_samples: int = 0

    def append(self, chunk: np.ndarray) -> None:
        if chunk.size == 0:
            return
        self.total_samples += int(chunk.size)
        if self.samples.size:
            self.samples = np.concatenate([self.samples, chunk])
        else:
            self.samples = chunk
        if self.max_samples > 0 and self.samples.size > self.max_samples:
            self.samples = self.samples[-self.max_samples :]


class PcmStreamReader:
    def __init__(self, stream: BinaryIO, *, sample_rate: int, target_sample_rate: int | None) -> None:
        self._stream = stream
        self._leftover = b""
        self._sample_rate = sample_rate
        self._target_sample_rate = target_sample_rate

    def read_waveform(self, chunk_size: int) -> tuple[np.ndarray, bool]:
        chunk = self._stream.read(chunk_size)
        if not chunk:
            return np.zeros(0, dtype=np.float32), True

        if self._leftover:
            chunk = self._leftover + chunk
            self._leftover = b""

        if len(chunk) % 2 != 0:
            self._leftover = chunk[-1:]
            chunk = chunk[:-1]

        if not chunk:
            return np.zeros(0, dtype=np.float32), False

        waveform = decode_pcm_s16le_bytes(
            chunk,
            sample_rate=self._sample_rate,
            target_sample_rate=self._target_sample_rate,
        )
        return waveform, False


def open_ffmpeg_pcm_stream(stdin: BinaryIO, *, target_sample_rate: int) -> tuple[Popen[bytes], BinaryIO]:
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
        str(target_sample_rate),
        "-",
    ]
    proc = Popen(cmd, stdin=stdin, stdout=PIPE, stderr=PIPE)
    assert proc.stdout is not None
    return proc, proc.stdout


__all__ = ["PcmRingBuffer", "PcmStreamReader", "open_ffmpeg_pcm_stream"]
