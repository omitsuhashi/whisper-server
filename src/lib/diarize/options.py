from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass
class DiarizeOptions:
    """
    Diarization 実行時のオプション。
    - model_name: 既定は pyannote/speaker-diarization-3.1
    - token_env:  トークンを読む環境変数（順に探索）
    - device:     "mps"（Noneでも自動で mps を選択）
    - require_mps: True 固定。MPS を前提とし、CPU にはフォールバックしない
    - *speakers:  人数の事前情報があると安定（num/min/max）
    """

    model_name: str = "pyannote/speaker-diarization-3.1"
    token: str | None = None
    token_env: Tuple[str, ...] = ("HF_TOKEN", "HUGGINGFACE_TOKEN", "PYANNOTE_TOKEN")
    device: str | None = None
    require_mps: bool = True
    num_speakers: int | None = None
    min_speakers: int | None = None
    max_speakers: int | None = None
    sample_rate: int = 16000  # waveform入力時の既定SR（pyannoteはSR指定可能）


__all__ = ["DiarizeOptions"]
