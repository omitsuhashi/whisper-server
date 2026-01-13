from __future__ import annotations

import math
from typing import List, Sequence, Tuple

import numpy as np


def estimate_speech_probabilities(
    waveform: np.ndarray,
    *,
    sample_rate: int,
    frame_bounds: Sequence[Tuple[int, int]],
    hop_duration: float,
    model_name: str,
) -> List[float]:
    """Whisper系(Transformer)から得られる no_speech_prob を speech_prob(=1-no_speech) に変換してフレーム系列へ投影する。

    - mlx_whisper が無い/失敗時は空配列を返す（上位でエネルギーVADへフォールバックさせる）。
    """

    if not model_name or hop_duration <= 0.0:
        return []

    try:
        from mlx_whisper import transcribe
    except Exception:
        return []

    try:
        raw = transcribe(waveform, path_or_hf_repo=model_name)
    except Exception:
        return []

    segments = raw.get("segments") or []
    probs = np.zeros(len(frame_bounds), dtype=np.float32)
    if not isinstance(segments, list) or probs.size == 0:
        return probs.tolist()

    for seg in segments:
        if not isinstance(seg, dict):
            continue
        start = seg.get("start")
        end = seg.get("end")
        if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
            continue
        start_f = float(start)
        end_f = float(end)
        if not math.isfinite(start_f) or not math.isfinite(end_f) or end_f <= start_f:
            continue

        no_speech = seg.get("no_speech_prob")
        if isinstance(no_speech, (int, float)) and math.isfinite(float(no_speech)):
            p = 1.0 - float(no_speech)
        else:
            p = 1.0
        if not math.isfinite(p):
            continue
        p = max(0.0, min(1.0, p))

        start_idx = int(max(0, math.floor(start_f / hop_duration)))
        end_idx = int(min(int(probs.size), math.ceil(end_f / hop_duration)))
        if end_idx <= start_idx:
            continue
        probs[start_idx:end_idx] = np.maximum(probs[start_idx:end_idx], float(p))

    return probs.tolist()


__all__ = ["estimate_speech_probabilities"]
