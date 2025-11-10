from __future__ import annotations

from dataclasses import dataclass
import math
from typing import List, Sequence, Tuple

import numpy as np


@dataclass(frozen=True, slots=True)
class VadConfig:
    """VAD パラメータ一式。シンプルな短時間エネルギー方式を想定する。"""

    frame_duration: float = 0.03
    """1 フレームの長さ [sec]。"""

    energy_threshold: float | None = None
    """固定しきい値（指定しない場合は波形に応じて推定）。"""

    min_energy: float = 1e-4
    """自動しきい値計算時の下限。"""

    dynamic_threshold_ratio: float = 2.5
    """ノイズ床に対する乗数。"""

    noise_quantile: float = 0.2
    """ノイズ推定に使う分位点 (0-1)。"""

    min_speech_duration: float = 0.3
    """この長さ未満の区間は除去する。"""

    min_silence_duration: float = 0.2
    """この長さ以下の無音ギャップは結合する。"""

    padding_duration: float = 0.05
    """両端に与えるフェード的な余白。"""


@dataclass(frozen=True, slots=True)
class SpeechSegment:
    """VAD 後の時間区間。サンプル位置も保持しておく。"""

    start: float
    end: float
    start_sample: int
    end_sample: int

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


def detect_voice_segments(waveform: np.ndarray, sample_rate: int, *, config: VadConfig | None = None) -> List[SpeechSegment]:
    """波形から音声区間を抽出して返す。"""

    if sample_rate <= 0:
        raise ValueError("sample_rate は正の値で指定してください")

    cfg = config or VadConfig()
    wf = _ensure_mono_waveform(waveform)
    total_samples = wf.size
    if total_samples == 0:
        return []

    frame_size = max(int(cfg.frame_duration * sample_rate), 1)
    frame_bounds, energies = _calc_frame_energies(wf, frame_size)
    if not energies:
        return []

    threshold = _estimate_threshold(energies, cfg)
    voiced_flags = [energy >= threshold for energy in energies]
    raw_segments = _collect_segments(voiced_flags)
    if not raw_segments:
        return []

    index_segments = _frames_to_samples(raw_segments, frame_bounds, total_samples)

    min_gap_samples = _seconds_to_samples(cfg.min_silence_duration, sample_rate)
    merged_segments = _merge_short_gaps(index_segments, min_gap_samples)

    min_speech_samples = _seconds_to_samples(cfg.min_speech_duration, sample_rate)
    filtered_segments = [seg for seg in merged_segments if (seg[1] - seg[0]) >= min_speech_samples]
    if not filtered_segments:
        return []

    padding = _seconds_to_samples(cfg.padding_duration, sample_rate)
    padded_segments = [_apply_padding(seg, padding, total_samples) for seg in filtered_segments]

    return [
        SpeechSegment(
            start=start / sample_rate,
            end=end / sample_rate,
            start_sample=start,
            end_sample=end,
        )
        for start, end in padded_segments
        if end > start
    ]


def segment_waveform(
    waveform: np.ndarray,
    sample_rate: int,
    *,
    vad_config: VadConfig | None = None,
    max_segment_duration: float | None = None,
) -> List[SpeechSegment]:
    """VAD で抽出した区間をさらに最大長で分割して返す。"""

    segments = detect_voice_segments(waveform, sample_rate, config=vad_config)
    if not segments:
        return []
    if max_segment_duration is None or max_segment_duration <= 0:
        return segments

    max_samples = _seconds_to_samples(max_segment_duration, sample_rate)
    if max_samples <= 0:
        return segments

    sliced: List[SpeechSegment] = []
    for seg in segments:
        start = seg.start_sample
        while start < seg.end_sample:
            end = min(start + max_samples, seg.end_sample)
            sliced.append(
                SpeechSegment(
                    start=start / sample_rate,
                    end=end / sample_rate,
                    start_sample=start,
                    end_sample=end,
                )
            )
            start = end
    return sliced


def _ensure_mono_waveform(waveform: np.ndarray) -> np.ndarray:
    wf = np.asarray(waveform, dtype=np.float32)
    if wf.ndim == 0:
        return wf.reshape(1)
    if wf.ndim == 1:
        return wf
    return np.mean(wf, axis=-1)


def _calc_frame_energies(waveform: np.ndarray, frame_size: int) -> Tuple[List[Tuple[int, int]], List[float]]:
    bounds: List[Tuple[int, int]] = []
    energies: List[float] = []
    total = waveform.size
    for start in range(0, total, frame_size):
        end = min(total, start + frame_size)
        frame = waveform[start:end]
        if frame.size == 0:
            continue
        energy = float(np.mean(np.abs(frame)))
        bounds.append((start, end))
        energies.append(energy)
    return bounds, energies


def _estimate_threshold(energies: Sequence[float], cfg: VadConfig) -> float:
    if not energies:
        return cfg.min_energy
    if cfg.energy_threshold is not None and cfg.energy_threshold > 0:
        return cfg.energy_threshold
    quantile = np.clip(cfg.noise_quantile, 0.0, 1.0)
    noise_floor = float(np.quantile(energies, quantile))
    peak = max(float(max(energies)), cfg.min_energy)
    margin = max(peak - noise_floor, 0.0)
    ratio_based = max(noise_floor * cfg.dynamic_threshold_ratio, cfg.min_energy)
    spread_based = noise_floor + margin * 0.3
    candidate = min(ratio_based, spread_based if margin > 0 else peak)
    if not math.isfinite(candidate) or candidate <= 0:
        return cfg.min_energy
    return min(candidate, peak * 0.95)


def _collect_segments(flags: Sequence[bool]) -> List[Tuple[int, int]]:
    segments: List[Tuple[int, int]] = []
    start_idx: int | None = None
    for idx, flag in enumerate(flags):
        if flag:
            if start_idx is None:
                start_idx = idx
        else:
            if start_idx is not None:
                segments.append((start_idx, idx))
                start_idx = None
    if start_idx is not None:
        segments.append((start_idx, len(flags)))
    return segments


def _frames_to_samples(
    segments: Sequence[Tuple[int, int]],
    frame_bounds: Sequence[Tuple[int, int]],
    total_samples: int,
) -> List[Tuple[int, int]]:
    results: List[Tuple[int, int]] = []
    for start_idx, end_idx in segments:
        if start_idx >= len(frame_bounds):
            continue
        start_sample = frame_bounds[start_idx][0]
        last_frame_idx = min(max(end_idx - 1, 0), len(frame_bounds) - 1)
        end_sample = frame_bounds[last_frame_idx][1]
        end_sample = min(end_sample, total_samples)
        if end_sample > start_sample:
            results.append((start_sample, end_sample))
    return results


def _merge_short_gaps(segments: Sequence[Tuple[int, int]], min_gap_samples: int) -> List[Tuple[int, int]]:
    if not segments:
        return []
    merged: List[Tuple[int, int]] = [segments[0]]
    for start, end in segments[1:]:
        prev_start, prev_end = merged[-1]
        if start - prev_end <= min_gap_samples:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def _apply_padding(segment: Tuple[int, int], padding: int, total_samples: int) -> Tuple[int, int]:
    start, end = segment
    start = max(0, start - padding)
    end = min(total_samples, end + padding)
    return start, end


def _seconds_to_samples(value: float, sample_rate: int) -> int:
    if value is None:
        return 0
    return max(int(round(float(value) * sample_rate)), 0)


__all__ = [
    "VadConfig",
    "SpeechSegment",
    "detect_voice_segments",
    "segment_waveform",
]
