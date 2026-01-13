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

    frame_hop_duration: float = 0.01
    """フレームのホップ [sec]（<= frame_duration）。境界精度と誤検出抑制に効く。"""

    energy_threshold: float | None = None
    """固定しきい値（指定しない場合は波形に応じて推定）。"""

    min_energy: float = 5e-4
    """自動しきい値計算時の下限。"""

    dynamic_threshold_ratio: float = 2.5
    """ノイズ床に対する乗数。"""

    noise_quantile: float = 0.2
    """ノイズ推定に使う分位点 (0-1)。"""

    energy_smoothing_frames: int = 5
    """エネルギー時系列の移動平均フレーム数（>=1）。瞬間ノイズを抑える。"""

    hysteresis_ratio: float = 0.7
    """OFF しきい値 = ON しきい値 * hysteresis_ratio（0-1）。チャタリング抑制。"""

    start_trigger_frames: int = 2
    """ON へ遷移するのに必要な連続フレーム数。瞬間スパイクを抑える。"""

    end_trigger_frames: int = 3
    """OFF へ遷移するのに必要な連続フレーム数。語尾切れを抑える。"""

    min_speech_duration: float = 0.3
    """この長さ未満の区間は除去する。"""

    min_silence_duration: float = 0.2
    """この長さ以下の無音ギャップは結合する。"""

    padding_duration: float = 0.05
    """両端に与えるフェード的な余白。"""

    fusion_enabled: bool = False
    """エネルギーVADとTransformer由来のspeech確率を時系列で融合して判定する。"""

    transformer_model: str | None = None
    """Transformer VAD 用モデル名（mlx_whisper の path_or_hf_repo を想定）。"""

    fusion_energy_weight: float = 0.4
    """融合スコアにおけるエネルギー側の重み。"""

    fusion_transformer_weight: float = 0.6
    """融合スコアにおけるTransformer側の重み。"""

    fusion_on_threshold: float = 0.6
    """融合スコアのON閾値（0-1）。"""

    fusion_off_threshold: float = 0.4
    """融合スコアのOFF閾値（0-1）。"""

    fusion_gate_threshold: float = 0.15
    """Transformerが低確率のときにエネルギー寄与を抑えるゲート（<=0で無効）。"""

    fusion_smoothing_frames: int | None = None
    """融合スコアの平滑化フレーム数（Noneなら energy_smoothing_frames を利用）。"""

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
    hop_duration = float(cfg.frame_hop_duration if cfg.frame_hop_duration is not None else cfg.frame_duration)
    hop_duration = max(0.0, min(hop_duration, float(cfg.frame_duration)))
    hop_size = max(int(hop_duration * sample_rate), 1)
    frame_bounds, energies = _calc_frame_energies(wf, frame_size, hop_size)
    if not energies:
        return []

    energies = _smooth_series(energies, window=max(int(cfg.energy_smoothing_frames), 1))
    threshold = _estimate_threshold(energies, cfg)
    voiced_flags: List[bool] = _hysteresis_binarize(
        energies,
        on_threshold=threshold,
        off_threshold=threshold * float(np.clip(cfg.hysteresis_ratio, 0.0, 1.0)),
        start_trigger=max(int(cfg.start_trigger_frames), 1),
        end_trigger=max(int(cfg.end_trigger_frames), 1),
    )
    fused = _maybe_fuse_with_transformer(
        waveform=wf,
        sample_rate=sample_rate,
        frame_bounds=frame_bounds,
        hop_duration=hop_duration,
        energies=energies,
        energy_threshold=threshold,
        cfg=cfg,
    )
    if fused is not None:
        voiced_flags = fused
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


def _maybe_fuse_with_transformer(
    *,
    waveform: np.ndarray,
    sample_rate: int,
    frame_bounds: Sequence[Tuple[int, int]],
    hop_duration: float,
    energies: Sequence[float],
    energy_threshold: float,
    cfg: VadConfig,
) -> List[bool] | None:
    if not cfg.fusion_enabled:
        return None
    model = (cfg.transformer_model or "").strip()
    if not model:
        return None
    if hop_duration <= 0.0:
        return None

    try:
        from .transformer import estimate_speech_probabilities
    except Exception:
        return None

    try:
        probs = estimate_speech_probabilities(
            waveform,
            sample_rate=sample_rate,
            frame_bounds=frame_bounds,
            hop_duration=hop_duration,
            model_name=model,
        )
    except Exception:
        return None

    energy_list = list(energies)
    if not probs or len(probs) != len(energy_list):
        return None

    w_e = float(cfg.fusion_energy_weight) if math.isfinite(float(cfg.fusion_energy_weight)) else 0.0
    w_t = float(cfg.fusion_transformer_weight) if math.isfinite(float(cfg.fusion_transformer_weight)) else 0.0
    w_e = max(w_e, 0.0)
    w_t = max(w_t, 0.0)
    total_w = w_e + w_t
    if total_w <= 0:
        return None
    w_e /= total_w
    w_t /= total_w

    denom = float(energy_threshold) * 2.0
    if not math.isfinite(denom) or denom <= 0:
        denom = max(float(cfg.min_energy) * 2.0, 1e-12)

    energy_arr = np.asarray(energy_list, dtype=np.float32)
    energy_score = np.clip(energy_arr / float(denom), 0.0, 1.0)

    smooth_frames = cfg.fusion_smoothing_frames
    if smooth_frames is None:
        smooth_frames = cfg.energy_smoothing_frames
    smooth_frames = max(int(smooth_frames), 1)

    prob_arr = np.asarray(probs, dtype=np.float32)
    prob_arr = np.clip(prob_arr, 0.0, 1.0)
    prob_arr = np.asarray(_smooth_series(prob_arr.tolist(), window=smooth_frames), dtype=np.float32)

    gate = float(cfg.fusion_gate_threshold)
    if math.isfinite(gate) and gate > 0:
        gate_scale = np.clip(prob_arr / gate, 0.0, 1.0)
        energy_score = energy_score * gate_scale

    fused = (w_e * energy_score) + (w_t * prob_arr)
    fused_list = _smooth_series(fused.tolist(), window=smooth_frames)

    on_th = float(cfg.fusion_on_threshold)
    off_th = float(cfg.fusion_off_threshold)
    if not math.isfinite(on_th):
        on_th = 0.6
    if not math.isfinite(off_th):
        off_th = 0.4
    on_th = float(np.clip(on_th, 0.0, 1.0))
    off_th = float(np.clip(off_th, 0.0, on_th))

    return _hysteresis_binarize(
        fused_list,
        on_threshold=on_th,
        off_threshold=off_th,
        start_trigger=max(int(cfg.start_trigger_frames), 1),
        end_trigger=max(int(cfg.end_trigger_frames), 1),
    )


def _ensure_mono_waveform(waveform: np.ndarray) -> np.ndarray:
    wf = np.asarray(waveform, dtype=np.float32)
    if wf.ndim == 0:
        return wf.reshape(1)
    if wf.ndim == 1:
        return wf
    return np.mean(wf, axis=-1)


def _calc_frame_energies(
    waveform: np.ndarray,
    frame_size: int,
    hop_size: int,
) -> Tuple[List[Tuple[int, int]], List[float]]:
    bounds: List[Tuple[int, int]] = []
    energies: List[float] = []
    total = waveform.size
    hop = max(int(hop_size), 1)
    size = max(int(frame_size), 1)
    for start in range(0, total, hop):
        end = min(total, start + size)
        frame = waveform[start:end]
        if frame.size == 0:
            continue
        energy = float(np.mean(np.abs(frame)))
        bounds.append((start, end))
        energies.append(energy)
    return bounds, energies


def _smooth_series(values: Sequence[float], *, window: int) -> List[float]:
    if window <= 1:
        return [float(v) for v in values]
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0:
        return []
    kernel = np.ones(int(window), dtype=np.float32) / float(window)
    smoothed = np.convolve(arr, kernel, mode="same")
    return [float(x) for x in smoothed]


def _hysteresis_binarize(
    energies: Sequence[float],
    *,
    on_threshold: float,
    off_threshold: float,
    start_trigger: int,
    end_trigger: int,
) -> List[bool]:
    if not energies:
        return []
    on_th = float(on_threshold) if math.isfinite(float(on_threshold)) else 0.0
    off_th = float(off_threshold) if math.isfinite(float(off_threshold)) else 0.0
    if off_th > on_th:
        off_th = on_th
    start_trigger = max(int(start_trigger), 1)
    end_trigger = max(int(end_trigger), 1)

    flags = [False] * len(energies)
    in_speech = False
    above = 0
    below = 0

    for i, e in enumerate(energies):
        energy = float(e)
        if not math.isfinite(energy):
            energy = 0.0

        if not in_speech:
            if energy >= on_th:
                above += 1
                if above >= start_trigger:
                    in_speech = True
                    start_idx = i - start_trigger + 1
                    for j in range(start_idx, i + 1):
                        flags[j] = True
                    above = 0
                    below = 0
            else:
                above = 0
            continue

        flags[i] = True
        if energy <= off_th:
            below += 1
            if below >= end_trigger:
                end_idx = i - end_trigger + 1
                for j in range(end_idx, i + 1):
                    flags[j] = False
                in_speech = False
                above = 0
                below = 0
        else:
            below = 0

    return flags


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
