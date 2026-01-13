from __future__ import annotations

import os
from dataclasses import replace

from .detector import VadConfig


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return (raw or "").strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str) -> int | None:
    raw = os.getenv(name)
    if raw is None:
        return None
    try:
        value = int(float(raw))
    except ValueError:
        return None
    return value if value > 0 else None


def resolve_vad_config() -> VadConfig:
    """実行環境のENVからVadConfigを組み立てる（未指定時は現行デフォルト互換）。"""

    base = VadConfig()
    mode = (os.getenv("ASR_VAD_MODE") or "").strip().lower()
    enabled = _env_bool("ASR_VAD_FUSION", False) or mode in {"fused", "fusion", "hybrid"}
    model = (os.getenv("ASR_VAD_TRANSFORMER_MODEL") or "").strip()
    if not enabled or not model:
        return base

    smoothing = _env_int("ASR_VAD_FUSION_SMOOTHING_FRAMES")
    return replace(
        base,
        fusion_enabled=True,
        transformer_model=model,
        fusion_energy_weight=_env_float("ASR_VAD_FUSION_WEIGHT_ENERGY", base.fusion_energy_weight),
        fusion_transformer_weight=_env_float(
            "ASR_VAD_FUSION_WEIGHT_TRANSFORMER", base.fusion_transformer_weight
        ),
        fusion_on_threshold=_env_float("ASR_VAD_FUSION_ON", base.fusion_on_threshold),
        fusion_off_threshold=_env_float("ASR_VAD_FUSION_OFF", base.fusion_off_threshold),
        fusion_gate_threshold=_env_float("ASR_VAD_FUSION_GATE", base.fusion_gate_threshold),
        fusion_smoothing_frames=smoothing,
    )


__all__ = ["resolve_vad_config"]
