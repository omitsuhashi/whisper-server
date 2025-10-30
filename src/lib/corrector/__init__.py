"""Streaming correction pipeline scaffolding."""
from __future__ import annotations

from .integration import apply_corrections_to_results
from .pipeline import CorrectionPipeline, run_correction
from .types import (
    CorrectionOptions,
    CorrectionPatch,
    CorrectionRequest,
    CorrectionResult,
    CorrectionSpan,
    apply_patches,
)

__all__ = [
    "CorrectionPipeline",
    "run_correction",
    "apply_corrections_to_results",
    "CorrectionOptions",
    "CorrectionPatch",
    "CorrectionRequest",
    "CorrectionResult",
    "CorrectionSpan",
    "apply_patches",
]
