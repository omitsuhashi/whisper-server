"""Streaming correction pipeline scaffolding."""
from __future__ import annotations

from .editor import EditorConfig, GlossaryEditor
from .integration import apply_corrections_to_results
from .pipeline import CorrectionPipeline, run_correction
from .tagger import BertCrfTagger, BertCrfTaggerConfig, RuleBasedPunctTagger, TaggerConfig
from .types import (
    CorrectionError,
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
    "GlossaryEditor",
    "EditorConfig",
    "RuleBasedPunctTagger",
    "TaggerConfig",
    "BertCrfTagger",
    "BertCrfTaggerConfig",
    "CorrectionOptions",
    "CorrectionPatch",
    "CorrectionRequest",
    "CorrectionResult",
    "CorrectionSpan",
    "apply_patches",
    "CorrectionError",
]
