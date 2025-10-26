"""Lazy exporters for ASR utilities."""
from __future__ import annotations

from importlib import import_module
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - 型チェッカー向けの擬似インポート
    from src.lib.asr.main import transcribe_all, transcribe_all_bytes
    from src.lib.asr.models import TranscriptionResult, TranscriptionSegment
    from src.lib.asr.options import TranscribeOptions

__all__ = (
    "TranscribeOptions",
    "TranscriptionResult",
    "TranscriptionSegment",
    "transcribe_all",
    "transcribe_all_bytes",
)


def __getattr__(name: str) -> Any:
    if name == "TranscribeOptions":
        module = import_module("src.lib.asr.options")
        return getattr(module, name)

    if name in {"TranscriptionResult", "TranscriptionSegment"}:
        module = import_module("src.lib.asr.models")
        return getattr(module, name)

    if name in {"transcribe_all", "transcribe_all_bytes"}:
        module = import_module("src.lib.asr.main")
        return getattr(module, name)

    raise AttributeError(name)


def __dir__() -> list[str]:  # pragma: no cover - 補助関数
    return sorted(__all__)
