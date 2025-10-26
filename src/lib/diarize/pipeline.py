from __future__ import annotations

import os
from typing import Any, Dict, List, TYPE_CHECKING

from .models import SpeakerTurn
from .options import DiarizeOptions


# ---- Torch (MPS) ----
try:
    import torch  # type: ignore

    _MPS_OK = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
except Exception:  # pragma: no cover - torch optional at import time
    _MPS_OK = False

# ---- pyannote ----
try:  # pragma: no cover - 依存がない環境では失敗させない
    from pyannote.audio import Pipeline as _PyannotePipeline  # type: ignore
    _pyannote_import_error: Exception | None = None
except Exception as e:  # pragma: no cover
    _PyannotePipeline = None
    _pyannote_import_error = e

if TYPE_CHECKING:  # pragma: no cover
    from pyannote.audio import Pipeline as PyannotePipeline  # type: ignore[import]
else:
    PyannotePipeline = Any


def load_pipeline(opt: DiarizeOptions) -> PyannotePipeline:
    if _PyannotePipeline is None:
        raise RuntimeError(
            "pyannote.audio が見つかりません。`pip install 'pyannote.audio>=3.1,<3.3' torch torchaudio soundfile` を実行してください。"
        ) from _pyannote_import_error
    token = _resolve_token(opt)
    try:
        pipeline = _PyannotePipeline.from_pretrained(opt.model_name, token=token)  # pyannote>=3.1
    except TypeError:
        pipeline = _PyannotePipeline.from_pretrained(opt.model_name)  # 旧署名
    import torch as _torch

    device_name = _resolve_device(opt)
    device = _torch.device(device_name)
    pipeline.to(device)  # type: ignore[attr-defined]
    return pipeline


def build_diarization_kwargs(opt: DiarizeOptions) -> Dict[str, int]:
    candidates = {
        "num_speakers": opt.num_speakers,
        "min_speakers": opt.min_speakers,
        "max_speakers": opt.max_speakers,
    }
    return {key: value for key, value in candidates.items() if value is not None}


def annotation_to_turns(annotation: Any) -> List[SpeakerTurn]:
    """
    pyannote.audio 3.x は `Annotation` を、4.x は `DiarizeOutput` を返す。
    どちらの場合でも `speaker_diarization` に含まれるトラックを抽出する。
    """

    if hasattr(annotation, "speaker_diarization"):
        source = annotation.speaker_diarization
    else:
        source = annotation

    if not hasattr(source, "itertracks"):
        raise TypeError(f"サポートされていない話者分離結果: {type(annotation)!r}")

    turns = [
        SpeakerTurn(start=float(turn.start), end=float(turn.end), speaker=str(speaker))
        for turn, _, speaker in source.itertracks(yield_label=True)
    ]
    turns.sort(key=lambda t: (t.start, t.end))
    return turns


def _resolve_device(opt: DiarizeOptions) -> str:
    if opt.device not in (None, "mps"):
        raise RuntimeError(
            f"この構成では MPS (Metal) のみサポートしています。指定されたデバイス: {opt.device!r}"
        )
    if not _MPS_OK:
        raise RuntimeError(
            "MPS (Metal) が利用できません。PyTorch を MPS 対応ビルドに更新するか、"
            "macOS 側で Metal が利用可能な環境を用意してください。"
        )
    return "mps"


def _resolve_token(opt: DiarizeOptions) -> str:
    if opt.token:
        return opt.token
    for key in opt.token_env:
        value = os.environ.get(key)
        if value:
            return value
    raise RuntimeError(
        "Hugging Face のアクセストークンが見つかりません。"
        "環境変数 HF_TOKEN / HUGGINGFACE_TOKEN / PYANNOTE_TOKEN のいずれかに設定してください。"
    )


__all__ = [
    "annotation_to_turns",
    "build_diarization_kwargs",
    "load_pipeline",
]
