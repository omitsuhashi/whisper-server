from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import torch
from sqlalchemy import select, update
from transformers import AutoModel, AutoTokenizer

from src.lib.kb.db import session_scope
from src.lib.kb.models import Unit

DEFAULT_MODEL_NAME = os.getenv("KB_EMBED_MODEL", "mlx-community/embeddinggemma-300m")
MAX_LENGTH = int(os.getenv("KB_EMBED_MAX_LENGTH", "768"))
BATCH_SIZE = int(os.getenv("KB_EMBED_BATCH_SIZE", "16"))

_tokenizer: AutoTokenizer | None = None
_model: AutoModel | None = None
_device: torch.device | None = None


@dataclass
class EmbeddingJobResult:
    processed_units: int = 0
    embedded_units: int = 0


def _ensure_model() -> tuple[AutoTokenizer, AutoModel, torch.device]:
    global _tokenizer, _model, _device
    if _tokenizer is not None and _model is not None and _device is not None:
        return _tokenizer, _model, _device

    _tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL_NAME)
    _model = AutoModel.from_pretrained(DEFAULT_MODEL_NAME)
    _model.eval()

    if torch.cuda.is_available():
        _device = torch.device("cuda")
    elif torch.backends.mps.is_available():  # type: ignore[attr-defined]
        _device = torch.device("mps")
    else:
        _device = torch.device("cpu")

    _model.to(_device)
    return _tokenizer, _model, _device


def _chunk(sequence: Sequence[str], size: int) -> Iterable[Sequence[str]]:
    for idx in range(0, len(sequence), size):
        yield sequence[idx : idx + size]


def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    masked = last_hidden_state * mask
    sum_hidden = masked.sum(dim=1)
    lengths = mask.sum(dim=1).clamp(min=1e-9)
    return sum_hidden / lengths


def _encode(texts: Sequence[str]) -> np.ndarray:
    if not texts:
        return np.empty((0, 0))
    tokenizer, model, device = _ensure_model()
    outputs: list[np.ndarray] = []
    with torch.inference_mode():
        for batch in _chunk(list(texts), BATCH_SIZE):
            tokens = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            )
            tokens = {k: v.to(device) for k, v in tokens.items()}
            model_out = model(**tokens)
            pooled = _mean_pool(model_out.last_hidden_state, tokens["attention_mask"])
            normalized = torch.nn.functional.normalize(pooled, p=2, dim=1)
            outputs.append(normalized.cpu().numpy())
    return np.vstack(outputs)


def encode_passages(passages: Sequence[str]) -> np.ndarray:
    prefixed = [f"passage: {text.strip()}" for text in passages]
    return _encode(prefixed)


def encode_queries(queries: Sequence[str]) -> np.ndarray:
    prefixed = [f"query: {text.strip()}" for text in queries]
    return _encode(prefixed)


def populate_missing_embeddings(batch_size: int = 16, limit: int | None = None) -> EmbeddingJobResult:
    """units.embed が NULL の行に対してベクトルを埋める。"""

    result = EmbeddingJobResult()
    with session_scope() as session:
        stmt = select(Unit.id, Unit.text).where(Unit.embed.is_(None)).order_by(Unit.id)
        if limit is not None and limit > 0:
            stmt = stmt.limit(limit)
        rows = session.execute(stmt).all()
        if not rows:
            return result

        total = len(rows)
        for start in range(0, total, batch_size):
            chunk = rows[start : start + batch_size]
            chunk_texts = [row.text or "" for row in chunk]
            embeddings = encode_passages(chunk_texts)
            for (row, vector) in zip(chunk, embeddings):
                session.execute(
                    update(Unit)
                    .where(Unit.id == row.id)
                    .values(embed=vector.tolist()),
                )
            result.embedded_units += len(chunk)
            result.processed_units += len(chunk)

    return result
