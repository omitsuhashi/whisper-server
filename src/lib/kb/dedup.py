from __future__ import annotations

from typing import Iterable

import hashlib
from datasketch import MinHash


def tokenize_for_dedup(text: str) -> list[str]:
    """簡易トークナイザ。言語混在を考慮しシンプルに空白区切りとする。"""

    return text.lower().split()


def build_minhash(tokens: Iterable[str], num_perm: int = 128) -> MinHash:
    """MinHash 署名を生成する。"""

    minhash = MinHash(num_perm=num_perm)
    for token in tokens:
        minhash.update(token.encode("utf-8", errors="ignore"))
    return minhash


def simhash_hex(text: str) -> str:
    """SimHash 互換のフィンガープリントを 16 進文字列で返す。"""

    tokens = tokenize_for_dedup(text) or [text]
    bits = [0] * 64
    for token in tokens:
        digest = hashlib.blake2b(token.encode("utf-8", errors="ignore"), digest_size=8).digest()
        value = int.from_bytes(digest, byteorder="big", signed=False)
        for idx in range(64):
            weight = 1 if value & (1 << idx) else -1
            bits[idx] += weight
    fingerprint = 0
    for idx, weight in enumerate(bits):
        if weight >= 0:
            fingerprint |= 1 << idx
    return f"{fingerprint:016x}"
