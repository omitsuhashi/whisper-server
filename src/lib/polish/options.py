from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Tuple


@dataclass
class PolishOptions:
    """校正オプション（すべて任意・安全なデフォルト）"""

    style: str = "ですます"  # "常体" も指定可
    use_ginza: bool = True  # GiNZA を利用した文分割
    ginza_model: str = "ja_ginza_electra"
    ginza_model_fallbacks: Tuple[str, ...] = ("ja_ginza",)
    ginza_fallback_to_heuristics: bool = True  # GiNZA 失敗時にヒューリスティクスへフォールバックするか
    remove_fillers: bool = True
    filler_patterns: Tuple[str, ...] = (
        r"(えー|あー|えっと|そのー|まー|なんか)(?:\s|$)",
        r"\b(uh|um|erm|you know|like)\b",
    )
    normalize_width: bool = True  # 全角/半角の正規化(NFKC)
    space_collapse: bool = True  # 余剰スペースの圧縮
    remove_repeated_chars: bool = True
    max_char_repeat: int = 2  # 連続文字の上限（!？ーなども含む）
    term_pairs: Tuple[Tuple[str, str], ...] = field(default_factory=tuple)  # 置換辞書（左→右）
    term_pairs_regex: bool = False  # 左側を正規表現として扱う
    protected_token_patterns: Tuple[str, ...] = (
        r"[A-Z]{2,}",  # AWS, URL など
        r"v\d+\.\d+",  # v1.2
        r"[A-Za-z]+\d{2,}",  # 型番系
    )
    # GiNZA 無し時のヒューリスティクス
    period_heuristics: Tuple[str, ...] = (
        r"(です|ます|でした|だ|である|でしょう|ですね|でしたら|でしたよね)$",
        r"([!?！？]+)$",
    )
    min_sentence_len: int = 6
    max_sentence_len: int = 200

    def model_dump(self, *, exclude_none: bool = False, **_: Any) -> Dict[str, Any]:
        """Pydantic BaseModel 互換のダンプメソッド。"""

        data: Dict[str, Any] = dict(vars(self))
        if exclude_none:
            data = {key: value for key, value in data.items() if value is not None}
        return data


__all__ = ["PolishOptions"]
