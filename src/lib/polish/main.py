# filepath: src/lib/polish/main.py
from __future__ import annotations

import json
import logging
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import spacy
from pydantic import BaseModel, ConfigDict, Field
from spacy.language import Language

logger = logging.getLogger(__name__)

_GINZA_MODEL_NAME = "ja_ginza_electra"
_GINZA_NLP: Language | None = None


def _load_ginza(model_name: str) -> Language:
    """GiNZA モデルをロードし、グローバルにキャッシュする。"""

    global _GINZA_MODEL_NAME, _GINZA_NLP
    if _GINZA_NLP is not None and _GINZA_MODEL_NAME == model_name:
        return _GINZA_NLP

    try:
        nlp = spacy.load(model_name)
    except Exception as exc:  # noqa: BLE001 - モデル依存なので詳細な例外はラップしない
        raise RuntimeError(
            f"GiNZA モデル '{model_name}' の読み込みに失敗しました。"
            " pip install ginza ja-ginza-electra 等で依存関係を満たしてください。"
        ) from exc

    _GINZA_MODEL_NAME = model_name
    _GINZA_NLP = nlp
    return nlp

# asr モジュールのモデルをそのまま受け取れるように型だけ参照
try:
    from ..asr import TranscriptionResult, TranscriptionSegment  # type: ignore
except Exception:
    # 型チェックを緩和（依存先がまだ読み込めない状況でも利用可）
    class TranscriptionSegment(BaseModel):  # type: ignore
        start: float = 0.0
        end: float = 0.0
        text: str = ""
    class TranscriptionResult(BaseModel):  # type: ignore
        filename: str = "unknown"
        text: str = ""
        language: str | None = None
        duration: float | None = None
        segments: List[TranscriptionSegment] = Field(default_factory=list)


# ------------------------------
# モデル
# ------------------------------
class PolishedSentence(BaseModel):
    """校正後の“文”。SRT 等に使えるよう区間を保持。"""
    model_config = ConfigDict(extra="ignore")
    start: float
    end: float
    text: str


class PolishedDocument(BaseModel):
    """校正済みドキュメント全体。"""
    model_config = ConfigDict(extra="ignore")
    filename: str
    sentences: List[PolishedSentence] = Field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n".join(s.text for s in self.sentences)


@dataclass
class PolishOptions:
    """校正オプション（すべて任意・安全なデフォルト）"""
    style: str = "ですます"          # "常体" も指定可
    use_ginza: bool = True           # GiNZA を利用した文分割
    ginza_model: str = "ja_ginza_electra"
    ginza_fallback_to_heuristics: bool = True  # GiNZA 失敗時にヒューリスティクスへフォールバックするか
    remove_fillers: bool = True
    filler_patterns: Tuple[str, ...] = (
        r"(えー|あー|えっと|そのー|まー|なんか)(?:\s|$)",
        r"\b(uh|um|erm|you know|like)\b",
    )
    normalize_width: bool = True     # 全角/半角の正規化(NFKC)
    space_collapse: bool = True      # 余剰スペースの圧縮
    remove_repeated_chars: bool = True
    max_char_repeat: int = 2         # 連続文字の上限（!？ーなども含む）
    term_pairs: Tuple[Tuple[str, str], ...] = field(default_factory=tuple)  # 置換辞書（左→右）
    term_pairs_regex: bool = False   # 左側を正規表現として扱う
    protected_token_patterns: Tuple[str, ...] = (
        r"[A-Z]{2,}",           # AWS, URL など
        r"v\d+\.\d+",           # v1.2
        r"[A-Za-z]+\d{2,}",     # 型番系
    )
    # GiNZA 無し時のヒューリスティクス
    period_heuristics: Tuple[str, ...] = (
        r"(です|ます|でした|だ|である|でしょう|ですね|でしたら|でしたよね)$",
        r"([!?！？]+)$",
    )
    min_sentence_len: int = 6
    max_sentence_len: int = 200


# ------------------------------
# ユーティリティ（保存）
# ------------------------------
def save_txt(path: str | Path, text: str) -> None:
    Path(path).write_text(text + "\n", encoding="utf-8")


def save_json_doc(path: str | Path, doc: PolishedDocument) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc.model_dump(), f, ensure_ascii=False, indent=2)


def save_srt(path: str | Path, sentences: Iterable[PolishedSentence]) -> None:
    def fmt_time(t: float) -> str:
        t = max(0.0, t)
        h = int(t // 3600)
        m = int((t % 3600) // 60)
        s = int(t % 60)
        ms = int(round((t - int(t)) * 1000))
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    lines: List[str] = []
    for i, s in enumerate(sentences, 1):
        start = max(0.0, s.start)
        end = max(start + 0.01, s.end)
        lines.append(str(i))
        lines.append(f"{fmt_time(start)} --> {fmt_time(end)}")
        lines.append(s.text)
        lines.append("")
    Path(path).write_text("\n".join(lines), encoding="utf-8")


# ------------------------------
# 前処理（正規化・フィラー等）
# ------------------------------
def _normalize_text(s: str, opt: PolishOptions) -> str:
    if opt.normalize_width:
        s = unicodedata.normalize("NFKC", s)
    if opt.space_collapse:
        s = re.sub(r"[ \t\u3000]+", " ", s).strip()
    # 日本語の句読点を統一（任意の軽微整形）
    s = s.replace("，", "、").replace("．", "。")
    return s


def _remove_fillers(s: str, opt: PolishOptions) -> str:
    if not opt.remove_fillers:
        return s
    for p in opt.filler_patterns:
        s = re.sub(p, "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s{2,}", " ", s).strip()
    return s


def _limit_repeats(s: str, max_repeat: int) -> str:
    # 「すごーーーい」→「すごーい」 / 「!!!」→「!!」
    def repl(m: re.Match[str]) -> str:
        ch = m.group(1)
        return ch * max_repeat
    return re.sub(r"(.)\1{%d,}" % (max_repeat), repl, s)


# ------------------------------
# 用語辞書（置換 & 保護）
# ------------------------------
def _compile_protected(patterns: Iterable[str]) -> List[re.Pattern[str]]:
    out: List[re.Pattern[str]] = []
    for p in patterns:
        p = p.strip()
        if not p:
            continue
        out.append(re.compile(p))
    return out


def _protect_tokens(s: str, prot_res: List[re.Pattern[str]]) -> Tuple[str, Dict[str, str]]:
    mapping: Dict[str, str] = {}
    for idx, pat in enumerate(prot_res):
        def repl(m: re.Match[str]) -> str:
            token = m.group(0)
            key = f"__PROT_{idx}_{len(mapping)}__"
            mapping[key] = token
            return key
        s = pat.sub(repl, s)
    return s, mapping


def _unprotect_tokens(s: str, mapping: Dict[str, str]) -> str:
    for k, v in mapping.items():
        s = s.replace(k, v)
    return s


def _apply_terms(s: str, pairs: Tuple[Tuple[str, str], ...], regex: bool) -> str:
    if not pairs:
        return s
    for src, dst in pairs:
        pat = src if regex else re.escape(src)
        s = re.sub(pat, dst)
    return s


# ------------------------------
# 文分割
# ------------------------------
def _sent_split_ginza(text: str, opt: PolishOptions) -> List[str]:
    """GiNZA による文分割。"""

    nlp = _load_ginza(opt.ginza_model)
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]


def _sent_split_heuristics(text: str, opt: PolishOptions) -> List[str]:
    # 既存の句点や改行で切る → 足りなければ語尾ヒューリスティクスで補句点
    pieces = re.split(r"(?<=[。！？!?])\s*|\n+", text)
    pieces = [p.strip() for p in pieces if p and p.strip()]
    out: List[str] = []
    buf = ""
    for chunk in pieces:
        buf = (buf + " " + chunk).strip() if buf else chunk
        too_long = len(buf) >= opt.max_sentence_len
        matched_tail = any(re.search(p, buf) for p in opt.period_heuristics)
        if too_long or matched_tail:
            if not re.search(r"[。！？!?]\s*$", buf):
                buf += "。"
            out.append(buf)
            buf = ""
    if buf:
        if not re.search(r"[。！？!?]\s*$", buf):
            buf += "。"
        out.append(buf)

    # 短すぎる断片は前に吸収
    merged: List[str] = []
    for s in out:
        if merged and len(s) < opt.min_sentence_len:
            merged[-1] = (merged[-1].rstrip("。") + " " + s).strip()
            if not merged[-1].endswith(("。", "！", "？", "!", "?")):
                merged[-1] += "。"
        else:
            merged.append(s)
    return merged


def _split_into_sentences(text: str, opt: PolishOptions) -> List[str]:
    if opt.use_ginza:
        try:
            sents = _sent_split_ginza(text, opt)
            if sents:
                return sents
        except Exception as exc:  # noqa: BLE001
            if not opt.ginza_fallback_to_heuristics:
                raise
            logger.warning("GiNZAによる文分割に失敗したためヒューリスティクスへフォールバックします: %s", exc)
    return _sent_split_heuristics(text, opt)


# ------------------------------
# タイムスタンプ再割当（貪欲・文字長比）
# ------------------------------
def _reassign_timestamps(sentences: List[str], segs: List[TranscriptionSegment]) -> List[PolishedSentence]:
    out: List[PolishedSentence] = []
    if not sentences:
        return out
    if not segs:
        # セグメントが無い場合はフラットに割当
        step = 2.0
        t = 0.0
        for s in sentences:
            out.append(PolishedSentence(start=t, end=t+step, text=s))
            t += step
        return out

    i = 0
    for sent in sentences:
        sent_len = max(1, len(re.sub(r"\s+", "", sent)))
        # 文のテキスト長に応じてセグメントを消費
        start = segs[i].start if i < len(segs) else (out[-1].end if out else 0.0)
        acc = ""
        j = i
        while j < len(segs) and len(re.sub(r"\s+", "", acc)) < sent_len:
            acc = (acc + " " + segs[j].text).strip()
            j += 1
        end = segs[min(j-1, len(segs)-1)].end if j > i else (segs[-1].end if segs else start + 1.5)
        out.append(PolishedSentence(start=float(start), end=float(max(end, start + 0.01)), text=sent))
        i = max(j, i + 1)
    # 重なり修正
    for k in range(1, len(out)):
        if out[k].start < out[k-1].end:
            out[k].start = out[k-1].end
    return out


# ------------------------------
# エントリポイント
# ------------------------------
def polish_text_from_segments(segments: Iterable[TranscriptionSegment], *, options: Optional[PolishOptions] = None) -> List[PolishedSentence]:
    """ASRセグメント列から校正済みの文列を作成。"""
    opt = options or PolishOptions()

    # 1) セグメント単位のクリーニング
    cleaned: List[TranscriptionSegment] = []
    for s in segments:
        t = _normalize_text(s.text or "", opt)
        t = _remove_fillers(t, opt)
        if opt.remove_repeated_chars:
            t = _limit_repeats(t, opt.max_char_repeat)
        seg = TranscriptionSegment.model_validate({"start": float(s.start or 0.0), "end": float(s.end or 0.0), "text": t})
        cleaned.append(seg)

    joined = " ".join(seg.text for seg in cleaned).strip()

    # 2) 置換の破壊から守るトークンを保護
    prot_res = _compile_protected(opt.protected_token_patterns)
    masked, mapping = _protect_tokens(joined, prot_res)

    # 3) 用語辞書（正規表記へ）
    masked = _apply_terms(masked, opt.term_pairs, regex=opt.term_pairs_regex)

    # 4) 文分割（GiNZA→フォールバック）
    sentences = _split_into_sentences(masked, opt)

    # 5) マスク解除
    sentences = [_unprotect_tokens(s, mapping) for s in sentences]

    # 6) スタイル統一
    if opt.style == "常体":
        sentences = [s.replace("です。", "だ。").replace("ます。", "る。") for s in sentences]

    # 7) タイムスタンプ再割当
    return _reassign_timestamps(sentences, cleaned)


def polish_result(result: TranscriptionResult, *, options: Optional[PolishOptions] = None) -> PolishedDocument:
    """TranscriptionResult -> PolishedDocument"""
    sentences = polish_text_from_segments(result.segments, options=options)
    return PolishedDocument(filename=result.filename, sentences=sentences)
