from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np
from pydantic import BaseModel, Field


def _avg_hash(image_bgr: np.ndarray, hash_size: int = 8) -> str:
    """平均ハッシュ（aHash）を16進文字列で返す。依存ライブラリ不要の簡易指標。"""

    try:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    except Exception:
        gray = image_bgr
    small = cv2.resize(gray, (hash_size, hash_size), interpolation=cv2.INTER_AREA)
    avg = float(small.mean())
    bits = (small > avg).astype(np.uint8).flatten()
    # 8bitごとにまとめて16進化
    accum = 0
    hex_digits = []
    for i, b in enumerate(bits, 1):
        accum = (accum << 1) | int(b)
        if i % 4 == 0:
            hex_digits.append(f"{accum:01x}")
            accum = 0
    if len(bits) % 4 != 0:
        hex_digits.append(f"{accum:01x}")
    return "".join(hex_digits)


def _dominant_colors(image_bgr: np.ndarray, k: int = 3) -> List[str]:
    """KMeans でざっくり代表色名を返す（UIの雰囲気把握用）。"""

    img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pixels = img.reshape((-1, 3)).astype(np.float32)
    # OpenCV の kmeans でクラスタリング
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    k = max(1, min(k, len(pixels)))
    _compact, labels, centers = cv2.kmeans(pixels, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    centers = centers.astype(int)
    names: List[str] = []
    for r, g, b in centers:
        # 簡易な色名化
        if r > 200 and g > 200 and b > 200:
            names.append("white")
        elif r < 50 and g < 50 and b < 50:
            names.append("black")
        elif r > g and r > b:
            names.append("red-ish")
        elif g > r and g > b:
            names.append("green-ish")
        elif b > r and b > g:
            names.append("blue-ish")
        else:
            names.append("mixed")
    return names


def _edge_density(image_bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return float(edges.mean()) / 255.0


def _ocr_text(image_bgr: np.ndarray) -> str:
    """利用可能なら pytesseract で OCR。無ければ空文字。"""

    try:
        import pytesseract  # type: ignore
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        # 日本語環境が無い場合に備えて eng と jpn を順に試す
        for lang in ("jpn", "eng"):
            try:
                text = pytesseract.image_to_string(rgb, lang=lang)
                if text and text.strip():
                    return text.strip()
            except Exception:
                continue
    except Exception:
        pass
    return ""


def _keywords_from_text(text: str) -> List[str]:
    """OCR文字列から簡易キーワードを抽出（固有名/サービス名のヒューリスティック）。"""

    lowered = text.lower()
    candidates = {
        "figma": "Figma",
        "google docs": "Google Docs",
        "google slides": "Google Slides",
        "slides": "Slides",
        "notion": "Notion",
        "slack": "Slack",
        "zoom": "Zoom",
        "chrome": "Chrome",
        "safari": "Safari",
        "vscode": "VS Code",
        "visual studio code": "VS Code",
        "xcode": "Xcode",
        "github": "GitHub",
    }
    found = []
    for needle, label in candidates.items():
        if needle in lowered:
            found.append(label)
    # 短い英単語を素朴に抽出（日本語は分かち書きしない）
    for tok in {t for t in lowered.replace("\n", " ").split(" ") if len(t) >= 5}:
        if tok.isalpha() and tok not in candidates:
            found.append(tok)
    # 重複排除
    seen = set()
    uniq = []
    for f in found:
        if f not in seen:
            seen.add(f)
            uniq.append(f)
    return uniq[:10]


class ImageContext(BaseModel):
    """1フレーム分の画面コンテキスト。"""

    timestamp: float
    frame_index: int
    ocr_text: str = ""
    tags: List[str] = Field(default_factory=list)
    hash: str = ""
    dominant_colors: List[str] = Field(default_factory=list)
    edge_density: float = 0.0
    path: Optional[str] = None

    @property
    def label(self) -> str:
        if self.tags:
            return ", ".join(self.tags[:2])
        if self.ocr_text:
            return self.ocr_text.splitlines()[0][:40]
        return f"frame#{self.frame_index}"


def analyze_frame_bgr(image_bgr: np.ndarray, *, frame_index: int, timestamp: float, path: Optional[str] = None) -> ImageContext:
    """BGR画像から簡易的な画面コンテキストを生成。"""

    text = _ocr_text(image_bgr)
    tags = _keywords_from_text(text)
    return ImageContext(
        timestamp=float(timestamp),
        frame_index=int(frame_index),
        ocr_text=text,
        tags=tags,
        hash=_avg_hash(image_bgr),
        dominant_colors=_dominant_colors(image_bgr),
        edge_density=_edge_density(image_bgr),
        path=path,
    )

