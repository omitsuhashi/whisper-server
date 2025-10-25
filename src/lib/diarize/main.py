from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

# ---- Torch (MPS) ----
try:
    import torch  # type: ignore
    _MPS_OK = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    try:
        torch.set_float32_matmul_precision("high")  # 古いtorchではno-op
    except Exception:
        pass
except Exception:
    _MPS_OK = False

# ---- pyannote ----
try:
    from pyannote.audio import Pipeline  # type: ignore
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "pyannote.audio が見つかりません。`pip install 'pyannote.audio>=3.1,<3.3' torch torchaudio soundfile` を実行してください。"
    ) from e

# ---- ASR 型（参照のみ） ----
try:
    from ..asr import TranscriptionResult, TranscriptionSegment  # type: ignore
except Exception:
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
class SpeakerTurn(BaseModel):
    """話者区間（start/end は秒）。"""
    model_config = ConfigDict(extra="ignore")
    start: float
    end: float
    speaker: str

class DiarizationResult(BaseModel):
    """話者分離の結果（ファイル単位）。"""
    model_config = ConfigDict(extra="ignore")
    filename: str
    duration: float | None = None
    turns: List[SpeakerTurn] = Field(default_factory=list)

    @property
    def speakers(self) -> List[str]:
        return sorted({t.speaker for t in self.turns})

class SpeakerSegment(BaseModel):
    """話者ラベル付き ASR セグメント。"""
    model_config = ConfigDict(extra="ignore")
    id: int | None = None
    start: float
    end: float
    text: str
    speaker: str = "UNK"

class SpeakerAnnotatedTranscript(BaseModel):
    """話者ラベルを付与済みの書き起こし全体。"""
    model_config = ConfigDict(extra="ignore")
    filename: str
    segments: List[SpeakerSegment] = Field(default_factory=list)

    @property
    def speakers(self) -> List[str]:
        return sorted({s.speaker for s in self.segments})

# ------------------------------
# オプション
# ------------------------------
@dataclass
class DiarizeOptions:
    """
    Diarization 実行時のオプション。
    - model_name: 既定は pyannote/speaker-diarization-3.1
    - token_env:  トークンを読む環境変数（順に探索）
    - device:     "mps"|"cpu"|"cuda"（Noneで自動）
    - require_mps: True で MPS 未検出時に例外（Metal前提の安全弁）
    - *speakers:  人数の事前情報があると安定（num/min/max）
    """
    model_name: str = "pyannote/speaker-diarization-3.1"
    token: str | None = None
    token_env: Tuple[str, ...] = ("HF_TOKEN", "HUGGINGFACE_TOKEN", "PYANNOTE_TOKEN")
    device: str | None = None
    require_mps: bool = True
    num_speakers: int | None = None
    min_speakers: int | None = None
    max_speakers: int | None = None
    sample_rate: int = 16000  # waveform入力時の既定SR（pyannoteはSR指定可能）

# ------------------------------
# パイプライン準備
# ------------------------------
def _resolve_device(opt: DiarizeOptions) -> str:
    if opt.device:
        return opt.device
    if _MPS_OK:
        return "mps"
    if opt.require_mps:
        raise RuntimeError(
            "MPS (Metal) が利用できません。Apple Silicon + PyTorch(MPS対応) が必要です。"
            " Intel Mac／MPS非対応環境では DiarizeOptions(require_mps=False) で CPU 実行へ切替えてください。"
        )
    return "cpu"

def _resolve_token(opt: DiarizeOptions) -> str:
    if opt.token:
        return opt.token
    for key in opt.token_env:
        v = os.environ.get(key)
        if v:
            return v
    raise RuntimeError(
        "Hugging Face のアクセストークンが見つかりません。"
        "環境変数 HF_TOKEN / HUGGINGFACE_TOKEN / PYANNOTE_TOKEN のいずれかに設定してください。"
    )

def _load_pipeline(opt: DiarizeOptions) -> Pipeline:
    token = _resolve_token(opt)
    try:
        pipeline = Pipeline.from_pretrained(opt.model_name, token=token)  # pyannote>=3.1
    except TypeError:
        pipeline = Pipeline.from_pretrained(opt.model_name, use_auth_token=token)  # 旧署名
    dev = _resolve_device(opt)
    try:
        pipeline.to(dev)  # type: ignore[attr-defined]
    except Exception:
        pass
    return pipeline

# ------------------------------
# 実行（ファイル / 波形 / バイト）
# ------------------------------
def diarize_file(audio_path: str | Path, *, options: Optional[DiarizeOptions] = None) -> DiarizationResult:
    """音声ファイルから話者分離。"""
    opt = options or DiarizeOptions()
    pipeline = _load_pipeline(opt)
    path = str(Path(audio_path))

    kwargs = {}
    if opt.num_speakers is not None:
        kwargs["num_speakers"] = opt.num_speakers
    if opt.min_speakers is not None:
        kwargs["min_speakers"] = opt.min_speakers
    if opt.max_speakers is not None:
        kwargs["max_speakers"] = opt.max_speakers

    annotation = pipeline(path, **kwargs)
    turns = []
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        turns.append(SpeakerTurn(start=float(turn.start), end=float(turn.end), speaker=str(speaker)))
    turns.sort(key=lambda t: (t.start, t.end))

    return DiarizationResult(filename=Path(path).name, duration=None, turns=turns)

def diarize_waveform(waveform: np.ndarray, *, sample_rate: int, display_name: str = "stream", options: Optional[DiarizeOptions] = None) -> DiarizationResult:
    """波形（-1..1 の float32, mono）から話者分離。"""
    opt = options or DiarizeOptions()
    pipeline = _load_pipeline(opt)

    if waveform.ndim > 1:
        waveform = np.mean(waveform, axis=-1)

    kwargs = {}
    if opt.num_speakers is not None:
        kwargs["num_speakers"] = opt.num_speakers
    if opt.min_speakers is not None:
        kwargs["min_speakers"] = opt.min_speakers
    if opt.max_speakers is not None:
        kwargs["max_speakers"] = opt.max_speakers

    annotation = pipeline({"waveform": waveform, "sample_rate": sample_rate}, **kwargs)
    turns = []
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        turns.append(SpeakerTurn(start=float(turn.start), end=float(turn.end), speaker=str(speaker)))
    turns.sort(key=lambda t: (t.start, t.end))

    return DiarizationResult(filename=str(display_name), duration=float(len(waveform) / sample_rate), turns=turns)

def diarize_all(audio_paths: Iterable[str | Path], *, options: Optional[DiarizeOptions] = None) -> List[DiarizationResult]:
    """複数ファイルをまとめて話者分離。"""
    opt = options or DiarizeOptions()
    pipeline = _load_pipeline(opt)

    results: List[DiarizationResult] = []
    for ap in audio_paths:
        path = str(Path(ap))
        kwargs = {}
        if opt.num_speakers is not None:
            kwargs["num_speakers"] = opt.num_speakers
        if opt.min_speakers is not None:
            kwargs["min_speakers"] = opt.min_speakers
        if opt.max_speakers is not None:
            kwargs["max_speakers"] = opt.max_speakers

        annotation = pipeline(path, **kwargs)
        turns = [SpeakerTurn(start=float(t.start), end=float(t.end), speaker=str(spk))
                 for t, _, spk in annotation.itertracks(yield_label=True)]
        turns.sort(key=lambda t: (t.start, t.end))
        results.append(DiarizationResult(filename=Path(path).name, turns=turns))
    return results

def diarize_all_bytes(audio_blobs: Iterable[bytes | bytearray | memoryview], *, names: Sequence[str] | None = None, options: Optional[DiarizeOptions] = None, sample_rate: int = 16000) -> List[DiarizationResult]:
    """メモリ上の音声データをまとめて話者分離（簡易デコード付）。"""
    results: List[DiarizationResult] = []
    opt = options or DiarizeOptions()
    name_overrides = list(names or [])
    pipeline = _load_pipeline(opt)

    for idx, blob in enumerate(audio_blobs):
        audio_bytes = _coerce_to_bytes(blob)
        display_name = name_overrides[idx] if idx < len(name_overrides) else f"stream_{idx+1}"
        wav = _decode_audio_bytes(audio_bytes, sample_rate)  # float32 mono

        kwargs = {}
        if opt.num_speakers is not None:
            kwargs["num_speakers"] = opt.num_speakers
        if opt.min_speakers is not None:
            kwargs["min_speakers"] = opt.min_speakers
        if opt.max_speakers is not None:
            kwargs["max_speakers"] = opt.max_speakers

        annotation = pipeline({"waveform": wav, "sample_rate": sample_rate}, **kwargs)
        turns = [SpeakerTurn(start=float(t.start), end=float(t.end), speaker=str(spk))
                 for t, _, spk in annotation.itertracks(yield_label=True)]
        turns.sort(key=lambda t: (t.start, t.end))
        results.append(DiarizationResult(filename=display_name, duration=float(len(wav) / sample_rate), turns=turns))
    return results

# ------------------------------
# ASR への話者ラベル付与
# ------------------------------
def attach_speaker_labels(asr: TranscriptionResult, diar: DiarizationResult, *, default_label: str = "UNK") -> SpeakerAnnotatedTranscript:
    """
    ASR セグメントに、diarization の話者ラベルを重ねる。
    ルール：時間重なりの合計が最大の話者を採用。重なり無しは“最も近い区間”の話者。
    """
    turns = diar.turns
    labeled: List[SpeakerSegment] = []
    for seg in asr.segments:
        best_label = default_label
        best_overlap = 0.0
        seg_s, seg_e = float(seg.start or 0.0), float(seg.end or 0.0)
        for t in turns:
            ov = _overlap(seg_s, seg_e, t.start, t.end)
            if ov > best_overlap:
                best_overlap = ov
                best_label = t.speaker
        if best_overlap <= 0.0 and turns:
            mid = 0.5 * (seg_s + seg_e)
            best_label = min(turns, key=lambda tt: _dist(mid, 0.5 * (tt.start + tt.end))).speaker
        labeled.append(
            SpeakerSegment(
                id=getattr(seg, "id", None),
                start=seg_s,
                end=seg_e,
                text=seg.text or "",
                speaker=best_label,
            )
        )
    return SpeakerAnnotatedTranscript(filename=asr.filename, segments=labeled)

def _overlap(a_s: float, a_e: float, b_s: float, b_e: float) -> float:
    s = max(a_s, b_s); e = min(a_e, b_e)
    return max(0.0, e - s)

def _dist(x: float, y: float) -> float:
    return abs(x - y)

# ------------------------------
# 保存ユーティリティ
# ------------------------------
def save_rttm(path: str | Path, diar: DiarizationResult, *, uri: str | None = None) -> None:
    """RTTM 形式で保存。"""
    uri = uri or diar.filename
    lines: List[str] = []
    for t in diar.turns:
        dur = max(0.0, t.end - t.start)
        lines.append(f"SPEAKER {uri} 1 {t.start:.3f} {dur:.3f} <NA> <NA> {t.speaker} <NA> <NA>")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")

def save_json_diarization(path: str | Path, diar: DiarizationResult) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(diar.model_dump(), f, ensure_ascii=False, indent=2)

def save_srt_with_speaker(path: str | Path, segments: Iterable[SpeakerSegment]) -> None:
    def fmt_time(t: float) -> str:
        t = max(0.0, t)
        h = int(t // 3600); m = int((t % 3600) // 60); s = int(t % 60)
        ms = int(round((t - int(t)) * 1000))
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
    lines: List[str] = []
    for i, seg in enumerate(segments, 1):
        start = max(0.0, seg.start)
        end = max(start + 0.01, seg.end)
        lines.append(str(i))
        lines.append(f"{fmt_time(start)} --> {fmt_time(end)}")
        lines.append(f"{seg.speaker}: {seg.text}")
        lines.append("")
    Path(path).write_text("\n".join(lines), encoding="utf-8")

# ------------------------------
# 内部：デコード（bytes→wav）
# ------------------------------
def _coerce_to_bytes(blob: bytes | bytearray | memoryview) -> bytes:
    if isinstance(blob, bytes):
        return blob
    if isinstance(blob, bytearray):
        return bytes(blob)
    return bytes(memoryview(blob))

def _decode_audio_bytes(audio_bytes: bytes, sample_rate: int) -> np.ndarray:
    if not audio_bytes:
        raise ValueError("音声データが空です。")
    cmd = [
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
        "-i", "pipe:0", "-threads", "0",
        "-f", "s16le", "-ac", "1", "-acodec", "pcm_s16le", "-ar", str(sample_rate), "-"
    ]
    try:
        completed = subprocess.run(cmd, input=audio_bytes, capture_output=True, check=True)
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg が見つかりません（diarize_bytes）") from exc
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode(errors="ignore") if exc.stderr else "unknown error"
        raise RuntimeError(f"音声デコードに失敗しました（diarize_bytes）: {stderr}") from exc
    waveform = np.frombuffer(completed.stdout, dtype=np.int16).astype(np.float32)
    if waveform.size == 0:
        raise RuntimeError("音声デコード結果が空でした（diarize_bytes）。")
    return waveform / 32768.0
