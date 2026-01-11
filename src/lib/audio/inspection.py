from __future__ import annotations

import json
import logging
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from uuid import uuid4
from subprocess import CalledProcessError, run

import numpy as np

from mlx_whisper.audio import SAMPLE_RATE

from .utils import AudioDecodeError, decode_audio_bytes

logger = logging.getLogger(__name__)


_SILENCE_CACHE: dict[Path, bool] = {}


def _resolve_cache_key(path: Path) -> Path:
    try:
        return path.resolve()
    except OSError:
        return path


def _get_cached_silence(path: Path) -> Optional[bool]:
    key = _resolve_cache_key(path)
    return _SILENCE_CACHE.get(key)


def _set_cached_silence(path: Path, silent: bool) -> None:
    key = _resolve_cache_key(path)
    _SILENCE_CACHE[key] = silent


@dataclass(slots=True)
class PreparedAudio:
    path: Path
    display_name: str
    silent: bool


class InvalidAudioError(Exception):
    """音声バリデーション失敗時に送出される例外。"""

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


def infer_suffix(filename: Optional[str]) -> str:
    if not filename:
        return ".tmp"
    suffix = Path(filename).suffix
    return suffix if suffix else ".tmp"


def prepare_audio(temp_path: Path, original_name: Optional[str]) -> PreparedAudio:
    validate_audio_file(temp_path, original_name)
    dump_audio_for_debug(temp_path, original_name)
    silent = is_silent_audio(temp_path)
    display_name = original_name or temp_path.name
    return PreparedAudio(path=temp_path, display_name=display_name, silent=silent)


def validate_audio_file(path: Path, original_name: Optional[str]) -> None:
    if not path.exists():
        raise InvalidAudioError("音声ファイルを一時ディスクへ保存できませんでした")
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise InvalidAudioError("音声ファイルを読み取れませんでした") from exc

    if size <= 0:
        raise InvalidAudioError("音声ファイルが空です")

    display_name = original_name or path.name

    try:
        metadata = _probe_audio_metadata(path)
    except FileNotFoundError:
        metadata = None
        logger.warning("ffprobe が見つかりません。音声検証を簡易チェックにフォールバックします。")
    except InvalidAudioError:
        raise

    if metadata is None:
        _decode_audio_bytes(path)
        logger.debug("transcribe_validation: %s (fallback decode)", display_name)
        return

    duration = metadata.get("duration")
    if duration is not None and duration <= 0:
        raise InvalidAudioError("音声の長さが0秒と判定されました")

    logger.debug(
        "transcribe_validation: %s codec=%s channels=%s sample_rate=%s duration=%.3f",
        display_name,
        metadata.get("codec"),
        metadata.get("channels"),
        metadata.get("sample_rate"),
        duration or -1.0,
    )


def is_silent_audio(path: Path, *, threshold: float = 1.5e-3) -> bool:
    cached = _get_cached_silence(path)
    if cached is not None:
        return cached

    try:
        waveform = _decode_audio_bytes(path)
    except InvalidAudioError:
        _set_cached_silence(path, False)
        return False

    if waveform.size == 0:
        _set_cached_silence(path, True)
        return True

    energy = float(np.mean(np.abs(waveform)))
    peak = float(np.max(np.abs(waveform))) if waveform.size else 0.0
    logger.debug(
        "transcribe_silence_metrics: samples=%d energy=%.6f peak=%.6f",
        waveform.size,
        energy,
        peak,
    )
    silent = energy < threshold and peak < threshold * 5
    _set_cached_silence(path, silent)
    return silent


def dump_audio_for_debug(path: Path, original_name: Optional[str]) -> Optional[Path]:
    enabled = (os.getenv("ASR_DUMP_AUDIO") or "").strip().lower() in {"1", "true", "yes", "on"}
    if not enabled:
        return None

    raw_dir = (os.getenv("ASR_DUMP_AUDIO_DIR") or "").strip()
    dest_dir = Path(raw_dir) if raw_dir else Path.cwd()
    try:
        dest_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        logger.debug("transcribe_debug_dump_dir_failed: %s", dest_dir, exc_info=True)
        return None

    stem = Path(original_name).stem if original_name else path.stem
    suffix = Path(original_name).suffix if original_name else path.suffix
    if not suffix:
        suffix = path.suffix or ".wav"

    safe_stem = stem.replace("/", "_").replace("\\", "_") or "audio"
    dest = dest_dir / f"debug_{safe_stem}_{uuid4().hex}{suffix}"

    try:
        shutil.copy2(path, dest)
        logger.debug("transcribe_debug_dump: %s", dest)
        return dest
    except OSError:
        logger.debug("transcribe_debug_dump_failed: %s", dest, exc_info=True)
        return None


def _decode_audio_bytes(path: Path) -> np.ndarray:
    try:
        audio_bytes = path.read_bytes()
    except OSError as exc:
        raise InvalidAudioError("音声ファイルの読み込みに失敗しました") from exc
    try:
        return decode_audio_bytes(audio_bytes, sample_rate=SAMPLE_RATE)
    except AudioDecodeError as exc:
        raise InvalidAudioError("音声ファイルをデコードできませんでした") from exc


def _probe_audio_metadata(path: Path) -> dict[str, float | int | str | None]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_streams",
        "-show_format",
        "-select_streams",
        "a:0",
        str(path),
    ]

    try:
        completed = run(cmd, capture_output=True, text=True, check=True)
    except FileNotFoundError as exc:
        raise FileNotFoundError from exc
    except CalledProcessError as exc:
        stderr = exc.stderr.strip() if exc.stderr else "unknown"
        raise InvalidAudioError(f"音声ファイルの解析に失敗しました: {stderr}") from exc

    if not completed.stdout:
        raise InvalidAudioError("音声ストリーム情報を取得できませんでした")

    try:
        probe = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise InvalidAudioError("音声メタデータの解析に失敗しました") from exc

    streams = probe.get("streams") or []
    if not streams:
        raise InvalidAudioError("音声トラックが含まれていません")

    stream = streams[0]

    def _to_float(value: object) -> float | None:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _to_int(value: object) -> int | None:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    duration = _to_float(stream.get("duration")) or _to_float(probe.get("format", {}).get("duration"))
    sample_rate = _to_int(stream.get("sample_rate"))
    channels = _to_int(stream.get("channels"))

    return {
        "codec": stream.get("codec_name"),
        "channels": channels,
        "sample_rate": sample_rate,
        "duration": duration,
    }


__all__ = [
    "InvalidAudioError",
    "PreparedAudio",
    "dump_audio_for_debug",
    "infer_suffix",
    "is_silent_audio",
    "prepare_audio",
    "validate_audio_file",
]
