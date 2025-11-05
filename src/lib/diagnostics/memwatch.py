from __future__ import annotations

"""常時稼働用の軽量メモリ監視ユーティリティ。

`MEM_WATCH` を有効化するとバックグラウンドスレッドが一定間隔で RSS を計測し、
閾値に応じてログへ通知する。`MEM_DIAG` の詳細スナップショットと比べると
最低限の計測だけを行うため、書き起こし処理の前後に挟んでもオーバーヘッドを
感じにくいのが特徴。しきい値超過時に限定して GC を挟めるため、緩やかな
リーク検知とメモリ回復のきっかけづくりにも利用できる。
"""

import gc
import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

logger = logging.getLogger("diag.memwatch")


def _env_flag(name: str, *, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "on", "yes"}


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def _format_bytes(num: int | float | None) -> str:
    if num is None or num < 0:
        return "0B"
    units = ("B", "KB", "MB", "GB", "TB")
    size = float(num)
    index = 0
    while size >= 1024.0 and index < len(units) - 1:
        size /= 1024.0
        index += 1
    return f"{size:.2f}{units[index]}"


def _rss_bytes() -> Optional[int]:
    try:
        import psutil  # type: ignore

        return int(psutil.Process().memory_info().rss)
    except Exception:
        try:
            import resource  # type: ignore

            usage = resource.getrusage(resource.RUSAGE_SELF)
            raw = usage.ru_maxrss
            return int(raw if raw > 1e12 else raw * 1024)
        except Exception:
            return None


@dataclass(frozen=True)
class MemoryWatchConfig:
    enabled: bool = False
    interval_seconds: float = 60.0
    warmup_seconds: float = 5.0
    warn_threshold_bytes: int = 0
    critical_threshold_bytes: int = 0
    gc_on_warning: bool = False
    log_level: int = logging.INFO
    emit_debug_delta: bool = False
    include_children: bool = False

    @classmethod
    def from_env(cls) -> "MemoryWatchConfig":
        enabled = _env_flag("MEM_WATCH", default=False)
        interval = max(_env_float("MEM_WATCH_INTERVAL", 60.0), 0.0)
        warmup = max(_env_float("MEM_WATCH_WARMUP", 5.0), 0.0)
        warn_mb = max(_env_float("MEM_WATCH_WARN_MB", 0.0), 0.0)
        critical_mb = max(_env_float("MEM_WATCH_CRITICAL_MB", 0.0), 0.0)
        gc_on_warning = _env_flag("MEM_WATCH_GC", default=False)
        emit_debug_delta = _env_flag("MEM_WATCH_DEBUG_DELTA", default=False)
        include_children = _env_flag("MEM_WATCH_INCLUDE_CHILDREN", default=False)
        level_value = _env_int("MEM_WATCH_LOG_LEVEL", logging.INFO)

        return cls(
            enabled=enabled and interval > 0.0,
            interval_seconds=interval,
            warmup_seconds=warmup,
            warn_threshold_bytes=int(warn_mb * 1024 * 1024) if warn_mb > 0 else 0,
            critical_threshold_bytes=int(critical_mb * 1024 * 1024) if critical_mb > 0 else 0,
            gc_on_warning=gc_on_warning,
            log_level=level_value,
            emit_debug_delta=emit_debug_delta,
            include_children=include_children,
        )


class MemoryWatchdog:
    """対象プロセスの RSS を定期的に記録する。"""

    def __init__(self, config: MemoryWatchConfig, *, sampler: Callable[[], Optional[int]] = _rss_bytes) -> None:
        self.config = config
        self._sampler = sampler
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_value: int | None = None
        self._peak_value: int = 0
        self._started_at: float = time.time()

    def start(self) -> bool:
        if self._thread and self._thread.is_alive():
            return False

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="memory-watchdog", daemon=True)
        self._thread.start()
        logger.log(
            self.config.log_level,
            "memory_watchdog_started: interval=%ss warmup=%ss warn=%s critical=%s gc_on_warn=%s",
            self.config.interval_seconds,
            self.config.warmup_seconds,
            _format_bytes(self.config.warn_threshold_bytes or 0),
            _format_bytes(self.config.critical_threshold_bytes or 0),
            self.config.gc_on_warning,
        )
        return True

    def stop(self, *, timeout: float = 1.0) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)

    def _run(self) -> None:
        if self.config.warmup_seconds > 0:
            if self._stop_event.wait(self.config.warmup_seconds):
                return
        while not self._stop_event.wait(self.config.interval_seconds):
            self._sample()

    def _sample(self) -> None:
        rss = self._sampler()
        if rss is None:
            logger.debug("memory_watchdog_sampler_unavailable")
            return

        measured_rss = rss
        rss_self = rss
        rss_children = 0

        # 子プロセスを合算したい場合は psutil で集計する（利用可能な時のみ）。
        if self.config.include_children:
            try:
                import psutil  # type: ignore

                proc = psutil.Process()
                children = proc.children(recursive=True)
                for ch in children:
                    try:
                        rss_children += int(ch.memory_info().rss)
                    except Exception:
                        pass
                measured_rss = rss_self + rss_children
            except Exception:
                # psutil が無い/失敗時は自己 RSS のみ
                measured_rss = rss

        previous = self._last_value
        self._last_value = measured_rss
        self._peak_value = max(self._peak_value, measured_rss)

        delta = measured_rss - previous if previous is not None else 0
        delta_text = _format_bytes(delta)
        now = time.time()
        uptime = now - self._started_at

        level = self.config.log_level
        message = "memory_usage"
        extra: dict[str, object] = {
            "rss": measured_rss,
            "rss_h": _format_bytes(measured_rss),
            "delta": delta,
            "delta_h": delta_text,
            "peak": self._peak_value,
            "peak_h": _format_bytes(self._peak_value),
            "uptime_sec": int(uptime),
        }

        if self.config.include_children:
            extra.update(
                {
                    "rss_self": rss_self,
                    "rss_self_h": _format_bytes(rss_self),
                    "rss_children": rss_children,
                    "rss_children_h": _format_bytes(rss_children),
                }
            )

        if self.config.critical_threshold_bytes and measured_rss >= self.config.critical_threshold_bytes:
            level = logging.ERROR
            message = "memory_usage_critical"
        elif self.config.warn_threshold_bytes and measured_rss >= self.config.warn_threshold_bytes:
            level = logging.WARNING
            message = "memory_usage_warning"

        if level >= logging.WARNING and self.config.gc_on_warning:
            collected = gc.collect()
            post_gc = self._sampler() or measured_rss
            extra.update({
                "gc_collected": collected,
                "rss_after_gc": post_gc,
                "rss_after_gc_h": _format_bytes(post_gc),
            })

        if level >= logging.WARNING or self.config.emit_debug_delta:
            logger.log(level, "%s: %s", message, extra)
        else:
            if self.config.include_children:
                logger.log(
                    level,
                    "%s: rss=%s (self=%s children=%s) delta=%s peak=%s",
                    message,
                    extra["rss_h"],
                    extra.get("rss_self_h", "0B"),
                    extra.get("rss_children_h", "0B"),
                    delta_text,
                    extra["peak_h"],
                )
            else:
                logger.log(level, "%s: rss=%s delta=%s peak=%s", message, extra["rss_h"], delta_text, extra["peak_h"])


_WATCHDOG: MemoryWatchdog | None = None
_WATCHDOG_LOCK = threading.Lock()


def ensure_memory_watchdog() -> MemoryWatchdog | None:
    """環境変数に応じてウォッチャーを起動する。"""

    global _WATCHDOG

    config = MemoryWatchConfig.from_env()
    if not config.enabled:
        return None

    with _WATCHDOG_LOCK:
        if _WATCHDOG is None:
            watchdog = MemoryWatchdog(config)
            try:
                watchdog.start()
            except Exception:
                logger.exception("memory_watchdog_start_failed")
                return None
            _WATCHDOG = watchdog
        return _WATCHDOG


__all__ = ["MemoryWatchConfig", "MemoryWatchdog", "ensure_memory_watchdog"]
