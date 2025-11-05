from __future__ import annotations

import gc
import json
import logging
import os
import time
import tracemalloc
from dataclasses import dataclass
from typing import Any, Dict

logger = logging.getLogger("diag.mem")


def _enabled() -> bool:
    return os.getenv("MEM_DIAG", "0").lower() in {"1", "true", "on", "yes"}


def _format_bytes(n: int | float | None) -> str:
    if not n or n < 0:
        return "0B"
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(n)
    i = 0
    while size >= 1024.0 and i < len(units) - 1:
        size /= 1024.0
        i += 1
    return f"{size:.2f}{units[i]}"


def _rss_bytes() -> int | None:
    try:
        import psutil  # type: ignore

        total = int(psutil.Process().memory_info().rss)
        # 子プロセスも含めたい場合は環境変数で切り替え
        if os.getenv("MEM_DIAG_INCLUDE_CHILDREN", "0").lower() in {"1", "true", "on", "yes"}:
            try:
                proc = psutil.Process()
                for ch in proc.children(recursive=True):
                    try:
                        total += int(ch.memory_info().rss)
                    except Exception:
                        pass
            except Exception:
                pass
        return total
    except Exception:
        try:
            import resource  # type: ignore

            usage = resource.getrusage(resource.RUSAGE_SELF)
            ru = usage.ru_maxrss
            # macOS: bytes, Linux: kilobytes
            return int(ru if ru > 1e12 else ru * 1024)
        except Exception:
            return None


def _mps_bytes() -> Dict[str, int] | None:
    try:
        import torch  # type: ignore

        out: Dict[str, int] = {}
        for name in ("current_allocated_memory", "driver_allocated_memory", "reserved_memory"):
            try:
                fn = getattr(torch.mps, name, None)  # type: ignore[attr-defined]
                if callable(fn):
                    out[name] = int(fn())  # type: ignore[misc]
            except Exception:
                pass
        return out or None
    except Exception:
        return None


_TRACEMALLOC_STARTED = False


def ensure_tracemalloc() -> None:
    global _TRACEMALLOC_STARTED
    if not _enabled():
        return
    if not _TRACEMALLOC_STARTED:
        try:
            tracemalloc.start(25)
            _TRACEMALLOC_STARTED = True
        except Exception:
            _TRACEMALLOC_STARTED = False


def snapshot(tag: str, **extra: Any) -> None:
    """メモリ状況をログに出す（MEM_DIAG=1 で有効）。"""

    if not _enabled():
        return

    ensure_tracemalloc()
    rss = _rss_bytes()
    mps = _mps_bytes()
    top = []
    try:
        if tracemalloc.is_tracing():
            snap = tracemalloc.take_snapshot()
            stats = snap.statistics("lineno")
            for s in stats[:10]:
                top.append({"trace": str(s.traceback[0]), "size": s.size, "count": s.count})
    except Exception:
        pass

    payload = {
        "ts": time.time(),
        "tag": tag,
        "rss": rss,
        "rss_h": _format_bytes(rss or 0),
        "mps": {k: v for k, v in (mps or {}).items()},
        "mps_h": {k: _format_bytes(v) for k, v in (mps or {}).items()},
        **extra,
        "gc_counts": {"gen0": gc.get_count()[0], "gen1": gc.get_count()[1], "gen2": gc.get_count()[2]},
        "top_allocs": top,
    }
    try:
        logger.debug("MEM_DIAG %s", json.dumps(payload, ensure_ascii=False))
    except Exception:
        logger.debug("MEM_DIAG tag=%s rss=%s", tag, payload.get("rss_h"))


__all__ = ["snapshot", "ensure_tracemalloc"]
