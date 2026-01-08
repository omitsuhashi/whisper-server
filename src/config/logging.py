from __future__ import annotations

import logging
import os

from src.lib.diagnostics.request_context import get_request_id, get_session_id

_LEVEL_ALIASES: dict[str, int] = {
    "WARN": logging.WARNING,
    "TRACE": logging.DEBUG,
}


class RequestContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = get_request_id()
        record.session_id = get_session_id()
        return True


def _normalize(value: str | int | None) -> int:
    if isinstance(value, int):
        return value

    if value is None:
        return logging.INFO

    text = value.strip()
    if not text:
        return logging.INFO

    try:
        return int(text)
    except ValueError:
        pass

    upper = text.upper()
    if upper in _LEVEL_ALIASES:
        return _LEVEL_ALIASES[upper]

    return getattr(logging, upper, logging.INFO)


def resolve_log_level(*candidates: str | int | None, default: str | int | None = None) -> int:
    for candidate in candidates:
        if candidate is None:
            continue
        return _normalize(candidate)

    if default is not None:
        return _normalize(default)

    return logging.INFO


def setup_logging(
    level: str | int | None = None,
    *,
    env_key: str = "LOG_LEVEL",
    default: str | int | None = logging.INFO,
) -> int:
    resolved = resolve_log_level(level, os.getenv(env_key), default=default)
    fmt = os.getenv("LOG_FORMAT") or "%(asctime)s %(levelname)s %(name)s [req=%(request_id)s sess=%(session_id)s] %(message)s"
    logging.basicConfig(level=resolved, force=True, format=fmt)
    root = logging.getLogger()
    root.setLevel(resolved)
    for handler in root.handlers:
        handler.addFilter(RequestContextFilter())
    return resolved
