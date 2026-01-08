from __future__ import annotations

import contextvars

_request_id: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")
_session_id: contextvars.ContextVar[str] = contextvars.ContextVar("session_id", default="-")


def set_request_context(request_id: str, session_id: str | None) -> tuple[contextvars.Token[str], contextvars.Token[str]]:
    tok_req = _request_id.set((request_id or "-").strip() or "-")
    tok_sess = _session_id.set((session_id or "-").strip() or "-")
    return tok_req, tok_sess


def reset_request_context(tokens: tuple[contextvars.Token[str], contextvars.Token[str]]) -> None:
    tok_req, tok_sess = tokens
    _request_id.reset(tok_req)
    _session_id.reset(tok_sess)


def get_request_id() -> str:
    return _request_id.get()


def get_session_id() -> str:
    return _session_id.get()
