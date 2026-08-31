"""Request correlation: one trace_id per request, visible in logs and responses."""

import re
import secrets
import time
from contextvars import ContextVar, Token
from typing import Optional

_TRACE_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

_trace_id: ContextVar[str] = ContextVar("trace_id", default="-")


def new_trace_id() -> str:
    """Lexicographically sortable id: millisecond timestamp + random suffix.

    ULID-like without taking a dependency; sorting by id sorts by arrival time,
    which is what makes it useful when grepping logs.
    """
    return f"{int(time.time() * 1000):012x}{secrets.token_hex(8)}"


def is_valid_trace_id(value: str) -> bool:
    return bool(value and _TRACE_ID_PATTERN.match(value))


def set_trace_id(value: str) -> Token:
    return _trace_id.set(value)


def reset_trace_id(token: Token) -> None:
    _trace_id.reset(token)


def get_trace_id() -> str:
    return _trace_id.get()


def resolve_inbound_trace_id(header_value: Optional[str]) -> str:
    """Honours a caller-supplied id when it is well-formed, else mints one."""
    if header_value and is_valid_trace_id(header_value):
        return header_value
    return new_trace_id()
