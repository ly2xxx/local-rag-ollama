"""Server-Sent Events envelope (DESIGN.md §5.2).

One event type per message; `data` is always a single-line JSON object, because
a raw newline inside `data:` would split the frame.
"""

import json
from typing import Any, Dict, Iterable, Iterator, List, Optional

TOKEN = "token"
NODE = "node"
TOOL_CALL = "tool_call"
OBSERVATION = "observation"
CITATION = "citation"
INTERRUPT = "interrupt"
DONE = "done"
ERROR = "error"

KNOWN_EVENTS = frozenset(
    {TOKEN, NODE, TOOL_CALL, OBSERVATION, CITATION, INTERRUPT, DONE, ERROR}
)

TERMINAL_EVENTS = frozenset({DONE, ERROR, INTERRUPT})


def format_event(event: str, data: Dict[str, Any]) -> str:
    """Serialises one SSE frame.

    `json.dumps` escapes newlines, so the payload can never break the framing.
    """
    if event not in KNOWN_EVENTS:
        raise ValueError(f"Unknown SSE event type: {event!r}")
    payload = json.dumps(data, ensure_ascii=False, default=str)
    return f"event: {event}\ndata: {payload}\n\n"


def format_comment(text: str = "ping") -> str:
    """SSE comment frame — keeps idle proxies from dropping the connection."""
    safe = text.replace("\n", " ")
    return f": {safe}\n\n"


def chunk_text(text: str, size: int) -> Iterator[str]:
    """Splits text into ~`size`-char pieces, preferring whitespace boundaries."""
    if not text:
        return
    if size <= 0:
        yield text
        return

    start = 0
    length = len(text)
    while start < length:
        end = min(start + size, length)
        if end < length:
            boundary = text.rfind(" ", start + 1, end + 1)
            if boundary > start:
                end = boundary + 1
        yield text[start:end]
        start = end


def truncate_preview(value: Any, max_bytes: int) -> str:
    """Bounds an observation preview so a large tool result cannot flood the stream."""
    text = value if isinstance(value, str) else str(value)
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text
    return encoded[:max_bytes].decode("utf-8", errors="ignore") + "… [truncated]"


def parse_stream(raw: str) -> Iterable[Dict[str, Any]]:
    """Parses an SSE body back into events. Test helper, also useful for clients."""
    events: List[Dict[str, Any]] = []
    for frame in raw.split("\n\n"):
        frame = frame.strip("\n")
        if not frame or frame.startswith(":"):  # blank or comment/heartbeat
            continue

        event_name: Optional[str] = None
        data_lines: List[str] = []
        for line in frame.split("\n"):
            if line.startswith("event: "):
                event_name = line.removeprefix("event: ")
            elif line.startswith("data: "):
                data_lines.append(line.removeprefix("data: "))

        if event_name is None:
            continue
        raw_data = "\n".join(data_lines)
        events.append(
            {"event": event_name, "data": json.loads(raw_data) if raw_data else {}}
        )
    return events
