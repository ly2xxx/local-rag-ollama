import json

import pytest

from agentic_rag.api import sse

pytestmark = pytest.mark.unit


def test_event_serialisation_is_valid_sse_framing():
    frame = sse.format_event(sse.TOKEN, {"delta": "hello"})
    assert frame == 'event: token\ndata: {"delta": "hello"}\n\n'
    assert frame.endswith("\n\n")
    assert frame.count("\n\n") == 1


def test_multiline_payload_is_escaped():
    # A raw newline inside `data:` would split the frame and corrupt the stream.
    frame = sse.format_event(sse.TOKEN, {"delta": "line one\nline two\r\nthree"})
    body = frame.split("\n\n")[0]
    assert len(body.split("\n")) == 2  # exactly "event:" and "data:"
    assert "\\n" in frame
    payload = json.loads(body.split("data: ", 1)[1])
    assert payload["delta"] == "line one\nline two\r\nthree"


def test_unknown_event_type_is_rejected():
    with pytest.raises(ValueError, match="Unknown SSE event"):
        sse.format_event("not_an_event", {})


def test_comment_frame_is_a_heartbeat():
    assert sse.format_comment() == ": ping\n\n"
    assert "\n\n" == sse.format_comment("a\nb")[-2:]


def test_terminal_events_are_declared():
    assert sse.TERMINAL_EVENTS == {sse.DONE, sse.ERROR, sse.INTERRUPT}


@pytest.mark.parametrize("size", [1, 4, 8, 1000])
def test_chunk_text_preserves_content(size):
    text = "The quick brown fox jumps over the lazy dog."
    assert "".join(sse.chunk_text(text, size)) == text


def test_chunk_text_prefers_word_boundaries():
    chunks = list(sse.chunk_text("alpha beta gamma", 8))
    assert len(chunks) > 1
    assert chunks[0] == "alpha "


def test_chunk_text_handles_empty_and_nonpositive_size():
    assert list(sse.chunk_text("", 8)) == []
    assert list(sse.chunk_text("abc", 0)) == ["abc"]


def test_truncate_preview_bounds_large_observations():
    preview = sse.truncate_preview("x" * 5000, 100)
    assert len(preview.encode("utf-8")) <= 100 + len("… [truncated]".encode("utf-8"))
    assert preview.endswith("[truncated]")


def test_truncate_preview_leaves_small_values_alone():
    assert sse.truncate_preview("short", 100) == "short"


def test_parse_stream_roundtrips_events():
    raw = (
        sse.format_event(sse.NODE, {"name": "agent", "status": "start"})
        + sse.format_comment("ping")
        + sse.format_event(sse.TOKEN, {"delta": "hi"})
        + sse.format_event(sse.DONE, {"answer": "hi"})
    )
    events = list(sse.parse_stream(raw))
    assert [e["event"] for e in events] == ["node", "token", "done"]
    assert events[1]["data"]["delta"] == "hi"
