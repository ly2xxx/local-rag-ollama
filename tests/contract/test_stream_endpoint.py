import time

import pytest

from agentic_rag.api import sse
from agentic_rag.api.errors import AppError, ErrorCode

pytestmark = pytest.mark.contract


async def _collect(client, headers, payload=None, settings_note=None):
    payload = payload or {"query": "hello"}
    chunks = []
    async with client.stream(
        "POST", "/api/v1/chat/stream", json=payload, headers=headers
    ) as response:
        status = response.status_code
        content_type = response.headers.get("content-type", "")
        async for chunk in response.aiter_text():
            chunks.append(chunk)
    return status, content_type, "".join(chunks)


async def test_stream_emits_tokens_then_done(client, auth_a):
    status, content_type, raw = await _collect(client, auth_a)

    assert status == 200
    assert content_type.startswith("text/event-stream")

    events = list(sse.parse_stream(raw))
    names = [e["event"] for e in events]

    assert names[0] == "node"
    assert names[-1] == "done"
    assert "token" in names
    assert names.index("token") < names.index("done")

    reassembled = "".join(e["data"]["delta"] for e in events if e["event"] == "token")
    assert reassembled == "The answer is 42."
    assert events[-1]["data"]["answer"] == "The answer is 42."
    assert events[-1]["data"]["trace_id"]


async def test_stream_emits_tool_events_before_tokens(client, auth_a, fake_service):
    fake_service.scratchpad = [
        {"type": "tool_call", "name": "query_document_knowledge_base", "args": {"query": "x"}, "id": "c1"},
        {"type": "tool_observation", "name": "query_document_knowledge_base",
         "content": "match text", "tool_call_id": "c1"},
    ]
    _, _, raw = await _collect(client, auth_a)
    names = [e["event"] for e in sse.parse_stream(raw)]

    assert names.index("tool_call") < names.index("token")
    assert names.index("observation") < names.index("token")


async def test_observation_preview_is_truncated(client, auth_a, fake_service, settings):
    fake_service.scratchpad = [
        {"type": "tool_observation", "name": "t", "content": "y" * 50_000, "tool_call_id": "c1"}
    ]
    _, _, raw = await _collect(client, auth_a)
    observation = next(e for e in sse.parse_stream(raw) if e["event"] == "observation")

    assert len(observation["data"]["preview"]) < 50_000
    assert observation["data"]["preview"].endswith("[truncated]")


async def test_stream_emits_error_event_when_llm_unavailable(client, auth_a, fake_service):
    fake_service.error = AppError(ErrorCode.LLM_UNAVAILABLE)

    status, _, raw = await _collect(client, auth_a)
    events = list(sse.parse_stream(raw))

    # The status line is already sent, so the failure has to arrive in-band.
    assert status == 200
    assert events[-1]["event"] == "error"
    assert events[-1]["data"]["code"] == "LLM_UNAVAILABLE"
    assert events[-1]["data"]["retryable"] is True
    assert "token" not in [e["event"] for e in events]


async def test_stream_error_event_does_not_leak_internals(client, auth_a, fake_service):
    fake_service.error = RuntimeError("secret-value-abc at C:/internal.py")

    _, _, raw = await _collect(client, auth_a)
    events = list(sse.parse_stream(raw))

    assert events[-1]["event"] == "error"
    assert events[-1]["data"]["code"] == "INTERNAL"
    assert "secret-value-abc" not in raw


async def test_stream_sends_heartbeat_when_idle(settings, fake_service, auth_a):
    from httpx import ASGITransport, AsyncClient

    from agentic_rag.api.main import create_app

    settings.sse_heartbeat_seconds = 0.05
    fake_service.delay = 0.35
    app = create_app(settings=settings, service=fake_service)

    transport = ASGITransport(app=app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        _, _, raw = await _collect(ac, auth_a)

    assert ": ping" in raw
    # The turn still completes normally after the heartbeats.
    assert [e["event"] for e in sse.parse_stream(raw)][-1] == "done"


async def test_stream_requires_authentication(client):
    response = await client.post("/api/v1/chat/stream", json={"query": "hi"})

    assert response.status_code == 401
    assert response.json()["error"]["code"] == "UNAUTHENTICATED"


async def test_stream_rejects_invalid_body_before_streaming(client, auth_a):
    response = await client.post("/api/v1/chat/stream", json={"query": ""}, headers=auth_a)

    assert response.status_code == 400
    assert response.headers["content-type"].startswith("application/json")


async def test_stream_sets_no_buffering_headers(client, auth_a):
    async with client.stream(
        "POST", "/api/v1/chat/stream", json={"query": "hi"}, headers=auth_a
    ) as response:
        assert response.headers["cache-control"] == "no-cache"
        assert response.headers["x-accel-buffering"] == "no"
        await response.aread()


async def test_frames_are_produced_before_the_turn_completes(settings, fake_service):
    """The generator must yield as it goes, not build one block at the end.

    Asserted against the generator itself because `httpx.ASGITransport` buffers
    the whole response body — arrival timing over a real socket is covered by
    tests/integration/test_streaming_live.py.
    """
    from agentic_rag.api.deps import Principal
    from agentic_rag.api.routes.chat import _stream_turn
    from agentic_rag.api.schemas import ChatRequest

    settings.sse_heartbeat_seconds = 0.05
    fake_service.delay = 0.3

    stream = _stream_turn(
        fake_service,
        Principal(api_key_id="k", tenant_id="tenant_a"),
        ChatRequest(query="hi"),
        settings,
        "trace-1",
    )

    start = time.perf_counter()
    first = await stream.__anext__()
    first_at = time.perf_counter() - start

    # The opening frame lands long before the agent's 0.3s of work is done.
    assert first.startswith("event: node")
    assert first_at < 0.15

    heartbeat_at = None
    frames = [first]
    async for frame in stream:
        frames.append(frame)
        if heartbeat_at is None and frame.startswith(": ping"):
            heartbeat_at = time.perf_counter() - start

    assert heartbeat_at is not None and heartbeat_at < 0.3
    assert frames[-1].startswith("event: done")
