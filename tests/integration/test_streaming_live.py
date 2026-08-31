"""Streaming over a real socket, through a real uvicorn server.

`httpx.ASGITransport` buffers the response body, so it cannot prove that frames
reach a client incrementally. This is the test that backs the Phase 1 DoD item
"curl -N prints tokens incrementally, not one block at the end".
"""

import socket
import threading
import time

import pytest
import uvicorn
from httpx import AsyncClient

from agentic_rag.api import sse
from agentic_rag.api.main import create_app
from agentic_rag.settings import Settings
from tests.fakes.service import FakeAgentService

pytestmark = pytest.mark.integration

API_KEY = "key-tenant-a"


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def live_server():
    """Runs the app under uvicorn on a free port for the duration of one test."""
    service = FakeAgentService(answer="Streaming works end to end, one chunk at a time.")
    service.delay = 0.4
    settings = Settings(
        _env_file=None,
        api_keys=f"{API_KEY}:tenant_a",
        warm_embeddings_on_startup=False,
        sse_heartbeat_seconds=0.1,
        sse_token_chunk_chars=8,
    )
    app = create_app(settings=settings, service=service)

    port = _free_port()
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    deadline = time.time() + 15
    while not server.started and time.time() < deadline:
        time.sleep(0.05)
    if not server.started:
        server.should_exit = True
        pytest.fail("uvicorn did not start within 15s")

    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=10)


async def test_stream_arrives_incrementally_over_a_real_socket(live_server):
    arrivals = []

    async with AsyncClient(base_url=live_server, timeout=30.0) as client:
        start = time.perf_counter()
        async with client.stream(
            "POST",
            "/api/v1/chat/stream",
            json={"query": "hello"},
            headers={"X-API-Key": API_KEY},
        ) as response:
            assert response.status_code == 200
            assert response.headers["content-type"].startswith("text/event-stream")
            body = []
            async for chunk in response.aiter_text():
                arrivals.append((time.perf_counter() - start, chunk))
                body.append(chunk)

    # More than one network read, and the first landed before the agent finished
    # its 0.4s of work — i.e. this is a stream, not a buffered response.
    assert len(arrivals) > 1, "response arrived as a single block"
    assert arrivals[0][0] < 0.35
    assert arrivals[-1][0] > arrivals[0][0]

    events = list(sse.parse_stream("".join(body)))
    assert events[0]["event"] == "node"
    assert events[-1]["event"] == "done"
    reassembled = "".join(e["data"]["delta"] for e in events if e["event"] == "token")
    assert reassembled == "Streaming works end to end, one chunk at a time."


async def test_heartbeats_reach_the_client_while_the_agent_works(live_server):
    async with AsyncClient(base_url=live_server, timeout=30.0) as client:
        async with client.stream(
            "POST",
            "/api/v1/chat/stream",
            json={"query": "hello"},
            headers={"X-API-Key": API_KEY},
        ) as response:
            raw = "".join([chunk async for chunk in response.aiter_text()])

    assert ": ping" in raw


async def test_healthz_over_a_real_socket(live_server):
    async with AsyncClient(base_url=live_server, timeout=10.0) as client:
        response = await client.get("/healthz")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.headers["X-Trace-Id"]
