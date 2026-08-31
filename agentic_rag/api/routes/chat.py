"""Chat endpoints: buffered JSON and SSE streaming.

====================================================================================================
SUMMARY COMPARISON: Buffered (/chat) vs. Streaming (/chat/stream)
====================================================================================================
| Feature / Concept   | Buffered Endpoint (/chat)            | Streaming Endpoint (/chat/stream)             |
|---------------------|--------------------------------------|------------------------------------------------|
| Return Type         | ChatResponse (application/json)      | StreamingResponse (text/event-stream)          |
| Execution Model     | Single-shot `await`, builds and      | Async generator `yield`, pushes event sequence |
|                     | returns full JSON once turn finishes | incrementally in real-time                     |
| Error Handling      | Standard HTTP 4xx/500 JSON body      | In-stream `event: error` SSE frame (HTTP 200)  |
| UX / Client Feel    | Wait spinner -> Instant bulk answer  | Typewriter effect + live thoughts/tool calls   |
| Reverse Proxy Setup | Standard response headers            | Cache-Control: no-cache, X-Accel-Buffering: no |
====================================================================================================
"""

import asyncio
import logging
import time
from typing import Any, AsyncIterator, Dict, List, Optional, Sequence

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from ...settings import Settings
from .. import sse
from ..deps import Principal, get_principal, get_service, get_settings
from ..errors import AppError, ErrorCode, default_message_for, error_body
from ..schemas import ChatRequest, ChatResponse, ErrorResponse, ScratchpadEntry
from ..service import AgentService
from ..tracing import get_trace_id

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["chat"])

_ERROR_RESPONSES = {
    400: {"model": ErrorResponse},
    401: {"model": ErrorResponse},
    500: {"model": ErrorResponse},
    503: {"model": ErrorResponse},
}


# #### 1.5 - Defensive Parsing & Fault Tolerance
# - Related Step: Step 1 (schemas.py) & Error Handling
# - Core Concept: Unpack dictionary entries (**item) into Pydantic models with try-except,
#   ensuring unparseable debug or trajectory entries do not crash the user's request.
def _to_entries(
    scratchpad: Optional[Sequence[Dict[str, Any]]],
) -> List[ScratchpadEntry]:
    entries: List[ScratchpadEntry] = []
    for item in scratchpad or []:
        try:
            entries.append(ScratchpadEntry(**item))
        except Exception:  # a malformed trajectory entry must not fail the turn
            logger.debug("Skipping unparseable scratchpad entry: %r", item)
    return entries


# #### 1.1 - Route Decorator & API Contract Definition
# - Related Step: Step 2 (main.py / routes) & Step 1 (schemas.py) & Step 3 (errors.py)
# - Core Concept: Declare endpoint metadata, automatic response filtering (response_model=ChatResponse),
#   and standardized error schemas for OpenAPI / Swagger documentation.
@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Ask the agent (buffered)",
    responses=_ERROR_RESPONSES,
)
# #### 1.2 - Request Body Validation & Dependency Injection
# - Related Step: Step 1 (schemas.py) & Step 3 (deps.py)
# - Core Concept: Pydantic deserialization + validation on `payload`, and FastAPI `Depends`
#   for authentication (Principal) and service injection (AgentService).
async def chat(
    payload: ChatRequest,
    principal: Principal = Depends(get_principal),
    service: AgentService = Depends(get_service),
) -> ChatResponse:
    started = time.perf_counter()
    # #### 1.3 - Asynchronous Non-blocking Invocation
    # - Related Step: Step 2 (service.py) & Async/Await
    # - Core Concept: Yields control back to the asyncio event loop while waiting for LLM / I/O,
    #   preventing Worker threads from blocking concurrent requests.
    result = await service.ask(
        tenant_id=principal.tenant_id,
        thread_id=payload.thread_id,
        query=payload.query,
    )
    # #### 1.4 - Response Construction & Distributed Tracing
    # - Related Step: Step 1 (schemas.py) & Step 3 (middleware.py / tracing.py)
    # - Core Concept: Pack result into Pydantic model with ContextVar-backed trace_id,
    #   high-precision latency calculation, and degraded fallback flags.
    return ChatResponse(
        answer=result.answer,
        thread_id=payload.thread_id,
        trace_id=get_trace_id(),
        scratchpad=_to_entries(result.scratchpad),
        latency_ms=int((time.perf_counter() - started) * 1000),
        degraded=result.degraded,
    )


# #### 1.6 - Gateway Keep-Alive & Task Shielding
# - Related Step: Step 4 (sse.py) & Async Programming
# - Core Concept: Emits SSE comment `: ping\n\n` periodically to prevent reverse-proxy timeouts.
#   Uses `asyncio.shield` so the agent task won't be cancelled when the heartbeat timer triggers.
async def _heartbeat_until_done(
    task: asyncio.Future, interval: float
) -> AsyncIterator[str]:
    """Yields an SSE comment every `interval` seconds while `task` runs.

    `shield` keeps the agent running when the heartbeat timer fires. Whatever
    the task raises propagates to the caller.
    """
    while not task.done():
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=interval)
        except asyncio.TimeoutError:
            yield sse.format_comment("ping")


# #### 1.7 - SSE Event Lifecycle & In-Stream Error Handling
# - Related Step: Step 4 (sse.py) & Step 3 (errors.py)
# - Core Concept: Async generator orchestrating lifecycle events (start -> tool_calls -> tokens -> done).
#   Since HTTP 200 is already committed, errors are delivered as `event: error` SSE payloads.
async def _stream_turn(
    service: AgentService,
    principal: Principal,
    payload: ChatRequest,
    settings: Settings,
    trace_id: str,
) -> AsyncIterator[str]:
    """Emits the DESIGN §5.2 event sequence for one turn.

    Phase 1 sources the text from a completed agent run and chunks it, so the
    framing, heartbeats and terminal events are real while generation is not yet
    incremental. Phase 2 replaces the body of this loop with `astream_events`
    without changing a single byte of the wire format.
    """
    started = time.perf_counter()

    def elapsed_ms() -> int:
        return int((time.perf_counter() - started) * 1000)

    yield sse.format_event(sse.NODE, {"name": "agent", "status": "start"})

    task = asyncio.ensure_future(
        service.ask(
            tenant_id=principal.tenant_id,
            thread_id=payload.thread_id,
            query=payload.query,
        )
    )

    try:
        async for heartbeat in _heartbeat_until_done(task, settings.sse_heartbeat_seconds):
            yield heartbeat
        result = task.result()
    except AppError as e:
        logger.warning("Stream failed (%s): %s", e.code.value, e.message)
        yield sse.format_event(
            sse.ERROR, error_body(e.code, e.message, trace_id, retryable=e.retryable)
        )
        return
    except Exception:
        logger.exception("Unhandled error while streaming")
        yield sse.format_event(
            sse.ERROR,
            error_body(
                ErrorCode.INTERNAL, default_message_for(ErrorCode.INTERNAL), trace_id
            ),
        )
        return

    yield sse.format_event(
        sse.NODE, {"name": "agent", "status": "end", "ms": elapsed_ms()}
    )

    for item in result.scratchpad or []:
        if item.get("type") == "tool_call":
            yield sse.format_event(
                sse.TOOL_CALL,
                {"id": item.get("id"), "name": item.get("name"), "args": item.get("args") or {}},
            )
        elif item.get("type") == "tool_observation":
            yield sse.format_event(
                sse.OBSERVATION,
                {
                    "id": item.get("tool_call_id"),
                    "name": item.get("name"),
                    "preview": sse.truncate_preview(
                        item.get("content", ""), settings.sse_observation_preview_bytes
                    ),
                },
            )

    for delta in sse.chunk_text(result.answer, settings.sse_token_chunk_chars):
        yield sse.format_event(sse.TOKEN, {"delta": delta})

    yield sse.format_event(
        sse.DONE,
        {
            "answer": result.answer,
            "citations": [],
            "usage": None,
            "latency_ms": elapsed_ms(),
            "trace_id": trace_id,
            "degraded": result.degraded,
        },
    )


# #### 1.8 - Streaming Endpoint & Proxy Buffering Control
# - Related Step: Step 2 (main.py) & Step 4 (sse.py)
# - Core Concept: Return `StreamingResponse` with `text/event-stream`. Header `X-Accel-Buffering: no`
#   instructs Nginx/proxies to disable response buffering so tokens stream immediately to frontend.
@router.post(
    "/chat/stream",
    summary="Ask the agent (SSE stream)",
    response_class=StreamingResponse,
    responses={
        200: {
            "content": {"text/event-stream": {}},
            "description": "SSE stream: node, tool_call, observation, token, done | error.",
        },
        **_ERROR_RESPONSES,
    },
)
async def chat_stream(
    payload: ChatRequest,
    principal: Principal = Depends(get_principal),
    service: AgentService = Depends(get_service),
    settings: Settings = Depends(get_settings),
) -> StreamingResponse:
    trace_id = get_trace_id()
    return StreamingResponse(
        _stream_turn(service, principal, payload, settings, trace_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            # Tells nginx-family proxies not to buffer. Traefik needs its own
            # annotation; see DESIGN.md Phase 6.
            "X-Accel-Buffering": "no",
            "X-Trace-Id": trace_id,
        },
    )
