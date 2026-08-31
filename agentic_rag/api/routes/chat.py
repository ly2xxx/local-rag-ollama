"""Chat endpoints: buffered JSON and SSE streaming."""

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


@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Ask the agent (buffered)",
    responses=_ERROR_RESPONSES,
)
async def chat(
    payload: ChatRequest,
    principal: Principal = Depends(get_principal),
    service: AgentService = Depends(get_service),
) -> ChatResponse:
    started = time.perf_counter()
    result = await service.ask(
        tenant_id=principal.tenant_id,
        thread_id=payload.thread_id,
        query=payload.query,
    )
    return ChatResponse(
        answer=result.answer,
        thread_id=payload.thread_id,
        trace_id=get_trace_id(),
        scratchpad=_to_entries(result.scratchpad),
        latency_ms=int((time.perf_counter() - started) * 1000),
        degraded=result.degraded,
    )


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
