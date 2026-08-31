"""Request middleware: trace correlation and access logging."""

import logging
import time

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from .tracing import reset_trace_id, resolve_inbound_trace_id, set_trace_id

logger = logging.getLogger("agentic_rag.access")

TRACE_HEADER = "X-Trace-Id"
REQUEST_ID_HEADER = "X-Request-Id"


class TraceIdMiddleware(BaseHTTPMiddleware):
    """Binds a trace_id for the request and echoes it on the response.

    Everything downstream — log records, error envelopes, the SSE `done` event —
    reads it from the contextvar, so one grep reconstructs a whole turn.
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        trace_id = resolve_inbound_trace_id(request.headers.get(REQUEST_ID_HEADER))
        token = set_trace_id(trace_id)
        request.state.trace_id = trace_id
        started = time.perf_counter()
        context = {
            "trace_id": trace_id,
            "method": request.method,
            "path": request.url.path,
        }

        def elapsed_ms() -> int:
            return int((time.perf_counter() - started) * 1000)

        try:
            response = await call_next(request)
        except Exception:
            logger.exception("request failed", extra={**context, "duration_ms": elapsed_ms()})
            raise
        else:
            duration_ms = elapsed_ms()
            response.headers[TRACE_HEADER] = trace_id
            logger.info(
                "%s %s -> %s (%dms)",
                request.method,
                request.url.path,
                response.status_code,
                duration_ms,
                extra={**context, "status": response.status_code, "duration_ms": duration_ms},
            )
            return response
        finally:
            reset_trace_id(token)
