"""FastAPI application factory.

`create_app()` builds a fully wired app. Tests pass an injected `service` so no
model, Redis connection, or vector store is touched.
"""

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from ..settings import Settings, get_settings
from .errors import (
    AppError,
    ErrorCode,
    code_for_status,
    default_message_for,
    error_payload,
    status_for,
)
from .middleware import TraceIdMiddleware
from .routes import chat, documents, health
from .service import AgentService, AgenticRAGService
from .tracing import get_trace_id

logger = logging.getLogger(__name__)

DESCRIPTION = """
Enterprise Agentic RAG API.

* `POST /api/v1/chat` — buffered answer
* `POST /api/v1/chat/stream` — SSE stream (`node`, `tool_call`, `observation`, `token`, `done` | `error`)
* `POST|GET|DELETE /api/v1/documents` — per-session knowledge base
* `GET /healthz`, `GET /readyz` — probes

Authenticate with `X-API-Key`. The tenant is derived from the key, never from the
request body, so a caller cannot address another tenant's data.
"""


def _error_response(
    code: ErrorCode,
    message: Optional[str] = None,
    *,
    status_code: Optional[int] = None,
    details: Optional[dict] = None,
) -> JSONResponse:
    """Renders one error envelope. Both `message` and `status_code` default to
    the taxonomy entry for `code`."""
    return JSONResponse(
        status_code=status_code if status_code is not None else status_for(code),
        content=error_payload(
            code, message or default_message_for(code), get_trace_id(), details=details
        ),
    )


def _register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def _handle_app_error(request: Request, exc: AppError) -> JSONResponse:
        logger.info("AppError %s on %s: %s", exc.code.value, request.url.path, exc.message)
        return JSONResponse(
            status_code=exc.status_code, content=exc.to_payload(get_trace_id())
        )

    @app.exception_handler(RequestValidationError)
    async def _handle_validation(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        # FastAPI defaults to 422; the taxonomy says a malformed request is 400.
        fields = [
            {"loc": ".".join(str(p) for p in err.get("loc", [])), "msg": err.get("msg", "")}
            for err in exc.errors()
        ]
        return _error_response(ErrorCode.INVALID_REQUEST, details={"fields": fields})

    @app.exception_handler(StarletteHTTPException)
    async def _handle_http_exception(
        request: Request, exc: StarletteHTTPException
    ) -> JSONResponse:
        # Keep the framework's own status (405, 422, ...) and only borrow the
        # taxonomy for the code/retryable fields.
        message = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
        return _error_response(
            code_for_status(exc.status_code), message, status_code=exc.status_code
        )

    @app.exception_handler(Exception)
    async def _handle_unexpected(request: Request, exc: Exception) -> JSONResponse:
        # The trace_id is the only bridge between this response and the logged
        # traceback; the client never sees the exception itself.
        logger.exception(
            "Unhandled exception on %s [trace_id=%s]", request.url.path, get_trace_id()
        )
        return _error_response(ErrorCode.INTERNAL)


def create_app(
    settings: Optional[Settings] = None,
    service: Optional[AgentService] = None,
) -> FastAPI:
    """Builds the app. Injecting `service` skips all real upstream construction."""
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if app.state.service is None:
            logger.info("Constructing AgenticRAGService")
            app.state.service = AgenticRAGService(settings)

        if settings.auth_enabled and not settings.api_key_map:
            logger.warning(
                "auth_enabled is true but API_KEYS is empty — every authenticated "
                "endpoint will return 401. Set API_KEYS='<key>:<tenant>'."
            )

        # Injected test doubles need not implement the whole protocol.
        warm_up = getattr(app.state.service, "warm_up", None)
        if warm_up is not None:
            await warm_up()

        yield

        logger.info("Shutting down %s", settings.app_name)

    app = FastAPI(
        title="Agentic RAG API",
        version=settings.app_version,
        description=DESCRIPTION,
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.service = service

    app.add_middleware(TraceIdMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Trace-Id"],
    )

    _register_exception_handlers(app)

    app.include_router(health.router)
    app.include_router(chat.router)
    app.include_router(documents.router)

    return app


app = create_app()
