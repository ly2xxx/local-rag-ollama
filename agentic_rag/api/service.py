"""Service layer between HTTP routes and the agent core.

Routes stay thin (DESIGN.md Phase 1): they validate, resolve tenancy, and call
this. The synchronous agent core is offloaded to a worker thread so it never
blocks the event loop; `anyio.to_thread.run_sync` copies the context, which the
Phase 0 namespace binding relies on.
"""

import logging
import uuid
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Protocol

import anyio

from ..settings import Settings
from .errors import AppError, ErrorCode

logger = logging.getLogger(__name__)


def scope_id(tenant_id: str, thread_id: str) -> str:
    """Namespace for both the vector store and the LangGraph checkpoint.

    The tenant is part of the key, so two tenants using the same `thread_id`
    share neither documents nor conversation state.
    """
    return f"{tenant_id}::{thread_id}"


@dataclass
class AskResult:
    answer: str
    scratchpad: List[Dict[str, Any]] = field(default_factory=list)
    degraded: List[str] = field(default_factory=list)


@dataclass
class IngestResult:
    document: str
    detail: str
    status: str = "completed"


class AgentService(Protocol):
    """The surface the routes depend on. Tests substitute a fake."""

    async def ask(self, *, tenant_id: str, thread_id: str, query: str) -> AskResult: ...

    async def ingest(
        self, *, tenant_id: str, thread_id: str, file_path: str, display_name: str
    ) -> IngestResult: ...

    async def list_documents(self, *, tenant_id: str, thread_id: str) -> List[str]: ...

    async def clear_documents(self, *, tenant_id: str, thread_id: str) -> None: ...

    async def check_dependencies(self) -> List[Dict[str, Any]]: ...

    async def warm_up(self) -> None: ...


def classify_agent_failure(message: str) -> ErrorCode:
    """Maps a core failure string onto the taxonomy.

    Phase 2 replaces this with typed exceptions raised by the provider layer;
    until then the core only reports failures as text.
    """
    lowered = (message or "").lower()
    if "timeout" in lowered or "timed out" in lowered:
        return ErrorCode.LLM_TIMEOUT
    # "connect" also covers "connection", "connecting", "ConnectError".
    if any(token in lowered for token in ("connect", "refused", "unreachable")):
        return ErrorCode.LLM_UNAVAILABLE
    return ErrorCode.INTERNAL


class AgenticRAGService:
    """Real implementation backed by `AgenticRAGHelper`."""

    def __init__(self, settings: Settings, helper: Optional[Any] = None):
        self.settings = settings
        if helper is None:
            from ..agent import AgenticRAGHelper

            helper = AgenticRAGHelper(
                model_name=settings.ollama_model,
                base_url=settings.ollama_base_url,
                redis_url=settings.redis_url,
            )
        self.helper = helper

    async def warm_up(self) -> None:
        """Loads the embedding model up front so the first request is not slow.

        Also the reason the Phase 6 manifests need a startup probe distinct from
        liveness — this takes tens of seconds on a cold pod.
        """
        if not self.settings.warm_embeddings_on_startup:
            return

        def _load():
            from ..tools import get_embeddings

            get_embeddings()

        try:
            await anyio.to_thread.run_sync(_load)
            logger.info("Embedding model warmed")
        except Exception as e:  # pragma: no cover - environment dependent
            logger.warning("Embedding warm-up failed: %s", e)

    async def ask(self, *, tenant_id: str, thread_id: str, query: str) -> AskResult:
        scope = scope_id(tenant_id, thread_id)
        result = await anyio.to_thread.run_sync(
            partial(self.helper.ask, query, thread_id=scope, namespace=scope)
        )

        if not result.get("success", False):
            message = str(result.get("answer", ""))
            code = classify_agent_failure(message)
            logger.error("Agent run failed (%s): %s", code.value, message)
            raise AppError(code)

        degraded: List[str] = []
        if not self.helper.get_redis_status().get("active", False):
            degraded.append("redis")

        return AskResult(
            answer=result.get("answer", ""),
            scratchpad=result.get("scratchpad", []),
            degraded=degraded,
        )

    async def ingest(
        self, *, tenant_id: str, thread_id: str, file_path: str, display_name: str
    ) -> IngestResult:
        scope = scope_id(tenant_id, thread_id)
        detail = await anyio.to_thread.run_sync(
            partial(
                self.helper.ingest_document,
                file_path,
                display_name=display_name,
                thread_id=scope,
            )
        )
        if isinstance(detail, str) and detail.lower().startswith("error"):
            raise AppError(ErrorCode.RETRIEVAL_UNAVAILABLE, detail)
        return IngestResult(document=display_name, detail=detail)

    async def list_documents(self, *, tenant_id: str, thread_id: str) -> List[str]:
        scope = scope_id(tenant_id, thread_id)
        return await anyio.to_thread.run_sync(
            partial(self.helper.get_ingested_files, thread_id=scope)
        )

    async def clear_documents(self, *, tenant_id: str, thread_id: str) -> None:
        scope = scope_id(tenant_id, thread_id)
        await anyio.to_thread.run_sync(
            partial(self.helper.clear_documents, thread_id=scope)
        )

    async def _probe(
        self, name: str, check: Callable[[], Any], *, required: bool
    ) -> Dict[str, Any]:
        """Runs one blocking health check off the event loop, never raising."""
        try:
            await anyio.to_thread.run_sync(check)
        except Exception as e:
            return {
                "name": name,
                "healthy": False,
                "required": required,
                "detail": str(e)[:200],
            }
        return {"name": name, "healthy": True, "required": required}

    async def check_dependencies(self) -> List[Dict[str, Any]]:
        """Live probe of every backing service — not a cached startup value."""

        def _ping_redis() -> None:
            import redis

            timeout = self.settings.redis_ping_timeout_seconds
            redis.from_url(
                self.settings.redis_url,
                socket_connect_timeout=timeout,
                socket_timeout=timeout,
            ).ping()

        def _probe_store() -> None:
            from ..tools import doc_registry

            doc_registry.get("__readyz_probe__").count()

        return [
            await self._probe(
                "redis", _ping_redis, required=self.settings.readyz_require_redis
            ),
            await self._probe("vector_store", _probe_store, required=True),
        ]


def new_job_id() -> str:
    return uuid.uuid4().hex
