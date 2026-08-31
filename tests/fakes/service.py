"""Fakes for the service layer, so contract tests never touch a model or Redis."""

import asyncio
from typing import Any, Dict, List, Optional

from agentic_rag.api.service import AskResult, IngestResult, scope_id


class FakeAgentService:
    """Implements the `AgentService` protocol with scripted behaviour.

    Documents are stored per scope, so tenant-isolation assertions at the HTTP
    layer are meaningful rather than decorative.
    """

    def __init__(
        self,
        answer: str = "The answer is 42.",
        scratchpad: Optional[List[Dict[str, Any]]] = None,
        delay: float = 0.0,
        error: Optional[BaseException] = None,
        dependencies: Optional[List[Dict[str, Any]]] = None,
        degraded: Optional[List[str]] = None,
    ):
        self.answer = answer
        self.scratchpad = scratchpad if scratchpad is not None else []
        self.delay = delay
        self.error = error
        self.degraded = degraded or []
        self.dependencies = dependencies if dependencies is not None else [
            {"name": "redis", "healthy": True, "required": True},
            {"name": "vector_store", "healthy": True, "required": True},
        ]
        self.dependencies_error: Optional[BaseException] = None
        self.documents: Dict[str, List[str]] = {}
        self.ask_calls: List[Dict[str, Any]] = []
        self.warm_up_calls = 0

    async def ask(self, *, tenant_id: str, thread_id: str, query: str) -> AskResult:
        self.ask_calls.append(
            {"tenant_id": tenant_id, "thread_id": thread_id, "query": query,
             "scope": scope_id(tenant_id, thread_id)}
        )
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.error:
            raise self.error
        return AskResult(
            answer=self.answer, scratchpad=list(self.scratchpad), degraded=list(self.degraded)
        )

    async def ingest(
        self, *, tenant_id: str, thread_id: str, file_path: str, display_name: str
    ) -> IngestResult:
        if self.error:
            raise self.error
        self.documents.setdefault(scope_id(tenant_id, thread_id), []).append(display_name)
        return IngestResult(
            document=display_name, detail=f"Successfully ingested '{display_name}' (3 chunks)."
        )

    async def list_documents(self, *, tenant_id: str, thread_id: str) -> List[str]:
        return sorted(self.documents.get(scope_id(tenant_id, thread_id), []))

    async def clear_documents(self, *, tenant_id: str, thread_id: str) -> None:
        self.documents.pop(scope_id(tenant_id, thread_id), None)

    async def check_dependencies(self) -> List[Dict[str, Any]]:
        if self.dependencies_error:
            raise self.dependencies_error
        return list(self.dependencies)

    async def warm_up(self) -> None:
        self.warm_up_calls += 1


class FakeHelper:
    """Stands in for `AgenticRAGHelper` when testing `AgenticRAGService` itself."""

    def __init__(self, result: Optional[Dict[str, Any]] = None, redis_active: bool = True):
        self.result = result or {"answer": "ok", "scratchpad": [], "success": True}
        self.redis_active = redis_active
        self.calls: List[Dict[str, Any]] = []
        self.documents: Dict[str, List[str]] = {}

    def ask(self, query: str, thread_id: str = "default", namespace: Optional[str] = None):
        self.calls.append({"query": query, "thread_id": thread_id, "namespace": namespace})
        return self.result

    def ingest_document(self, file_path, display_name=None, thread_id=None, namespace=None):
        self.documents.setdefault(thread_id, []).append(display_name)
        return f"Successfully ingested '{display_name}' (1 chunks)."

    def get_ingested_files(self, thread_id=None, namespace=None):
        return sorted(self.documents.get(thread_id, []))

    def clear_documents(self, thread_id=None, namespace=None):
        self.documents.pop(thread_id, None)

    def get_redis_status(self):
        return {"active": self.redis_active, "message": "", "url": "redis://fake"}
