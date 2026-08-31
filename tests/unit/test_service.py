import pytest

from agentic_rag.api.errors import AppError, ErrorCode
from agentic_rag.api.service import (
    AgenticRAGService,
    classify_agent_failure,
    scope_id,
)
from tests.fakes.service import FakeHelper

pytestmark = pytest.mark.unit


def test_scope_id_includes_the_tenant():
    assert scope_id("tenant_a", "session_1") == "tenant_a::session_1"


def test_same_thread_id_across_tenants_yields_different_scopes():
    # The leak this prevents: two tenants both using "session_1" would otherwise
    # share a vector-store namespace *and* a LangGraph checkpoint.
    assert scope_id("tenant_a", "session_1") != scope_id("tenant_b", "session_1")


@pytest.mark.parametrize(
    "message,expected",
    [
        ("Connection refused to 127.0.0.1:11434", ErrorCode.LLM_UNAVAILABLE),
        ("httpx.ConnectError: [Errno 111]", ErrorCode.LLM_UNAVAILABLE),
        ("Request timed out after 60s", ErrorCode.LLM_TIMEOUT),
        ("ReadTimeout", ErrorCode.LLM_TIMEOUT),
        ("something else entirely", ErrorCode.INTERNAL),
        ("", ErrorCode.INTERNAL),
    ],
)
def test_classify_agent_failure(message, expected):
    assert classify_agent_failure(message) is expected


async def test_ask_scopes_namespace_and_thread_to_the_tenant(settings):
    helper = FakeHelper({"answer": "hi", "scratchpad": [], "success": True})
    service = AgenticRAGService(settings, helper=helper)

    await service.ask(tenant_id="tenant_a", thread_id="session_1", query="hello")

    call = helper.calls[0]
    assert call["thread_id"] == "tenant_a::session_1"
    assert call["namespace"] == "tenant_a::session_1"


async def test_ask_raises_typed_error_on_core_failure(settings):
    helper = FakeHelper({"answer": "Connection refused", "scratchpad": [], "success": False})
    service = AgenticRAGService(settings, helper=helper)

    with pytest.raises(AppError) as excinfo:
        await service.ask(tenant_id="t", thread_id="s", query="hello")

    assert excinfo.value.code is ErrorCode.LLM_UNAVAILABLE


async def test_ask_reports_redis_degradation(settings):
    helper = FakeHelper({"answer": "hi", "scratchpad": [], "success": True}, redis_active=False)
    service = AgenticRAGService(settings, helper=helper)

    result = await service.ask(tenant_id="t", thread_id="s", query="hello")

    assert result.degraded == ["redis"]


async def test_ingest_error_string_becomes_typed_error(settings, monkeypatch):
    helper = FakeHelper()
    monkeypatch.setattr(
        helper, "ingest_document", lambda *a, **k: "Error reading PDF 'x.pdf': broken"
    )
    service = AgenticRAGService(settings, helper=helper)

    with pytest.raises(AppError) as excinfo:
        await service.ingest(
            tenant_id="t", thread_id="s", file_path="x.pdf", display_name="x.pdf"
        )

    assert excinfo.value.code is ErrorCode.RETRIEVAL_UNAVAILABLE


async def test_documents_are_scoped_per_tenant(settings):
    service = AgenticRAGService(settings, helper=FakeHelper())

    await service.ingest(
        tenant_id="tenant_a", thread_id="s", file_path="a.pdf", display_name="a.pdf"
    )

    assert await service.list_documents(tenant_id="tenant_a", thread_id="s") == ["a.pdf"]
    assert await service.list_documents(tenant_id="tenant_b", thread_id="s") == []


async def test_warm_up_is_skipped_when_disabled(settings):
    # settings fixture sets warm_embeddings_on_startup=False
    service = AgenticRAGService(settings, helper=FakeHelper())
    await service.warm_up()  # must not attempt to load the model
