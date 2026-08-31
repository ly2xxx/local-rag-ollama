"""Exercises the real service against real Chroma (no LLM involved).

Proves the HTTP layer, the tenant scoping and the vector store agree with each
other — the isolation guarantee is asserted through the API, not just in-process.
"""

import pytest
from httpx import ASGITransport, AsyncClient

from agentic_rag.api.main import create_app
from agentic_rag.api.service import AgenticRAGService
from agentic_rag.settings import Settings
from tests.fakes.service import FakeHelper

pytestmark = pytest.mark.integration

KEY_A = "key-tenant-a"
KEY_B = "key-tenant-b"


@pytest.fixture
def real_store_app(tmp_path, monkeypatch):
    """Points the document registry at a temp directory for the duration."""
    from agentic_rag import tools

    monkeypatch.setattr(tools, "CHROMA_PERSIST_DIR", str(tmp_path))
    monkeypatch.setattr(tools.doc_registry, "_managers", {})

    original_get = tools.DocumentStoreRegistry.get

    def scoped_get(self, namespace=None):
        resolved = (namespace or tools.get_active_namespace() or "default").strip()
        with self._lock:
            if resolved not in self._managers:
                self._managers[resolved] = tools.DocumentRetrieverManager(
                    namespace=resolved, persist_directory=str(tmp_path)
                )
            return self._managers[resolved]

    monkeypatch.setattr(tools.DocumentStoreRegistry, "get", scoped_get)

    settings = Settings(
        _env_file=None,
        api_keys=f"{KEY_A}:tenant_a,{KEY_B}:tenant_b",
        auth_enabled=True,
        warm_embeddings_on_startup=False,
    )
    # Without auth there is exactly one tenant, and every isolation assertion
    # below would be vacuous rather than wrong. Assert the premise.
    assert settings.auth_enabled, "tenant isolation is meaningless with auth disabled"
    assert len(settings.api_key_map) == 2

    service = AgenticRAGService(settings, helper=_RealDocsHelper())
    return create_app(settings=settings, service=service)


class _RealDocsHelper(FakeHelper):
    """Fake LLM, real document store."""

    def ingest_document(self, file_path, display_name=None, thread_id=None, namespace=None):
        from agentic_rag.tools import doc_registry

        return doc_registry.get(thread_id).ingest_pdf(file_path, display_name=display_name)

    def get_ingested_files(self, thread_id=None, namespace=None):
        from agentic_rag.tools import doc_registry

        return list(doc_registry.get(thread_id).ingested_files)

    def clear_documents(self, thread_id=None, namespace=None):
        from agentic_rag.tools import doc_registry

        doc_registry.get(thread_id).clear()


async def test_documents_are_isolated_between_tenants_through_http(real_store_app, sample_pdf):
    pdf = sample_pdf.read_bytes()
    transport = ASGITransport(app=real_store_app)

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        upload = await client.post(
            "/api/v1/documents",
            files={"file": ("confidential.pdf", pdf, "application/pdf")},
            data={"thread_id": "shared"},
            headers={"X-API-Key": KEY_A},
        )
        assert upload.status_code == 202

        mine = await client.get(
            "/api/v1/documents", params={"thread_id": "shared"}, headers={"X-API-Key": KEY_A}
        )
        theirs = await client.get(
            "/api/v1/documents", params={"thread_id": "shared"}, headers={"X-API-Key": KEY_B}
        )

    assert mine.json()["documents"] == ["confidential.pdf"]
    assert theirs.json()["documents"] == []


async def test_reingest_through_http_is_idempotent(real_store_app, sample_pdf):
    from agentic_rag.tools import doc_registry

    pdf = sample_pdf.read_bytes()
    transport = ASGITransport(app=real_store_app)

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        for _ in range(2):
            await client.post(
                "/api/v1/documents",
                files={"file": ("paper.pdf", pdf, "application/pdf")},
                data={"thread_id": "s1"},
                headers={"X-API-Key": KEY_A},
            )

    manager = doc_registry.get("tenant_a::s1")
    counts = manager.count()
    assert counts > 0
    assert manager.ingested_files == ["paper.pdf"]


async def test_delete_through_http_clears_the_store(real_store_app, sample_pdf):
    pdf = sample_pdf.read_bytes()
    transport = ASGITransport(app=real_store_app)

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        await client.post(
            "/api/v1/documents",
            files={"file": ("paper.pdf", pdf, "application/pdf")},
            data={"thread_id": "s1"},
            headers={"X-API-Key": KEY_A},
        )
        await client.delete(
            "/api/v1/documents", params={"thread_id": "s1"}, headers={"X-API-Key": KEY_A}
        )
        listed = await client.get(
            "/api/v1/documents", params={"thread_id": "s1"}, headers={"X-API-Key": KEY_A}
        )

    assert listed.json()["documents"] == []
