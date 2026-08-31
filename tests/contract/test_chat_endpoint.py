import pytest

pytestmark = pytest.mark.contract


async def test_chat_returns_answer_and_trace_id(client, auth_a):
    response = await client.post("/api/v1/chat", json={"query": "hello"}, headers=auth_a)

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "The answer is 42."
    assert body["thread_id"] == "default_session"
    assert body["trace_id"]
    assert body["trace_id"] == response.headers["X-Trace-Id"]
    assert isinstance(body["latency_ms"], int)


async def test_chat_is_401_without_api_key(client):
    response = await client.post("/api/v1/chat", json={"query": "hello"})

    assert response.status_code == 401
    assert response.json()["error"]["code"] == "UNAUTHENTICATED"


async def test_chat_is_401_with_unknown_api_key(client):
    response = await client.post(
        "/api/v1/chat", json={"query": "hello"}, headers={"X-API-Key": "not-a-real-key"}
    )
    assert response.status_code == 401


async def test_blank_query_is_400_not_422(client, auth_a):
    response = await client.post("/api/v1/chat", json={"query": "   "}, headers=auth_a)

    assert response.status_code == 400
    error = response.json()["error"]
    assert error["code"] == "INVALID_REQUEST"
    assert error["retryable"] is False
    assert "fields" in error["details"]


async def test_bad_thread_id_is_rejected(client, auth_a):
    response = await client.post(
        "/api/v1/chat", json={"query": "hi", "thread_id": "../../etc"}, headers=auth_a
    )
    assert response.status_code == 400


async def test_tenant_id_cannot_be_supplied_by_the_client(client, auth_a, fake_service):
    response = await client.post(
        "/api/v1/chat",
        json={"query": "hi", "tenant_id": "tenant_b"},
        headers=auth_a,
    )
    assert response.status_code == 400  # extra="forbid"


async def test_scope_is_derived_from_the_api_key(client, auth_a, auth_b, fake_service):
    await client.post("/api/v1/chat", json={"query": "hi", "thread_id": "shared"}, headers=auth_a)
    await client.post("/api/v1/chat", json={"query": "hi", "thread_id": "shared"}, headers=auth_b)

    scopes = [call["scope"] for call in fake_service.ask_calls]
    assert scopes == ["tenant_a::shared", "tenant_b::shared"]


async def test_tenant_cannot_read_other_tenant_documents(client, auth_a, auth_b, sample_pdf):
    upload = {"file": ("secret.pdf", sample_pdf.read_bytes(), "application/pdf")}
    ingest = await client.post(
        "/api/v1/documents", files=upload, data={"thread_id": "shared"}, headers=auth_a
    )
    assert ingest.status_code == 202

    mine = await client.get("/api/v1/documents", params={"thread_id": "shared"}, headers=auth_a)
    theirs = await client.get("/api/v1/documents", params={"thread_id": "shared"}, headers=auth_b)

    assert mine.json()["documents"] == ["secret.pdf"]
    assert theirs.json()["documents"] == []


async def test_upstream_failure_maps_to_503(client, auth_a, fake_service):
    from agentic_rag.api.errors import AppError, ErrorCode

    fake_service.error = AppError(ErrorCode.LLM_UNAVAILABLE)
    response = await client.post("/api/v1/chat", json={"query": "hi"}, headers=auth_a)

    assert response.status_code == 503
    body = response.json()["error"]
    assert body["code"] == "LLM_UNAVAILABLE"
    assert body["retryable"] is True


async def test_scratchpad_is_returned(client, auth_a, fake_service):
    fake_service.scratchpad = [
        {"type": "tool_call", "name": "query_document_knowledge_base", "args": {"query": "x"}, "id": "c1"},
        {"type": "tool_observation", "name": "query_document_knowledge_base", "content": "match", "id": "c1"},
    ]
    response = await client.post("/api/v1/chat", json={"query": "hi"}, headers=auth_a)

    entries = response.json()["scratchpad"]
    assert [e["type"] for e in entries] == ["tool_call", "tool_observation"]


async def test_inbound_request_id_is_echoed_as_trace_id(client, auth_a):
    response = await client.post(
        "/api/v1/chat",
        json={"query": "hi"},
        headers={**auth_a, "X-Request-Id": "my-correlation-id"},
    )
    assert response.headers["X-Trace-Id"] == "my-correlation-id"
    assert response.json()["trace_id"] == "my-correlation-id"


async def test_malformed_inbound_request_id_is_replaced(client, auth_a):
    response = await client.post(
        "/api/v1/chat", json={"query": "hi"}, headers={**auth_a, "X-Request-Id": "bad id!!"}
    )
    assert response.headers["X-Trace-Id"] != "bad id!!"


async def test_unknown_route_uses_the_error_envelope(client, auth_a):
    response = await client.get("/api/v1/nope", headers=auth_a)

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "NOT_FOUND"
