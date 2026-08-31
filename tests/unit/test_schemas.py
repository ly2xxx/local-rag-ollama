import json

import pytest
from pydantic import ValidationError

from agentic_rag.api.schemas import ChatRequest, ChatResponse, Citation

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("blank", ["", "   ", "\n\t "])
def test_chat_request_rejects_blank_query(blank):
    with pytest.raises(ValidationError):
        ChatRequest(query=blank)


def test_chat_request_strips_surrounding_whitespace():
    assert ChatRequest(query="  what is RAG?  ").query == "what is RAG?"


@pytest.mark.parametrize(
    "thread_id",
    ["../../etc/passwd", "a/b", "with space", "semi;colon", "x" * 65, "", "tenant::thread"],
)
def test_thread_id_pattern_rejects_path_traversal(thread_id):
    with pytest.raises(ValidationError):
        ChatRequest(query="hello", thread_id=thread_id)


@pytest.mark.parametrize("thread_id", ["session_1", "abc-123", "A_b-9", "x" * 64])
def test_thread_id_accepts_safe_values(thread_id):
    assert ChatRequest(query="hello", thread_id=thread_id).thread_id == thread_id


def test_chat_request_forbids_unknown_fields():
    # Blocks a client trying to smuggle in a tenant or namespace override.
    with pytest.raises(ValidationError):
        ChatRequest(query="hello", tenant_id="someone_else")


def test_chat_response_serialises_citations_stably():
    response = ChatResponse(
        answer="42",
        thread_id="session_1",
        trace_id="abc123",
        citations=[
            Citation(n=1, chunk_id="c1", source="paper.pdf", page=3, score=0.91),
            Citation(n=2, chunk_id="c2", source="paper.pdf", page=7, score=0.80),
        ],
        latency_ms=120,
    )
    first = response.model_dump_json()
    second = response.model_dump_json()
    assert first == second

    payload = json.loads(first)
    assert [c["n"] for c in payload["citations"]] == [1, 2]
    assert payload["citations"][0]["chunk_id"] == "c1"
    assert payload["degraded"] == []


def test_citation_rejects_zero_marker():
    with pytest.raises(ValidationError):
        Citation(n=0, chunk_id="c1", source="paper.pdf")
