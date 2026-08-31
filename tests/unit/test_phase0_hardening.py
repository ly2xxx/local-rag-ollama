"""Retrofit of the Phase 0 hardening checks (DESIGN.md §6.5).

These assert the fixes that closed a real RCE-shaped hole and a real
cross-tenant data leak, so they are permanent regression tests, not archaeology.
"""

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agentic_rag.agent import _extract_final_answer
from agentic_rag.tools import (
    DocumentRetrieverManager,
    UnsafeExpressionError,
    calculate_expression,
    doc_registry,
    namespace_to_collection,
    read_local_file,
    safe_eval_expression,
)
from tests.fakes.embeddings import DeterministicFakeEmbeddings

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------- #
# Fix 1 — calculator sandbox
# --------------------------------------------------------------------------- #

ESCAPE_PAYLOADS = [
    "(1).__class__.__base__.__subclasses__()",
    "__import__('os').system('echo pwned')",
    "open('C:/Windows/win.ini').read()",
    "[c for c in ().__class__.__mro__]",
    "9**9**9",
    "math._sin",
    "().__class__",
    "lambda: 1",
]


@pytest.mark.parametrize("payload", ESCAPE_PAYLOADS)
def test_sandbox_rejects_dunder_escape(payload):
    with pytest.raises(UnsafeExpressionError):
        safe_eval_expression(payload)


@pytest.mark.parametrize(
    "expression,expected",
    [
        ("sum([80, 90, 99, 70])/4", 84.75),
        ("math.sqrt(144)", 12.0),
        ("2**10", 1024),
        ("round(math.pi, 4)", 3.1416),
        ("max([3, 9, 2]) - min([3, 9, 2])", 7),
        ("-(4 + 6) * 2", -20),
    ],
)
def test_sandbox_still_computes_valid_math(expression, expected):
    assert safe_eval_expression(expression) == expected


def test_sandbox_rejects_oversized_expression():
    with pytest.raises(UnsafeExpressionError):
        safe_eval_expression("1+" * 400 + "1")


def test_calculator_tool_returns_message_instead_of_raising():
    result = calculate_expression.invoke({"expression": "().__class__"})
    assert result.startswith("Rejected expression")


def test_calculator_tool_handles_division_by_zero():
    assert "Error calculating" in calculate_expression.invoke({"expression": "1/0"})


# --------------------------------------------------------------------------- #
# Fix 2 + 3 — namespace isolation, persistence, idempotent ingest
# --------------------------------------------------------------------------- #


@pytest.fixture
def fake_embeddings(monkeypatch):
    """Swaps the sentence-transformer for hash vectors so this stays a unit test."""
    embeddings = DeterministicFakeEmbeddings(size=32)
    monkeypatch.setattr(
        "agentic_rag.tools.get_embeddings", lambda model_name=None: embeddings
    )
    return embeddings


def _manager(namespace, tmp_path, **kwargs):
    return DocumentRetrieverManager(
        namespace=namespace, persist_directory=str(tmp_path), **kwargs
    )


def test_namespaces_map_to_distinct_collections():
    assert namespace_to_collection("tenant_a::s1") != namespace_to_collection("tenant_b::s1")


def test_namespace_collection_name_is_chroma_safe():
    name = namespace_to_collection("../../weird name!!")
    assert name[0].isalnum() and name[-1].isalnum()
    assert all(ch.isalnum() or ch in "-_" for ch in name)
    assert 3 <= len(name) <= 63


def test_namespaces_are_isolated(tmp_path, fake_embeddings, sample_pdf):
    alice = _manager("tenant_a::s1", tmp_path)
    bob = _manager("tenant_b::s1", tmp_path)

    alice.ingest_pdf(str(sample_pdf), display_name="quarterly.pdf")

    assert alice.count() > 0
    assert bob.count() == 0
    assert "No documents have been ingested" in bob.query("what is this about")
    assert "Document Match #1" in alice.query("what is this about")
    assert alice.ingested_files == ["quarterly.pdf"]
    assert bob.ingested_files == []


def test_reingest_is_idempotent(tmp_path, fake_embeddings, sample_pdf):
    manager = _manager("tenant_a::s1", tmp_path)
    manager.ingest_pdf(str(sample_pdf), display_name="quarterly.pdf")
    first_count = manager.count()

    manager.ingest_pdf(str(sample_pdf), display_name="quarterly.pdf")

    assert manager.count() == first_count


def test_knowledge_base_survives_restart(tmp_path, fake_embeddings, sample_pdf):
    original = _manager("tenant_a::s1", tmp_path)
    original.ingest_pdf(str(sample_pdf), display_name="quarterly.pdf")
    expected = original.count()

    reopened = _manager("tenant_a::s1", tmp_path)  # simulates a fresh process

    assert reopened.count() == expected
    assert reopened.ingested_files == ["quarterly.pdf"]


def test_display_name_is_used_not_temp_filename(tmp_path, fake_embeddings, sample_pdf):
    manager = _manager("tenant_a::s1", tmp_path)
    manager.ingest_pdf(str(sample_pdf), display_name="real_name.pdf")
    assert manager.ingested_files == ["real_name.pdf"]


def test_registry_returns_one_manager_per_namespace():
    assert doc_registry.get("ns_one") is doc_registry.get("ns_one")
    assert doc_registry.get("ns_one") is not doc_registry.get("ns_two")


# --------------------------------------------------------------------------- #
# Fix 4 — final answer extraction
# --------------------------------------------------------------------------- #


def test_final_answer_skips_pre_tool_reasoning():
    trajectory = [
        HumanMessage(content="What does the paper say about latency?"),
        AIMessage(
            content="I should search the knowledge base first.",
            tool_calls=[
                {"name": "query_document_knowledge_base", "args": {"query": "latency"}, "id": "c1"}
            ],
        ),
        ToolMessage(
            content="--- [Document Match #1] ...",
            name="query_document_knowledge_base",
            tool_call_id="c1",
        ),
        AIMessage(content="The paper reports p99 latency of 12ms."),
    ]
    assert _extract_final_answer(trajectory) == "The paper reports p99 latency of 12ms."


def test_final_answer_handles_list_of_content_blocks():
    messages = [HumanMessage(content="hi"), AIMessage(content=[{"type": "text", "text": "Hello."}])]
    assert _extract_final_answer(messages) == "Hello."


def test_final_answer_falls_back_to_tool_call_message():
    messages = [
        HumanMessage(content="hi"),
        AIMessage(content="Let me check.", tool_calls=[{"name": "t", "args": {}, "id": "c2"}]),
    ]
    assert _extract_final_answer(messages) == "Let me check."


def test_final_answer_on_empty_trajectory():
    assert _extract_final_answer([]) == "No response generated."


# --------------------------------------------------------------------------- #
# Fix 5 — file tool scoping
# --------------------------------------------------------------------------- #


def test_out_of_scope_file_read_is_denied(tmp_path, monkeypatch):
    outside = tmp_path / "secret.txt"
    outside.write_text("classified")
    monkeypatch.setattr("agentic_rag.tools.FILE_TOOL_ROOTS", str(tmp_path / "allowed"))
    (tmp_path / "allowed").mkdir()

    result = read_local_file.invoke({"file_path": str(outside)})

    assert "Access denied" in result
    assert "classified" not in result


def test_in_scope_file_read_is_allowed(tmp_path, monkeypatch):
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    target = allowed / "notes.txt"
    target.write_text("readable content")
    monkeypatch.setattr("agentic_rag.tools.FILE_TOOL_ROOTS", str(allowed))

    assert read_local_file.invoke({"file_path": str(target)}) == "readable content"
