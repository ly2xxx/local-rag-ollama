"""Regression net for the synchronous agent path that Streamlit tab 3 uses.

`app.py` never touches the API tier. It goes:

    AgenticRAGHelper.ask -> AgentCore.run -> graph.invoke

entirely synchronously. Phase 2 edits exactly that path — model construction
moves out to `llm/provider.py`, and the checkpointer gains an async variant — so
these tests exist to prove the path still executes before, during and after
those changes.

CI's `py_compile app.py` step only proves the file parses; it would not catch
any of the three Phase 2 risks (DESIGN.md Phase 2 · R1/R2/R3). These tests do.
"""

import pytest
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver

from agentic_rag.agent import AgenticRAGHelper
from agentic_rag.memory.short_term import ShortTermMemoryManager
from tests.fakes.chat_model import FakeToolCallingModel

pytestmark = pytest.mark.unit


def _tool_call(name: str, args: dict, call_id: str) -> dict:
    return {"name": name, "args": args, "id": call_id, "type": "tool_call"}


@pytest.fixture
def in_memory_checkpointer(monkeypatch):
    """Keeps the turn off Redis so this stays a hermetic unit test.

    The real Redis checkpointer is covered by
    tests/integration/test_agent_sync_path_redis.py.
    """
    monkeypatch.setattr(
        ShortTermMemoryManager,
        "_init_checkpointer",
        lambda self: (MemorySaver(), False, "test in-memory checkpointer"),
    )


@pytest.fixture
def fake_model(monkeypatch):
    """Replaces the model AgentCore builds for itself.

    `AgentCore` constructs `ChatOllama` inline today; Phase 2 replaces that with
    `get_chat_model("chat")`. Patching the name `core` resolves means this test
    keeps working across that change as long as the factory returns a plain
    `BaseChatModel` — which is exactly the R3 guardrail.
    """

    def _install(*responses: AIMessage) -> FakeToolCallingModel:
        model = FakeToolCallingModel(responses=list(responses))
        monkeypatch.setattr("agentic_rag.core.ChatOllama", lambda **kwargs: model)
        return model

    return _install


def test_helper_answers_a_direct_question(fake_model, in_memory_checkpointer):
    fake_model(AIMessage(content="RAG retrieves documents before generating."))

    result = AgenticRAGHelper().ask("what is RAG?", thread_id="session_1")

    assert result["success"] is True
    assert result["answer"] == "RAG retrieves documents before generating."
    assert result["scratchpad"] == []


def test_helper_runs_a_tool_and_reports_the_observation(fake_model, in_memory_checkpointer):
    """The exact shape app.py renders in its scratchpad expander."""
    fake_model(
        AIMessage(
            content="I should compute that.",
            tool_calls=[_tool_call("calculate_expression", {"expression": "sum([80,90,99,70])/4"}, "c1")],
        ),
        AIMessage(content="The average is 84.75."),
    )

    result = AgenticRAGHelper().ask("what is the average?", thread_id="session_1")

    assert result["success"] is True
    assert result["answer"] == "The average is 84.75."

    kinds = [entry["type"] for entry in result["scratchpad"]]
    assert kinds == ["tool_call", "tool_observation"]

    call, observation = result["scratchpad"]
    assert call["name"] == "calculate_expression"
    assert call["args"] == {"expression": "sum([80,90,99,70])/4"}
    assert "84.75" in observation["content"]


def test_tools_see_the_bound_namespace(monkeypatch, fake_model, in_memory_checkpointer):
    """Phase 0's contextvar binding must survive any AgentCore rewrite.

    If this breaks, retrieval tools reach the wrong tenant's documents — the
    failure is silent and the data leak is real, so it is asserted directly.
    """
    seen = []

    @tool
    def probe_namespace(query: str) -> str:
        """Records the namespace bound during execution."""
        from agentic_rag.tools import get_active_namespace

        seen.append(get_active_namespace())
        return "recorded"

    monkeypatch.setattr("agentic_rag.core.get_default_tools", lambda: [probe_namespace])
    fake_model(
        AIMessage(content="", tool_calls=[_tool_call("probe_namespace", {"query": "x"}, "c1")]),
        AIMessage(content="done"),
    )

    AgenticRAGHelper().ask("anything", thread_id="tenant_a::session_1")

    assert seen == ["tenant_a::session_1"]


def test_explicit_namespace_overrides_the_thread_id(monkeypatch, fake_model, in_memory_checkpointer):
    seen = []

    @tool
    def probe_namespace(query: str) -> str:
        """Records the namespace bound during execution."""
        from agentic_rag.tools import get_active_namespace

        seen.append(get_active_namespace())
        return "recorded"

    monkeypatch.setattr("agentic_rag.core.get_default_tools", lambda: [probe_namespace])
    fake_model(
        AIMessage(content="", tool_calls=[_tool_call("probe_namespace", {"query": "x"}, "c1")]),
        AIMessage(content="done"),
    )

    AgenticRAGHelper().ask("anything", thread_id="thread_only", namespace="tenant_b::kb")

    assert seen == ["tenant_b::kb"]


def test_multi_turn_state_is_checkpointed(fake_model, in_memory_checkpointer):
    """Second turn must see the first — this is what tab 3's thread_id selector demos."""
    model = fake_model(
        AIMessage(content="First answer."),
        AIMessage(content="Second answer."),
    )
    helper = AgenticRAGHelper()

    helper.ask("first question", thread_id="session_1")
    second = helper.ask("second question", thread_id="session_1")

    assert second["answer"] == "Second answer."
    contents = [
        m.content for m in second["messages"] if getattr(m, "content", None)
    ]
    assert "first question" in contents
    assert "First answer." in contents
    assert model.index == 2


def test_threads_do_not_leak_into_each_other(fake_model, in_memory_checkpointer):
    fake_model(AIMessage(content="A."), AIMessage(content="B."))
    helper = AgenticRAGHelper()

    helper.ask("question for thread one", thread_id="thread_one")
    other = helper.ask("question for thread two", thread_id="thread_two")

    contents = [m.content for m in other["messages"] if getattr(m, "content", None)]
    assert "question for thread one" not in contents


def test_core_failure_is_reported_not_raised(monkeypatch, fake_model, in_memory_checkpointer):
    """app.py has no try/except around ask(); a raise would blank the tab."""
    fake_model(AIMessage(content="unused"))
    helper = AgenticRAGHelper()
    monkeypatch.setattr(
        helper.core, "run", lambda **kwargs: (_ for _ in ()).throw(RuntimeError("boom"))
    )

    result = helper.ask("anything", thread_id="session_1")

    assert result["success"] is False
    assert "boom" in result["answer"]
    assert result["scratchpad"] == []


def test_redis_status_shape_for_the_ui(fake_model, in_memory_checkpointer):
    """app.py reads ["active"] and ["url"] to render its status badge."""
    status = AgenticRAGHelper().get_redis_status()

    assert set(status) == {"active", "message", "url"}
    assert isinstance(status["active"], bool)
    assert status["url"]


def test_graph_is_exposed_for_the_topology_view(fake_model, in_memory_checkpointer):
    """app.py passes helper.core.graph to render_agentic_rag_graph()."""
    graph = AgenticRAGHelper().core.graph

    assert graph is not None
    assert hasattr(graph, "get_graph")


def test_legacy_config_constants_remain_importable():
    """Guards R1: `tools`, `core` and `agent` import these at module load.

    DESIGN.md Phase 2 · D-6 defers folding `config.py` into `Settings` to Phase 4
    precisely so this stays true. If the fold happens early, this fails loudly
    instead of Streamlit failing at import.
    """
    from agentic_rag import config

    for name in (
        "OLLAMA_MODEL",
        "OLLAMA_BASE_URL",
        "REDIS_URL",
        "EMBEDDING_MODEL",
        "CHROMA_PERSIST_DIR",
        "DEFAULT_NAMESPACE",
        "FILE_TOOL_ROOTS",
    ):
        assert getattr(config, name, None) is not None, f"config.{name} disappeared"


def test_tools_module_keeps_patchable_module_level_settings():
    """`tests/unit/test_phase0_hardening.py` monkeypatches this attribute.

    Turning it into a Settings lookup would make that test silently stop
    testing anything, so the attribute itself is pinned here.
    """
    import agentic_rag.tools as tools

    assert isinstance(tools.FILE_TOOL_ROOTS, str)
