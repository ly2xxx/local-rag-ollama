"""The synchronous agent path against the REAL Redis checkpointer.

Covers DESIGN.md Phase 2 · R2. `RedisSaver` implements the sync checkpoint
methods; `AsyncRedisSaver` implements sync `get_tuple`/`put`/`put_writes` but
NOT sync `list`. If Phase 2 swaps the manager's default saver globally instead
of handing one out per calling context, this is where it shows up — the hermetic
unit tests use `MemorySaver` and would stay green.

The model is faked; only the checkpointer is real.
"""

import uuid

import pytest
from langchain_core.messages import AIMessage

from agentic_rag.agent import AgenticRAGHelper
from agentic_rag.settings import Settings
from tests.fakes.chat_model import FakeToolCallingModel

pytestmark = pytest.mark.integration


def _redis_available(url: str) -> bool:
    import redis

    try:
        redis.from_url(url, socket_connect_timeout=1.0, socket_timeout=1.0).ping()
        return True
    except Exception:
        return False


@pytest.fixture
def redis_url():
    url = Settings(_env_file=None).redis_url
    if not _redis_available(url):
        pytest.skip(f"Redis not reachable at {url}")
    return url


@pytest.fixture
def fake_model(monkeypatch):
    def _install(*responses: AIMessage) -> FakeToolCallingModel:
        model = FakeToolCallingModel(responses=list(responses))
        monkeypatch.setattr("agentic_rag.core.ChatOllama", lambda **kwargs: model)
        return model

    return _install


@pytest.fixture
def thread_id():
    # Unique per run so repeated runs never inherit stale checkpoint state.
    return f"itest-{uuid.uuid4().hex[:12]}"


def test_sync_turn_completes_with_the_real_redis_checkpointer(
    fake_model, redis_url, thread_id
):
    fake_model(AIMessage(content="Answered through the Redis checkpointer."))
    helper = AgenticRAGHelper(redis_url=redis_url)

    assert helper.get_redis_status()["active"] is True, "expected the real RedisSaver"

    result = helper.ask("hello", thread_id=thread_id)

    assert result["success"] is True
    assert result["answer"] == "Answered through the Redis checkpointer."


def test_conversation_survives_a_new_helper_instance(fake_model, redis_url, thread_id):
    """Simulates a Streamlit restart: state lives in Redis, not the process."""
    fake_model(AIMessage(content="First answer."), AIMessage(content="Second answer."))

    first_helper = AgenticRAGHelper(redis_url=redis_url)
    first_helper.ask("remember this question", thread_id=thread_id)

    second_helper = AgenticRAGHelper(redis_url=redis_url)
    second = second_helper.ask("follow up", thread_id=thread_id)

    contents = [m.content for m in second["messages"] if getattr(m, "content", None)]
    assert "remember this question" in contents
    assert "First answer." in contents


def test_inspect_state_reads_back_the_checkpoint(fake_model, redis_url, thread_id):
    """`ShortTermMemoryManager.inspect_state` uses get_state -> sync get_tuple."""
    fake_model(AIMessage(content="Stored."))
    helper = AgenticRAGHelper(redis_url=redis_url)
    helper.ask("a question worth checkpointing", thread_id=thread_id)

    state = helper.core.memory_manager.inspect_state(thread_id, helper.core.graph)

    assert state is not None
    contents = [m.content for m in state["messages"] if getattr(m, "content", None)]
    assert "a question worth checkpointing" in contents
