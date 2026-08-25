"""3. Short-term Memory Component (Multi-turn History & Agent Scratchpads).

Provides short-term working memory and state checkpointing for LangGraph agents.
Backs multi-turn conversation states and agent scratchpads to Redis via LangGraph's
RedisSaver / AsyncRedisSaver checkpointer, with automatic in-memory fallback (MemorySaver)
if Redis is unreachable.
"""

import logging
from typing import Tuple, Any, Optional, Dict, List
import redis
from langgraph.checkpoint.memory import MemorySaver

logger = logging.getLogger(__name__)

# Check for RedisSaver / AsyncRedisSaver
try:
    from langgraph.checkpoint.redis import RedisSaver
except ImportError:
    RedisSaver = None

try:
    from langgraph.checkpoint.redis.aio import AsyncRedisSaver
except ImportError:
    AsyncRedisSaver = None


class ShortTermMemoryManager:
    """Manages short-term state persistence and checkpointers for LangGraph agents."""

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_url = redis_url
        self.checkpointer, self.is_redis_active, self.status_message = self._init_checkpointer()

    def _init_checkpointer(self) -> Tuple[Any, bool, str]:
        """Initializes the Redis checkpointer or falls back to MemorySaver if Redis is offline."""
        if RedisSaver is None:
            msg = "langgraph-checkpoint-redis not installed. Using in-memory MemorySaver fallback."
            logger.warning(msg)
            return MemorySaver(), False, msg

        try:
            # Test connection with a short timeout
            client = redis.from_url(self.redis_url, socket_connect_timeout=1.5, socket_timeout=1.5)
            client.ping()
            
            # Redis is active, construct RedisSaver
            checkpointer = RedisSaver(redis_url=self.redis_url)
            # Ensure setup/index creation if available
            if hasattr(checkpointer, "setup"):
                try:
                    checkpointer.setup()
                except Exception as setup_err:
                    logger.debug(f"RedisSaver setup notice: {setup_err}")

            msg = f"Connected to Redis checkpointer at {self.redis_url}"
            logger.info(msg)
            return checkpointer, True, msg
        except Exception as e:
            fallback_msg = f"Redis offline at {self.redis_url} ({e}). Using in-memory checkpointer fallback."
            logger.warning(fallback_msg)
            return MemorySaver(), False, fallback_msg

    def get_checkpointer(self) -> Any:
        """Returns the active LangGraph checkpointer instance."""
        return self.checkpointer

    def get_thread_config(self, thread_id: str) -> Dict[str, Any]:
        """Generates the LangGraph runnable config dict for a given thread_id."""
        return {"configurable": {"thread_id": thread_id}}

    def inspect_state(self, thread_id: str, graph: Any) -> Optional[Dict[str, Any]]:
        """Extracts the persisted state and messages for a thread from the checkpointer."""
        try:
            config = self.get_thread_config(thread_id)
            state = graph.get_state(config)
            if state and state.values:
                return state.values
        except Exception as e:
            logger.warning(f"Could not retrieve state for thread {thread_id}: {e}")
        return None


def get_redis_checkpointer(redis_url: str = "redis://localhost:6379") -> ShortTermMemoryManager:
    """Helper factory to obtain a ShortTermMemoryManager instance."""
    return ShortTermMemoryManager(redis_url=redis_url)
