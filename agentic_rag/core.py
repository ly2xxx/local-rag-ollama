"""Agent Core Component (ReAct Reasoning Loop & LangGraph State Machine).

Implements the central ReAct (Reasoning + Action) Agent Core using LangGraph,
binding Ollama LLM, dynamic tools, system prompt profile, and the Redis checkpointer
as described in Section 3 of Agent.md.
"""

import logging
from typing import List, Optional, Any
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent

from .profile import AgentProfile
from .tools import get_default_tools, doc_manager
from .memory.short_term import ShortTermMemoryManager

logger = logging.getLogger(__name__)


class AgentCore:
    """The central Agent Core orchestrating Thought, Action, and Observation loops."""

    def __init__(
        self,
        model_name: str = "glm-5.2:cloud",
        base_url: str = "http://127.0.0.1:11434",
        redis_url: str = "redis://localhost:6379",
        profile: Optional[AgentProfile] = None,
        custom_tools: Optional[List[Any]] = None,
    ):
        self.model_name = model_name
        self.base_url = base_url
        self.profile = profile or AgentProfile()
        self.memory_manager = ShortTermMemoryManager(redis_url=redis_url)
        
        # Initialize Ollama model
        self.llm = ChatOllama(
            model=self.model_name,
            base_url=self.base_url,
            temperature=0.2,
        )

        # Register tools
        self.tools = custom_tools or get_default_tools()

        # Build LangGraph ReAct Agent with checkpointer
        self.graph = self._build_graph()

    def _build_graph(self):
        """Compiles the LangGraph ReAct agent with checkpointer and profile instructions."""
        checkpointer = self.memory_manager.get_checkpointer()
        system_prompt = self.profile.get_full_prompt()
        
        # LangGraph create_react_agent accepts state_modifier / prompt
        try:
            agent = create_react_agent(
                model=self.llm,
                tools=self.tools,
                checkpointer=checkpointer,
                prompt=system_prompt,
            )
        except TypeError:
            # Fallback for older/alternate signature using state_modifier
            agent = create_react_agent(
                model=self.llm,
                tools=self.tools,
                checkpointer=checkpointer,
                state_modifier=system_prompt,
            )
        return agent

    def run(self, message: str, thread_id: str = "default_session") -> dict:
        """Executes the agent for a user message within a thread session."""
        config = self.memory_manager.get_thread_config(thread_id)
        input_payload = {"messages": [("user", message)]}
        
        # Invoke LangGraph
        result = self.graph.invoke(input_payload, config=config)
        return result
