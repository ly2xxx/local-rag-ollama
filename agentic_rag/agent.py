"""Agentic RAG Helper & Facade.

High-level interface for Streamlit and external applications to interact with
the LangGraph Agentic RAG architecture, document manager, and Redis memory checkpointer.
"""

import os
import logging
from typing import List, Tuple, Dict, Any, Optional
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from .core import AgentCore
from .tools import doc_registry
from .profile import AgentProfile
from .config import OLLAMA_MODEL, OLLAMA_BASE_URL, REDIS_URL, DEFAULT_NAMESPACE

logger = logging.getLogger(__name__)


def _message_text(content: Any) -> str:
    """Flattens message content (a string, or a list of content blocks) to text."""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        return "\n".join(part for part in parts if part).strip()
    return ""


def _extract_final_answer(messages: List[Any]) -> str:
    """Returns the agent's final answer from a message trajectory.

    Scans backwards for the last AI message that is *not* a tool-call request.
    Scanning forwards would let reasoning text emitted alongside a tool call
    overwrite the real answer.
    """
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None):
            text = _message_text(msg.content)
            if text:
                return text

    # Fallback: a model that packs its answer into the same message as a tool call.
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            text = _message_text(msg.content)
            if text:
                return text

    return "No response generated."


class AgenticRAGHelper:
    """Facade for the Agentic RAG System."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        base_url: Optional[str] = None,
        redis_url: Optional[str] = None,
        system_prompt: Optional[str] = None,
        namespace: Optional[str] = None,
    ):
        self.model_name = model_name or OLLAMA_MODEL
        self.base_url = base_url or OLLAMA_BASE_URL
        self.redis_url = redis_url or REDIS_URL
        # When set, pins every thread to one knowledge base. When left as None,
        # each thread_id gets its own isolated knowledge base.
        self.namespace = namespace

        self.profile = AgentProfile(system_prompt=system_prompt)
        self.core = AgentCore(
            model_name=self.model_name,
            base_url=self.base_url,
            redis_url=self.redis_url,
            profile=self.profile,
        )

    def _resolve_namespace(
        self, namespace: Optional[str] = None, thread_id: Optional[str] = None
    ) -> str:
        return namespace or self.namespace or thread_id or DEFAULT_NAMESPACE

    def ingest_document(
        self,
        file_path: str,
        display_name: Optional[str] = None,
        thread_id: Optional[str] = None,
        namespace: Optional[str] = None,
    ) -> str:
        """Ingests a PDF document into the knowledge base for this session.

        `display_name` preserves the original filename when the caller hands us
        a temp file (as Streamlit's uploader does).
        """
        manager = doc_registry.get(self._resolve_namespace(namespace, thread_id))
        return manager.ingest_pdf(file_path, display_name=display_name)

    def get_ingested_files(
        self, thread_id: Optional[str] = None, namespace: Optional[str] = None
    ) -> List[str]:
        """Returns the list of ingested document filenames for this session."""
        manager = doc_registry.get(self._resolve_namespace(namespace, thread_id))
        return list(manager.ingested_files)

    def clear_documents(
        self, thread_id: Optional[str] = None, namespace: Optional[str] = None
    ):
        """Clears the ingested document knowledge base for this session."""
        doc_registry.get(self._resolve_namespace(namespace, thread_id)).clear()

    def get_redis_status(self) -> Dict[str, Any]:
        """Returns connection status details for the Redis memory checkpointer."""
        return {
            "active": self.core.memory_manager.is_redis_active,
            "message": self.core.memory_manager.status_message,
            "url": self.redis_url,
        }

    def ask(
        self,
        query: str,
        thread_id: str = "default_session",
        namespace: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Runs the ReAct agent on the query within a thread session.

        Returns a dictionary containing:
        - 'answer': The final textual response from the agent.
        - 'scratchpad': List of intermediate tool calls and observations.
        - 'messages': All message objects from the graph execution.
        """
        try:
            result = self.core.run(
                message=query,
                thread_id=thread_id,
                namespace=self._resolve_namespace(namespace, thread_id),
            )
            messages = result.get("messages", [])

            scratchpad: List[Dict[str, Any]] = []

            # Walk the trajectory forwards for the scratchpad (chronological order).
            for msg in messages:
                if isinstance(msg, AIMessage):
                    for tc in getattr(msg, "tool_calls", None) or []:
                        scratchpad.append({
                            "type": "tool_call",
                            "name": tc.get("name"),
                            "args": tc.get("args"),
                            "id": tc.get("id"),
                        })
                elif isinstance(msg, ToolMessage):
                    scratchpad.append({
                        "type": "tool_observation",
                        "name": msg.name,
                        "content": msg.content,
                        "tool_call_id": msg.tool_call_id,
                    })

            return {
                "answer": _extract_final_answer(messages),
                "scratchpad": scratchpad,
                "messages": messages,
                "success": True,
            }
        except Exception as e:
            logger.error(f"Error running Agentic RAG: {e}", exc_info=True)
            return {
                "answer": f"Error running Agentic RAG: {str(e)}",
                "scratchpad": [],
                "messages": [],
                "success": False,
            }
