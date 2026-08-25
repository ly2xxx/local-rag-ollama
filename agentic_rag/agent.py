"""Agentic RAG Helper & Facade.

High-level interface for Streamlit and external applications to interact with
the LangGraph Agentic RAG architecture, document manager, and Redis memory checkpointer.
"""

import os
import logging
from typing import List, Tuple, Dict, Any, Optional
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from .core import AgentCore
from .tools import doc_manager
from .profile import AgentProfile

logger = logging.getLogger(__name__)


class AgenticRAGHelper:
    """Facade for the Agentic RAG System."""

    def __init__(
        self,
        model_name: str = "glm-5.2:cloud",
        base_url: str = "http://127.0.0.1:11434",
        redis_url: str = "redis://localhost:6379",
        system_prompt: Optional[str] = None,
    ):
        self.model_name = model_name
        self.base_url = base_url
        self.redis_url = redis_url
        
        self.profile = AgentProfile(system_prompt=system_prompt)
        self.core = AgentCore(
            model_name=self.model_name,
            base_url=self.base_url,
            redis_url=self.redis_url,
            profile=self.profile,
        )

    def ingest_document(self, file_path: str) -> str:
        """Ingests a PDF document into the RAG vector store."""
        return doc_manager.ingest_pdf(file_path)

    def get_ingested_files(self) -> List[str]:
        """Returns the list of ingested document filenames."""
        return list(doc_manager.ingested_files)

    def clear_documents(self):
        """Clears the ingested document knowledge base."""
        doc_manager.clear()

    def get_redis_status(self) -> Dict[str, Any]:
        """Returns connection status details for the Redis memory checkpointer."""
        return {
            "active": self.core.memory_manager.is_redis_active,
            "message": self.core.memory_manager.status_message,
            "url": self.redis_url,
        }

    def ask(self, query: str, thread_id: str = "default_session") -> Dict[str, Any]:
        """Runs the ReAct agent on the query within a thread session.

        Returns a dictionary containing:
        - 'answer': The final textual response from the agent.
        - 'scratchpad': List of intermediate tool calls and observations.
        - 'messages': All message objects from the graph execution.
        """
        try:
            result = self.core.run(message=query, thread_id=thread_id)
            messages = result.get("messages", [])
            
            final_answer = "No response generated."
            scratchpad: List[Dict[str, Any]] = []

            # Parse message trajectory for scratchpad and final answer
            for msg in messages:
                if isinstance(msg, AIMessage):
                    if msg.tool_calls:
                        for tc in msg.tool_calls:
                            scratchpad.append({
                                "type": "tool_call",
                                "name": tc.get("name"),
                                "args": tc.get("args"),
                                "id": tc.get("id"),
                            })
                    if msg.content and isinstance(msg.content, str) and msg.content.strip():
                        # The latest non-empty AI content represents the answer (or intermediate reasoning)
                        final_answer = msg.content.strip()
                elif isinstance(msg, ToolMessage):
                    scratchpad.append({
                        "type": "tool_observation",
                        "name": msg.name,
                        "content": msg.content,
                        "tool_call_id": msg.tool_call_id,
                    })

            return {
                "answer": final_answer,
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
