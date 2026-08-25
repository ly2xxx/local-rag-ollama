"""1. Profile Component (Persona, Role & System Prompt Constraints).

Defines agent personas, system prompt templates, domain roles, and boundary rules
as described in Section 3 of Agent.md.
"""

from typing import Dict, Optional


class AgentProfile:
    """Represents the profile, persona, and role boundaries of an AI Agent."""

    DEFAULT_SYSTEM_PROMPT = (
        "You are an expert Agentic RAG Assistant equipped with dynamic tools for knowledge retrieval, "
        "document analysis, calculation, and file inspection.\n\n"
        "Guidelines:\n"
        "1. When answering questions about uploaded documents or domain facts, use the 'query_document_knowledge_base' tool.\n"
        "2. If you need to inspect raw file contents or paths, use the 'read_local_file' tool.\n"
        "3. For mathematical calculations, statistics, or evaluations, use the 'calculate_expression' tool.\n"
        "4. Always reason step-by-step (Thought -> Action -> Observation) before formulating your final answer.\n"
        "5. If retrieved context is insufficient or if you don't know the answer, honestly state your limitations."
    )

    def __init__(
        self,
        name: str = "AgenticRAG_Core",
        role: str = "Knowledge Retrieval & Reasoning Specialist",
        system_prompt: Optional[str] = None,
        custom_instructions: Optional[str] = None,
    ):
        self.name = name
        self.role = role
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        self.custom_instructions = custom_instructions or ""

    def get_full_prompt(self) -> str:
        """Returns the formatted system prompt combining persona and custom instructions."""
        if self.custom_instructions:
            return f"{self.system_prompt}\n\nAdditional Instructions:\n{self.custom_instructions}"
        return self.system_prompt

    def to_dict(self) -> Dict[str, str]:
        return {
            "name": self.name,
            "role": self.role,
            "system_prompt": self.system_prompt,
            "custom_instructions": self.custom_instructions,
        }
