"""6. Observation Component (Environment Feedback & Perception).

Parses tool return values, execution logs, API status codes, and environment feedback
to feed back into the Agent Core reasoning loop as described in Section 3 of Agent.md.
"""

from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class AgentObservation(BaseModel):
    """Represents the observation returned from tool or environment execution."""

    tool_name: str = Field(description="Tool that produced this observation")
    raw_output: Any = Field(description="Raw return value or exception from the environment")
    parsed_content: str = Field(description="Cleaned, human/LLM-readable observation string")
    is_error: bool = Field(default=False, description="Whether the tool execution failed")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Execution metadata (latency, status, tokens)")


class ObservationProcessor:
    """Processes environment returns into structured observations for agent perception."""

    @staticmethod
    def process_result(tool_name: str, result: Any) -> AgentObservation:
        if isinstance(result, Exception):
            return AgentObservation(
                tool_name=tool_name,
                raw_output=str(result),
                parsed_content=f"Error executing tool '{tool_name}': {str(result)}",
                is_error=True,
            )
        return AgentObservation(
            tool_name=tool_name,
            raw_output=result,
            parsed_content=str(result),
            is_error=False,
        )
