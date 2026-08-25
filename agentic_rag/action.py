"""5. Action Component (Execution & Output Layer).

Manages tool execution formatting, parameter validation, and outbound action dispatch
as described in Section 3 of Agent.md.
"""

from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class AgentAction(BaseModel):
    """Represents an action decided by the agent core to be executed."""

    tool_name: str = Field(description="The name of the target tool to invoke")
    tool_input: Dict[str, Any] = Field(default_factory=dict, description="Parameters to pass to the tool")
    thought: Optional[str] = Field(default=None, description="The agent's internal reasoning leading to this action")


class ActionExecutor:
    """Dispatches and executes actions against registered tool handlers.

    (Extensible placeholder for sandboxed code execution, API dispatch, and structured outputs).
    """

    def __init__(self, tools_map: Optional[Dict[str, Any]] = None):
        self.tools_map = tools_map or {}

    def execute(self, action: AgentAction) -> Any:
        """Executes a given tool action and returns the raw output."""
        if action.tool_name not in self.tools_map:
            raise ValueError(f"Tool '{action.tool_name}' not found in registered tools.")
        tool = self.tools_map[action.tool_name]
        return tool.invoke(action.tool_input)
