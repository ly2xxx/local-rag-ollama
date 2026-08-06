"""Agent module using smolagents and LiteLLMModel connected to Ollama."""

import os
from smolagents import LiteLLMModel, ToolCallingAgent


class SmolAgentHelper:
    def __init__(
        self,
        model_id: str = "ollama/glm-5.2:cloud",
        api_base: str = "http://127.0.0.1:11434",
        add_base_tools: bool = False,
    ):
        self.model_id = model_id
        self.api_base = api_base
        self.add_base_tools = add_base_tools

        self.model = LiteLLMModel(
            model_id=self.model_id,
            api_base=self.api_base,
            num_ctx=8192,
        )

        self.agent = ToolCallingAgent(
            tools=[],
            model=self.model,
            add_base_tools=self.add_base_tools,
        )

    def ask(self, prompt: str) -> str:
        """Run the agent with a given prompt and return the result."""
        result = self.agent.run(prompt)
        return str(result)


def create_agent(
    model_id: str = "ollama/glm-5.2:cloud",
    api_base: str = "http://127.0.0.1:11434",
    add_base_tools: bool = False,
) -> SmolAgentHelper:
    return SmolAgentHelper(
        model_id=model_id,
        api_base=api_base,
        add_base_tools=add_base_tools,
    )


if __name__ == "__main__":
    print("Testing smolagents + LiteLLMModel agent...")
    agent_helper = create_agent(add_base_tools=False)
    response = agent_helper.ask("Where is Singapore located?")
    print("Agent Response:", response)
