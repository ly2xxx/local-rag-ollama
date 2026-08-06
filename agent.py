"""Agent module using smolagents, LiteLLMModel, and custom PDF / file reading tools."""

import os
import pypdf
from smolagents import CodeAgent, LiteLLMModel, ToolCallingAgent, tool


@tool
def read_pdf(file_path: str) -> str:
    """Reads a PDF file and returns its text content page by page.

    Args:
        file_path: The file path to the PDF document to read.

    Returns:
        str: The extracted text content from the PDF file.
    """
    if not os.path.exists(file_path):
        return f"Error: File not found at path '{file_path}'"

    try:
        reader = pypdf.PdfReader(file_path)
        pages_text = []
        for idx, page in enumerate(reader.pages):
            text = page.extract_text()
            if text and text.strip():
                pages_text.append(f"--- Page {idx + 1} ---\n{text.strip()}")
        if not pages_text:
            return f"PDF file '{file_path}' contains no extractable text."
        return "\n\n".join(pages_text)
    except Exception as e:
        return f"Error reading PDF '{file_path}': {str(e)}"


@tool
def read_file(file_path: str) -> str:
    """Opens a text or JSON file from the local directory and reads its contents.

    Args:
        file_path: The path or name of the file to open.

    Returns:
        str: The contents of the specified file.
    """
    if not os.path.exists(file_path):
        return f"Error: File not found at path '{file_path}'"

    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception as e:
        return f"Error reading file '{file_path}': {str(e)}"


class SmolAgentHelper:
    def __init__(
        self,
        model_id: str = "ollama/glm-5.2:cloud",
        api_base: str = "http://127.0.0.1:11434",
        add_base_tools: bool = False,
        use_code_agent: bool = False,
    ):
        self.model_id = model_id
        self.api_base = api_base
        self.add_base_tools = add_base_tools

        self.model = LiteLLMModel(
            model_id=self.model_id,
            api_base=self.api_base,
            num_ctx=8192,
        )

        tools = [read_pdf, read_file]

        if use_code_agent:
            self.agent = CodeAgent(
                tools=tools,
                model=self.model,
                add_base_tools=self.add_base_tools,
            )
        else:
            self.agent = ToolCallingAgent(
                tools=tools,
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
    use_code_agent: bool = False,
) -> SmolAgentHelper:
    return SmolAgentHelper(
        model_id=model_id,
        api_base=api_base,
        add_base_tools=add_base_tools,
        use_code_agent=use_code_agent,
    )


if __name__ == "__main__":
    print("Testing smolagents + read_pdf / read_file tools...")
    agent_helper = create_agent(add_base_tools=False)
    response = agent_helper.ask("Read the file test.txt using your tools.")
    print("Agent Response:", response)
