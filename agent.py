"""Agent module with dynamic LLM routing between ToolCallingAgent and CodeAgent using LiteLLM."""

import os
import pypdf
import litellm
from dotenv import load_dotenv
from smolagents import CodeAgent, LiteLLMModel, ToolCallingAgent, tool

load_dotenv()


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
        model_id: str = None,
        api_base: str = None,
        add_base_tools: bool = False,
    ):
        raw_model = os.getenv("OLLAMA_MODEL", "glm-5.2:cloud")
        default_model_id = f"ollama/{raw_model}" if not raw_model.startswith("ollama/") else raw_model
        self.model_id = model_id or default_model_id
        self.api_base = api_base or os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
        self.add_base_tools = add_base_tools

        self.model = LiteLLMModel(
            model_id=self.model_id,
            api_base=self.api_base,
            num_ctx=8192,
        )
        self.tools = [read_pdf, read_file]

    def route_agent_type(self, prompt: str) -> str:
        """Use LiteLLM to classify whether user input needs CodeAgent or ToolCallingAgent."""
        routing_prompt = (
            "You are a router that decides which AI agent mode to use for a request.\n\n"
            "Agent Modes:\n"
            "- CODE_AGENT: Use when the user asks for math calculations, averages, statistics, writing code, executing python, multi-step logic, or data processing.\n"
            "- TOOL_CALLING: Use for standard document retrieval, simple file reading, questions about files, or general conversational Q&A.\n\n"
            f"User Request: {prompt}\n\n"
            "Return EXACTLY ONE WORD choice: CODE_AGENT or TOOL_CALLING."
        )
        try:
            response = litellm.completion(
                model=self.model_id,
                api_base=self.api_base,
                messages=[{"role": "user", "content": routing_prompt}],
            )
            content = str(response.choices[0].message.content).strip().upper()
            print(f"[LLM Router Output]: '{content}'")
            if "CODE" in content:
                return "CODE_AGENT"
            return "TOOL_CALLING"
        except Exception as e:
            print(f"[Router Warning] Routing failed ({e}), defaulting to TOOL_CALLING")
            return "TOOL_CALLING"

    def ask(self, prompt: str) -> str:
        """Dynamically route and run the query with CodeAgent or ToolCallingAgent."""
        agent_mode = self.route_agent_type(prompt)
        print(f"[LLM Router] Prompt routed to -> {agent_mode}")

        if agent_mode == "CODE_AGENT":
            agent = CodeAgent(
                tools=[],
                model=self.model,
                add_base_tools=False,
            )
        else:
            agent = ToolCallingAgent(
                tools=self.tools,
                model=self.model,
                add_base_tools=self.add_base_tools,
            )

        result = agent.run(prompt)
        return f"**(Selected Mode: `{agent_mode}`)**\n\n{result}"


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
    print("Testing smolagents + litellm router...")
    helper = create_agent()
    res = helper.ask("what's the average of 80, 90, 99, 70 ?")
    print(res)
