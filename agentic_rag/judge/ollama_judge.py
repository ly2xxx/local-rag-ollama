"""A DeepEval judge model backed by an Ollama endpoint.

DeepEval calls generate() with an optional pydantic schema when it wants
structured verdicts; we ask the model for JSON and validate it back.
"""

from __future__ import annotations

import json
import os

from typing import Optional, Any
from openai import OpenAI

try:
    from deepeval.models import DeepEvalBaseLLM
except ImportError:
    # Graceful fallback if deepeval is not installed
    class DeepEvalBaseLLM:
        pass


class OllamaJudge(DeepEvalBaseLLM):
    def __init__(
        self,
        model: str = "glm-5.2:cloud",
        base_url: Optional[str] = None,
        api_key: str = "ollama",
    ):
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
        if not self.base_url.endswith("/v1"):
            self.base_url = self.base_url.rstrip("/") + "/v1"
        self.api_key = os.getenv("OLLAMA_API_KEY", api_key)
        self.model = model
        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)

    def load_model(self):
        return self.client

    def generate(self, prompt: str, schema: Any = None) -> str:
        if schema is not None and hasattr(schema, "model_json_schema"):
            prompt += (
                "\n\nReply with ONLY a JSON object matching this schema, no prose:\n"
                + json.dumps(schema.model_json_schema())
            )
        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.choices[0].message.content or ""
        if schema is not None and hasattr(schema, "model_validate_json"):
            cleaned = text.strip()
            if "{" in cleaned and "}" in cleaned:
                cleaned = cleaned[cleaned.index("{") : cleaned.rindex("}") + 1]
            return schema.model_validate_json(cleaned)
        return text

    async def a_generate(self, prompt: str, schema=None):
        return self.generate(prompt, schema)

    def get_model_name(self):
        return f"OllamaJudge({self.model})"
