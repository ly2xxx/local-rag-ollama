"""A chat model that emits scripted tool calls.

`langchain_core`'s `GenericFakeChatModel` cannot script `tool_calls`, which is
exactly what graph-path tests need, so this is written locally. Phase 1 only
needs it for message-shape tests; Phase 4 drives every graph branch with it.
"""

from typing import Any, Iterator, List, Optional, Sequence

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult


class FakeToolCallingModel(BaseChatModel):
    """Replays a fixed list of `AIMessage`s, one per invocation."""

    responses: List[AIMessage] = []
    index: int = 0
    bound_tools: Optional[Sequence[Any]] = None

    @property
    def _llm_type(self) -> str:
        return "fake-tool-calling"

    def _next(self) -> AIMessage:
        if not self.responses:
            return AIMessage(content="")
        message = self.responses[min(self.index, len(self.responses) - 1)]
        self.index += 1
        return message

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=self._next())])

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> "FakeToolCallingModel":
        self.bound_tools = list(tools)
        return self

    def reset(self) -> None:
        self.index = 0
