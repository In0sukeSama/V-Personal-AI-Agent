"""
Model provider abstraction for V.

The Brain talks to this interface, never to a specific LLM SDK directly.
Providers translate V's neutral message/tool format into whatever their
backend needs, and normalize the response back into ModelResponse/ToolCall -
so no provider-specific object (Groq's, Gemini's, etc.) ever reaches brain.py.

Kept deliberately minimal - just what V's brain actually uses today.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class ToolCall:
    """One tool call requested by the model, already parsed - brain.py
    never touches a provider SDK's raw tool-call object."""
    id: str
    name: str
    arguments: dict


@dataclass
class ModelResponse:
    """Normalized model output. `raw_assistant_message` is a plain dict in
    V's own wire format (role/content/tool_calls) - it's what gets appended
    back into the conversation history so the next turn has the right
    context, without brain.py ever calling a provider SDK's own
    serialization method (e.g. Groq's message.model_dump())."""
    text: str
    tool_calls: list = field(default_factory=list)  # list[ToolCall]
    raw_assistant_message: dict = None


class ModelProviderError(Exception):
    """Normalized provider failure. Providers translate their own
    SDK-specific exceptions into this, so brain.py never needs
    provider-specific except clauses."""

    def __init__(self, error_type: str, message: str, retryable: bool = False):
        super().__init__(message)
        self.error_type = error_type  # "auth" | "rate_limit" | "timeout" | "invalid_response" | "unknown"
        self.message = message
        self.retryable = retryable


class ModelProvider(ABC):
    """Minimal interface every provider implements. Brain only ever calls
    generate() - nothing provider-specific is exposed beyond this."""

    @abstractmethod
    def generate(self, messages: list, tools: list = None, **kwargs) -> ModelResponse:
        """
        messages: list of {"role": ..., "content": ...} dicts (V's own wire
                  format - already includes tool-role results where needed).
        tools: tool schemas in the OpenAI-function-calling shape, exactly
               what ToolRegistry.get_llm_tools() already produces. Providers
               that need a different shape translate internally.
        kwargs: optional generation params (max_tokens, temperature, etc.) -
                a provider uses what it supports and ignores the rest.

        Returns a ModelResponse. Raises ModelProviderError on failure -
        never a raw SDK exception.
        """
        raise NotImplementedError
