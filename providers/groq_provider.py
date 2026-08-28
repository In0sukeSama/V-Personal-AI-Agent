"""
Groq provider - the only fully implemented provider for now.

Wraps the OpenAI-compatible SDK pointed at Groq's endpoint (exactly what
brain.py used to do directly). All Groq/OpenAI-SDK-specific code lives here
and nowhere else - brain.py only ever sees ModelResponse/ToolCall/
ModelProviderError from providers/base.py.
"""

import json
import openai
from openai import OpenAI

from providers.base import ModelProvider, ModelResponse, ToolCall, ModelProviderError

DEFAULT_MODEL = "openai/gpt-oss-120b"  # Groq's free-tier flagship (llama-3.3-70b was deprecated)


class GroqProvider(ModelProvider):
    def __init__(self, api_key: str = None, model: str = None):
        self._client = OpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
        )
        self._model = model or DEFAULT_MODEL

    def generate(self, messages: list, tools: list = None, **kwargs) -> ModelResponse:
        request_kwargs = {"model": self._model, "messages": messages}
        if tools:
            request_kwargs["tools"] = tools
        if "max_tokens" in kwargs:
            request_kwargs["max_tokens"] = kwargs["max_tokens"]
        if "frequency_penalty" in kwargs:
            request_kwargs["frequency_penalty"] = kwargs["frequency_penalty"]
        if "temperature" in kwargs:
            request_kwargs["temperature"] = kwargs["temperature"]

        try:
            response = self._client.chat.completions.create(**request_kwargs)
        except openai.AuthenticationError as e:
            raise ModelProviderError("auth", str(e), retryable=False)
        except openai.RateLimitError as e:
            raise ModelProviderError("rate_limit", str(e), retryable=True)
        except openai.APITimeoutError as e:
            raise ModelProviderError("timeout", str(e), retryable=True)
        except openai.APIError as e:
            raise ModelProviderError("unknown", str(e), retryable=False)
        except Exception as e:
            raise ModelProviderError("unknown", str(e), retryable=False)

        return self._normalize(response)

    def _normalize(self, response) -> ModelResponse:
        try:
            choice = response.choices[0]
            message = choice.message
        except (AttributeError, IndexError) as e:
            raise ModelProviderError("invalid_response", f"Unexpected response shape: {e}", retryable=False)

        tool_calls = []
        for tc in (message.tool_calls or []):
            try:
                arguments = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                arguments = {}
            tool_calls.append(ToolCall(id=tc.id, name=tc.function.name, arguments=arguments))

        # Plain dict, in V's own wire format - this is what gets appended
        # back into conversation history for the next turn. Built here, once,
        # so brain.py never calls a provider SDK's own serialization method.
        raw_assistant_message = message.model_dump(exclude_none=True)

        return ModelResponse(
            text=message.content or "",
            tool_calls=tool_calls,
            raw_assistant_message=raw_assistant_message,
        )
