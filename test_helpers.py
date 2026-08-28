"""
Shared test helpers for faking the ModelProvider interface (providers/base.py)
instead of a specific SDK's raw response objects. Used across test files so
tests exercise brain.py exactly the way it's actually wired - through the
provider abstraction, not a mocked-up Groq/OpenAI client.
"""

import json
from providers.base import ModelProvider, ModelResponse, ToolCall


class FakeProvider(ModelProvider):
    """Returns pre-scripted ModelResponse objects in order, one per
    generate() call. Tracks call_count so tests can assert exactly how many
    model calls actually happened."""

    def __init__(self, responses):
        self._responses = iter(responses)
        self.call_count = 0

    def generate(self, messages, tools=None, **kwargs):
        self.call_count += 1
        return next(self._responses)


def make_tool_call_response(tool_calls_spec):
    """Build a ModelResponse representing the model requesting one or more
    tool calls. tool_calls_spec: list of (name, args_dict, call_id) tuples."""
    tool_calls = [ToolCall(id=cid, name=name, arguments=args) for name, args, cid in tool_calls_spec]
    raw_assistant_message = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": cid,
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(args)},
            }
            for name, args, cid in tool_calls_spec
        ],
    }
    return ModelResponse(text="", tool_calls=tool_calls, raw_assistant_message=raw_assistant_message)


def make_text_response(text):
    """Build a ModelResponse representing a plain final text reply, no tool calls."""
    return ModelResponse(text=text or "", tool_calls=[], raw_assistant_message={"role": "assistant", "content": text})
