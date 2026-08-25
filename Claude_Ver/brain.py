"""
The "brain" of JARVIS: sends messages to the Claude API, handles tool calls,
and returns a final spoken-ready text response.
"""

import os
from anthropic import Anthropic
from tools import TOOL_SCHEMAS, execute_tool

MODEL = "claude-sonnet-4-6"

SYSTEM_PROMPT = """You are JARVIS, a witty, articulate AI assistant inspired by the one from Iron Man.
You address the user as "sir" (or their name if they tell you one), speak with dry British wit,
stay calm and composed at all times, and are extremely competent and efficient.

Keep responses concise and conversational since they will be read aloud by a text-to-speech
system - avoid long lists, avoid markdown formatting, avoid asterisks or headers. Speak in
natural spoken sentences.

When the user asks you to open an application, visit a website, or check the weather,
use the tools available to you rather than just describing what you would do.
"""


class JarvisBrain:
    def __init__(self, api_key: str = None):
        self.client = Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))
        self.history = []

    def ask(self, user_text: str) -> str:
        """Send user text to Claude, handle any tool calls, return final reply text."""
        self.history.append({"role": "user", "content": user_text})

        response = self.client.messages.create(
            model=MODEL,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            tools=TOOL_SCHEMAS,
            messages=self.history,
        )

        # Keep handling tool calls until Claude gives a final text answer
        while response.stop_reason == "tool_use":
            self.history.append({"role": "assistant", "content": response.content})

            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    result_text = execute_tool(block.name, block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result_text,
                    })

            self.history.append({"role": "user", "content": tool_results})

            response = self.client.messages.create(
                model=MODEL,
                max_tokens=1024,
                system=SYSTEM_PROMPT,
                tools=TOOL_SCHEMAS,
                messages=self.history,
            )

        # Final text response
        final_text = "".join(
            block.text for block in response.content if block.type == "text"
        )
        self.history.append({"role": "assistant", "content": response.content})
        return final_text.strip()
