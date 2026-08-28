"""
Provider factory - selects and constructs a ModelProvider based on
configuration, so brain.py never hardcodes which provider it's using.
"""

import os

from providers.base import ModelProvider, ModelResponse, ToolCall, ModelProviderError  # noqa: F401


def create_model_provider(provider_name: str = None, api_key: str = None, model: str = None) -> ModelProvider:
    """Build a configured ModelProvider. Reads V_MODEL_PROVIDER from the
    environment if provider_name isn't given explicitly. Defaults to Groq -
    the only fully implemented provider today."""
    provider_name = (provider_name or os.getenv("V_MODEL_PROVIDER", "groq")).lower()

    if provider_name == "groq":
        from providers.groq_provider import GroqProvider
        return GroqProvider(api_key=api_key or os.getenv("GROQ_API_KEY"), model=model)

    raise ValueError(
        f"Unknown model provider: '{provider_name}'. Only 'groq' is currently implemented. "
        f"Set V_MODEL_PROVIDER=groq or leave it unset."
    )
