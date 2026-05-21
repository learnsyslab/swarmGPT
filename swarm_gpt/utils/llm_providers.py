"""Shared LLM routing helpers (OpenAI vs Ollama)."""

from __future__ import annotations

import json
import logging
import os
from typing import Final, Literal
from urllib.request import urlopen

from openai import OpenAI

logger = logging.getLogger(__name__)

# Ollama exposes an OpenAI-compatible /v1/responses API.
OLLAMA_OPENAI_BASE_URL: Final = os.getenv("OLLAMA_OPENAI_BASE_URL", "http://localhost:11434/v1")
OLLAMA_API_KEY: Final = os.getenv("OLLAMA_API_KEY", "ollama")

RESPONSES_MAX_OUTPUT_TOKENS: Final = 4096
RESPONSES_TEMPERATURE: Final = 0.0

LLMProvider = Literal["openai", "ollama"]

PROVIDER_LABEL_OPENAI: Final = "ChatGPT / OpenAI"
PROVIDER_LABEL_OLLAMA: Final = "Ollama (local)"

# Shown in the browser UI; users can type another OpenAI id if needed.
DEFAULT_OPENAI_MODEL_CHOICES: tuple[str, ...] = (
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4-turbo",
    "gpt-3.5-turbo",
    "o3-mini",
)


def default_openai_model() -> str:
    """Default OpenAI model id for ``responses.create``."""
    return DEFAULT_OPENAI_MODEL_CHOICES[0]


def prepare_responses_messages(
    messages: list[dict[str, str]],
) -> tuple[list[dict[str, str]], str | None]:
    """Split chat-style messages for ``responses.create``.

    Ollama's ``/v1/responses`` returns empty ``output_text`` when multiple ``system``
    messages are interleaved with ``user``/``assistant`` turns. Hoist all system content
    into ``instructions`` and keep only dialogue roles in ``input``.
    """
    system_parts: list[str] = []
    input_messages: list[dict[str, str]] = []
    for message in messages:
        if message["role"] == "system":
            system_parts.append(message["content"])
        else:
            input_messages.append(message)
    instructions = "\n\n".join(system_parts) if system_parts else None
    return input_messages, instructions


def openai_client_for_provider(provider: LLMProvider) -> OpenAI:
    """Build an OpenAI SDK client for cloud OpenAI or local Ollama (``responses`` API)."""
    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Export it or select Ollama (local) in the UI."
            )
        return OpenAI(api_key=api_key)
    if provider == "ollama":
        return OpenAI(base_url=OLLAMA_OPENAI_BASE_URL, api_key=OLLAMA_API_KEY)
    raise ValueError(f"Unknown LLM provider: {provider!r}")


def ollama_installed_model_names() -> list[str]:
    """Return sorted Ollama model names, or [] if the daemon is unreachable."""
    try:
        base_url = OLLAMA_OPENAI_BASE_URL.rsplit("/v1", 1)[0].rstrip("/")
        with urlopen(f"{base_url}/api/tags", timeout=0.75) as response:
            payload = json.loads(response.read().decode("utf-8"))
        names = []
        for model in payload.get("models", []):
            name = model.get("model") or model.get("name")
            if name:
                names.append(str(name))
        return sorted(set(names))
    except Exception as e:
        logger.warning("Could not list Ollama models (is `ollama serve` running?): %s", e)
        return []
