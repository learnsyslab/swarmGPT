"""Shared LLM routing helpers (OpenAI vs Ollama)."""

from __future__ import annotations

import json
import logging
import os
import threading
from typing import Any, Final, Literal
from urllib.request import urlopen

import httpx
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

_lock = threading.Lock()
_active_ollama_clients: list[Any] = []
_active_ollama_models: set[str] = set()


def _ollama_api_base() -> str:
    return OLLAMA_OPENAI_BASE_URL.rsplit("/v1", 1)[0].rstrip("/")


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
        with urlopen(f"{_ollama_api_base()}/api/tags", timeout=0.75) as response:
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


def register_ollama_client(client: Any) -> None:
    """Track a client whose ``close()`` aborts an in-flight Ollama request."""
    with _lock:
        _active_ollama_clients.append(client)


def unregister_ollama_client(client: Any) -> None:
    """Stop tracking an Ollama client once its request has completed."""
    with _lock:
        try:
            _active_ollama_clients.remove(client)
        except ValueError:
            pass


def note_ollama_model(model: str | None) -> None:
    """Remember an Ollama model that may need unloading during shutdown."""
    if model:
        with _lock:
            _active_ollama_models.add(model)


def cancellable_ollama_chat(**kwargs: Any) -> Any:
    """Run ``Client.chat`` on a dedicated client that can be closed on app shutdown."""
    from ollama import Client  # pyright: ignore[reportMissingImports]

    note_ollama_model(kwargs.get("model"))
    client = Client()
    register_ollama_client(client)
    try:
        return client.chat(**kwargs)
    finally:
        unregister_ollama_client(client)
        try:
            client.close()
        except Exception:
            pass


def _close_registered_ollama_clients() -> int:
    with _lock:
        clients = list(_active_ollama_clients)
        _active_ollama_clients.clear()
    closed = 0
    for client in clients:
        try:
            close = getattr(client, "close", None)
            if callable(close):
                close()
                closed += 1
        except Exception as exc:
            logger.debug("Failed to close tracked Ollama client: %s", exc)
    return closed


def _close_module_default_ollama_client() -> None:
    try:
        import ollama  # pyright: ignore[reportMissingImports]

        ollama._client.close()  # noqa: SLF001 — module singleton used by ollama.chat()
    except Exception as exc:
        logger.debug("Failed to close default Ollama client: %s", exc)


def _unload_ollama_models(model_names: set[str]) -> None:
    if not model_names:
        return
    base = _ollama_api_base()
    for model in model_names:
        try:
            httpx.post(
                f"{base}/api/generate",
                json={"model": model, "prompt": " ", "keep_alive": 0},
                timeout=10.0,
            )
        except Exception as exc:
            logger.debug("Failed to unload Ollama model %r: %s", model, exc)


def _unload_ollama_models_from_ps() -> None:
    """Unload every model Ollama reports as loaded (frees VRAM after abrupt shutdown)."""
    base = _ollama_api_base()
    try:
        response = httpx.get(f"{base}/api/ps", timeout=5.0)
        response.raise_for_status()
        models = [entry.get("name") for entry in response.json().get("models", [])]
    except Exception as exc:
        logger.debug("Could not list Ollama processes: %s", exc)
        return
    _unload_ollama_models({name for name in models if name})


def shutdown_ollama_generation() -> None:
    """Abort in-flight Ollama generations and unload models. Safe to call multiple times."""
    with _lock:
        models = set(_active_ollama_models)
        _active_ollama_models.clear()

    closed = _close_registered_ollama_clients()
    _close_module_default_ollama_client()
    _unload_ollama_models(models)
    _unload_ollama_models_from_ps()

    if closed:
        logger.info("Closed %d in-flight Ollama HTTP client(s)", closed)
