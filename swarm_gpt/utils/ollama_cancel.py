"""Cancel in-flight Ollama HTTP requests when the SwarmGPT app shuts down."""

from __future__ import annotations

import logging
import threading
from typing import Any

import httpx

from swarm_gpt.utils.llm_providers import OLLAMA_OPENAI_BASE_URL

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_active_clients: list[Any] = []
_active_models: set[str] = set()


def _ollama_api_base() -> str:
    return OLLAMA_OPENAI_BASE_URL.rsplit("/v1", 1)[0].rstrip("/")


def register_ollama_client(client: Any) -> None:
    """Track a client whose ``close()`` aborts an in-flight Ollama request."""
    with _lock:
        _active_clients.append(client)


def unregister_ollama_client(client: Any) -> None:
    with _lock:
        try:
            _active_clients.remove(client)
        except ValueError:
            pass


def note_ollama_model(model: str | None) -> None:
    if model:
        with _lock:
            _active_models.add(model)


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


def _close_registered_clients() -> int:
    with _lock:
        clients = list(_active_clients)
        _active_clients.clear()
    closed = 0
    for client in clients:
        try:
            close = getattr(client, "close", None)
            if callable(close):
                close()
                closed += 1
        except Exception as exc:
            logger.debug("Failed to close tracked client: %s", exc)
    return closed


def _close_module_default_client() -> None:
    try:
        import ollama  # pyright: ignore[reportMissingImports]

        ollama._client.close()  # noqa: SLF001 — module singleton used by ollama.chat()
    except Exception as exc:
        logger.debug("Failed to close default Ollama client: %s", exc)


def _unload_models(model_names: set[str]) -> None:
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


def _unload_models_from_ps() -> None:
    """Unload every model Ollama reports as loaded (frees VRAM after abrupt shutdown)."""
    base = _ollama_api_base()
    try:
        response = httpx.get(f"{base}/api/ps", timeout=5.0)
        response.raise_for_status()
        models = [entry.get("name") for entry in response.json().get("models", [])]
    except Exception as exc:
        logger.debug("Could not list Ollama processes: %s", exc)
        return
    _unload_models({name for name in models if name})


def shutdown_ollama_generation() -> None:
    """Abort in-flight generations and unload models. Safe to call multiple times."""
    with _lock:
        models = set(_active_models)
        _active_models.clear()

    closed = _close_registered_clients()
    _close_module_default_client()
    _unload_models(models)
    _unload_models_from_ps()

    if closed:
        logger.info("Closed %d in-flight Ollama HTTP client(s)", closed)
