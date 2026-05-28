"""Tests for Ollama shutdown cancellation in llm_providers."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

from swarm_gpt.utils import llm_providers

if TYPE_CHECKING:
    from pytest import MonkeyPatch


def test_shutdown_closes_registered_clients(monkeypatch: MonkeyPatch):
    llm_providers._active_ollama_clients.clear()
    llm_providers._active_ollama_models.clear()

    client = MagicMock()
    llm_providers.register_ollama_client(client)

    monkeypatch.setattr(llm_providers, "_close_module_default_ollama_client", lambda: None)
    monkeypatch.setattr(llm_providers, "_unload_ollama_models", lambda _models: None)
    monkeypatch.setattr(llm_providers, "_unload_ollama_models_from_ps", lambda: None)

    llm_providers.shutdown_ollama_generation()

    client.close.assert_called_once()
    assert llm_providers._active_ollama_clients == []
