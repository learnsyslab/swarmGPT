"""Tests for Ollama shutdown cancellation helpers."""

from __future__ import annotations

from unittest.mock import MagicMock

from swarm_gpt.utils import ollama_cancel


def test_shutdown_closes_registered_clients(monkeypatch):
    ollama_cancel._active_clients.clear()
    ollama_cancel._active_models.clear()

    client = MagicMock()
    ollama_cancel.register_ollama_client(client)

    monkeypatch.setattr(ollama_cancel, "_close_module_default_client", lambda: None)
    monkeypatch.setattr(ollama_cancel, "_unload_models", lambda _models: None)
    monkeypatch.setattr(ollama_cancel, "_unload_models_from_ps", lambda: None)

    ollama_cancel.shutdown_ollama_generation()

    client.close.assert_called_once()
    assert ollama_cancel._active_clients == []
