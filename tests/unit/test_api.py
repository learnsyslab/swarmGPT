from pathlib import Path
from types import SimpleNamespace
from urllib.parse import quote

import numpy as np
import pytest
from fastapi.testclient import TestClient

from swarm_gpt.api.server import ApiConfig, _backend_from_config, create_app, normalize_playback
from swarm_gpt.utils.llm_providers import DEFAULT_OPENAI_MODEL_CHOICES


def test_normalize_playback_schema():
    backend = SimpleNamespace(
        settings={"axswarm": {"pos_min": [-1, -1, 0], "pos_max": [1, 1, 2]}},
        music_manager=SimpleNamespace(song="Example Song"),
    )
    states = np.zeros((2, 3, 13))
    states[:, :, 3:7] = [0, 0, 0, 1]
    payload = normalize_playback(
        {"timestamps": np.array([0.0, 0.02]), "states": states, "num_drones": 3},
        backend,
    )
    assert payload["schemaVersion"] == 1
    assert payload["audioUrl"] == "/api/media/music/Example%20Song"
    assert payload["numDrones"] == 3
    assert payload["fields"]["pos"] == [0, 3]
    assert len(payload["states"]) == len(payload["timestamps"])


def test_normalize_playback_rejects_mismatched_states():
    backend = SimpleNamespace(
        settings={"axswarm": {"pos_min": [-1, -1, 0], "pos_max": [1, 1, 2]}},
        music_manager=SimpleNamespace(song="Example Song"),
    )
    with pytest.raises(ValueError, match="State/timestamp mismatch"):
        normalize_playback(
            {
                "timestamps": np.array([0.0, 0.02]),
                "states": np.zeros((1, 3, 13)),
                "num_drones": 3,
            },
            backend,
        )


def test_app_and_library_metadata_build(tmp_path: Path):
    (tmp_path / "Test Song.mp3").write_bytes(b"")
    app = create_app(ApiConfig(music_dir=tmp_path))
    backend = _backend_from_config(ApiConfig(music_dir=tmp_path), "openai", "gpt-4o")

    assert app.title == "SwarmGPT Browser API"
    assert backend.songs == ["Test Song"]
    assert DEFAULT_OPENAI_MODEL_CHOICES[0] == "gpt-4o"


def test_library_returns_preset_display_metadata_and_delete(tmp_path: Path):
    (tmp_path / "Test Song.mp3").write_bytes(b"")
    preset_dir = tmp_path / "presets"
    preset_id = "Test Song | 6 | 20260521_123456"
    (preset_dir / preset_id).mkdir(parents=True)

    client = TestClient(create_app(ApiConfig(music_dir=tmp_path, preset_dir=preset_dir)))
    response = client.get("/api/library")
    response.raise_for_status()
    data = response.json()

    assert data["presets"] == [
        {
            "id": preset_id,
            "label": "Test Song",
            "kind": "preset",
            "previewUrl": "/api/media/music/Test%20Song",
            "song": "Test Song",
            "numDrones": 6,
            "createdAt": "2026-05-21T12:34:56",
            "createdLabel": "2026-05-21 12:34",
        }
    ]

    delete_response = client.delete(f"/api/presets/{quote(preset_id, safe='')}")
    delete_response.raise_for_status()
    assert delete_response.json() == {"deleted": preset_id}
    assert not (preset_dir / preset_id).exists()
