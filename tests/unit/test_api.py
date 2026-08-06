import threading
import time
from collections.abc import Generator
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import quote

import numpy as np
import pytest
from fastapi.testclient import TestClient

import swarm_gpt.api.server as server
from swarm_gpt.api.server import ApiConfig, _backend_from_config, create_app, normalize_playback
from swarm_gpt.utils.llm_providers import DEFAULT_OPENAI_MODEL_CHOICES


def _lighting_stub(num_drones: int) -> dict[str, list[dict[str, list]]]:
    """A stand-in for `AppBackend.browser_cues`; the real adapter is covered in test_backend.py."""
    return {
        deck: [{"times": [0.0, 1.0], "rgb": [[255, 0, 0], [0, 0, 0]]} for _ in range(num_drones)]
        for deck in ("top", "bot")
    }


def test_normalize_playback_schema():
    backend = SimpleNamespace(
        settings={"axswarm": {"pos_min": [-1, -1, 0], "pos_max": [1, 1, 2]}},
        music_manager=SimpleNamespace(song="Example Song"),
        crop_window=lambda song: (0.0, 60.0),
        browser_cues=lambda: _lighting_stub(3),
    )
    states = np.zeros((2, 3, 13))
    states[:, :, 3:7] = [0, 0, 0, 1]
    payload = normalize_playback(
        {"timestamps": np.array([0.0, 0.02]), "states": states, "num_drones": 3}, backend
    )
    assert payload["schemaVersion"] == 2
    assert payload["audioUrl"] == "/api/media/music/Example%20Song"
    assert payload["audioOffset"] == 0.0
    assert payload["numDrones"] == 3
    assert payload["fields"]["pos"] == [0, 3]
    assert len(payload["states"]) == len(payload["timestamps"])
    # §9.3: the timeline is the single colour source, so the static `colors` array is gone and the
    # payload carries one cue list per deck per drone instead.
    assert "colors" not in payload
    assert payload["lighting"] == _lighting_stub(3)


def test_normalize_playback_rejects_mismatched_states():
    backend = SimpleNamespace(
        settings={"axswarm": {"pos_min": [-1, -1, 0], "pos_max": [1, 1, 2]}},
        music_manager=SimpleNamespace(song="Example Song"),
    )
    with pytest.raises(ValueError, match="State/timestamp mismatch"):
        normalize_playback(
            {"timestamps": np.array([0.0, 0.02]), "states": np.zeros((1, 3, 13)), "num_drones": 3},
            backend,
        )


def test_app_and_library_metadata_build(tmp_path: Path):
    (tmp_path / "Test Song.mp3").write_bytes(b"")
    app = create_app(ApiConfig(music_dir=tmp_path))
    backend = _backend_from_config(ApiConfig(music_dir=tmp_path), "openai", "gpt-5.4-nano")

    assert app.title == "SwarmGPT Browser API"
    assert backend.songs == ["Test Song"]
    assert DEFAULT_OPENAI_MODEL_CHOICES[0] == "gpt-5.6-luna"


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


def test_emergency_stop_endpoint_runs_while_deploy_is_active(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class DeployingBackend:
        def __init__(self) -> None:
            self.songs = ["Test Song"]
            self.presets: list[str] = []
            self.settings = {"axswarm": {"pos_min": [-1, -1, 0], "pos_max": [1, 1, 2]}}
            self.music_manager = SimpleNamespace(song="Test Song")
            self.splines: dict[int, object] = {}
            self.deploy_entered = threading.Event()
            self.stop_requested = threading.Event()
            self.emergency_stop_calls = 0

        def initial_prompt(self, selection: str) -> list[dict[str, str]]:
            return []

        def simulate(self) -> Generator[None, None, dict[str, object]]:
            self.splines[0] = object()
            states = np.zeros((1, 1, 13))
            states[:, :, 3:7] = [0, 0, 0, 1]
            if False:
                yield None
            return {"timestamps": np.array([0.0]), "states": states, "num_drones": 1}

        def crop_window(self, song: str) -> tuple[float, float]:
            return (0.0, 60.0)

        def deploy(self) -> bool:
            self.deploy_entered.set()
            self.stop_requested.wait(timeout=2.0)
            return True

        def emergency_stop_active_swarm(self) -> None:
            self.emergency_stop_calls += 1
            self.stop_requested.set()

    backends: list[DeployingBackend] = []

    def backend_from_config(config: ApiConfig, provider: str, model_id: str) -> DeployingBackend:
        backend = DeployingBackend()
        backends.append(backend)
        return backend

    (tmp_path / "Test Song.mp3").write_bytes(b"")
    monkeypatch.setattr(server, "_backend_from_config", backend_from_config)
    client = TestClient(create_app(ApiConfig(music_dir=tmp_path)))

    create_response = client.post(
        "/api/jobs", json={"selection": "Test Song", "provider": "openai", "modelId": "gpt"}
    )
    create_response.raise_for_status()
    job_id = create_response.json()["jobId"]
    backend = backends[0]
    for _ in range(50):
        if client.get(f"/api/jobs/{job_id}").json()["status"] == "ready":
            break
        time.sleep(0.01)

    deploy_response = client.post(f"/api/jobs/{job_id}/deploy")
    deploy_response.raise_for_status()
    assert backend.deploy_entered.wait(timeout=1.0)

    stop_response = client.post(f"/api/jobs/{job_id}/emergency-stop")
    stop_response.raise_for_status()

    assert stop_response.json() == {"jobId": job_id, "emergencyStopped": True}
    assert backend.emergency_stop_calls == 1
