from pathlib import Path

import numpy as np
import pytest
from conftest import virtual_crazyswarm_config

import swarm_gpt.core.drone_swarm as drone_swarm
from swarm_gpt.core.backend import AppBackend


def test_backend_init():
    config_path = virtual_crazyswarm_config(n_drones=4)
    app = AppBackend(config_file=config_path)
    assert app.choreographer.num_drones == 4
    assert app.choreographer.messages == []


def test_songs():
    config_path = virtual_crazyswarm_config(n_drones=4)
    app = AppBackend(config_file=config_path)
    assert isinstance(app.songs, list)
    available_songs = [s.stem for s in app.music_manager.music_dir.glob("*.mp3")]
    for song in app.songs:
        assert isinstance(song, str), f"Song {song} is not a string"
        assert song in available_songs, f"Song {song} is not in the available songs"


def test_presets():
    config_path = virtual_crazyswarm_config(n_drones=4)
    app = AppBackend(config_file=config_path)
    assert isinstance(app.presets, list)
    for preset in app.presets:
        assert isinstance(preset, str), f"Preset {preset} is not a string"


def test_preset_metadata_and_delete(tmp_path: Path):
    config_path = virtual_crazyswarm_config(n_drones=4)
    preset_dir = tmp_path / "presets"
    preset_id = "Example Song | 4 | 20260521_123456"
    (preset_dir / preset_id).mkdir(parents=True)

    app = AppBackend(config_file=config_path, preset_dir=preset_dir)
    metadata = app.preset_metadata(preset_id)

    assert metadata["song"] == "Example Song"
    assert metadata["numDrones"] == 4
    assert metadata["createdAt"] == "2026-05-21T12:34:56"
    assert metadata["createdLabel"] == "2026-05-21 12:34"

    app.delete_preset(preset_id)
    assert not (preset_dir / preset_id).exists()


def test_deploy_emergency_stops_before_close_on_keyboard_interrupt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class InterruptingSwarm:
        instances = []

        def __init__(self, drones: dict, *, lighthouse: bool) -> None:
            self.drones_by_uri = {drone["uri"]: drone for drone in drones.values()}
            self.calls: list[str] = []
            self.instances.append(self)

        def get_obs(self, uri: str) -> dict[str, np.ndarray]:
            pos = np.asarray(self.drones_by_uri[uri]["pos"], dtype=float)
            if "goto" in self.calls:
                pos = pos + np.array([0.0, 0.0, 0.5])
            return {"pos": pos, "quat": np.array([0.0, 0.0, 0.0, 1.0])}

        def goto(self, target: dict[str, np.ndarray], duration: float = 3.0) -> None:
            self.calls.append("goto")

        def is_active(self, uri: str) -> bool:
            return uri in self.drones_by_uri

        def execute_choreography(self, *args: object, **kwargs: object) -> None:
            self.calls.append("execute_choreography")
            raise KeyboardInterrupt

        def emergency_stop(self) -> None:
            self.calls.append("emergency_stop")

        def land(self, height: float = 0.0, duration: float = 3.0) -> None:
            self.calls.append("land")

        def close(self) -> None:
            self.calls.append("close")

    config_path = virtual_crazyswarm_config(n_drones=1)
    app = AppBackend(config_file=config_path)
    app.settings["lighthouse"] = True
    app.settings["land_on_docks"] = False
    app.music_manager.song = "Crazyflie Drones Theme"
    app.music_manager.verify_libvlc = lambda: True
    app.music_manager.play = lambda *, wait, start_s, end_s: True
    app.music_manager.stop = lambda: None
    app.waypoints = {"time": np.array([[0.0, 1.0]])}
    app.splines[0] = lambda t: np.array([0.0, 0.0, 1.0])
    monkeypatch.setattr(drone_swarm, "DroneSwarm", InterruptingSwarm)

    with pytest.raises(KeyboardInterrupt):
        app.deploy()

    swarm = InterruptingSwarm.instances[0]
    assert swarm.calls == ["goto", "execute_choreography", "emergency_stop", "close"]


def test_emergency_stop_active_swarm_stops_live_swarm_and_music() -> None:
    class ActiveSwarm:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def emergency_stop(self) -> None:
            self.calls.append("emergency_stop")

    config_path = virtual_crazyswarm_config(n_drones=1)
    app = AppBackend(config_file=config_path)
    swarm = ActiveSwarm()
    music_calls: list[str] = []
    app._active_swarm = swarm
    app.music_manager.stop = lambda: music_calls.append("stop")

    app.emergency_stop_active_swarm()

    assert swarm.calls == ["emergency_stop"]
    assert music_calls == ["stop"]
