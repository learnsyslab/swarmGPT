from pathlib import Path

from conftest import virtual_crazyswarm_config

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
