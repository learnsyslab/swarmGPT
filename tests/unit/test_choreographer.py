"""Tests for choreographer orchestration helpers (F3) and form_* motion primitives (F1)."""

from pathlib import Path

import numpy as np
from conftest import virtual_crazyswarm_config

from swarm_gpt.core.choreographer import (
    Choreographer,
    _form_should_drop_holds,
    _overlapping_drone_set,
)


def test_form_should_drop_holds_when_overlapping_motion_follows():
    """form_star (full swarm) + rotate (full swarm) → overlap, drop holds."""
    action_list = [{"form_star": (100, 60, 80, 1.0)}, {"rotate": (45, "z")}]
    assert _form_should_drop_holds(action_list, 0, num_drones=10) is True


def test_form_should_not_drop_holds_when_motion_targets_disjoint_drones():
    """form_circle drones 1-5 + move_z drones 6-10 → no overlap, keep holds."""
    action_list = [{"form_circle": ([1, 2, 3, 4, 5], 150, 1.0)}, {"move_z": ([6, 7, 8, 9, 10], 50)}]
    assert _form_should_drop_holds(action_list, 0, num_drones=10) is False


def test_form_should_not_drop_holds_when_followed_by_another_form():
    """form_star followed only by form_circle → no motion primitive, keep holds."""
    action_list = [{"form_star": (100, 60, 80, 1.0)}, {"form_circle": ([1, 2, 3, 4, 5], 150, 1.0)}]
    assert _form_should_drop_holds(action_list, 0, num_drones=10) is False


def test_form_should_drop_holds_with_three_deep_stack():
    """form_star; rotate; move_z — both later entries are motion, still drops holds."""
    action_list = [
        {"form_star": (100, 60, 80, 1.0)},
        {"rotate": (45, "z")},
        {"move_z": ([1, 2, 3], 30)},
    ]
    assert _form_should_drop_holds(action_list, 0, num_drones=10) is True


def test_overlapping_drone_set_full_swarm():
    """Primitives without drone subset args touch the full swarm."""
    assert _overlapping_drone_set({"form_star": (100, 60, 80, 1.0)}, num_drones=5) == frozenset(
        {0, 1, 2, 3, 4}
    )
    assert _overlapping_drone_set({"rotate": (45, "z")}, num_drones=3) == frozenset({0, 1, 2})


def test_overlapping_drone_set_subset():
    """form_circle / move_z / center return 0-indexed drone IDs from their first arg."""
    assert _overlapping_drone_set(
        {"form_circle": ([1, 3, 5], 150, 1.0)}, num_drones=10
    ) == frozenset({0, 2, 4})
    assert _overlapping_drone_set({"move_z": ([2, 4], 50)}, num_drones=10) == frozenset({1, 3})
    assert _overlapping_drone_set({"center": ([1, 2, 3],)}, num_drones=10) == frozenset({0, 1, 2})


def test_overlapping_drone_set_swap():
    """swap returns exactly the two drone IDs (0-indexed)."""
    assert _overlapping_drone_set({"swap": (3, 7)}, num_drones=10) == frozenset({2, 6})


def test_overlapping_drone_set_move():
    """move returns the single target drone ID (0-indexed, 4th arg)."""
    assert _overlapping_drone_set({"move": (100, 0, 150, 5)}, num_drones=10) == frozenset({4})


def test_schema_allows_multiple_actions_per_entry():
    """After F3 rollback, action_list must not have maxItems: 1."""
    from swarm_gpt.core.structured_output_schema import build_motion_primitive_response_schema

    schema = build_motion_primitive_response_schema(
        all_keys=[(1, 1, 1)], required_keys=[(1, 1, 1)], num_drones=10
    )
    action_list_schema = schema["$defs"]["action_list"]
    assert "maxItems" not in action_list_schema or action_list_schema["maxItems"] > 1
    assert action_list_schema["minItems"] == 1


def test_form_star_hold_pruning_in_pipeline():
    """form_star + rotate: arrival waypoint kept, holds stripped before merge."""
    config_path = virtual_crazyswarm_config(n_drones=10)
    choreographer = Choreographer(
        config_file=config_path, llm_provider="openai", use_motion_primitives=True
    )
    # 10 drones arranged in a line at z=100 cm
    for i in choreographer.starting_pos:
        choreographer.starting_pos[i] = np.array([(i - 5) * 0.3, 0.0, 1.0])

    from swarm_gpt.utils.music_analyzer import Bar, Beat, Segment, SongStructure

    t = 0.0
    beats = [Beat(id=j + 1, time_s=t + j * 0.5, position_in_bar=j + 1) for j in range(4)]
    bar = Bar(id=1, start_s=0.0, beats=beats)
    seg = Segment(id=1, label="chorus", start_s=0.0, end_s=20.0, bars=[bar])
    structure = SongStructure(
        schema_version=2,
        source_path="test.mp3",
        song_sha256="abc",
        analyzer="test",
        bpm=120,
        segments=[seg],
    )

    # Build a choreography with form_star followed by rotate at the same key
    choreography = {(1, 1, 1): "form_star(100, 60, 80, 1.0); rotate(45, 'z')"}
    waypoints = choreographer._choreo2waypoints(choreography, structure)

    # The position array must have shape (n_drones, T, 3)
    assert waypoints["pos"].shape[0] == 10
    # Must have more than one timestep (rotate emits dense waypoints)
    assert waypoints["pos"].shape[1] > 2


def test_load_drone_config_uses_active_list(tmp_path: Path) -> None:
    """Loader must respect active list order and build uri from addr and channel."""
    cfg = tmp_path / "drones.toml"
    cfg.write_text(
        'active = ["cf41", "cf31"]\n'
        "[cf31]\naddr = 0x1F\nchannel = 30\npos = [0.0, 0.0, 0.0]\n"
        "[cf41]\naddr = 0x29\nchannel = 40\npos = [1.0, 0.0, 0.0]\n"
    )
    import yaml

    c = Choreographer.__new__(Choreographer)
    c.agents = {}
    c.uris = {}
    c.starting_pos = {}
    c.num_drones = 0
    settings_path = Path(__file__).resolve().parents[2] / "swarm_gpt/data/settings.yaml"
    c.settings = yaml.safe_load(settings_path.read_text())
    c.load_drone_config(config_file=cfg)

    # active = ["cf41", "cf31"] → swarm index 0 = cf41, 1 = cf31
    assert c.num_drones == 2
    assert c.uris[0] == "radio://0/40/2M/E7E7E7E729"  # cf41, channel=40, addr=0x29
    assert c.uris[1] == "radio://0/30/2M/E7E7E7E71F"  # cf31, channel=30, addr=0x1F
