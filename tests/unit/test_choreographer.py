"""Tests for choreographer spline composition (WS1) and schema helpers (F3)."""

from pathlib import Path

import numpy as np
import pytest
from conftest import virtual_crazyswarm_config

from swarm_gpt.core.choreographer import Choreographer
from swarm_gpt.core.spline import PiecewiseSpline, Spline


def test_schema_allows_multiple_actions_per_entry():
    """After F3 rollback, action_list must not have maxItems: 1."""
    from swarm_gpt.core.structured_output_schema import build_motion_primitive_response_schema

    schema = build_motion_primitive_response_schema(
        all_keys=[(1, 1, 1)], required_keys=[(1, 1, 1)], num_drones=10
    )
    action_list_schema = schema["$defs"]["action_list"]
    assert "maxItems" not in action_list_schema or action_list_schema["maxItems"] > 1
    assert action_list_schema["minItems"] == 1


def _single_bar_structure(end_s: float = 20.0):  # noqa: ANN202
    """A one-segment, one-bar, four-beat structure for composition tests."""
    from swarm_gpt.utils.music_analyzer import Bar, Beat, Segment, SongStructure

    beats = [Beat(id=j + 1, time_s=j * 0.5, position_in_bar=j + 1) for j in range(4)]
    bar = Bar(id=1, start_s=0.0, beats=beats)
    seg = Segment(id=1, label="chorus", start_s=0.0, end_s=end_s, bars=[bar])
    return SongStructure(
        schema_version=2,
        source_path="test.mp3",
        song_sha256="abc",
        analyzer="test",
        bpm=120,
        segments=[seg],
    )


def _spline_choreographer(n_drones: int = 10) -> Choreographer:
    """A motion-primitive choreographer with drones on a line at z=1 m."""
    config_path = virtual_crazyswarm_config(n_drones=n_drones)
    c = Choreographer(config_file=config_path, llm_provider="openai", use_motion_primitives=True)
    for i in c.starting_pos:
        c.starting_pos[i] = np.array([(i - 5) * 0.3, 0.0, 1.0])
    return c


def test_choreo2splines_returns_one_fragment_list_per_drone():
    """form_star + rotate at one beat → every drone gets a spline-1 fragment for the block."""
    c = _spline_choreographer(10)
    structure = _single_bar_structure()
    choreography = {
        (
            1,
            1,
            1,
        ): "form_star([1, 2, 3, 4, 5, 6], 100, 60, 80, 1.0); rotate([1, 2, 3, 4, 5, 6], 45, 'z')"
    }

    per_drone = c._choreo2splines(choreography, structure)

    assert set(per_drone) == set(range(10))
    for fragments in per_drone.values():
        assert len(fragments) >= 1
        assert all(isinstance(s, (Spline, PiecewiseSpline)) for s in fragments)


def test_last_fn_wins_so_rotate_supersedes_the_form_hold():
    """A same-beat rotate replaces form_star's hold: drones end up moving, not held."""
    c = _spline_choreographer(10)
    structure = _single_bar_structure()
    held = c._choreo2splines(
        {(1, 1, 1): "form_star([1, 2, 3, 4, 5, 6], 100, 60, 80, 1.0)"}, structure
    )
    rotated = c._choreo2splines(
        {
            (
                1,
                1,
                1,
            ): "form_star([1, 2, 3, 4, 5, 6], 100, 60, 80, 1.0); rotate([1, 2, 3, 4, 5, 6], 45, 'z')"
        },
        structure,
    )
    # Pure hold has zero boundary velocity; the rotate block must move at least one drone.
    assert all(np.allclose(frags[0].end_state()[1], 0.0) for frags in held.values())
    assert any(np.any(np.abs(frags[0].end_state()[1]) > 1e-6) for frags in rotated.values())


def test_response2splines_threads_real_boundary_velocity():
    """The spline path yields real (nonzero) endpoint velocities, not legacy zeros."""
    c = _spline_choreographer(6)
    structure = _single_bar_structure()
    text = "choreography:\n  s1b1t1: spiral([1, 2, 3, 4, 5, 6], 4, 150)"
    per_drone = c.response2splines(text, structure)
    moved = any(
        np.any(np.abs(frag.end_state()[1]) > 1e-6) for frags in per_drone.values() for frag in frags
    )
    assert moved


def test_response2trajectory_is_continuous_and_returns_home():
    """End-to-end: each drone gets one C2 curve that leads in from and returns to home."""
    c = _spline_choreographer(6)
    structure = _single_bar_structure()
    text = "choreography:\n  s1b1t1: form_circle([1, 2, 3, 4, 5, 6], 120, 100, 1.0)"
    trajectories = c.response2trajectory(text, structure)

    assert set(trajectories) == set(range(6))
    for drone_id, traj in trajectories.items():
        assert isinstance(traj, PiecewiseSpline)
        home_cm = c.starting_pos[drone_id] * 100.0
        np.testing.assert_allclose(traj.start_state()[0], home_cm, atol=1e-6)
        np.testing.assert_allclose(traj.end_state()[0], home_cm, atol=1e-6)
        # C2 at every join the assembler created.
        for left, right in zip(traj.segments[:-1], traj.segments[1:]):
            for d in range(3):
                np.testing.assert_allclose(left.end_state()[d], right.start_state()[d], atol=1e-6)


def test_response2trajectory_handles_arc_primitives():
    """A spiral yields PiecewiseSpline (arc) fragments; assembly must run end-to-end on them."""
    c = _spline_choreographer(6)
    structure = _single_bar_structure()
    text = "choreography:\n  s1b1t1: spiral([1, 2, 3, 4, 5, 6], 4, 150)"
    trajectories = c.response2trajectory(text, structure)
    assert set(trajectories) == set(range(6))
    for drone_id, traj in trajectories.items():
        assert isinstance(traj, PiecewiseSpline)
        home_cm = c.starting_pos[drone_id] * 100.0
        np.testing.assert_allclose(traj.start_state()[0], home_cm, atol=1e-6)
        np.testing.assert_allclose(traj.end_state()[0], home_cm, atol=1e-6)


def _multi_bar_structure(n_beats: int = 8, beat_dt: float = 2.0, end_s: float = 32.0):  # noqa: ANN202
    """A one-segment, one-bar structure with ``n_beats`` evenly spaced beats."""
    from swarm_gpt.utils.music_analyzer import Bar, Beat, Segment, SongStructure

    beats = [Beat(id=j + 1, time_s=j * beat_dt, position_in_bar=j + 1) for j in range(n_beats)]
    bar = Bar(id=1, start_s=0.0, beats=beats)
    seg = Segment(id=1, label="x", start_s=0.0, end_s=end_s, bars=[bar])
    return SongStructure(
        schema_version=2, source_path="a", song_sha256="b", analyzer="t", bpm=120, segments=[seg]
    )


def test_response2trajectory_multi_block_formations_and_motion():
    """A realistic mixed sequence (formation, spiral, formation, rotate) composes end-to-end."""
    c = _spline_choreographer(6)
    structure = _multi_bar_structure(n_beats=8)
    # Real primitives and TRANSITION markers strictly alternate, starting and ending with one.
    text = (
        "choreography:\n"
        "  s1b1t1: form_circle([1, 2, 3, 4, 5, 6], 120, 100, 1.0)\n"
        "  s1b1t2: TRANSITION\n"
        "  s1b1t3: spiral([1, 2, 3, 4, 5, 6], 4, 150)\n"
        "  s1b1t4: TRANSITION\n"
        "  s1b1t5: form_star([1, 2, 3, 4, 5, 6], 100, 60, 80, 1.0)\n"
        "  s1b1t6: TRANSITION\n"
        "  s1b1t7: rotate([1, 2, 3, 4, 5, 6], 45, 'z')"
    )
    trajectories = c.response2trajectory(text, structure)
    assert set(trajectories) == set(range(6))
    for drone_id, traj in trajectories.items():
        assert isinstance(traj, PiecewiseSpline)
        home_cm = c.starting_pos[drone_id] * 100.0
        np.testing.assert_allclose(traj.start_state()[0], home_cm, atol=1e-6)
        np.testing.assert_allclose(traj.end_state()[0], home_cm, atol=1e-6)
        samples = traj.evaluate(np.linspace(traj.t0, traj.t1, 300))
        assert np.all(np.isfinite(samples))


def test_explicit_transition_between_primitives_is_continuous():
    """An explicit TRANSITION between two primitives yields a continuous C2 trajectory."""
    c = _spline_choreographer(6)
    structure = _multi_bar_structure(n_beats=4)
    text = (
        "choreography:\n"
        "  s1b1t1: form_circle([1, 2, 3, 4, 5, 6], 120, 100, 1.0)\n"
        "  s1b1t2: TRANSITION\n"
        "  s1b1t3: form_star([1, 2, 3, 4, 5, 6], 100, 60, 80, 1.0)"
    )
    trajectories = c.response2trajectory(text, structure)
    assert set(trajectories) == set(range(6))
    for drone_id, traj in trajectories.items():
        assert isinstance(traj, PiecewiseSpline)
        home_cm = c.starting_pos[drone_id] * 100.0
        np.testing.assert_allclose(traj.start_state()[0], home_cm, atol=1e-6)
        np.testing.assert_allclose(traj.end_state()[0], home_cm, atol=1e-6)
        for left, right in zip(traj.segments[:-1], traj.segments[1:]):
            for d in range(3):
                np.testing.assert_allclose(left.end_state()[d], right.start_state()[d], atol=1e-6)


def test_two_primitives_without_transition_raise_validation_error():
    """Two back-to-back primitives with no TRANSITION between them are rejected."""
    import pytest

    from swarm_gpt.exception import LLMResponseProcessingError

    c = _spline_choreographer(6)
    structure = _multi_bar_structure(n_beats=4)
    text = (
        "choreography:\n"
        "  s1b1t1: form_circle([1, 2, 3, 4, 5, 6], 120, 100, 1.0)\n"
        "  s1b1t2: form_star([1, 2, 3, 4, 5, 6], 100, 60, 80, 1.0)"
    )
    with pytest.raises(LLMResponseProcessingError, match="without a TRANSITION"):
        c.response2trajectory(text, structure)


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


def test_response2waypoints_produces_a_grid_axswarm_can_consume():
    """The WS4 deliverable end to end: nothing else in the suite calls `response2waypoints`."""
    import yaml

    c = _spline_choreographer(6)
    structure = _single_bar_structure()
    text = (
        "choreography:\n"
        "  s1b1t1: form_circle([1, 2, 3, 4, 5, 6], 120, 100, 1.0)\n"
        "  s1b1t3: TRANSITION\n"
        "  s1b1t4: helix([1, 2, 3], 2, 40, 90)\n"
    )
    wp = c.response2waypoints(text, structure, strict=False)
    settings = yaml.safe_load(
        (Path(__file__).resolve().parents[2] / "swarm_gpt/data/settings.yaml").read_text()
    )
    freq, horizon = settings["axswarm"]["freq"], settings["axswarm"]["K"]
    t = wp["time"][0]

    assert wp["pos"].shape == (6, t.size, 3)
    assert t[0] == pytest.approx(0.0), "the sim clock starts at 0 and SolverData reads column 0"
    gaps = np.diff(t)
    assert gaps.min() >= 1.0 / freq, "two waypoints would collapse onto one MPC index"
    assert gaps.max() < horizon / freq, "a gap this wide empties the lookahead, silently"
    assert len({round(float(x) * freq) for x in t}) == t.size, "MPC index collision"
