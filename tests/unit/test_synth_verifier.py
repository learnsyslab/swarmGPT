import numpy as np
import pytest

from swarm_gpt.synth.sandbox import SynthError, compile_invariants, compile_primitive
from swarm_gpt.synth.verifier import authored_trajectory, check_invariants, measure

SETTINGS = {"axswarm": {"collision_envelope": [0.25, 0.25, 0.6], "freq": 10, "vel_max": 1.73}}

RISE_SOURCE = """
def rise(params, swarm_pos, tstart, tend, limits):
    delta, = params
    pos = swarm_pos.copy()
    waypoints = {}
    for t in np.linspace(tstart, tend, 3)[1:]:
        pos = pos + np.array([0.0, 0.0, delta / 2])
        waypoints[float(t)] = {i: p.copy() for i, p in enumerate(pos)}
    return pos, waypoints
"""

LIMITS = {"lower": np.array([-2.0, -2.0, 0.0]), "upper": np.array([2.0, 2.0, 2.0])}


def two_drone_pass(gap_m: float, n_steps: int = 21) -> dict:
    """Two drones on the x axis, closing to ``gap_m`` at the midpoint and separating again."""
    t = np.linspace(0.0, 2.0, n_steps)
    half = gap_m / 2 + np.abs(t - 1.0)
    pos = np.zeros((n_steps, 2, 3))
    pos[:, 0, 0] = -half
    pos[:, 1, 0] = half
    pos[:, :, 2] = 1.0
    return {
        "time": t,
        "pos": pos,
        "vel": np.zeros_like(pos),
        "success": np.ones(n_steps, dtype=bool),
    }


def test_authored_trajectory_holds_untouched_drones_and_appends_a_tail():
    fn = compile_primitive(RISE_SOURCE, "rise")
    start = np.zeros((3, 3))
    start[:, 2] = 1.0
    authored = authored_trajectory(fn, (50,), start, 0.0, 4.0, LIMITS)
    assert authored["pos"].shape == (3, 4, 3)  # t=0, two emitted, one tail hold
    assert authored["time"][0, -1] > 4.0
    assert np.allclose(authored["pos"][:, -1], authored["pos"][:, -2])


def test_authored_trajectory_clips_to_the_arena():
    fn = compile_primitive(RISE_SOURCE, "rise")
    authored = authored_trajectory(fn, (500,), np.zeros((2, 3)), 0.0, 4.0, LIMITS)
    assert authored["pos"][:, :, 2].max() <= LIMITS["upper"][2]


def test_measure_finds_the_hand_computed_closest_approach():
    repaired = two_drone_pass(gap_m=0.2)
    authored = {"time": repaired["time"][None, :], "pos": repaired["pos"].transpose(1, 0, 2)}
    m = measure(authored, repaired, SETTINGS, (0.0, 2.0))
    assert m["min_sep_m"] == pytest.approx(0.2, abs=1e-6)
    assert m["worst_time_s"] == pytest.approx(1.0, abs=1e-6)
    assert sorted(m["worst_pair"]) == [1, 2]
    # Approach is pure x, so the required separation is the x semi-axis of the envelope.
    assert m["required_sep_m"] == pytest.approx(0.25, abs=1e-6)
    assert m["min_sep_norm"] == pytest.approx(0.8, abs=1e-6)
    assert m["steps_inside_envelope"] > 0


def test_measure_reports_zero_deviation_when_the_filter_changed_nothing():
    repaired = two_drone_pass(gap_m=1.0)
    authored = {"time": repaired["time"][None, :], "pos": repaired["pos"].transpose(1, 0, 2)}
    m = measure(authored, repaired, SETTINGS, (0.0, 2.0))
    assert m["deviation_mean_m"] == pytest.approx(0.0, abs=1e-9)
    assert m["deviation_max_m"] == pytest.approx(0.0, abs=1e-9)
    assert m["steps_inside_envelope"] == 0


def test_measure_excludes_the_settle_tail():
    repaired = two_drone_pass(gap_m=0.2, n_steps=21)
    authored = {"time": repaired["time"][None, :], "pos": repaired["pos"].transpose(1, 0, 2)}
    full = measure(authored, repaired, SETTINGS, (0.0, 2.0))
    early = measure(authored, repaired, SETTINGS, (0.0, 0.5))
    assert early["n_steps"] < full["n_steps"]
    # The closest approach happens at t=1.0, outside the early window.
    assert early["min_sep_m"] > full["min_sep_m"]


def test_check_invariants_normalizes_triples_and_uses_centimetres():
    check = compile_invariants(
        "def check(pos, time, params):\n"
        "    return [('height', bool(pos[:, :, 2].max() > 50), 'z in cm')]\n"
    )
    repaired = two_drone_pass(gap_m=1.0)  # z = 1.0 m, so 100 cm
    result = check_invariants(check, repaired, (), (0.0, 2.0))
    assert result == [{"name": "height", "ok": True, "detail": "z in cm"}]


def test_check_invariants_rejects_a_non_triple_return():
    check = compile_invariants("def check(pos, time, params):\n    return ['nope']\n")
    with pytest.raises(SynthError, match="triple"):
        check_invariants(check, two_drone_pass(gap_m=1.0), (), (0.0, 2.0))


def test_check_invariants_rejects_an_empty_return():
    check = compile_invariants("def check(pos, time, params):\n    return []\n")
    with pytest.raises(SynthError, match="non-empty list"):
        check_invariants(check, two_drone_pass(gap_m=1.0), (), (0.0, 2.0))
