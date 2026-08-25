import numpy as np
import pytest
import yaml

from swarm_gpt.synth import loop as loopmod
from swarm_gpt.synth.loop import SynthesisLoop
from swarm_gpt.synth.manifest import ParamSpec, PrimitiveManifest
from swarm_gpt.synth.sandbox import SynthError, compile_invariants, compile_primitive
from swarm_gpt.synth.verifier import authored_trajectory, check_invariants, measure, screen_authored

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


# Every drone flies to the same point, so the authored geometry is infeasible before any solve.
COLLIDING = """
def pile(params, swarm_pos, tstart, tend, limits):
    reach, = params
    pos = swarm_pos.copy()
    waypoints = {}
    for t in np.linspace(tstart, tend, 4)[1:]:
        pos = np.zeros_like(swarm_pos) + np.array([0.0, 0.0, reach])
        waypoints[float(t)] = {i: p.copy() for i, p in enumerate(pos)}
    return pos, waypoints
"""

# Drones hold their (well-separated) dock positions and only drift upward together.
SEPARATED = """
def lift(params, swarm_pos, tstart, tend, limits):
    delta, = params
    pos = swarm_pos.copy()
    waypoints = {}
    for t in np.linspace(tstart, tend, 4)[1:]:
        pos = pos + np.array([0.0, 0.0, delta / 3.0])
        waypoints[float(t)] = {i: p.copy() for i, p in enumerate(pos)}
    return pos, waypoints
"""

# Well separated (drones are spread in x and swing together in y), but crosses the arena
# between adjacent waypoints, which no drone can fly in the time allowed.
TELEPORTS = """
def dash(params, swarm_pos, tstart, tend, limits):
    reach, = params
    pos = swarm_pos.copy()
    waypoints = {}
    for k, t in enumerate(np.linspace(tstart, tend, 4)[1:]):
        pos = swarm_pos + np.array([0.0, reach if k % 2 == 0 else -reach, 0.0])
        waypoints[float(t)] = {i: p.copy() for i, p in enumerate(pos)}
    return pos, waypoints
"""

CHECK = """
def check(pos, time, params):
    return [("moved", bool(np.any(pos[:, -1, 2] != pos[:, 0, 2])), "z changed")]
"""


class SolverReached(Exception):
    pass


def _settings() -> dict:
    return yaml.safe_load(open("swarm_gpt/data/settings.yaml").read())


def _manifest(name: str, source: str, lo: float, hi: float) -> PrimitiveManifest:
    return PrimitiveManifest(
        name=name,
        intent="test fixture",
        params=(ParamSpec(name="a", type="float", minimum=lo, maximum=hi),),
        source=source,
        invariants=CHECK,
    )


def _spread_positions(n: int = 6) -> np.ndarray:
    pos = np.zeros((n, 3))
    pos[:, 0] = np.linspace(-1.5, 1.5, n)
    pos[:, 2] = 1.0
    return pos


def _loop(monkeypatch: pytest.MonkeyPatch, *, screen: bool) -> SynthesisLoop:
    monkeypatch.setattr(loopmod, "openai_client_for_provider", lambda *a, **k: object())
    return SynthesisLoop(
        settings=_settings(),
        start_pos_m=_spread_positions(),
        arm="absolute",
        model_id="test",
        duration_s=6.0,
        screen=screen,
    )


def _authored(manifest: PrimitiveManifest, value: float, settings: dict) -> dict:
    fn, _check = manifest.compile()
    limits = {
        "lower": np.asarray(settings["axswarm"]["pos_min"], dtype=float),
        "upper": np.asarray(settings["axswarm"]["pos_max"], dtype=float),
    }
    return authored_trajectory(fn, manifest.bind([value]), _spread_positions(), 0.0, 6.0, limits)


def test_screen_flags_a_colliding_trajectory():
    settings = _settings()
    authored = _authored(_manifest("pile", COLLIDING, 0.5, 1.5), 1.0, settings)
    result, _violations = screen_authored(authored, settings, (0.0, 6.0))
    assert result["authored_min_sep_norm"] < 1.0
    assert len(result["worst_pair"]) == 2
    assert 0.0 <= result["worst_time_s"] <= 6.0


def test_screen_passes_a_separated_trajectory():
    settings = _settings()
    authored = _authored(_manifest("lift", SEPARATED, 0.1, 0.6), 0.3, settings)
    assert screen_authored(authored, settings, (0.0, 6.0))[0]["authored_min_sep_norm"] >= 1.0


def test_screened_candidate_never_reaches_the_solver(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        loopmod, "solve_only", lambda *a, **k: (_ for _ in ()).throw(SolverReached())
    )
    loop = _loop(monkeypatch, screen=True)
    record = loop._evaluate(_manifest("pile", COLLIDING, 0.5, 1.5), [1.0])
    assert record.stage == "screened"
    assert record.error is None
    assert record.metrics["authored_min_sep_norm"] < 1.0
    assert "separation" in record.feedback


def test_a_separated_candidate_still_reaches_the_solver(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        loopmod, "solve_only", lambda *a, **k: (_ for _ in ()).throw(SolverReached())
    )
    loop = _loop(monkeypatch, screen=True)
    with pytest.raises(SolverReached):
        loop._evaluate(_manifest("lift", SEPARATED, 0.1, 0.6), [0.3])


def test_screen_is_off_by_default_so_the_measured_path_is_unchanged(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        loopmod, "solve_only", lambda *a, **k: (_ for _ in ()).throw(SolverReached())
    )
    loop = _loop(monkeypatch, screen=False)
    with pytest.raises(SolverReached):
        loop._evaluate(_manifest("pile", COLLIDING, 0.5, 1.5), [1.0])


def test_screen_feedback_carries_the_magnitude(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        loopmod, "solve_only", lambda *a, **k: (_ for _ in ()).throw(SolverReached())
    )
    loop = _loop(monkeypatch, screen=True)
    record = loop._evaluate(_manifest("pile", COLLIDING, 0.5, 1.5), [1.0])
    norm = record.metrics["authored_min_sep_norm"]
    assert f"{norm:.3f}" in record.feedback
    assert "1.0" in record.feedback


def test_screen_reports_speed_and_acceleration():
    settings = _settings()
    authored = _authored(_manifest("lift", SEPARATED, 0.1, 0.6), 0.3, settings)
    result, _violations = screen_authored(authored, settings, (0.0, 6.0))
    assert result["authored_max_speed_mps"] >= 0.0
    assert result["authored_max_accel_mps2"] >= 0.0


def test_a_slow_separated_trajectory_breaks_no_limit():
    settings = _settings()
    authored = _authored(_manifest("lift", SEPARATED, 0.1, 0.6), 0.3, settings)
    assert screen_authored(authored, settings, (0.0, 6.0))[1] == []


def test_teleporting_trajectory_breaks_the_speed_limit():
    settings = _settings()
    authored = _authored(_manifest("dash", TELEPORTS, 1.0, 400.0), 300.0, settings)
    result, violations = screen_authored(authored, settings, (0.0, 6.0))
    assert result["authored_max_speed_mps"] > settings["axswarm"]["vel_max"]
    assert any("speed" in v for v in violations)


def test_separation_and_kinematics_are_reported_separately():
    settings = _settings()
    authored = _authored(_manifest("pile", COLLIDING, 0.5, 1.5), 1.0, settings)
    violations = screen_authored(authored, settings, (0.0, 6.0))[1]
    assert any("separation" in v for v in violations)


def test_a_kinematically_impossible_candidate_never_reaches_the_solver(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        loopmod, "solve_only", lambda *a, **k: (_ for _ in ()).throw(SolverReached())
    )
    loop = _loop(monkeypatch, screen=True)
    record = loop._evaluate(_manifest("dash", TELEPORTS, 1.0, 400.0), [300.0])
    assert record.stage == "screened"
    assert "speed" in record.feedback


def test_screen_feedback_names_the_limit_that_was_broken(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        loopmod, "solve_only", lambda *a, **k: (_ for _ in ()).throw(SolverReached())
    )
    loop = _loop(monkeypatch, screen=True)
    record = loop._evaluate(_manifest("dash", TELEPORTS, 1.0, 400.0), [300.0])
    assert str(_settings()["axswarm"]["vel_max"]) in record.feedback
