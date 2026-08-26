import numpy as np
import pytest
import yaml

from swarm_gpt.synth import loop as loopmod
from swarm_gpt.synth.loop import SynthesisLoop
from swarm_gpt.synth.manifest import ParamSpec, PrimitiveManifest
from swarm_gpt.synth.verifier import authored_trajectory, measure, screen_authored

SETTINGS = {"axswarm": {"collision_envelope": [0.25, 0.25, 0.6], "freq": 10, "vel_max": 1.73}}

# A line of drones at a chosen height, well clear of each other in x.
RISE_SOURCE = """
def rise(params, n_drones):
    height, = params
    x = np.linspace(-150.0, 150.0, n_drones)
    return np.stack([x, np.zeros(n_drones), np.full(n_drones, height)], axis=-1)
"""

# The whole shape sits at the far corner of the arena, so the swarm cannot reach it in the window.
FAR_SOURCE = """
def far(params, n_drones):
    reach, = params
    y = np.linspace(-150.0, 150.0, n_drones)
    return np.stack([np.full(n_drones, reach), y, np.full(n_drones, 100.0)], axis=-1)
"""

LIMITS = {"lower": np.array([-2.0, -2.0, 0.0]), "upper": np.array([2.0, 2.0, 2.0])}


class SolverReached(Exception):
    pass


class _UncalledClient:
    """Stands in for the OpenAI client in tests that never reach the model."""

    def with_options(self, **_kwargs: object) -> "_UncalledClient":
        return self


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


def as_authored(flown: dict) -> dict:
    """Reshape a flown (T, D, 3) trajectory into the (D, T, 3) form the screen reads."""
    return {"time": flown["time"][None, :], "pos": flown["pos"].transpose(1, 0, 2)}


def _settings() -> dict:
    return yaml.safe_load(open("swarm_gpt/data/settings.yaml").read())


def _manifest(name: str, source: str, lo: float, hi: float) -> PrimitiveManifest:
    return PrimitiveManifest(
        name=name,
        intent="test fixture",
        params=(ParamSpec(name="a", type="float", minimum=lo, maximum=hi),),
        source=source,
    )


def _spread_positions(n: int = 6) -> np.ndarray:
    pos = np.zeros((n, 3))
    pos[:, 0] = np.linspace(-1.5, 1.5, n)
    pos[:, 2] = 1.0
    return pos


def _loop(monkeypatch: pytest.MonkeyPatch, *, screen: bool) -> SynthesisLoop:
    monkeypatch.setattr(loopmod, "openai_client_for_provider", lambda *a, **k: _UncalledClient())
    return SynthesisLoop(
        settings=_settings(),
        start_pos_m=_spread_positions(),
        arm="absolute",
        model_id="test",
        duration_s=6.0,
        screen=screen,
    )


def _authored(
    manifest: PrimitiveManifest, value: float, settings: dict, duration: float = 6.0
) -> dict:
    fn, _shape_fn = manifest.compile()
    limits = {
        "lower": np.asarray(settings["axswarm"]["pos_min"], dtype=float),
        "upper": np.asarray(settings["axswarm"]["pos_max"], dtype=float),
    }
    args = manifest.bind([value])
    return authored_trajectory(fn, args, _spread_positions(), 0.0, duration, limits)


def test_authored_trajectory_holds_the_pose_and_appends_a_tail():
    fn, _shape_fn = _manifest("rise", RISE_SOURCE, 20.0, 180.0).compile()
    start = np.zeros((3, 3))
    start[:, 2] = 1.0
    authored = authored_trajectory(fn, (150.0,), start, 0.0, 4.0, LIMITS)
    assert authored["time"][0, -1] > 4.0
    assert np.allclose(authored["pos"][:, -1], authored["pos"][:, -2])


def test_authored_trajectory_clips_to_the_arena():
    fn, _shape_fn = _manifest("rise", RISE_SOURCE, 20.0, 500.0).compile()
    authored = authored_trajectory(fn, (500.0,), np.zeros((2, 3)), 0.0, 4.0, LIMITS)
    assert authored["pos"][:, :, 2].max() <= LIMITS["upper"][2]


def test_measure_finds_the_hand_computed_closest_approach():
    repaired = two_drone_pass(gap_m=0.2)
    m = measure(as_authored(repaired), repaired, SETTINGS, (0.0, 2.0))
    assert m["min_sep_m"] == pytest.approx(0.2, abs=1e-6)
    assert m["worst_time_s"] == pytest.approx(1.0, abs=1e-6)
    assert sorted(m["worst_pair"]) == [1, 2]
    # Approach is pure x, so the required separation is the x semi-axis of the envelope.
    assert m["required_sep_m"] == pytest.approx(0.25, abs=1e-6)
    assert m["min_sep_norm"] == pytest.approx(0.8, abs=1e-6)
    assert m["steps_inside_envelope"] > 0


def test_measure_reports_zero_deviation_when_the_filter_changed_nothing():
    repaired = two_drone_pass(gap_m=1.0)
    m = measure(as_authored(repaired), repaired, SETTINGS, (0.0, 2.0))
    assert m["deviation_mean_m"] == pytest.approx(0.0, abs=1e-9)
    assert m["deviation_max_m"] == pytest.approx(0.0, abs=1e-9)
    assert m["steps_inside_envelope"] == 0


def test_measure_excludes_the_settle_tail():
    repaired = two_drone_pass(gap_m=0.2, n_steps=21)
    authored = as_authored(repaired)
    full = measure(authored, repaired, SETTINGS, (0.0, 2.0))
    early = measure(authored, repaired, SETTINGS, (0.0, 0.5))
    assert early["n_steps"] < full["n_steps"]
    # The closest approach happens at t=1.0, outside the early window.
    assert early["min_sep_m"] > full["min_sep_m"]


def test_screen_flags_a_colliding_trajectory():
    result, violations = screen_authored(
        as_authored(two_drone_pass(gap_m=0.05)), _settings(), (0.0, 2.0)
    )
    assert result["authored_min_sep_norm"] < 1.0
    assert len(result["worst_pair"]) == 2
    assert any("separation" in v for v in violations)


def test_screen_passes_a_separated_trajectory():
    result, violations = screen_authored(
        as_authored(two_drone_pass(gap_m=1.0)), _settings(), (0.0, 2.0)
    )
    assert result["authored_min_sep_norm"] >= 1.0
    assert violations == []


def test_screen_reports_speed_and_acceleration():
    result, _violations = screen_authored(
        as_authored(two_drone_pass(gap_m=1.0)), _settings(), (0.0, 2.0)
    )
    assert result["authored_max_speed_mps"] >= 0.0
    assert result["authored_max_accel_mps2"] >= 0.0


def test_a_shape_out_of_reach_in_its_window_breaks_the_speed_limit():
    settings = _settings()
    authored = _authored(_manifest("far", FAR_SOURCE, 10.0, 195.0), 195.0, settings, 0.4)
    result, violations = screen_authored(authored, settings, (0.0, 0.4))
    assert result["authored_max_speed_mps"] > settings["axswarm"]["vel_max"]
    assert any("speed" in v for v in violations)


def test_a_screened_candidate_never_reaches_the_solver(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        loopmod, "solve_only", lambda *a, **k: (_ for _ in ()).throw(SolverReached())
    )
    loop = _loop(monkeypatch, screen=True)
    loop.duration_s = 0.4
    record = loop._evaluate(_manifest("far", FAR_SOURCE, 10.0, 195.0), [195.0])
    assert record.stage == "screened"
    assert record.error is None
    assert record.metrics["authored_max_speed_mps"] > loop.settings["axswarm"]["vel_max"]
    assert "speed" in record.feedback


def test_a_reachable_shape_still_reaches_the_solver(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        loopmod, "solve_only", lambda *a, **k: (_ for _ in ()).throw(SolverReached())
    )
    loop = _loop(monkeypatch, screen=True)
    with pytest.raises(SolverReached):
        loop._evaluate(_manifest("rise", RISE_SOURCE, 40.0, 160.0), [120.0])


def test_screen_is_off_by_default_so_the_measured_path_is_unchanged(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        loopmod, "solve_only", lambda *a, **k: (_ for _ in ()).throw(SolverReached())
    )
    loop = _loop(monkeypatch, screen=False)
    loop.duration_s = 0.4
    with pytest.raises(SolverReached):
        loop._evaluate(_manifest("far", FAR_SOURCE, 10.0, 195.0), [195.0])


def test_a_screened_record_carries_each_broken_limit_verbatim(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        loopmod, "solve_only", lambda *a, **k: (_ for _ in ()).throw(SolverReached())
    )
    loop = _loop(monkeypatch, screen=True)
    loop.duration_s = 0.4
    record = loop._evaluate(_manifest("far", FAR_SOURCE, 10.0, 195.0), [195.0])
    assert record.violations
    assert all(v in record.feedback for v in record.violations)
