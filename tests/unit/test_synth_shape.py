import numpy as np
import pytest
import yaml

from swarm_gpt.synth import loop as loopmod
from swarm_gpt.synth.loop import _RESPONSE_SCHEMA, SynthesisLoop
from swarm_gpt.synth.manifest import ParamSpec, PrimitiveManifest
from swarm_gpt.synth.sandbox import SynthError
from swarm_gpt.synth.shape import screen_shape, targets
from swarm_gpt.synth.verifier import authored_trajectory, screen_authored

RING = """
def ring(params, n_drones):
    radius, = params
    angles = np.linspace(0, 2 * np.pi, n_drones, endpoint=False)
    return np.stack(
        [radius * np.cos(angles), radius * np.sin(angles), np.full(n_drones, 100.0)], axis=-1
    )
"""

STACK = """
def stack(params, n_drones):
    spacing, = params
    z = 60.0 + spacing * np.arange(n_drones)
    return np.stack([np.zeros(n_drones), np.zeros(n_drones), z], axis=-1)
"""

WRONG_SHAPE = """
def flat(params, n_drones):
    size, = params
    return np.full(3, size)
"""

LIMITS = {"lower": np.array([-2.0, -2.0, 0.25]), "upper": np.array([2.0, 2.0, 1.7])}


class SolverReached(Exception):
    pass


class _UncalledClient:
    """Stands in for the OpenAI client in tests that never reach the model."""

    def with_options(self, **_kwargs: object) -> "_UncalledClient":
        return self


def _settings() -> dict:
    return yaml.safe_load(open("swarm_gpt/data/settings.yaml").read())


def _manifest(name: str, source: str, lo: float, hi: float) -> PrimitiveManifest:
    return PrimitiveManifest(
        name=name,
        intent="test fixture",
        params=(ParamSpec(name="a", type="float", minimum=lo, maximum=hi),),
        source=source,
    )


def _docked(n: int = 8) -> np.ndarray:
    """The lab's dock ring, at the show's start height."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([1.5 * np.cos(angles), 1.5 * np.sin(angles), np.full(n, 1.0)], axis=-1)


def _loop(monkeypatch: pytest.MonkeyPatch) -> SynthesisLoop:
    monkeypatch.setattr(loopmod, "openai_client_for_provider", lambda *a, **k: _UncalledClient())
    return SynthesisLoop(
        settings=_settings(),
        start_pos_m=_docked(),
        arm="absolute",
        model_id="test",
        duration_s=8.0,
        screen=True,
    )


def test_a_shape_is_wrapped_into_the_primitive_contract():
    manifest = _manifest("ring", RING, 50.0, 180.0)
    fn, _shape_fn = manifest.compile()
    docked = _docked() * 100
    final_pos, waypoints = fn(manifest.bind([150.0]), docked, 0.0, 8.0, LIMITS)
    assert final_pos.shape == (8, 3)
    assert waypoints and all(0.0 < t <= 8.0 for t in waypoints)


def test_the_swarm_arrives_within_the_speed_limit_without_the_author_scheduling_it():
    settings = _settings()
    manifest = _manifest("ring", RING, 50.0, 180.0)
    fn, _shape_fn = manifest.compile()
    authored = authored_trajectory(fn, manifest.bind([50.0]), _docked(), 0.0, 8.0, LIMITS)
    measured, violations = screen_authored(authored, settings, (0.0, 8.0))
    assert measured["authored_max_speed_mps"] <= settings["axswarm"]["vel_max"]
    assert violations == []


def test_the_arrival_corner_is_not_reported_as_a_broken_acceleration_limit():
    """The corner reads far above acc_max however gently the swarm flies through it."""
    settings = _settings()
    manifest = _manifest("ring", RING, 50.0, 180.0)
    fn, _shape_fn = manifest.compile()
    authored = authored_trajectory(fn, manifest.bind([50.0]), _docked(), 0.0, 8.0, LIMITS)
    measured, violations = screen_authored(authored, settings, (0.0, 8.0))
    assert measured["authored_max_accel_mps2"] > settings["axswarm"]["acc_max"]
    assert violations == []


def test_screen_flags_a_crowded_shape_with_the_gap_in_centimetres():
    manifest = _manifest("stack", STACK, 5.0, 100.0)
    _fn, shape_fn = manifest.compile()
    des_pos = targets(shape_fn, manifest.bind([20.0]), 8)
    measured, violations = screen_shape(des_pos, _settings())
    assert measured["shape_min_sep_norm"] < 1.0
    assert measured["shape_worst_gap_cm"] == pytest.approx(20.0)
    assert "20 cm apart" in violations[0]


def test_screen_passes_a_shape_that_clears_the_envelope():
    manifest = _manifest("ring", RING, 50.0, 180.0)
    _fn, shape_fn = manifest.compile()
    des_pos = targets(shape_fn, manifest.bind([150.0]), 8)
    measured, violations = screen_shape(des_pos, _settings())
    assert measured["shape_min_sep_norm"] >= 1.0
    assert violations == []


def test_a_crowded_shape_never_reaches_the_solver(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        loopmod, "solve_only", lambda *a, **k: (_ for _ in ()).throw(SolverReached())
    )
    loop = _loop(monkeypatch)
    record = loop._evaluate(_manifest("stack", STACK, 5.0, 100.0), [20.0])
    assert record.stage == "shaped"
    assert record.error is None
    assert record.metrics["shape_min_sep_norm"] < 1.0
    assert "too close together" in record.feedback


def test_a_spread_shape_still_reaches_the_solver(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        loopmod, "solve_only", lambda *a, **k: (_ for _ in ()).throw(SolverReached())
    )
    loop = _loop(monkeypatch)
    with pytest.raises(SolverReached):
        loop._evaluate(_manifest("ring", RING, 50.0, 180.0), [150.0])


def test_a_shape_returning_the_wrong_array_is_told_what_was_expected():
    manifest = _manifest("flat", WRONG_SHAPE, 1.0, 10.0)
    _fn, shape_fn = manifest.compile()
    with pytest.raises(SynthError, match=r"one \(x, y, z\) position per drone"):
        targets(shape_fn, manifest.bind([5.0]), 8)


def test_the_schema_asks_for_a_shape_and_nothing_else():
    manifest = _RESPONSE_SCHEMA["properties"]["manifest"]
    assert sorted(manifest["required"]) == ["intent", "name", "params", "source"]
    assert "invariants" not in manifest["properties"]
