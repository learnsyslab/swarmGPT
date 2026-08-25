import numpy as np
import pytest

from swarm_gpt.synth.manifest import ParamSpec, PrimitiveManifest
from swarm_gpt.synth.sandbox import (
    SynthError,
    call_guarded,
    compile_invariants,
    compile_primitive,
    validate_waypoints,
)

GOOD_SOURCE = """
def rise(params, swarm_pos, tstart, tend, limits):
    delta, = params
    pos = swarm_pos.copy()
    waypoints = {}
    for t in np.linspace(tstart, tend, 3)[1:]:
        pos = pos + np.array([0.0, 0.0, delta / 2])
        waypoints[float(t)] = {i: p.copy() for i, p in enumerate(pos)}
    return pos, waypoints
"""

GOOD_CHECK = """
def check(pos, time, params):
    climbed = pos[:, -1, 2] - pos[:, 0, 2]
    return [("rises", bool(np.all(climbed > 0)), "every drone ended higher than it started")]
"""


def run_good(n_drones: int = 4, tstart: float = 0.0, tend: float = 4.0) -> tuple:
    fn = compile_primitive(GOOD_SOURCE, "rise")
    swarm_pos = np.zeros((n_drones, 3))
    limits = {"lower": np.array([-2.0, -2.0, 0.0]), "upper": np.array([2.0, 2.0, 2.0])}
    return fn((50.0,), swarm_pos, tstart, tend, limits)


def test_compiles_and_runs_a_valid_primitive():
    final_pos, waypoints = validate_waypoints(run_good(), 4, 0.0, 4.0)
    assert final_pos.shape == (4, 3)
    assert all(0.0 < t <= 4.0 for t in waypoints)
    assert np.allclose(final_pos[:, 2], 50.0)


def test_compiled_check_returns_triples():
    check = compile_invariants(GOOD_CHECK)
    pos = np.zeros((3, 5, 3))
    pos[:, :, 2] = np.linspace(0, 100, 5)
    result = check(pos, np.linspace(0, 4, 5), (50.0,))
    assert result == [("rises", True, "every drone ended higher than it started")]


@pytest.mark.parametrize(
    "source",
    [
        "import os\ndef f(params, swarm_pos, tstart, tend, limits):\n    return None, {}",
        "def f(params, swarm_pos, tstart, tend, limits):\n    from os import system\n    return 1",
        "def f(params, swarm_pos, tstart, tend, limits):\n    return f.__globals__, {}",
        "def f(params, swarm_pos, tstart, tend, limits):\n    return eval('1'), {}",
        "def f(params, swarm_pos, tstart, tend, limits):\n    return open('x'), {}",
    ],
)
def test_rejects_escapes(source: str):
    with pytest.raises(SynthError):
        compile_primitive(source, "f")


def test_rejects_wrong_signature():
    with pytest.raises(SynthError, match="must take exactly"):
        compile_primitive("def f(a, b):\n    return a, b", "f")


def test_rejects_missing_function():
    with pytest.raises(SynthError, match="must define a function named"):
        compile_primitive("def other(params, swarm_pos, tstart, tend, limits):\n    return 1", "f")


def test_rejects_top_level_statements():
    source = "x = 1\ndef f(params, swarm_pos, tstart, tend, limits):\n    return x, {}"
    with pytest.raises(SynthError, match="only function definitions"):
        compile_primitive(source, "f")


def test_rejects_syntax_error():
    with pytest.raises(SynthError, match="not valid Python"):
        compile_primitive("def f(:", "f")


def test_errors_name_the_manifest_field_they_came_from():
    """Both fields hold Python, so an unlabelled parse error sends the author to the wrong one."""
    prose = "The check tests radial placement and diametric opposition."
    with pytest.raises(SynthError, match="`invariants` field is not valid Python"):
        compile_invariants(prose)
    with pytest.raises(SynthError, match="`source` field is not valid Python"):
        compile_primitive(prose, "f")


def test_call_guarded_converts_exceptions():
    fn = compile_primitive("def f(params, swarm_pos, tstart, tend, limits):\n    return 1 / 0", "f")
    with pytest.raises(SynthError, match="ZeroDivisionError"):
        call_guarded(fn, (), np.zeros((2, 3)), 0.0, 1.0, {})


def test_call_guarded_times_out():
    fn = compile_primitive(
        "def f(params, swarm_pos, tstart, tend, limits):\n    while True:\n        pass\n", "f"
    )
    with pytest.raises(SynthError, match="exceeded"):
        call_guarded(fn, (), np.zeros((2, 3)), 0.0, 1.0, {})


def test_rejects_nan_positions():
    result = (np.zeros((2, 3)), {1.0: {0: np.array([np.nan, 0.0, 0.0]), 1: np.zeros(3)}})
    with pytest.raises(SynthError, match="not finite"):
        validate_waypoints(result, 2, 0.0, 2.0)


def test_rejects_wrong_position_shape():
    result = (np.zeros((2, 3)), {1.0: {0: np.zeros(2), 1: np.zeros(3)}})
    with pytest.raises(SynthError, match=r"shape \(3,\)"):
        validate_waypoints(result, 2, 0.0, 2.0)


def test_rejects_time_outside_window():
    result = (np.zeros((2, 3)), {9.0: {0: np.zeros(3), 1: np.zeros(3)}})
    with pytest.raises(SynthError, match="outside the interval"):
        validate_waypoints(result, 2, 0.0, 2.0)


def test_rejects_time_at_window_start():
    result = (np.zeros((2, 3)), {0.0: {0: np.zeros(3), 1: np.zeros(3)}})
    with pytest.raises(SynthError, match="outside the interval"):
        validate_waypoints(result, 2, 0.0, 2.0)


def test_rejects_out_of_range_drone_id():
    result = (np.zeros((2, 3)), {1.0: {5: np.zeros(3)}})
    with pytest.raises(SynthError, match="drone ids are"):
        validate_waypoints(result, 2, 0.0, 2.0)


def test_rejects_wrong_final_pos_shape():
    with pytest.raises(SynthError, match="final_pos must have shape"):
        validate_waypoints((np.zeros((3, 3)), {1.0: {0: np.zeros(3)}}), 2, 0.0, 2.0)


def test_accepts_numpy_scalar_times():
    times = np.linspace(0.0, 2.0, 3)[1:]
    result = (np.zeros((1, 3)), {t: {0: np.zeros(3)} for t in times})
    _, waypoints = validate_waypoints(result, 1, 0.0, 2.0)
    assert len(waypoints) == 2


def test_manifest_binds_and_range_checks():
    manifest = PrimitiveManifest(
        name="rise",
        intent="go up",
        params=(ParamSpec("delta", "int", 0, 100),),
        source=GOOD_SOURCE,
        invariants=GOOD_CHECK,
    )
    assert manifest.bind([50.4]) == (50,)
    with pytest.raises(SynthError, match="outside its declared range"):
        manifest.bind([500])
    with pytest.raises(SynthError, match="takes 1 arguments"):
        manifest.bind([1, 2])


def test_manifest_from_payload_rejects_bad_params():
    payload = {
        "name": "f",
        "intent": "i",
        "source": GOOD_SOURCE,
        "invariants": GOOD_CHECK,
        "params": [{"name": "a", "type": "str", "minimum": 0, "maximum": 1}],
    }
    with pytest.raises(SynthError, match="use one of"):
        PrimitiveManifest.from_payload(payload)


def test_manifest_from_payload_rejects_missing_field():
    with pytest.raises(SynthError, match="missing required fields"):
        PrimitiveManifest.from_payload({"name": "f"})


ASSIGNS = """
def ring(params, swarm_pos, tstart, tend, limits):
    radius, = params
    n = swarm_pos.shape[0]
    targets = np.zeros_like(swarm_pos)
    for i in range(n):
        a = 2.0 * np.pi * i / n
        targets[i] = np.array([radius * np.cos(a), radius * np.sin(a), swarm_pos[i, 2]])
    slots = assign(swarm_pos, targets)
    placed = np.zeros_like(swarm_pos)
    for i in range(n):
        placed[i] = targets[slots[i]]
    t = arrival_time(placed, swarm_pos, tstart, tend)
    return placed, {float(t): {i: p.copy() for i, p in enumerate(placed)}}
"""


def test_helpers_are_bound_in_the_sandbox():
    fn = compile_primitive(ASSIGNS, "ring")
    swarm = np.zeros((6, 3))
    swarm[:, 0] = np.linspace(-150, 150, 6)
    limits = {"lower": np.array([-2.0, -2.0, 0.0]), "upper": np.array([2.0, 2.0, 2.0])}
    placed, waypoints = fn((120.0,), swarm, 0.0, 10.0, limits)
    assert placed.shape == (6, 3)
    assert len(waypoints) == 1


def test_assign_returns_a_permutation():
    from swarm_gpt.synth.sandbox import HELPERS

    current = np.array([[0.0, 0.0, 0.0], [100.0, 0.0, 0.0], [200.0, 0.0, 0.0]])
    targets = current[::-1].copy()
    slots = HELPERS["assign"](current, targets)
    assert sorted(slots) == [0, 1, 2]
    # Each drone is sent to the target that is already where it stands, not to its own index.
    assert [int(s) for s in slots] == [2, 1, 0]


def test_arrival_time_respects_the_interval():
    from swarm_gpt.synth.sandbox import HELPERS

    current = np.zeros((4, 3))
    far = np.tile(np.array([300.0, 0.0, 0.0]), (4, 1))
    assert HELPERS["arrival_time"](far, current, 0.0, 1.0) == 1.0
    assert 0.0 < HELPERS["arrival_time"](far, current, 0.0, 30.0) < 30.0
