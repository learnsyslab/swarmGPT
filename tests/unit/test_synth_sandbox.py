import threading
from collections.abc import Callable

import numpy as np
import pytest

from swarm_gpt.synth.manifest import ParamSpec, PrimitiveManifest
from swarm_gpt.synth.sandbox import (
    SynthError,
    call_guarded,
    compile_shape,
    validate_shape,
    validate_waypoints,
)

GOOD_SOURCE = """
def rise(params, n_drones):
    delta, = params
    x = np.linspace(-150.0, 150.0, n_drones)
    return np.stack([x, np.zeros(n_drones), np.full(n_drones, delta)], axis=-1)
"""

SPIN = "def f(params, n_drones):\n    while True:\n        pass\n"
DIVIDES_BY_ZERO = "def f(params, n_drones):\n    return 1 / 0\n"


def run_good(n_drones: int = 4) -> np.ndarray:
    return compile_shape(GOOD_SOURCE, "rise")((50.0,), n_drones)


def test_compiles_and_runs_a_valid_shape():
    positions = validate_shape(run_good(), 4)
    assert positions.shape == (4, 3)
    assert np.allclose(positions[:, 2], 50.0)


@pytest.mark.parametrize(
    "source",
    [
        "import os\ndef f(params, n_drones):\n    return None",
        "def f(params, n_drones):\n    from os import system\n    return 1",
        "def f(params, n_drones):\n    return f.__globals__",
        "def f(params, n_drones):\n    return eval('1')",
        "def f(params, n_drones):\n    return open('x')",
    ],
)
def test_rejects_escapes(source: str):
    with pytest.raises(SynthError):
        compile_shape(source, "f")


def test_rejects_wrong_signature():
    with pytest.raises(SynthError, match="must take exactly"):
        compile_shape("def f(a, b):\n    return a, b", "f")


def test_rejects_missing_function():
    with pytest.raises(SynthError, match="must define a function named"):
        compile_shape("def other(params, n_drones):\n    return 1", "f")


def test_rejects_top_level_statements():
    with pytest.raises(SynthError, match="only function definitions"):
        compile_shape("x = 1\ndef f(params, n_drones):\n    return x", "f")


def test_rejects_syntax_error():
    with pytest.raises(SynthError, match="not valid Python"):
        compile_shape("def f(:", "f")


def test_a_prose_answer_is_named_as_a_source_error():
    prose = "The shape places drones around a heart-shaped curve."
    with pytest.raises(SynthError, match="`source` field is not valid Python"):
        compile_shape(prose, "f")


def test_the_formation_helpers_are_not_reachable_from_a_shape():
    """A shape is geometry; assignment and arrival timing happen outside the sandbox."""
    with pytest.raises(SynthError, match="NameError"):
        call_guarded(compile_shape("def f(params, n_drones):\n    return assign(1, 2)", "f"), (), 2)


def test_call_guarded_converts_exceptions():
    with pytest.raises(SynthError, match="ZeroDivisionError"):
        call_guarded(compile_shape(DIVIDES_BY_ZERO, "f"), (), 2)


def test_call_guarded_times_out():
    with pytest.raises(SynthError, match="exceeded"):
        call_guarded(compile_shape(SPIN, "f"), (), 2)


def test_rejects_a_shape_of_the_wrong_size():
    with pytest.raises(SynthError, match="one .x, y, z. position per drone"):
        validate_shape(np.zeros((3, 3)), 2)


def test_rejects_a_shape_with_a_non_finite_point():
    positions = np.zeros((2, 3))
    positions[1, 0] = np.nan
    with pytest.raises(SynthError, match="NaN or inf"):
        validate_shape(positions, 2)


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
        name="rise", intent="go up", params=(ParamSpec("delta", "int", 0, 100),), source=GOOD_SOURCE
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
        "params": [{"name": "a", "type": "str", "minimum": 0, "maximum": 1}],
    }
    with pytest.raises(SynthError, match="use one of"):
        PrimitiveManifest.from_payload(payload)


def test_manifest_from_payload_rejects_missing_field():
    with pytest.raises(SynthError, match="missing required fields"):
        PrimitiveManifest.from_payload({"name": "f"})


def _in_thread(target: Callable[[], None]) -> threading.Thread:
    worker = threading.Thread(target=target)
    worker.start()
    worker.join(30.0)
    return worker


def test_call_guarded_runs_off_the_main_thread():
    result = {}

    def _run():
        result["value"] = call_guarded(compile_shape(GOOD_SOURCE, "rise"), (50.0,), 4)

    assert not _in_thread(_run).is_alive()
    assert result["value"].shape == (4, 3)


def test_call_guarded_times_out_off_the_main_thread():
    fn = compile_shape(SPIN, "f")
    result = {}

    def _run():
        try:
            call_guarded(fn, (), 2)
        except SynthError as e:
            result["error"] = str(e)

    assert not _in_thread(_run).is_alive()
    assert "exceeded" in result["error"]


def test_call_guarded_reports_a_raise_off_the_main_thread():
    fn = compile_shape(DIVIDES_BY_ZERO, "f")
    result = {}

    def _run():
        try:
            call_guarded(fn, (), 2)
        except SynthError as e:
            result["error"] = str(e)

    assert not _in_thread(_run).is_alive()
    assert "ZeroDivisionError" in result["error"]
