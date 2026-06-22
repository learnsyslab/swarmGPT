"""Tests for F1: time_to_finish_s on formation primitives."""

import numpy as np

from swarm_gpt.core.motion_primitives import form_circle, form_cone, form_star


def _limits() -> dict:
    return {"lower": np.array([-2.2, -2.7, 0.25]), "upper": np.array([2.2, 2.7, 1.7])}


def _swarm_10() -> np.ndarray:
    return np.array(
        [[x, y, 100] for x in (-200, -100, 0, 100, 200) for y in (-100, 100)], dtype=float
    )


def test_form_star_respects_time_to_finish():
    """Large time_to_finish_s → arrival should be ~5s into a 10s interval, not at physics min."""
    swarm = _swarm_10()
    limits = _limits()
    _, wps = form_star((100, 60, 80, 5.0), swarm, 0.0, 10.0, limits)
    times = sorted(wps.keys())
    assert 4.5 <= times[0] <= 5.5, f"expected arrival ~5s, got {times[0]}"


def test_form_star_clamps_below_physics_min():
    """Tiny time_to_finish_s → should clamp UP to the physics floor (>= T_MIN = 0.5s)."""
    swarm = _swarm_10()
    limits = _limits()
    _, wps = form_star((100, 60, 80, 0.05), swarm, 0.0, 10.0, limits)
    times = sorted(wps.keys())
    assert times[0] >= 0.5, f"expected clamp to physics floor, got {times[0]}"


def test_form_star_clamps_above_interval():
    """time_to_finish_s larger than the interval → arrival should be clamped to tend."""
    swarm = _swarm_10()
    limits = _limits()
    _, wps = form_star((100, 60, 80, 999.0), swarm, 0.0, 5.0, limits)
    times = sorted(wps.keys())
    # Arrival must not exceed tend
    assert times[0] <= 5.0, f"arrival {times[0]} exceeds tend=5.0"


def test_form_circle_respects_time_to_finish():
    """form_circle with large time_to_finish_s should arrive late in the interval."""
    swarm = _swarm_10()
    limits = _limits()
    drone_ids = list(range(1, 6))  # drones 1-5
    _, wps = form_circle((drone_ids, 100, 100, 8.0), swarm, 0.0, 10.0, limits)
    times = sorted(wps.keys())
    assert times[0] >= 7.0, f"expected late arrival, got {times[0]}"


def test_form_cone_respects_time_to_finish():
    """form_cone with time_to_finish_s close to tend → arrival near end of interval."""
    swarm = _swarm_10()
    limits = _limits()
    _, wps = form_cone((50, 60, 0, 8.0), swarm, 0.0, 10.0, limits)
    times = sorted(wps.keys())
    assert times[0] >= 7.0, f"expected late arrival, got {times[0]}"
