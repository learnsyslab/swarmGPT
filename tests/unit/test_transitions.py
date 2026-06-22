"""Unit tests for WS2 min-snap transitions and trajectory assembly."""

import numpy as np
import pytest

from swarm_gpt.core.spline import PiecewiseSpline, Spline
from swarm_gpt.core.transitions import assemble_trajectory, transition_spline


def _hold(point, t0, t1):  # noqa: ANN001, ANN202
    return Spline(np.asarray(point, dtype=float)[None, :], t0=t0, t1=t1)


def _assert_continuous(pw: PiecewiseSpline, order: int, atol: float = 1e-6) -> None:
    """Assert C^order continuity across every internal join of a piecewise spline."""
    for left, right in zip(pw.segments[:-1], pw.segments[1:]):
        for d in range(order + 1):
            np.testing.assert_allclose(left.end_state()[d], right.start_state()[d], atol=atol)


# --- transition_spline ---------------------------------------------------------------


def test_transition_meets_both_boundary_states_exactly():
    rng = np.random.default_rng(0)
    s0 = (rng.standard_normal(3), rng.standard_normal(3), rng.standard_normal(3))
    s1 = (rng.standard_normal(3), rng.standard_normal(3), rng.standard_normal(3))
    tr = transition_spline(s0, s1, 1.0, 3.0)
    assert tr.degree == 8
    for d in range(3):
        np.testing.assert_allclose(tr.start_state()[d], s0[d], atol=1e-8)
        np.testing.assert_allclose(tr.end_state()[d], s1[d], atol=1e-8)


def test_rest_to_rest_transition_is_symmetric():
    s0 = (np.zeros(3), np.zeros(3), np.zeros(3))
    s1 = (np.array([2.0, 0.0, 0.0]), np.zeros(3), np.zeros(3))
    tr = transition_spline(s0, s1, 0.0, 1.0)
    np.testing.assert_allclose(tr.evaluate(0.5), [1.0, 0.0, 0.0], atol=1e-9)


def test_transition_rejects_empty_interval():
    s = (np.zeros(3), np.zeros(3), np.zeros(3))
    with pytest.raises(ValueError, match="t1 > t0"):
        transition_spline(s, s, 2.0, 2.0)


# --- assemble_trajectory -------------------------------------------------------------


def test_assembly_is_contiguous_and_c2_for_formations():
    # Consecutive fragments have a gap [3, 4] that is the explicit transition window.
    fragments = [_hold([100.0, 0.0, 100.0], 0.0, 3.0), _hold([0.0, 150.0, 120.0], 4.0, 6.0)]
    home = (np.array([0.0, 0.0, 100.0]), np.zeros(3), np.zeros(3))
    traj = assemble_trajectory(fragments, home)
    assert isinstance(traj, PiecewiseSpline)
    _assert_continuous(traj, order=2)


def test_assembly_rejects_contiguous_fragments_missing_transition():
    # Gapless fragments mean an upstream TRANSITION is missing; assembly must reject them.
    fragments = [_hold([100.0, 0.0, 100.0], 0.0, 3.0), _hold([0.0, 150.0, 120.0], 3.0, 6.0)]
    home = (np.array([0.0, 0.0, 100.0]), np.zeros(3), np.zeros(3))
    with pytest.raises(ValueError, match="TRANSITION must"):
        assemble_trajectory(fragments, home)


def test_assembly_leads_in_from_home_and_returns_home():
    fragments = [_hold([100.0, 50.0, 120.0], 0.0, 4.0)]
    home = (np.array([0.0, 0.0, 100.0]), np.zeros(3), np.zeros(3))
    traj = assemble_trajectory(fragments, home)
    assert traj.t0 == 0.0
    assert traj.t1 > 4.0  # return-to-home is appended after the last fragment
    np.testing.assert_allclose(traj.start_state()[0], home[0], atol=1e-8)
    np.testing.assert_allclose(traj.end_state()[0], home[0], atol=1e-8)
    np.testing.assert_allclose(traj.start_state()[1], 0.0, atol=1e-8)  # starts from hover
    np.testing.assert_allclose(traj.end_state()[1], 0.0, atol=1e-8)  # ends at hover


def test_assembly_handles_piecewise_fragments():
    # A multi-segment (arc-like) fragment must work: assemble uses .duration/.subdivide on it.
    seg_a = Spline(np.array([[0.0, 0.0, 100.0], [50.0, 0.0, 100.0]]), t0=0.0, t1=2.0)
    seg_b = Spline(np.array([[50.0, 0.0, 100.0], [50.0, 80.0, 100.0]]), t0=2.0, t1=4.0)
    fragment = PiecewiseSpline([seg_a, seg_b])
    home = (np.array([0.0, 0.0, 100.0]), np.zeros(3), np.zeros(3))
    traj = assemble_trajectory([fragment], home)
    assert traj.t0 == 0.0
    np.testing.assert_allclose(traj.start_state()[0], home[0], atol=1e-8)
    np.testing.assert_allclose(traj.end_state()[0], home[0], atol=1e-8)


def test_assembly_rejects_empty_fragments():
    home = (np.zeros(3), np.zeros(3), np.zeros(3))
    with pytest.raises(ValueError, match="at least one"):
        assemble_trajectory([], home)
