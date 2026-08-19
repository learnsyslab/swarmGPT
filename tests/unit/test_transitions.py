"""Unit tests for WS2 min-snap transitions and trajectory assembly."""

import logging
import re

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


# --- transition trimming -------------------------------------------------------------


def _speed_bound(pw: PiecewiseSpline) -> float:
    """Peak control-point speed bound over every segment, in m/s (positions are cm)."""
    return (
        max(np.linalg.norm(s.derivative().control_points, axis=-1).max() for s in pw.segments) / 100
    )


def test_generous_gap_leaves_the_preceding_fragment_untouched():
    # 20 cm apart over a 4 s gap needs 0.5 s: nothing to trim.
    fragments = [_hold([0.0, 0.0, 100.0], 0.0, 3.0), _hold([20.0, 0.0, 100.0], 7.0, 9.0)]
    home = (np.array([0.0, 0.0, 100.0]), np.zeros(3), np.zeros(3))
    traj = assemble_trajectory(fragments, home)
    holds = [s for s in traj.segments if s.degree == 0]
    assert any(np.isclose(s.t1, 3.0) for s in holds), "first fragment should still end at t=3"


def test_short_gap_trims_the_preceding_fragment_and_keeps_the_next_on_its_beat():
    # 400 cm apart over a 0.5 s gap needs 5.2 s at V_EFF=1.0 m/s with 1.3 headroom.
    fragments = [_hold([0.0, 0.0, 100.0], 0.0, 8.0), _hold([400.0, 0.0, 100.0], 8.5, 10.0)]
    home = (np.array([0.0, 0.0, 100.0]), np.zeros(3), np.zeros(3))
    traj = assemble_trajectory(fragments, home)
    holds = [s for s in traj.segments if s.degree == 0]
    first = min(holds, key=lambda s: s.t0)
    assert first.t1 < 8.0, "the preceding fragment must give up part of its tail"
    assert any(np.isclose(s.t0, 8.5) for s in holds), "the next fragment still starts on its beat"


def test_trimming_slows_the_transition_leg_itself():
    # Scored on the transition into t=8.5 alone: the return-to-home leg dominates the whole-
    # trajectory maximum and would mask the effect.
    fragments = [_hold([0.0, 0.0, 100.0], 0.0, 8.0), _hold([400.0, 0.0, 100.0], 8.5, 10.0)]
    home = (np.array([0.0, 0.0, 100.0]), np.zeros(3), np.zeros(3))
    traj = assemble_trajectory(fragments, home)
    leg = next(s for s in traj.segments if np.isclose(s.t1, 8.5) and s.degree > 0)
    speed = np.linalg.norm(leg.derivative().control_points, axis=-1).max() / 100
    # Untrimmed, the same 4 m move over the authored 0.5 s gap bounds above 24 m/s.
    assert speed < 2.5, f"transition leg still at {speed:.2f} m/s"
    assert leg.t0 < 8.0, "the leg must have been widened backwards into the fragment"


@pytest.fixture
def transitions_log(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> pytest.LogCaptureFixture:
    """`caplog`, wired so it actually sees this module's records.

    The ROS `launch` pytest plugin calls `logging.setLoggerClass` with ``propagate = False``, so
    every logger built after it bypasses the root logger `caplog` listens on.
    """
    logger = logging.getLogger("swarm_gpt.core.transitions")
    monkeypatch.setattr(logger, "propagate", True)
    caplog.set_level(logging.WARNING, logger=logger.name)
    return caplog


def test_trim_is_capped_and_warns_rather_than_raising(
    transitions_log: pytest.LogCaptureFixture,
) -> None:
    # A short preceding fragment cannot buy enough time even when fully consumed.
    fragments = [_hold([0.0, 0.0, 100.0], 0.0, 0.4), _hold([400.0, 0.0, 100.0], 0.5, 2.0)]
    home = (np.array([0.0, 0.0, 100.0]), np.zeros(3), np.zeros(3))
    traj = assemble_trajectory(fragments, home)
    assert "flown faster than the effective speed limit" in transitions_log.text
    _assert_continuous(traj, order=2)


def test_trimmed_assembly_stays_c2_and_ends_when_it_should():
    fragments = [_hold([0.0, 0.0, 100.0], 0.0, 8.0), _hold([400.0, 0.0, 100.0], 8.5, 10.0)]
    home = (np.array([0.0, 0.0, 100.0]), np.zeros(3), np.zeros(3))
    traj = assemble_trajectory(fragments, home)
    _assert_continuous(traj, order=2)
    assert traj.t1 > 10.0  # only the return-to-home extends past the last fragment
    np.testing.assert_allclose(traj.end_state()[0], home[0], atol=1e-8)


def test_trimming_works_on_a_piecewise_fragment():
    seg_a = Spline(np.array([[0.0, 0.0, 100.0], [50.0, 0.0, 100.0]]), t0=0.0, t1=4.0)
    seg_b = Spline(np.array([[50.0, 0.0, 100.0], [50.0, 80.0, 100.0]]), t0=4.0, t1=8.0)
    fragments = [PiecewiseSpline([seg_a, seg_b]), _hold([400.0, 0.0, 100.0], 8.5, 10.0)]
    home = (np.array([0.0, 0.0, 100.0]), np.zeros(3), np.zeros(3))
    traj = assemble_trajectory(fragments, home)
    _assert_continuous(traj, order=2)
    assert any(np.isclose(s.t0, 8.5) for s in traj.segments)


def test_warning_reports_the_state_after_trimming_not_before(
    transitions_log: pytest.LogCaptureFixture,
) -> None:
    # A curved fragment moves its own end state as it is cut, so figures measured before the last
    # cut are stale. Whatever the warning claims must hold of the trajectory actually produced.
    arc = Spline(
        np.array([[0.0, 0.0, 100.0], [120.0, 60.0, 100.0], [200.0, 0.0, 100.0]]), t0=0.0, t1=6.0
    )
    fragments = [arc, _hold([260.0, 0.0, 100.0], 6.4, 8.0)]
    home = (np.array([0.0, 0.0, 100.0]), np.zeros(3), np.zeros(3))
    assemble_trajectory(fragments, home)
    assert transitions_log.records, "this fixture is meant to bottom out and warn"
    for record in transitions_log.records:
        needed, available = (float(x) for x in re.findall(r"(\d+\.\d+)s", record.message)[-2:])
        assert needed > available, f"warned but budget was met: {record.message}"


def test_no_warning_when_the_final_trim_meets_the_budget(
    transitions_log: pytest.LogCaptureFixture,
) -> None:
    # Tuned so the third and last cut is the one that satisfies the budget (3.9167s needed against
    # 3.9180s available). Checking only at the top of each pass misses it and warns falsely.
    arc = Spline(
        np.array([[-88.4, 62.0, 100.0], [193.7, -236.0, 100.0], [-60.0, 73.3, 100.0]]),
        t0=0.0,
        t1=6.17,
    )
    fragments = [arc, _hold([-240.4, -60.2, 100.0], 7.05, 9.05)]
    home = (np.array([0.0, 0.0, 100.0]), np.zeros(3), np.zeros(3))
    assemble_trajectory(fragments, home)
    assert transitions_log.records == []


def test_trimming_across_three_fragments_keeps_every_beat():
    fragments = [
        _hold([0.0, 0.0, 100.0], 0.0, 6.0),
        _hold([300.0, 0.0, 100.0], 6.3, 12.0),
        _hold([0.0, 300.0, 100.0], 12.3, 14.0),
    ]
    home = (np.array([0.0, 0.0, 100.0]), np.zeros(3), np.zeros(3))
    traj = assemble_trajectory(fragments, home)
    _assert_continuous(traj, order=2)
    starts = {round(s.t0, 3) for s in traj.segments}
    assert 6.3 in starts and 12.3 in starts, "both later fragments must still start on their beats"
    holds = [s for s in traj.segments if s.degree == 0]
    assert max(s.t1 for s in holds if s.t0 < 6.0) < 6.0, "first fragment must give up its tail"
    assert max(s.t1 for s in holds if 6.0 < s.t0 < 12.0) < 12.0, "so must the middle one"
