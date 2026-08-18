"""Unit tests for the axswarm sampling bridge (WS4)."""

from pathlib import Path

import numpy as np
import pytest
import yaml

from swarm_gpt.core.sampling import _Curve, adaptive_times, sample_trajectories
from swarm_gpt.core.spline import PiecewiseSpline, Spline

SETTINGS = yaml.safe_load(
    (Path(__file__).resolve().parents[2] / "swarm_gpt/data/settings.yaml").read_text()
)
AX = SETTINGS["axswarm"]
MIN_DT = 1.0 / AX["freq"]
MAX_DT = AX["K"] / AX["freq"]
TOL_CM = AX["waypoints_pos_tol"] * 100.0


def line(t0: float = 0.0, t1: float = 4.0, length_cm: float = 300.0) -> Spline:
    """A straight segment: zero curvature, so no interior sample is needed for shape."""
    return Spline(np.array([[0.0, 0.0, 100.0], [length_cm, 0.0, 100.0]]), t0, t1)


def arc(t0: float = 0.0, t1: float = 4.0, radius: float = 150.0) -> Spline:
    """A cubic quarter-circle: genuinely curved, so it must subdivide."""
    k = 4.0 / 3.0 * np.tan(np.pi / 8.0)
    cp = np.array(
        [
            [radius, 0.0, 100.0],
            [radius, k * radius, 100.0],
            [k * radius, radius, 100.0],
            [0.0, radius, 100.0],
        ]
    )
    return Spline(cp, t0, t1)


def max_chord_error_cm(curve: _Curve, times: list[float]) -> float:
    """Worst deviation of the curve from linear interpolation between the chosen samples."""
    worst = 0.0
    for a, b in zip(times, times[1:]):
        probe = np.linspace(a, b, 25)
        exact = curve.evaluate(probe)
        s = ((probe - a) / (b - a))[:, None]
        chord = (1 - s) * curve.evaluate(np.array([a])) + s * curve.evaluate(np.array([b]))
        worst = max(worst, float(np.linalg.norm(exact - chord, axis=-1).max()))
    return worst


def test_a_straight_line_needs_no_interior_samples_for_shape():
    assert adaptive_times(line(), TOL_CM) == [0.0, 4.0]


def test_a_curve_subdivides_until_within_tolerance():
    times = adaptive_times(arc(), tol_cm=5.0)
    assert len(times) > 2
    assert max_chord_error_cm(arc(), times) <= 5.0


def test_tightening_the_tolerance_adds_samples():
    coarse = adaptive_times(arc(), tol_cm=20.0)
    fine = adaptive_times(arc(), tol_cm=1.0)
    assert len(fine) > len(coarse)


def test_segment_boundaries_always_survive():
    """A formation lands on its beat at a segment boundary; dropping one moves the arrival."""
    curve = PiecewiseSpline([line(0.0, 3.0), line(3.0, 6.0, length_cm=10.0)])
    assert 3.0 in adaptive_times(curve, TOL_CM)


def _sample(curve: _Curve, n_drones: int = 2) -> dict:
    trajectories = {i: curve for i in range(n_drones)}
    start = np.tile(np.array([0.0, 0.0, 1.0]), (n_drones, 1))
    return sample_trajectories(trajectories, SETTINGS, start)


def test_no_two_waypoints_collapse_onto_one_mpc_index():
    """Below 1/freq two waypoints round to the same index and hard-constrain it twice."""
    wp = _sample(arc(0.0, 2.0))
    gaps = np.diff(wp["time"][0])
    assert gaps.min() >= MIN_DT - 1e-9, f"closest pair {gaps.min():.4f}s < {MIN_DT}s"


def test_no_gap_empties_the_mpc_lookahead():
    """A long straight move samples to two points on shape alone; the horizon guard must fill it."""
    wp = _sample(line(0.0, 40.0, length_cm=50.0))
    gaps = np.diff(wp["time"][0])
    assert gaps.max() < MAX_DT, f"widest gap {gaps.max():.2f}s >= horizon {MAX_DT}s"


def test_the_grid_starts_at_zero_so_the_sim_and_solver_clocks_agree():
    """SolverData.init reads current_time from column 0 while the sim clock starts at 0."""
    wp = _sample(arc(5.0, 9.0))
    assert wp["time"][0, 0] == pytest.approx(0.0)
    assert wp["pos"][0, 0] == pytest.approx([0.0, 0.0, 1.0])


def test_the_grid_ends_at_the_true_end_of_show():
    """render.py and backend.py read time[0, -1] as the show duration."""
    wp = _sample(arc(0.0, 7.0))
    assert wp["time"][0, -1] == pytest.approx(7.0)


def test_output_is_metres_and_correctly_shaped():
    wp = _sample(arc(0.0, 4.0), n_drones=3)
    t = wp["time"].shape[1]
    assert wp["time"].shape == (3, t)
    assert wp["pos"].shape == wp["vel"].shape == wp["acc"].shape == (3, t, 3)
    # The arc lives at radius 150 cm, so every sample is ~1.5 m from the axis, not ~150.
    assert np.linalg.norm(wp["pos"][0, -1, :2]) == pytest.approx(1.5, abs=0.01)


def test_velocity_comes_from_the_spline_derivative_not_finite_differences():
    """Inert under the shipped config (vel_constraints false), so this pins a future config."""
    curve = arc(0.0, 4.0)
    wp = _sample(curve)
    expected = curve.derivative().evaluate(wp["time"][0]) / 100.0
    assert np.allclose(wp["vel"][0], expected)


def test_the_union_grid_keeps_a_time_only_one_drone_needed():
    """Dropping it would put that drone back over tolerance."""
    trajectories = {0: line(0.0, 6.0), 1: arc(0.0, 6.0)}
    start = np.tile(np.array([0.0, 0.0, 1.0]), (2, 1))
    wp = sample_trajectories(trajectories, SETTINGS, start)
    curved_times = adaptive_times(arc(0.0, 6.0), AX["waypoints_pos_tol"] * 100.0 * 0.5)
    grid = wp["time"][0]
    # Every time the curved drone needed survives, up to the MPC-spacing thinning.
    assert sum(np.isclose(grid, t).any() for t in curved_times) >= len(curved_times) - 1


def test_drones_whose_curve_starts_late_hold_their_home():
    trajectories = {0: arc(0.0, 6.0), 1: arc(4.0, 6.0)}
    start = np.array([[0.0, 0.0, 1.0], [1.0, 1.0, 1.0]])
    wp = sample_trajectories(trajectories, SETTINGS, start)
    early = wp["time"][0] < 4.0
    assert np.allclose(wp["pos"][1, early], np.array([1.0, 1.0, 1.0]))


def test_a_late_first_key_does_not_leave_an_unfilled_leading_gap():
    """The t=0 column must join the union before spacing is enforced, or its gap is never filled.

    A show whose first key resolves past the horizon would otherwise empty axswarm's lookahead --
    and empty it silently, because its own guard cannot detect an all-False mask.
    """
    wp = _sample(arc(20.0, 26.0))
    gaps = np.diff(wp["time"][0])
    assert wp["time"][0, 0] == pytest.approx(0.0)
    assert gaps.max() < MAX_DT, f"leading gap {gaps.max():.1f}s >= horizon {MAX_DT}s"
