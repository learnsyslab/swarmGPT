"""Sample continuous trajectories into the discrete waypoints axswarm consumes.

**This module exists only because axswarm takes discrete waypoints.** WS1 composes primitives into
splines and WS2 joins them into one continuous C2 curve per drone; everything upstream is
continuous. When amswarm-continuous replaces axswarm, delete this file and hand the solver the
`PiecewiseSpline` directly -- nothing else needs to change.

The sampling rate is not a free choice. axswarm takes waypoints as *sparse constraints*, not as a
trajectory, and three of its properties bound the grid from both sides:

- Waypoint times snap to the MPC grid (``round((t - now) * freq)``), so two samples closer than
  ``1 / freq`` collapse onto one index, so one of the two is silently discarded.
- Only waypoints inside ``(0, K]`` steps are in the horizon. A gap wider than ``K / freq`` empties
  it -- and empties it *silently*: with an all-False mask axswarm's ``argmax`` bounds still produce
  a non-empty range, so its own "no waypoints within current horizon" guard never fires and the QP
  is corrupted rather than rejected.
- Between waypoints the solver follows its smoothness objective, not a straight line, so the chord
  criterion below is a shape-preservation heuristic and not a bound on flown error.

Geometry therefore sets the *upper* density and the MPC grid sets the lower and upper spacing.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from swarm_gpt.core.spline import PiecewiseSpline, Spline

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)

_Curve = Spline | PiecewiseSpline

# Half of `waypoints_pos_tol`, not all of it: that tolerance is already the slack on how far the
# solver may miss a waypoint, so spending the whole budget on chord deviation stacks the two.
_TOL_FRACTION = 0.5
# Guard against a pathological curve subdividing without end. Hit in practice only by a curve whose
# hull bound does not shrink, which would itself be a spline bug.
_MAX_DEPTH = 12


def _chord_bound_cm(segment: Spline) -> float:
    """Bound the segment's deviation from the straight chord joining its endpoints, in cm.

    Uses convex-hull containment on the difference curve rather than sampling it, which makes the
    result a guaranteed upper bound. Conservative at coarse subdivision, tightening quadratically.
    """
    start, end = segment.control_points[0], segment.control_points[-1]
    chord = Spline(np.stack([start, end]), segment.t0, segment.t1)
    lower, upper = (segment + chord * -1.0).axis_bounds()
    return float(np.linalg.norm(np.maximum(np.abs(lower), np.abs(upper))))


def _subdivide_until_flat(segment: Spline, tol_cm: float, depth: int = 0) -> list[float]:
    """Return the interior split times a segment needs to stay within ``tol_cm`` of its chords."""
    if depth >= _MAX_DEPTH or _chord_bound_cm(segment) <= tol_cm:
        return []
    mid = 0.5 * (segment.t0 + segment.t1)
    left, right = segment.subdivide(mid)
    return [
        *_subdivide_until_flat(left, tol_cm, depth + 1),
        mid,
        *_subdivide_until_flat(right, tol_cm, depth + 1),
    ]


def adaptive_times(curve: _Curve, tol_cm: float) -> list[float]:
    """Times at which ``curve`` must be sampled to stay within ``tol_cm`` of its chords.

    Segment boundaries are always kept: they are where a formation lands on its beat, and dropping
    one would move the moment the shape arrives.
    """
    segments = curve.segments if isinstance(curve, PiecewiseSpline) else [curve]
    times = [float(segments[0].t0)]
    for segment in segments:
        times.extend(_subdivide_until_flat(segment, tol_cm))
        times.append(float(segment.t1))
    return sorted(set(times))


def _enforce_spacing(times: list[float], min_dt: float, max_dt: float) -> list[float]:
    """Thin times closer than ``min_dt`` and fill gaps wider than ``max_dt``.

    Both bounds come from axswarm, not from the geometry: below ``min_dt`` two waypoints collide on
    one MPC index, and above ``max_dt`` the lookahead empties. The last time is always kept, so
    thinning never shortens the show.
    """
    kept = [times[0]]
    for t in times[1:-1]:
        if t - kept[-1] >= min_dt:
            kept.append(t)
    if times[-1] - kept[-1] < min_dt and len(kept) > 1:
        kept.pop()  # drop the neighbour rather than the true end of the curve
    kept.append(times[-1])

    filled = [kept[0]]
    for t in kept[1:]:
        gap = t - filled[-1]
        if gap > max_dt:
            n_fill = int(np.ceil(gap / max_dt))
            filled.extend(np.linspace(filled[-1], t, n_fill + 1)[1:-1].tolist())
        filled.append(t)
    return filled


def sample_trajectories(
    trajectories: dict[int, _Curve], settings: dict, start_pos_m: NDArray
) -> dict[str, NDArray]:
    """Sample per-drone trajectories into the ``{time, pos, vel, acc}`` dict axswarm consumes.

    Trajectories are in **cm** (WS1/WS2 convention) and the output is in **metres**: this is the
    axswarm boundary where the conversion belongs.

    Every drone shares one time grid, because the waypoints dict has a single time axis. The grid is
    the *union* of the drones' adaptive times -- dropping a time some drone needed would put that
    drone back over tolerance. It is prepended with ``t = 0`` at the hover home, because
    ``SolverData.init`` reads ``current_time`` from the first column while the sim clock starts at
    zero, and the two must agree.

    Args:
        trajectories: Drone id -> continuous curve in cm.
        settings: The parsed ``settings.yaml``; reads the ``axswarm`` block.
        start_pos_m: ``(D, 3)`` hover homes in metres, for the prepended zero column.

    Returns:
        ``time`` of shape ``(D, T)`` and ``pos``/``vel``/``acc`` of shape ``(D, T, 3)``, in metres.
    """
    axswarm = settings["axswarm"]
    freq, horizon_steps = float(axswarm["freq"]), int(axswarm["K"])
    tol_cm = float(axswarm["waypoints_pos_tol"]) * 100.0 * _TOL_FRACTION
    # A rounding margin: a pair at exactly 1/freq relies on the snap surviving float error.
    min_dt = 1.05 / freq
    # A margin under the true horizon: a waypoint landing exactly at K rounds to the boundary, and
    # the first solve happens before the swarm has moved, when the lookahead is emptiest.
    max_dt = 0.8 * horizon_steps / freq

    drone_ids = sorted(trajectories)
    union: set[float] = set()
    for drone_id in drone_ids:
        union.update(adaptive_times(trajectories[drone_id], tol_cm))
    # Zero joins the union *before* spacing is enforced. Appending it afterwards leaves a leading
    # gap nothing fills, and a show whose first key resolves past the horizon then empties the
    # lookahead silently -- the exact failure this module exists to prevent.
    union.add(0.0)
    times = _enforce_spacing(sorted(union), min_dt, max_dt)

    grid = np.asarray(times, dtype=float)
    n_drones = len(drone_ids)
    pos = np.zeros((n_drones, grid.size, 3))
    vel = np.zeros_like(pos)
    acc = np.zeros_like(pos)
    for row, drone_id in enumerate(drone_ids):
        curve = trajectories[drone_id]
        # Each drone's return-to-hover is distance-dependent, so the curves end at different times.
        # Sampling past a curve's end holds its final state, which is hover, and is what we want --
        # but only because that transition ends at rest.
        inside = grid >= curve.t0
        d_vel = curve.derivative()
        pos[row, inside] = curve.evaluate(grid[inside]) / 100.0
        vel[row, inside] = d_vel.evaluate(grid[inside]) / 100.0
        acc[row, inside] = d_vel.derivative().evaluate(grid[inside]) / 100.0
        pos[row, ~inside] = start_pos_m[row]
    logger.debug(
        "Sampled %d drones onto %d waypoints over %.1fs (%.2f Hz mean)",
        n_drones,
        grid.size,
        grid[-1],
        grid.size / max(grid[-1], 1e-9),
    )
    return {"time": np.tile(grid, (n_drones, 1)), "pos": pos, "vel": vel, "acc": acc}
