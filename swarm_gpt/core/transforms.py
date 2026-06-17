"""M(t) transform builders for WS1 primitive composition.

Polynomial transforms (translate, scale, shear) use the WS0 ``Spline`` algebra exactly.
Rotation uses a frozen canonical cubic quarter-circle, affine-placed per drone. No
construction-time sampling. See ``docs/specs/2026-06-17-swarmgpt2-trajectory-rewrite.md``.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from swarm_gpt.core.spline import PiecewiseSpline, Spline

if TYPE_CHECKING:
    from numpy.typing import NDArray

_K_ARC = 4.0 / 3.0 * math.tan(math.pi / 8.0)
CANONICAL_ARC: NDArray = np.array([[1.0, 0.0], [1.0, _K_ARC], [_K_ARC, 1.0], [0.0, 1.0]])
"""Control points of a unit cubic Bezier quarter-circle (0 to pi/2), ~0.0273 % radial error."""


def _subdivide_bezier(control_points: NDArray, t: float) -> NDArray:
    """Return control points of the sub-curve over ``[0, t]`` via de Casteljau.

    Args:
        control_points: Bezier control points, shape ``(n + 1, dim)``.
        t: Split parameter in ``[0, 1]``.

    Returns:
        Control points of the left sub-curve, same shape as the input.
    """
    pts = np.asarray(control_points, dtype=float).copy()
    left = [pts[0].copy()]
    for _ in range(1, len(pts)):
        pts = (1.0 - t) * pts[:-1] + t * pts[1:]
        left.append(pts[0].copy())
    return np.array(left)


def _quarter_arc_xy(phi0: float) -> NDArray:
    """Return canonical quarter-arc control points rotated to start at angle ``phi0``.

    Args:
        phi0: Start angle in radians; the quarter sweeps ``phi0`` to ``phi0 + pi/2``.

    Returns:
        Control points of shape ``(4, 2)`` on the unit circle.
    """
    c, s = math.cos(phi0), math.sin(phi0)
    return CANONICAL_ARC @ np.array([[c, -s], [s, c]]).T


def arc_spline(
    center: NDArray, radius: float, phi0: float, dphi: float, t0: float, t1: float
) -> PiecewiseSpline:
    """Build a circular arc spline of angular span ``dphi`` for one drone.

    Assembled from affine-placed canonical quarter-circles (plus one subdivided partial
    quarter), at constant height ``center[2]``. Time is split evenly across quarters.

    Args:
        center: Arc center ``(x, y, z)`` in cm.
        radius: Arc radius in cm.
        phi0: Start angle in radians.
        dphi: Signed angular sweep in radians.
        t0: Block start time in seconds.
        t1: Block end time in seconds.

    Returns:
        A ``PiecewiseSpline`` of cubic 3-D segments over ``[t0, t1]``.
    """
    sign = 1.0 if dphi >= 0 else -1.0
    n_quarters = max(1, int(math.ceil(abs(dphi) / (math.pi / 2) - 1e-9)))
    edges = np.linspace(t0, t1, n_quarters + 1)
    segments: list[Spline] = []
    angle, remaining = phi0, abs(dphi)
    for q in range(n_quarters):
        step = min(remaining, math.pi / 2)
        unit = _quarter_arc_xy(angle if sign > 0 else angle - math.pi / 2)
        if sign < 0:
            unit = unit[::-1]
        sub = _subdivide_bezier(unit, step / (math.pi / 2))
        xy = sub * radius + center[:2]
        segments.append(
            Spline(np.hstack([xy, np.full((4, 1), center[2])]), t0=edges[q], t1=edges[q + 1])
        )
        angle += sign * step
        remaining -= step
    return PiecewiseSpline(segments)


def linear_translate(start: NDArray, end: NDArray, t0: float, t1: float) -> Spline:
    """Return a degree-1 straight-line spline from ``start`` to ``end`` (cm).

    Args:
        start: Start position in cm, shape ``(3,)``.
        end: End position in cm, shape ``(3,)``.
        t0: Block start time in seconds.
        t1: Block end time in seconds.

    Returns:
        A degree-1 ``Spline`` over ``[t0, t1]``.
    """
    return Spline(np.vstack([start, end]), t0=t0, t1=t1)


def linear_scale(s0: float, s1: float, t0: float, t1: float) -> Spline:
    """Return a degree-1 scalar (dim=1) spline ramping ``s0`` to ``s1``.

    Args:
        s0: Start scalar value.
        s1: End scalar value.
        t0: Block start time in seconds.
        t1: Block end time in seconds.

    Returns:
        A degree-1 ``Spline`` of dimension 1 over ``[t0, t1]``.
    """
    return Spline(np.array([[s0], [s1]]), t0=t0, t1=t1)


def affine_offset(
    home: NDArray, center: NDArray, matrix_end: NDArray, t0: float, t1: float
) -> Spline:
    """Return the displacement spline of a one-way affine applied about ``center``.

    The displacement ramps linearly from zero to ``matrix_end @ (home-center) - (home-center)``,
    i.e. the final affine map (scale/shear) is reached at ``t1``. Used for ``scale``/``shear``.

    Args:
        home: Drone home position in cm.
        center: Affine center in cm (e.g. the formation centroid).
        matrix_end: The 3x3 linear map at ``t1`` (e.g. ``diag(s,s,1)`` or a shear).
        t0: Block start time in seconds.
        t1: Block end time in seconds.

    Returns:
        A degree-1 3-D displacement ``Spline`` (zero at ``t0``).
    """
    offset = home - center
    end_disp = matrix_end @ offset - offset
    return Spline(np.vstack([np.zeros(3), end_disp]), t0=t0, t1=t1)


def zigzag_translate(
    start: NDArray, steps: int, delta_xy: NDArray, delta_z: NDArray, t0: float, t1: float
) -> PiecewiseSpline:
    """Return a piecewise-linear zig-zag translate (one segment per step).

    Each step alternates the sign of the horizontal displacement while the vertical
    displacement accumulates upward, matching the legacy ``zig_zag``.

    Args:
        start: Start position in cm.
        steps: Number of zig-zag segments.
        delta_xy: Per-step horizontal displacement ``(dx, dy, 0)`` in cm.
        delta_z: Per-step vertical displacement ``(0, 0, dz)`` in cm.
        t0: Block start time in seconds.
        t1: Block end time in seconds.

    Returns:
        A ``PiecewiseSpline`` of ``steps`` degree-1 segments.
    """
    edges = np.linspace(t0, t1, steps + 1)
    pos = np.asarray(start, dtype=float).copy()
    segments: list[Spline] = []
    for i in range(steps):
        nxt = pos + (-1.0) ** i * delta_xy + delta_z
        segments.append(Spline(np.vstack([pos, nxt]), t0=edges[i], t1=edges[i + 1]))
        pos = nxt
    return PiecewiseSpline(segments)
