"""Min-snap transition generation and trajectory assembly (WS2).

Joins WS1's per-drone spline-1 fragments into one continuous, C2 ``PiecewiseSpline`` per
drone. A degree-8 min-snap Bezier smooths every seam; each transition's duration is borrowed
from the tail of the preceding fragment so formations land on their beat.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from swarm_gpt.core.motion_primitives import (
    _FORMATION_HEADROOM,
    _FORMATION_T_MIN_S,
    _FORMATION_V_EFF_MPS,
)
from swarm_gpt.core.spline import PiecewiseSpline, Spline

if TYPE_CHECKING:
    from numpy.typing import NDArray

State = tuple["NDArray", "NDArray", "NDArray"]
"""A boundary state ``(position, velocity, acceleration)``, each shape ``(dim,)``."""

_Curve = Spline | PiecewiseSpline
_TRANSITION_DEGREE = 8


def _snap_gram(degree: int) -> NDArray:
    """Return the ``(degree+1, degree+1)`` Gram matrix of the snap functional in the basis.

    For a degree-``n`` Bezier the 4th derivative is a degree-``n-4`` Bezier whose control
    points are 4th forward differences of the originals. ``Q = Mᵀ G M`` where ``M`` maps
    control points to those differences and ``G`` is the degree-``n-4`` Bernstein mass matrix.
    The overall positive scalar (powers of duration) is dropped — it does not change the
    minimiser.

    Args:
        degree: Bezier degree ``n`` (must be at least 4).

    Returns:
        The snap Gram matrix ``Q`` of shape ``(degree + 1, degree + 1)``.
    """
    n = degree
    r = 4
    diff = np.array([(-1) ** (r - k) * math.comb(r, k) for k in range(r + 1)], dtype=float)
    m_rows = n - r + 1
    m = np.zeros((m_rows, n + 1))
    for i in range(m_rows):
        m[i, i : i + r + 1] = diff
    md = n - r
    g = np.zeros((md + 1, md + 1))
    for i in range(md + 1):
        for j in range(md + 1):
            g[i, j] = (
                math.comb(md, i) * math.comb(md, j) / ((2 * md + 1) * math.comb(2 * md, i + j))
            )
    return m.T @ g @ m


def transition_spline(start_state: State, end_state: State, t0: float, t1: float) -> Spline:
    """Build a degree-8 min-snap transition between two C2 boundary states.

    The first/last three control points are pinned by the boundary positions, velocities and
    accelerations (Bernstein endpoint locality); the middle three minimise the snap integral
    ``∫(d⁴/dt⁴)²``. A tiny ridge keeps the (otherwise near-singular) free block well-posed.

    Args:
        start_state: ``(p, v, a)`` at ``t0``, each shape ``(dim,)``.
        end_state: ``(p, v, a)`` at ``t1``, each shape ``(dim,)``.
        t0: Transition start time in seconds.
        t1: Transition end time in seconds (must exceed ``t0``).

    Returns:
        A degree-8 ``Spline`` over ``[t0, t1]`` meeting both boundary states exactly.

    Raises:
        ValueError: If ``t1 <= t0``.
    """
    if t1 <= t0:
        raise ValueError(f"transition interval must satisfy t1 > t0, got [{t0}, {t1}]")
    n = _TRANSITION_DEGREE
    duration = t1 - t0
    p0, v0, a0 = (np.asarray(s, dtype=float) for s in start_state)
    p1, v1, a1 = (np.asarray(s, dtype=float) for s in end_state)
    dim = p0.shape[0]

    cp = np.zeros((n + 1, dim))
    cp[0] = p0
    cp[1] = p0 + v0 * duration / n
    cp[2] = a0 * duration**2 / (n * (n - 1)) + 2 * cp[1] - cp[0]
    cp[n] = p1
    cp[n - 1] = p1 - v1 * duration / n
    cp[n - 2] = a1 * duration**2 / (n * (n - 1)) + 2 * cp[n - 1] - cp[n]

    q = _snap_gram(n)
    free = [3, 4, 5]
    fixed = [0, 1, 2, 6, 7, 8]
    q_ff = q[np.ix_(free, free)]
    q_fx = q[np.ix_(free, fixed)]
    ridge = 1e-9 * (np.trace(q_ff) / len(free) + 1.0)
    cp[free] = np.linalg.solve(q_ff + ridge * np.eye(len(free)), -q_fx @ cp[fixed])
    return Spline(cp, t0, t1)


def _transition_duration(displacement_cm: NDArray) -> float:
    """Min-snap travel-time floor for a displacement, reusing the formation physics constants.

    Args:
        displacement_cm: Per-drone displacement vector(s) in cm; the bottleneck (max) norm sets
            the duration.

    Returns:
        Transition duration in seconds, at least ``_FORMATION_T_MIN_S``.
    """
    max_travel_m = float(np.linalg.norm(np.atleast_2d(displacement_cm), axis=-1).max()) / 100.0
    return max(max_travel_m / _FORMATION_V_EFF_MPS * _FORMATION_HEADROOM, _FORMATION_T_MIN_S)


def _segments(curve: _Curve) -> list[Spline]:
    """Return a curve's component segments as a list of single ``Spline`` pieces.

    Args:
        curve: A ``Spline`` or ``PiecewiseSpline``.

    Returns:
        The component segments (a one-element list for a single ``Spline``).
    """
    return curve.segments if isinstance(curve, PiecewiseSpline) else [curve]


def assemble_trajectory(fragments: list[_Curve], home_state: State) -> PiecewiseSpline:
    """Join per-drone spline-1 fragments into one continuous, C2 trajectory.

    Every seam is smoothed by a min-snap transition whose duration is borrowed from the tail of
    the preceding fragment (so formations land on their beat). The first transition borrows from
    the head of fragment 0 (nothing precedes it), leading in from ``home_state``; a
    return-to-home transition is appended after the last fragment.

    Args:
        fragments: Ordered, time-contiguous spline-1 fragments for one drone (cm).
        home_state: The drone's hover state ``(home, 0, 0)`` to lead in from and return to.

    Returns:
        One continuous C2 ``PiecewiseSpline`` over ``[fragments[0].t0, last.t1 + return]``.

    Raises:
        ValueError: If ``fragments`` is empty.
    """
    if not fragments:
        raise ValueError("assemble_trajectory needs at least one fragment")
    home_pos = np.asarray(home_state[0], dtype=float)

    out: list[Spline] = []
    # First fragment: lead in from hover, borrowing from its head.
    first = fragments[0]
    delta = min(_transition_duration(first.start_state()[0] - home_pos), 0.9 * first.duration)
    kept = first.subdivide(first.t0 + delta)[1]
    out += _segments(transition_spline(home_state, kept.start_state(), first.t0, first.t0 + delta))
    pending = kept

    for frag in fragments[1:]:
        # Borrow delta from the pending fragment's tail; transition into frag.
        delta = min(
            _transition_duration(frag.start_state()[0] - pending.end_state()[0]),
            0.9 * pending.duration,
        )
        cut = pending.t1 - delta
        kept_prev = pending.subdivide(cut)[0]
        out += _segments(kept_prev)
        out += _segments(
            transition_spline(kept_prev.end_state(), frag.start_state(), cut, pending.t1)
        )
        pending = frag

    out += _segments(pending)
    # Return to hover after the last fragment (appended; nothing follows it).
    ret_delta = _transition_duration(home_pos - pending.end_state()[0])
    out += _segments(
        transition_spline(pending.end_state(), home_state, pending.t1, pending.t1 + ret_delta)
    )
    return PiecewiseSpline(out)
