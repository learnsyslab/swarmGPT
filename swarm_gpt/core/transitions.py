"""Min-snap transition generation and trajectory assembly (WS2).

Joins WS1's per-drone spline-1 fragments into one continuous, C2 ``PiecewiseSpline`` per
drone. A degree-8 min-snap Bezier fills the explicit gap each ``TRANSITION`` marker leaves
between consecutive fragments; the lead-in from and return-to hover are added automatically.

An authored gap too short for the distance it must cover is widened by giving up the tail of the
preceding fragment, so the next fragment still starts on its beat.
"""

from __future__ import annotations

import logging
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
# Largest share of the preceding fragment a transition may consume. Fragment 0 arrives already
# head-carved by the lead-in, so the two together can leave very little of what the LLM authored.
_MAX_TRIM = 0.9
# Trimming moves the fragment's end state, which changes the distance to cover; re-solve this often.
_TRIM_ITERS = 3

logger = logging.getLogger(__name__)


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


def _trim_for_transition(prev: _Curve, target_pos: NDArray, t_end: float) -> _Curve:
    """Shorten ``prev``'s tail until the transition to ``target_pos`` fits its speed budget.

    Args:
        prev: The preceding fragment, whose tail may be given up.
        target_pos: Position the transition must reach, in cm.
        t_end: Beat-locked time the transition must arrive at.

    Returns:
        ``prev`` unchanged, or its leading part over a shortened interval.
    """
    trimmed = prev
    for _ in range(_TRIM_ITERS):
        need = _transition_duration(target_pos - trimmed.end_state()[0])
        have = t_end - trimmed.t1
        if need <= have:
            return trimmed
        # Measured against the original fragment: successive cuts must not compound past the cap.
        extra = min(need - have, _MAX_TRIM * prev.duration - (prev.t1 - trimmed.t1))
        if extra <= 1e-9:
            break
        trimmed = trimmed.subdivide(trimmed.t1 - extra)[0]
    # A curved fragment moves its own end state as it is cut, so the last pass may have succeeded.
    need = _transition_duration(target_pos - trimmed.end_state()[0])
    have = t_end - trimmed.t1
    if need > have:
        logger.warning(
            f"Transition into t={t_end:.2f}s needs {need:.2f}s but only {have:.2f}s is available "
            "after trimming the preceding primitive; it will be flown faster than the effective "
            "speed limit"
        )
    return trimmed


def assemble_trajectory(fragments: list[_Curve], home_state: State) -> PiecewiseSpline:
    """Join per-drone spline-1 fragments into one continuous, C2 trajectory.

    The fragments are non-contiguous: an explicit ``TRANSITION`` marker upstream leaves a real
    time gap ``[prev.t1, frag.t0]`` between every consecutive pair. Each gap is filled by one
    min-snap transition connecting ``prev.end_state()`` to ``frag.start_state()`` over that
    window, widened by :func:`_trim_for_transition` when the gap is too short for the distance.
    The lead-in from hover stays automatic: a ``delta`` is carved from the head of
    fragment 0 (nothing precedes it) and a min-snap transition leads in from ``home_state``; a
    return-to-home transition is appended after the last fragment. Lead-in and return use
    :func:`_transition_duration`.

    Args:
        fragments: Ordered, non-contiguous spline-1 fragments for one drone (cm). Consecutive
            fragments must have a gap (an explicit ``TRANSITION`` window) between them.
        home_state: The drone's hover state ``(home, 0, 0)`` to lead in from and return to.

    Returns:
        One continuous C2 ``PiecewiseSpline`` over ``[fragments[0].t0, last.t1 + return]``.

    Raises:
        ValueError: If ``fragments`` is empty, or if consecutive fragments are contiguous
            (``frag.t0 <= prev.t1``), which means an upstream ``TRANSITION`` is missing.
    """
    if not fragments:
        raise ValueError("assemble_trajectory needs at least one fragment")
    home_pos = np.asarray(home_state[0], dtype=float)

    out: list[Spline] = []
    # First fragment: lead in from hover, carving delta from its head.
    first = fragments[0]
    delta = min(_transition_duration(first.start_state()[0] - home_pos), 0.9 * first.duration)
    kept = first.subdivide(first.t0 + delta)[1]
    out += _segments(transition_spline(home_state, kept.start_state(), first.t0, first.t0 + delta))
    prev = kept

    for frag in fragments[1:]:
        # The gap [prev.t1, frag.t0] is the explicit transition window.
        if frag.t0 <= prev.t1:
            raise ValueError(
                f"fragments are contiguous over [{prev.t1}, {frag.t0}]: a TRANSITION must "
                "separate consecutive primitives so a transition window exists between them"
            )
        prev = _trim_for_transition(prev, frag.start_state()[0], frag.t0)
        out += _segments(prev)
        out += _segments(transition_spline(prev.end_state(), frag.start_state(), prev.t1, frag.t0))
        prev = frag

    out += _segments(prev)
    # Return to hover after the last fragment (appended; nothing follows it).
    ret_delta = _transition_duration(home_pos - prev.end_state()[0])
    out += _segments(transition_spline(prev.end_state(), home_state, prev.t1, prev.t1 + ret_delta))
    return PiecewiseSpline(out)


def authored_span(trajectories: dict[int, PiecewiseSpline]) -> tuple[float, float]:
    """The window in which every drone is flying material the LLM actually wrote.

    :func:`assemble_trajectory` brackets each drone's fragments with a lead-in from hover and a
    return to it. Neither is authored and neither is collision-aware, so blaming the model for a
    conflict there asks it to fix something it cannot address.

    Args:
        trajectories: Drone id -> assembled trajectory, as built by :func:`assemble_trajectory`.

    Returns:
        ``(t_start, t_end)``: after the last drone's lead-in, before the first drone's return.
    """
    return (
        max(t.segments[0].t1 for t in trajectories.values()),
        min(t.segments[-1].t0 for t in trajectories.values()),
    )
