"""Field (D) generators for WS1: per-drone time-varying displacement splines.

All temporal oscillation reduces to a single frozen canonical quarter-sine, tiled and
amplitude-scaled. No construction-time sampling or least-squares fitting: continuous
per-drone phase is handled by ``sin(theta+phi) = sin(theta) cos(phi) + cos(theta) sin(phi)``
applied at the control-point level. Fields stack additively, which removes the legacy
last-write-wins merge.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from swarm_gpt.core.spline import PiecewiseSpline, Spline

if TYPE_CHECKING:
    from numpy.typing import NDArray

_K_SINE = math.pi / 6.0
CANONICAL_SINE: NDArray = np.array([0.0, _K_SINE, 1.0, 1.0])
"""Cubic-Hermite quarter-sine control points approximating sin(pi*u/2) on [0,1]."""

# Signed/reversed quarters tiling one full sine period (boundary values 0,1,0,-1,0).
_SIN_QUARTERS: NDArray = np.array(
    [
        [0.0, _K_SINE, 1.0, 1.0],
        [1.0, 1.0, _K_SINE, 0.0],
        [0.0, -_K_SINE, -1.0, -1.0],
        [-1.0, -1.0, -_K_SINE, 0.0],
    ]
)
# Cosine quarters are the sine quarters advanced by one quarter-period.
_COS_QUARTERS: NDArray = np.roll(_SIN_QUARTERS, -1, axis=0)


def sine_field(
    homes: NDArray, t0: float, t1: float, periods: int, amplitude: NDArray, phase: NDArray
) -> dict[int, PiecewiseSpline]:
    """Return a per-drone sinusoidal displacement field, built from frozen constants.

    Displacement is ``amplitude_i * sin(2*pi*periods*(t-t0)/(t1-t0) + phase_i)``. The
    period is tiled from four canonical quarters per period; per-drone phase is applied
    via the angle-sum identity on the quarter control points, so no curve is sampled.

    Args:
        homes: Drone home positions in cm, shape ``(n, 3)`` (used only for the index set).
        t0: Block start time in seconds.
        t1: Block end time in seconds.
        periods: Number of full oscillation periods over the block (>= 1).
        amplitude: Per-drone amplitude vectors ``(n, 3)`` in cm (direction x magnitude).
        phase: Per-drone phase offsets ``(n,)`` in radians.

    Returns:
        Mapping of drone id to a 3-D displacement ``PiecewiseSpline``.
    """
    n_seg = 4 * periods
    edges = np.linspace(t0, t1, n_seg + 1)
    field: dict[int, PiecewiseSpline] = {}
    for i in range(len(homes)):
        cphi, sphi = math.cos(phase[i]), math.sin(phase[i])
        segments: list[Spline] = []
        for q in range(n_seg):
            qi = q % 4
            value = cphi * _SIN_QUARTERS[qi] + sphi * _COS_QUARTERS[qi]  # (4,) scalar cps
            cp = value[:, None] * amplitude[i][None, :]  # (4, 3)
            segments.append(Spline(cp, t0=edges[q], t1=edges[q + 1]))
        field[i] = PiecewiseSpline(segments)
    return field
