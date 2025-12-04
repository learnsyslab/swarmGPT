"""Collection of utility functions for the swarm_gpt package."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from numpy.typing import NDArray as Array


def discretize_bspline(
    bsplines: dict[int, list], t_end: float, freq: float = 100, derivative: int = 0
) -> dict[int, Array]:
    """Discretizes bsplines with the given frequency up to t_end."""
    assert derivative >= 0, f"Derivative must be >=0, was {derivative}"
    for _ in range(derivative):
        bsplines = {i: [s.derivative() for s in bsplines[i]] for i in bsplines}
    waypoints = {i: [] for i in bsplines}
    des_time = np.arange(0, t_end, 1.0 / freq)
    for t in des_time:
        for i in bsplines:
            waypoints[i].append([s(t) for s in bsplines[i]])
    for i in waypoints.keys():
        waypoints[i] = np.array(waypoints[i])
    return waypoints
