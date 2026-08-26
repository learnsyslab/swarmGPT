"""A synthesized primitive that is only its geometry.

`form_circle` is three lines of trigonometry followed by `_assign_positions` and
`_formation_waypoints`: the equation is the primitive's contribution, and the library flies the
swarm there. Asking the model for the equation alone and wrapping it the same way removes the
layer -- picking an arrival time, interpolating, keeping the fly-in apart -- that every rejected
run fell over in.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from swarm_gpt.core.motion_primitives import _assign_positions, _formation_waypoints
from swarm_gpt.synth.sandbox import validate_shape

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray


def targets(shape_fn: Callable[..., Any], params: tuple, n_drones: int) -> NDArray:
    """Evaluate a shape function and check its output contract.

    Returns:
        The ``(n_drones, 3)`` target positions, in cm.
    """
    return validate_shape(shape_fn(params, n_drones), n_drones)


def as_primitive(shape_fn: Callable[..., Any]) -> Callable[..., tuple]:
    """Wrap a shape function in the five-argument primitive contract.

    Returns:
        A primitive that flies the swarm into the shape and holds it for the interval.
    """

    def primitive(
        params: tuple, swarm_pos: NDArray, tstart: float, tend: float, limits: dict[str, NDArray]
    ) -> tuple[NDArray, dict[float, dict[int, NDArray]]]:
        """Fly the swarm into the authored shape and hold it."""
        des_pos = targets(shape_fn, params, swarm_pos.shape[0])
        des_pos = np.clip(des_pos, limits["lower"] * 100, limits["upper"] * 100)
        target = des_pos[_assign_positions(swarm_pos, des_pos)]
        # 0.0 asks for the earliest arrival physics allows, leaving the rest of the interval to
        # hold the shape -- there is no author-chosen duration to honour.
        waypoints = _formation_waypoints(target, swarm_pos, tstart, tend, 0.0)
        return target, waypoints

    return primitive


def screen_shape(des_pos: NDArray, settings: dict) -> tuple[dict[str, Any], list[str]]:
    """Judge the geometry alone, before anything is flown.

    Separating this from the trajectory screen is what makes the feedback act on the equation: a
    close pair here is the shape being too dense, not the fly-in crossing.

    Returns:
        The measured dict, and one sentence per pair too close together (empty if the shape is
        spaced well enough to fly).
    """
    envelope = np.asarray(settings["axswarm"]["collision_envelope"], dtype=float) * 100
    scaled = des_pos / envelope
    norm = np.linalg.norm(scaled[:, None, :] - scaled[None, :, :], axis=-1)
    norm = norm + np.eye(len(des_pos)) * 1e6
    i, j = np.unravel_index(norm.argmin(), norm.shape)
    gap = float(np.linalg.norm(des_pos[i] - des_pos[j]))
    measured = {
        "shape_min_sep_norm": float(norm[i, j]),
        "shape_worst_pair": [int(i), int(j)],
        "shape_worst_gap_cm": gap,
    }
    if measured["shape_min_sep_norm"] >= 1.0:
        return measured, []
    return measured, [
        f"points {i} and {j} of your shape are {gap:.0f} cm apart, which is "
        f"{measured['shape_min_sep_norm']:.3f} of the separation two drones need there. Spread "
        f"the shape out, or sample fewer points along the crowded part of it."
    ]
