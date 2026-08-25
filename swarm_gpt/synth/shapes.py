"""Shape predicates the primitive's author does not write.

The model's own invariants pass on trajectories that are not the requested shape: two flat
counter-rotating rings satisfied all five checks it wrote for a double helix. These predicates are
hand-written, selected by the person making the request, and never shown to the model as source,
so satisfying one means building the shape rather than describing it.

They run on the flown trajectory in the same form the author's own checks get: ``(D, T, 3)`` in cm.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

# Below this the strands sit at one height, which reads as a ring however fast it turns.
MIN_STRAND_CLIMB_CM = 20.0
# Fraction of the climb that must advance in one direction; a fraction rather than all-or-nothing
# because the filter perturbs flown positions and near-equal levels can swap order harmlessly.
MIN_CLIMB_MONOTONICITY = 0.8
# Total twist from bottom to top. Less than this is a ladder, not a helix.
MIN_TWIST_RAD = np.pi / 2
# Heights must cluster into pairs: the gap between levels this many times the gap within one.
# A ratio rather than an absolute tolerance, because drones evenly spaced in a single file
# sit exactly on any half-the-spacing bound, and the filter's noise moves an absolute one.
PAIR_SEPARATION_RATIO = 3.0
# How far a pair may sit from truly opposite before the strands stop reading as interleaved.
OPPOSED_TOL_RAD = np.deg2rad(40.0)


def _final_geometry(pos: NDArray) -> tuple[NDArray, NDArray]:
    """Each drone's angle about the swarm axis and its height, at the formed pose."""
    centre = pos[:, :, :2].mean(axis=(0, 1))
    final = pos[:, -1, :]
    return np.arctan2(final[:, 1] - centre[1], final[:, 0] - centre[0]), final[:, 2]


def _double_helix(pos: NDArray, time: NDArray) -> list[tuple[str, bool, str]]:
    """Two strands half a turn apart, both climbing, twisting together about a common axis.

    Both strands share a handedness: strands that counter-rotate sweep through each other and
    cannot be flown. What separates a double helix from a single one is the second strand held
    opposite the first at every height, so that is what is checked here rather than rotation.
    """
    del time
    n = pos.shape[0]
    if n < 4 or n % 2:
        return [("paired_heights", False, f"{n} drones cannot be split into two equal strands")]

    angle, z = _final_geometry(pos)
    pairs = np.argsort(z).reshape(-1, 2)
    levels = z[pairs].mean(axis=1)
    order = np.argsort(levels)

    # Walking up the sorted heights, the gaps alternate: inside a pair, then between levels.
    gaps = np.diff(np.sort(z))
    within = float(np.median(gaps[0::2]))
    spacing = float(np.median(gaps[1::2])) if gaps[1::2].size else 0.0
    paired = spacing > 0.0 and spacing >= PAIR_SEPARATION_RATIO * within

    apart = np.abs(np.mod(angle[pairs[:, 0]] - angle[pairs[:, 1]] + np.pi, 2 * np.pi) - np.pi)
    opposed = float(np.degrees(apart.min()))

    span = float(levels.max() - levels.min())

    # A pair's orientation is a line, not a direction, so it is read modulo pi -- which also makes
    # it independent of which drone of the pair you happen to pick.
    twist = np.unwrap(np.mod(angle[pairs[order, 0]], np.pi), period=np.pi)
    steps = np.diff(twist)
    advance = max((steps >= 0).sum(), (steps <= 0).sum()) / steps.size if steps.size else 0.0
    total = float(abs(twist[-1] - twist[0]))

    return [
        (
            "paired_heights",
            bool(paired),
            f"heights sit {within:.1f} cm apart within a pair and {spacing:.1f} cm between "
            f"levels, a ratio of {spacing / within if within else float('inf'):.1f} "
            f"(needs {PAIR_SEPARATION_RATIO:.0f})",
        ),
        (
            "strands_opposed",
            bool(apart.min() >= np.pi - OPPOSED_TOL_RAD),
            f"the least opposed pair sits {opposed:.0f} deg apart, needs "
            f"{180 - np.degrees(OPPOSED_TOL_RAD):.0f}-180 deg",
        ),
        (
            "strands_climb",
            span >= MIN_STRAND_CLIMB_CM,
            f"the strands span {span:.1f} cm of height, needs {MIN_STRAND_CLIMB_CM:.0f} cm",
        ),
        (
            "twists_with_height",
            bool(advance >= MIN_CLIMB_MONOTONICITY and total >= MIN_TWIST_RAD),
            f"the pair axis turns {np.degrees(total):.0f} deg from bottom to top, advancing "
            f"steadily for {advance:.0%} of the climb; needs "
            f"{np.degrees(MIN_TWIST_RAD):.0f} deg at {MIN_CLIMB_MONOTONICITY:.0%}",
        ),
    ]


SHAPES: dict[str, tuple[Callable[[NDArray, NDArray], list[tuple[str, bool, str]]], str]] = {
    "double_helix": (
        _double_helix,
        "two strands winding around a common vertical axis, HALF A TURN APART at every height "
        "and both turning the same way. Drones pair up: at each height there is one drone from "
        "each strand, on opposite sides of the axis. Going up, the pair's orientation must rotate "
        "steadily, at least 90 degrees from bottom to top, which is what makes it a helix rather "
        "than a ladder. Two flat rings at two altitudes is NOT a double helix, however fast they "
        "turn. Strands that turn in OPPOSITE directions sweep through each other and cannot be "
        "flown -- give both strands the same handedness and keep them opposed by phase.",
    )
}


def check_shape(name: str, pos_cm: NDArray, time: NDArray) -> list[dict[str, Any]]:
    """Run the named shape predicate over a flown trajectory.

    Args:
        name: A key of ``SHAPES``.
        pos_cm: Flown positions, ``(D, T, 3)`` in cm.
        time: Timestamps, ``(T,)`` in seconds.

    Returns:
        One ``{"name", "ok", "detail"}`` entry per property.

    Raises:
        KeyError: If ``name`` is not a known shape.
    """
    predicate, _description = SHAPES[name]
    return [
        {"name": str(n), "ok": bool(ok), "detail": str(detail)}
        for n, ok, detail in predicate(np.asarray(pos_cm, dtype=float), np.asarray(time))
    ]


def describe_shape(name: str) -> str:
    """Return the prose the requester's shape requirement is stated to the model as.

    Raises:
        KeyError: If ``name`` is not a known shape.
    """
    _predicate, description = SHAPES[name]
    return description
