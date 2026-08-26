"""Render one measurement dict as the feedback forms the ablation compares.

All three arms read the same dict from `verifier.measure`, so they differ in wording alone and a
reviewer cannot object that a weaker arm was handed weaker information. The arms are:

- ``categorical`` -- what swarmGPT sends today (`Choreographer._collision_check`): who and roughly
  when, plus canned advice. No magnitudes.
- ``absolute`` -- the same events with their measured magnitudes in metres.
- ``relative`` -- the same magnitudes as ratios and comparatives, with no raw units anywhere.
  This is the arm that tests whether the claim needs absolute numbers or only ordered ones.
"""

from __future__ import annotations

from typing import Any

ARMS = ("categorical", "absolute", "relative")

# Ratio bins for the relative arm. Ordered coarse-to-fine so a lookup takes the first bin the
# ratio falls under; deliberately vague, because the point of the arm is to carry order without
# carrying a number.
_RATIO_BINS: tuple[tuple[float, str], ...] = (
    (0.15, "a small fraction of"),
    (0.35, "about a third of"),
    (0.6, "about half of"),
    (0.85, "somewhat less than"),
    (1.15, "about"),
    (1.75, "somewhat more than"),
    (2.5, "roughly twice"),
    (4.0, "roughly three times"),
    (8.0, "several times"),
)
_RATIO_LARGE = "many times"


def _ratio_words(ratio: float) -> str:
    """Describe ``ratio`` as a comparative phrase carrying its order but no digits."""
    for bound, words in _RATIO_BINS:
        if ratio < bound:
            return words
    return _RATIO_LARGE


def _fraction_words(part: int, whole: int) -> str:
    """Describe ``part`` out of ``whole`` as a coarse frequency phrase."""
    if part == 0:
        return "none"
    share = part / max(whole, 1)
    if share < 0.02:
        return "a handful"
    if share < 0.1:
        return "a small minority"
    if share < 0.35:
        return "a sizeable minority"
    if share < 0.65:
        return "about half"
    return "most"


def categorical(m: dict[str, Any]) -> str:
    """Located-category feedback: which drones, roughly when, and generic separation advice."""
    i, j = m["worst_pair"]
    lines = []
    if m["authored_min_sep_norm"] < 1.0:
        lines.append(
            f"Drones {i} and {j} get too close to each other around t≈{m['worst_time_s']:.1f}s. "
            "Separate them there by height (z), radius, or x/y center."
        )
    else:
        lines.append("No pair came too close in what you authored.")
    if m["steps_inside_envelope"]:
        lines.append("The filter could not keep every pair clear the whole way through.")
    if m["failed_solves"]:
        lines.append("The safety filter failed to find a solution at some points.")
    lines.append("The filter had to move the swarm away from the trajectory you authored.")
    return "\n".join(lines)


def absolute(m: dict[str, Any]) -> str:
    """Certified-magnitude feedback: the same events, reported in metres."""
    i, j = m["worst_pair"]
    lines = [
        f"Closest approach after filtering: {m['min_sep_m']:.2f} m between drones {i} and {j} at "
        f"t={m['worst_time_s']:.1f}s, against a required {m['required_sep_m']:.2f} m along that "
        f"approach direction.",
        f"What you authored came to {m['authored_min_sep_norm']:.2f} of the required separation "
        f"at its worst (1.00 is exactly clear).",
        f"Repairing it moved the swarm {m['deviation_mean_m']:.2f} m on average and "
        f"{m['deviation_max_m']:.2f} m at worst from the trajectory you authored; drone "
        f"{m['deviation_worst_drone']} was displaced most.",
        f"{m['steps_inside_envelope']} of {m['n_steps']} solver steps were inside the collision "
        f"envelope, and {m['failed_solves']} failed to solve.",
        f"Peak flown speed {m['max_speed_mps']:.2f} m/s against a {m['vel_max_mps']:.2f} m/s "
        f"limit; lowest altitude {m['min_z_m']:.2f} m.",
    ]
    return "\n".join(lines)


def relative(m: dict[str, Any]) -> str:
    """Comparative feedback: the same magnitudes as ratios, with no units and no raw figures."""
    i, j = m["worst_pair"]
    approach = _ratio_words(m["min_sep_norm"])
    authored = _ratio_words(m["authored_min_sep_norm"])
    next_worst = m["next_worst_norm"]
    severity = (
        _ratio_words(next_worst / m["min_sep_norm"]) if m["min_sep_norm"] > 0 else _RATIO_LARGE
    )
    spread = (
        _ratio_words(m["deviation_max_m"] / m["deviation_mean_m"])
        if m["deviation_mean_m"] > 0
        else "about"
    )
    lines = [
        f"At its worst moment the trajectory you authored brought drones {i} and {j} "
        f"{authored} the separation they need.",
        f"After the filter repaired it, that same pair ends up {approach} the separation they "
        f"need, and the next worst moment in the show is {severity} as clear as this one.",
        f"Repairing it pulled the swarm off the shape you authored. Drone "
        f"{m['deviation_worst_drone']} came off worst, {spread} as far off as the typical drone.",
        f"Of all the solver steps, {_fraction_words(m['steps_inside_envelope'], m['n_steps'])} "
        f"were still too close, and {_fraction_words(m['failed_solves'], m['n_steps'])} could not "
        f"be solved at all.",
        f"The fastest the swarm flew was {_ratio_words(m['max_speed_mps'] / m['vel_max_mps'])} "
        f"the speed limit.",
    ]
    return "\n".join(lines)


def render(arm: str, m: dict[str, Any]) -> str:
    """Render measurements under the named feedback arm.

    Raises:
        ValueError: If ``arm`` is not one of `ARMS`.
    """
    if arm not in ARMS:
        raise ValueError(f"Unknown feedback arm {arm!r}; expected one of {ARMS}")
    return {"categorical": categorical, "absolute": absolute, "relative": relative}[arm](m)
