import json
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from swarm_gpt.synth.shapes import SHAPES, check_shape, describe_shape

T_STEPS = 60
N_LEVELS = 5
RADIUS_CM = 120.0


def _helix(
    *,
    twist_turns: float = 0.5,
    pitch_cm: float = 25.0,
    offset_deg: float = 180.0,
    n_levels: int = N_LEVELS,
) -> tuple[NDArray, NDArray]:
    """A double helix: one drone per strand per level, ``offset_deg`` apart, twisting as it climbs."""
    pos = np.zeros((n_levels * 2, T_STEPS, 3))
    for level in range(n_levels):
        frac = level / (n_levels - 1)
        for strand in (0, 1):
            i = level * 2 + strand
            angle = 2 * np.pi * twist_turns * frac + np.deg2rad(offset_deg) * strand
            pos[i, :, 0] = RADIUS_CM * np.cos(angle)
            pos[i, :, 1] = RADIUS_CM * np.sin(angle)
            pos[i, :, 2] = 60.0 + pitch_cm * level
    return pos, np.linspace(0.0, 12.0, T_STEPS)


def _flat_rings() -> tuple[NDArray, NDArray]:
    """Two rings at two altitudes: the shape the model kept producing and self-certifying."""
    pos = np.zeros((N_LEVELS * 2, T_STEPS, 3))
    for i in range(N_LEVELS * 2):
        ring, j = i // N_LEVELS, i % N_LEVELS
        angle = 2 * np.pi * j / N_LEVELS
        pos[i, :, 0] = RADIUS_CM * np.cos(angle)
        pos[i, :, 1] = RADIUS_CM * np.sin(angle)
        pos[i, :, 2] = 68.0 if ring == 0 else 162.0
    return pos, np.linspace(0.0, 12.0, T_STEPS)


def _named(checks: list[dict[str, Any]], name: str) -> dict[str, Any]:
    return next(c for c in checks if c["name"] == name)


def test_double_helix_is_registered():
    assert "double_helix" in SHAPES


def test_unknown_shape_is_rejected():
    with pytest.raises(KeyError):
        check_shape("banana", *_helix())


def test_a_true_double_helix_passes_every_check():
    checks = check_shape("double_helix", *_helix())
    assert all(c["ok"] for c in checks), [c for c in checks if not c["ok"]]


def test_two_flat_rings_are_rejected():
    # The twist reading is not asserted on: once the heights do not pair, it means nothing.
    checks = check_shape("double_helix", *_flat_rings())
    assert _named(checks, "paired_heights")["ok"] is False
    assert _named(checks, "strands_opposed")["ok"] is False


def test_strands_must_be_opposed_not_merely_two():
    checks = check_shape("double_helix", *_helix(offset_deg=25.0))
    assert _named(checks, "strands_opposed")["ok"] is False


def test_a_ladder_without_twist_is_rejected():
    checks = check_shape("double_helix", *_helix(twist_turns=0.0))
    assert _named(checks, "strands_climb")["ok"] is True
    assert _named(checks, "strands_opposed")["ok"] is True
    assert _named(checks, "twists_with_height")["ok"] is False


def test_a_shallow_climb_is_rejected_as_a_ring():
    checks = check_shape("double_helix", *_helix(pitch_cm=1.0))
    assert _named(checks, "strands_climb")["ok"] is False


def test_an_odd_swarm_cannot_pair_up():
    pos, time = _helix()
    checks = check_shape("double_helix", pos[:-1], time)
    assert _named(checks, "paired_heights")["ok"] is False


def test_evenly_spaced_single_file_is_not_paired():
    # One drone per height, evenly spaced: a single strand, not two opposed ones.
    pos, time = _helix()
    pos[:, :, 2] = 60.0 + 25.0 * np.arange(pos.shape[0])[:, None]
    assert _named(check_shape("double_helix", pos, time), "paired_heights")["ok"] is False


def test_twist_is_read_independently_of_which_drone_of_a_pair_is_picked():
    pos, time = _helix()
    swapped = pos.copy()
    swapped[0::2], swapped[1::2] = pos[1::2], pos[0::2]
    assert check_shape("double_helix", pos, time) == check_shape("double_helix", swapped, time)


def test_checks_are_json_serialisable_triples():
    checks = check_shape("double_helix", *_helix())
    assert json.loads(json.dumps(checks)) == checks
    assert all(set(c) == {"name", "ok", "detail"} for c in checks)


def test_description_states_same_handedness_and_gives_no_code():
    text = describe_shape("double_helix")
    assert "strand" in text.lower()
    assert "same way" in text.lower()
    assert "def " not in text
