"""Tests for time_to_finish_s on formation primitives and compact drone-id addressing."""

import numpy as np
import pytest

from swarm_gpt.core.motion_primitives import (
    _sanitize_drone_ids,
    expand_drone_id_spec,
    form_circle,
    form_cone,
    form_star,
)
from swarm_gpt.exception import LLMFormatError


def _limits() -> dict:
    return {"lower": np.array([-2.2, -2.7, 0.25]), "upper": np.array([2.2, 2.7, 1.7])}


def _swarm_10() -> np.ndarray:
    return np.array(
        [[x, y, 100] for x in (-200, -100, 0, 100, 200) for y in (-100, 100)], dtype=float
    )


def test_form_star_respects_time_to_finish():
    """Large time_to_finish_s → arrival should be ~5s into a 10s interval, not at physics min."""
    swarm = _swarm_10()
    limits = _limits()
    _, wps = form_star((100, 60, 80, 5.0), swarm, 0.0, 10.0, limits)
    times = sorted(wps.keys())
    assert 4.5 <= times[0] <= 5.5, f"expected arrival ~5s, got {times[0]}"


def test_form_star_clamps_below_physics_min():
    """Tiny time_to_finish_s → should clamp UP to the physics floor (>= T_MIN = 0.5s)."""
    swarm = _swarm_10()
    limits = _limits()
    _, wps = form_star((100, 60, 80, 0.05), swarm, 0.0, 10.0, limits)
    times = sorted(wps.keys())
    assert times[0] >= 0.5, f"expected clamp to physics floor, got {times[0]}"


def test_form_star_clamps_above_interval():
    """time_to_finish_s larger than the interval → arrival should be clamped to tend."""
    swarm = _swarm_10()
    limits = _limits()
    _, wps = form_star((100, 60, 80, 999.0), swarm, 0.0, 5.0, limits)
    times = sorted(wps.keys())
    # Arrival must not exceed tend
    assert times[0] <= 5.0, f"arrival {times[0]} exceeds tend=5.0"


def test_form_circle_respects_time_to_finish():
    """form_circle with large time_to_finish_s should arrive late in the interval."""
    swarm = _swarm_10()
    limits = _limits()
    drone_ids = list(range(1, 6))  # drones 1-5
    _, wps = form_circle((drone_ids, 100, 100, 8.0), swarm, 0.0, 10.0, limits)
    times = sorted(wps.keys())
    assert times[0] >= 7.0, f"expected late arrival, got {times[0]}"


def test_form_cone_respects_time_to_finish():
    """form_cone with time_to_finish_s close to tend → arrival near end of interval."""
    swarm = _swarm_10()
    limits = _limits()
    _, wps = form_cone((50, 60, 0, 8.0), swarm, 0.0, 10.0, limits)
    times = sorted(wps.keys())
    assert times[0] >= 7.0, f"expected late arrival, got {times[0]}"


def test_range_endpoints_are_inclusive_at_both_ends():
    """ "1-50" over 100 drones is exactly 0..49 -- it already contains drone 50, not 51."""
    ids = _sanitize_drone_ids("1-50", 100)
    assert len(ids) == 50
    assert ids[0] == 0  # drone 1
    assert ids[-1] == 49  # drone 50
    assert 50 not in ids  # drone 51 is NOT in "1-50"


def test_consecutive_blocks_partition_the_swarm():
    """The split the prompt mandates: "1-50" then "51-100" is disjoint and covers everything."""
    lower = _sanitize_drone_ids("1-50", 100)
    upper = _sanitize_drone_ids("51-100", 100)
    assert set(lower).isdisjoint(upper)
    assert sorted(lower + upper) == list(range(100))
    assert upper[0] == 50  # drone 51
    assert upper[-1] == 99  # drone 100


def test_single_id_needs_no_range():
    assert _sanitize_drone_ids("7", 10) == [6]


def test_mixed_singles_and_ranges():
    assert _sanitize_drone_ids("1-3,7,10-11", 20) == [0, 1, 2, 6, 9, 10]


def test_spec_preserves_the_order_it_is_written_in():
    """Order is not normalised: the explicit list form never sorted either."""
    assert _sanitize_drone_ids("5,1-2", 10) == [4, 0, 1]


def test_spec_matches_the_explicit_list_it_replaces():
    """The whole point: the compact form must select exactly what the old list selected."""
    assert _sanitize_drone_ids("1-5", 10) == _sanitize_drone_ids([1, 2, 3, 4, 5], 10)
    assert _sanitize_drone_ids("1-100", 100) == _sanitize_drone_ids(list(range(1, 101)), 100)


def test_whitespace_around_tokens_is_tolerated():
    """Hand-written presets and `lighting:` blocks are not rejected over a space."""
    assert _sanitize_drone_ids(" 1-3 , 7 ", 10) == [0, 1, 2, 6]


def test_plain_integer_lists_still_work():
    """Saved presets store the raw response text, so the old list form must keep loading."""
    assert _sanitize_drone_ids([1, 3, 5], 10) == [0, 2, 4]


def test_ellipsis_still_means_the_whole_swarm():
    assert _sanitize_drone_ids([...], 6) == [0, 1, 2, 3, 4, 5]


@pytest.mark.parametrize("spec", ["1-50,50-100", "1-5,3", "3,3", "1-5,5", "10-20,15-25", "5,1-10"])
def test_overlapping_selections_are_rejected(spec: str):
    """A drone named twice has no defined target -- reject rather than double-assign."""
    with pytest.raises(LLMFormatError, match="more than once"):
        _sanitize_drone_ids(spec, 100)


def test_overlap_error_names_the_shared_drone():
    """The message must be actionable for the reprompt loop."""
    with pytest.raises(LLMFormatError, match=r"\[50\]"):
        _sanitize_drone_ids("1-50,50-100", 100)


def test_reversed_range_is_rejected():
    with pytest.raises(LLMFormatError, match="runs backwards"):
        _sanitize_drone_ids("50-1", 100)


@pytest.mark.parametrize("spec", ["0", "0-5", "1-3,0"])
def test_drone_zero_is_rejected(spec: str):
    """Ids are 1-indexed; drone 0 would shift to -1 and silently select the last drone."""
    with pytest.raises(LLMFormatError, match="1-indexed"):
        _sanitize_drone_ids(spec, 10)


def test_lowest_id_is_in_bounds():
    """The other side of the drone-0 boundary: drone 1 is valid."""
    assert _sanitize_drone_ids("1", 10) == [0]


def test_highest_id_is_in_bounds():
    assert _sanitize_drone_ids("10", 10) == [9]


@pytest.mark.parametrize("spec", ["11", "1-11", "1-5,11", "101"])
def test_ids_above_the_swarm_are_rejected(spec: str):
    with pytest.raises(LLMFormatError, match=r"outside the 1\.\.10 swarm"):
        _sanitize_drone_ids(spec, 10)


@pytest.mark.parametrize("spec", ["", "1-", "-5", "a-b", "1--5", "1,,2", "1 - 5", "1;2", "1-2-3"])
def test_malformed_specs_are_rejected(spec: str):
    with pytest.raises(LLMFormatError, match="malformed"):
        _sanitize_drone_ids(spec, 10)


def test_non_string_non_list_is_rejected():
    with pytest.raises(LLMFormatError, match="range string"):
        _sanitize_drone_ids(5, 10)


def test_expand_returns_one_indexed_ids():
    """`expand_drone_id_spec` stays 1-indexed; only `_sanitize_drone_ids` shifts to 0."""
    assert expand_drone_id_spec("1-3,9") == [1, 2, 3, 9]


def test_form_circle_selects_the_same_drones_as_the_explicit_list():
    """End-to-end at the primitive: the range form must not change which drones move."""
    limits = _limits()
    by_range, wps_range = form_circle(("1-5", 100, 100, 1.0), _swarm_10(), 0.0, 10.0, limits)
    by_list, wps_list = form_circle(
        ([1, 2, 3, 4, 5], 100, 100, 1.0), _swarm_10(), 0.0, 10.0, limits
    )
    np.testing.assert_allclose(by_range, by_list)
    assert sorted(wps_range[min(wps_range)]) == [0, 1, 2, 3, 4]
    assert sorted(wps_range) == sorted(wps_list)


def test_form_circle_leaves_unselected_drones_where_they_were():
    swarm = _swarm_10()
    pos, _ = form_circle(("1-5", 100, 100, 1.0), swarm.copy(), 0.0, 10.0, _limits())
    np.testing.assert_allclose(pos[5:], swarm[5:])
