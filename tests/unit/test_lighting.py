"""Unit tests for the lighting engine: selectors, waveforms, spreads, layers and the timeline."""

import dataclasses
import logging
from pathlib import Path

import numpy as np
import pytest
from conftest import with_ulp_noise

from swarm_gpt.core.lighting import (
    BrightnessLayer,
    ColourLayer,
    LightingConfig,
    LightingTimeline,
    Look,
    hue_to_wrgb,
    load_lighting_config,
    select,
    spread_offsets,
    waveform,
)


@pytest.fixture
def lighting_log(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> pytest.LogCaptureFixture:
    """`caplog`, wired so it actually sees this module's records.

    The ROS `launch` pytest plugin calls `logging.setLoggerClass` with ``propagate = False``, so
    every logger built after it bypasses the root logger `caplog` listens on.
    """
    logger = logging.getLogger("swarm_gpt.core.lighting")
    monkeypatch.setattr(logger, "propagate", True)
    caplog.set_level(logging.WARNING, logger=logger.name)
    return caplog


# The names the prompt offers the LLM as the `color` enum.
PALETTE_NAMES = (
    "red",
    "orange",
    "amber",
    "yellow",
    "green",
    "teal",
    "cyan",
    "azure",
    "blue",
    "indigo",
    "magenta",
    "pink",
    "white",
)

# Six drones spread along +x, no two sharing an x and none sitting on the x centroid (0.5), so the
# left/right split is unambiguous and flipping `stage_axis` swaps the two halves exactly.
POSITIONS_6 = np.array(
    [
        [-2.0, 0.0, 1.0],
        [-1.0, 1.0, 1.2],
        [0.0, -1.0, 1.4],
        [1.0, 0.5, 1.6],
        [2.0, -0.5, 1.8],
        [3.0, 1.0, 2.0],
    ]
)

# Six drones whose height split crosses their stage split: z climbs with the index while x and y
# both alternate about their means, so `upper` cannot coincide with `right` on either stage axis.
# `POSITIONS_6` rises in x and z together, which makes `upper` there `right` by coincidence.
POSITIONS_CROSSED_6 = np.array(
    [
        [2.0, 0.0, 0.5],
        [-2.0, 1.0, 0.6],
        [2.0, -1.0, 0.7],
        [-2.0, 0.5, 2.3],
        [2.0, -0.5, 2.4],
        [-2.0, 1.0, 2.5],
    ]
)

# Two drones parked well above four, the unequal stack the mean split exists for: the z mean is
# 1.33, so `upper` takes the high two, where a rank split would take three.
POSITIONS_TOP_HEAVY_6 = np.array(
    [
        [-1.0, 0.0, 0.5],
        [0.0, 1.0, 0.5],
        [1.0, -1.0, 0.5],
        [2.0, 0.5, 0.5],
        [-0.5, -0.5, 3.0],
        [0.5, 1.0, 3.0],
    ]
)


def _staged(axis: str) -> LightingConfig:
    """The shipped config with `stage_axis` pinned, for tests whose fixture picks the axis.

    Which axis faces the audience is a property of the room, checked once in
    `test_shipped_stage_axis_matches_the_lab_geometry`. Tests of the split mechanism state the axis
    their fixture is laid out along, so re-rigging the room cannot fail them.
    """
    return dataclasses.replace(load_lighting_config(), stage_axis=axis)


def test_shipped_stage_axis_matches_the_lab_geometry():
    """The audience views from +x with the show on x = 0, so their right hand points along +y.

    Facing -x with z up puts right at ``(-1,0,0) x (0,0,1) = (0,1,0)``. Getting this wrong is not a
    crash: `left`/`right` silently split near/far from the audience, which reads as no split at all.
    """
    assert load_lighting_config().stage_axis == "+y"


def test_palette_entries_are_wrgb_vectors_in_range():
    cfg = load_lighting_config()
    assert len(cfg.palette) >= 12
    for name, entry in cfg.palette.items():
        assert entry.shape == (4,), f"{name} is not a WRGB 4-vector"
        assert np.all(entry >= 0.0) and np.all(entry <= 255.0), f"{name} outside [0, 255]"


def test_palette_holds_every_prompt_colour():
    cfg = load_lighting_config()
    missing = [name for name in PALETTE_NAMES if name not in cfg.palette]
    assert not missing, f"palette is missing prompt colours: {missing}"


def test_calibration_constants_present_and_typed():
    cfg = load_lighting_config()
    assert isinstance(cfg.gamma, float) and cfg.gamma > 0.0
    assert isinstance(cfg.b_min, float) and 0.0 <= cfg.b_min < 1.0
    assert isinstance(cfg.hue_steps, int) and cfg.hue_steps > 0
    # Quantization on both axes, not just hue: the unquantized brightness waveforms were the
    # expensive ones to compile.
    assert isinstance(cfg.brightness_steps, int) and cfg.brightness_steps > 0
    assert cfg.channel_gain.shape == (4,)
    assert np.all(cfg.channel_gain > 0.0)
    assert cfg.stage_axis in ("+x", "-x", "+y", "-y")
    # The cue rate is data, not a constant duplicated next to `DroneSwarm.col_freq`: it sets both
    # the compile grid and the Nyquist floor the primitives clamp against.
    assert isinstance(cfg.col_freq, float) and cfg.col_freq > 0.0


def test_blue_is_dimmed_relative_to_constant_channel_sum():
    """The backend.py:314 blue dim survives as a per-colour value, not a global multiply."""
    cfg = load_lighting_config()
    assert cfg.palette["blue"][3] == pytest.approx(255.0 * 0.8)
    assert cfg.palette["red"].sum() == pytest.approx(255.0)


def test_every_palette_entry_carries_the_same_gain_corrected_output():
    """One invariant over the whole palette, which is what says the thirteen are calibrated alike.

    The constant quantity is the *gain-corrected* channel sum, the one `hue_to_wrgb` holds across
    the wheel. Spot-checking two entries lets a miscalibrated `indigo` through.
    """
    cfg = load_lighting_config()
    for name, entry in cfg.palette.items():
        assert (entry / cfg.channel_gain).sum() == pytest.approx(255.0, abs=0.01), name


# Every key `load_lighting_config` indexes directly, and the fragment of TOML that supplies it. The
# palette is a table, so it is rendered last whatever the omitted key is.
_CONFIG_LINES = {
    "gamma": "gamma = 2.2",
    "b_min": "b_min = 0.02",
    "col_freq": "col_freq = 10.0",
    "hue_steps": "hue_steps = 24",
    "brightness_steps": "brightness_steps = 16",
    "channel_gain": "channel_gain = [1.0, 1.0, 1.0, 0.8]",
    "stage_axis": 'stage_axis = "+y"',
    "palette": "[palette]\nred = [0.0, 255.0, 0.0, 0.0]",
}


def _config_toml(omit: str = "") -> str:
    """The shipped config's shape, with one required key left out."""
    return "\n".join(line for key, line in _CONFIG_LINES.items() if key != omit) + "\n"


def test_the_synthetic_config_fixture_loads_when_it_is_complete(tmp_path: Path):
    """The positive control for the parametrization below, which would otherwise pass vacuously."""
    path = tmp_path / "lighting.toml"
    path.write_text(_config_toml())
    assert load_lighting_config(path).stage_axis == "+y"


@pytest.mark.parametrize("key", sorted(_CONFIG_LINES))
def test_every_required_key_is_indexed_directly(key: str, tmp_path: Path):
    """The loader indexes required keys directly, so a truncated file fails loudly (CLAUDE.md 6.2).

    One case per key: a fixture missing two is satisfied by whichever the loader reads first, so
    seven of the eight could become `.get(key, default)` and still fly on a silent default.
    """
    path = tmp_path / "lighting.toml"
    path.write_text(_config_toml(omit=key))
    with pytest.raises(KeyError, match=key):
        load_lighting_config(path)


def test_select_all_covers_every_drone():
    cfg = load_lighting_config()
    assert np.all(select(("all", ()), 6, POSITIONS_6, cfg))


def test_select_ids_is_one_indexed():
    """`ids` is LLM-facing and 1-indexed like move_z/form_circle; the mask is 0-indexed."""
    cfg = load_lighting_config()
    mask = select(("ids", (1, 3, 5)), 6, POSITIONS_6, cfg)
    assert list(mask) == [True, False, True, False, True, False]


@pytest.mark.parametrize("sel", [("ids", (7,)), ("ids", (0,)), ("ids", (1, -2))])
def test_select_ids_outside_the_swarm_raise(sel: tuple):
    """Both ends, not just the top one — the schema is not the only path into `select`.

    An id of 0 shifts to -1 and quietly selects the *last* drone. The schema's ``minimum: 1``
    blocks that, but a preset or hand-written `lighting:` block reaches `build_look` without one.
    """
    cfg = load_lighting_config()
    with pytest.raises(IndexError, match="1..6"):
        select(sel, 6, POSITIONS_6, cfg)


@pytest.mark.parametrize("count", [0, -1, 7])
def test_select_first_outside_the_swarm_raises(count: int):
    """`first(k)` clamped silently against the slice: `first(99)` selected the whole swarm."""
    cfg = load_lighting_config()
    with pytest.raises(IndexError, match="1..6"):
        select(("first", (count,)), 6, POSITIONS_6, cfg)


def test_select_even_and_odd_split_on_the_parity_of_the_1_indexed_id():
    """`even` must mean the drones the LLM calls 2, 4, 6, not the array slots at 0, 2, 4.

    Every other way of naming a drone -- `ids`, `first`, the motion primitives -- is 1-indexed, so
    an array-parity `even` is the exact complement of what an author writing `ids([2, 4, 6])` gets.
    """
    cfg = load_lighting_config()
    even = select(("even", ()), 6, POSITIONS_6, cfg)
    odd = select(("odd", ()), 6, POSITIONS_6, cfg)
    assert list(even) == [False, True, False, True, False, True]
    assert list(odd) == [True, False, True, False, True, False]
    assert np.all(even ^ odd)
    assert list(even) == list(select(("ids", (2, 4, 6)), 6, POSITIONS_6, cfg))


def test_select_first_n_takes_the_lowest_indices():
    cfg = load_lighting_config()
    mask = select(("first", (4,)), 6, POSITIONS_6, cfg)
    assert list(mask) == [True, True, True, True, False, False]


def test_select_left_and_right_partition_about_the_centroid():
    """x centroid of the fixture is 0.5, so drones 3-5 are stage right when the axis is "+x"."""
    cfg = _staged("+x")
    right = select(("right", ()), 6, POSITIONS_6, cfg)
    left = select(("left", ()), 6, POSITIONS_6, cfg)
    assert list(right) == [False, False, False, True, True, True]
    assert list(left) == [True, True, True, False, False, False]
    assert np.all(left ^ right), "left and right must be exact complements"


def test_flipping_stage_axis_swaps_left_and_right():
    """Reorienting the show is a one-line data change, never a code change (CLAUDE.md 6.6)."""
    cfg = _staged("+x")
    flipped = _staged("-x")
    assert list(select(("right", ()), 6, POSITIONS_6, flipped)) == list(
        select(("left", ()), 6, POSITIONS_6, cfg)
    )
    assert list(select(("left", ()), 6, POSITIONS_6, flipped)) == list(
        select(("right", ()), 6, POSITIONS_6, cfg)
    )


def test_select_upper_and_lower_partition_about_the_mean_height():
    """The fixture's z mean is 1.5, and its stage split crosses its height split.

    `POSITIONS_6` rises monotonically in both x and z, so `upper` there is `right` by coincidence
    and a test on it passes just as well when `upper` reads the wrong column.
    """
    cfg = load_lighting_config()
    upper = select(("upper", ()), 6, POSITIONS_CROSSED_6, cfg)
    lower = select(("lower", ()), 6, POSITIONS_CROSSED_6, cfg)
    assert list(upper) == [False, False, False, True, True, True]
    assert list(lower) == [True, True, True, False, False, False]
    assert np.all(upper ^ lower), "upper and lower must be exact complements"
    assert list(select(("right", ()), 6, POSITIONS_CROSSED_6, cfg)) != list(upper), (
        "the fixture must separate the height split from the stage split"
    )


def test_upper_splits_unequal_stacks_along_the_gap_between_them():
    """Two formations at different heights part where they actually part, whatever their sizes.

    This is the mean rule's reason for being: a rank split cuts at the halfway drone, so a small
    high formation over a large low one drags the low one's top drones up with it.
    """
    cfg = load_lighting_config()
    upper = select(("upper", ()), 6, POSITIONS_TOP_HEAVY_6, cfg)
    assert list(upper) == [False, False, False, False, True, True]
    assert upper.sum() == 2, "a rank split would take three, one of them from the lower stack"


def test_upper_and_lower_are_independent_of_stage_axis():
    """Height needs no calibration, so unlike left/right it cannot be flipped by the config."""
    cfg = _staged("+x")
    flipped = _staged("-y")
    assert list(select(("upper", ()), 6, POSITIONS_CROSSED_6, flipped)) == list(
        select(("upper", ()), 6, POSITIONS_CROSSED_6, cfg)
    )


def test_select_ids_naming_no_drones_raises():
    """`ids([])` yielded an all-False mask and a layer that did nothing at all.

    `first(0)` -- the other way to ask for nothing -- already raises, and the two must not
    disagree. `IndexError` is one of the three `_build_look` turns into an `LLMFormatError`.
    """
    cfg = load_lighting_config()
    with pytest.raises(IndexError, match="no drones"):
        select(("ids", ()), 6, POSITIONS_6, cfg)


@pytest.mark.parametrize(
    "sel", [("all", (1, 2, 3)), ("even", (2,)), ("left", (1,)), ("first", ()), ("first", (2, 3))]
)
def test_select_with_the_wrong_argument_count_raises(sel: tuple):
    """Arity was never checked: `all` dropped extra arguments silently and `first` had none to drop.

    A hand-written `("all", (1, 2, 3))` reads as "drones 1-3" and selected all of them, while
    `("first", ())` raised a bare tuple-index `IndexError` naming neither selector nor problem.
    """
    cfg = load_lighting_config()
    with pytest.raises(IndexError, match="takes"):
        select(sel, 6, POSITIONS_6, cfg)


def test_unknown_selector_raises():
    cfg = load_lighting_config()
    with pytest.raises(KeyError):
        select(("middle", ()), 6, POSITIONS_6, cfg)


QUARTER_PHASES = np.array([0.0, 0.25, 0.5, 0.75])


def test_waveform_sine_at_quarter_phases():
    """0.5 * (1 + cos(2*pi*phi))."""
    assert waveform("sine", QUARTER_PHASES) == pytest.approx([1.0, 0.5, 0.0, 0.5])


def test_waveform_square_at_quarter_phases():
    """1 while frac(phi) < duty, else 0."""
    assert waveform("square", QUARTER_PHASES) == pytest.approx([1.0, 1.0, 0.0, 0.0])


def test_waveform_ramp_at_quarter_phases():
    """1 - frac(phi): full on the beat, decaying out."""
    assert waveform("ramp", QUARTER_PHASES) == pytest.approx([1.0, 0.75, 0.5, 0.25])


def test_every_waveform_peaks_on_the_beat():
    """Effects land *on* the beat, not between beats -- including at wrapped phases."""
    on_beat = np.array([-2.0, -1.0, 0.0, 1.0, 3.0])
    for kind in ("sine", "square", "ramp"):
        assert waveform(kind, on_beat) == pytest.approx(np.ones(5)), kind


def test_waveform_duty_defaults_to_half():
    assert waveform("square", np.array([0.49])) == pytest.approx([1.0])
    assert waveform("square", np.array([0.51])) == pytest.approx([0.0])


def test_waveform_duty_is_clamped_to_the_open_unit_interval():
    """duty <= 0 still lights the beat instant; duty > 1 is solid on."""
    pinched = waveform("square", np.array([0.0, 0.01, 0.5]), duty=0.0)
    assert pinched == pytest.approx([1.0, 0.0, 0.0])
    solid = waveform("square", np.array([0.0, 0.5, 0.99]), duty=5.0)
    assert solid == pytest.approx([1.0, 1.0, 1.0])


def test_waveform_outputs_stay_within_the_unit_interval():
    phases = np.linspace(-3.0, 3.0, 401)
    for kind in ("sine", "square", "ramp"):
        out = waveform(kind, phases)
        assert np.all(out >= 0.0) and np.all(out <= 1.0), kind


def test_unknown_waveform_raises():
    with pytest.raises(KeyError):
        waveform("sawtooth", QUARTER_PHASES)


# Ten drones evenly spaced along +x, so the x centroid is 4.5 and the spatial spreads have
# hand-computable normalizations.
POSITIONS_10 = np.stack([np.arange(10.0), np.zeros(10), np.ones(10)], axis=1)
ALL_10 = np.ones(10, dtype=bool)


def test_spread_none_is_all_zeros():
    cfg = load_lighting_config()
    assert spread_offsets("none", ALL_10, POSITIONS_10, 1, cfg) == pytest.approx(np.zeros(10))


def test_spread_index_ranks_within_the_selected_subset():
    """A chase over ids([2, 5, 9]) runs across three drones, not across gaps in the swarm."""
    cfg = load_lighting_config()
    mask = select(("ids", (2, 5, 9)), 10, POSITIONS_10, cfg)
    offsets = spread_offsets("index", mask, POSITIONS_10, 1, cfg)
    assert offsets[[1, 4, 8]] == pytest.approx([0.0, 1 / 3, 2 / 3])
    # The full-swarm ranking would have given these rows 0.1, 0.4 and 0.8 instead.
    assert offsets[4] != pytest.approx(0.4)


def test_spread_index_over_the_whole_swarm_is_rank_over_n():
    cfg = load_lighting_config()
    offsets = spread_offsets("index", ALL_10, POSITIONS_10, 1, cfg)
    assert offsets == pytest.approx(np.arange(10) / 10)


#
# Every formation primitive routes through `_assign_positions` (motion_primitives.py:590), a
# Hungarian assignment that returns whichever drone -> slot permutation is cheapest to fly. Drone 6
# is therefore as likely to sit beside drone 1 as beside drone 5, and the fixtures below all model
# that: a formation laid out slot by slot, plus a permutation saying which drone flies which slot.
# A fixture in id order could not tell `neighbour` from `index` at all.

_RING_ANGLES = 2.0 * np.pi * np.arange(10) / 10
# Ten ring slots in ring order, radius 2 at a constant height.
RING_SLOTS = np.stack(
    [2.0 * np.cos(_RING_ANGLES), 2.0 * np.sin(_RING_ANGLES), np.full(10, 1.5)], axis=1
)
# Drone i flies ring slot SCRAMBLE[i]. Deliberately not a rotation: no cyclic shift of the ids
# reproduces it, so `index` cannot pass the ring assertions by luck.
SCRAMBLE = np.array([3, 7, 0, 9, 4, 1, 8, 2, 6, 5])
RING_10 = RING_SLOTS[SCRAMBLE]

# Eight unevenly spaced slots along +x -- uneven because even spacing makes a nearest-neighbour
# walk trivially correct in both directions -- and again a scrambled assignment.
LINE_SLOTS = np.stack(
    [np.array([0.0, 0.7, 1.1, 2.4, 2.9, 4.2, 5.0, 6.3]), np.zeros(8), np.full(8, 1.2)], axis=1
)
LINE_SCRAMBLE = np.array([5, 2, 7, 0, 4, 1, 6, 3])
LINE_8 = LINE_SLOTS[LINE_SCRAMBLE]

# A 3x3 grid at 0.5 m pitch, walked as a snake, with the ids scrambled across it.
GRID_SLOTS = np.array([[0.5 * i, 0.5 * j, 1.0] for i in range(3) for j in range(3)])
GRID_SCRAMBLE = np.array([4, 0, 8, 2, 6, 1, 7, 3, 5])
GRID_9 = GRID_SLOTS[GRID_SCRAMBLE]


def _ranks(offsets: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Recover integer walk ranks from per-drone offsets, which are ``rank / n_sel``."""
    return np.round(offsets[mask] * int(mask.sum())).astype(int)


def test_spread_neighbour_recovers_ring_order_from_a_scrambled_id_assignment():
    """The bug `neighbour` exists for, on the formation that shows it worst.

    Under an arbitrary id rotation an `index` chase jumps clean across the stage. So the assertion
    is on *ring* order: consecutive ranks land on adjacent slots, all the way round, one direction.
    """
    cfg = load_lighting_config()
    offsets = spread_offsets("neighbour", ALL_10, RING_10, 1, cfg)
    assert np.all(offsets >= 0.0) and np.all(offsets < 1.0)
    ranks = _ranks(offsets, ALL_10)
    assert sorted(ranks) == list(range(10)), "every drone takes a distinct rank"
    steps = set(np.diff(SCRAMBLE[np.argsort(ranks)]) % 10)
    assert steps in ({1}, {9}), f"the walk must run slot by slot around the ring, got {steps}"
    # And the fixture really is scrambled, so id order does not run round the ring by accident.
    assert set(np.diff(SCRAMBLE) % 10) not in ({1}, {9})


def test_spread_neighbour_recovers_line_order_from_a_scrambled_id_assignment():
    """On a line the walk is unambiguous, so the ranks are pinned exactly rather than up to sign."""
    cfg = load_lighting_config()
    mask = np.ones(8, dtype=bool)
    offsets = spread_offsets("neighbour", mask, LINE_8, 1, cfg)
    assert list(_ranks(offsets, mask)) == list(LINE_SCRAMBLE), "rank is the slot along the line"
    assert offsets != pytest.approx(spread_offsets("index", mask, LINE_8, 1, cfg))


def test_spread_neighbour_on_a_grid_keeps_adjacent_ranks_spatially_adjacent():
    """The requirement in general form: neighbouring ranks are neighbouring drones, on any shape."""
    cfg = load_lighting_config()
    mask = np.ones(9, dtype=bool)
    ranks = _ranks(spread_offsets("neighbour", mask, GRID_9, 1, cfg), mask)
    walk = GRID_9[np.argsort(ranks)]
    hops = np.linalg.norm(np.diff(walk, axis=0), axis=1)
    assert hops == pytest.approx(np.full(8, 0.5)), "a 3x3 grid snakes with no long jump"


def test_spread_neighbour_ranks_the_same_positions_however_the_ids_are_permuted():
    """Determinism: the ranking is a function of the positions alone, never of the id assignment.

    A ring is hardest: the start's two neighbours are equidistant, so breaking that tie by array
    order would let the permutation choose which way round the walk runs.
    """
    cfg = load_lighting_config()
    slot_ranks = _ranks(spread_offsets("neighbour", ALL_10, RING_SLOTS, 1, cfg), ALL_10)
    for perm in (SCRAMBLE, np.arange(10)[::-1], np.roll(np.arange(10), 4)):
        ranks = _ranks(spread_offsets("neighbour", ALL_10, RING_SLOTS[perm], 1, cfg), ALL_10)
        assert list(ranks) == list(slot_ranks[perm]), perm


def test_spread_neighbour_ranks_within_the_selected_subset():
    """A chase over three drones walks those three, not the gaps in the full swarm."""
    cfg = load_lighting_config()
    # `ids(1, 2, 6)`, unevenly spread through the swarm.
    mask = select(("ids", (1, 2, 6)), 8, LINE_8, cfg)
    offsets = spread_offsets("neighbour", mask, LINE_8, 1, cfg)
    # Those three sit at line slots 5, 2 and 1, so along the line the order is drone 6, 2, 1.
    assert offsets[[0, 1, 5]] == pytest.approx([2 / 3, 1 / 3, 0.0])
    # Ranking over the whole swarm instead would have given these rows 5/8, 2/8 and 1/8.
    assert offsets[0] != pytest.approx(5 / 8)
    assert np.all(offsets[[2, 3, 4, 6, 7]] == 0.0), "unselected rows stay 0, as for every spread"


def test_spread_neighbour_takes_the_same_group_quantization_as_index():
    """`group_size` buckets the walk ranks exactly as it buckets the id ranks."""
    cfg = load_lighting_config()
    ranks = _ranks(spread_offsets("neighbour", ALL_10, RING_10, 1, cfg), ALL_10)
    by_three = spread_offsets("neighbour", ALL_10, RING_10, 3, cfg)
    assert by_three == pytest.approx((ranks // 3) / 4)
    assert len(set(by_three)) == 4  # ceil(10 / 3)


def test_spread_neighbour_of_a_single_drone_is_a_zero_offset():
    """A one-drone walk has no step to take; `n_sel = 1` must not divide by zero or wrap to 1.0."""
    cfg = load_lighting_config()
    mask = select(("ids", (4,)), 10, RING_10, cfg)
    assert spread_offsets("neighbour", mask, RING_10, 1, cfg) == pytest.approx(np.zeros(10))


def test_spread_alternate_parity_gives_two_offsets_half_a_turn_apart():
    cfg = load_lighting_config()
    offsets = spread_offsets("alternate_parity", ALL_10, POSITIONS_10, 1, cfg)
    assert sorted(set(offsets)) == pytest.approx([0.0, 0.5])
    assert offsets[0::2] == pytest.approx(np.zeros(5))
    assert offsets[1::2] == pytest.approx(np.full(5, 0.5))


def test_spread_alternate_side_gives_two_offsets_half_a_turn_apart():
    """Stage right (x > 4.5) is half a period behind stage left."""
    cfg = _staged("+x")
    offsets = spread_offsets("alternate_side", ALL_10, POSITIONS_10, 1, cfg)
    assert sorted(set(offsets)) == pytest.approx([0.0, 0.5])
    assert offsets[:5] == pytest.approx(np.zeros(5))
    assert offsets[5:] == pytest.approx(np.full(5, 0.5))


def test_spatial_spreads_normalize_into_the_half_open_unit_interval():
    """Offsets must stay in [0, 1) so no two ends of a sweep sit at the same phase."""
    cfg = load_lighting_config()
    for kind in ("x", "y", "z", "radius"):
        offsets = spread_offsets(kind, ALL_10, POSITIONS_10, 1, cfg)
        assert np.all(offsets >= 0.0) and np.all(offsets < 1.0), kind


def test_spread_x_matches_index_for_evenly_spaced_drones():
    """The spatial normalization uses the same (n_sel - 1) / n_sel convention as `index`."""
    cfg = load_lighting_config()
    assert spread_offsets("x", ALL_10, POSITIONS_10, 1, cfg) == pytest.approx(
        spread_offsets("index", ALL_10, POSITIONS_10, 1, cfg)
    )


def test_spread_radius_ripples_out_from_the_centre():
    """Distances from the x centroid (4.5) are 0.5 .. 4.5, normalized over that span."""
    cfg = load_lighting_config()
    offsets = spread_offsets("radius", ALL_10, POSITIONS_10, 1, cfg)
    expected = [0.9, 0.675, 0.45, 0.225, 0.0, 0.0, 0.225, 0.45, 0.675, 0.9]
    assert offsets == pytest.approx(expected)


def test_spread_radius_on_a_cos_sin_ring_collapses_rather_than_amplifying_float_noise():
    """A ring's radii are equal only to ~1e-16, so an exact-zero span test is the wrong test.

    Against exact zero a `cos`/`sin` ring takes the non-degenerate branch and divides by that noise,
    giving a random per-drone phase. Equal literals cannot catch it, so the fixture is a real ring.
    """
    cfg = load_lighting_config()
    ring = with_ulp_noise(RING_10)
    radii = np.linalg.norm(ring - ring.mean(axis=0), axis=1)
    span = float(radii.max() - radii.min())
    assert 0.0 < span < 1e-12, (
        f"the fixture must be degenerate only to within float noise, got a span of {span}; "
        "an exactly-equal ring passes this test without the tolerance and pins nothing"
    )
    assert spread_offsets("radius", ALL_10, ring, 1, cfg) == pytest.approx(np.zeros(10))


def test_spread_group_size_quantizes_into_ceil_buckets():
    """group_size = k advances the pattern group-by-group over ceil(n_sel / k) buckets."""
    cfg = load_lighting_config()
    by_three = spread_offsets("index", ALL_10, POSITIONS_10, 3, cfg)
    assert by_three == pytest.approx([0.0, 0.0, 0.0, 0.25, 0.25, 0.25, 0.5, 0.5, 0.5, 0.75])
    assert len(set(by_three)) == 4  # ceil(10 / 3)
    by_four = spread_offsets("index", ALL_10, POSITIONS_10, 4, cfg)
    assert len(set(by_four)) == 3  # ceil(10 / 4)
    per_drone = spread_offsets("index", ALL_10, POSITIONS_10, 1, cfg)
    assert len(set(per_drone)) == 10


@pytest.mark.parametrize(
    "kind", ["neighbour", "index", "alternate_parity", "alternate_side", "radius", "x", "y", "z"]
)
def test_spread_offsets_of_an_empty_selection_are_all_zero(kind: str):
    """An empty selection is reachable: `right` on a formation with no extent along the stage axis.

    The layers are then no-ops, but the spatial spreads reduce over the selected rows --
    `values.max()` raises on a zero-size array -- so the offsets are short-circuited first.
    """
    cfg = load_lighting_config()
    empty = np.zeros(10, dtype=bool)
    assert spread_offsets(kind, empty, POSITIONS_10, 1, cfg) == pytest.approx(np.zeros(10))


@pytest.mark.parametrize("kind", ["neighbour", "index", "none", "radius", "x", "alternate_side"])
def test_spread_group_size_below_one_raises_whatever_the_spread(kind: str):
    """The check used to sit inside the ranked-spread branch, so only ranked spreads saw it.

    `group_size=0` with `spread="x"` then succeeded silently while `spread="neighbour"` raised, and
    the catalogue lists `group_size` as a plain parameter with no spread restriction.
    """
    cfg = load_lighting_config()
    with pytest.raises(ValueError, match="group_size"):
        spread_offsets(kind, ALL_10, POSITIONS_10, 0, cfg)


@pytest.mark.parametrize("kind", ["none", "radius", "x", "y", "z", "alternate_parity"])
def test_spread_group_size_above_one_is_rejected_by_the_spreads_that_cannot_honour_it(kind: str):
    """Bucketing is defined over `rank_i`, which only `neighbour` and `index` produce.

    A spatial spread carries a normalized coordinate, so an evenly bucketed `x` is a different
    effect from a proportional one. Rejected rather than accepted-and-ignored, so it reprompts.
    """
    cfg = load_lighting_config()
    with pytest.raises(ValueError, match="group_size"):
        spread_offsets(kind, ALL_10, POSITIONS_10, 2, cfg)


@pytest.mark.parametrize(
    "kind", ["none", "neighbour", "index", "alternate_parity", "alternate_side", "radius", "x"]
)
def test_the_default_group_size_stays_legal_on_every_spread(kind: str):
    """`group_size = 1` is per-drone and means "no bucketing", so no spread can object to it."""
    cfg = load_lighting_config()
    assert spread_offsets(kind, ALL_10, POSITIONS_10, 1, cfg).shape == (10,)


def test_a_stage_axis_with_no_extent_warns_that_the_split_collapsed(
    lighting_log: pytest.LogCaptureFixture,
):
    """`left`/`right` on a planar formation is a silent no-op, and must not stay silent.

    A formation with no extent along the stage axis goes entirely left, so `light_color(right,...)`
    paints nobody. The show still runs, hence a warning -- but it has to name the axis.
    """
    cfg = _staged("+x")
    flat = np.stack([np.zeros(6), np.arange(6.0), np.ones(6)], axis=1)
    assert not select(("right", ()), 6, flat, cfg).any()
    assert select(("left", ()), 6, flat, cfg).all()
    assert lighting_log.records, "a silent no-op is the failure mode being fixed"
    assert lighting_log.records[0].levelno == logging.WARNING
    assert cfg.stage_axis in lighting_log.records[0].getMessage(), "name the axis"


def test_a_stage_axis_degenerate_only_to_float_noise_collapses_left_rather_than_splitting(
    lighting_log: pytest.LogCaptureFixture,
):
    """The stage axis collapses on edge-on formations, whose coordinate is equal only to ~1e-16.

    A heading perpendicular to the stage axis has a `cos` of 6.1e-17, not 0, so `coord >
    coord.mean()` deals the swarm into arbitrary halves on noise and the warning stays silent.
    """
    cfg = _staged("+x")
    angles = np.linspace(0.0, 2.0 * np.pi, 10, endpoint=False)
    # A radius-2.5 ring standing in the plane spanned by z and the horizontal heading at pi/2,
    # which is edge-on to the "+x" stage axis. `np.cos(np.pi / 2)` is 6.1e-17, not 0, and that is
    # the whole point of the fixture: writing the x column as the literal 0.5 pins nothing.
    edge_on = with_ulp_noise(
        np.stack(
            [
                0.5 + 2.5 * np.cos(angles) * np.cos(np.pi / 2),
                2.5 * np.cos(angles) * np.sin(np.pi / 2),
                3.0 + 2.5 * np.sin(angles),
            ],
            axis=1,
        )
    )
    span = float(edge_on[:, 0].max() - edge_on[:, 0].min())
    assert 0.0 < span < 1e-12, (
        f"the fixture must be degenerate only to within float noise, got a span of {span}; "
        "an exactly-equal x column passes this test without the tolerance and pins nothing"
    )
    assert not select(("right", ()), 10, edge_on, cfg).any()
    assert select(("left", ()), 10, edge_on, cfg).all()
    assert spread_offsets("alternate_side", ALL_10, edge_on, 1, cfg) == pytest.approx(np.zeros(10))
    assert lighting_log.records, "the collapse must be reported, not decided by rounding"
    assert lighting_log.records[0].levelno == logging.WARNING
    assert cfg.stage_axis in lighting_log.records[0].getMessage(), "name the axis"


def test_a_stage_axis_with_extent_does_not_warn(lighting_log: pytest.LogCaptureFixture):
    cfg = _staged("+x")
    assert select(("right", ()), 6, POSITIONS_6, cfg).any()
    assert not lighting_log.records


def test_a_flat_formation_warns_that_the_height_split_collapsed(
    lighting_log: pytest.LogCaptureFixture,
):
    """The failure mode `upper`/`lower` invites: every formation that is not vertical is flat.

    A ring or a grid at one altitude has no upper half, so `upper` paints nobody while `lower`
    covers the swarm. The show still runs, hence a warning -- but it has to name the axis.
    """
    cfg = load_lighting_config()
    flat = np.stack([np.arange(6.0), np.arange(6.0), np.full(6, 1.2)], axis=1)
    assert not select(("upper", ()), 6, flat, cfg).any()
    assert select(("lower", ()), 6, flat, cfg).all()
    assert lighting_log.records, "a silent no-op is the failure mode being fixed"
    assert lighting_log.records[0].levelno == logging.WARNING
    assert "z" in lighting_log.records[0].getMessage(), "name the axis"


def test_a_formation_with_height_does_not_warn(lighting_log: pytest.LogCaptureFixture):
    cfg = load_lighting_config()
    assert select(("upper", ()), 6, POSITIONS_CROSSED_6, cfg).any()
    assert not lighting_log.records


def test_unknown_spread_raises():
    cfg = load_lighting_config()
    with pytest.raises(KeyError):
        spread_offsets("spiral", ALL_10, POSITIONS_10, 1, cfg)


# Hand-computed from HSV at full saturation and value, normalized to a constant channel sum of 255,
# and only then multiplied by channel_gain [1, 1, 1, 0.8].
PRIMARY_WRGB = {
    0 / 6: [0.0, 255.0, 0.0, 0.0],
    1 / 6: [0.0, 127.5, 127.5, 0.0],
    2 / 6: [0.0, 0.0, 255.0, 0.0],
    3 / 6: [0.0, 0.0, 127.5, 0.8 * 127.5],
    4 / 6: [0.0, 0.0, 0.0, 0.8 * 255.0],
    5 / 6: [0.0, 127.5, 0.0, 0.8 * 127.5],
}


def test_hue_wheel_closes():
    cfg = load_lighting_config()
    assert hue_to_wrgb(np.array(1.0), cfg) == pytest.approx(hue_to_wrgb(np.array(0.0), cfg))


def test_hue_primaries_land_where_expected():
    cfg = load_lighting_config()
    for hue, expected in PRIMARY_WRGB.items():
        assert hue_to_wrgb(np.array(hue), cfg) == pytest.approx(expected), hue


def test_hue_output_shape_follows_the_input():
    cfg = load_lighting_config()
    assert hue_to_wrgb(np.array(0.3), cfg).shape == (4,)
    assert hue_to_wrgb(np.zeros(7), cfg).shape == (7, 4)
    assert hue_to_wrgb(np.zeros((2, 3)), cfg).shape == (2, 3, 4)


def test_generated_hues_never_drive_the_white_led():
    cfg = load_lighting_config()
    out = hue_to_wrgb(np.linspace(0.0, 1.0, 101), cfg)
    assert np.all(out[:, 0] == 0.0)


def test_every_hue_stays_within_the_addressable_range():
    cfg = load_lighting_config()
    out = hue_to_wrgb(np.linspace(0.0, 1.0, 101), cfg)
    assert np.all(out >= 0.0) and np.all(out <= 255.0)


def test_hue_gain_holds_perceived_output_constant_across_the_wheel():
    """A rainbow that visibly throbs as it sweeps would show up here as a varying output.

    The constant quantity is the *gain-corrected* sum; asserting on the raw sum would pass even
    with the gain applied before the normalization, which is the ordering bug this pins.
    """
    cfg = load_lighting_config()
    out = hue_to_wrgb(np.arange(cfg.hue_steps) / cfg.hue_steps, cfg)
    corrected = (out / cfg.channel_gain).sum(axis=-1)
    assert np.ptp(corrected) < 0.01 * corrected.mean()


def test_generated_pure_blue_matches_the_palette_entry():
    """The gain must land *after* the normalization, or pure blue comes out at 255 instead of 204."""
    cfg = load_lighting_config()
    assert hue_to_wrgb(np.array(4 / 6), cfg) == pytest.approx(cfg.palette["blue"], abs=0.5)


def test_colour_layer_named_returns_the_palette_entry_on_masked_rows():
    cfg = load_lighting_config()
    mask = np.array([True, False, True])
    layer = ColourLayer(mask, ("top", "bot"), "named", {"color": "amber"})
    out = layer.evaluate(0.0, cfg)
    assert out[0] == pytest.approx(cfg.palette["amber"])
    assert out[2] == pytest.approx(cfg.palette["amber"])
    assert out[1] == pytest.approx(np.zeros(4)), "unselected rows are left dark for the LTP merge"


def test_colour_layer_named_is_static_in_time():
    cfg = load_lighting_config()
    layer = ColourLayer(np.ones(3, dtype=bool), ("top",), "named", {"color": "teal"})
    assert layer.evaluate(0.0, cfg) == pytest.approx(layer.evaluate(97.3, cfg))


def test_colour_layer_named_rejects_an_unknown_colour():
    cfg = load_lighting_config()
    layer = ColourLayer(np.ones(2, dtype=bool), ("top",), "named", {"color": "chartreuse"})
    with pytest.raises(KeyError):
        layer.evaluate(0.0, cfg)


def test_colour_layer_gradient_hits_both_endpoints_and_their_average():
    cfg = load_lighting_config()
    params = {"color_a": "red", "color_b": "blue", "s": np.array([0.0, 0.5, 1.0])}
    out = ColourLayer(np.ones(3, dtype=bool), ("top",), "gradient", params).evaluate(0.0, cfg)
    assert out[0] == pytest.approx(cfg.palette["red"])
    assert out[2] == pytest.approx(cfg.palette["blue"])
    assert out[1] == pytest.approx(0.5 * (cfg.palette["red"] + cfg.palette["blue"]))


def test_colour_layer_gradient_is_static_in_time():
    """`gradient` costs the same as a named colour to compile because it never changes."""
    cfg = load_lighting_config()
    params = {"color_a": "green", "color_b": "pink", "s": np.array([0.0, 0.25, 1.0])}
    layer = ColourLayer(np.ones(3, dtype=bool), ("top",), "gradient", params)
    assert layer.evaluate(0.0, cfg) == pytest.approx(layer.evaluate(42.0, cfg))


def test_colour_layer_cycled_advances_hue_with_time():
    """A quarter of the way through the period the hue is a quarter of the way round the wheel."""
    cfg = load_lighting_config()
    params = {"period_s": 4.0, "offsets": np.zeros(2)}
    layer = ColourLayer(np.ones(2, dtype=bool), ("top",), "cycled", params)
    assert layer.evaluate(0.0, cfg)[0] == pytest.approx(hue_to_wrgb(np.array(0.0), cfg))
    assert layer.evaluate(1.0, cfg)[0] == pytest.approx(hue_to_wrgb(np.array(0.25), cfg))


def test_colour_layer_cycled_applies_the_spread_offsets():
    """The same offsets that make a chase run along drone order make a rainbow travel along it."""
    cfg = load_lighting_config()
    params = {"period_s": 4.0, "offsets": np.array([0.0, 0.5])}
    out = ColourLayer(np.ones(2, dtype=bool), ("top",), "cycled", params).evaluate(0.0, cfg)
    assert out[0] == pytest.approx(hue_to_wrgb(np.array(0.0), cfg))
    assert out[1] == pytest.approx(hue_to_wrgb(np.array(0.5), cfg))


def test_colour_layer_cycled_travels_forward_along_the_offsets():
    """The hue a drone carries is handed on to the *next* drone in the spread, never the previous.

    Offsets of 0 and 0.5 cannot see the direction `phase = t / period - offset` sets: 0.5 is its
    own negative mod 1. Quarter-turn offsets break that symmetry.
    """
    cfg = load_lighting_config()
    offsets = np.array([0.0, 0.25, 0.5, 0.75])
    params = {"period_s": 4.0, "offsets": offsets}
    layer = ColourLayer(np.ones(4, dtype=bool), ("top",), "cycled", params)
    # Quarter-turn offsets land on exact `hue_steps` boundaries, so quantization is the identity.
    assert layer.evaluate(0.0, cfg) == pytest.approx(
        hue_to_wrgb(np.array([0.0, 0.75, 0.5, 0.25]), cfg)
    )
    # A quarter period later, drone 1 carries what drone 0 carried -- the spectrum travelled along
    # drone order. Reversed, it would have gone to drone 3.
    assert layer.evaluate(1.0, cfg)[1] == pytest.approx(layer.evaluate(0.0, cfg)[0])


def test_colour_layer_cycled_is_quantized_to_hue_steps_per_period():
    """Quantization is what restores cue dedup for the one primitive that defeats it."""
    cfg = load_lighting_config()
    period = 4.0
    params = {"period_s": period, "offsets": np.zeros(1)}
    layer = ColourLayer(np.ones(1, dtype=bool), ("top",), "cycled", params)
    seen = {
        tuple(layer.evaluate(t, cfg)[0])
        for t in np.linspace(0.0, period, 100 * cfg.hue_steps, endpoint=False)
    }
    assert len(seen) == cfg.hue_steps


def test_colour_layer_rejects_an_unknown_kind():
    cfg = load_lighting_config()
    with pytest.raises(KeyError):
        ColourLayer(np.ones(2, dtype=bool), ("top",), "strobe", {}).evaluate(0.0, cfg)


# Six drones, so even/odd split them three and three. The merge rules are position-free -- they see
# the swarm only through the masks the caller hands them -- so these tests never need positions.
N6 = 6
ALL_6 = np.ones(N6, dtype=bool)
EVEN_6 = np.array([True, False, True, False, True, False])
ODD_6 = ~EVEN_6
BOTH_DECKS = ("top", "bot")

# Deck indices into an (n, 2, 4) evaluate() result.
TOP, BOT = 0, 1

# Rounded palette entries, which is what a full-brightness read-out must produce.
RED = np.array([0.0, 255.0, 0.0, 0.0])
GREEN = np.array([0.0, 0.0, 255.0, 0.0])
WHITE = np.array([255.0, 0.0, 0.0, 0.0])
AMBER = np.array([0.0, 146.0, 109.0, 0.0])


def _linear_cfg() -> LightingConfig:
    """Shipped config with gamma = 1, so a merged brightness reads straight off the WRGB output.

    Otherwise retuning gamma by eye on hardware breaks tests that are not about gamma.
    """
    return dataclasses.replace(load_lighting_config(), gamma=1.0)


def _ramp(
    mask: np.ndarray,
    period_s: float = 4.0,
    offsets: np.ndarray | None = None,
    decks: tuple[str, ...] = BOTH_DECKS,
) -> BrightnessLayer:
    """A ramp brightness layer. At period 4 the quarter phases give exact 0.75/0.5/0.25 values."""
    return BrightnessLayer(
        mask, decks, "ramp", period_s, 0.5, np.zeros(N6) if offsets is None else offsets
    )


def _on(mask: np.ndarray, decks: tuple[str, ...] = BOTH_DECKS) -> BrightnessLayer:
    """A `light_on` layer: an HTP participant pinned at 1.0."""
    return BrightnessLayer(mask, decks, "constant", 0.0, 0.5, np.zeros(N6))


def _named(mask: np.ndarray, color: str, decks: tuple[str, ...] = BOTH_DECKS) -> ColourLayer:
    return ColourLayer(mask, decks, "named", {"color": color})


def _look(
    t_start: float,
    colours: tuple[ColourLayer, ...] = (),
    brightnesses: tuple[BrightnessLayer, ...] = (),
    off: np.ndarray | None = None,
    positions: np.ndarray | None = None,
) -> Look:
    return Look(
        t_start,
        tuple(colours),
        tuple(brightnesses),
        np.zeros((N6, 2), dtype=bool) if off is None else off,
        positions,
    )


def _both_decks(mask: np.ndarray) -> np.ndarray:
    """Lift an (n,) selection into the (n, 2) per-deck shape `Look.off_mask` carries."""
    return np.stack([mask, mask], axis=1)


def _base_colours() -> np.ndarray:
    """The base colour of a look-less timeline: drone i carries hue i / n, in id order.

    Six drones land on the six primaries, so drone 0 is red, 2 green and 4 blue.
    """
    return np.round(hue_to_wrgb(np.arange(N6) / N6, load_lighting_config()))


# Six unevenly spaced slots along +x, with the ids scrambled across them the way `_assign_positions`
# scrambles them. Drone i sits at slot SLOT_OF[i], counting from the -x end, so its
# base hue is SLOT_OF[i] / 6 rather than i / 6. A line rather than a ring here because the
# walk over it is unambiguous, which makes the expected hue per drone hand-computable.
LINE_SLOTS_6 = np.stack(
    [np.array([-2.0, -1.4, -0.2, 0.9, 1.5, 2.8]), np.zeros(N6), np.full(N6, 1.3)], axis=1
)
SLOT_OF = np.array([2, 5, 0, 3, 1, 4])
LINE_6 = LINE_SLOTS_6[SLOT_OF]


def test_brightness_layer_constant_is_one_on_the_masked_rows():
    assert _on(EVEN_6).evaluate(3.7) == pytest.approx([1.0, 0.0, 1.0, 0.0, 1.0, 0.0])


def test_brightness_layer_leaves_unselected_rows_at_zero():
    """Unselected rows are zero so the HTP merge can reduce with a plain `max`."""
    assert _ramp(EVEN_6).evaluate(0.0) == pytest.approx([1.0, 0.0, 1.0, 0.0, 1.0, 0.0])


def test_brightness_layer_applies_its_phase_offsets():
    """The same offsets that make a chase run along drone order phase-shift its brightness."""
    offsets = np.array([0.0, 0.5, 0.0, 0.5, 0.0, 0.5])
    out = _ramp(ALL_6, period_s=4.0, offsets=offsets).evaluate(1.0)
    assert out == pytest.approx([0.75, 0.25, 0.75, 0.25, 0.75, 0.25])


def test_brightness_layer_travels_forward_along_the_offsets():
    """A chase runs from low drone index to high, and `phase = t / period - offset` says so.

    At offsets of 0 or 0.5 the sign of the offset term is invisible and a backwards chase reads
    identically, so this is pinned on quarter-turn offsets, where the two differ.
    """
    offsets = np.array([0.0, 0.25, 0.5, 0.75])
    layer = BrightnessLayer(np.ones(4, dtype=bool), BOTH_DECKS, "square", 4.0, 0.25, offsets)
    lit = []
    for t in (0.0, 1.0, 2.0, 3.0):
        on = np.flatnonzero(layer.evaluate(t))
        assert on.size == 1, f"a quarter duty over four evenly spread phases lights one, at t={t}"
        lit.append(int(on[0]))
    assert lit == [0, 1, 2, 3], "the lit drone must advance with time, not retreat"


def test_brightness_layer_rejects_an_unknown_kind():
    with pytest.raises(KeyError):
        BrightnessLayer(ALL_6, BOTH_DECKS, "sawtooth", 4.0, 0.5, np.zeros(N6)).evaluate(0.0)


def test_final_wrgb_is_colour_times_brightness():
    """At t = 1 a period-4 ramp is exactly 0.75, so 255 * 0.75 = 191.25 rounds to 191."""
    cfg = _linear_cfg()
    timeline = LightingTimeline(
        [_look(0.0, (_named(ALL_6, "red"),), (_ramp(ALL_6),))], N6, 100.0, cfg
    )
    assert timeline.evaluate(1.0)[0, TOP] == pytest.approx([0.0, 191.0, 0.0, 0.0])


def test_gamma_applies_to_the_brightness_and_not_to_the_colour():
    """gamma = 2 turns b = 0.75 into 0.5625, so 255 * 0.5625 = 143.4 rounds to 143.

    At full brightness the two gammas must agree, which says gamma never touches the colour.
    """
    squared = dataclasses.replace(load_lighting_config(), gamma=2.0)
    dimmed = LightingTimeline(
        [_look(0.0, (_named(ALL_6, "red"),), (_ramp(ALL_6),))], N6, 100.0, squared
    )
    assert dimmed.evaluate(1.0)[0, TOP] == pytest.approx([0.0, 143.0, 0.0, 0.0])
    for cfg in (squared, _linear_cfg()):
        full = LightingTimeline(
            [_look(0.0, (_named(ALL_6, "red"),), (_on(ALL_6),))], N6, 100.0, cfg
        )
        assert full.evaluate(1.0)[0, TOP] == pytest.approx(RED)


def test_brightness_never_changes_the_hue():
    """Dimming scales all four channels uniformly, so the channel ratio survives it."""
    cfg = _linear_cfg()
    full = LightingTimeline([_look(0.0, (_named(ALL_6, "amber"),), (_on(ALL_6),))], N6, 100.0, cfg)
    dim = LightingTimeline([_look(0.0, (_named(ALL_6, "amber"),), (_ramp(ALL_6),))], N6, 100.0, cfg)
    bright, faded = full.evaluate(1.0)[0, TOP], dim.evaluate(1.0)[0, TOP]
    assert bright == pytest.approx(AMBER)
    assert faded == pytest.approx([0.0, 109.0, 82.0, 0.0])
    assert faded[1] / faded[2] == pytest.approx(bright[1] / bright[2], rel=0.02)


def test_a_colour_layer_never_changes_the_brightness():
    """Two different colours under the same brightness layer are both scaled by the same factor."""
    cfg = _linear_cfg()
    for color, expected in (("red", [0.0, 191.0, 0.0, 0.0]), ("amber", [0.0, 109.0, 82.0, 0.0])):
        timeline = LightingTimeline(
            [_look(0.0, (_named(ALL_6, color),), (_ramp(ALL_6),))], N6, 100.0, cfg
        )
        assert timeline.evaluate(1.0)[0, TOP] == pytest.approx(expected), color


def test_brightness_below_b_min_goes_fully_dark():
    """Below b_min the LED is quantization noise and coloured fringing, so it is cut.

    The floor precedes quantization, or it is inert -- the smallest non-zero bucket exceeds any
    b_min. Hence the finer dimmer: at 16 steps the bottom bucket answers for every such value.
    """
    cfg = dataclasses.replace(_linear_cfg(), brightness_steps=255)
    timeline = LightingTimeline(
        [_look(0.0, (_named(ALL_6, "red"),), (_ramp(ALL_6, period_s=1.0),))], N6, 100.0, cfg
    )
    assert cfg.b_min == 0.02
    assert timeline.evaluate(0.99)[0, TOP] == pytest.approx(np.zeros(4))  # b = 0.01 < b_min
    # b = 0.021 is above b_min and stays lit, even though its bucket floor (5/255 = 0.0196) is
    # below it: the comparison reads the continuous value, not the quantized one.
    assert timeline.evaluate(0.979)[0, TOP] == pytest.approx([0.0, 5.0, 0.0, 0.0])
    # Clear of the floor, quantization is the only thing acting: b = 0.25 -> bucket 63/255.
    assert timeline.evaluate(0.75)[0, TOP] == pytest.approx([0.0, 63.0, 0.0, 0.0])


def test_brightness_is_quantized_to_brightness_steps_before_the_multiply():
    """`brightness_steps` buckets the merged brightness, which is what lets dedup collapse a ramp.

    Sixteen samples over a period-4 ramp land on four bucket floors plus the exact-1.0 peak.
    """
    cfg = dataclasses.replace(_linear_cfg(), brightness_steps=4)
    timeline = LightingTimeline(
        [_look(0.0, (_named(ALL_6, "red"),), (_ramp(ALL_6),))], N6, 100.0, cfg
    )
    reds = [timeline.evaluate(0.25 * k)[0, TOP][1] for k in range(16)]
    assert sorted(set(reds)) == [0.0, 64.0, 128.0, 191.0, 255.0]


def test_brightness_quantization_leaves_a_ramp_monotone():
    """The visual result is still a recognisable ramp: piecewise constant, but never non-monotone."""
    cfg = dataclasses.replace(_linear_cfg(), brightness_steps=8)
    timeline = LightingTimeline(
        [_look(0.0, (_named(ALL_6, "red"),), (_ramp(ALL_6),))], N6, 100.0, cfg
    )
    # One descending ramp segment, sampled far finer than the quantizer's own resolution.
    reds = np.array([timeline.evaluate(0.02 * k)[0, TOP][1] for k in range(200)])
    assert np.all(np.diff(reds) <= 0.0)
    assert len(set(reds)) >= 6  # still a ramp, not a single hold


def test_quantization_floors_rather_than_rounds_so_it_can_only_darken():
    """`floor` and `round` differ only at the bottom of the range, and `floor` is the darker one.

    A brightness in ``[1/(2*steps), 1/steps)`` rounds *up* into the first lit bucket but floors to
    zero, and only darkening is safe. `brightness_steps` is explicit: the shipped value is a lever.
    """
    cfg = dataclasses.replace(_linear_cfg(), brightness_steps=16)
    timeline = LightingTimeline(
        [_look(0.0, (_named(ALL_6, "red"),), (_ramp(ALL_6, period_s=1.0),))], N6, 100.0, cfg
    )
    # b = 0.05, inside [1/32, 1/16) and well above b_min: floor gives 0, round gives 1/16 and a
    # visibly lit LED at 16/255.
    assert timeline.evaluate(0.95)[0, TOP] == pytest.approx(np.zeros(4))


def test_quantization_never_dims_a_fully_lit_drone():
    """A brightness of exactly 1.0 is the top bucket, so `light_on` and the base state are exact."""
    for steps in (4, 8, 16, 64):
        cfg = dataclasses.replace(_linear_cfg(), brightness_steps=steps)
        timeline = LightingTimeline(
            [_look(0.0, (_named(ALL_6, "red"),), (_on(ALL_6),))], N6, 100.0, cfg
        )
        assert timeline.evaluate(1.0)[0, TOP] == pytest.approx(RED), steps


def test_colour_ltp_later_layer_wins_on_its_subset():
    """light_color(all, blue) then light_color(even, amber) is two colours from two primitives."""
    cfg = _linear_cfg()
    look = _look(0.0, (_named(ALL_6, "blue"), _named(EVEN_6, "amber")))
    out = LightingTimeline([look], N6, 100.0, cfg).evaluate(0.0)[:, TOP]
    assert out[EVEN_6] == pytest.approx(np.tile(AMBER, (3, 1)))
    assert out[ODD_6] == pytest.approx(np.tile([0.0, 0.0, 0.0, 204.0], (3, 1)))


def test_colour_ltp_overwrites_by_mask_not_by_non_zero_channels():
    """A merge that keeps whichever channel is non-zero blends a layer into whatever it covers.

    Drone 2's base green and `red` share no non-zero channel, so a "non-zero wins" merge yields
    yellow. The stacked case pins the same trap between two layers rather than against the base.
    """
    cfg = _linear_cfg()
    over_base = LightingTimeline([_look(0.0, (_named(EVEN_6, "red"),))], N6, 100.0, cfg)
    out = over_base.evaluate(0.0)[:, TOP]
    assert out[EVEN_6] == pytest.approx(np.tile(RED, (3, 1)))
    assert out[ODD_6] == pytest.approx(_base_colours()[ODD_6])
    stacked = LightingTimeline(
        [_look(0.0, (_named(ALL_6, "white"), _named(EVEN_6, "red")))], N6, 100.0, cfg
    )
    stacked_out = stacked.evaluate(0.0)[:, TOP]
    assert stacked_out[EVEN_6] == pytest.approx(np.tile(RED, (3, 1)))
    assert stacked_out[ODD_6] == pytest.approx(np.tile(WHITE, (3, 1)))


def test_colour_ltp_precedence_is_by_position_not_by_kind():
    """A named layer after a cycled one freezes its drones while the rest keep cycling."""
    cfg = _linear_cfg()
    cycled = ColourLayer(ALL_6, BOTH_DECKS, "cycled", {"period_s": 4.0, "offsets": np.zeros(N6)})
    named_last = LightingTimeline([_look(0.0, (cycled, _named(EVEN_6, "amber")))], N6, 100.0, cfg)
    frozen = named_last.evaluate(0.0)[:, TOP], named_last.evaluate(1.0)[:, TOP]
    assert frozen[0][EVEN_6] == pytest.approx(np.tile(AMBER, (3, 1)))
    assert frozen[1][EVEN_6] == pytest.approx(np.tile(AMBER, (3, 1)))
    assert frozen[0][ODD_6] != pytest.approx(frozen[1][ODD_6]), "odd drones must keep cycling"
    cycled_last = LightingTimeline([_look(0.0, (_named(EVEN_6, "amber"), cycled))], N6, 100.0, cfg)
    reversed_out = cycled_last.evaluate(0.0)[:, TOP]
    assert reversed_out == pytest.approx(
        np.tile(np.round(hue_to_wrgb(np.array(0.0), cfg)), (N6, 1))
    )


def test_brightness_htp_takes_the_max_not_the_sum():
    """Two ramps half a turn apart read 0.75 and 0.25 at t = 1: max is 191, a sum would be 255."""
    cfg = _linear_cfg()
    ahead = _ramp(ALL_6)
    behind = _ramp(ALL_6, offsets=np.full(N6, 0.5))
    look = _look(0.0, (_named(ALL_6, "red"),), (ahead, behind))
    assert LightingTimeline([look], N6, 100.0, cfg).evaluate(1.0)[0, TOP] == pytest.approx(
        [0.0, 191.0, 0.0, 0.0]
    )


def test_light_off_kills_drones_that_a_competing_layer_lights():
    """light_off is a post-reduction kill mask, not an HTP participant.

    The ramp runs over *every* drone deliberately: as an HTP participant light_off contributes 0
    and `max(0, ramp)` is just `ramp`. Sample times stay clear of the ramp's dark last bucket.
    """
    cfg = _linear_cfg()
    look = _look(0.0, (_named(ALL_6, "red"),), (_ramp(ALL_6),), off=_both_decks(EVEN_6))
    timeline = LightingTimeline([look], N6, 100.0, cfg)
    for t in (0.0, 1.0, 2.0, 3.0, 3.5):
        out = timeline.evaluate(t)
        assert np.all(out[EVEN_6] == 0.0), f"light_off drones must be dark at t={t}"
        assert np.all(out[ODD_6, :, 1] > 0.0), f"the competing layer must still light odd at t={t}"


def test_light_on_dominates_a_pulse_underneath_it():
    """light_on is an HTP participant at 1.0, so it swallows every other layer."""
    cfg = _linear_cfg()
    look = _look(0.0, (_named(ALL_6, "red"),), (_ramp(ALL_6), _on(ALL_6)))
    timeline = LightingTimeline([look], N6, 100.0, cfg)
    for t in (0.0, 1.0, 2.5, 3.9):
        assert timeline.evaluate(t)[0, TOP] == pytest.approx(RED), t


def test_a_layer_reading_zero_still_suppresses_the_base():
    """Base suppression is decided by the layer's mask, never by the value it happens to return.

    A square wave off-phase returns exactly 0, so deciding from the value inverts the blink.
    """
    cfg = _linear_cfg()
    blink = BrightnessLayer(
        np.array([True, False, False, False, False, False]),
        BOTH_DECKS,
        "square",
        4.0,
        0.5,
        np.zeros(N6),
    )
    timeline = LightingTimeline([_look(0.0, (_named(ALL_6, "red"),), (blink,))], N6, 100.0, cfg)
    out = timeline.evaluate(3.0)  # phase 0.75, past the duty, so the layer reads 0
    assert out[0, TOP] == pytest.approx(np.zeros(4)), "covered drone stays dark in the off phase"
    assert out[1, TOP] == pytest.approx(RED), "an uncovered drone still sits on the base"


def test_a_look_does_not_inherit_the_previous_looks_layers():
    """Each key defines a complete look; the next key replaces it, deltas included."""
    cfg = _linear_cfg()
    first = _look(0.0, (_named(ALL_6, "red"),), (_ramp(ALL_6),))
    second = _look(10.0, (_named(ALL_6, "green"),))
    timeline = LightingTimeline([first, second], N6, 100.0, cfg)
    assert timeline.evaluate(1.0)[0, TOP] == pytest.approx([0.0, 191.0, 0.0, 0.0])
    # Inheriting the ramp would put t = 11 at phase 0.75, i.e. green at 0.25 -> 64, not 255.
    assert timeline.evaluate(11.0)[0, TOP] == pytest.approx(GREEN)


def test_look_dispatch_is_by_start_time_and_independent_of_emission_order():
    """Lighting keys carry no ordering guarantee, so dispatch sorts by t_start."""
    cfg = _linear_cfg()
    late = _look(10.0, (_named(ALL_6, "green"),))
    early = _look(2.0, (_named(ALL_6, "red"),))
    timeline = LightingTimeline([late, early], N6, 100.0, cfg)
    # Drone 1's base hue is neither look colour, so all three states are distinguishable on it.
    base = _base_colours()[1]
    assert timeline.evaluate(0.0)[1, TOP] == pytest.approx(base), "before any look, the base state"
    assert timeline.evaluate(2.0)[1, TOP] == pytest.approx(RED), "a look owns its own start instant"
    assert timeline.evaluate(9.99)[1, TOP] == pytest.approx(RED)
    assert timeline.evaluate(10.0)[1, TOP] == pytest.approx(GREEN)
    # Re-entered last: a merge that painted over the stored base would have destroyed it by now.
    assert timeline.evaluate(0.0)[1, TOP] == pytest.approx(base), "the base survives a look"


def test_an_empty_timeline_is_full_on():
    """One forgetful emission must not black out the show, so the base is a fallback at 1.0."""
    cfg = _linear_cfg()
    empty = LightingTimeline([], N6, 100.0, cfg).evaluate(5.0)
    forced = LightingTimeline([_look(0.0, (), (_on(ALL_6),))], N6, 100.0, cfg).evaluate(5.0)
    assert empty.shape == (N6, 2, 4)
    assert empty == pytest.approx(forced), "an empty stack must match an explicit light_on(all)"
    assert empty.max() == 255.0, "and be undimmed, not merely non-zero"


def test_the_base_colour_is_each_drones_own_hue_off_the_wheel():
    """A colour-less emission reproduces today's per-drone colouring exactly.

    A swarm flattened to one hue is unidentifiable in the viewer and in flight. Same wheel as
    `generate_default_colors`, with the blue dim now carried by `channel_gain`.
    """
    cfg = _linear_cfg()
    out = LightingTimeline([], N6, 100.0, cfg).evaluate(0.0)[:, TOP]
    for i in range(N6):
        assert out[i] == pytest.approx(np.round(hue_to_wrgb(np.array(i / N6), cfg))), i
    assert len({tuple(row) for row in out}) == N6, "no two drones may share a base colour"
    assert out[4] == pytest.approx([0.0, 0.0, 0.0, 204.0]), "and blue still carries its 0.8 dim"


def test_the_base_colour_within_a_look_follows_neighbour_order_not_id_order():
    """The default wheel must read as a wheel *around the formation*, not around the id list.

    `_assign_positions` hands ids out by cheapest assignment, so an id-keyed wheel puts unrelated
    hues side by side. Within a look the wheel follows that look's nearest-neighbour walk.
    """
    cfg = _linear_cfg()
    out = LightingTimeline([_look(0.0, positions=LINE_6)], N6, 100.0, cfg).evaluate(3.0)[:, TOP]
    for drone, slot in enumerate(SLOT_OF):
        expected = np.round(hue_to_wrgb(np.array(slot / N6), cfg))
        assert out[drone] == pytest.approx(expected), drone
    assert len({tuple(row) for row in out}) == N6, "still one distinct hue per drone"
    assert not np.allclose(out, _base_colours()), "ring order and id order must actually differ"


def test_no_drone_changes_colour_when_the_first_look_takes_over():
    """A timeline whose first key resolves to t > 0 must not re-colour the whole swarm at it.

    Without a snapshot the pre-show look falls back to the id-order scramble the walk exists to
    remove, and the swarm snaps out of it in one frame -- measured, 10 of 10 drones changed.
    """
    cfg = _linear_cfg()
    timeline = LightingTimeline([_look(20.0, positions=LINE_6)], N6, 100.0, cfg)
    before, after = timeline.evaluate(19.9)[:, TOP], timeline.evaluate(20.0)[:, TOP]
    assert before == pytest.approx(after), "the first look must not repaint the swarm"
    for drone, slot in enumerate(SLOT_OF):
        expected = np.round(hue_to_wrgb(np.array(slot / N6), cfg))
        assert before[drone] == pytest.approx(expected), drone
    assert not np.allclose(before, _base_colours()), "id order and walk order must actually differ"


def test_the_pre_show_base_takes_the_earliest_looks_snapshot_whatever_the_emission_order():
    """The looks arrive in emission order, so "first" means earliest in time, not first in the list."""
    cfg = _linear_cfg()
    reformed = LINE_SLOTS_6[(N6 - 1) - SLOT_OF]
    timeline = LightingTimeline(
        [_look(40.0, positions=reformed), _look(20.0, positions=LINE_6)], N6, 100.0, cfg
    )
    for drone, slot in enumerate(SLOT_OF):
        expected = np.round(hue_to_wrgb(np.array(slot / N6), cfg))
        assert timeline.evaluate(5.0)[drone, TOP] == pytest.approx(expected), drone


def test_a_timeline_with_no_looks_keeps_the_id_ordered_base():
    """Nothing was authored and there is no snapshot to order against, so id order stands.

    A show carrying no lighting reproduces `generate_default_colors` drone for drone.
    """
    cfg = _linear_cfg()
    out = LightingTimeline([], N6, 100.0, cfg).evaluate(3.0)[:, TOP]
    assert out == pytest.approx(_base_colours())


def test_each_look_orders_the_base_colour_against_its_own_snapshot():
    """The wheel re-sorts as formations change, which is why the snapshot rides on the look.

    The second look mirrors the assignment end to end, so every base hue must mirror with it --
    which a single wheel computed once for the timeline could not do.
    """
    cfg = _linear_cfg()
    reformed = LINE_SLOTS_6[(N6 - 1) - SLOT_OF]
    timeline = LightingTimeline(
        [_look(0.0, positions=LINE_6), _look(20.0, positions=reformed)], N6, 100.0, cfg
    )
    before, after = timeline.evaluate(5.0)[:, TOP], timeline.evaluate(25.0)[:, TOP]
    for drone, slot in enumerate(SLOT_OF):
        held = np.round(hue_to_wrgb(np.array(slot / N6), cfg))
        assert before[drone] == pytest.approx(held), drone
        mirrored = np.round(hue_to_wrgb(np.array((N6 - 1 - slot) / N6), cfg))
        assert after[drone] == pytest.approx(mirrored), drone


def test_the_base_colour_of_a_one_drone_swarm_is_the_top_of_the_wheel():
    """The walk over a single point has no step to take, and `rank / n` must stay 0, not wrap."""
    cfg = _linear_cfg()
    solo = Look(0.0, (), (), np.zeros((1, 2), dtype=bool), np.array([[0.4, -1.0, 1.1]]))
    out = LightingTimeline([solo], 1, 100.0, cfg).evaluate(3.0)
    assert out.shape == (1, 2, 4)
    assert out[0, TOP] == pytest.approx(np.round(hue_to_wrgb(np.array(0.0), cfg)))


def test_a_layer_on_one_drone_leaves_the_others_on_the_base():
    """The base is suppressed per drone, not for the whole swarm."""
    cfg = _linear_cfg()
    covered = np.zeros(N6, dtype=bool)
    covered[3] = True
    timeline = LightingTimeline(
        [_look(0.0, (_named(ALL_6, "red"),), (_ramp(covered),))], N6, 100.0, cfg
    )
    out = timeline.evaluate(1.0)[:, TOP]
    assert out[3] == pytest.approx([0.0, 191.0, 0.0, 0.0]), "drone 3 is driven by the layer"
    assert out[5] == pytest.approx(RED), "drone 5 has no layer, so it sits at brightness 1.0"


def test_a_colour_only_look_is_full_brightness():
    """The failure-safe property: colour primitives alone reproduce today's lights-on behaviour."""
    cfg = _linear_cfg()
    timeline = LightingTimeline([_look(0.0, (_named(ALL_6, "teal"),))], N6, 100.0, cfg)
    for t in (0.0, 12.5, 60.0):
        assert timeline.evaluate(t) == pytest.approx(np.tile([0.0, 0.0, 146.0, 87.0], (N6, 2, 1)))


def test_a_top_only_brightness_effect_leaves_bot_on_the_base():
    cfg = _linear_cfg()
    look = _look(0.0, (_named(ALL_6, "red"),), (_ramp(ALL_6, decks=("top",)),))
    out = LightingTimeline([look], N6, 100.0, cfg).evaluate(1.0)
    assert out[0, TOP] == pytest.approx([0.0, 191.0, 0.0, 0.0])
    assert out[0, BOT] == pytest.approx(RED)


def test_colour_stacks_resolve_independently_per_deck():
    """A slow wash on bot under a different colour on top is free -- the hardware has two decks."""
    cfg = _linear_cfg()
    look = _look(
        0.0, (_named(ALL_6, "red", decks=("top",)), _named(ALL_6, "green", decks=("bot",)))
    )
    out = LightingTimeline([look], N6, 100.0, cfg).evaluate(0.0)
    assert out[0, TOP] == pytest.approx(RED)
    assert out[0, BOT] == pytest.approx(GREEN)


def test_light_off_is_per_deck():
    cfg = _linear_cfg()
    off = np.zeros((N6, 2), dtype=bool)
    off[:, TOP] = True
    look = _look(0.0, (_named(ALL_6, "red"),), (_on(ALL_6),), off=off)
    out = LightingTimeline([look], N6, 100.0, cfg).evaluate(0.0)
    assert np.all(out[:, TOP] == 0.0)
    assert out[:, BOT] == pytest.approx(np.tile(RED, (N6, 1)))


def test_the_terminal_blackout_is_present_whatever_was_emitted():
    """The drones never land lit, and this is not the LLM's to override."""
    cfg = _linear_cfg()
    look = _look(0.0, (_named(ALL_6, "white"),), (_on(ALL_6),))
    timeline = LightingTimeline([look], N6, 10.0, cfg)
    assert timeline.evaluate(9.8)[0, TOP] == pytest.approx(WHITE)
    for t in (9.9, 9.95, 10.0, 50.0):
        assert np.all(timeline.evaluate(t) == 0.0), f"the swarm must be dark at t={t}"


def test_a_look_after_the_blackout_cannot_relight_the_swarm():
    cfg = _linear_cfg()
    late = _look(9.95, (_named(ALL_6, "white"),), (_on(ALL_6),))
    assert np.all(LightingTimeline([late], N6, 10.0, cfg).evaluate(9.96) == 0.0)


def test_evaluate_returns_integral_wrgb_per_drone_and_deck():
    """`_apply_drone_color` asserts integral 0-255 values, so the read-out rounds."""
    cfg = load_lighting_config()
    look = _look(0.0, (_named(ALL_6, "amber"),), (_ramp(ALL_6),))
    out = LightingTimeline([look], N6, 100.0, cfg).evaluate(1.3)
    assert out.shape == (N6, 2, 4)
    assert np.all(out == np.round(out)), "every emitted WRGB must be integral"
    assert np.all(out >= 0.0) and np.all(out <= 255.0)


def test_evaluate_rgb01_folds_white_into_rgb_and_selects_the_deck():
    """The viewer has no white channel, and a marker shows the top deck."""
    cfg = _linear_cfg()
    look = _look(
        0.0, (_named(ALL_6, "red", decks=("top",)), _named(ALL_6, "green", decks=("bot",)))
    )
    timeline = LightingTimeline([look], N6, 100.0, cfg)
    assert timeline.evaluate_rgb01(0.0).shape == (N6, 3)
    assert timeline.evaluate_rgb01(0.0)[0] == pytest.approx([1.0, 0.0, 0.0])
    assert timeline.evaluate_rgb01(0.0, "bot")[0] == pytest.approx([0.0, 1.0, 0.0])
    white = LightingTimeline([_look(0.0, (_named(ALL_6, "white"),))], N6, 100.0, cfg)
    assert white.evaluate_rgb01(0.0)[0] == pytest.approx([1.0, 1.0, 1.0]), "W folds into all three"
