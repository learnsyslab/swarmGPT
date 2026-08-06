"""Unit tests for the LLM-facing lighting primitive vocabulary (spec §10.2).

Companion to ``test_lighting.py``, which covers the engine underneath (selectors, waveforms,
spreads, layers and the timeline). This file covers only the twelve catalogued primitives and the
``build_look`` dispatch that turns an emitted actions array into a ``Look``.
"""

import dataclasses
import logging

import numpy as np
import pytest

from swarm_gpt.core.lighting import LightingTimeline, Look, hue_to_wrgb, load_lighting_config
from swarm_gpt.core.lighting_primitives import LIGHTING_PRIMITIVES, build_look

CFG = load_lighting_config()

# The §9.1 aliasing floor: an effect faster than `col_freq / 2` cannot be represented by the cue
# stream. Derived here the same way the clamp derives it, from the one configured cue rate.
MIN_PERIOD_S = 2.0 / CFG.col_freq

# 120 BPM makes one beat exactly 0.5 s, so every `period_beats` below converts to a round number.
BPM = 120.0

N6 = 6
# Six drones spread along +x, no two sharing a coordinate and none on the x centroid (0.5), so the
# left/right split is unambiguous. z runs 1.0 .. 2.0 evenly, which makes the `z` spread exact.
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

N8 = 8
# Eight drones evenly spaced along +x, so the centroid is 3.5 and `radius` is hand-computable.
POSITIONS_8 = np.stack([np.arange(8.0), np.zeros(8), np.ones(8)], axis=1)

ALL = ("all", ())


def _action(primitive: str, **params: object) -> dict:
    """One entry of the emitted actions array."""
    return {"primitive": primitive, "params": params}


def _build(
    action: dict, positions: np.ndarray = POSITIONS_6, n: int = N6, bpm: float = BPM
) -> Look:
    """Build a single-action look against the six-drone fixture."""
    return build_look([action], 0.0, positions, n, CFG, bpm)


@pytest.fixture
def clamp_log(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> pytest.LogCaptureFixture:
    """`caplog`, wired so it actually sees this module's records.

    The ROS `launch` package — autoloaded here as a pytest plugin via `launch_testing` — calls
    `logging.setLoggerClass`, and its logger class defaults to ``propagate = False``. Every logger
    created after that point therefore bypasses the root logger `caplog` listens on. Forcing
    propagation back on for the one module under test keeps the assertion about the *logger*
    rather than about whichever handler the environment happens to have installed.
    """
    logger = logging.getLogger("swarm_gpt.core.lighting_primitives")
    monkeypatch.setattr(logger, "propagate", True)
    caplog.set_level(logging.WARNING, logger=logger.name)
    return caplog


# --- catalogue -------------------------------------------------------------------------


def test_the_catalogue_holds_exactly_the_twelve_spec_primitives():
    assert set(LIGHTING_PRIMITIVES) == {
        "light_color",
        "gradient",
        "rainbow",
        "light_on",
        "light_off",
        "pulse",
        "blink",
        "strobe_decay",
        "chase",
        "sweep",
        "ripple_light",
        "alternate_blink",
    }


def test_unknown_primitive_raises():
    with pytest.raises(KeyError):
        _build(_action("laser_show", sel=ALL, deck="both"))


def test_build_look_carries_its_start_time():
    look = build_look(
        [_action("light_color", sel=ALL, color="red", deck="both")], 12.5, POSITIONS_6, N6, CFG, BPM
    )
    assert look.t_start == 12.5


def test_deck_choice_maps_onto_the_deck_tuple():
    for deck, expected in (("top", ("top",)), ("bot", ("bot",)), ("both", ("top", "bot"))):
        look = _build(_action("light_color", sel=ALL, color="red", deck=deck))
        assert look.colour_layers[0].decks == expected, deck


def test_unknown_deck_raises():
    with pytest.raises(KeyError):
        _build(_action("light_color", sel=ALL, color="red", deck="middle"))


def test_colour_layers_keep_their_order_in_the_actions_array():
    """§8.2 resolves colour LTP by position in the actions array, so build_look must not reorder."""
    look = build_look(
        [
            _action("light_color", sel=ALL, color="blue", deck="both"),
            _action("pulse", sel=ALL, period_beats=4.0, deck="both"),
            _action("light_color", sel=("even", ()), color="amber", deck="both"),
        ],
        0.0,
        POSITIONS_6,
        N6,
        CFG,
        BPM,
    )
    assert [layer.params["color"] for layer in look.colour_layers] == ["blue", "amber"]
    assert len(look.brightness_layers) == 1


# --- colour: light_color, gradient, rainbow (spec §10.2) -------------------------------


def test_light_color_builds_a_named_colour_layer():
    look = _build(_action("light_color", sel=("even", ()), color="amber", deck="both"))
    assert look.brightness_layers == ()
    (layer,) = look.colour_layers
    assert layer.kind == "named"
    assert layer.params == {"color": "amber"}
    assert list(layer.mask) == [True, False, True, False, True, False]
    assert layer.decks == ("top", "bot")


@pytest.mark.parametrize(
    "params",
    [
        {"primitive": "light_color", "color": "chartreuse"},
        {"primitive": "gradient", "color_a": "chartreuse", "color_b": "red", "by": "x"},
        {"primitive": "gradient", "color_a": "red", "color_b": "chartreuse", "by": "x"},
    ],
)
def test_an_off_palette_colour_is_rejected_when_the_layer_is_built(params: dict):
    """The palette is resolved lazily at read-out, so an unchecked name fails far too late.

    `ColourLayer.evaluate` indexes `cfg.palette` per frame and inside `compile_cues`, which puts a
    bare `KeyError('chartreuse')` mid-render or mid-deploy — long past `response2lighting`'s
    `try/except` and the reprompt loop it feeds.
    """
    params = dict(params)
    with pytest.raises(KeyError, match="chartreuse"):
        _build(_action(params.pop("primitive"), sel=("all", ()), deck="both", **params))


def test_gradient_interpolation_parameter_is_inclusive_so_color_b_is_reached():
    """`s` must hit 1.0 exactly at the far end.

    ``spread_offsets`` is deliberately half-open (it must not collapse the two ends of a sweep onto
    the same phase), so reusing it here would leave `color_b` unreachable — the far drone would sit
    at `lerp(a, b, 5/6)` instead of at `b`.
    """
    look = _build(
        _action("gradient", sel=ALL, color_a="red", color_b="blue", by="index", deck="both")
    )
    (layer,) = look.colour_layers
    assert layer.kind == "gradient"
    assert layer.params["color_a"] == "red"
    assert layer.params["color_b"] == "blue"
    s = layer.params["s"]
    assert s.shape == (N6,)
    assert s == pytest.approx([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    out = layer.evaluate(0.0, CFG)
    assert out[0] == pytest.approx(CFG.palette["red"]), "the near end is exactly color_a"
    assert out[5] == pytest.approx(CFG.palette["blue"]), "the far end is exactly color_b"


def test_gradient_by_index_ranks_within_the_selected_subset():
    """The subset is unevenly spaced on purpose: rank-in-subset and absolute index differ on it.

    `ids(2, 4, 6)` cannot tell the two apart -- evenly spaced ids normalize to the same [0, 0.5, 1]
    either way, so the assertion the name makes would hold against a gradient ranked over the whole
    swarm (spec 11). `ids(1, 2, 6)` ranks to [0, 0.5, 1] but normalizes absolutely to [0, 0.2, 1].
    """
    look = _build(
        _action(
            "gradient",
            sel=("ids", (1, 2, 6)),
            color_a="red",
            color_b="blue",
            by="index",
            deck="top",
        )
    )
    s = look.colour_layers[0].params["s"]
    assert s[[0, 1, 5]] == pytest.approx([0.0, 0.5, 1.0])
    assert s[1] != pytest.approx(0.2), "the absolute drone index would have put drone 2 here"
    assert s[[2, 3, 4]] == pytest.approx([0.0, 0.0, 0.0]), "unselected rows stay at 0"


def test_gradient_by_axis_min_max_normalizes_the_coordinate():
    """x runs -2 .. 3 over the fixture, a span of 5."""
    look = _build(_action("gradient", sel=ALL, color_a="red", color_b="green", by="x", deck="both"))
    assert look.colour_layers[0].params["s"] == pytest.approx([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])


def test_gradient_by_radius_normalizes_distance_from_the_subset_centroid():
    """Distances from the x centroid (3.5) are 0.5 .. 3.5, so the span is 3.0."""
    look = _build(
        _action("gradient", sel=ALL, color_a="red", color_b="blue", by="radius", deck="both"),
        positions=POSITIONS_8,
        n=N8,
    )
    assert look.colour_layers[0].params["s"] == pytest.approx(
        [1.0, 2 / 3, 1 / 3, 0.0, 0.0, 1 / 3, 2 / 3, 1.0]
    )


def test_gradient_over_a_formation_with_no_extent_stays_finite():
    """A flat formation gives `gradient(by="z")` a zero span, which unguarded is 0 / 0.

    Not an exotic case: any planar formation -- which is most of them -- has no extent in z, and
    the resulting `nan` propagates through the whole read-out to `_apply_drone_color`, whose
    `int()` packing and 0-255 assertion are the last thing between it and the radio.
    """
    flat = np.stack([np.arange(6.0), np.zeros(6), np.full(6, 1.5)], axis=1)
    look = _build(
        _action("gradient", sel=ALL, color_a="red", color_b="blue", by="z", deck="both"),
        positions=flat,
    )
    (layer,) = look.colour_layers
    assert layer.params["s"] == pytest.approx(np.zeros(N6))
    out = layer.evaluate(0.0, CFG)
    assert np.all(np.isfinite(out)), "a span-free gradient must not emit nan"
    assert out == pytest.approx(np.tile(CFG.palette["red"], (N6, 1))), "it collapses onto color_a"


def test_gradient_rejects_an_unknown_by_axis():
    with pytest.raises(KeyError):
        _build(
            _action("gradient", sel=ALL, color_a="red", color_b="blue", by="spiral", deck="both")
        )


def test_rainbow_builds_a_cycled_layer_carrying_the_spread_offsets():
    look = _build(_action("rainbow", sel=ALL, period_beats=8.0, spread="index", deck="both"))
    assert look.brightness_layers == ()
    (layer,) = look.colour_layers
    assert layer.kind == "cycled"
    assert layer.params["period_s"] == pytest.approx(4.0), "8 beats at 120 BPM"
    assert layer.params["offsets"] == pytest.approx(np.arange(N6) / N6)


def test_rainbow_takes_the_whole_spread_vocabulary():
    """`none` cycles the swarm in sync; `x` sweeps the spectrum across the stage (§10.2)."""
    synced = _build(_action("rainbow", sel=ALL, period_beats=8.0, spread="none", deck="both"))
    assert synced.colour_layers[0].params["offsets"] == pytest.approx(np.zeros(N6))
    swept = _build(_action("rainbow", sel=ALL, period_beats=8.0, spread="x", deck="both"))
    assert swept.colour_layers[0].params["offsets"] == pytest.approx(np.arange(N6) / N6)


# --- brightness: the nine (spec §10.2) --------------------------------------------------


def test_light_on_builds_a_constant_brightness_layer():
    look = _build(_action("light_on", sel=("first", (2,)), deck="top"))
    assert look.colour_layers == ()
    (layer,) = look.brightness_layers
    assert layer.kind == "constant"
    assert layer.decks == ("top",)
    assert list(layer.mask) == [True, True, False, False, False, False]
    assert layer.offsets.shape == (N6,)
    assert layer.evaluate(7.3) == pytest.approx([1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    assert not look.off_mask.any(), "light_on is a layer, not a mask"


def test_light_off_becomes_an_off_mask_bit_not_a_brightness_layer():
    """§8.3: under a plain `max` a layer contributing 0 is a no-op, so light_off is a kill mask."""
    look = _build(_action("light_off", sel=("odd", ()), deck="both"))
    assert look.brightness_layers == ()
    assert look.colour_layers == ()
    assert look.off_mask.shape == (N6, 2)
    assert list(look.off_mask[:, 0]) == [False, True, False, True, False, True]
    assert list(look.off_mask[:, 1]) == [False, True, False, True, False, True]


def test_light_off_marks_only_the_named_deck():
    look = _build(_action("light_off", sel=ALL, deck="bot"))
    assert not look.off_mask[:, 0].any(), "top is untouched"
    assert look.off_mask[:, 1].all()


def test_pulse_is_a_sine_with_the_whole_group_in_sync():
    look = _build(_action("pulse", sel=ALL, period_beats=2.0, deck="both"))
    (layer,) = look.brightness_layers
    assert layer.kind == "sine"
    assert layer.period_s == pytest.approx(1.0)
    assert layer.offsets == pytest.approx(np.zeros(N6))


def test_blink_is_a_square_carrying_its_own_duty():
    look = _build(_action("blink", sel=ALL, period_beats=1.0, duty=0.25, deck="both"))
    (layer,) = look.brightness_layers
    assert layer.kind == "square"
    assert layer.period_s == pytest.approx(0.5)
    assert layer.duty == pytest.approx(0.25)
    assert layer.offsets == pytest.approx(np.zeros(N6))


def test_strobe_decay_is_a_ramp_flashing_on_the_beat():
    look = _build(_action("strobe_decay", sel=ALL, period_beats=1.0, deck="both"))
    (layer,) = look.brightness_layers
    assert layer.kind == "ramp"
    assert layer.period_s == pytest.approx(0.5)
    assert layer.evaluate(0.0) == pytest.approx(np.ones(N6)), "full on the beat"
    assert layer.evaluate(0.25) == pytest.approx(np.full(N6, 0.5)), "decayed halfway"


def _chase_action(**overrides: object) -> dict:
    """A `chase` action, `spread` included -- §10.2 gives it one and the schema requires it."""
    params = dict(
        sel=ALL, period_beats=4.0, length=2, group_size=1, spread="neighbour", deck="both"
    )
    return _action("chase", **(params | overrides))


def test_chase_is_a_square_running_along_the_neighbour_spread():
    """`chase` used to hardcode `index`, which is spatially scrambled by construction (§7.3).

    POSITIONS_6's y coordinates wander, so it is not a straight line and the walk skips past drone
    2 -- picking it up last, off a long jump back from drone 5. That is the §7.3 seam, and it is the
    accepted cost. An evenly spaced line would rank identically to `index` and so could not tell the
    two spreads apart at all.
    """
    (layer,) = _build(_chase_action()).brightness_layers
    assert layer.kind == "square"
    assert layer.period_s == pytest.approx(2.0)
    assert layer.duty == pytest.approx(2 / N6), "duty = length / n_sel"
    assert layer.offsets == pytest.approx(np.array([0, 1, 5, 2, 3, 4]) / N6)
    assert layer.offsets != pytest.approx(np.arange(N6) / N6), "and not drone id order"


def test_chase_honours_whichever_spread_it_is_given():
    """`index` stays reachable for a choreography that deliberately addresses drones by id (§7.3).

    The engine must read the emitted `spread` rather than pin its own: with `neighbour` recommended
    everywhere the model reads, a `chase` that quietly ignored the parameter would look right on
    every fixture whose walk happens to agree with id order.
    """
    by_walk = _build(_chase_action()).brightness_layers[0]
    by_id = _build(_chase_action(spread="index")).brightness_layers[0]
    assert by_walk.offsets == pytest.approx(np.array([0, 1, 5, 2, 3, 4]) / N6)
    assert by_id.offsets == pytest.approx(np.arange(N6) / N6)
    assert by_id.offsets != pytest.approx(by_walk.offsets)


def test_chase_without_a_spread_fails_loudly():
    """`spread` is required, so a short emission raises rather than flying on a silent default.

    The schema declares it and `_parse_lighting_actions` checks arity, so neither production path
    can reach here missing it -- and a `.get(default)` would therefore be unreachable code masking a
    required field (CLAUDE.md §6.2). `KeyError` is one of the three `Choreographer._build_look`
    turns into an `LLMFormatError`, so a hand-written block reprompts rather than escaping.
    """
    params = {"sel": ALL, "period_beats": 4.0, "length": 2, "group_size": 1, "deck": "both"}
    with pytest.raises(KeyError):
        _build({"primitive": "chase", "params": params})


def test_a_built_look_colours_the_swarm_along_its_own_snapshot():
    """§8.5 end to end, through `build_look` rather than a hand-assembled `Look`.

    The base colour is ranked against the look's snapshot at read-out time, so the snapshot has to
    ride on the `Look` -- resolving the masks and offsets while building it is not enough. Dropping
    it there is invisible to every other test in this file, and the swarm silently falls back to the
    id-ordered wheel this spread exists to undo (§7.3).

    POSITIONS_6 walks as drones 0, 1, 3, 4, 5, 2, so drone 2 takes the *last* hue rather than the
    third: walk order and id order genuinely disagree on this fixture. `first(4)` rather than `all`
    per §11 -- and it changes nothing here, because both the covered and the uncovered drones sit at
    brightness 1.0, which is what makes the read-out the pure base colour.
    """
    look = _build(_action("light_on", sel=("first", (4,)), deck="both"))
    out = LightingTimeline([look], N6, 100.0, CFG).evaluate(3.0)[:, 0]
    for drone, rank in enumerate([0, 1, 5, 2, 3, 4]):
        assert out[drone] == pytest.approx(np.round(hue_to_wrgb(np.array(rank / N6), CFG))), drone


def test_chase_lights_exactly_length_drones_at_any_instant():
    """`duty = length / n_sel` is what makes the running light a window of fixed width (§10.2).

    The period and sample times are chosen so every phase is an exact binary fraction: at 120 BPM
    16 beats is 8 s, and 0.25 s steps put `t / period` on multiples of 1/32 against offsets of 1/8.
    """
    look = _build(_chase_action(period_beats=16.0, length=3), positions=POSITIONS_8, n=N8)
    (layer,) = look.brightness_layers
    assert layer.period_s == pytest.approx(8.0)
    assert layer.duty == pytest.approx(3 / N8)
    for t in np.arange(0.0, 8.0, 0.25):
        assert int(layer.evaluate(float(t)).sum()) == 3, t


def test_chase_duty_is_computed_over_the_selection_not_the_whole_swarm():
    """`duty = length / n_sel` is over the *selected* subset, which `sel=all` cannot distinguish.

    On the full swarm the mask's population and its length agree, so a duty divided by the swarm
    size passes every existing assertion (spec 11). Over `first(4)` of eight they differ by a
    factor of two, and `chase(first(4), length=2)` must light 2 drones at a time, not 1.
    """
    look = _build(
        _chase_action(sel=("first", (4,)), period_beats=16.0), positions=POSITIONS_8, n=N8
    )
    (layer,) = look.brightness_layers
    assert layer.duty == pytest.approx(2 / 4), "over the four selected, not the eight in the swarm"
    # 16 beats is 8 s at 120 BPM, so 0.25 s steps keep every phase an exact binary fraction.
    for t in np.arange(0.0, 8.0, 0.25):
        assert int(layer.evaluate(float(t)).sum()) == 2, t


def test_chase_over_an_empty_selection_builds_instead_of_dividing_by_zero():
    """`ids([])` now raises (§7.1), but an empty selection is still reachable through `right`.

    `_right_mask` splits on a strict `>` against the mean, so a formation with no extent along the
    stage axis puts every drone stage left and none stage right. Every layer is then a no-op, so
    the duty is irrelevant; the floor exists only because `chase` is the one primitive that would
    divide by zero on it. The resulting `ZeroDivisionError` is not one of the `(KeyError,
    ValueError, IndexError)` `Choreographer._build_look` catches, so it would escape the reprompt
    path entirely rather than reporting as a format error.
    """
    no_x_extent = np.stack([np.zeros(N6), np.arange(6.0), np.ones(N6)], axis=1)
    look = _build(_chase_action(sel=("right", ())), positions=no_x_extent)
    (layer,) = look.brightness_layers
    assert not layer.mask.any()
    assert layer.evaluate(1.3) == pytest.approx(np.zeros(N6))


def test_chase_group_size_quantizes_whichever_spread_it_runs_along():
    """The supervisor's "blinking with different group size": the window advances group-by-group.

    POSITIONS_8 is an evenly spaced line, so the `neighbour` walk over it *is* id order and the
    buckets are hand-writable. What the fixture pins is the bucketing, not which order it buckets.
    """
    look = _build(_chase_action(group_size=2), positions=POSITIONS_8, n=N8)
    (layer,) = look.brightness_layers
    assert layer.offsets == pytest.approx([0.0, 0.0, 0.25, 0.25, 0.5, 0.5, 0.75, 0.75])


@pytest.mark.parametrize("group_size", [0, 2])
def test_chase_group_size_is_never_silently_inert(group_size: int):
    """`group_size` is a plain `chase` parameter in §10.2, carrying no spread restriction.

    The validation used to sit inside the ranked-spread branch of `spread_offsets`, so
    `chase(..., group_size=0, spread="x")` built happily while the identical call with
    `spread="neighbour"` raised -- and a `group_size` of 2 on a spatial spread was accepted and
    then ignored. `ValueError` is one of the three `Choreographer._build_look` turns into an
    `LLMFormatError`, so the model is told rather than left to wonder why nothing changed.
    """
    with pytest.raises(ValueError, match="group_size"):
        _build(_chase_action(group_size=group_size, spread="x"))


def test_sweep_uses_the_named_axis_spread():
    look = _build(_action("sweep", sel=ALL, period_beats=4.0, axis="z", deck="both"))
    (layer,) = look.brightness_layers
    assert layer.kind == "square"
    assert layer.duty == pytest.approx(0.5), "sweep takes the default duty (§7.2)"
    # z runs 1.0 .. 2.0 evenly, so the half-open spatial normalization gives exactly rank / n.
    assert layer.offsets == pytest.approx(np.arange(N6) / N6)
    across = _build(_action("sweep", sel=ALL, period_beats=4.0, axis="y", deck="both"))
    assert across.brightness_layers[0].offsets != pytest.approx(layer.offsets)


def test_ripple_light_is_a_sine_over_the_radius_spread():
    """Distances from the centroid are 0.5 .. 3.5; the half-open normalization scales by 7/8."""
    look = _build(
        _action("ripple_light", sel=ALL, period_beats=4.0, deck="both"), positions=POSITIONS_8, n=N8
    )
    (layer,) = look.brightness_layers
    assert layer.kind == "sine"
    assert layer.offsets == pytest.approx([7 / 8, 7 / 12, 7 / 24, 0.0, 0.0, 7 / 24, 7 / 12, 7 / 8])


def test_alternate_blink_maps_by_onto_the_two_alternate_spreads():
    parity = _build(
        _action("alternate_blink", sel=ALL, period_beats=2.0, by="parity", deck="both")
    ).brightness_layers[0]
    side = _build(
        _action("alternate_blink", sel=ALL, period_beats=2.0, by="side", deck="both")
    ).brightness_layers[0]
    assert parity.kind == "square" and side.kind == "square"
    assert parity.offsets == pytest.approx([0.0, 0.5, 0.0, 0.5, 0.0, 0.5])
    # The fixture's x centroid is 0.5, so drones 3-5 are stage right under stage_axis "+x".
    assert side.offsets == pytest.approx([0.0, 0.0, 0.0, 0.5, 0.5, 0.5])


def test_alternate_blink_puts_its_two_halves_in_antiphase():
    """This is why the primitive exists: `blink` has no phase parameter, so two `blink` calls
    cannot express a ping-pong — the half-period offset can only come from a spread (§10.2)."""
    (layer,) = _build(
        _action("alternate_blink", sel=ALL, period_beats=2.0, by="parity", deck="both")
    ).brightness_layers
    assert layer.period_s == pytest.approx(1.0)
    assert layer.evaluate(0.0) == pytest.approx([1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
    assert layer.evaluate(0.5) == pytest.approx([0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
    for t in np.linspace(0.0, 1.0, 21):
        out = layer.evaluate(float(t))
        assert out[0::2] == pytest.approx(1.0 - out[1::2]), f"halves must never coincide at t={t}"


def test_rainbow_and_chase_share_the_spread_offsets():
    """They differ only in the attribute driven — hue versus intensity (§10.2)."""
    rainbow = _build(
        _action("rainbow", sel=ALL, period_beats=8.0, spread="radius", deck="both")
    ).colour_layers[0]
    chase = _build(_chase_action(period_beats=8.0, length=1, spread="radius")).brightness_layers[0]
    assert rainbow.params["offsets"] == pytest.approx(chase.offsets)
    assert rainbow.params["period_s"] == pytest.approx(chase.period_s)


def test_offsets_are_full_swarm_shaped_even_for_a_subset():
    """`BrightnessLayer.evaluate` indexes offsets by absolute drone index, so a subset-shaped
    array would silently phase-shift the wrong drones."""
    look = _build(_chase_action(sel=("ids", (4, 5, 6)), length=1))
    (layer,) = look.brightness_layers
    assert layer.offsets.shape == (N6,)
    assert layer.offsets[[3, 4, 5]] == pytest.approx([0.0, 1 / 3, 2 / 3])
    assert layer.offsets[[0, 1, 2]] == pytest.approx([0.0, 0.0, 0.0])


# --- degenerate formations collapse a spatial spread (spec §7.3) ------------------------

# A horizontal ring, built exactly the way `form_circle` builds one (`motion_primitives.py:473`):
# no extent in z at all, and every drone the same distance from the centre. `cos`/`sin` rather than
# hand-written axis coordinates, because that leaves the radii equal only to within a couple of
# ulps — the case `_normalize_span`'s tolerance exists for, and the one an exactly-equal fixture
# cannot reach.
_RING_ANGLES = np.linspace(0.0, 2.0 * np.pi, N6, endpoint=False)
RING_6 = np.stack(
    [2.0 * np.cos(_RING_ANGLES), 2.0 * np.sin(_RING_ANGLES), np.full(N6, 1.4)], axis=1
)


def _spread_offsets_of(look: Look) -> np.ndarray:
    """The phase offsets of a single-action look's layer, whichever attribute it drives."""
    if look.brightness_layers:
        return look.brightness_layers[0].offsets
    return look.colour_layers[0].params["offsets"]


@pytest.mark.parametrize(
    ("action", "axis"),
    [
        (_action("sweep", sel=ALL, period_beats=4.0, axis="z", deck="both"), "z"),
        (_action("ripple_light", sel=ALL, period_beats=4.0, deck="both"), "radius"),
        (_action("rainbow", sel=ALL, period_beats=4.0, spread="z", deck="both"), "z"),
    ],
)
def test_a_spread_with_no_extent_to_run_along_warns(
    action: dict, axis: str, clamp_log: pytest.LogCaptureFixture
):
    """`sweep(axis="z")` on a planar formation degrades into a synchronised blink, silently.

    Every offset comes out 0, so the "sweep" fires the whole swarm at once — and `ripple_light` on
    a ring is the same thing, since every radius is equal. Both are natural things for a model to
    author over a `form_circle`, and both are legal: the show runs, so this warns rather than
    raising. What it must not do is leave the author with no way to tell a collapsed effect from a
    working one.
    """
    look = _build(action, positions=RING_6, n=N6)
    assert _spread_offsets_of(look) == pytest.approx(np.zeros(N6)), "the collapse being reported"
    records = [r for r in clamp_log.records if r.name == "swarm_gpt.core.lighting_primitives"]
    assert len(records) == 1
    assert records[0].levelno == logging.WARNING
    message = records[0].getMessage()
    assert action["primitive"] in message, "name the primitive"
    assert axis in message, "and the axis it had nothing to run along"


def test_a_spread_with_extent_does_not_warn(clamp_log: pytest.LogCaptureFixture):
    """POSITIONS_6 runs 1.0 .. 2.0 in z, so the same `sweep` has a real gradient to follow."""
    _build(_action("sweep", sel=ALL, period_beats=4.0, axis="z", deck="both"))
    assert not clamp_log.records


def test_a_single_drone_selection_does_not_warn(clamp_log: pytest.LogCaptureFixture):
    """One drone has nothing to spread against by definition, which is not a degenerate formation."""
    _build(_action("sweep", sel=("first", (1,)), period_beats=4.0, axis="z", deck="both"))
    assert not clamp_log.records


def test_gradient_by_radius_on_a_cos_sin_ring_collapses_rather_than_amplifying_float_noise():
    """The colour-axis twin of `_normalize_span`'s tolerance, and the same fixture forces it.

    `form_circle` lays its slots out with `cos`/`sin`, so a ring's radii are equal in exact
    arithmetic and equal only to within a couple of ulps in floats. Testing the span against exact
    zero takes the *non-degenerate* branch there and divides by that rounding noise: `s` comes out
    as [1, 1/3, 0, 0, 0, 1/3] rather than flat, so each drone lands somewhere between `color_a` and
    `color_b` in whatever order the rounding happened to fall -- a per-drone random colour, not the
    one flat colour a span-free axis has to give. Hand-written exactly-equal coordinates cannot
    catch that (they take the degenerate branch either way), so the fixture has to be a real ring,
    and the guard below is what keeps it one.
    """
    radii = np.linalg.norm(RING_6 - RING_6.mean(axis=0), axis=1)
    span = float(radii.max() - radii.min())
    assert 0.0 < span < 1e-12, (
        f"the fixture must be degenerate only to within float noise, got a span of {span}; "
        "an exactly-equal ring passes this test without the tolerance and pins nothing"
    )
    look = _build(
        _action("gradient", sel=ALL, color_a="red", color_b="blue", by="radius", deck="both"),
        positions=RING_6,
    )
    (layer,) = look.colour_layers
    assert layer.params["s"] == pytest.approx(np.zeros(N6))
    assert layer.evaluate(0.0, CFG) == pytest.approx(np.tile(CFG.palette["red"], (N6, 1))), (
        "it collapses onto color_a"
    )


def test_a_gradient_with_no_extent_to_run_along_warns(clamp_log: pytest.LogCaptureFixture):
    """A collapsed gradient paints one flat colour, which no emission distinguishes from intent.

    The same argument `_spread` makes for a collapsed sweep: `gradient(by="radius")` over a ring
    puts every drone on `color_a`, which is `light_color` -- legal, so the show runs, but not what
    was asked for, and nothing in the emission says so.
    """
    _build(
        _action("gradient", sel=ALL, color_a="red", color_b="blue", by="radius", deck="both"),
        positions=RING_6,
    )
    records = [r for r in clamp_log.records if r.name == "swarm_gpt.core.lighting_primitives"]
    assert len(records) == 1
    assert records[0].levelno == logging.WARNING
    message = records[0].getMessage()
    assert "gradient" in message, "name the primitive"
    assert "radius" in message, "and the axis it had nothing to run along"


def test_a_gradient_with_extent_does_not_warn(clamp_log: pytest.LogCaptureFixture):
    """POSITIONS_6 runs 1.0 .. 2.0 in z, so the same `gradient` has a real span to interpolate."""
    _build(_action("gradient", sel=ALL, color_a="red", color_b="blue", by="z", deck="both"))
    assert not clamp_log.records


# --- beats -> seconds, and the Nyquist clamp (spec §9.1) --------------------------------


def test_period_beats_converts_through_the_song_bpm():
    """`period_beats` is beats, not seconds — the same emission is faster at a faster tempo."""
    for bpm, expected in ((120.0, 2.0), (60.0, 4.0), (90.0, 8.0 / 3.0)):
        look = _build(_action("pulse", sel=ALL, period_beats=4.0, deck="both"), bpm=bpm)
        assert look.brightness_layers[0].period_s == pytest.approx(expected), bpm


def test_nyquist_clamp_warns_and_clamps_rather_than_rejecting(clamp_log: pytest.LogCaptureFixture):
    """An effect faster than col_freq / 2 aliases, so it is slowed — never rejected, because one
    over-eager LLM parameter must not fail a whole show (§9.1)."""
    look = _build(_action("blink", sel=ALL, period_beats=0.1, duty=0.5, deck="both"))
    assert look.brightness_layers[0].period_s == pytest.approx(MIN_PERIOD_S), (
        "clamped, not rejected"
    )
    records = [r for r in clamp_log.records if r.name == "swarm_gpt.core.lighting_primitives"]
    assert len(records) == 1
    assert records[0].levelno == logging.WARNING
    message = records[0].getMessage()
    assert "0.050" in message, "the requested period must be reported"
    assert "0.200" in message, "and so must the applied one"


def test_the_clamp_guards_each_drones_lit_window_not_the_period(
    clamp_log: pytest.LogCaptureFixture,
):
    """`chase` divides the period into `length / n_sel`, so the period is the wrong quantity (§9.1).

    `chase(all, 1 beat, length=1)` over eight drones is a 0.5 s period at 120 BPM, comfortably
    above the 0.2 s Nyquist floor — while each drone is lit for 0.0625 s, well under one 0.1 s cue
    tick. Its whole on-interval then fits between two grid ticks and the drone is never lit in the
    compiled cues at all. The condition that has to hold is `period_s >= n_sel / (length x
    col_freq)`, four times the period-only floor here.
    """
    look = _build(_chase_action(period_beats=1.0, length=1), positions=POSITIONS_8, n=N8)
    assert look.brightness_layers[0].period_s == pytest.approx(N8 / CFG.col_freq)
    assert look.brightness_layers[0].period_s > MIN_PERIOD_S, "the Nyquist floor alone lets it pass"
    message = clamp_log.records[0].getMessage()
    assert "0.062" in message, "the requested lit window, which is the quantity being clamped"
    assert "0.100" in message, "and the applied one, which is one cue tick"


def test_the_clamp_never_returns_a_period_below_the_nyquist_floor():
    """A `chase` wide enough to want a sub-Nyquist period still cannot have one (§9.1).

    `length = 8` of eight is a duty of 1.0, whose lit-window floor is a single tick — but a
    waveform sampled fewer than twice per period aliases whatever its duty, so the two floors are a
    `max`, not a replacement.
    """
    look = _build(
        _chase_action(period_beats=0.1, length=N8), positions=POSITIONS_8, n=N8
    ).brightness_layers[0]
    assert look.period_s == pytest.approx(MIN_PERIOD_S)


def test_an_author_set_blink_duty_is_not_clamped_against_its_lit_window(
    clamp_log: pytest.LogCaptureFixture,
):
    """`blink` runs at `spread="none"`, where no drone can be skipped, so it keeps the Nyquist floor.

    Every offset is 0, so the whole selection shares one phase: a short `duty` coarsens the flash
    for all of them together rather than dropping it from some, which is the failure the lit-window
    floor exists to prevent. Clamping `blink` against `duty` too would slow a legal 0.1-duty stab
    by a factor of five for no benefit (§9.1).
    """
    look = _build(_action("blink", sel=ALL, period_beats=1.0, duty=0.1, deck="both"))
    assert look.brightness_layers[0].period_s == pytest.approx(0.5)
    assert look.brightness_layers[0].duty == pytest.approx(0.1)
    assert not clamp_log.records


def test_the_nyquist_clamp_also_covers_rainbow():
    """`rainbow` drives hue off the same period, so it aliases the same way."""
    look = _build(_action("rainbow", sel=ALL, period_beats=0.01, spread="none", deck="both"))
    assert look.colour_layers[0].params["period_s"] == pytest.approx(MIN_PERIOD_S)


def test_a_legal_period_is_left_alone(clamp_log: pytest.LogCaptureFixture):
    look = _build(_action("pulse", sel=ALL, period_beats=1.0, deck="both"))
    assert look.brightness_layers[0].period_s == pytest.approx(0.5)
    assert [r for r in clamp_log.records if r.name == "swarm_gpt.core.lighting_primitives"] == []


def test_the_clamp_floor_tracks_the_configured_cue_rate(clamp_log: pytest.LogCaptureFixture):
    """The floor is `2 / col_freq`, read from the one place the cue rate is written down.

    A second hardcoded copy would silently mis-tune the moment `DroneSwarm.col_freq` moved: raise
    the rate and the clamp throttles effects for no reason, lower it and the clamp stops guarding
    against the §3.1 drift at all. The same emission must therefore survive at 20 Hz and be clamped
    at 10 Hz, with the warning firing only in the second case.
    """
    # 0.3 beats is 0.15 s at 120 BPM: above the 0.1 s floor at 20 Hz, below the 0.2 s floor at 10.
    action = _action("blink", sel=ALL, period_beats=0.3, duty=0.5, deck="both")
    faster = dataclasses.replace(CFG, col_freq=20.0)
    at_20hz = build_look([action], 0.0, POSITIONS_6, N6, faster, BPM)
    assert at_20hz.brightness_layers[0].period_s == pytest.approx(0.15), "legal at 20 Hz"
    assert not clamp_log.records, "and so must not warn"
    at_10hz = build_look([action], 0.0, POSITIONS_6, N6, CFG, BPM)
    assert at_10hz.brightness_layers[0].period_s == pytest.approx(0.2), "clamped at 10 Hz"
    assert len(clamp_log.records) == 1
