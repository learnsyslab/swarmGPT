"""Unit tests for the hardware cue read-out (spec §9.1, and §3.1 for the failure it prevents)."""

import dataclasses

import numpy as np
import pytest

from swarm_gpt.core.lighting import (
    LightingConfig,
    LightingTimeline,
    hue_to_wrgb,
    load_lighting_config,
)
from swarm_gpt.core.lighting_compile import compile_cues
from swarm_gpt.core.lighting_primitives import build_look

CFG = load_lighting_config()

BPM = 120.0
N6 = 6
POSITIONS_6 = np.stack([np.arange(6.0), np.zeros(6), np.ones(6)], axis=1)
URIS_6 = [f"radio://0/80/2M/E7E7E7E70{i}" for i in range(N6)]

N10 = 10
# Ten drones, which is the swarm the §9.1 chase measurements were taken on. The ids are scrambled
# across the line the way `_assign_positions` scrambles them, so the `neighbour` walk that orders
# the chase disagrees with id order and a per-drone assertion cannot pass by accident.
LINE_10 = np.stack([np.arange(10.0), np.zeros(10), np.ones(10)], axis=1)
POSITIONS_10 = LINE_10[[4, 9, 0, 6, 2, 8, 1, 7, 3, 5]]
URIS_10 = [f"radio://0/80/2M/E7E7E7E7{i:02d}" for i in range(N10)]

# `DroneSwarm.col_freq` defaults to 10 Hz (drone_swarm.py:48) and caps cue consumption.
COL_FREQ = 10.0

# How far before the end of the show the unconditional blackout lands (§8.7).
BLACKOUT_LEAD_S = 0.1

ALL = ("all", ())

# Float slack for comparing a decimal cue grid: 0.1 is not representable in binary, so consecutive
# differences of k / 10.0 land ~4e-16 below 0.1. Eight orders of magnitude below the tolerance used
# here, and eight above it lies the period itself -- this is float noise, not semantic slack.
GRID_SLACK = 1e-9


def _action(primitive: str, **params: object) -> dict:
    """One entry of the emitted actions array."""
    return {"primitive": primitive, "params": params}


def _timeline(actions: list[dict], t_end: float, cfg: LightingConfig = CFG) -> LightingTimeline:
    """A one-look timeline over the six-drone fixture, the look starting at t = 0."""
    return LightingTimeline([build_look(actions, 0.0, POSITIONS_6, N6, cfg, BPM)], N6, t_end, cfg)


def _lit_grid_samples(cues: dict[float, np.ndarray], t_end: float) -> int:
    """Count the grid ticks on which a drone-deck's compiled cue stream is not dark.

    The cues are step events under zero-order hold, so what the hardware and the browser actually
    show at a tick is the last cue at or before it — which is what this replays. The grid is
    rebuilt the way `_sample_times` builds it, ``k / col_freq`` rather than ``k * (1 / col_freq)``,
    so the query times are bit-identical to the ones that were compiled.
    """
    times = np.arange(int(round((t_end - BLACKOUT_LEAD_S) * COL_FREQ))) / COL_FREQ
    cue_times = np.array(sorted(cues))
    holding = np.searchsorted(cue_times, times, "right") - 1
    held = np.stack([cues[float(cue_times[k])] for k in holding])
    return int(np.count_nonzero(held.any(axis=1)))


# --- §3.1: the cue-drift regression -----------------------------------------------------


def test_minimum_cue_spacing_is_never_denser_than_col_freq():
    """The §3.1 regression test, and the sharpest constraint in the design.

    `_stream_reference` consumes at most one cue per deck per `1 / col_freq` tick, in order, and
    never drops. A denser cue list therefore plays back *slowed*, and the lag accumulates for the
    remainder of the show — the lights desynchronize from the music permanently rather than
    glitching once. Uniform sampling at exactly `col_freq` makes that structurally impossible, so
    this holds for every drone, every deck, and the fastest effects the vocabulary can express.

    The `t_end` values matter: the §8.7 blackout cue is emitted at `t_end - 0.1` whatever the grid
    does, so a show whose end does *not* fall on a `col_freq` tick is exactly where an unguarded
    implementation crowds that final cue up against the tick before it. An on-grid `t_end` hides
    that, because the two land on the same time and collapse into one dict key.
    """
    # `period_beats = 0.05` is 0.025 s at 120 BPM, so `rainbow` and `blink` are both clamped to the
    # 0.2 s Nyquist floor: the show does contain the fastest legal effect. `sweep` sits just above
    # it on purpose, and `chase` is held at 0.3 s by the §9.1 lit-window floor its `length = 2` of
    # six implies. A floor-period square wave lands on exactly two samples per period,
    # which aliases into a *static* lit/unlit pattern — a legitimate compile result, but one that
    # leaves some drone-deck permanently dark and so makes its spacing assertion vacuous.
    fastest = [
        _action("rainbow", sel=ALL, period_beats=0.05, spread="index", deck="both"),
        _action(
            "chase",
            sel=ALL,
            period_beats=0.5,
            length=2,
            group_size=1,
            spread="neighbour",
            deck="both",
        ),
        _action("blink", sel=("odd", ()), period_beats=0.05, duty=0.5, deck="top"),
        _action("sweep", sel=ALL, period_beats=0.7, axis="x", deck="bot"),
    ]
    period = 1.0 / COL_FREQ
    for t_end in (30.0, 30.05, 27.37):
        top, bot = compile_cues(_timeline(fastest, t_end), URIS_6, COL_FREQ, t_end)
        for deck, cues in (("top", top), ("bot", bot)):
            for uri, deck_cues in cues.items():
                assert len(deck_cues) > 150, f"{deck} {uri} must be dense, not trivially deduped"
                gaps = np.diff(sorted(deck_cues))
                assert gaps.min() >= period - GRID_SLACK, f"{deck} {uri} at t_end={t_end}"


# --- §9.1: the clamp guards the lit window, not the period -------------------------------


@pytest.mark.parametrize("bpm", [120.0, 120.0000001])
def test_a_chase_lights_every_drone_on_the_compile_grid(bpm: float):
    """The quantity that has to survive the sampling grid is each drone's lit window (§9.1).

    A `chase` spreads the phases evenly, so one drone's on-interval is `period_s x length / n_sel`
    — a tenth of the period at `length = 1` on ten drones. Clamping the *period* to `2 / col_freq`
    leaves that window at 0.05 s, half a grid tick, so a drone's whole on-interval can fall between
    two ticks and it is never lit in the compiled cues. Continuously-sampled `evaluate` gives all
    ten equal on-time, so the MuJoCo preview looks right while the hardware and the browser are
    broken — the preview-versus-flight divergence §5 exists to prevent.

    The second tempo is the same emission a ten-millionth of a BPM away. Under the period-only
    clamp it flips which drones are dropped (measured: five of ten never light at all), because
    ticks and phase offsets are exact multiples of the same rational and the last bit decides.
    Under the correct clamp both tempos land on the same 1.0 s period and neither drops anyone.

    The evenness bound is deliberately loose. §9.1 clamps the lit window to *one* tick, not two, so
    a window straddles a tick boundary on some periods and not others: measured on-times run 71-94
    grid samples where the continuous on-time is 6.01 s for all ten. That ±1 tick per period is
    inherent to the boundary the spec sets. A drone getting a quarter of another's on-time, which
    is what the period-only clamp produces, is not.
    """
    t_end = 60.1  # so the grid is exactly 0.0 .. 59.9 plus the blackout at 60.0
    action = _action(
        "chase", sel=ALL, period_beats=1.0, length=1, group_size=1, spread="neighbour", deck="both"
    )
    look = build_look([action], 0.0, POSITIONS_10, N10, CFG, bpm)
    timeline = LightingTimeline([look], N10, t_end, CFG)
    top, bot = compile_cues(timeline, URIS_10, COL_FREQ, t_end)
    for deck, cues in (("top", top), ("bot", bot)):
        lit = np.array([_lit_grid_samples(cues[uri], t_end) for uri in URIS_10])
        assert lit.min() > 0, f"{deck} leaves drones {np.flatnonzero(lit == 0)} dark all show"
        assert lit.max() - lit.min() <= 0.5 * lit.mean(), f"{deck} on-time is uneven: {lit}"


def test_a_chase_period_the_clamp_stretches_still_lights_every_drone():
    """The case where the old clamp fired and still produced a broken show (§9.1).

    `chase(all, 0.25 beats, length=1)` is 0.125 s at 120 BPM, so the period-only clamp *did* fire
    and logged "clamping to 0.200 s" — as though the effect were now representable. It was not:
    measured over ten drones, six of them never lit at all.
    """
    t_end = 60.1
    action = _action(
        "chase", sel=ALL, period_beats=0.25, length=1, group_size=1, spread="neighbour", deck="both"
    )
    look = build_look([action], 0.0, POSITIONS_10, N10, CFG, BPM)
    top, _ = compile_cues(LightingTimeline([look], N10, t_end, CFG), URIS_10, COL_FREQ, t_end)
    lit = np.array([_lit_grid_samples(top[uri], t_end) for uri in URIS_10])
    assert lit.min() > 0, f"drones {np.flatnonzero(lit == 0)} are dark all show"
    assert lit.max() - lit.min() <= 0.5 * lit.mean(), f"uneven on-time: {lit}"


# --- §9.1: dedup is what makes the design cheap -----------------------------------------


def test_a_static_look_dedups_to_one_cue_plus_the_terminal_blackout():
    """Dedup means a static look costs ~1 cue, not `10 x duration` (§9.1)."""
    t_end = 60.0
    action = _action("light_color", sel=ALL, color="teal", deck="both")
    top, bot = compile_cues(_timeline([action], t_end), URIS_6, COL_FREQ, t_end)
    for cues in (top, bot):
        for uri in URIS_6:
            times = sorted(cues[uri])
            assert len(times) == 2, "one content cue, then the §8.7 blackout"
            assert times[0] == 0.0
            assert times[1] == pytest.approx(t_end - BLACKOUT_LEAD_S)
            assert cues[uri][times[0]] == pytest.approx(np.round(CFG.palette["teal"]))
            assert cues[uri][times[1]] == pytest.approx(np.zeros(4))


def test_a_gradient_look_dedups_like_a_static_one():
    """`gradient` is time-invariant, so it costs exactly what a named colour costs (§9.1)."""
    t_end = 60.0
    action = _action("gradient", sel=ALL, color_a="red", color_b="blue", by="index", deck="both")
    top, _ = compile_cues(_timeline([action], t_end), URIS_6, COL_FREQ, t_end)
    for uri in URIS_6:
        assert len(top[uri]) == 2
    near, far = top[URIS_6[0]][0.0], top[URIS_6[5]][0.0]
    assert near == pytest.approx(np.round(CFG.palette["red"])), "the near end is exactly color_a"
    assert far == pytest.approx(np.round(CFG.palette["blue"])), "the far end is exactly color_b"


def test_rainbow_cue_count_tracks_hue_steps_over_the_period():
    """`hue_steps` is what keeps a continuously advancing hue from defeating dedup.

    A continuously advancing hue changes on every tick, so nothing collapses; quantizing to
    `hue_steps` makes it piecewise-constant and brings the rate to `min(col_freq, hue_steps /
    period)`. At 24 steps over a 12 s cycle that is 2 Hz, a fifth of the 10 Hz ceiling — the
    property that keeps the radio budget viable (§9.1, §12.1).
    """
    period_s = 12.0
    t_end = period_s + BLACKOUT_LEAD_S  # the sample grid then covers exactly one cycle
    action = _action(
        "rainbow", sel=ALL, period_beats=period_s * BPM / 60.0, spread="none", deck="both"
    )
    top, bot = compile_cues(_timeline([action], t_end), URIS_6, COL_FREQ, t_end)
    rate = min(COL_FREQ, CFG.hue_steps / period_s)
    expected = int(rate * period_s) + 1  # one cue per hue step, plus the blackout
    for cues in (top, bot):
        for uri in URIS_6:
            assert len(cues[uri]) == expected
    assert expected < COL_FREQ * period_s / 4, "and far below the undeduped ceiling"


def test_brightness_steps_collapses_a_slow_brightness_waveform():
    """`brightness_steps` is to `sine`/`ramp` what `hue_steps` is to `rainbow` (§9.1).

    Unquantized, a 16 s breathe changes on nearly every tick and compiles to most of the undeduped
    ceiling. Bucketing the merged brightness makes it piecewise-constant, so dedup collapses the
    runs and the rate drops to roughly two buckets' worth of edges per period.
    """
    t_end = 60.0
    action = _action("pulse", sel=ALL, period_beats=32.0, deck="both")
    # 255 buckets is one per addressable output level, i.e. the unquantized behaviour.
    unquantized = dataclasses.replace(CFG, brightness_steps=255)
    raw, _ = compile_cues(_timeline([action], t_end, unquantized), URIS_6, COL_FREQ, t_end)
    quantized, _ = compile_cues(_timeline([action], t_end), URIS_6, COL_FREQ, t_end)
    for uri in URIS_6:
        assert len(raw[uri]) > 0.5 * COL_FREQ * t_end, "the cost this is mitigating"
        assert len(quantized[uri]) < len(raw[uri]) / 3


def test_square_wave_primitives_are_unaffected_by_brightness_quantization():
    """`blink`, `chase` and `sweep` never paid the cost: `square` is already piecewise-constant."""
    t_end = 60.0
    cases = [
        _action("blink", sel=ALL, period_beats=2.0, duty=0.5, deck="both"),
        _action(
            "chase",
            sel=ALL,
            period_beats=4.0,
            length=3,
            group_size=1,
            spread="neighbour",
            deck="both",
        ),
        _action("sweep", sel=ALL, period_beats=4.0, axis="x", deck="both"),
    ]
    unquantized = dataclasses.replace(CFG, brightness_steps=255)
    for action in cases:
        raw, _ = compile_cues(_timeline([action], t_end, unquantized), URIS_6, COL_FREQ, t_end)
        quantized, _ = compile_cues(_timeline([action], t_end), URIS_6, COL_FREQ, t_end)
        for uri in URIS_6:
            assert sorted(raw[uri]) == sorted(quantized[uri]), action["primitive"]
            for t, wrgb in raw[uri].items():
                assert quantized[uri][t] == pytest.approx(wrgb), action["primitive"]


def test_decks_compile_independently():
    """A top-only effect leaves bot deduped down to its base colour plus the blackout (§8.6)."""
    t_end = 20.0
    actions = [
        _action("light_color", sel=ALL, color="red", deck="both"),
        _action("blink", sel=ALL, period_beats=1.0, duty=0.5, deck="top"),
    ]
    top, bot = compile_cues(_timeline(actions, t_end), URIS_6, COL_FREQ, t_end)
    assert len(bot[URIS_6[0]]) == 2
    assert len(top[URIS_6[0]]) > 20


# --- §8.7: the terminal blackout --------------------------------------------------------


def test_the_terminal_blackout_cue_is_emitted_explicitly():
    """The timeline implements the blackout as an early return, which guarantees zeros *from*
    `t_end - 0.1` but does not put a sample there: a uniform grid anchored at 0 lands on that
    instant only by luck, so `compile_cues` must emit it itself (§8.7)."""
    t_end = 12.34  # deliberately off the 10 Hz grid
    action = _action("light_on", sel=ALL, deck="both")
    top, bot = compile_cues(_timeline([action], t_end), URIS_6, COL_FREQ, t_end)
    blackout = t_end - BLACKOUT_LEAD_S
    for cues in (top, bot):
        for uri in URIS_6:
            assert blackout in cues[uri], "the drones must never land lit"
            assert cues[uri][blackout] == pytest.approx(np.zeros(4))
            assert max(cues[uri]) == blackout, "and nothing may be emitted after it"


# --- the interface the deploy path consumes ---------------------------------------------


def test_cue_dicts_are_keyed_by_uri_for_both_decks():
    """The two returned dicts drop straight into `execute_choreography(color_top=, color_bot=)`."""
    t_end = 10.0
    top, bot = compile_cues(LightingTimeline([], N6, t_end, CFG), URIS_6, COL_FREQ, t_end)
    assert list(top) == URIS_6
    assert list(bot) == URIS_6


def test_every_compiled_wrgb_is_integral_and_in_range():
    """`_apply_drone_color` asserts 0-255 and packs the channels with `int()` (drone_swarm.py:553)."""
    t_end = 20.0
    actions = [
        _action("rainbow", sel=ALL, period_beats=4.0, spread="index", deck="both"),
        _action("pulse", sel=ALL, period_beats=2.0, deck="both"),
    ]
    top, bot = compile_cues(_timeline(actions, t_end), URIS_6, COL_FREQ, t_end)
    for cues in (top, bot):
        for deck_cues in cues.values():
            for wrgb in deck_cues.values():
                assert wrgb.shape == (4,)
                assert np.all(wrgb == np.round(wrgb)), f"non-integral WRGB {wrgb}"
                assert np.all(wrgb >= 0.0) and np.all(wrgb <= 255.0)


def test_a_lighting_less_show_compiles_to_todays_static_cue_structure():
    """The §8.5 failure-safe property, end to end: an emission carrying no lighting at all
    reproduces the two-cue-per-drone stub the deploy path uses today (backend.py:330-337)."""
    t_end = 45.0
    top, bot = compile_cues(LightingTimeline([], N6, t_end, CFG), URIS_6, COL_FREQ, t_end)
    for cues in (top, bot):
        for i, uri in enumerate(URIS_6):
            times = sorted(cues[uri])
            assert len(times) == 2
            assert times[0] == 0.0
            assert times[1] == pytest.approx(t_end - BLACKOUT_LEAD_S)
            assert cues[uri][times[0]] == pytest.approx(
                np.round(hue_to_wrgb(np.array(i / N6), CFG))
            )
            assert cues[uri][times[1]] == pytest.approx(np.zeros(4))


@pytest.mark.parametrize("t_end", [0.0, 0.05, 0.15, 0.19])
def test_a_show_too_short_for_the_cue_grid_raises(t_end: float):
    """Below `1 / col_freq + 0.1` the grid cannot open at 0 and can open before it (§9.3).

    The blackout at `t_end - 0.1` is appended unconditionally, and grid ticks it would crowd are
    dropped first, so a show shorter than one tick plus the blackout lead loses every tick and
    keeps only the blackout: `t_end = 0.15` compiles to a single cue at 0.05 and `t_end = 0.05` to
    a single cue at **-0.05**. The §9.3 payload contract says every cue list is non-empty and
    starts at t = 0, and a negative time would be handed straight to `DroneSwarm`.

    Unreachable through `deploy` — `response2waypoints` appends return-to-home legs, so a real
    flight is minutes long — but `compile_cues` is a public entry point, and this is the one input
    on which it silently produces a payload no consumer can honour.
    """
    with pytest.raises(ValueError, match="too short"):
        compile_cues(LightingTimeline([], N6, t_end, CFG), URIS_6, COL_FREQ, t_end)


def test_the_shortest_compilable_show_still_opens_at_zero():
    """One tick plus the blackout lead is the boundary, and it must be on the legal side of it."""
    t_end = 1.0 / COL_FREQ + BLACKOUT_LEAD_S
    top, bot = compile_cues(LightingTimeline([], N6, t_end, CFG), URIS_6, COL_FREQ, t_end)
    for cues in (top, bot):
        for uri in URIS_6:
            times = sorted(cues[uri])
            assert times[0] == 0.0, "the §9.3 contract's initial colour has to be defined"
            assert times[-1] == pytest.approx(t_end - BLACKOUT_LEAD_S)


def test_the_short_show_guard_tracks_the_configured_cue_rate():
    """The boundary is one cue tick, not a hardcoded 0.2 s: `col_freq` is the source of truth."""
    t_end = 0.16
    faster = dataclasses.replace(CFG, col_freq=20.0)
    top, _ = compile_cues(LightingTimeline([], N6, t_end, faster), URIS_6, 20.0, t_end)
    assert sorted(top[URIS_6[0]])[0] == 0.0, "legal at 20 Hz, where a tick is 0.05 s"
    with pytest.raises(ValueError, match="too short"):
        compile_cues(LightingTimeline([], N6, t_end, CFG), URIS_6, COL_FREQ, t_end)


def test_a_uri_list_that_does_not_cover_the_swarm_raises():
    """Silently zipping short would leave the uncovered drones dark for a whole show."""
    t_end = 10.0
    with pytest.raises(ValueError):
        compile_cues(LightingTimeline([], N6, t_end, CFG), URIS_6[:3], COL_FREQ, t_end)
