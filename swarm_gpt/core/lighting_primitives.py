"""The LLM-facing lighting primitive vocabulary: the twelve primitives of spec §10.2.

`lighting.py` holds the engine — selectors, waveforms, phase spreads, colour sources and the merge
rules. This module holds the *names the LLM may say* and the mapping from each name's parameters
onto that engine, which is deliberately thin: `chase` and `sweep` are both "square wave plus a
spread", and `rainbow` and `chase` differ only in whether the shared spread drives hue or intensity
(§4, §10.2).

Spec: ``docs/specs/2026-08-05-lighting-primitives-design.md``. Like `lighting.py` this is pure NumPy
plus stdlib; it imports neither the backend, the simulator nor JAX. The three private names it does
import from `lighting` — ``_DECKS``, ``_SPAN_REL_TOL`` and ``_SPREAD_AXES`` — are shared constants,
and restating them here would give the deck order, the no-extent tolerance and the axis mapping two
sources of truth.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from swarm_gpt.core.lighting import (
    _DECKS,
    _SPAN_REL_TOL,
    _SPREAD_AXES,
    BrightnessLayer,
    ColourLayer,
    Look,
    select,
    spread_offsets,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

    from swarm_gpt.core.lighting import LightingConfig

    # A primitive builder. `light_off` is the one that returns no layer: §8.3 makes it a
    # post-reduction kill mask, which `build_look` collects into `Look.off_mask` instead.
    _Builder = Callable[[dict, "_BuildContext"], ColourLayer | BrightnessLayer | None]

logger = logging.getLogger(__name__)

# The `deck` every primitive takes (§8.6), mapped onto the decks its layer covers.
_DECK_CHOICES = {"top": ("top",), "bot": ("bot",), "both": _DECKS}

# `alternate_blink`'s `by`, mapped onto the §7.3 spread that splits the group into two halves half
# a period apart. The spec writes these `alternate(parity)` and `alternate(side)`.
_ALTERNATE_SPREADS = {"parity": "alternate_parity", "side": "alternate_side"}

# Waveform duty for every brightness primitive that does not set its own (§7.2).
_DEFAULT_DUTY = 0.5

# The spreads that read the snapshot's geometry, and so are the ones a formation can leave with
# nothing to run along (§7.3). `neighbour` is not among them: a walk ranks whatever it is given.
# `gradient`'s measured `by` axes are the same four, and are checked against this for the same
# reason -- the other one, `index`, ranks rather than measures (§7.5).
_SPATIAL_SPREADS = ("radius", *_SPREAD_AXES)


@dataclass(frozen=True)
class _BuildContext:
    """Everything a primitive builder reads besides its own parameters.

    Bundled so all twelve builders share one signature and can live in a flat dispatch table. Most
    builders ignore most of it — `light_color` needs only the mask and the decks.

    Attributes:
        primitive: The §10.2 name this action was emitted under, for diagnostics.
        mask: (n,) boolean mask the action's ``sel`` resolved to.
        decks: The decks the action's ``deck`` resolved to, a subset of ("top", "bot").
        positions: (n, 3) position snapshot, frozen at the look's start time (§7.3).
        cfg: Lighting config, for the stage axis the spatial spreads read.
        bpm: Song tempo in beats per minute, which converts `period_beats` into seconds.
    """

    primitive: str
    mask: NDArray
    decks: tuple[str, ...]
    positions: NDArray
    cfg: LightingConfig
    bpm: float


def _period_seconds(
    period_beats: float, ctx: _BuildContext, lit_fraction: float = _DEFAULT_DUTY
) -> float:
    """Convert an LLM-facing beat period into seconds, held off the aliasing floor (§9.1).

    **The clamped quantity is the lit window, not the period.** What has to survive the sampling
    grid is the shortest feature the effect asks a single drone to show: ``period_s x
    lit_fraction``. Below one ``1 / col_freq`` tick that window can fall entirely between two grid
    ticks, and the drone is never lit in the compiled cues at all — not merely quantized. Clamping
    the period alone guards the right quantity only at ``lit_fraction = 0.5``, which is where every
    primitive but `chase` sits; `chase` divides the period into ``length / n_sel`` and can drop
    whole drones while passing a period-only floor by a factor of five (§9.1).

    The clamp never returns a period below ``2 / col_freq`` either: a waveform sampled fewer than
    twice per period aliases whatever its duty, which is the Nyquist bound the two floors coincide
    on at ``lit_fraction = 0.5``.

    Over-fast effects are *clamped* rather than rejected: one over-eager LLM parameter must not
    fail a whole show. Both floors are derived from ``cfg.col_freq`` rather than hardcoded, so they
    track whatever rate the show is actually compiled and flown at — the same number `compile_cues`
    samples on and `DroneSwarm` drains at.

    Args:
        period_beats: Effect period in beats, as emitted.
        ctx: The action's build context, for the song tempo and the configured cue rate.
        lit_fraction: Fraction of each period a single drone must be distinguishable for. The
            default is the half period every un-spread primitive shows; only `chase` narrows it.

    Returns:
        The period in seconds, never so short that ``period_s x lit_fraction`` falls below one cue
        tick, and never below ``2 / cfg.col_freq``.
    """
    tick_s = 1.0 / ctx.cfg.col_freq
    min_period_s = max(2.0 * tick_s, tick_s / lit_fraction)
    period_s = float(period_beats) * 60.0 / ctx.bpm
    if period_s >= min_period_s:
        return period_s
    logger.warning(
        "Lighting period_beats=%g is %.3f s at %.1f BPM, leaving each drone lit for %.3f s — "
        "below the %.3f s cue tick at %.1f Hz, which drops the effect from some drones entirely "
        "rather than merely coarsening it. Clamping that window up to %.3f s, which stretches the "
        "period to %.3f s (%.3f beats).",
        period_beats,
        period_s,
        ctx.bpm,
        period_s * lit_fraction,
        tick_s,
        ctx.cfg.col_freq,
        min_period_s * lit_fraction,
        min_period_s,
        min_period_s * ctx.bpm / 60.0,
    )
    return min_period_s


def _gradient_s(by: str, mask: NDArray, positions: NDArray) -> NDArray:
    """Compute `gradient`'s interpolation parameter along ``by`` (§7.5).

    Min-max normalized over the selected subset onto the **inclusive** [0, 1], which is what makes
    the far drone reproduce ``color_b`` exactly. This is deliberately *not* `spread_offsets`: that
    is half-open so the two ends of a sweep never share a phase, and reusing it here would leave
    ``color_b`` unreachable.

    The no-extent test is `_normalize_span`'s, on the colour axis instead of the phase axis, and
    for the same reason: a ring's radii are equal only to within an ulp or two, so an exact-zero
    test divides by that rounding noise and hands every drone a different point along the ramp
    rather than one flat colour. `_SPAN_REL_TOL` is imported rather than restated so the two axes
    agree on what "no extent" means.

    Args:
        by: One of "index", "x", "y", "z" or "radius".
        mask: (n,) boolean mask of the selected drones.
        positions: (n, 3) frozen position snapshot.

    Returns:
        (n,) values in [0, 1]. Unselected drones are 0, as is every drone when the subset has no
        extent along ``by``.

    Raises:
        KeyError: If ``by`` is not one of the five axes.
    """
    s = np.zeros(mask.shape[0])
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return s
    if by == "index":
        values = np.arange(idx.size, dtype=float)
    elif by == "radius":
        values = np.linalg.norm(positions[idx] - positions[idx].mean(axis=0), axis=1)
    elif by in _SPREAD_AXES:
        values = positions[idx, _SPREAD_AXES[by]].astype(float)
    else:
        raise KeyError(f"Unknown gradient axis {by}")
    span = values.max() - values.min()
    if span > _SPAN_REL_TOL * np.abs(values).max():
        s[idx] = (values - values.min()) / span
    return s


def _spread(ctx: _BuildContext, kind: str, group_size: int = 1) -> NDArray:
    """Resolve a phase spread against the frozen snapshot, reporting one it collapses on (§7.3).

    A spatial spread normalizes by the selection's extent along its axis, so a selection with no
    extent there gets every offset 0 and the effect degrades into a synchronised blink: `sweep`
    with ``axis="z"`` over any planar formation — every `form_circle` and `form_star` output is
    planar — and `ripple_light` over a ring, where every radius is equal. Both are natural things
    to author and both are legal, so the show runs; what it must not do is run silently, since a
    collapsed effect and a working one are indistinguishable from the emission alone.

    A single-drone selection is excluded: it has nothing to spread against by definition, and that
    is not a degenerate formation.

    Args:
        ctx: The action's build context, which carries the emitting primitive's name.
        kind: The §7.3 spread name that sets the per-drone phase offsets.
        group_size: Drones per phase bucket; quantizes the ranked spreads only.

    Returns:
        (n,) offsets in turns, full-swarm-shaped.
    """
    offsets = spread_offsets(kind, ctx.mask, ctx.positions, group_size, ctx.cfg)
    if kind in _SPATIAL_SPREADS and ctx.mask.sum() > 1 and not offsets[ctx.mask].any():
        logger.warning(
            "Lighting %s covers drones with no extent along %s, so every phase offset is 0 and it "
            "fires as one synchronised flash instead of travelling. It needs a formation spread "
            "out along that axis.",
            ctx.primitive,
            kind,
        )
    return offsets


def _brightness(
    ctx: _BuildContext, kind: str, period_s: float, duty: float, spread: str, group_size: int
) -> BrightnessLayer:
    """Assemble a brightness layer, resolving its phase spread against the frozen snapshot.

    Args:
        ctx: The action's build context.
        kind: "constant", or one of the §7.2 waveform names.
        period_s: Waveform period in seconds. Unused by "constant".
        duty: Fraction of each period a "square" waveform stays on.
        spread: The §7.3 spread name that sets the per-drone phase offsets.
        group_size: Drones per phase bucket; quantizes the "index" spread only.

    Returns:
        The layer, with full-swarm-shaped (n,) offsets.
    """
    offsets = _spread(ctx, spread, group_size)
    return BrightnessLayer(ctx.mask, ctx.decks, kind, period_s, duty, offsets)


def _palette_colour(name: str, cfg: LightingConfig) -> str:
    """Check a colour name against the palette and return it (§7.5).

    `ColourLayer` resolves palette names lazily, at every read-out, so an unchecked name would
    surface as a bare ``KeyError`` per frame mid-render or inside `compile_cues` during a deploy —
    never through `response2lighting`'s ``try/except``, and so never as a reprompt.

    Args:
        name: The emitted colour name.
        cfg: Lighting config, holding the palette.

    Returns:
        ``name``, unchanged.

    Raises:
        KeyError: If the name is not a palette entry.
    """
    if name not in cfg.palette:
        raise KeyError(f"Unknown lighting colour {name}")
    return name


def _light_color(params: dict, ctx: _BuildContext) -> ColourLayer:
    """`light_color(sel, color, deck)`: assign a calibrated palette colour to a subset."""
    return ColourLayer(
        ctx.mask, ctx.decks, "named", {"color": _palette_colour(params["color"], ctx.cfg)}
    )


def _gradient(params: dict, ctx: _BuildContext) -> ColourLayer:
    """`gradient(sel, color_a, color_b, by, deck)`: interpolate two palette colours across it.

    A `by` axis the selection has no extent along is reported for the reason `_spread` reports a
    collapsed spread: every drone lands on ``color_a``, which is a legal show and an exact
    `light_color`, so nothing downstream distinguishes it from one that was asked for. `index` is
    excluded because it ranks rather than measures, and so always has extent.
    """
    s = _gradient_s(params["by"], ctx.mask, ctx.positions)
    if params["by"] in _SPATIAL_SPREADS and ctx.mask.sum() > 1 and not s[ctx.mask].any():
        logger.warning(
            "Lighting gradient covers drones with no extent along %s, so every drone lands on "
            "color_a and it paints one flat colour instead of interpolating. It needs a formation "
            "spread out along that axis.",
            params["by"],
        )
    return ColourLayer(
        ctx.mask,
        ctx.decks,
        "gradient",
        {
            "color_a": _palette_colour(params["color_a"], ctx.cfg),
            "color_b": _palette_colour(params["color_b"], ctx.cfg),
            "s": s,
        },
    )


def _rainbow(params: dict, ctx: _BuildContext) -> ColourLayer:
    """`rainbow(sel, period_beats, spread, deck)`: a spectrum cycle along the chosen spread."""
    return ColourLayer(
        ctx.mask,
        ctx.decks,
        "cycled",
        {
            "period_s": _period_seconds(params["period_beats"], ctx),
            "offsets": _spread(ctx, params["spread"]),
        },
    )


def _light_on(params: dict, ctx: _BuildContext) -> BrightnessLayer:
    """`light_on(sel, deck)`: force full on. An HTP participant at 1.0, so it dominates (§8.3)."""
    return _brightness(ctx, "constant", 0.0, _DEFAULT_DUTY, "none", 1)


def _light_off(params: dict, ctx: _BuildContext) -> None:
    """`light_off(sel, deck)`: force dark. Produces no layer at all.

    Under the §8.3 Highest-Takes-Precedence reduction a layer contributing 0 is a no-op the instant
    anything else covers the same drone, so `light_off` is a post-reduction kill mask instead.
    `build_look` reads this ``None`` and sets the bits on ``Look.off_mask``.
    """
    return None


def _pulse(params: dict, ctx: _BuildContext) -> BrightnessLayer:
    """`pulse(sel, period_beats, deck)`: the whole group breathes together."""
    period_s = _period_seconds(params["period_beats"], ctx)
    return _brightness(ctx, "sine", period_s, _DEFAULT_DUTY, "none", 1)


def _blink(params: dict, ctx: _BuildContext) -> BrightnessLayer:
    """`blink(sel, period_beats, duty, deck)`: hard on/off flash, the group in sync."""
    period_s = _period_seconds(params["period_beats"], ctx)
    return _brightness(ctx, "square", period_s, float(params["duty"]), "none", 1)


def _strobe_decay(params: dict, ctx: _BuildContext) -> BrightnessLayer:
    """`strobe_decay(sel, period_beats, deck)`: flash on the beat, decay out."""
    period_s = _period_seconds(params["period_beats"], ctx)
    return _brightness(ctx, "ramp", period_s, _DEFAULT_DUTY, "none", 1)


def _chase(params: dict, ctx: _BuildContext) -> BrightnessLayer:
    """`chase(sel, period_beats, length, group_size, spread, deck)`: a running light along `spread`.

    ``length`` is how many drones are lit at once, which is exactly ``duty = length / n_sel``: the
    square wave's on-window then covers that fraction of the evenly spread phases at every instant.

    ``spread`` used to be fixed at `index`, and id order is not spatial order: `_assign_positions`
    leaves the ids in an arbitrary rotation around a formation, so an index chase visits 1->5 and
    then jumps to 10->6 instead of running around the ring (§7.3). The prompt now tells the model to
    say `neighbour` unless it specifically wants id order, and `index` stays available for a
    choreography that deliberately addresses drones by number.

    `chase` is also the one primitive whose per-drone lit window is narrower than half a period, so
    it is the one that has to tell `_period_seconds` what window the cue grid must resolve (§9.1).
    """
    # An empty selection makes every layer a no-op, so the duty is irrelevant there; the floor is
    # only here because `chase` is the one primitive that would divide by zero on it. `length` is
    # clipped into the 1 .. n_sel the schema declares, which keeps the lit window positive.
    n_sel = max(int(ctx.mask.sum()), 1)
    duty = float(np.clip(int(params["length"]) / n_sel, 1.0 / n_sel, 1.0))
    period_s = _period_seconds(params["period_beats"], ctx, duty)
    return _brightness(ctx, "square", period_s, duty, params["spread"], int(params["group_size"]))


def _sweep(params: dict, ctx: _BuildContext) -> BrightnessLayer:
    """`sweep(sel, period_beats, axis, deck)`: a directional sweep across the stage."""
    period_s = _period_seconds(params["period_beats"], ctx)
    return _brightness(ctx, "square", period_s, _DEFAULT_DUTY, params["axis"], 1)


def _ripple_light(params: dict, ctx: _BuildContext) -> BrightnessLayer:
    """`ripple_light(sel, period_beats, deck)`: a wave out from the swarm centre."""
    period_s = _period_seconds(params["period_beats"], ctx)
    return _brightness(ctx, "sine", period_s, _DEFAULT_DUTY, "radius", 1)


def _alternate_blink(params: dict, ctx: _BuildContext) -> BrightnessLayer:
    """`alternate_blink(sel, period_beats, by, deck)`: ping-pong between two halves.

    Not redundant with two `blink` calls: `blink` has no phase parameter, so the half-period offset
    that makes the ping-pong read can only come from a spread (§10.2).
    """
    period_s = _period_seconds(params["period_beats"], ctx)
    spread = _ALTERNATE_SPREADS[params["by"]]
    return _brightness(ctx, "square", period_s, _DEFAULT_DUTY, spread, 1)


# The catalogue of §10.2, flat, in spec order: three colour primitives then nine brightness ones.
# This is the vocabulary the prompt documents and the structured-output schema enumerates.
LIGHTING_PRIMITIVES: dict[str, _Builder] = {
    "light_color": _light_color,
    "gradient": _gradient,
    "rainbow": _rainbow,
    "light_on": _light_on,
    "light_off": _light_off,
    "pulse": _pulse,
    "blink": _blink,
    "strobe_decay": _strobe_decay,
    "chase": _chase,
    "sweep": _sweep,
    "ripple_light": _ripple_light,
    "alternate_blink": _alternate_blink,
}


def build_look(
    actions: list[dict], t_start: float, positions: NDArray, n: int, cfg: LightingConfig, bpm: float
) -> Look:
    """Compile one emitted lighting key's actions into a `Look` (§6, §10.2).

    Colour layers keep their order in ``actions``, because §8.2 resolves colour
    Latest-Takes-Precedence by position rather than by kind. Brightness layers merge
    Highest-Takes-Precedence, so their order is immaterial, but it is preserved too.

    Args:
        actions: The key's actions, each ``{"primitive": name, "params": {...}}``. Every action's
            params carry ``sel`` (a `Selector`, i.e. a ``(name, args)`` pair) and ``deck`` ("top",
            "bot" or "both") alongside that primitive's own parameters from §10.2.
        t_start: Show time in seconds at which the look takes over.
        positions: (n, 3) position snapshot, frozen at ``t_start`` (§7.3). The spatial selectors and
            spreads resolve against it here; it is also carried on the returned look, because the
            §8.5 base colour is ranked against it at read-out time.
        n: Number of drones in the swarm.
        cfg: Lighting config, for the palette and the stage axis.
        bpm: Song tempo in beats per minute, which converts every `period_beats` into seconds.

    Returns:
        The assembled look.

    Raises:
        KeyError: If a primitive, deck, selector, spread, gradient axis or colour name is unknown.
    """
    colour_layers: list[ColourLayer] = []
    brightness_layers: list[BrightnessLayer] = []
    off_mask = np.zeros((n, 2), dtype=bool)
    for action in actions:
        name = action["primitive"]
        if name not in LIGHTING_PRIMITIVES:
            raise KeyError(f"Unknown lighting primitive {name}")
        params = action["params"]
        decks = _DECK_CHOICES[params["deck"]]
        mask = select(params["sel"], n, positions, cfg)
        ctx = _BuildContext(name, mask, decks, positions, cfg, bpm)
        layer = LIGHTING_PRIMITIVES[name](params, ctx)
        if layer is None:
            for deck in decks:
                off_mask[mask, _DECKS.index(deck)] = True
        elif isinstance(layer, ColourLayer):
            colour_layers.append(layer)
        else:
            brightness_layers.append(layer)
    return Look(t_start, tuple(colour_layers), tuple(brightness_layers), off_mask, positions)
