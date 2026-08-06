"""The lighting engine: config, selectors, waveforms, spreads, colour sources, primitives, cues.

Colour and brightness are independent layers that multiply at the read-out, so any effect composes
with any colour source without needing a primitive per combination. Four sections, bottom up: the
engine, the `Look` / `LightingTimeline` read-out, the primitives the LLM may emit, and the
`compile_cues` bake into hardware colour cues.
"""

from __future__ import annotations

import logging
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from swarm_gpt.core.motion_primitives import _sanitize_drone_ids

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

    _Builder = Callable[[dict, "_BuildContext"], "ColourLayer | BrightnessLayer | None"]

logger = logging.getLogger(__name__)

# World axis index and sign that point to the audience's right, keyed by `stage_axis`.
_STAGE_AXES = {"+x": (0, 1.0), "-x": (0, -1.0), "+y": (1, 1.0), "-y": (1, -1.0)}

# Coordinate index a directional spread reads, keyed by spread name.
_SPREAD_AXES = {"x": 0, "y": 1, "z": 2}

# `duty` is clamped to (0, 1]; the open lower bound needs a positive floor.
_DUTY_MIN = 1e-6

# Deck axis order, shared by `Look.off_mask` and the `LightingTimeline` read-outs.
_DECKS = ("top", "bot")

# How far before the end of the show the blackout lands, so drones never land lit.
_BLACKOUT_LEAD_S = 0.1

# A selector is a name plus its arguments, e.g. ("all", ()), ("ids", (1, 3, 5)), ("first", (4,)).
Selector = tuple[str, tuple]

# Arguments each fixed-arity selector takes. `ids` is variadic and so absent.
_SELECTOR_ARITY = {"all": 0, "even": 0, "odd": 0, "left": 0, "right": 0, "first": 1}

# The spreads that rank the selection, and so the only ones `group_size` can bucket.
_RANKED_SPREADS = ("neighbour", "index")

# Fraction of the coordinate magnitude below which a span counts as no extent. Relative rather than
# exact-zero: a cos/sin ring is degenerate only to ~1e-16, so an equality test would divide by that.
_SPAN_REL_TOL = 1e-9

# The `deck` every primitive takes, mapped onto the decks its layer covers.
_DECK_CHOICES = {"top": ("top",), "bot": ("bot",), "both": _DECKS}

# `alternate_blink`'s `by`, mapped onto the spread that splits the group into antiphase halves.
_ALTERNATE_SPREADS = {"parity": "alternate_parity", "side": "alternate_side"}

# Waveform duty for every brightness primitive that does not set its own.
_DEFAULT_DUTY = 0.5

# Spreads that measure geometry, so a formation can leave them nothing to run along.
_SPATIAL_SPREADS = ("radius", *_SPREAD_AXES)


# --- Config ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class LightingConfig:
    """Palette and calibration constants loaded from ``swarm_gpt/data/lighting.toml``.

    Attributes:
        palette: Colour name -> (4,) calibrated WRGB in [0, 255] at full brightness.
        gamma: Perceived-brightness exponent applied to the merged brightness scalar.
        b_min: Merged brightness below which the LED goes fully dark.
        hue_steps: Hue quantization steps per `rainbow` cycle.
        brightness_steps: Quantization buckets the merged brightness is floored into.
        channel_gain: (4,) per-channel WRGB multiplier for generated hues.
        stage_axis: World axis pointing to the audience's right; "+x", "-x", "+y" or "-y".
        col_freq: Maximum colour-cue rate in Hz. Must match `DroneSwarm`'s.
    """

    palette: dict[str, NDArray]
    gamma: float
    b_min: float
    hue_steps: int
    brightness_steps: int
    channel_gain: NDArray
    stage_axis: str
    col_freq: float


def load_lighting_config(path: Path | None = None) -> LightingConfig:
    """Load the lighting palette and calibration constants.

    Args:
        path: Path to the TOML config. Defaults to ``swarm_gpt/data/lighting.toml``.

    Returns:
        The parsed configuration.
    """
    if path is None:
        path = Path(__file__).resolve().parents[1] / "data/lighting.toml"
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    return LightingConfig(
        palette={name: np.asarray(v, dtype=float) for name, v in raw["palette"].items()},
        gamma=float(raw["gamma"]),
        b_min=float(raw["b_min"]),
        hue_steps=int(raw["hue_steps"]),
        brightness_steps=int(raw["brightness_steps"]),
        channel_gain=np.asarray(raw["channel_gain"], dtype=float),
        stage_axis=raw["stage_axis"],
        col_freq=float(raw["col_freq"]),
    )


# --- Selectors ------------------------------------------------------------------------------


def _right_mask(positions: NDArray, cfg: LightingConfig) -> NDArray:
    """Mark the drones stage right of the swarm centroid along the stage axis.

    A formation with no extent along that axis goes entirely stage left and warns, rather than
    being dealt into halves by ``coord > coord.mean()`` on float noise.

    Args:
        positions: (n, 3) drone positions.
        cfg: Lighting config, whose ``stage_axis`` says which world axis points audience-right.

    Returns:
        (n,) boolean mask. Its complement is stage left, so the two partition the swarm.
    """
    axis, sign = _STAGE_AXES[cfg.stage_axis]
    coord = sign * positions[:, axis]
    if coord.size > 1 and coord.max() - coord.min() <= _SPAN_REL_TOL * np.abs(coord).max():
        logger.warning(
            "Lighting left/right split has no extent along stage_axis %s: every drone is stage "
            "left and `right` selects nobody, so a left/right effect covers the whole swarm or "
            "none of it.",
            cfg.stage_axis,
        )
        return np.zeros(coord.size, dtype=bool)
    return coord > coord.mean()


def select(sel: Selector, n: int, positions: NDArray, cfg: LightingConfig) -> NDArray:
    """Resolve a selector into the set of drones a lighting layer covers.

    Bounds are checked here rather than in the LLM output schema, because presets and hand-written
    ``lighting:`` blocks bypass it and every unchecked case is silently wrong rather than an error:
    ``ids(0)`` shifts to -1 and selects the last drone, ``first(99)`` selects everything.

    Args:
        sel: The selector, as a ``(name, args)`` pair.
        n: Number of drones in the swarm.
        positions: (n, 3) snapshot frozen once for the look; only spatial selectors read it.
        cfg: Lighting config, for the stage axis.

    Returns:
        (n,) boolean mask over 0-indexed drone indices.

    Raises:
        KeyError: If the selector name is unknown.
        LLMFormatError: If an ``ids`` entry is not an integer, from `_sanitize_drone_ids`.
        IndexError: If the selector carries the wrong number of arguments, names no drones, or an
            ``ids`` entry or `first` count falls outside ``1..n``.
    """
    kind, args = sel
    if kind in _SELECTOR_ARITY and len(args) != _SELECTOR_ARITY[kind]:
        raise IndexError(
            f"Lighting selector {kind} takes {_SELECTOR_ARITY[kind]} arguments, got {len(args)}"
        )
    if kind == "all":
        return np.ones(n, dtype=bool)
    if kind == "left":
        return ~_right_mask(positions, cfg)
    if kind == "right":
        return _right_mask(positions, cfg)
    mask = np.zeros(n, dtype=bool)
    if kind == "ids":
        if not args:
            raise IndexError("Lighting selector ids() names no drones")
        # `ids` is 1-indexed on the LLM side; _sanitize_drone_ids shifts it and validates the shape.
        ids = _sanitize_drone_ids(list(args), n)
        if out_of_range := sorted(i + 1 for i in ids if not 0 <= i < n):
            raise IndexError(f"Lighting drone ids {out_of_range} are outside the 1..{n} swarm")
        mask[ids] = True
    elif kind == "even":
        mask[0::2] = True
    elif kind == "odd":
        mask[1::2] = True
    elif kind == "first":
        count = int(args[0])
        if not 1 <= count <= n:
            raise IndexError(f"Lighting first({count}) is outside the 1..{n} swarm")
        mask[:count] = True
    else:
        raise KeyError(f"Unknown lighting selector {kind}")
    return mask


# --- Waveforms and phase spreads ------------------------------------------------------------


def waveform(kind: str, phase: NDArray, duty: float = 0.5) -> NDArray:
    """Evaluate an effect waveform. All three peak at ``phase = 0``, and the phase wraps.

    Args:
        kind: One of "sine", "square" or "ramp".
        phase: Phase in turns, of any shape.
        duty: Fraction of each period "square" stays on, clamped to (0, 1]. Ignored by the others.

    Returns:
        Waveform values in [0, 1], the same shape as ``phase``.

    Raises:
        KeyError: If the waveform name is unknown.
    """
    frac = np.mod(phase, 1.0)
    if kind == "sine":
        return 0.5 * (1.0 + np.cos(2.0 * np.pi * frac))
    if kind == "square":
        return (frac < np.clip(duty, _DUTY_MIN, 1.0)).astype(float)
    if kind == "ramp":
        return 1.0 - frac
    raise KeyError(f"Unknown lighting waveform {kind}")


def _normalize_span(values: NDArray) -> NDArray:
    """Normalize spatial values into the half-open [0, 1), as the `index` spread does.

    The ``(n - 1) / n`` scaling keeps the far drone off 1.0, which is the near drone's phase.

    Args:
        values: (n_sel,) coordinates or distances for the selected drones.

    Returns:
        (n_sel,) offsets in [0, 1); all zeros if the span is inside `_SPAN_REL_TOL`.
    """
    span = values.max() - values.min()
    if span <= _SPAN_REL_TOL * np.abs(values).max():
        return np.zeros_like(values)
    n_sel = values.size
    return (values - values.min()) / span * (n_sel - 1) / n_sel


def _neighbour_ranks(points: NDArray) -> NDArray:
    """Rank points along a greedy nearest-neighbour walk over them.

    Drone id order is not spatial order, because formation primitives assign drones to slots by
    whichever permutation is cheapest to fly. The walk recovers spatial order instead, and runs over
    the lexicographically sorted points so the id permutation cannot break a ring's first-step tie.

    Args:
        points: (m, 3) positions to rank; a caller ranking a subset passes the subset's rows.

    Returns:
        (m,) integer ranks in ``0 .. m - 1``, one per input row, in walk order.
    """
    lex = np.lexsort(points.T[::-1])
    walk = points[lex]
    order = [0]
    unvisited = np.ones(walk.shape[0], dtype=bool)
    unvisited[0] = False
    while unvisited.any():
        distance = np.linalg.norm(walk - walk[order[-1]], axis=1)
        distance[~unvisited] = np.inf
        nearest = int(np.argmin(distance))
        order.append(nearest)
        unvisited[nearest] = False
    ranks = np.empty(walk.shape[0], dtype=int)
    ranks[lex[order]] = np.arange(walk.shape[0])
    return ranks


def spread_offsets(
    kind: str, mask: NDArray, positions: NDArray, group_size: int, cfg: LightingConfig
) -> NDArray:
    """Compute the per-drone phase offsets that turn one waveform into a family of effects.

    Offsets are relative to the selected subset, except "alternate_side", which splits against the
    swarm centroid so it matches the `left`/`right` selectors. ``group_size`` above 1 is rejected
    outside the ranked spreads rather than ignored, since bucketing a coordinate evenly would turn
    a proportional sweep into an evenly stepped one without the emitting model ever knowing.

    Args:
        kind: "none", "neighbour", "index", "alternate_parity", "alternate_side", "radius", "x",
            "y" or "z".
        mask: (n,) boolean mask of the selected drones.
        positions: (n, 3) frozen position snapshot.
        group_size: Drones per phase bucket for the ranked spreads; 1 is per-drone.
        cfg: Lighting config, for the stage axis.

    Returns:
        (n,) offsets in turns, in [0, 1). Unselected drones are 0.

    Raises:
        KeyError: If the spread name is unknown.
        ValueError: If ``group_size`` is below 1, or above 1 on a spread that cannot bucket it.
    """
    if group_size < 1:
        raise ValueError(f"group_size must be >= 1, got {group_size}")
    if group_size > 1 and kind not in _RANKED_SPREADS:
        raise ValueError(
            f"group_size={group_size} needs a ranked spread ({' or '.join(_RANKED_SPREADS)}), "
            f"got {kind}"
        )
    offsets = np.zeros(mask.shape[0])
    idx = np.flatnonzero(mask)
    if kind == "none" or idx.size == 0:
        return offsets
    if kind in _RANKED_SPREADS:
        ranks = _neighbour_ranks(positions[idx]) if kind == "neighbour" else np.arange(idx.size)
        n_groups = int(np.ceil(idx.size / group_size))
        offsets[idx] = (ranks // group_size) / n_groups
    elif kind == "alternate_parity":
        offsets[idx] = 0.5 * (idx % 2)
    elif kind == "alternate_side":
        offsets[idx] = 0.5 * _right_mask(positions, cfg)[idx]
    elif kind == "radius":
        centroid = positions[idx].mean(axis=0)
        offsets[idx] = _normalize_span(np.linalg.norm(positions[idx] - centroid, axis=1))
    elif kind in _SPREAD_AXES:
        offsets[idx] = _normalize_span(positions[idx, _SPREAD_AXES[kind]])
    else:
        raise KeyError(f"Unknown lighting spread {kind}")
    return offsets


# --- Colour sources and layers --------------------------------------------------------------


def hue_to_wrgb(hue: NDArray, cfg: LightingConfig) -> NDArray:
    """Convert hues on the colour wheel to calibrated full-brightness WRGB.

    The order is load-bearing: normalize to a constant channel sum first, then apply
    ``channel_gain``. The other way divides the gain back out for any hue on a single channel.

    Args:
        hue: Hues in turns, of any shape. Values outside [0, 1) wrap.
        cfg: Lighting config, for ``channel_gain``.

    Returns:
        WRGB values in [0, 255], shaped ``hue.shape + (4,)``.
    """
    h6 = 6.0 * np.mod(np.asarray(hue, dtype=float), 1.0)
    rgb = np.stack(
        [
            np.clip(np.abs(h6 - 3.0) - 1.0, 0.0, 1.0),
            np.clip(2.0 - np.abs(h6 - 2.0), 0.0, 1.0),
            np.clip(2.0 - np.abs(h6 - 4.0), 0.0, 1.0),
        ],
        axis=-1,
    )
    rgb = 255.0 * rgb / rgb.sum(axis=-1, keepdims=True)
    rgb = rgb * cfg.channel_gain[1:]
    return np.concatenate([np.zeros(rgb.shape[:-1] + (1,)), rgb], axis=-1)


@dataclass(frozen=True)
class ColourLayer:
    """One colour source covering a subset of the swarm on one or both decks.

    Within a look, later colour layers overwrite earlier ones over the rows their mask covers.

    Attributes:
        mask: (n,) boolean mask of the drones this layer covers.
        decks: Which decks it applies to, a subset of ("top", "bot").
        kind: One of "named", "gradient" or "cycled".
        params: ``{"color"}`` for "named"; ``{"color_a", "color_b", "s"}`` for "gradient", where
            ``s`` is (n,) on the inclusive [0, 1] so the extremes hit the endpoints exactly;
            ``{"period_s", "offsets"}`` for "cycled", the offsets being `spread_offsets`'.
    """

    mask: NDArray
    decks: tuple[str, ...]
    kind: str
    params: dict

    def evaluate(self, t: float, cfg: LightingConfig) -> NDArray:
        """Evaluate the layer's full-brightness colour at time ``t``.

        Args:
            t: Show time in seconds.
            cfg: Lighting config, for the palette and hue calibration.

        Returns:
            (n, 4) WRGB in [0, 255]. Rows outside the mask are zero.

        Raises:
            KeyError: If the layer kind, or a named colour, is unknown.
        """
        colours = np.zeros((self.mask.shape[0], 4))
        idx = np.flatnonzero(self.mask)
        if idx.size == 0:
            return colours
        if self.kind == "named":
            colours[idx] = cfg.palette[self.params["color"]]
        elif self.kind == "gradient":
            s = np.asarray(self.params["s"], dtype=float)[idx, None]
            colours[idx] = (1.0 - s) * cfg.palette[self.params["color_a"]] + s * cfg.palette[
                self.params["color_b"]
            ]
        elif self.kind == "cycled":
            offsets = np.asarray(self.params["offsets"], dtype=float)[idx]
            hue = np.mod(t / self.params["period_s"] - offsets, 1.0)
            colours[idx] = hue_to_wrgb(np.floor(hue * cfg.hue_steps) / cfg.hue_steps, cfg)
        else:
            raise KeyError(f"Unknown colour layer kind {self.kind}")
        return colours


@dataclass(frozen=True)
class BrightnessLayer:
    """One brightness effect covering a subset of the swarm on one or both decks.

    Within a look, brightness layers reduce with ``max``. ``light_on`` is one of these, a "constant"
    layer at 1.0; ``light_off`` is a kill mask applied after the reduction instead, since a layer
    contributing 0 would be a no-op under ``max``.

    Attributes:
        mask: (n,) boolean mask of the drones this layer covers.
        decks: Which decks it applies to, a subset of ("top", "bot").
        kind: "constant", or a waveform name: "sine", "square" or "ramp".
        period_s: Waveform period in seconds. Unused by "constant".
        duty: Fraction of each period "square" stays on. Unused by the others.
        offsets: (n,) phase offsets in turns, as produced by `spread_offsets`.
    """

    mask: NDArray
    decks: tuple[str, ...]
    kind: str
    period_s: float
    duty: float
    offsets: NDArray

    def evaluate(self, t: float) -> NDArray:
        """Evaluate the layer's brightness at time ``t``.

        Args:
            t: Show time in seconds.

        Returns:
            (n,) brightness in [0, 1]. Rows outside the mask are zero.

        Raises:
            KeyError: If the kind is neither "constant" nor a known waveform.
        """
        out = np.zeros(self.mask.shape[0])
        idx = np.flatnonzero(self.mask)
        if idx.size == 0:
            return out
        if self.kind == "constant":
            out[idx] = 1.0
            return out
        phase = t / self.period_s - np.asarray(self.offsets, dtype=float)[idx]
        out[idx] = waveform(self.kind, phase, self.duty)
        return out


# --- Looks and the timeline read-out --------------------------------------------------------


@dataclass(frozen=True)
class Look:
    """The complete lighting state from one emitted key until the next.

    The next look replaces this one rather than layering onto it, so a colour that should persist
    has to be restated.

    Attributes:
        t_start: Show time in seconds at which this look takes over.
        colour_layers: Colour sources in emission order; later ones overwrite earlier ones.
        brightness_layers: Brightness effects, reduced with ``max``.
        off_mask: (n, 2) boolean `light_off` kill mask, per deck in ``_DECKS`` order, applied after
            that reduction so it beats every layer covering the same drone.
        positions: (n, 3) snapshot frozen once for the look, used for the default hue wheel.
            ``None`` falls back to id order, only ever the case on a timeline with no looks at all.
    """

    t_start: float
    colour_layers: tuple[ColourLayer, ...]
    brightness_layers: tuple[BrightnessLayer, ...]
    off_mask: NDArray
    positions: NDArray | None = None


class LightingTimeline:
    """An ordered list of looks, evaluable at any show time.

    A pure function of ``t`` plus the snapshots already frozen into its layers, so the per-frame sim
    read-out and the baked hardware cues see exactly the same thing.
    """

    def __init__(self, looks: list[Look], n: int, t_end: float, cfg: LightingConfig) -> None:
        """Assemble the timeline.

        Args:
            looks: The emitted looks, in any order. Sorted stably by ``t_start``, so two looks on
                the same time resolve in favour of the later one.
            n: Number of drones in the swarm.
            t_end: Show duration in seconds; the blackout lands ``_BLACKOUT_LEAD_S`` before it.
            cfg: Lighting config, for the hue calibration, gamma and the dim floor.
        """
        self._n = n
        self._cfg = cfg
        self._t_blackout = t_end - _BLACKOUT_LEAD_S
        ordered = sorted(looks, key=lambda look: look.t_start)
        # A layerless look covering everything before the first emitted key, so the lookup never has
        # to special-case "no look yet". It borrows the first look's snapshot so the hue order does
        # not change when that look takes over.
        snapshot = ordered[0].positions if ordered else None
        base = Look(-np.inf, (), (), np.zeros((n, 2), dtype=bool), snapshot)
        self._looks = [base, *ordered]
        self._starts = np.array([look.t_start for look in self._looks])
        self._base_colours = [self._base_colour(look) for look in self._looks]

    def _base_colour(self, look: Look) -> NDArray:
        """Assign the default hue wheel across the swarm in one look's `neighbour` order.

        Args:
            look: The look to colour for.

        Returns:
            (n, 4) full-brightness WRGB, one row per drone.
        """
        ranks = np.arange(self._n) if look.positions is None else _neighbour_ranks(look.positions)
        return hue_to_wrgb(ranks / self._n, self._cfg)

    def _look_index_at(self, t: float) -> int:
        """Find the index of the look covering ``t``.

        Args:
            t: Show time in seconds.

        Returns:
            Index into ``self._looks``, which is 0 -- the base look -- before the first emitted one.
        """
        return int(np.searchsorted(self._starts, t, side="right")) - 1

    def _merge_colour(self, look: Look, base: NDArray, t: float, deck: str) -> NDArray:
        """Merge one deck's colour layers, later layers overwriting earlier ones.

        The overwrite is driven by each layer's mask, not by whether its output is non-zero: an
        unselected row and a legitimately dark drone both read as zeros.

        Args:
            look: The look in force.
            base: That look's (n, 4) default colour, from `_base_colour`.
            t: Show time in seconds.
            deck: Which deck to resolve.

        Returns:
            (n, 4) full-brightness WRGB. Drones no layer covers carry the default colour.
        """
        colours = base.copy()
        for layer in look.colour_layers:
            if deck in layer.decks:
                colours[layer.mask] = layer.evaluate(t, self._cfg)[layer.mask]
        return colours

    def _merge_brightness(self, look: Look, t: float, deck: str, deck_idx: int) -> NDArray:
        """Reduce one deck's brightness layers with ``max``, then apply the kill mask.

        Coverage comes from the layer masks, not the merged values: a `square` layer in its off
        phase legitimately contributes 0, and reading the base state off the value would invert it.

        Args:
            look: The look in force.
            t: Show time in seconds.
            deck: Which deck to resolve.
            deck_idx: That deck's index into ``look.off_mask``.

        Returns:
            (n,) brightness in [0, 1].
        """
        brightness = np.zeros(self._n)
        covered = np.zeros(self._n, dtype=bool)
        for layer in look.brightness_layers:
            if deck in layer.decks:
                brightness = np.maximum(brightness, layer.evaluate(t))
                covered |= layer.mask
        # Full-on is a fallback, not a participant in the max: it applies only where nothing else
        # does.
        brightness[~covered] = 1.0
        brightness[look.off_mask[:, deck_idx]] = 0.0
        return brightness

    def evaluate(self, t: float) -> NDArray:
        """Evaluate every drone's colour on both decks at show time ``t``.

        Brightness is floored into ``brightness_steps`` buckets before the multiply, which makes the
        continuous waveforms piecewise-constant so `compile_cues` can dedup the runs. ``b_min`` is
        applied before that, or quantizing would make the floor inert.

        Args:
            t: Show time in seconds.

        Returns:
            (n, 2, 4) integral WRGB in [0, 255], with the deck axis ordered (top, bot).
        """
        if t >= self._t_blackout:
            return np.zeros((self._n, 2, 4))
        index = self._look_index_at(t)
        look = self._looks[index]
        steps = self._cfg.brightness_steps
        out = np.empty((self._n, 2, 4))
        for deck_idx, deck in enumerate(_DECKS):
            colours = self._merge_colour(look, self._base_colours[index], t, deck)
            merged = self._merge_brightness(look, t, deck, deck_idx)
            merged[merged < self._cfg.b_min] = 0.0
            brightness = (np.floor(merged * steps) / steps)[:, None]
            out[:, deck_idx] = np.round(colours * brightness**self._cfg.gamma)
        return out

    def evaluate_rgb01(self, t: float, deck: str = "top") -> NDArray:
        """Evaluate one deck as RGB in [0, 1] for the 3D viewer, folding W into all three channels.

        Args:
            t: Show time in seconds.
            deck: Which deck to read, "top" or "bot".

        Returns:
            (n, 3) RGB in [0, 1].

        Raises:
            ValueError: If ``deck`` is not one of the two deck names.
        """
        wrgb = self.evaluate(t)[:, _DECKS.index(deck)]
        return np.clip((wrgb[:, 1:] + wrgb[:, :1]) / 255.0, 0.0, 1.0)


# --- The primitives the LLM may emit --------------------------------------------------------
#
# The mapping onto the engine above is thin: `chase` and `sweep` are both "square wave plus a
# spread", and `rainbow` and `chase` differ only in whether the spread drives hue.


@dataclass(frozen=True)
class _BuildContext:
    """Everything a primitive builder reads besides its own parameters.

    Attributes:
        primitive: The name this action was emitted under, for diagnostics.
        mask: (n,) boolean mask the action's ``sel`` resolved to.
        decks: The decks the action's ``deck`` resolved to, a subset of ("top", "bot").
        positions: (n, 3) position snapshot, frozen once for the look.
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
    """Convert an emitted beat period into seconds, held off the cue-rate aliasing floor.

    The clamped quantity is the lit window, not the period: below one ``1 / col_freq`` tick,
    ``period_s x lit_fraction`` can fall entirely between two grid ticks and the drone is then never
    lit in the compiled cues at all. Over-fast effects are clamped rather than rejected.

    Args:
        period_beats: Effect period in beats, as emitted.
        ctx: The action's build context, for the song tempo and the configured cue rate.
        lit_fraction: Fraction of each period one drone must be distinguishable for; only `chase`
            narrows it below the default half period.

    Returns:
        The period in seconds, at least ``max(2 / col_freq, 1 / (col_freq x lit_fraction))``.
    """
    tick_s = 1.0 / ctx.cfg.col_freq
    min_period_s = max(2.0 * tick_s, tick_s / lit_fraction)
    period_s = float(period_beats) * 60.0 / ctx.bpm
    if period_s >= min_period_s:
        return period_s
    logger.warning(
        "Lighting period_beats=%g is %.3f s at %.1f BPM, leaving each drone lit for %.3f s — under "
        "the %.3f s cue tick, which drops the effect from some drones entirely rather than merely "
        "coarsening it. Clamping that window to %.3f s, stretching the period to %.3f s.",
        period_beats,
        period_s,
        ctx.bpm,
        period_s * lit_fraction,
        tick_s,
        min_period_s * lit_fraction,
        min_period_s,
    )
    return min_period_s


def _gradient_s(by: str, mask: NDArray, positions: NDArray) -> NDArray:
    """Compute `gradient`'s interpolation parameter along ``by``.

    Normalized onto the inclusive [0, 1] so the far drone reproduces ``color_b`` exactly, which is
    why this is not `spread_offsets` -- that range is half-open and leaves ``color_b`` unreachable.

    Args:
        by: One of "index", "x", "y", "z" or "radius".
        mask: (n,) boolean mask of the selected drones.
        positions: (n, 3) frozen position snapshot.

    Returns:
        (n,) values in [0, 1]. Zero for unselected drones, and for all of them when the subset has
        no extent along ``by``.

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
    """Resolve a phase spread against the frozen snapshot, reporting one it collapses on.

    A selection with no extent along a spatial spread's axis gets every offset 0 and the effect
    degrades into a synchronised blink, which is legal but indistinguishable from a working one.

    Args:
        ctx: The action's build context, which carries the emitting primitive's name.
        kind: The spread name that sets the per-drone phase offsets.
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
        kind: "constant", or one of the waveform names.
        period_s: Waveform period in seconds. Unused by "constant".
        duty: Fraction of each period a "square" waveform stays on.
        spread: The spread name that sets the per-drone phase offsets.
        group_size: Drones per phase bucket; quantizes the ranked spreads only.

    Returns:
        The layer, with full-swarm-shaped (n,) offsets.
    """
    offsets = _spread(ctx, spread, group_size)
    return BrightnessLayer(ctx.mask, ctx.decks, kind, period_s, duty, offsets)


def _palette_colour(name: str, cfg: LightingConfig) -> str:
    """Check a colour name against the palette and return it.

    `ColourLayer` resolves palette names lazily, so an unchecked one would surface as a bare
    ``KeyError`` mid-render or mid-deploy rather than as a reprompt.

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

    A `by` axis with no extent lands every drone on ``color_a``, so it is reported for the reason
    `_spread` reports a collapsed spread.
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
    """`light_on(sel, deck)`: force full on. Contributes 1.0 to the max, so it dominates."""
    return _brightness(ctx, "constant", 0.0, _DEFAULT_DUTY, "none", 1)


def _light_off(params: dict, ctx: _BuildContext) -> None:
    """`light_off(sel, deck)`: force dark, via `Look.off_mask` rather than a layer.

    A layer contributing 0 is a no-op under the ``max`` reduction the instant anything else covers
    the same drone, so `build_look` reads this ``None`` and sets the mask bits instead.
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

    ``length`` is how many drones are lit at once, i.e. ``duty = length / n_sel``. That makes it the
    one primitive whose lit window is narrower than half a period, hence the ``duty`` passed on.
    """
    # The `n_sel` floor is only here because `chase` is the one primitive that would divide by zero
    # on an empty selection, where the duty is irrelevant anyway.
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

    Not two `blink` calls: `blink` has no phase parameter, so the half-period offset that makes the
    ping-pong read can only come from a spread.
    """
    period_s = _period_seconds(params["period_beats"], ctx)
    spread = _ALTERNATE_SPREADS[params["by"]]
    return _brightness(ctx, "square", period_s, _DEFAULT_DUTY, spread, 1)


# The catalogue the prompt documents and the LLM output schema enumerates, in prompt order.
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
    """Compile one emitted lighting key's actions into a `Look`.

    Colour layers keep their order in ``actions``, since a later colour overwrites an earlier one by
    position rather than by kind.

    Args:
        actions: The key's actions, each ``{"primitive": name, "params": {...}}``, whose params
            carry ``sel`` and ``deck`` alongside that primitive's own parameters.
        t_start: Show time in seconds at which the look takes over.
        positions: (n, 3) snapshot the spatial selectors and spreads resolve against, also
            carried on the look for the default hue wheel. The caller picks when to sample it --
            not necessarily ``t_start``, since a formation emitted at that address has not
            arrived yet.
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


# --- The hardware read-out ------------------------------------------------------------------
#
# `DroneSwarm` drains at most one cue per deck per `1 / col_freq` tick and never drops, so a denser
# cue list plays back slowed and drifts out of sync with the music permanently. Sampling on a
# uniform `col_freq` grid and dropping consecutive duplicates rules that out structurally.


def _sample_times(col_freq: float, t_end: float) -> NDArray:
    """Build the sample grid, terminated by the blackout instant.

    The blackout is appended explicitly, since a grid anchored at 0 lands on it only by luck; ticks
    it would crowd are dropped first, so it cannot itself break the minimum spacing between cues.

    Args:
        col_freq: Maximum colour-cue rate in Hz, matching ``DroneSwarm.col_freq``.
        t_end: Show duration in seconds.

    Returns:
        Strictly increasing sample times at least ``1 / col_freq`` apart, opening at 0 and ending at
        the blackout.

    Raises:
        ValueError: If the show ends less than one cue period after the blackout instant.
    """
    period = 1.0 / col_freq
    t_blackout = t_end - _BLACKOUT_LEAD_S
    if t_blackout < period:
        raise ValueError(
            f"A {t_end} s show is too short to compile lighting cues: it leaves {t_blackout} s "
            f"before the blackout, under the {period} s cue period at {col_freq} Hz"
        )
    ticks = np.arange(int(np.floor(t_blackout * col_freq)) + 1) / col_freq
    return np.append(ticks[t_blackout - ticks >= period], t_blackout)


def compile_cues(
    timeline: LightingTimeline, uris: list[str], col_freq: float, t_end: float
) -> tuple[dict[str, dict[float, NDArray]], dict[str, dict[float, NDArray]]]:
    """Bake a lighting timeline into per-deck colour cues for `DroneSwarm`.

    Args:
        timeline: The lighting timeline, already carrying its frozen position snapshots.
        uris: Radio URI per drone, in the timeline's drone-index order.
        col_freq: Maximum colour-cue rate in Hz, matching ``DroneSwarm.col_freq``.
        t_end: Show duration in seconds.

    Returns:
        ``(color_top, color_bot)``, each ``{uri: {time: (4,) WRGB}}``, ready for
        ``execute_choreography``.

    Raises:
        ValueError: If ``uris`` does not cover the swarm the timeline was built for, or if the show
            is too short for the sample grid to open at 0.
    """
    times = _sample_times(col_freq, t_end)
    frames = np.stack([timeline.evaluate(float(t)) for t in times])  # (n_samples, n, 2, 4)
    n = frames.shape[1]
    if len(uris) != n:
        raise ValueError(f"Got {len(uris)} URIs for a {n}-drone lighting timeline")
    top: dict[str, dict[float, NDArray]] = {}
    bot: dict[str, dict[float, NDArray]] = {}
    for deck_idx, cues in enumerate((top, bot)):  # the deck axis is ordered (top, bot) throughout
        for i, uri in enumerate(uris):
            track = frames[:, i, deck_idx]
            changed = np.ones(times.size, dtype=bool)
            changed[1:] = np.any(track[1:] != track[:-1], axis=1)
            cues[uri] = {float(times[k]): track[k] for k in np.flatnonzero(changed)}
    return top, bot
