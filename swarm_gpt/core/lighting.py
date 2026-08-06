"""Lighting primitive layer: config, selectors, waveforms, phase spreads and colour sources.

Spec: ``docs/specs/2026-08-05-lighting-primitives-design.md``. Lighting factors into two orthogonal
layers that multiply at the read-out — colour (which hue a drone carries) and brightness (a per-drone
scalar in ``[0, 1]``) — so any brightness effect composes with any colour source without a
combinatorial catalogue.

This module is pure NumPy plus stdlib. It never imports the backend, the simulator or JAX, which is
what lets the whole lighting engine be tested without a trajectory or a radio.
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
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# World axis index and sign that point to the audience's right, keyed by `stage_axis` (§7.1).
_STAGE_AXES = {"+x": (0, 1.0), "-x": (0, -1.0), "+y": (1, 1.0), "-y": (1, -1.0)}

# Coordinate index a directional spread reads, keyed by spread name (§7.3).
_SPREAD_AXES = {"x": 0, "y": 1, "z": 2}

# `duty` is clamped to (0, 1]; the open lower bound needs a positive floor (§7.2).
_DUTY_MIN = 1e-6

# Deck axis order, shared by `Look.off_mask` and the `LightingTimeline` read-outs (§6, §8.6).
_DECKS = ("top", "bot")

# How far before the end of the show the unconditional blackout lands (§8.7). This is the existing
# terminal cue offset from `backend.py:332`, kept so the drones never land lit.
_BLACKOUT_LEAD_S = 0.1

# A selector is a name plus its arguments, e.g. ("all", ()), ("ids", (1, 3, 5)), ("first", (4,)).
Selector = tuple[str, tuple]

# How many arguments each fixed-arity selector takes (§7.1). `ids` is variadic and is absent, but
# still has to name at least one drone.
_SELECTOR_ARITY = {"all": 0, "even": 0, "odd": 0, "left": 0, "right": 0, "first": 1}

# The two spreads that rank the selection, and so are the two `group_size` can bucket (§7.3).
_RANKED_SPREADS = ("neighbour", "index")

# Fraction of the coordinate magnitude below which a spatial span counts as no extent at all.
# Relative rather than absolute because the values are metres of whatever the formation happens to
# be sized at, and `span <= 2 * max(|values|)` always, so the ratio is bounded. Three users:
# `_normalize_span` and `_right_mask` here, `_gradient_s` in `lighting_primitives`. An exact-zero
# test is not enough for any of them: a ring built from cos/sin is degenerate only to ~1e-16, so it
# slips past equality and the code then divides by float noise.
_SPAN_REL_TOL = 1e-9


@dataclass(frozen=True)
class LightingConfig:
    """Palette and calibration constants loaded from ``swarm_gpt/data/lighting.toml``.

    Attributes:
        palette: Colour name -> (4,) WRGB float in [0, 255] at full brightness, already calibrated.
        gamma: Perceived-brightness exponent applied to the merged brightness scalar.
        b_min: Merged brightness below which the LED goes fully dark.
        hue_steps: Hue quantization steps per `rainbow` cycle.
        brightness_steps: Quantization buckets the merged brightness is floored into.
        channel_gain: (4,) per-channel WRGB multiplier for generated hues.
        stage_axis: Which world axis points to the audience's right; one of "+x", "-x", "+y", "-y".
        col_freq: Maximum colour-cue rate in Hz. Sets both the `compile_cues` sample grid and the
            Nyquist floor effect periods are clamped against (§9.1), and must match the `col_freq`
            given to `DroneSwarm` (`drone_swarm.py:48`).
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

    Every key is required and indexed directly, so a truncated config fails loudly rather than
    flying with a silent default (CLAUDE.md §6.2).

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


def _right_mask(positions: NDArray, cfg: LightingConfig) -> NDArray:
    """Mark the drones on the audience's right of the swarm centroid along the stage axis.

    The split is a strict ``>`` against the mean, so a formation with no extent along the stage
    axis -- a vertical line, or a circle seen edge-on -- puts every drone stage left and none stage
    right. `light_color(right, ...)` is then a no-op that paints nobody, which is legal but almost
    never what was meant, so it is logged.

    "No extent" is `_normalize_span`'s test -- the span against ``_SPAN_REL_TOL`` times the
    coordinate magnitude, not against exact zero -- and the degenerate branch returns that all-left
    split explicitly rather than leaving it to the comparison. An edge-on ring forces both: its
    stage-axis coordinate is projected through a heading whose cosine is 6.1e-17 rather than 0, so
    it comes out equal only to within an ulp or two. An exact-zero test takes the *non-degenerate*
    branch there and ``coord > coord.mean()`` deals the swarm into two arbitrary halves on that
    rounding noise -- with the warning silent, and with `alternate_side` putting the same arbitrary
    half into antiphase. The explicit return matters even for an exactly degenerate axis: the mean
    of equal non-zero coordinates need not round back to the value they all share, and a mean an
    ulp low would send *every* drone stage right.

    Args:
        positions: (n, 3) drone positions.
        cfg: Lighting config, whose ``stage_axis`` says which world axis points audience-right.

    Returns:
        An (n,) boolean mask. The complement is stage left, so the two partition the swarm.
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
    """Resolve a selector into the set of drones a lighting layer covers (§7.1).

    Both bounds of ``ids`` and ``first`` are checked **here**, not left to the structured-output
    schema's ``minimum: 1`` / ``maximum: num_drones``: presets and hand-written ``lighting:``
    blocks reach `build_look` without ever passing through the schema, and that is the path the
    shipped demo preset takes. `_sanitize_drone_ids` checks neither bound — it only shifts the
    1-indexed emission down by one — so without this an ``ids`` entry of 0 would become -1 and
    quietly select the *last* drone, and ``first(99)`` would clamp against the slice and select
    everything. Both are silent wrong answers rather than errors, which is the worst failure mode
    available. The checks live in `select` rather than in `_sanitize_drone_ids` so the motion path
    keeps its existing behaviour (CLAUDE.md §3).

    The **argument count** and an **empty ``ids``** are checked here for the same reason and report
    the same way. ``("all", (1, 2, 3))`` reads as "drones 1-3" and used to drop the extras;
    ``ids([])`` selected nothing, so its whole layer did nothing, which is the silent wrong answer
    ``first(0)`` already raised on.

    Args:
        sel: The selector, as a ``(name, args)`` pair.
        n: Number of drones in the swarm.
        positions: (n, 3) position snapshot, frozen at the look's start time (§7.3). Only the
            spatial selectors read it.
        cfg: Lighting config, for the stage axis.

    Returns:
        An (n,) boolean mask over 0-indexed drone indices.

    Raises:
        KeyError: If the selector name is not one of the §7.1 vocabulary.
        LLMFormatError: If an ``ids`` entry is not an integer, from `_sanitize_drone_ids`.
        IndexError: If the selector carries the wrong number of arguments, names no drones at all,
            or an ``ids`` entry or a `first` count falls outside ``1..n`` -- matching how an
            out-of-range motion drone id fails.
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


def waveform(kind: str, phase: NDArray, duty: float = 0.5) -> NDArray:
    """Evaluate an effect waveform (§7.2).

    All three waveforms peak at ``phase = 0`` so effects land *on* the beat rather than between
    beats. The phase wraps, so negative phases -- which a spread offset routinely produces near
    ``t = 0`` -- behave the same as their positive equivalents.

    Args:
        kind: One of "sine", "square" or "ramp".
        phase: Phase in turns, of any shape.
        duty: Fraction of each period the "square" waveform stays on, clamped to (0, 1]. Ignored by
            the other waveforms.

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
    """Normalize spatial values into [0, 1) using the same convention as the `index` spread.

    Min-max normalization alone lands the far drone at exactly 1.0, which is the same phase as the
    near drone at 0.0 and so collapses the two ends of a sweep. Scaling by ``(n - 1) / n`` closes
    the gap the way ``rank / n`` does for `index`, and makes evenly spaced drones produce exactly
    the `index` offsets.

    The no-extent test is against ``_SPAN_REL_TOL`` times the coordinate magnitude, not against
    exact zero. A ring is the case that forces it: `form_circle` lays its slots out with ``cos``
    and ``sin``, so every radius is equal in exact arithmetic but equal only to within an ulp or
    two in floats. An exact-zero test takes the *non-degenerate* branch there and divides by that
    rounding noise, which spreads the offsets across the whole turn in whatever order the rounding
    fell — a random per-drone phase for `ripple_light` on the one formation it is most obviously
    authored over, and non-zero enough that `_spread`'s collapse warning stays silent too. With the
    tolerance a numerically degenerate span takes the same branch an exactly degenerate one does.

    Args:
        values: (n_sel,) coordinates or distances for the selected drones.

    Returns:
        (n_sel,) offsets in [0, 1). All zeros if every value is identical to within the tolerance.
    """
    span = values.max() - values.min()
    if span <= _SPAN_REL_TOL * np.abs(values).max():
        return np.zeros_like(values)
    n_sel = values.size
    return (values - values.min()) / span * (n_sel - 1) / n_sel


def _neighbour_ranks(points: NDArray) -> NDArray:
    """Rank points along a greedy nearest-neighbour walk over them (§7.3).

    Drone id order has no relationship to spatial order: every formation primitive routes through
    `_assign_positions`, a Hungarian assignment that returns whichever drone->slot permutation is
    cheapest to fly, so on a ring the ids land in an arbitrary rotation. Anything keyed by index is
    therefore spatially scrambled by construction. This walk recovers spatial order instead: start
    at the lexicographically smallest position, then repeatedly step to the nearest unvisited point.
    On a ring it recovers ring order, on a line line order, on a grid a snake.

    **The walk runs over the lexicographically sorted points, not the caller's order.** That is what
    makes the ranking independent of drone id, and it is not a nicety: a ring hands out an exact
    distance tie at the very first step, because the start's two ring-neighbours are equidistant
    from it. Resolving that tie by array order would let the id permutation choose which way round
    the ring the walk goes, and the whole point of the spread is that it does not care about ids.
    Sorting first makes "the lexicographically smaller candidate wins" the tie-break everywhere.

    The known failure mode (§7.3): on a clustered or highly symmetric formation the walk can exhaust
    one cluster and make a single long jump, putting two distant points adjacent in the ordering --
    one seam in an otherwise smooth sweep. Accepted, because a true shortest-path ordering is a
    travelling salesman problem and one seam is far smaller than the scrambling `index` produces.

    O(n^2) over the whole walk, which on ten drones is nothing (CLAUDE.md §2).

    Args:
        points: (m, 3) positions to rank. Ranks cover exactly these rows, so a caller ranking a
            selected subset passes the subset's rows, not the full swarm's.

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
    """Compute the per-drone phase offsets that turn one waveform into a family of effects (§7.3).

    The offsets are relative to the *selected subset*, so a chase over three drones runs across
    those three rather than across gaps in the full swarm. The one exception is
    "alternate_side", whose left/right split is defined against the swarm centroid (§7.1) so that
    it matches the `left`/`right` selectors.

    ``group_size`` quantizes the two ranked spreads, "neighbour" and "index"; the catalogue pairs it
    with `chase`, which is ranked (§10.2). The two differ only in what they rank by -- spatial order
    against id order -- so they share the bucketing exactly.

    Every other spread **rejects** a ``group_size`` above 1 rather than ignoring it. §7.3 defines
    the bucketing over ``rank_i``, and the spatial spreads carry a normalized coordinate instead of
    a rank: bucketing one evenly would silently turn a proportional sweep into an evenly stepped
    one on any unevenly spaced formation, which is a different effect, not a quantization of the
    same effect. §10.2 lists ``group_size`` as a plain `chase` parameter with no spread
    restriction, so the emitting model has no way to know the combination is inert -- and inert is
    the one outcome it must not be.

    This function is time-free: spatial spreads read the ``positions`` snapshot the caller froze at
    the look's start time (§7.3).

    Args:
        kind: One of "none", "neighbour", "index", "alternate_parity", "alternate_side", "radius",
            "x", "y", "z".
        mask: (n,) boolean mask of the selected drones.
        positions: (n, 3) frozen position snapshot.
        group_size: Drones per phase bucket for the "neighbour" and "index" spreads; 1 is per-drone.
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


def hue_to_wrgb(hue: NDArray, cfg: LightingConfig) -> NDArray:
    """Convert hues on the colour wheel to calibrated full-brightness WRGB (§7.5).

    A generated hue has no palette entry, so it carries its own calibration. The hue is first
    normalized to a constant channel sum -- constant nominal output across the wheel -- and
    ``channel_gain`` is applied *after*, as a correction multiplier for the LEDs not being equally
    efficient per channel. A gain of 0.8 on blue means the blue LED emits about 1/0.8 as much light
    per commanded unit, so commanding 0.8x equalizes it.

    **Order matters.** Normalizing *after* the gain would divide it straight back out for any hue
    that lands on a single channel, leaving pure blue at 255 instead of 204 and making a sweeping
    rainbow throb. With this order the six primaries reproduce their palette entries exactly, and
    it is the gain-corrected sum that is constant across the wheel, not the raw one.

    The RGB conversion is the closed form of ``colorsys.hsv_to_rgb`` at full saturation and value,
    written out so it vectorizes over an arbitrary hue shape. The white LED is never driven: the
    hue wheel cannot reach it, so ``channel_gain[0]`` is inert here.

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
    """One colour source covering a subset of the swarm on one or both decks (§7.5).

    Layers merge Latest-Takes-Precedence by their order within a look, which is why ``evaluate``
    leaves unselected rows at zero: the merge overwrites only the rows this layer's mask covers.

    ``params`` depends on ``kind``:

    - ``"named"``: ``{"color": str}`` -- a key into ``cfg.palette``.
    - ``"gradient"``: ``{"color_a": str, "color_b": str, "s": NDArray}`` where ``s`` is (n,) in
      [0, 1] **inclusive**, min-max normalized over the selected subset along the primitive's
      ``by`` axis, so the two extremes reproduce the endpoint colours exactly. Note this differs
      from ``spread_offsets``, which is half-open on purpose.
    - ``"cycled"``: ``{"period_s": float, "offsets": NDArray}`` -- the (n,) phase offsets from
      ``spread_offsets``, so a rainbow travels along whatever order a chase would.

    Attributes:
        mask: (n,) boolean mask of the drones this layer covers.
        decks: Which decks it applies to, a subset of ("top", "bot").
        kind: One of "named", "gradient" or "cycled".
        params: Kind-specific parameters, as above.
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
    """One brightness effect covering a subset of the swarm on one or both decks (§7.2, §7.3).

    Layers merge Highest-Takes-Precedence within a look, which is why ``evaluate`` leaves unselected
    rows at zero: the merge reduces with ``max`` over the layers covering each drone. ``light_on`` is
    one of these, as a ``"constant"`` layer contributing 1.0, and therefore dominates everything else
    covering the same drone. ``light_off`` is *not* — it is a post-reduction kill mask carried on the
    look, because a layer contributing 0 would be a no-op under ``max`` (§8.3).

    Attributes:
        mask: (n,) boolean mask of the drones this layer covers.
        decks: Which decks it applies to, a subset of ("top", "bot").
        kind: "constant", or one of the §7.2 waveform names: "sine", "square" or "ramp".
        period_s: Waveform period in seconds. Unused by "constant".
        duty: Fraction of each period the "square" waveform stays on. Unused by the others.
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


@dataclass(frozen=True)
class Look:
    """The complete lighting state from one emitted key until the next (§6, §8.4).

    A look is self-contained: the next look **replaces** it rather than layering onto it, so a colour
    that should persist has to be restated. That is the desk convention, and it keeps a look a unit a
    test can assert on.

    Attributes:
        t_start: Show time in seconds at which this look takes over.
        colour_layers: Colour sources in emission order, merged Latest-Takes-Precedence (§8.2).
        brightness_layers: Brightness effects, merged Highest-Takes-Precedence (§8.3).
        off_mask: (n, 2) boolean `light_off` kill mask, per deck in ``_DECKS`` order. Applied after
            the HTP reduction, so it beats every layer covering the same drone (§8.3).
        positions: (n, 3) position snapshot frozen at ``t_start`` (§7.3), which the §8.5 base colour
            is assigned in `neighbour` order over. The layers already carry their own resolved
            masks and offsets, so this is here for the base alone -- and because each look holds its
            own snapshot, the default hue wheel re-sorts as formations change instead of following
            the order `_assign_positions` happened to hand out. ``None`` means the look has no
            snapshot to order against, and the base falls back to id order (§8.5) -- which on a
            timeline carrying looks is only ever the case when it carries none of them at all,
            since the synthetic pre-show look borrows the first emitted one's snapshot.
    """

    t_start: float
    colour_layers: tuple[ColourLayer, ...]
    brightness_layers: tuple[BrightnessLayer, ...]
    off_mask: NDArray
    positions: NDArray | None = None


class LightingTimeline:
    """An ordered list of looks, evaluable at any show time (§5, §6).

    The timeline is a pure function of ``t`` plus the position snapshots already frozen into its
    layers, so the sim read-out (`evaluate` per frame) and the hardware read-out (a baked cue dict)
    see exactly the same thing, and neither needs a trajectory or a radio to test.
    """

    def __init__(self, looks: list[Look], n: int, t_end: float, cfg: LightingConfig) -> None:
        """Assemble the timeline.

        Args:
            looks: The emitted looks, in any order. They are sorted by ``t_start``; the sort is
                stable, so two looks landing on the same time resolve in favour of the later one.
            n: Number of drones in the swarm.
            t_end: Show duration in seconds. The blackout lands ``_BLACKOUT_LEAD_S`` before it.
            cfg: Lighting config, for the hue calibration, gamma and the dim floor.
        """
        self._n = n
        self._cfg = cfg
        self._t_blackout = t_end - _BLACKOUT_LEAD_S
        ordered = sorted(looks, key=lambda look: look.t_start)
        # A layerless look covering everything before the first emitted key, so the lookup never has
        # to special-case "no look yet": with no layers it evaluates to the §8.5 base state. It
        # borrows the first emitted look's snapshot, because the only thing a layerless look decides
        # is the base hue order (§8.5) and the two must agree: ordering the pre-show state by id and
        # the first look by its walk would fly the §7.3 scramble until that look and then re-colour
        # every drone at once. A timeline with no looks at all has nothing to borrow and keeps id
        # order, which is what makes §8.5's failure-safe claim exact.
        snapshot = ordered[0].positions if ordered else None
        base = Look(-np.inf, (), (), np.zeros((n, 2), dtype=bool), snapshot)
        self._looks = [base, *ordered]
        self._starts = np.array([look.t_start for look in self._looks])
        self._base_colours = [self._base_colour(look) for look in self._looks]

    def _base_colour(self, look: Look) -> NDArray:
        """Assign the §8.5 base hue wheel across the swarm in one look's `neighbour` order.

        The wheel itself is the construction `generate_default_colors` uses for today's deploy and
        viewer colours -- evenly spaced hues, HSV at full saturation and value, normalized to a
        constant channel sum -- with its separate blue dim now carried by ``channel_gain``. What
        changes is *which drone gets which hue*: ranking by the nearest-neighbour walk over the
        look's snapshot makes the default gradient read as a smooth wheel around the formation,
        where id order reads as the scramble `_assign_positions` hands out (§7.3).

        A look with no snapshot keeps id order. That is a timeline with no looks at all, where
        nothing was authored and there is nothing to order against -- so a lighting-less show still
        reproduces today's per-drone colouring exactly (§8.5). The pre-show base look is not that
        case: it borrows the first emitted look's snapshot, so the swarm is already in the order
        that look will hold it in and nothing re-colours when it takes over.

        Args:
            look: The look to colour for.

        Returns:
            (n, 4) full-brightness WRGB, one row per drone.
        """
        ranks = np.arange(self._n) if look.positions is None else _neighbour_ranks(look.positions)
        return hue_to_wrgb(ranks / self._n, self._cfg)

    def _look_index_at(self, t: float) -> int:
        """Find the index of the look covering ``t``.

        A sorted-boundary search rather than a linear scan over the looks, because the renderer
        calls this once per rendered frame. The index rather than the look itself, because the base
        colours are precomputed per look and read alongside it.

        Args:
            t: Show time in seconds.

        Returns:
            The index into ``self._looks`` of the look in force at ``t``, which is 0 -- the base
            look -- before the first emitted one.
        """
        return int(np.searchsorted(self._starts, t, side="right")) - 1

    def _merge_colour(self, look: Look, base: NDArray, t: float, deck: str) -> NDArray:
        """Merge one deck's colour layers Latest-Takes-Precedence (§8.2).

        The overwrite is driven by each layer's *mask*, never by whether its output is non-zero: an
        unselected row and a legitimately dark drone both read as zeros, and only the mask tells them
        apart. Testing values instead also mixes channels — a red layer over a green one would come
        out as both.

        Args:
            look: The look in force.
            base: That look's (n, 4) §8.5 base colour, from `_base_colour`.
            t: Show time in seconds.
            deck: Which deck to resolve.

        Returns:
            (n, 4) full-brightness WRGB. Drones no layer covers carry the §8.5 base colour.
        """
        colours = base.copy()
        for layer in look.colour_layers:
            if deck in layer.decks:
                colours[layer.mask] = layer.evaluate(t, self._cfg)[layer.mask]
        return colours

    def _merge_brightness(self, look: Look, t: float, deck: str, deck_idx: int) -> NDArray:
        """Merge one deck's brightness layers Highest-Takes-Precedence (§8.3, §8.5).

        Coverage is tracked from the layer masks, not from the merged values: a `square` layer in its
        off phase legitimately contributes 0, and reading the base state off the value would light
        those drones full-on during every off phase — inverting the blink.

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
        # The base is a fallback, not an HTP participant: it applies only where nothing else does.
        brightness[~covered] = 1.0
        brightness[look.off_mask[:, deck_idx]] = 0.0
        return brightness

    def evaluate(self, t: float) -> NDArray:
        """Evaluate every drone's colour on both decks at show time ``t`` (§7.4).

        The merged brightness is floored into ``brightness_steps`` buckets *before* the multiply,
        which is the brightness-axis twin of what ``hue_steps`` does for `rainbow`: it turns the
        continuous `sine` and `ramp` waveforms into piecewise-constant ones so `compile_cues` can
        dedup the runs (§9.1).

        ``b_min`` is applied to the merged brightness *before* that quantization. The two are
        different ideas — a hard dark floor and a resolution — and the floor has to act on the
        continuous value: quantizing first makes it inert, because the smallest non-zero bucket is
        larger than any ``b_min`` anyone would set (§7.4, §9.1).

        The quantizer floors rather than rounds. The two differ only at the *bottom* of the range —
        `floor` darkens ``[0, 1/steps)`` where `round` darkens only ``[0, 1/(2·steps))`` — and
        `floor` is chosen so quantization can only ever darken relative to intent, never brighten,
        which is the conservative direction for a physical output. It is not that flooring protects
        the top bucket: ``round(1.0 × steps) / steps`` is also exactly 1.0, so `light_on` and the
        §8.5 base state are undimmed either way.

        Args:
            t: Show time in seconds.

        Returns:
            (n, 2, 4) integral WRGB in [0, 255], with the deck axis ordered (top, bot).
        """
        if t >= self._t_blackout:
            # Unconditional, appended after the last look, and not the LLM's to override (§8.7).
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
        """Evaluate one deck as RGB in [0, 1], the convenience read-out for the 3D viewer (§9.2).

        The viewer has no separate white channel, so W folds into all three. The default is the top
        deck because that is the face a drone marker represents.

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
