"""The hardware read-out: bake a `LightingTimeline` into `DroneSwarm` colour cues (spec §9.1).

The cue interface is `{uri: {time: wrgb}}` — step events, no interpolation — and `_stream_reference`
drains **at most one cue per deck per `1 / col_freq` tick, in order, and never drops**
(`drone_swarm.py:611-618`). A cue list denser than that therefore plays back *slowed*, and the lag
accumulates for the remainder of the show, so the lights desynchronize from the music permanently
rather than glitching once (§3.1).

That failure mode is eliminated structurally rather than by care: sample the timeline on a uniform
grid at exactly `col_freq`, then drop consecutive duplicates. The cue list can never be denser than
the consumer, whatever the primitives above it do; dedup is what keeps the common case (long holds)
at ~1 cue rather than `col_freq x duration`.

Pure NumPy plus stdlib, like the two modules it sits on: no backend, no simulator, no JAX.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from swarm_gpt.core.lighting import _BLACKOUT_LEAD_S

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from swarm_gpt.core.lighting import LightingTimeline


def _sample_times(col_freq: float, t_end: float) -> NDArray:
    """Build the sample grid, terminated by the unconditional blackout instant (§8.7, §9.1).

    The grid is uniform at ``col_freq``, which is what bounds the cue rate. The blackout at
    ``t_end - 0.1`` is appended *explicitly*: the timeline implements it as an early return, so it
    guarantees zeros from that instant but a grid anchored at 0 lands on it only by luck. Grid ticks
    the blackout would crowd are dropped first — appending a cue less than one period after the tick
    before it would violate the §3.1 spacing guarantee with the very cue added to satisfy §8.7.

    A show has to leave at least one whole period before the blackout, or the blackout crowds out
    the tick at 0 as well and the grid opens *after* it — at ``t_end - 0.1``, which for a show
    shorter than the blackout lead is negative. `DroneSwarm` would be handed a cue at a negative
    time, and the §9.3 browser contract (every list non-empty and starting at ``t = 0``) has no
    reading under which that holds. So it raises rather than clamping: there is no show to salvage,
    and both ways of salvaging one lie about it — moving the blackout later fabricates a duration
    the caller did not ask for, and the blackout is what keeps the drones from landing lit, while
    clamping the times to 0 emits a blackout at 0 and calls a dark show a compiled one.

    Args:
        col_freq: Maximum colour-cue rate in Hz, matching ``DroneSwarm.col_freq``.
        t_end: Show duration in seconds.

    Returns:
        Strictly increasing sample times, at least ``1 / col_freq`` apart, opening at 0 and ending
        at the blackout.

    Raises:
        ValueError: If the show ends less than one cue period after the blackout instant.
    """
    period = 1.0 / col_freq
    t_blackout = t_end - _BLACKOUT_LEAD_S
    if t_blackout < period:
        raise ValueError(
            f"A {t_end} s show is too short to compile lighting cues: it leaves {t_blackout} s "
            f"before the §8.7 blackout, under the {period} s cue period at {col_freq} Hz"
        )
    ticks = np.arange(int(np.floor(t_blackout * col_freq)) + 1) / col_freq
    return np.append(ticks[t_blackout - ticks >= period], t_blackout)


def compile_cues(
    timeline: LightingTimeline, uris: list[str], col_freq: float, t_end: float
) -> tuple[dict[str, dict[float, NDArray]], dict[str, dict[float, NDArray]]]:
    """Bake a lighting timeline into per-deck colour cues for `DroneSwarm` (§9.1).

    The two returned dicts drop straight into
    ``execute_choreography(color_top=..., color_bot=...)``.

    Args:
        timeline: The lighting timeline, already carrying its frozen position snapshots.
        uris: Radio URI per drone, in the timeline's drone-index order.
        col_freq: Maximum colour-cue rate in Hz, matching ``DroneSwarm.col_freq``.
        t_end: Show duration in seconds.

    Returns:
        ``(color_top, color_bot)``, each ``{uri: {time: (4,) WRGB}}``.

    Raises:
        ValueError: If ``uris`` does not cover the swarm the timeline was built for -- zipping
            short would silently leave the uncovered drones dark for the whole show -- or if the
            show is too short for the sample grid to open at 0 (see `_sample_times`).
    """
    times = _sample_times(col_freq, t_end)
    frames = np.stack([timeline.evaluate(float(t)) for t in times])  # (n_samples, n, 2, 4)
    n = frames.shape[1]
    if len(uris) != n:
        raise ValueError(f"Got {len(uris)} URIs for a {n}-drone lighting timeline")
    top: dict[str, dict[float, NDArray]] = {}
    bot: dict[str, dict[float, NDArray]] = {}
    # The deck axis is ordered (top, bot) everywhere in the lighting layer (§6).
    for deck_idx, cues in enumerate((top, bot)):
        for i, uri in enumerate(uris):
            track = frames[:, i, deck_idx]
            changed = np.ones(times.size, dtype=bool)
            changed[1:] = np.any(track[1:] != track[:-1], axis=1)
            cues[uri] = {float(times[k]): track[k] for k in np.flatnonzero(changed)}
    return top, bot
