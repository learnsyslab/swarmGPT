"""Extract benchmark choreographies to spline artifacts for the MAPF benchmark.

Runs on the swarmGPT ``swarmgpt2-spline-foundation`` branch. Feeds a hand-authored primitive text
through ``Choreographer.response2trajectory`` -- the native min-snap spline choreographer, no LLM /
OpenAI key needed -- and writes ``<name>.spline.npz``, the input format of
``mapf_benchmarking.solvers.amcont_wrapper.bake_choreo_reference``.

The per-drone piecewise Beziers are taken STRAIGHT off the returned ``PiecewiseSpline`` segments
(control points, degree, time window). There is no resampling and no least-squares refit, so the
artifact is the choreographer's own spline-1, losslessly. Do not "modernise" this onto
``response2waypoints``: that renders the same authored text as linearly interpolated waypoint
chords, which is a different intended path (measured metres apart mid-show) and not what swarmgpt2
will fly.

    python scripts/extract_spline_choreo.py --list
    python scripts/extract_spline_choreo.py --name stack_swap_D10 --out /tmp/choreo
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from swarm_gpt.core.choreographer import Choreographer
from swarm_gpt.utils.music_analyzer import Bar, Beat, Segment, SongStructure

BPM = 120
BEAT_S = 60.0 / BPM
BAR_S = 4 * BEAT_S  # 2.0 s at 120 BPM
HZ = 100  # dense sample rate for the conflict report only; not stored as the reference
D_MIN = 0.30  # physical separation target carried by the artifact [m]

LOW = "[1, 2, 3, 4, 5]"
HIGH = "[6, 7, 8, 9, 10]"
ALL = "[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]"

# ============================================================================
# AUTHOR CHOREOGRAPHIES HERE.
#
# Required downbeats on the structure below are s1b1t1, s1b3t1, s2b1t1, s2b3t1, s2b5t1, s2b7t1.
# A TRANSITION must sit on an optional slot between each consecutive pair of primitives, and may
# never sit on a required key. Several primitives in one slot: separate with ';'.
#
# Primitives on this branch: form_circle, rotate, helix, spiral, spiral_speed, wave, twister,
# form_star, form_cone, zig_zag, move_z, center. There is NO swap and NO move.
#
# Generating real conflict here is not a matter of picking a dramatic primitive. Every FORMATION
# primitive routes drones through `_assign_positions`, a Hungarian assignment over a min-snap cost
# matrix -- it picks the cheapest drone-to-slot permutation, which is almost always the
# crossing-free one. That is why the stock diametric_D10 show, despite containing a 180 degree
# rotation, has zero pairs closer than d_min: the assignment quietly resolved the conflict before
# the solver ever saw it.
#
# `move_z` is the exception: it takes an explicit drone-id list and translates exactly those
# drones, bypassing the assignment entirely. Two subset `form_circle` calls at the same radius put
# drone k and drone k+5 at the SAME (x, y) on different altitudes -- both subsets get the same
# `linspace(0, 2*pi, 5, endpoint=False)` angles -- and a simultaneous opposing `move_z` then drives
# them through each other vertically. Splitting the radii instead holds a known horizontal offset
# through the crossing, which grades the difficulty without the degenerate coincidence.
# ============================================================================
CHOREOGRAPHIES: dict[str, str] = {
    # Two co-located rings trade altitude twice. Drone k and k+5 share an (x, y) column, so each
    # exchange is an exact vertical head-on: five coincident pairs, collision direction degenerate.
    "stack_swap_D10": f"""\
song_mood: "stacked ring exchange"
choreography_plan: "two co-located rings trade altitude twice, then regroup"
choreography:
  s1b1t1: form_circle({LOW}, 130, 80, 1.5); form_circle({HIGH}, 130, 140, 1.5)
  s1b2t1: TRANSITION
  s1b3t1: move_z({LOW}, 60); move_z({HIGH}, -60)
  s1b4t1: TRANSITION
  s2b1t1: move_z({LOW}, -60); move_z({HIGH}, 60)
  s2b2t1: TRANSITION
  s2b3t1: rotate(72, 'z')
  s2b4t1: TRANSITION
  s2b5t1: helix(3, 40, 100)
  s2b6t1: TRANSITION
  s2b7t1: center({ALL})
  END
""",
    # The same exchange with the rings 25 cm apart in radius, so every pair holds a known
    # horizontal offset through the crossing. Sustained conflict, well-defined collision direction.
    "stack_near_D10": f"""\
song_mood: "offset stacked ring exchange"
choreography_plan: "two rings 25 cm apart in radius trade altitude twice, then regroup"
choreography:
  s1b1t1: form_circle({LOW}, 130, 80, 1.5); form_circle({HIGH}, 155, 140, 1.5)
  s1b2t1: TRANSITION
  s1b3t1: move_z({LOW}, 60); move_z({HIGH}, -60)
  s1b4t1: TRANSITION
  s2b1t1: move_z({LOW}, -60); move_z({HIGH}, 60)
  s2b2t1: TRANSITION
  s2b3t1: rotate(72, 'z')
  s2b4t1: TRANSITION
  s2b5t1: helix(3, 40, 100)
  s2b6t1: TRANSITION
  s2b7t1: rotate(-72, 'z')
  END
""",
    # Convergent rather than crossing: a wide ring collapses to `center`'s tight one, re-forms as a
    # cone, then inverts that cone. Assignment cannot help on the collapses -- every drone is
    # inbound at once, so the conflict is in the flow, not the permutation.
    "collapse_D10": f"""\
song_mood: "repeated collapse"
choreography_plan: "wide ring -> centre -> cone -> centre -> inverted cone -> twister"
choreography:
  s1b1t1: form_circle({ALL}, 180, 110, 1.5)
  s1b2t1: TRANSITION
  s1b3t1: center({ALL})
  s1b4t1: TRANSITION
  s2b1t1: form_cone(60, 60, False, 1.5)
  s2b2t1: TRANSITION
  s2b3t1: center({ALL})
  s2b4t1: TRANSITION
  s2b5t1: form_cone(60, 60, True, 1.5)
  s2b6t1: TRANSITION
  s2b7t1: twister(3, 15, 20)
  END
""",
}


def _make_bar(bar_id: int, bar_start: float, n_beats: int = 4) -> Bar:
    """Build one bar of ``n_beats`` evenly spaced beats starting at ``bar_start``."""
    beats = [
        Beat(id=i + 1, time_s=bar_start + i * BEAT_S, position_in_bar=i + 1) for i in range(n_beats)
    ]
    return Bar(id=bar_id, start_s=bar_start, beats=beats)


def build_structure() -> SongStructure:
    """Return the mock 4-bar intro + 8-bar main structure the choreographies are authored against."""
    seg1_bars = [_make_bar(b, (b - 1) * BAR_S) for b in range(1, 5)]
    seg2_start = 4 * BAR_S
    seg2_bars = [_make_bar(b, seg2_start + (b - 1) * BAR_S) for b in range(1, 9)]
    return SongStructure(
        schema_version=2,
        source_path="mock",
        song_sha256="0" * 64,
        analyzer="mock",
        bpm=BPM,
        segments=[
            Segment(id=1, label="intro", start_s=0.0, end_s=seg2_start, bars=seg1_bars),
            Segment(
                id=2, label="main", start_s=seg2_start, end_s=seg2_start + 8 * BAR_S, bars=seg2_bars
            ),
        ],
    )


def conflict_report(pts_cm: np.ndarray, times: np.ndarray) -> float:
    """Print how hard the intended (unfiltered) show is and return its min separation in metres.

    Args:
        pts_cm: Intended positions, shape ``(D, L, 3)`` in cm.
        times: Sample times, shape ``(L,)``.

    Returns:
        The minimum pairwise separation over the show, in metres.
    """
    p = pts_cm / 100.0
    gaps = np.linalg.norm(p[:, None] - p[None, :], axis=-1)
    d = p.shape[0]
    gaps[np.arange(d), np.arange(d)] = np.inf
    per_t = gaps.min(axis=(0, 1))
    speed = np.linalg.norm(np.gradient(p, times, axis=1), axis=-1)
    accel = np.linalg.norm(np.gradient(np.gradient(p, times, axis=1), times, axis=1), axis=-1)
    worst = np.unravel_index(np.argmin(gaps), gaps.shape)
    print("\n=== conflict analysis ===")
    print(f"  intended min separation : {per_t.min():.3f} m at t={times[per_t.argmin()]:.2f}s")
    print(f"  worst pair              : drones {worst[0] + 1} and {worst[1] + 1}")
    print(f"  time below d_min={D_MIN:.2f} m : {100 * (per_t < D_MIN).mean():.1f}%")
    print(
        f"  conflicting pairs       : {int((gaps.min(axis=2) < D_MIN).sum() // 2)}"
        f" of {d * (d - 1) // 2}"
    )
    print(f"  peak speed / accel      : {speed.max():.2f} m/s, {accel.max():.2f} m/s^2")
    return float(per_t.min())


def _provenance() -> tuple[str, str]:
    """Return the ``(branch, sha)`` of this checkout, with ``-dirty`` appended when uncommitted."""
    repo = Path(__file__).resolve().parents[1]
    git = ["git", "-C", str(repo)]

    def run(*args: str) -> str:
        return subprocess.run([*git, *args], capture_output=True, text=True, check=True).stdout

    branch = run("rev-parse", "--abbrev-ref", "HEAD").strip()
    if branch == "HEAD":  # detached, e.g. the extraction worktree
        branch = run("name-rev", "--name-only", "HEAD").strip()
    sha = run("rev-parse", "HEAD").strip()
    if run("status", "--porcelain").strip():
        sha += "-dirty"
    return branch, sha


def extract(name: str, out_dir: Path) -> Path:
    """Build one named choreography and write its spline artifact.

    Args:
        name: Key into :data:`CHOREOGRAPHIES`.
        out_dir: Directory to write ``<name>.spline.npz`` into.

    Returns:
        The path written.
    """
    text = CHOREOGRAPHIES[name]
    choreo = Choreographer(use_motion_primitives=True)
    trajectories = choreo.response2trajectory(text, build_structure())

    ids = sorted(trajectories)
    d = len(ids)
    t0 = min(t.t0 for t in trajectories.values())
    t1 = max(t.t1 for t in trajectories.values())
    times = np.linspace(t0, t1, int(round((t1 - t0) * HZ)) + 1)
    pts = np.stack([trajectories[i].evaluate(times) for i in ids])  # (D, L, 3) cm

    print(f"{name}: {d} drones, span [{t0:.2f}, {t1:.2f}] s")
    for i in ids[:2]:
        segs = trajectories[i].segments
        print(f"  drone {i}: {len(segs)} segs, degrees {[s.degree for s in segs]}")
    conflict_report(pts, times)

    # Straight off the choreographer's own spline; ragged across drones, hence object arrays.
    seg_cp, seg_tw = np.empty(d, dtype=object), np.empty(d, dtype=object)
    for k, i in enumerate(ids):
        segs = trajectories[i].segments
        seg_cp[k] = [np.asarray(s.control_points, dtype=float) for s in segs]
        seg_tw[k] = [(float(s.t0), float(s.t1)) for s in segs]

    branch, sha = _provenance()
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{name}.spline.npz"
    np.savez(
        out,
        drone_ids=np.asarray(ids),
        times=times,
        pts=pts,
        seg_cp=seg_cp,
        seg_tw=seg_tw,
        d_min=np.asarray(D_MIN),
        branch=np.asarray(branch),
        commit_sha=np.asarray(sha),
        primitive_script=np.asarray(text),
    )
    print(f"\nSaved -> {out}")
    return out


def main() -> None:
    """Parse arguments and extract the selected choreography."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", help="choreography to extract")
    parser.add_argument("--out", type=Path, default=Path("scripts"), help="output directory")
    parser.add_argument("--list", action="store_true", help="list the available choreographies")
    parser.add_argument("--all", action="store_true", help="extract every choreography")
    args = parser.parse_args()

    if args.list:
        for k, v in CHOREOGRAPHIES.items():
            plan = next(ln for ln in v.splitlines() if ln.startswith("choreography_plan"))
            print(f"{k:16s} {plan.split(':', 1)[1].strip()}")
        return
    if not args.all and args.name not in CHOREOGRAPHIES:
        parser.error(f"unknown choreography {args.name!r}; --list shows the options")
    for n in CHOREOGRAPHIES if args.all else [args.name]:
        extract(n, args.out)


if __name__ == "__main__":
    main()
