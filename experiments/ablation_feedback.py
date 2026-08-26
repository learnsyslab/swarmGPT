"""Run the synthesis loop across requests, repeats, and feedback arms, to beat down variance.

A single run per arm told us nothing: within-arm spread swamped between-arm differences. This
sweeps the grid so the comparison has enough runs behind it to mean something.

**Pre-registered before looking at the data**, so the verdict is not chosen to fit it. The 54-run
sweep of 2026-08 measured trajectory authoring, which no longer exists; this is the same question
asked of shape authoring, and the outcomes are restated because the failure mode changed. It used
to be "the filter has to drag the swarm a long way"; it is now "it takes N tries to write a shape
that fits the room".

- **Primary: ``iterations_to_clear``** -- how many turns until the first candidate that flies with
  no step inside the collision envelope, censored at ``--iters`` when none does. Lower wins. This
  is the outcome that decides whether the loop is usable live, and it is the one the arms should
  move if magnitudes help at all.
- **Secondary: ``cleared``** -- whether any candidate cleared both gates. A run that never gets
  there is a failure however good its last deviation looked.
- **Tertiary: ``deviation_max_m``** on the best clearing candidate, kept because it was the 2026-08
  primary and keeps the two sweeps commensurable.
- ``converged`` -- whether the model itself closed on "keep" -- is reported but is **not** an
  outcome. It measures the model's willingness to stop, and `gate()` no longer defers to it.
- The claim survives only if `absolute` and/or `relative` beat `categorical` on the primary
  outcome by more than the within-arm spread. If they do not, the claim is dead -- that is the
  point of running it.

Every iteration's feedback goes through the arm, whichever stage it reached: under shape authoring
most candidates are rejected by a screen rather than by the filter, so rendering screen rejections
arm-independently would leave the manipulation applying to a minority of turns. `screen` is on, so
the measured path is the one the app ships.

Results append to JSONL as each run finishes, so an interrupted sweep keeps what it had.

    pixi run python experiments/ablation_feedback.py --repeats 3 --duration 8.18
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
import statistics
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import toml
import yaml

from swarm_gpt.synth.feedback import ARMS
from swarm_gpt.synth.loop import SynthesisLoop

logger = logging.getLogger("ablation")

ROOT = Path(__file__).resolve().parents[1]

# Static formations, because a shape primitive is a pose and motion comes from composing it. Each
# is known to be buildable in this arena, so a failure is the loop's and not the geometry's -- the
# 2026-08 sweep spent three runs on a counter-rotating double helix that is impossible at any
# separation. They span open and closed curves, nested and single outlines, and both the easy
# horizontal plane and the expensive vertical one.
REQUESTS = [
    "a heart outline",
    "a five-pointed star outline",
    "a crescent moon",
    "two concentric rings, the inner one half the radius of the outer",
    "a spiral winding outward from the centre",
    "the outline of a cube",
]


def starting_positions() -> np.ndarray:
    """Read the active swarm's dock positions, at the show start height."""
    settings = yaml.safe_load((ROOT / "swarm_gpt/data/settings.yaml").read_text())
    raw = toml.load(ROOT / "swarm_gpt/data/drones.toml")
    positions = np.array([raw[name]["pos"] for name in raw["active"]], dtype=float)
    positions[:, 2] = settings["starting_height"]
    return positions


def outcome(history: list, max_iterations: int) -> dict[str, Any]:
    """Reduce one run's iterations to the pre-registered outcome measures.

    ``iterations_to_clear`` is censored at ``max_iterations`` when no candidate ever cleared, so
    the arms stay comparable whether or not they got there.
    """
    if not history:
        return {"ok": False}
    flown = [r for r in history if r.stage == "measured"]
    clear = [r for r in flown if r.metrics["steps_inside_envelope"] == 0]
    best = min(clear, key=lambda r: r.metrics["deviation_max_m"]) if clear else None
    stages = collections.Counter(r.stage for r in history)
    return {
        "ok": True,
        "n_iterations": len(history),
        "n_flown": len(flown),
        "cleared": bool(clear),
        "iterations_to_clear": clear[0].index if clear else max_iterations,
        "censored": not clear,
        "deviation_max_m": best.metrics["deviation_max_m"] if best else None,
        "deviation_mean_m": best.metrics["deviation_mean_m"] if best else None,
        "min_sep_norm": best.metrics["min_sep_norm"] if best else None,
        "authored_min_sep_norm": best.metrics["authored_min_sep_norm"] if best else None,
        # Where the arms bit: a run rejected on geometry every turn never reached the filter.
        "stage_counts": dict(stages),
        "converged": history[-1].closing_verdict == "keep",
        "compile_failures": sum(1 for r in history if r.error is not None),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=3, help="Runs per (request, arm) cell")
    parser.add_argument("--iters", type=int, default=6, help="Iterations per run")
    parser.add_argument(
        "--duration",
        type=float,
        default=8.18,
        help="Window the primitive gets, seconds. Default is Fearless2's narrowest required-key "
        "gap: verifying against a window no show will give it certifies nothing.",
    )
    parser.add_argument("--model", default="gpt-5.6-luna", help="LLM model id")
    parser.add_argument("--arms", nargs="*", default=list(ARMS), help="Feedback arms to sweep")
    parser.add_argument("--requests", nargs="*", default=REQUESTS, help="Motion requests")
    parser.add_argument("--out", type=Path, default=ROOT / "synth_runs", help="Log directory")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> Path:
    """Sweep the grid and report per-arm distributions.

    Returns:
        Path of the written JSONL log.
    """
    args = parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    logger.setLevel(logging.INFO)
    settings = yaml.safe_load((ROOT / "swarm_gpt/data/settings.yaml").read_text())
    start_pos = starting_positions()

    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"ablation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    total = len(args.arms) * len(args.requests) * args.repeats
    done = 0
    rows: list[dict[str, Any]] = []

    with open(path, "w") as f:
        f.write(json.dumps({"model": args.model, "repeats": args.repeats, "grid": total}) + "\n")
        f.flush()
        for request in args.requests:
            for arm in args.arms:
                for repeat in range(args.repeats):
                    done += 1
                    loop = SynthesisLoop(
                        settings=settings,
                        start_pos_m=start_pos,
                        arm=arm,
                        model_id=args.model,
                        duration_s=args.duration,
                        screen=True,
                    )
                    try:
                        history = loop.run(request, max_iterations=args.iters)
                        result = outcome(history, args.iters)
                        iterations = [asdict(r) for r in history]
                    except Exception as e:  # a run that dies must not take the sweep with it
                        logger.warning("run failed (%s, repeat %d): %s", arm, repeat, e)
                        result, iterations = {"ok": False, "error": str(e)}, []
                    row = {
                        "request": request,
                        "arm": arm,
                        "repeat": repeat,
                        **result,
                        "iterations": iterations,
                    }
                    rows.append(row)
                    f.write(json.dumps(row) + "\n")
                    f.flush()
                    logger.info(
                        "[%d/%d] %-11s r%d cleared=%s in %s  %s",
                        done,
                        total,
                        arm,
                        repeat,
                        "y" if result.get("cleared") else "n",
                        result.get("iterations_to_clear", "-"),
                        request[:40],
                    )

    report(rows)
    print(f"\nlog: {path}")
    return path


def _spread(values: list[float]) -> str:
    """Median and interquartile-ish range, robust to the small n a sweep like this affords."""
    if not values:
        return "-"
    if len(values) < 4:
        return f"{statistics.median(values):.2f} (n={len(values)}, min {min(values):.2f} max {max(values):.2f})"
    lo, hi = np.percentile(values, [25, 75])
    return f"{statistics.median(values):.2f} (IQR {lo:.2f}-{hi:.2f}, n={len(values)})"


def report(rows: list[dict[str, Any]]) -> None:
    """Print the per-arm distributions the pre-registered comparison reads."""
    ok = [r for r in rows if r.get("ok")]
    print(f"\n{len(ok)}/{len(rows)} runs completed\n")
    header = (
        f"{'arm':<12}{'iterations to clear (primary, lower better)':<48}"
        f"{'cleared':<10}{'deviation_max (m)':<32}model kept"
    )
    print(header)
    print("-" * len(header))
    for arm in dict.fromkeys(r["arm"] for r in rows):
        cell = [r for r in ok if r["arm"] == arm]
        iters = [r["iterations_to_clear"] for r in cell]
        dev = [r["deviation_max_m"] for r in cell if r["deviation_max_m"] is not None]
        cleared = sum(r["cleared"] for r in cell)
        kept = sum(r["converged"] for r in cell)
        print(
            f"{arm:<12}{_spread(iters):<48}{f'{cleared}/{len(cell)}':<10}"
            f"{_spread(dev):<32}{kept}/{len(cell)}"
        )
    censored = sum(r["censored"] for r in ok)
    if censored:
        print(f"\n{censored} run(s) never cleared, counted at the iteration cap.")

    # Where each arm's feedback was doing its work. A run rejected on geometry every turn never
    # reached the filter, so an arm that shifts this distribution is shifting what it teaches.
    stages = sorted({s for r in ok for s in r["stage_counts"]})
    print(f"\niterations by stage:\n{'arm':<12}" + "".join(f"{s:<12}" for s in stages))
    for arm in dict.fromkeys(r["arm"] for r in rows):
        cell = [r for r in ok if r["arm"] == arm]
        line = f"{arm:<12}"
        for stage in stages:
            line += f"{sum(r['stage_counts'].get(stage, 0) for r in cell):<12}"
        print(line)

    print(
        f"\nper request (iterations to clear, median):\n{'request':<44}"
        + "".join(f"{a:<14}" for a in ARMS)
    )
    for request in dict.fromkeys(r["request"] for r in rows):
        line = f"{request[:42]:<44}"
        for arm in ARMS:
            cell = [
                r["iterations_to_clear"] for r in ok if r["request"] == request and r["arm"] == arm
            ]
            line += f"{statistics.median(cell):<14.1f}" if cell else f"{'-':<14}"
        print(line)


if __name__ == "__main__":
    main()
