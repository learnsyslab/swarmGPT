"""Run the synthesis loop across requests, repeats, and feedback arms, to beat down variance.

A single run per arm told us nothing: within-arm spread swamped between-arm differences. This
sweeps the grid so the comparison has enough runs behind it to mean something.

**Pre-registered before looking at the data**, so the verdict is not chosen to fit it:

- Primary outcome: ``deviation_max`` on the final iteration -- how far the filter had to move the
  swarm from what the model authored. This is the faithfulness cost the claim rests on. Lower wins.
- Secondary: the fraction of the model's own shape checks that pass on the final iteration.
- Tertiary: ``min_sep_norm`` on the final iteration, i.e. how much room the filter ended up with.
- The claim survives only if `absolute` and/or `relative` beat `categorical` on the primary
  outcome by more than the within-arm spread. If they do not, the claim is dead -- that is the
  point of running it.

Results append to JSONL as each run finishes, so an interrupted sweep keeps what it had.

    pixi run python experiments/ablation_feedback.py --repeats 3
"""

from __future__ import annotations

import argparse
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

# Chosen to span the gap classes the decoy probe offers, not just the double helix, so no single
# geometry drives the result.
REQUESTS = [
    "a double helix: two counter-rotating strands winding upward around a common axis",
    "a heart outline standing upright, readable from the audience",
    "a firework: the swarm collapses to a point then bursts outward",
    "half the swarm orbits the other half, which holds still",
    "a travelling ripple spreading outward from one drone through the rest",
    "a flat wall of drones that sweeps across the arena",
]


def starting_positions() -> np.ndarray:
    """Read the active swarm's dock positions, at the show start height."""
    settings = yaml.safe_load((ROOT / "swarm_gpt/data/settings.yaml").read_text())
    raw = toml.load(ROOT / "swarm_gpt/data/drones.toml")
    positions = np.array([raw[name]["pos"] for name in raw["active"]], dtype=float)
    positions[:, 2] = settings["starting_height"]
    return positions


def outcome(history: list) -> dict[str, Any]:
    """Reduce one run's iterations to the pre-registered outcome measures."""
    scored = [r for r in history if r.metrics is not None]
    if not scored:
        return {"ok": False}
    final = scored[-1]
    checks = final.checks
    return {
        "ok": True,
        "n_iterations": len(history),
        "n_scored": len(scored),
        "deviation_max_m": final.metrics["deviation_max_m"],
        "deviation_mean_m": final.metrics["deviation_mean_m"],
        "min_sep_norm": final.metrics["min_sep_norm"],
        "authored_min_sep_norm": final.metrics["authored_min_sep_norm"],
        "check_pass_fraction": (
            sum(c["ok"] for c in checks) / len(checks) if checks else float("nan")
        ),
        "n_checks": len(checks),
        "converged": final.closing_verdict == "keep",
        "compile_failures": sum(1 for r in history if r.error is not None),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=3, help="Runs per (request, arm) cell")
    parser.add_argument("--iters", type=int, default=4, help="Iterations per run")
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
                        duration_s=12.0,
                    )
                    try:
                        history = loop.run(request, max_iterations=args.iters)
                        result = outcome(history)
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
                        "[%d/%d] %-11s r%d dev_max=%s checks=%s  %s",
                        done,
                        total,
                        arm,
                        repeat,
                        f"{result['deviation_max_m']:.2f}" if result.get("ok") else "-",
                        f"{result['check_pass_fraction']:.0%}" if result.get("ok") else "-",
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
    print(f"{'arm':<12}{'deviation_max (m), lower better':<44}{'checks pass':<26}converged")
    for arm in dict.fromkeys(r["arm"] for r in rows):
        cell = [r for r in ok if r["arm"] == arm]
        dev = [r["deviation_max_m"] for r in cell]
        chk = [
            r["check_pass_fraction"]
            for r in cell
            if r["check_pass_fraction"] == r["check_pass_fraction"]
        ]
        conv = sum(r["converged"] for r in cell)
        print(f"{arm:<12}{_spread(dev):<44}{_spread(chk):<26}{conv}/{len(cell)}")

    print(
        f"\nper request (deviation_max median):\n{'request':<44}"
        + "".join(f"{a:<14}" for a in ARMS)
    )
    for request in dict.fromkeys(r["request"] for r in rows):
        line = f"{request[:42]:<44}"
        for arm in ARMS:
            dev = [r["deviation_max_m"] for r in ok if r["request"] == request and r["arm"] == arm]
            line += f"{statistics.median(dev):<14.2f}" if dev else f"{'-':<14}"
        print(line)


if __name__ == "__main__":
    main()
