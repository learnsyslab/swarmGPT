"""Synthesize one motion primitive, verify it, and persist it as a loadable library entry.

The single entry point for the synthesis loop. Every run writes a JSONL log whatever the outcome;
a run that clears every gate is also promoted to `results/synthesized/<name>.json`, from where
`PrimitiveManifest.from_payload` reloads it into the schema, the prompt, and `primitive_by_name`.

Two gates, neither of which the model may overrule: the pre-solve screen (its own waypoints must
be collision-free and flyable) and the filter (the flown trajectory must clear the envelope).
Whether the result looks like what was asked for is a human call, reported, not gated.

    pixi run python experiments/synth_to_library.py \
        --request "a double helix: two strands half a turn apart at every height, both winding \
upward the same way around a common axis" --iters 14
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import toml
import yaml

from swarm_gpt.synth.feedback import ARMS
from swarm_gpt.synth.loop import Iteration, SynthesisLoop

logger = logging.getLogger("synth")

ROOT = Path(__file__).resolve().parents[1]


def starting_positions() -> np.ndarray:
    """Read the active swarm's dock positions, at the show start height."""
    settings = yaml.safe_load((ROOT / "swarm_gpt/data/settings.yaml").read_text())
    raw = toml.load(ROOT / "swarm_gpt/data/drones.toml")
    positions = np.array([raw[name]["pos"] for name in raw["active"]], dtype=float)
    positions[:, 2] = settings["starting_height"]
    return positions


def accepted(history: list[Iteration]) -> Iteration | None:
    """Return the iteration the model closed on with "keep", or None if it never accepted one."""
    if not history:
        return None
    last = history[-1]
    if last.closing_verdict == "keep" and last.error is None:
        return last
    return None


def unsafe_reason(record: Iteration) -> str | None:
    """Return why an accepted primitive must not be promoted, or None if it may be.

    The model owns the verdict inside the loop and has been observed keeping a trajectory whose
    closest approach was inside the collision envelope. Promotion does not defer to it: a
    primitive that flies the swarm through itself is not a library entry.
    """
    if record.stage != "measured":
        return f"never reached the safety filter (stopped at stage {record.stage!r})"
    metrics = record.metrics or {}
    # Judged on what flew, not on whether the solver converged. `failed_solves` counts steps where
    # axswarm hit max_iters with its K-step horizon still unsatisfied, but only the first step of
    # each horizon is ever executed and the next tick re-solves, so a run can carry failures and
    # still fly clean. It is reported, never gated on.
    if metrics["steps_inside_envelope"] == 0:
        return None
    return (
        f"the flown trajectory is inside the collision envelope on "
        f"{metrics['steps_inside_envelope']} of {metrics['n_steps']} steps, closest approach "
        f"{metrics['min_sep_norm']:.3f}x"
    )


def document(
    record: Iteration, *, request: str, arm: str, model: str, duration_s: float, n_drones: int
) -> dict[str, Any]:
    """Assemble the persisted entry: the manifest plus what it was measured under."""
    return {
        "manifest": record.manifest,
        "args": record.args,
        "provenance": {
            "request": request,
            "arm": arm,
            "model": model,
            "iterations": record.index,
            "duration_s": duration_s,
            "n_drones": n_drones,
            "synthesized_at": datetime.now().isoformat(timespec="seconds"),
        },
        "metrics": record.metrics,
        "checks": record.checks,
        "reasoning": record.closing_reasoning,
    }


def write_run_log(
    history: list[Iteration], status: str, *, args: argparse.Namespace, n_drones: int
) -> Path:
    """Write every iteration to JSONL whatever the outcome, and return the path.

    Promotion is gated; capture is not. A run that is refused still cost the same API time, and
    the refusal is usually the interesting part.
    """
    args.runs.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = args.runs / f"promote_{args.arm}_{stamp}.jsonl"
    with open(path, "w") as f:
        header = {
            "request": args.request,
            "arm": args.arm,
            "model": args.model,
            "duration_s": args.duration,
            "n_drones": n_drones,
            "max_iterations": args.iters,
            "status": status,
        }
        f.write(json.dumps(header) + "\n")
        for record in history:
            f.write(json.dumps(asdict(record)) + "\n")
    return path


def render_table(history: list[Iteration]) -> str:
    """One row per iteration: how far it got, what it broke, and how its two verdicts landed."""
    header = f"\n{'it':<4}{'stage':<10}{'authored':>9}{'flown':>8}{'failed':>8}{'own':>7}  verdict"
    rows = [header, "-" * len(header)]
    for r in history:
        m = r.metrics or {}
        own = f"{sum(c['ok'] for c in r.checks)}/{len(r.checks)}" if r.checks else "-"
        rows.append(
            f"{r.index:<4}{r.stage:<10}"
            f"{m.get('authored_min_sep_norm', float('nan')):>9.3f}"
            f"{m.get('min_sep_norm', float('nan')):>8.3f}"
            f"{m.get('failed_solves', -1):>8}{own:>7}  "
            f"{r.closing_verdict or r.verdict}"
        )
    return "\n".join(rows)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, help="Motion to author, in plain language")
    parser.add_argument("--arm", choices=ARMS, default="absolute", help="Feedback encoding")
    parser.add_argument("--iters", type=int, default=4, help="Maximum synthesis iterations")
    parser.add_argument("--model", default="gpt-5.6-luna", help="LLM model id")
    parser.add_argument("--duration", type=float, default=12.0, help="Primitive window, seconds")
    parser.add_argument(
        "--out", type=Path, default=ROOT / "results/synthesized", help="Where to persist the entry"
    )
    parser.add_argument(
        "--runs", type=Path, default=ROOT / "synth_runs", help="Where to write the run log"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run one synthesis loop and persist the accepted primitive.

    Returns:
        0 if written, 1 if the model never said "keep", 2 if what it kept was rejected.
    """
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    settings = yaml.safe_load((ROOT / "swarm_gpt/data/settings.yaml").read_text())
    start_pos = starting_positions()

    loop = SynthesisLoop(
        settings=settings,
        start_pos_m=start_pos,
        arm=args.arm,
        model_id=args.model,
        duration_s=args.duration,
        screen=True,
    )
    logger.info("request=%r arm=%s drones=%d", args.request, args.arm, start_pos.shape[0])
    history = loop.run(args.request, max_iterations=args.iters)

    n_drones = int(start_pos.shape[0])
    record = accepted(history)
    reason = None if record is None else unsafe_reason(record)
    if record is None:
        closing = history[-1].closing_verdict if history else ""
        status, code = f"not accepted; model closed on {closing or 'nothing'!r}", 1
    elif reason is not None:
        status, code = f"rejected: {reason}", 2
    else:
        status, code = "promoted", 0

    log_path = write_run_log(history, status, args=args, n_drones=n_drones)
    print(render_table(history))
    logger.info("run log: %s", log_path)

    if code:
        logger.error("%s", status)
        if reason is not None:
            logger.error("the model kept it anyway -- %r", record.closing_reasoning)
        return code

    entry = document(
        record,
        request=args.request,
        arm=args.arm,
        model=args.model,
        duration_s=args.duration,
        n_drones=n_drones,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"{record.manifest['name']}.json"
    path.write_text(json.dumps(entry, indent=2) + "\n")

    logger.info("promoted %s to %s", record.manifest["name"], path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
