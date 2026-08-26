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
import logging
from pathlib import Path

import yaml

from swarm_gpt.synth.feedback import ARMS
from swarm_gpt.synth.loop import Iteration, SynthesisLoop
from swarm_gpt.synth.promote import accepted, promote, starting_positions
from swarm_gpt.synth.run_log import write_run_log

logger = logging.getLogger("synth")

ROOT = Path(__file__).resolve().parents[1]


def render_table(history: list[Iteration]) -> str:
    """One row per iteration: how far it got, what it broke, and how its two verdicts landed."""
    header = f"\n{'it':<4}{'stage':<10}{'authored':>9}{'flown':>8}{'failed':>8}  verdict"
    rows = [header, "-" * len(header)]
    for r in history:
        m = r.metrics or {}
        # A run rejected on its geometry never built a trajectory, so its separation is the
        # shape's own -- the same quantity, measured a stage earlier.
        authored = m.get("authored_min_sep_norm", m.get("shape_min_sep_norm", float("nan")))
        rows.append(
            f"{r.index:<4}{r.stage:<10}{authored:>9.3f}"
            f"{m.get('min_sep_norm', float('nan')):>8.3f}"
            f"{m.get('failed_solves', -1):>8}  "
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
    code, status, path = promote(
        history,
        request=args.request,
        arm=args.arm,
        model=args.model,
        duration_s=args.duration,
        n_drones=n_drones,
        out_dir=args.out,
    )

    write_run_log(
        history,
        status,
        request=args.request,
        arm=args.arm,
        model=args.model,
        duration_s=args.duration,
        n_drones=n_drones,
        max_iterations=args.iters,
        runs_dir=args.runs,
    )
    print(render_table(history))

    if code:
        logger.error("%s", status)
        if code == 2:
            logger.error("the model kept it anyway -- %r", accepted(history).closing_reasoning)
        return code

    logger.info("promoted %s to %s", accepted(history).manifest["name"], path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
