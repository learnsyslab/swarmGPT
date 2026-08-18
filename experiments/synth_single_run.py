"""Run the primitive-synthesis loop once, under one feedback arm, and log every iteration.

pixi run python experiments/synth_single_run.py --request "a double helix" --arm absolute
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import toml
import yaml

from swarm_gpt.synth.feedback import ARMS
from swarm_gpt.synth.loop import SynthesisLoop

logger = logging.getLogger("synth")

ROOT = Path(__file__).resolve().parents[1]


def starting_positions() -> np.ndarray:
    """Read the active swarm's dock positions from ``drones.toml``, at the show start height."""
    settings = yaml.safe_load((ROOT / "swarm_gpt/data/settings.yaml").read_text())
    raw = toml.load(ROOT / "swarm_gpt/data/drones.toml")
    positions = np.array([raw[name]["pos"] for name in raw["active"]], dtype=float)
    positions[:, 2] = settings["starting_height"]
    return positions


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, help="Motion to author, in plain language")
    parser.add_argument("--arm", choices=ARMS, default="absolute", help="Feedback encoding")
    parser.add_argument("--iters", type=int, default=4, help="Maximum synthesis iterations")
    parser.add_argument("--model", default="gpt-5.6-luna", help="LLM model id")
    parser.add_argument("--duration", type=float, default=12.0, help="Primitive window, seconds")
    parser.add_argument("--out", type=Path, default=ROOT / "synth_runs", help="Run log directory")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> Path:
    """Run one synthesis loop and write its JSONL log.

    Returns:
        Path of the written run log.
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
    )
    logger.info("request=%r arm=%s drones=%d", args.request, args.arm, start_pos.shape[0])
    history = loop.run(args.request, max_iterations=args.iters)

    args.out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = args.out / f"{args.arm}_{stamp}.jsonl"
    with open(path, "w") as f:
        f.write(json.dumps({"request": args.request, "arm": args.arm, "model": args.model}) + "\n")
        for record in history:
            f.write(json.dumps(asdict(record)) + "\n")

    # A turn both judges the previous candidate and proposes the next, so a row's verdict is the
    # one attached to the following turn -- or `closing_verdict` on the row the run ended with.
    judged = {r.index: n.verdict for r, n in zip(history, history[1:])}
    print(f"\n{'iter':<5}{'stage':<10}{'sep':>7}{'dev_max':>9}  {'checks':<12}judged")
    for record in history:
        m = record.metrics or {}
        checks = (
            f"{sum(c['ok'] for c in record.checks)}/{len(record.checks)} pass"
            if record.checks
            else (record.error or "")[:40]
        )
        print(
            f"{record.index:<5}{record.stage:<10}"
            f"{m.get('min_sep_norm', float('nan')):>7.2f}"
            f"{m.get('deviation_max_m', float('nan')):>9.2f}  {checks:<12}"
            f"{record.closing_verdict or judged.get(record.index, '-')}"
        )
    print(f"\nlog: {path}")
    return path


if __name__ == "__main__":
    main()
