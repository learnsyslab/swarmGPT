"""Re-judge an unanchored run with per-drone ``move`` placement disallowed as a way to express shape.

``move(x, y, z, drone_id)`` sets ONE drone to ONE absolute coordinate. A judge that credits it with
expressing a formation is crediting the library with hand-specifying the shape drone by drone --
which is the work a primitive would exist to do, so the credit is circular. It is also
counterfactual: across 39 full choreographies the model emitted ``move`` zero times.

This re-runs the identical judging step over the identical intents with one rule added, so the
comparison is paired: every verdict shift is attributable to that rule and nothing else.

    pixi run python experiments/judge_against_vocabulary.py
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from swarm_gpt.utils.llm_providers import openai_client_for_provider

# `experiments` is not a package, so the sibling probe is imported by path. The judge prompt and schema
# must be the *same objects* the original run used, or the comparison stops being paired.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_unanchored import (  # noqa: E402
    _JUDGE_PROMPT,
    _JUDGE_SCHEMA,
    CONDITIONS,
    VERDICTS,
    call_structured,
    vocabulary_blocks,
)
from vocabularies import VARIANTS  # noqa: E402

logger = logging.getLogger("rejudge")

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "synth_runs"

_MOVE_RULE = """

ONE RULE ABOUT `move`: `move(x_cm, y_cm, z_cm, drone_id)` sets a single drone to a single absolute
coordinate. Writing one `move` per drone to build up a formation is NOT the library expressing that
shape -- it is hand-specifying the shape drone by drone, which is exactly the work a primitive
would exist to do. So do not count per-drone `move` placement as expressing a multi-drone shape,
path, or figure. Count `move` only for what it plainly is: a single-drone or two-drone accent.

Apply this rule symmetrically. It does not make an intent harder or easier than it is; it only
stops one primitive from being credited with capability it does not have."""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="Unanchored run JSON (default: newest)")
    parser.add_argument("--model", default="gpt-5.6-luna", help="Judge model")
    parser.add_argument("--limit", type=int, help="Re-judge only the first N intents per condition")
    parser.add_argument(
        "--vocabulary",
        choices=VARIANTS,
        default="current",
        help="Library to judge against. Blind intents carry no vocabulary, so judging the same "
        "ones against several libraries is a paired comparison.",
    )
    parser.add_argument(
        "--conditions",
        nargs="*",
        default=list(CONDITIONS),
        help="Restrict to these conditions. Only `blind` is vocabulary-independent.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> Path:
    """Re-judge and report the paired verdict shift.

    Returns:
        Path of the written comparison.
    """
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    source = args.source
    if source is None:
        found = sorted(glob.glob(str(RUNS / "unanchored_2026*.json")), key=os.path.getmtime)
        if not found:
            raise SystemExit(f"No unanchored run in {RUNS}")
        source = Path(found[-1])
    data = json.loads(source.read_text())
    motion_block, lighting_block = vocabulary_blocks(args.vocabulary)
    client = openai_client_for_provider("openai")

    original = [j for j in data["judged"] if j["condition"] in args.conditions]
    if args.limit:
        kept: list[dict] = []
        for condition in args.conditions:
            kept += [j for j in original if j["condition"] == condition][: args.limit]
        original = kept

    rejudged = []
    for i, j in enumerate(original, start=1):
        verdict = call_structured(
            client,
            args.model,
            _JUDGE_PROMPT.format(
                motion=j["motion"],
                lighting=j["lighting"],
                primitives=motion_block,
                lighting_primitives=lighting_block,
            )
            + _MOVE_RULE,
            _JUDGE_SCHEMA,
            "expressibility",
        )
        rejudged.append(
            {
                "song": j["song"],
                "condition": j["condition"],
                "key": j["key"],
                "motion": j["motion"],
                "lighting": j["lighting"],
                "before": j["verdict"],
                "after": verdict["verdict"],
                "missing": verdict["missing"],
                "primitives_used": verdict["primitives_used"],
            }
        )
        if i % 20 == 0:
            logger.info("  %d/%d re-judged", i, len(original))

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = RUNS / f"rejudge_{args.vocabulary}_{stamp}.json"
    out.write_text(
        json.dumps(
            {
                "source": source.name,
                "model": args.model,
                "vocabulary": args.vocabulary,
                "rejudged": rejudged,
            },
            indent=2,
        )
    )
    report(rejudged)
    print(f"\nwrote {out}")
    return out


def report(rows: list[dict]) -> None:
    """Print the paired before/after shift per condition."""
    for condition in CONDITIONS:
        cell = [r for r in rows if r["condition"] == condition]
        if not cell:
            continue
        n = len(cell)
        before, after = Counter(r["before"] for r in cell), Counter(r["after"] for r in cell)
        print(f"\n{condition.upper()}  n={n}")
        print(f"  {'verdict':<18}{'before':<14}after")
        for v in VERDICTS:
            print(
                f"  {v:<18}{before[v] / n:>5.0%} ({before[v]:>2})   {after[v] / n:>5.0%} ({after[v]:>2})"
            )
        short_b = (before["partial"] + before["not_expressible"]) / n
        short_a = (after["partial"] + after["not_expressible"]) / n
        print(f"  {'shortfall':<18}{short_b:>5.0%}        {short_a:>5.0%}")
        moves = defaultdict(int)
        for r in cell:
            if r["before"] != r["after"]:
                moves[f"{r['before']} -> {r['after']}"] += 1
        for k, v in sorted(moves.items(), key=lambda kv: -kv[1]):
            print(f"    {v:>3}  {k}")

    print("\nDELTA (blind minus anchored):")
    for label, field in (("before", "before"), ("after", "after")):
        vals = {}
        for condition in CONDITIONS:
            cell = [r for r in rows if r["condition"] == condition]
            if not cell:
                continue
            counts = Counter(r[field] for r in cell)
            vals[condition] = (counts["partial"] + counts["not_expressible"]) / len(cell)
        if len(vals) == 2:
            print(f"  {label:<8}{vals['blind'] - vals['anchored']:+.1%}")


if __name__ == "__main__":
    main()
