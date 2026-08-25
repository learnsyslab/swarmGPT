"""Measure the refine-message gap classifier against hand-labelled requests.

The introspective probe established that asking the choreographer what it lacks returns nothing.
The classifier asks a different question -- one concrete request against one concrete list -- and
that difference is an assumption, not a result, until it is measured. This is the measurement.

Labels are the author's judgement about the current hand-written library and are the weak point:
a few cases are genuinely arguable and are marked so. Report accuracy on the unambiguous ones.

    pixi run python experiments/probe_gap_classifier.py --repeats 1
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

from swarm_gpt.synth.trigger import catalogue, classify_gap

logger = logging.getLogger("probe")

ROOT = Path(__file__).resolve().parents[1]

# (message, needs_new, arguable) -- `arguable` marks a case the author cannot label with confidence.
CASES: list[tuple[str, bool, bool]] = [
    ("make the chorus more energetic", False, False),
    ("put a circle at the drop", False, False),
    ("add a star formation in the bridge", False, False),
    ("make the drones spiral higher during the solo", False, False),
    ("turn everything deep red at the drop", False, False),
    ("have them fly lower overall, it feels too high", False, False),
    ("add a cone at the very end as they land", False, False),
    ("the wave section is too busy, slow it down", False, False),
    ("split the swarm into two circles at different heights", False, False),
    ("spin the whole formation 180 degrees at the chorus", False, False),
    ("put a heart at the drop", True, False),
    ("have them form a butterfly during the bridge", True, False),
    ("spell out the letter S in the air", True, False),
    ("arrange them into the outline of a cube", True, False),
    ("make a crescent moon shape in the quiet section", True, False),
    ("form an arrow pointing straight up at the climax", True, False),
    ("make a DNA double helix at the climax", True, False),
    ("form a triangle standing upright facing the audience", True, True),
    ("tilt the ring on its side so it reads like a coin", True, True),
    ("have them form a grid that ripples outward from the centre", False, True),
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gpt-5.6-luna", help="LLM model id")
    parser.add_argument("--repeats", type=int, default=1, help="Times to run each case")
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "results/coverage/gap-classifier.json",
        help="Where to write the result",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Classify every labelled case and report the confusion matrix.

    Returns:
        0 always; the numbers are the output.
    """
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger.info("catalogue the classifier is shown:\n%s\n", catalogue())

    rows = []
    for message, label, arguable in CASES:
        for repeat in range(args.repeats):
            gap = classify_gap(message, model_id=args.model)
            rows.append(
                {
                    "message": message,
                    "label": label,
                    "arguable": arguable,
                    "repeat": repeat,
                    "predicted": gap is not None,
                    "name": gap.name if gap else "",
                    "request": gap.request if gap else "",
                    "reasoning": gap.reasoning if gap else "",
                }
            )
            mark = "ok " if rows[-1]["predicted"] == label else "MISS"
            logger.info("%s %-52s -> %s", mark, message[:52], rows[-1]["name"] or "(covered)")

    firm = [r for r in rows if not r["arguable"]]
    counts = {
        "true_positive": sum(r["label"] and r["predicted"] for r in firm),
        "false_negative": sum(r["label"] and not r["predicted"] for r in firm),
        "true_negative": sum(not r["label"] and not r["predicted"] for r in firm),
        "false_positive": sum(not r["label"] and r["predicted"] for r in firm),
    }
    counts["accuracy"] = (counts["true_positive"] + counts["true_negative"]) / max(len(firm), 1)
    logger.info("\nunambiguous cases (n=%d): %s", len(firm), counts)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "model": args.model,
                "repeats": args.repeats,
                "probed_at": datetime.now().isoformat(timespec="seconds"),
                "catalogue": catalogue(),
                "counts": counts,
                "rows": rows,
            },
            indent=2,
        )
        + "\n"
    )
    logger.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
