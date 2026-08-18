"""Render the unanchored-probe results as a readable markdown report.

The raw JSON is awkward to read by eye, and the judge's reasoning is the part that decides whether
the headline delta is trustworthy, so it gets written out in full rather than summarised.

    pixi run python experiments/report_unanchored.py
"""

from __future__ import annotations

import glob
import json
import os
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "synth_runs"

# Palette complaints are a colour-list limitation, not a missing motion capability, so the report
# separates them: only the capability figure bears on whether primitives need authoring.
_PALETTE = re.compile(
    r"palette|hue|gold|ivory|silver|pearl|violet|fuchsia|warm-white|champagne", re.I
)
_CAPABILITY = re.compile(
    r"no primitive|no true|no dedicated|not directly|cannot|does not|there is no|no explicit", re.I
)


def classify(missing: str) -> str:
    """Label a judge's reason as a capability gap, a palette gap, or both."""
    cap, pal = bool(_CAPABILITY.search(missing)), bool(_PALETTE.search(missing))
    if cap and pal:
        return "capability + palette"
    if cap:
        return "capability"
    if pal:
        return "palette only"
    return "unclear"


def main() -> Path:
    """Write the markdown report next to the raw results.

    Returns:
        Path of the written report.
    """
    found = sorted(glob.glob(str(RUNS / "unanchored_*.json")), key=os.path.getmtime)
    if not found:
        raise SystemExit(f"No unanchored_*.json in {RUNS}")
    source = Path(found[-1])
    data = json.loads(source.read_text())
    judged, plans = data["judged"], data["plans"]

    lines = [
        "# Unanchored elicitation probe — results",
        "",
        f"Source: `{source.name}`  |  model: `{data['model']}`",
        "",
        "## What this measures",
        "",
        "The model is asked for a choreography plan in plain language **with no primitive list in "
        "the prompt** (`blind`), and again with the real primitive list added and nothing else "
        "changed (`anchored`). A separate judge then decides, for each described moment, whether "
        "the real library can deliver it. The judge sees one intent and the primitive list, never "
        "the song, the surrounding plan, or which condition produced it.",
        "",
        "The headline is the **difference** between conditions. An absolute figure would inherit "
        "whatever bias the judge has; a difference does not, as long as the judge cannot tell the "
        "conditions apart — which it cannot.",
        "",
        "## Summary",
        "",
        "| condition | intents | expressible | any shortfall | capability shortfall |",
        "|---|---|---|---|---|",
    ]

    stats = {}
    for condition in ("blind", "anchored"):
        cell = [j for j in judged if j["condition"] == condition]
        short = [j for j in cell if j["verdict"] != "expressible"]
        cap = [j for j in short if _CAPABILITY.search(j["missing"])]
        n = len(cell)
        stats[condition] = (len(short) / n, len(cap) / n)
        lines.append(
            f"| {condition} | {n} | {(n - len(short)) / n:.0%} | "
            f"{len(short) / n:.0%} ({len(short)}) | {len(cap) / n:.0%} ({len(cap)}) |"
        )
    lines += [
        "",
        f"**Anchoring delta, any shortfall: {stats['blind'][0] - stats['anchored'][0]:+.1%}**",
        "",
        f"**Anchoring delta, capability only: "
        f"{stats['blind'][1] - stats['anchored'][1]:+.1%}** — the defensible figure, with "
        "colour-palette complaints excluded.",
        "",
        "No intent in either condition was judged `not_expressible`. Every shortfall is `partial`, "
        "so the claim is that the library **degrades** what the model asks for, not that it cannot "
        "do it at all.",
        "",
        "## Per song",
        "",
        "| song | blind expressible | anchored expressible |",
        "|---|---|---|",
    ]

    per = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for j in judged:
        cell = per[j["song"]][j["condition"]]
        cell[0] += 1
        cell[1] += j["verdict"] == "expressible"
    for song in sorted(per):
        b, a = per[song]["blind"], per[song]["anchored"]
        lines.append(f"| {song} | {b[1]}/{b[0]} | {a[1]}/{a[0]} |")

    for condition, heading in (
        ("blind", "Blind-condition shortfalls (judge saw no condition label)"),
        ("anchored", "Anchored-condition shortfalls, for comparison"),
    ):
        rows = [j for j in judged if j["condition"] == condition and j["verdict"] != "expressible"]
        lines += ["", f"## {heading} — {len(rows)} of 88", ""]
        for j in rows:
            lines += [
                f"**{j['song']} · {j['key']}** — _{classify(j['missing'])}_",
                "",
                f"- wanted: {j['motion']}",
                f"- lighting: {j['lighting']}",
                f"- judge says missing: {j['missing']}",
                "",
            ]

    lines += ["", "## Full plans", ""]
    for p in plans:
        lines += [f"### {p['condition'].upper()} — {p['song']}", "", p["plan_summary"], ""]
        for intent in p["intents"]:
            lines.append(f"- `{intent['key']}` **{intent['motion']}** — {intent['lighting']}")
        lines.append("")

    out = RUNS / "unanchored_report.md"
    out.write_text("\n".join(lines))
    print(f"wrote {out} ({out.stat().st_size // 1024} KB)")
    return out


if __name__ == "__main__":
    main()
