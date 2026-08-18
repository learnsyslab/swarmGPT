"""Inspect the newest result file from any of the experiment scripts.

Run from anywhere; paths resolve against the repo, not the shell's working directory.

    pixi run python experiments/inspect_runs.py gaps      # judge verdicts that were not "expressible"
    pixi run python experiments/inspect_runs.py plans     # blind vs anchored plan summaries
    pixi run python experiments/inspect_runs.py decoys    # which nonexistent primitives got chosen
    pixi run python experiments/inspect_runs.py ablation  # per-arm feedback comparison, live
    pixi run python experiments/inspect_runs.py status    # progress and ETA for anything running
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "synth_runs"
# The background runs log here; the scratchpad is session-scoped, so a missing log is not an error.
SCRATCH = Path(
    "/private/tmp/claude-501/-Users-yiyixu-Documents-School-ESROP-LSY-swarmGPT/"
    "5831b54f-b9cb-45eb-adcf-ff2e5ba1e309/scratchpad"
)


def newest(pattern: str) -> Path | None:
    """Newest file in ``synth_runs`` matching ``pattern``, or None if there is none."""
    found = sorted(glob.glob(str(RUNS / pattern)), key=os.path.getmtime)
    return Path(found[-1]) if found else None


def _require(pattern: str) -> dict | None:
    path = newest(pattern)
    if path is None:
        print(f"No {pattern} in {RUNS} yet.")
        return None
    print(f"# {path.name}  ({time.strftime('%H:%M', time.localtime(path.stat().st_mtime))})\n")
    return json.loads(path.read_text())


def show_gaps(song: str | None) -> None:
    """Print every judged intent the library could not fully deliver, with the judge's reason."""
    data = _require("unanchored_*.json")
    if data is None:
        return
    rows = [j for j in data["judged"] if j["verdict"] != "expressible"]
    if song:
        rows = [j for j in rows if j["song"] == song]
    for j in rows:
        print(f"[{j['condition']:<8}][{j['verdict']}] {j['song']}  {j['key']}")
        print(f"    wanted:  {j['motion']}")
        print(f"    missing: {j['missing']}\n")
    print(f"{len(rows)} intent(s) short of the library.")


def show_plans(song: str | None) -> None:
    """Print plan summaries side by side, so the two conditions can be compared directly."""
    data = _require("unanchored_*.json")
    if data is None:
        return
    for p in data["plans"]:
        if song and p["song"] != song:
            continue
        print(f"=== {p['condition'].upper():<9} {p['song']} ===")
        print(f"{p['plan_summary']}\n")
        for intent in p["intents"]:
            print(f"  {intent['key']:<10} {intent['motion'][:96]}")
        print()


def show_decoys() -> None:
    """Print which nonexistent primitives were chosen, split by gap vs redundant class."""
    from probe_decoy import LIGHTING_DECOYS, MOTION_DECOYS

    data = _require("decoy_*.json")
    if data is None:
        return
    songs = defaultdict(set)
    counts: Counter = Counter()
    for r in data["records"]:
        for track in ("motion", "lighting"):
            for name, n in r[track].items():
                if name in MOTION_DECOYS or name in LIGHTING_DECOYS:
                    counts[name] += n
                    songs[name].add(r["song"])
    n_songs = len({r["song"] for r in data["records"]})
    print(f"{'primitive':<18}{'class':<11}{'uses':<7}songs (of {n_songs})")
    for name, n in counts.most_common():
        kind = (MOTION_DECOYS.get(name) or LIGHTING_DECOYS[name])[0]
        print(f"{name:<18}{kind:<11}{n:<7}{len(songs[name])}")
    print("\n'gap' adds capability the library lacks; 'redundant' is a renamed duplicate.")
    print("Only a gap rate clearly above the redundant rate is evidence of unmet need.")


def show_ablation() -> None:
    """Print per-arm feedback results from the JSONL, which is written run by run."""
    path = newest("ablation_*.jsonl")
    if path is None:
        print("No ablation_*.jsonl yet.")
        return
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    header, rows = rows[0], [r for r in rows[1:] if r.get("ok")]
    print(f"# {path.name}: {len(rows)}/{header.get('grid', '?')} runs done\n")
    print(f"{'arm':<13}{'n':<5}{'dev_max median':<17}{'range':<16}checks median")
    by_arm = defaultdict(list)
    for r in rows:
        by_arm[r["arm"]].append(r)
    for arm, cell in by_arm.items():
        dev = [r["deviation_max_m"] for r in cell]
        chk = [
            r["check_pass_fraction"]
            for r in cell
            if r["check_pass_fraction"] == r["check_pass_fraction"]
        ]
        print(
            f"{arm:<13}{len(cell):<5}{statistics.median(dev):<17.2f}"
            f"{f'{min(dev):.2f}-{max(dev):.2f}':<16}"
            f"{statistics.median(chk):.0%}"
            if chk
            else ""
        )
    print("\nPrimary outcome is dev_max (lower better), fixed before the data existed.")
    print("Ranges overlapping means the arms are not separated yet, whatever the medians say.")


def show_status() -> None:
    """Print progress and a linear ETA for each background run that has a log."""
    now = time.time()
    for name, log, total, pattern in (
        ("ablation", "ablation.log", 54, r"^\["),
        ("unanchored", "unanchored.log", 26, r"(blind|anchored)\s+\d+ intents"),
        ("decoy", "decoy.log", 39, r"s\d: \d+ motion"),
    ):
        path = SCRATCH / log
        if not path.exists():
            continue
        done = len([ln for ln in path.read_text().splitlines() if re.search(pattern, ln)])
        started = path.stat().st_birthtime
        elapsed = now - started
        rate = elapsed / done if done else 0
        left = (total - done) * rate
        finished = done >= total
        eta = "done" if finished else time.strftime("%H:%M", time.localtime(now + left))
        print(
            f"{name:<12}{done:>3}/{total:<5}{elapsed / 60:>6.0f} min elapsed  "
            f"{rate / 60:>5.1f} min/unit  ETA {eta}"
        )


def main() -> None:
    """Dispatch to the requested view."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("view", choices=["gaps", "plans", "decoys", "ablation", "status"])
    parser.add_argument("--song", help="Restrict gaps/plans to one song")
    args = parser.parse_args()
    if args.view == "gaps":
        show_gaps(args.song)
    elif args.view == "plans":
        show_plans(args.song)
    elif args.view == "decoys":
        show_decoys()
    elif args.view == "ablation":
        show_ablation()
    else:
        show_status()


if __name__ == "__main__":
    main()
