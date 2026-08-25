"""Capture every synthesis run to JSONL, whatever the outcome.

Promotion is gated; capture is not. A run that is refused cost the same API time, and the refusal
is usually the interesting part. Logs land in the gitignored ``synth_runs/`` -- they are the record
of what was authored, and are deliberately not the load path.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from swarm_gpt.synth.loop import Iteration

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = ROOT / "synth_runs"


def write_run_log(
    history: list[Iteration],
    status: str,
    *,
    request: str,
    arm: str,
    model: str,
    duration_s: float,
    n_drones: int,
    max_iterations: int,
    runs_dir: Path = RUNS_DIR,
) -> Path:
    """Write one header line plus one line per iteration, and return the path."""
    runs_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = runs_dir / f"promote_{arm}_{stamp}.jsonl"
    header = {
        "request": request,
        "arm": arm,
        "model": model,
        "duration_s": duration_s,
        "n_drones": n_drones,
        "max_iterations": max_iterations,
        "status": status,
    }
    with open(path, "w") as f:
        f.write(json.dumps(header) + "\n")
        for record in history:
            f.write(json.dumps(asdict(record)) + "\n")
    logger.info("run log: %s", path)
    return path
