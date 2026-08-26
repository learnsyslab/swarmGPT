"""The gate a synthesis run must clear, and the on-disk library the CLI writes it to.

``gate`` decides whether a run may be trusted; ``promote`` is ``gate`` plus persistence. The two
are separate because they have different callers: the CLI is library authoring and persists, while
a primitive authored inside a browser refine is scoped to that one choreography and never written
to ``results/synthesized/``. ``load_promoted`` reads the directory back for offline tools; the
running app deliberately does not call it.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import toml
import yaml

from swarm_gpt.core.motion_primitives import clear_synthesized
from swarm_gpt.core.structured_output_schema import clear_synthesized_actions
from swarm_gpt.synth.manifest import PrimitiveManifest, clear_registered_manifests
from swarm_gpt.synth.sandbox import SynthError

if TYPE_CHECKING:
    from swarm_gpt.synth.loop import Iteration

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
PROMOTED_DIR = ROOT / "results" / "synthesized"


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
        "reasoning": record.closing_reasoning,
    }


def gate(history: list[Iteration]) -> tuple[int, str, Iteration | None]:
    """Decide whether a finished run produced a primitive that may be trusted.

    Returns:
        ``(code, status, record)`` -- code 0 cleared, 1 the model never said "keep", 2 a gate
        refused what it kept. ``record`` is set only on code 0.
    """
    record = accepted(history)
    if record is None:
        closing = history[-1].closing_verdict if history else ""
        return 1, f"not accepted; model closed on {closing or 'nothing'!r}", None
    reason = unsafe_reason(record)
    if reason is not None:
        return 2, f"rejected: {reason}", None
    return 0, "promoted", record


def promote(
    history: list[Iteration],
    *,
    request: str,
    arm: str,
    model: str,
    duration_s: float,
    n_drones: int,
    out_dir: Path = PROMOTED_DIR,
) -> tuple[int, str, Path | None]:
    """Apply the gate and write the entry to the on-disk library if it clears.

    Returns:
        ``(code, status, path)`` -- as ``gate``, with ``path`` set only on code 0.
    """
    code, status, record = gate(history)
    if record is None:
        return code, status, None

    entry = document(
        record, request=request, arm=arm, model=model, duration_s=duration_s, n_drones=n_drones
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{record.manifest['name']}.json"
    path.write_text(json.dumps(entry, indent=2) + "\n")
    return 0, "promoted", path


def reset_synthesized() -> None:
    """Forget every runtime-authored primitive, in all four places a signature lives."""
    clear_synthesized()
    clear_synthesized_actions()
    clear_registered_manifests()


def register_entry(entry: dict[str, Any]) -> PrimitiveManifest:
    """Compile and register one promoted entry, returning its manifest.

    Raises:
        SynthError: If the manifest is malformed or its source does not compile.
    """
    manifest = PrimitiveManifest.from_payload(entry["manifest"])
    fn, _check = manifest.compile()
    manifest.register(fn)
    return manifest


def load_promoted(directory: Path = PROMOTED_DIR, *, n_drones: int | None = None) -> list[str]:
    """Register every promoted primitive on disk, skipping any entry that will not load.

    A primitive is verified against one swarm size and may index drones by number, so an entry
    synthesized for a different ``n_drones`` is skipped rather than trusted.

    Returns:
        The names registered, in the order they loaded.
    """
    if not directory.is_dir():
        logger.info("No promoted primitives at %s", directory)
        return []
    names = []
    for path in sorted(directory.glob("*.json")):
        try:
            entry = json.loads(path.read_text())
            entry_drones = entry["provenance"]["n_drones"]
            if n_drones is not None and entry_drones != n_drones:
                logger.info(
                    "Skipping %s: verified for %d drones, this swarm has %d",
                    path.name,
                    entry_drones,
                    n_drones,
                )
                continue
            names.append(register_entry(entry).name)
        except (SynthError, ValueError, KeyError, json.JSONDecodeError) as exc:
            logger.error("Skipping promoted primitive %s: %s", path.name, exc)
    logger.info("Registered %d synthesized primitive(s): %s", len(names), ", ".join(names))
    return names
