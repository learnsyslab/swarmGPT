"""Plot a promoted primitive: the shape its equation draws, and the pose the filter actually flew.

    pixi run -e tests python experiments/plot_primitive.py results/synthesized/form_heart.json

Solves the primitive again rather than trusting the stored numbers, so the picture is of a
trajectory that was actually produced, not of the manifest's intent. The equation is drawn at
three swarm sizes as well, because a shape that only works at the size it was authored for is not
one.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import toml
import yaml

from swarm_gpt.synth.manifest import PrimitiveManifest
from swarm_gpt.synth.shape import targets
from swarm_gpt.synth.verifier import authored_trajectory, solve_only

ROOT = Path(__file__).resolve().parents[1]

_SIZES = (10, 20, 40)


def fly(entry: dict) -> tuple[np.ndarray, np.ndarray]:
    """Re-run the primitive and the filter, returning the flown (D, T, 3) array in m and the shape.

    Returns:
        The flown trajectory, and the authored shape in m at the show's own swarm size.
    """
    settings = yaml.safe_load((ROOT / "swarm_gpt/data/settings.yaml").read_text())
    raw = toml.load(ROOT / "swarm_gpt/data/drones.toml")
    start = np.array([raw[n]["pos"] for n in raw["active"]], dtype=float)
    start[:, 2] = settings["starting_height"]
    limits = {
        "lower": np.asarray(settings["axswarm"]["pos_min"], dtype=float),
        "upper": np.asarray(settings["axswarm"]["pos_max"], dtype=float),
    }
    manifest = PrimitiveManifest.from_payload(entry["manifest"])
    fn, shape_fn = manifest.compile()
    args = manifest.bind(entry["args"])
    duration = entry["provenance"]["duration_s"]
    authored = authored_trajectory(fn, args, start, 0.0, duration, limits)
    repaired = solve_only(authored, settings)
    return np.transpose(repaired["pos"], (1, 0, 2)), targets(shape_fn, args, len(start)) / 100


def _plane(shape: np.ndarray) -> tuple[int, int]:
    """The two axes the shape actually spreads over, so a flat shape is never drawn edge-on.

    Returns:
        The wider and the narrower of the two spread axes, as indices into x, y, z.
    """
    order = np.argsort(np.ptp(shape, axis=0))[::-1]
    return int(order[0]), int(order[1])


def plot(entry: dict, out: Path) -> Path:
    """Draw the flown pose in the shape's own plane, and the equation at three swarm sizes."""
    flown, shape = fly(entry)
    manifest = PrimitiveManifest.from_payload(entry["manifest"])
    _fn, shape_fn = manifest.compile()
    args = manifest.bind(entry["args"])
    horizontal, vertical = _plane(shape)
    labels = "xyz"

    fig, axes = plt.subplots(1, 1 + len(_SIZES), figsize=(4.2 * (1 + len(_SIZES)), 4.4))
    metrics = entry["metrics"]
    fig.suptitle(
        f"{manifest.signature()} = {entry['args']}   —   authored by "
        f"{entry['provenance']['model']}, flown separation {metrics['min_sep_norm']:.2f}x, "
        f"{metrics['steps_inside_envelope']} steps inside the envelope",
        fontsize=10,
    )

    ax = axes[0]
    for drone in flown:
        ax.plot(drone[:, horizontal], drone[:, vertical], color="0.75", lw=0.7, zorder=1)
    ax.plot(shape[:, horizontal], shape[:, vertical], "-", color="#d62728", lw=1.2, zorder=2)
    ax.scatter(flown[:, -1, horizontal], flown[:, -1, vertical], color="#1f77b4", s=40, zorder=3)
    ax.set_title("the pose it flew into, over its flight paths", fontsize=9)
    ax.set_aspect("equal")
    ax.set_xlabel(f"{labels[horizontal]} [m]")
    ax.set_ylabel(f"{labels[vertical]} [m]")

    for ax, n in zip(axes[1:], _SIZES):
        sampled = targets(shape_fn, args, n) / 100
        ax.plot(sampled[:, horizontal], sampled[:, vertical], "o-", color="#d62728", ms=5)
        ax.set_title(f"the equation, sampled for {n} drones", fontsize=9)
        ax.set_aspect("equal")
        ax.set_xlabel(f"{labels[horizontal]} [m]")
        ax.set_ylabel(f"{labels[vertical]} [m]")

    fig.tight_layout()
    fig.savefig(out, dpi=140)
    return out


def main(argv: list[str] | None = None) -> None:
    """Plot one promoted primitive to PNG."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("entry", type=Path, help="A results/synthesized/<name>.json")
    parser.add_argument("--out", type=Path, help="PNG path (default: alongside the entry)")
    args = parser.parse_args(argv)
    entry = json.loads(args.entry.read_text())
    out = args.out or args.entry.with_suffix(".png")
    print(f"wrote {plot(entry, out)}")


if __name__ == "__main__":
    main()
