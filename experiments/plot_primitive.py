"""Plot a promoted primitive: the flown trajectory, and the geometry its shape check measures.

    pixi run -e tests python experiments/plot_primitive.py results/synthesized/double_helix.json

Solves the primitive again rather than trusting the stored numbers, so the picture is of a
trajectory that was actually produced, not of the manifest's intent.
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
from swarm_gpt.synth.verifier import authored_trajectory, solve_only

ROOT = Path(__file__).resolve().parents[1]


def fly(entry: dict) -> tuple[np.ndarray, np.ndarray]:
    """Re-run the primitive and the filter, returning authored and flown (D, T, 3) arrays in m."""
    settings = yaml.safe_load((ROOT / "swarm_gpt/data/settings.yaml").read_text())
    raw = toml.load(ROOT / "swarm_gpt/data/drones.toml")
    start = np.array([raw[n]["pos"] for n in raw["active"]], dtype=float)
    start[:, 2] = settings["starting_height"]
    limits = {
        "lower": np.asarray(settings["axswarm"]["pos_min"], dtype=float),
        "upper": np.asarray(settings["axswarm"]["pos_max"], dtype=float),
    }
    manifest = PrimitiveManifest.from_payload(entry["manifest"])
    fn, _check = manifest.compile()
    duration = entry["provenance"]["duration_s"]
    authored = authored_trajectory(fn, manifest.bind(entry["args"]), start, 0.0, duration, limits)
    repaired = solve_only(authored, settings)
    return authored["pos"], np.transpose(repaired["pos"], (1, 0, 2))


def strands(final: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split the formed pose into its two strands by pairing drones on matched heights."""
    pairs = np.argsort(final[:, 2]).reshape(-1, 2)
    return pairs[:, 0], pairs[:, 1]


def plot(entry: dict, out: Path) -> Path:
    """Draw the flown trajectory, the pairing seen from above, and the twist against height."""
    authored, flown = fly(entry)
    final = flown[:, -1, :]
    left, right = strands(final)
    centre = final[:, :2].mean(axis=0)

    fig = plt.figure(figsize=(15, 5.2))
    name = entry["manifest"]["name"]
    fig.suptitle(
        f"{name}({', '.join(p['name'] for p in entry['manifest']['params'])}) "
        f"= {entry['args']}   —   authored by {entry['provenance']['model']}",
        fontsize=11,
    )

    ax = fig.add_subplot(131, projection="3d")
    for a, b in zip(left, right):
        ax.plot(*final[[a, b]].T, color="0.75", lw=1.0, zorder=1)
    for group, colour in ((left, "#1f77b4"), (right, "#d62728")):
        climb = group[np.argsort(final[group, 2])]
        ax.plot(*final[climb].T, color=colour, lw=2.6, zorder=3)
        ax.scatter(*final[group].T, color=colour, s=30, depthshade=False, zorder=4)
        for i in group:
            ax.plot(*flown[i].T, color=colour, lw=0.5, alpha=0.30, zorder=2)
    ax.set_title("the formed helix, over its faint flight paths", fontsize=9)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    # Near side-on, and the axes scaled to the structure rather than the arena, or half a turn of
    # helix reads as a zigzag.
    span = np.ptp(final[:, :2])
    ax.set_xlim(centre[0] - span / 2, centre[0] + span / 2)
    ax.set_ylim(centre[1] - span / 2, centre[1] + span / 2)
    ax.set_box_aspect((1, 1, 1.35))
    ax.view_init(elev=16, azim=-72)

    ax = fig.add_subplot(132)
    for a, b in zip(left, right):
        ax.plot(final[[a, b], 0], final[[a, b], 1], color="0.75", lw=0.8, zorder=1)
    ax.scatter(final[left, 0], final[left, 1], color="#1f77b4", s=34, zorder=2, label="strand A")
    ax.scatter(final[right, 0], final[right, 1], color="#d62728", s=34, zorder=2, label="strand B")
    ax.scatter(*centre, marker="+", color="k", s=60, zorder=3)
    ax.set_aspect("equal")
    ax.set_title("formed pose from above — each grey line is one pair", fontsize=9)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.legend(fontsize=8, loc="upper right")

    ax = fig.add_subplot(133)
    angle = np.degrees(np.arctan2(final[left, 1] - centre[1], final[left, 0] - centre[0])) % 180
    order = np.argsort(final[left, 2])
    ax.plot(np.unwrap(angle[order], period=180), final[left, 2][order], "o-", color="#2b6a3f")
    ax.set_title("pair axis vs height — a helix climbs as it turns", fontsize=9)
    ax.set_xlabel("pair axis angle [deg, unwrapped]")
    ax.set_ylabel("z [m]")
    ax.grid(alpha=0.3)

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
