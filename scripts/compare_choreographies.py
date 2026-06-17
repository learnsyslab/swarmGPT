"""Physically compare Euclidean vs min-snap assignment on a fixed 2-primitive sequence.

No LLM, no MPC sim. We run the *identical* primitive sequence

    1. rotate     (gives each drone tangential swirl momentum)
    2. form_star  (does the drone->target assignment)

twice: once passing the carried velocity into the assignment (min-snap) and once
withholding it (Euclidean). The only thing that differs is the assignment metric,
so the drone paths isolate its effect.

Outputs:
    scripts/compare_choreographies.png   side-by-side traced paths
    scripts/compare_choreographies.gif   side-by-side animation (drones flying)

Run with:
    pixi run python scripts/compare_choreographies.py
"""

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.animation as animation  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from swarm_gpt.core.motion_primitives import (  # noqa: E402
    _assign_positions,
    _minsnap_cost_matrix,
    form_star,
    rotate,
)

LIMITS = {"lower": np.array([-2.2, -2.7, 0.25]), "upper": np.array([2.2, 2.7, 1.7])}
SWARM = np.array([[x, y, 100.0] for x in (-200, -100, 0, 100, 200) for y in (-100, 100)])

T_ROT0, T_ROT1 = 0.0, 1.0
T_FORM0, T_FORM1 = 1.0, 4.0
ROTATE_ANGLE_DEG = 90
FORM_STAR_ARGS = (120, 60, 80, 2.0)  # height, min_spacing, delta_radius, time_to_finish_s


def _form_star_targets(n: int, args: tuple) -> np.ndarray:
    """Replicate form_star's target ring positions (cm) for snap-cost evaluation."""
    height, min_spacing, delta_radius, _ = args
    min_spacing = max(min_spacing, 40)
    delta_radius = max(delta_radius, 40)
    dpc = n // 2
    r = min_spacing / (2 * np.sin(np.pi / dpc))
    des = None
    for rr, off in zip([r, r + delta_radius], [0, 2 * np.pi / dpc]):
        ang = np.linspace(0, 2 * np.pi, dpc, endpoint=False) + off
        block = np.array([rr * np.cos(ang), rr * np.sin(ang), [height] * dpc]).T
        des = block if des is None else np.vstack([des, block])
    if n != dpc * 2:
        des = np.vstack([des, np.array([0, 0, height])])
    return des


def build_waypoints(use_snap: bool) -> tuple[np.ndarray, np.ndarray, float]:
    """Run rotate -> form_star; return (times, positions_cm, total_snap_cost).

    Args:
        use_snap: pass carried velocity to form_star (min-snap) or withhold it (Euclidean).

    Returns:
        times: sorted timestamps (T,).
        positions: per-drone positions in cm (n, T, 3).
        snap_cost: total min-snap cost of the resulting formation assignment.
    """
    n = SWARM.shape[0]
    pos = SWARM.copy()
    vel = np.zeros_like(pos)
    merged: dict[float, dict[int, np.ndarray]] = {0.0: {i: p.copy() for i, p in enumerate(pos)}}

    prev = pos.copy()
    pos, wps = rotate((ROTATE_ANGLE_DEG, "z"), pos, T_ROT0, T_ROT1, LIMITS, swarm_vel=vel)
    vel = (pos - prev) / (T_ROT1 - T_ROT0)
    for t, d in wps.items():
        merged.setdefault(t, {}).update({i: p.copy() for i, p in d.items()})

    # snap cost of the assignment this metric will pick (evaluated on the true snap cost)
    pos_ho, vel_ho = pos.copy(), vel.copy()
    targets = _form_star_targets(n, FORM_STAR_ARGS)
    cost_mat = _minsnap_cost_matrix(pos_ho, targets, vel_ho, T_FORM1 - T_FORM0)
    assign = _assign_positions(
        pos_ho, targets, swarm_vel=(vel_ho if use_snap else None), T=T_FORM1 - T_FORM0
    )
    snap_cost = float(cost_mat[np.arange(n), assign].sum())

    pos, wps = form_star(
        FORM_STAR_ARGS, pos, T_FORM0, T_FORM1, LIMITS, swarm_vel=(vel if use_snap else None)
    )
    for t, d in wps.items():
        merged.setdefault(t, {}).update({i: p.copy() for i, p in d.items()})

    times = np.array(sorted(merged.keys()))
    positions = np.zeros((n, len(times), 3))
    last = {i: SWARM[i].copy() for i in range(n)}
    for k, t in enumerate(times):
        for i in range(n):
            if i in merged[t]:
                last[i] = merged[t][i]
            positions[i, k] = last[i]
    return times, positions, snap_cost


def resample(times: np.ndarray, pos: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Linearly interpolate per-drone positions (n, T, 3) onto a common time grid."""
    n = pos.shape[0]
    out = np.zeros((n, len(grid), 3))
    for i in range(n):
        for c in range(3):
            out[i, :, c] = np.interp(grid, times, pos[i, :, c])
    return out


def main() -> None:
    """Build both choreographies, render a static PNG and an animation GIF."""
    te, pe, cost_e = build_waypoints(use_snap=False)
    ts, ps, cost_s = build_waypoints(use_snap=True)
    pe, ps = pe / 100.0, ps / 100.0  # cm -> m

    n = SWARM.shape[0]
    colors = plt.cm.tab10(np.linspace(0, 1, n))
    panels = [(pe, te, cost_e, "Euclidean assignment"), (ps, ts, cost_s, "Min-snap assignment")]

    # ---- static traced-paths PNG ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    fig.suptitle(
        "Same sequence (rotate → form_star), two assignment metrics", fontsize=14, fontweight="bold"
    )
    for ax, (pos, _t, cost, label) in zip(axes, panels):
        improve = (
            "" if label.startswith("Eucl") else f"  ({100 * (cost_e - cost_s) / cost_e:+.0f}%)"
        )
        ax.set_title(f"{label}\ntotal snap cost: {cost:.0f}{improve}", fontsize=12)
        ax.set_aspect("equal")
        ax.set_xlim(-2.2, 2.2)
        ax.set_ylim(-2.2, 2.2)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.axhline(0, color="gray", lw=0.4, ls="--")
        ax.axvline(0, color="gray", lw=0.4, ls="--")
        for i in range(n):
            ax.plot(pos[i, :, 0], pos[i, :, 1], color=colors[i], lw=1.5, alpha=0.85)
            ax.scatter(
                *pos[i, 0, :2], color=colors[i], s=60, zorder=5, edgecolors="k", linewidths=0.4
            )
            ax.scatter(
                *pos[i, -1, :2],
                color=colors[i],
                s=150,
                marker="*",
                zorder=6,
                edgecolors="k",
                linewidths=0.4,
            )
    plt.tight_layout()
    plt.savefig("scripts/compare_choreographies.png", dpi=150)
    print("saved scripts/compare_choreographies.png")

    # ---- side-by-side animation ----
    grid = np.linspace(0.0, T_FORM1, 160)
    pe_g, ps_g = resample(te, pe, grid), resample(ts, ps, grid)
    figa, axa = plt.subplots(1, 2, figsize=(14, 7))
    figa.suptitle(
        "rotate → form_star: Euclidean (left) vs Min-snap (right)", fontsize=13, fontweight="bold"
    )
    dots, trails = [], []
    for ax, (_pos, _t, cost, label) in zip(axa, panels):
        ax.set_title(label, fontsize=12)
        ax.set_aspect("equal")
        ax.set_xlim(-2.2, 2.2)
        ax.set_ylim(-2.2, 2.2)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.axhline(0, color="gray", lw=0.4, ls="--")
        ax.axvline(0, color="gray", lw=0.4, ls="--")
    for ax, grid_pos in [(axa[0], pe_g), (axa[1], ps_g)]:
        d = [ax.plot([], [], "o", color=colors[i], ms=8, mec="k", mew=0.4)[0] for i in range(n)]
        tr = [ax.plot([], [], "-", color=colors[i], lw=1.2, alpha=0.6)[0] for i in range(n)]
        dots.append((d, grid_pos))
        trails.append((tr, grid_pos))

    def update(frame: int) -> list:
        artists = []
        for d, gp in dots:
            for i in range(n):
                d[i].set_data([gp[i, frame, 0]], [gp[i, frame, 1]])
                artists.append(d[i])
        for tr, gp in trails:
            for i in range(n):
                tr[i].set_data(gp[i, : frame + 1, 0], gp[i, : frame + 1, 1])
                artists.append(tr[i])
        return artists

    anim = animation.FuncAnimation(figa, update, frames=len(grid), interval=40, blit=True)
    anim.save("scripts/compare_choreographies.gif", writer=animation.PillowWriter(fps=25))
    print("saved scripts/compare_choreographies.gif")

    print(f"\nEuclidean total snap cost: {cost_e:.1f}")
    print(
        f"Min-snap  total snap cost: {cost_s:.1f}  ({100 * (cost_e - cost_s) / cost_e:.0f}% lower)"
    )


if __name__ == "__main__":
    main()
