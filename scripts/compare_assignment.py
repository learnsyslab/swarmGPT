"""Compare Euclidean vs min-snap Hungarian assignment.

Scenario: 4 drones with diagonal velocities approach a scattered formation.
The velocity arrows clearly show that Euclidean sends drones to targets they are
moving *away* from, while min-snap aligns each drone with a momentum-compatible target.

Run with:
    pixi run python scripts/compare_assignment.py
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import minsnap_trajectories as ms
from scipy.optimize import linear_sum_assignment

matplotlib.use("Agg")  # headless-safe; comment out for an interactive window

from swarm_gpt.core.motion_primitives import _minsnap_cost_matrix  # noqa: E402


def trajectory_xy(p0: np.ndarray, v0: np.ndarray, pf: np.ndarray, T: float, n: int = 80) -> tuple:
    """Return (x, y) arrays for the min-snap trajectory from p0 to pf."""
    waypoints = [
        ms.Waypoint(time=0.0, position=p0, velocity=v0),
        ms.Waypoint(time=T, position=pf),
    ]
    traj = ms.generate_trajectory(waypoints, degree=8, idx_minimized_orders=4)
    t_samples = np.linspace(0.0, T, n)
    pos_t = ms.compute_trajectory_derivatives(traj, t_samples, num_orders=1)[0]
    return pos_t[:, 0], pos_t[:, 1]


def snap_cost(p0: np.ndarray, v0: np.ndarray, pf: np.ndarray, T: float) -> float:
    """Compute min-snap trajectory cost from (p0, v0) to (pf, 0) in time T."""
    waypoints = [
        ms.Waypoint(time=0.0, position=p0, velocity=v0),
        ms.Waypoint(time=T, position=pf),
    ]
    traj = ms.generate_trajectory(waypoints, degree=8, idx_minimized_orders=4)
    t_samples = np.linspace(0.0, T, 40)
    derivs = ms.compute_trajectory_derivatives(traj, t_samples, num_orders=5)
    return float(np.trapezoid(np.sum(derivs[4] ** 2, axis=-1), t_samples))


# ---------------------------------------------------------------------------
# Scenario: 4 drones with diagonal velocities, targets offset from headings
# ---------------------------------------------------------------------------

Z = 100.0
speed = 80.0  # cm/s

pos = np.array([
    [-100.0, -50.0, Z],   # drone 0
    [ 100.0,  50.0, Z],   # drone 1
    [-100.0,  80.0, Z],   # drone 2
    [ 100.0, -80.0, Z],   # drone 3
])

vel = np.array([
    [ speed,  speed * 0.3, 0.0],   # drone 0: mostly right
    [-speed, -speed * 0.3, 0.0],   # drone 1: mostly left
    [ speed * 0.3, -speed, 0.0],   # drone 2: mostly down
    [-speed * 0.3,  speed, 0.0],   # drone 3: mostly up
])

targets = np.array([
    [ 120.0,  -60.0, Z],   # T0
    [-120.0,   60.0, Z],   # T1
    [  60.0,  120.0, Z],   # T2
    [ -60.0, -120.0, Z],   # T3
])

T = 4.0  # seconds

# --- Assignments ---
dist_matrix = np.linalg.norm(pos[:, None] - targets[None, :], axis=-1)
euc_asgn = linear_sum_assignment(dist_matrix)[1]
ms_asgn = linear_sum_assignment(_minsnap_cost_matrix(pos, targets, vel, T))[1]

# --- Evaluate true snap cost for each ---
euc_total = sum(snap_cost(pos[i] / 100, vel[i] / 100, targets[euc_asgn[i]] / 100, T) for i in range(4))
ms_total = sum(snap_cost(pos[i] / 100, vel[i] / 100, targets[ms_asgn[i]] / 100, T) for i in range(4))

print(f"Euclidean assignment: {euc_asgn}  total snap cost: {euc_total:.1f}")
print(f"Min-snap assignment:  {ms_asgn}  total snap cost: {ms_total:.1f}")
print(f"Reduction: {(euc_total - ms_total) / euc_total * 100:.0f}%")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

pos_m, vel_ms, tgt_m = pos / 100, vel / 100, targets / 100
COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
t_samples = np.linspace(0.0, T, 80)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle(
    "Euclidean vs Min-Snap Assignment  |  4 drones with diagonal velocities, T = 4 s",
    fontsize=13,
    fontweight="bold",
)

for ax_idx, (assignment, label, total) in enumerate(
    [(euc_asgn, "Euclidean", euc_total), (ms_asgn, "Min-Snap", ms_total)]
):
    ax = axes[ax_idx]
    improvement = "" if ax_idx == 0 else f"  ({(euc_total - ms_total) / euc_total * 100:.0f}% less snap)"
    ax.set_title(f"{label} assignment\nTotal snap cost: {total:.1f}{improvement}", fontsize=11)
    ax.set_aspect("equal")
    ax.set_xlim(-2.0, 2.0)
    ax.set_ylim(-2.0, 2.0)
    ax.axhline(0, color="gray", lw=0.5, ls="--")
    ax.axvline(0, color="gray", lw=0.5, ls="--")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.scatter(tgt_m[:, 0], tgt_m[:, 1], marker="*", s=220, color="gold", zorder=5,
               edgecolors="goldenrod", linewidths=0.5)
    for j in range(4):
        ax.annotate(f"T{j}", tgt_m[j, :2] + 0.07, fontsize=8, color="goldenrod", fontweight="bold")
    for i in range(4):
        c = COLORS[i]
        j = assignment[i]
        xs, ys = trajectory_xy(pos_m[i], vel_ms[i], tgt_m[j], T)
        ax.plot(xs, ys, color=c, lw=2.0, alpha=0.85)
        ax.scatter(*pos_m[i, :2], color=c, s=90, zorder=6, edgecolors="k", linewidths=0.5)
        vn = vel_ms[i, :2]
        ax.annotate("", xy=pos_m[i, :2] + vn / np.linalg.norm(vn) * 0.35, xytext=pos_m[i, :2],
                    arrowprops=dict(arrowstyle="->", color=c, lw=2.0))
        ax.annotate(f"D{i}", pos_m[i, :2] + np.array([-0.18, 0.10]), fontsize=8, color=c, fontweight="bold")

ax_snap = axes[2]
ax_snap.set_title("Snap² over time per drone\n(solid = min-snap assignment, dashed = euclidean)", fontsize=10)
ax_snap.set_xlabel("time (s)")
ax_snap.set_ylabel("||snap||²  (m⁴/s⁸)")
for i in range(4):
    c = COLORS[i]
    for assignment, ls in [(euc_asgn, "--"), (ms_asgn, "-")]:
        j = assignment[i]
        wpts = [
            ms.Waypoint(time=0.0, position=pos_m[i], velocity=vel_ms[i]),
            ms.Waypoint(time=T, position=tgt_m[j]),
        ]
        traj = ms.generate_trajectory(wpts, degree=8, idx_minimized_orders=4)
        derivs = ms.compute_trajectory_derivatives(traj, t_samples, num_orders=5)
        snap_sq = np.sum(derivs[4] ** 2, axis=-1)
        ax_snap.plot(t_samples, snap_sq, color=c, ls=ls, lw=1.6, alpha=0.85)

legend_handles = (
    [plt.Line2D([0], [0], color=COLORS[i], lw=2, label=f"Drone {i}") for i in range(4)]
    + [
        plt.Line2D([0], [0], color="gray", lw=1.5, ls="--", label="euclidean assign."),
        plt.Line2D([0], [0], color="gray", lw=1.5, ls="-", label="min-snap assign."),
    ]
)
ax_snap.legend(handles=legend_handles, fontsize=8)

plt.tight_layout()
out_path = "scripts/compare_assignment.png"
plt.savefig(out_path, dpi=150)
print(f"\nPlot saved to {out_path}")
