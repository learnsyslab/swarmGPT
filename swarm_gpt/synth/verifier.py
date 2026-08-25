"""Run a primitive through the shipped safety filter and measure what the filter had to do.

The axswarm loop here is `swarm_gpt.core.sim.simulate_axswarm` minus crazyflow: that loop already
overwrites its state from ``solver_data.u_pos``/``u_vel`` rather than from the simulator, so the
MuJoCo pass only ever fed the log and the viewer. Dropping it costs no fidelity and makes the
synthesis loop fast enough to iterate.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import einops
import numpy as np
from axswarm import SolverData, SolverSettings, solve

from swarm_gpt.core.choreographer import dicts2arrays
from swarm_gpt.synth.sandbox import SynthError, call_guarded, validate_waypoints

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# Hold the final pose past the end of the primitive so the MPC's K-step lookahead is never empty
# over the interval being measured. K=50 at freq=10 Hz is a 5 s horizon.
_TAIL_S = 6.0


def authored_trajectory(
    fn: Callable[..., Any],
    args: tuple,
    start_pos_m: NDArray,
    tstart: float,
    tend: float,
    limits: dict[str, NDArray],
) -> dict[str, NDArray]:
    """Run a primitive and assemble the waypoint arrays the solver consumes.

    Returns:
        ``time`` of shape (D, T) and ``pos``/``vel``/``acc`` of shape (D, T, 3), positions in m.

    Raises:
        SynthError: If the primitive raises, times out, or breaks the output contract.
    """
    n_drones = start_pos_m.shape[0]
    swarm_pos_cm = np.asarray(start_pos_m, dtype=float) * 100
    result = call_guarded(fn, args, swarm_pos_cm.copy(), tstart, tend, limits)
    _, emitted = validate_waypoints(result, n_drones, tstart, tend)

    waypoints: dict[float, dict[int, NDArray]] = {
        0.0: {i: p.copy() for i, p in enumerate(swarm_pos_cm)}
    }
    for t, entry in emitted.items():
        waypoints[t] = entry if t not in waypoints else waypoints[t] | entry
    # A primitive may move a subset of drones, so every other drone holds its previous pose.
    ordered = sorted(waypoints)
    for previous_t, t in zip(ordered, ordered[1:]):
        for drone_id in range(n_drones):
            if drone_id not in waypoints[t]:
                waypoints[t][drone_id] = waypoints[previous_t][drone_id]
    final = waypoints[ordered[-1]]
    waypoints[ordered[-1] + _TAIL_S] = {i: p.copy() for i, p in final.items()}

    arrays = dicts2arrays(dict(sorted(waypoints.items())))
    pos = einops.rearrange(np.array(list(arrays.values())), "t d c -> d t c") / 100
    pos = np.clip(pos, limits["lower"], limits["upper"])
    time = np.tile(np.array(list(arrays.keys())), (n_drones, 1))
    return {"time": time, "pos": pos, "vel": np.zeros_like(pos), "acc": np.zeros_like(pos)}


def solve_only(waypoints: dict[str, NDArray], settings: dict) -> dict[str, NDArray]:
    """Drive axswarm over the authored waypoints and return the trajectory it actually commands.

    Returns:
        ``time`` of shape (T,), ``pos``/``vel`` of shape (T, D, 3), and a (T,) ``success`` mask.
    """
    solver_settings = SolverSettings(
        **{k: np.asarray(v) if isinstance(v, list) else v for k, v in settings["axswarm"].items()}
    )
    dynamics = settings["Dynamics"]
    solver_data = SolverData.init(
        waypoints=waypoints,
        K=solver_settings.K,
        N=solver_settings.N,
        A=np.asarray(dynamics["A"]),
        B=np.asarray(dynamics["B"]),
        A_prime=np.asarray(dynamics["A_prime"]),
        B_prime=np.asarray(dynamics["B_prime"]),
        freq=solver_settings.freq,
        smoothness_weight=solver_settings.smoothness_weight,
        input_smoothness_weight=solver_settings.input_smoothness_weight,
        input_continuity_weight=solver_settings.input_continuity_weight,
    )
    pos = np.asarray(waypoints["pos"][:, 0], dtype=np.float32)
    vel = np.zeros_like(pos)
    n_solves = int(float(waypoints["time"][0, -1]) * solver_settings.freq)

    positions, velocities, successes = [], [], []
    for step in range(n_solves):
        t = step / solver_settings.freq
        success, _, solver_data = solve(
            np.concat((pos, vel), axis=-1), t, solver_data, solver_settings
        )
        solver_data = solver_data.step(solver_data)
        pos, vel = np.asarray(solver_data.u_pos[:, 0]), np.asarray(solver_data.u_vel[:, 0])
        positions.append(pos.copy())
        velocities.append(vel.copy())
        successes.append(bool(np.all(success)))

    return {
        "time": np.arange(n_solves) / solver_settings.freq,
        "pos": np.stack(positions),
        "vel": np.stack(velocities),
        "success": np.array(successes),
    }


def _pairwise_min(pos: NDArray, envelope: NDArray) -> tuple[NDArray, NDArray, NDArray]:
    """Per-timestep closest pair under the envelope metric, as ``(norm, i, j)`` arrays."""
    scaled = pos / envelope
    diff = scaled[:, :, None, :] - scaled[:, None, :, :]
    norm = np.linalg.norm(diff, axis=-1)
    n = pos.shape[1]
    norm = norm + np.eye(n) * 1e6
    flat = norm.reshape(norm.shape[0], -1).argmin(axis=1)
    return norm.min(axis=(1, 2)), flat // n, flat % n


def _interpolate(authored: dict[str, NDArray], time: NDArray) -> NDArray:
    """Sample the authored waypoints onto ``time``, returning (T, D, 3) in m.

    Linear between waypoints: that is what the solver is asked to track, so it is also the honest
    reference for how far the repair moved things.
    """
    src_t, src_pos = authored["time"][0], authored["pos"]
    out = np.empty((len(time), src_pos.shape[0], 3))
    for d in range(src_pos.shape[0]):
        for axis in range(3):
            out[:, d, axis] = np.interp(time, src_t, src_pos[d, :, axis])
    return out


def screen_authored(
    authored: dict[str, NDArray], settings: dict, window: tuple[float, float]
) -> tuple[dict[str, Any], list[str]]:
    """Judge the authored trajectory on its own, before paying for a solve.

    A solve costs ~30 s and repairs nothing when the waypoints already collide or demand motion no
    drone can fly. Reuses the grid and metric `measure` uses, so ``authored_min_sep_norm`` here is
    the figure it would report.

    Returns:
        The measured dict, and one sentence per broken limit (empty if worth solving).
    """
    axswarm = settings["axswarm"]
    freq = axswarm["freq"]
    envelope = np.asarray(axswarm["collision_envelope"], dtype=float)
    time = np.arange(int(float(authored["time"][0, -1]) * freq)) / freq
    time = time[(time >= window[0]) & (time <= window[1])]
    reference = _interpolate(authored, time)

    norm, worst_i, worst_j = _pairwise_min(reference, envelope)
    step = int(norm.argmin())
    speed = np.linalg.norm(np.diff(reference, axis=0), axis=-1) * freq
    accel = np.linalg.norm(np.diff(reference, n=2, axis=0), axis=-1) * freq * freq
    measured = {
        "authored_min_sep_norm": float(norm[step]),
        "worst_pair": [int(worst_i[step]), int(worst_j[step])],
        "worst_time_s": float(time[step]),
        "authored_max_speed_mps": float(speed.max()) if speed.size else 0.0,
        "authored_max_accel_mps2": float(accel.max()) if accel.size else 0.0,
    }

    violations = []
    if measured["authored_min_sep_norm"] < 1.0:
        i, j = measured["worst_pair"]
        violations.append(
            f"drones {i} and {j} close to {measured['authored_min_sep_norm']:.3f} of the required "
            f"separation at t={measured['worst_time_s']:.1f} s, which must be at least 1.0"
        )
    if measured["authored_max_speed_mps"] > axswarm["vel_max"]:
        violations.append(
            f"peak speed {measured['authored_max_speed_mps']:.2f} m/s exceeds the drones' "
            f"{axswarm['vel_max']} m/s limit"
        )
    if measured["authored_max_accel_mps2"] > axswarm["acc_max"]:
        violations.append(
            f"peak acceleration {measured['authored_max_accel_mps2']:.2f} m/s^2 exceeds the "
            f"drones' {axswarm['acc_max']} m/s^2 limit"
        )
    return measured, violations


def measure(
    authored: dict[str, NDArray],
    repaired: dict[str, NDArray],
    settings: dict,
    window: tuple[float, float],
) -> dict[str, Any]:
    """Compare what the filter flew against what the primitive authored, over ``window``.

    The settle tail is excluded: holding a final pose is neither the primitive's intent nor its
    fault, and averaging over it would dilute every deviation figure.

    Returns:
        A flat dict of magnitudes; every feedback encoding renders from this one dict.
    """
    envelope = np.asarray(settings["axswarm"]["collision_envelope"], dtype=float)
    inside = (repaired["time"] >= window[0]) & (repaired["time"] <= window[1])
    pos, time = repaired["pos"][inside], repaired["time"][inside]
    success = repaired["success"][inside]
    vel = repaired["vel"][inside]
    reference = _interpolate(authored, time)

    norm, worst_i, worst_j = _pairwise_min(pos, envelope)
    worst_step = int(norm.argmin())
    i, j = int(worst_i[worst_step]), int(worst_j[worst_step])
    gap = pos[worst_step, i] - pos[worst_step, j]
    min_sep_m = float(np.linalg.norm(gap))
    # The separation the envelope demands along the direction this pair actually approached from.
    # A single metre figure is only meaningful once the direction is fixed, because the envelope
    # is much deeper in z than in x/y.
    required_m = min_sep_m / float(norm[worst_step]) if norm[worst_step] > 0 else float("inf")
    # Second-worst moment outside a 1 s neighbourhood of the worst, so "next worst" names a
    # different event rather than the adjacent step of the same one.
    freq = settings["axswarm"]["freq"]
    masked = norm.copy()
    masked[max(0, worst_step - freq) : worst_step + freq + 1] = np.inf
    next_worst_norm = float(masked.min()) if np.isfinite(masked).any() else float("inf")

    authored_norm, _, _ = _pairwise_min(reference, envelope)
    deviation = np.linalg.norm(pos - reference, axis=-1)  # (T, D)
    per_drone_max = deviation.max(axis=0)
    speed = np.linalg.norm(vel, axis=-1)
    accel = np.gradient(vel, 1.0 / freq, axis=0)

    return {
        "n_drones": int(pos.shape[1]),
        "duration_s": float(time[-1]),
        "n_steps": int(len(time)),
        "min_sep_m": min_sep_m,
        "min_sep_norm": float(norm[worst_step]),
        "required_sep_m": required_m,
        "worst_pair": (i + 1, j + 1),  # 1-indexed, as the LLM addresses drones
        "worst_time_s": float(time[worst_step]),
        "next_worst_norm": next_worst_norm,
        "steps_inside_envelope": int((norm < 1.0).sum()),
        "authored_min_sep_norm": float(authored_norm.min()),
        "deviation_mean_m": float(deviation.mean()),
        "deviation_max_m": float(deviation.max()),
        "deviation_per_drone_max_m": per_drone_max.tolist(),
        "deviation_worst_drone": int(per_drone_max.argmax()) + 1,
        "failed_solves": int((~success).sum()),
        "max_speed_mps": float(speed.max()),
        "max_accel_mps2": float(np.linalg.norm(accel, axis=-1).max()),
        "min_z_m": float(pos[:, :, 2].min()),
        "vel_max_mps": float(settings["axswarm"]["vel_max"]),
    }


def check_invariants(
    check_fn: Callable[..., Any],
    repaired: dict[str, NDArray],
    args: tuple,
    window: tuple[float, float],
) -> list[dict[str, Any]]:
    """Run the primitive author's own shape check over the flown trajectory inside ``window``.

    Positions are handed over in **cm** and shaped (D, T, 3), so the check works in the same units
    the primitive was written in.

    Returns:
        One ``{"name", "ok", "detail"}`` entry per declared invariant.

    Raises:
        SynthError: If the check raises or returns something other than (name, ok, detail) triples.
    """
    inside = (repaired["time"] >= window[0]) & (repaired["time"] <= window[1])
    pos_cm = np.transpose(repaired["pos"][inside], (1, 0, 2)) * 100
    result = call_guarded(check_fn, pos_cm, repaired["time"][inside], args)
    if not isinstance(result, (list, tuple)) or not result:
        raise SynthError(
            "check must return a non-empty list of (name, ok, detail) triples, got "
            f"{type(result).__name__}."
        )
    checks = []
    for entry in result:
        if not isinstance(entry, (list, tuple)) or len(entry) != 3:
            raise SynthError(f"Each check entry must be a (name, ok, detail) triple, got {entry!r}")
        name, ok, detail = entry
        checks.append({"name": str(name), "ok": bool(ok), "detail": str(detail)})
    return checks
