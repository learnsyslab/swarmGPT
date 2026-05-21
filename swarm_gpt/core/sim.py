"""Simulation module for swarm_gpt.

Before we deploy the choreography to the drones, we run a simulation to check if the modified paths
from AMSwarm are collision-free and can be executed. While there is no guarantee that the
trajectories work in reality, it is a good sanity check to ensure that the drones do not crash into
each other or have to perform infeasible maneuvers.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING

import jax
import numpy as np
from axswarm import SolverData, SolverSettings, solve
from crazyflow.control import Control
from crazyflow.sim import Physics, Sim
from crazyflow.sim.visualize import change_material, draw_line
from drone_models.core import load_params
from drone_models.transform import motor_force2rotor_vel
from tqdm import tqdm

from swarm_gpt.utils import MusicManager, generate_default_colors

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from scipy.interpolate import BSpline

    from swarm_gpt.utils import MusicManager


def simulate_axswarm(
    waypoints: dict[str, NDArray], settings: dict, gui: bool = False
) -> dict[int, NDArray]:
    """Run the crazyflow simulation from waypoints.

    Args:
        waypoints: The waypoints to fly to. Dictionary of drone IDs to waypoints. Each waypoint
            consists of [time, x, y, z, vx, vy, vz].
        settings: Settings for the simulation and AMSwarm.
        gui: Flag to render the simulation.

    Returns:
        A collection of data from the simulation.
    """
    # Set up the simulation
    sim = Sim(
        n_worlds=1,
        n_drones=waypoints["pos"].shape[0],
        drone_model="cf21B_500",
        physics=Physics.first_principles,
        control=Control.state,
        freq=settings["sim_freq"],
        attitude_freq=settings["attitude_freq"],
        state_freq=settings["state_freq"],
        device="cpu",
        xml_path=Path("swarm_gpt/data/scene.xml"),
    )
    fps = 60
    sim.max_visual_geom = 100_000

    # JIT compile the simulation
    sim.reset()
    sim.state_control(np.random.random((sim.n_worlds, sim.n_drones, 13)))
    sim.step(sim.freq // sim.control_freq)
    sim.reset()

    # Set up solver
    solver_settings = {
        k: v if not isinstance(v, list) else np.asarray(v) for k, v in settings["axswarm"].items()
    }
    solver_settings = SolverSettings(**solver_settings)
    dynamics = settings["Dynamics"]
    A, B = np.asarray(dynamics["A"]), np.asarray(dynamics["B"])
    A_prime, B_prime = np.asarray(dynamics["A_prime"]), np.asarray(dynamics["B_prime"])
    solver_data = SolverData.init(
        waypoints=waypoints,
        K=solver_settings.K,
        N=solver_settings.N,
        A=A,
        B=B,
        A_prime=A_prime,
        B_prime=B_prime,
        freq=solver_settings.freq,
        smoothness_weight=solver_settings.smoothness_weight,
        input_smoothness_weight=solver_settings.input_smoothness_weight,
        input_continuity_weight=solver_settings.input_continuity_weight,
    )
    n_steps = int(waypoints["time"][0, -1] * sim.control_freq)
    solve_every_n_steps = sim.control_freq // solver_settings.freq
    assert sim.freq % sim.control_freq == 0, (
        "control freq {sim.control_freq} must be divisible by sim.freq {sim.freq}"
    )
    assert sim.control_freq % solver_settings.freq == 0, (
        "control freq {sim.control_freq} must be divisible by amswarm freq {solver_settings.freq}"
    )

    # Set up initial states
    control = np.zeros((sim.n_worlds, sim.n_drones, 13), dtype=np.float32)
    pos = sim.data.states.pos.at[0, ...].set(waypoints["pos"][:, 0])
    sim.data = sim.data.replace(states=sim.data.states.replace(pos=pos))
    pos, vel = np.asarray(sim.data.states.pos[0]), np.asarray(sim.data.states.vel[0])
    states, controls, solve_times = [], [], []  # logging variables

    # Set up colours for tracking lines
    rng = np.random.default_rng(0)
    rgbas = rng.random((sim.n_drones, 4))
    rgbas[..., 3] = 1

    tstart = time.time()
    for step in tqdm(range(n_steps)):
        yield "progress", step + 1, n_steps
        t = step / sim.control_freq
        if step % solve_every_n_steps == 0:
            state = np.concat((pos, vel), axis=-1)
            t_solve = time.perf_counter()
            success, _, solver_data = solve(state, t, solver_data, solver_settings)
            jax.block_until_ready(solver_data)
            solve_times.append(time.perf_counter() - t_solve)
            if not all(success):
                logger.info("Solve failed")

            solver_data = solver_data.step(solver_data)
            pos, vel = solver_data.u_pos[:, 0], solver_data.u_vel[:, 0]
            control[0, :, :3] = solver_data.u_pos[:, 0]
            control[0, :, 3:6] = solver_data.u_vel[:, 0]

            # Log inputs
            controls.append(control[0, :, :6].copy())

        # Run the simulation
        sim.state_control(control)
        sim.step(sim.freq // sim.control_freq)

        # Store the state
        states.append(
            np.concatenate(
                (
                    np.asarray(sim.data.states.pos[0]),
                    np.asarray(sim.data.states.quat[0]),
                    np.asarray(sim.data.states.vel[0]),
                    np.asarray(sim.data.states.ang_vel[0]),
                ),
                axis=-1,
            )
        )

        # Render simulation with visualizations of the planned trajectories
        if ((step * fps) % sim.control_freq) < fps and gui:
            for i in range(sim.n_drones):
                draw_line(sim, solver_data.u_pos[i, :], rgba=rgbas[i % len(rgbas)])
            sim.render()
            if (dt := t - (time.time() - tstart)) > 0:
                time.sleep(dt)
    sim.close()

    timestamps = np.arange(n_steps) / sim.control_freq
    states = np.array(states)
    if len(states) != len(timestamps):
        raise RuntimeError(
            f"Simulation log mismatch: {len(states)} states for {len(timestamps)} timestamps"
        )
    if states.shape[1:] != (sim.n_drones, 13):
        raise RuntimeError(
            f"Expected states with shape (T, {sim.n_drones}, 13), got {states.shape}"
        )

    sim_log = {
        "num_drones": sim.n_drones,
        "log_freq": solver_settings.freq,
        "sim_freq": sim.freq,
        "timestamps": timestamps,
        "states": states,
        "controls": np.array(controls),
        "waypoints": waypoints,
        "simulation_freq": sim.freq,
        "amswarm_every_n_steps": solve_every_n_steps,
        "solve_times": np.array(solve_times),
    }
    yield "result", sim_log, "placeholder"
    # return sim_log


def replay_sim_states(
    sim_data: dict[str, NDArray], settings: dict, music_manager: MusicManager | None = None
) -> None:
    """Replay a previously recorded Crazyflow state log in MuJoCo.

    This is a debug viewer for the exact states produced by ``simulate_axswarm``. Unlike
    ``simulate_spline``, it does not run another controller/physics pass.
    """
    timestamps = np.asarray(sim_data["timestamps"], dtype=float)
    states = np.asarray(sim_data["states"], dtype=np.float32)
    if states.ndim != 3 or states.shape[-1] != 13:
        raise ValueError(f"Expected states with shape (T, drones, 13), got {states.shape}")
    if len(states) != len(timestamps):
        raise ValueError(f"State/timestamp mismatch: {len(states)} != {len(timestamps)}")
    if len(states) == 0:
        return

    fps = 60
    default_cam_config = {"distance": 4.0, "azimuth": 180, "elevation": -20, "lookat": [0, 0, 1]}
    sim = Sim(
        n_worlds=1,
        n_drones=int(sim_data["num_drones"]),
        drone_model="cf21B_500",
        physics=Physics.first_principles,
        control=Control.state,
        freq=settings["sim_freq"],
        attitude_freq=settings["attitude_freq"],
        state_freq=settings["state_freq"],
        device="cpu",
        xml_path=Path("swarm_gpt/data/scene.xml"),
    )
    sim.max_visual_geom = 100_000

    rgbas = np.ones((sim.n_drones, 4))
    rgbas[:, :3] = generate_default_colors(sim.n_drones, limit=1.0)
    swarm_pos = [deque(maxlen=100) for _ in range(sim.n_drones)]

    def sample_state(t: float) -> NDArray:
        if len(states) == 1:
            return states[0]
        t = float(np.clip(t, timestamps[0], timestamps[-1]))
        idx = int(np.searchsorted(timestamps, t, side="right") - 1)
        idx = min(max(idx, 0), len(timestamps) - 2)
        t0, t1 = timestamps[idx], timestamps[idx + 1]
        alpha = 0.0 if t1 <= t0 else (t - t0) / (t1 - t0)
        frame = (1 - alpha) * states[idx] + alpha * states[idx + 1]

        # Keep quaternion interpolation on the shortest arc and normalized.
        q0 = states[idx, :, 3:7]
        q1 = states[idx + 1, :, 3:7]
        q1 = np.where(np.sum(q0 * q1, axis=-1, keepdims=True) < 0, -q1, q1)
        quat = (1 - alpha) * q0 + alpha * q1
        quat /= np.linalg.norm(quat, axis=-1, keepdims=True) + 1e-8
        frame[:, 3:7] = quat
        return frame.astype(np.float32, copy=False)

    def set_state(frame: NDArray) -> None:
        nonlocal sim
        sim.data = sim.data.replace(
            states=sim.data.states.replace(
                pos=sim.data.states.pos.at[0, ...].set(frame[:, 0:3]),
                quat=sim.data.states.quat.at[0, ...].set(frame[:, 3:7]),
                vel=sim.data.states.vel.at[0, ...].set(frame[:, 7:10]),
                ang_vel=sim.data.states.ang_vel.at[0, ...].set(frame[:, 10:13]),
            ),
            core=sim.data.core.replace(mjx_synced=sim.data.core.mjx_synced.at[...].set(False)),
        )

    set_state(states[0])
    sim.render(cam_config=default_cam_config)
    time.sleep(0.5)  # Wait for the viewer to initialize
    assert music_manager is not None, "Music manager is required for debug replay"
    try:
        music_manager.play(wait=True)
    except RuntimeError as exc:
        logger.warning("Could not start music playback for debug replay: %s", exc)

    tstart = time.perf_counter()
    last_progress_time = timestamps[0]
    try:
        with tqdm(total=float(timestamps[-1] - timestamps[0]), unit="s") as progress:
            while True:
                t_frame_start = time.perf_counter()
                t_playback = timestamps[0] + (t_frame_start - tstart)

                t = float(np.clip(t_playback, timestamps[0], timestamps[-1]))
                frame = sample_state(t)
                progress.update(max(0.0, t - last_progress_time))
                last_progress_time = t

                set_state(frame)
                for j, dq in enumerate(swarm_pos):
                    dq.append(frame[j, 0:3])
                    draw_line(
                        sim, np.array(dq), rgba=rgbas[j % len(rgbas)], start_size=2, end_size=5
                    )

                change_material(
                    sim,
                    mat_name="led_top",
                    drone_ids=np.arange(sim.n_drones),
                    rgba=rgbas[np.arange(sim.n_drones) % len(rgbas)],
                    emission=np.ones((sim.n_drones,)),
                )
                change_material(
                    sim,
                    mat_name="led_bot",
                    drone_ids=np.arange(sim.n_drones),
                    rgba=rgbas[np.arange(sim.n_drones) % len(rgbas)],
                    emission=np.ones((sim.n_drones,)),
                )

                sim.render(cam_config=default_cam_config)
                if t_playback >= timestamps[-1]:
                    break
                if (dt := (1 / fps) - (time.perf_counter() - t_frame_start)) > 0:
                    time.sleep(dt)
    finally:
        if music_manager is not None:
            music_manager.stop()
        sim.close()
