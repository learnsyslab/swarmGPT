"""Render a saved preset with a scripted camera flythrough."""

from __future__ import annotations

import json
import logging
import math
import os
from collections import deque
from fractions import Fraction
from pathlib import Path

from drone_models.core import load_params
from drone_models.transform import motor_force2rotor_vel

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import imageio.v3 as imageio
import jax.numpy as jnp
import mujoco
import numpy as np
from crazyflow.control import Control
from crazyflow.sim import Physics, Sim
from crazyflow.sim.visualize import change_material, draw_line
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from swarm_gpt.core import AppBackend
from swarm_gpt.utils import generate_default_colors

ROOT = Path(__file__).resolve().parents[1]
MUSIC_DIR = ROOT / "music"
SCENE_XML = ROOT / "swarm_gpt/data/scene.xml"

# Pick a preset that matches the drone count in swarm_gpt/data/drones.toml.
PRESET_PATH = ROOT / "swarm_gpt/data/presets/Vivaldi Summer | 20 | 20260430_062211"
OUTPUT_PATH = ROOT / "renders/vivaldi_summer_flythrough.mp4"

RENDER_MODE = "rgb_array"
CAMERA_BODY_NAME = "render_camera_rig"
CAMERA_NAME = "cinema_cam"

CAMERA_MOVE_START_TIME = 2.0
CAMERA_MOVE_END_TIME = 10.0
CAMERA_START_POS = np.array([6.0, 0.0, 6.0], dtype=float)
CAMERA_END_POS = np.array([0.0, -2.98, 2.91], dtype=float)
CAMERA_LOOKAT = np.array([0.0, 0.0, 1.2], dtype=float)
CAMERA_UP = np.array([0.0, 0.0, 1.0], dtype=float)

WIDTH = 3840
HEIGHT = 2160
FPS = 60
TRAIL_LENGTH = 120

logger = logging.getLogger(__name__)


class FrameSink:
    """Write frames to a video through imageio."""

    def __init__(self, output_path: Path, fps: int):
        """Open an mp4 writer at the requested output path and frame rate."""
        self.output_path = output_path
        self.result_path = output_path
        self._writer = imageio.imopen(output_path, "w", plugin="pyav")
        self._writer.init_video_stream("libx264", fps=fps, pixel_format="yuv420p")
        stream = self._writer._video_stream
        if stream.codec_context.time_base is None:
            stream.codec_context.time_base = stream.time_base or Fraction(1, fps)

    @staticmethod
    def _normalize_frame(frame: np.ndarray) -> np.ndarray:
        """Convert a rendered frame into uint8 RGB data."""
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"Expected RGB frame with shape (H, W, 3), got {frame.shape}")
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(frame)

    def append_data(self, frame: np.ndarray) -> None:
        """Store a single rendered frame."""
        self._writer.write_frame(self._normalize_frame(frame))

    def close(self) -> None:
        """Close the video writer."""
        self._writer.close()
        self._writer.close = lambda: None


def camera_position_at(t: float) -> np.ndarray:
    """Orbit the camera around the look-at point and land at the configured end pose."""
    if CAMERA_MOVE_END_TIME <= CAMERA_MOVE_START_TIME:
        raise ValueError("CAMERA_MOVE_END_TIME must be larger than CAMERA_MOVE_START_TIME")
    alpha = (t - CAMERA_MOVE_START_TIME) / (CAMERA_MOVE_END_TIME - CAMERA_MOVE_START_TIME)
    alpha = float(np.clip(alpha, 0.0, 1.0))
    alpha = alpha * alpha * (3.0 - 2.0 * alpha)
    start_offset = CAMERA_START_POS - CAMERA_LOOKAT
    end_offset = CAMERA_END_POS - CAMERA_LOOKAT
    start_radius = np.linalg.norm(start_offset)
    end_radius = np.linalg.norm(end_offset)
    if start_radius == 0.0 or end_radius == 0.0:
        raise ValueError("Camera positions must differ from CAMERA_LOOKAT")

    start_azimuth = math.atan2(start_offset[1], start_offset[0])
    end_azimuth = math.atan2(end_offset[1], end_offset[0])
    azimuth_delta = (end_azimuth - start_azimuth + math.pi) % (2.0 * math.pi) - math.pi
    start_elevation = math.atan2(start_offset[2], np.linalg.norm(start_offset[:2]))
    end_elevation = math.atan2(end_offset[2], np.linalg.norm(end_offset[:2]))

    radius = (1.0 - alpha) * start_radius + alpha * end_radius
    azimuth = start_azimuth + alpha * azimuth_delta
    elevation = (1.0 - alpha) * start_elevation + alpha * end_elevation
    planar_radius = radius * math.cos(elevation)
    orbit_offset = np.array(
        [
            planar_radius * math.cos(azimuth),
            planar_radius * math.sin(azimuth),
            radius * math.sin(elevation),
        ],
        dtype=float,
    )
    return CAMERA_LOOKAT + orbit_offset


def look_at_quat(position: np.ndarray, target: np.ndarray, up_hint: np.ndarray) -> np.ndarray:
    """Build a MuJoCo quaternion so the camera points at a fixed target."""
    forward = target - position
    forward_norm = np.linalg.norm(forward)
    if forward_norm == 0.0:
        raise ValueError("Camera position and look-at point must differ")
    forward /= forward_norm

    up = up_hint / np.linalg.norm(up_hint)
    if abs(np.dot(forward, up)) > 0.99:
        up = np.array([0.0, 1.0, 0.0], dtype=float)

    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    true_up = np.cross(right, forward)
    true_up /= np.linalg.norm(true_up)

    rotation = np.column_stack((right, true_up, -forward))
    quat_xyzw = Rotation.from_matrix(rotation).as_quat()
    return np.roll(quat_xyzw, 1)


def get_camera_mocap_id(sim: Sim) -> int:
    """Resolve the mocap slot that drives the camera rig."""
    body_id = mujoco.mj_name2id(sim.mj_model, mujoco.mjtObj.mjOBJ_BODY, CAMERA_BODY_NAME)
    if body_id < 0:
        raise ValueError(f"Body {CAMERA_BODY_NAME!r} not found in {SCENE_XML}")
    mocap_id = int(sim.mj_model.body_mocapid[body_id])
    if mocap_id < 0:
        raise ValueError(f"Body {CAMERA_BODY_NAME!r} is not configured as a mocap body")
    camera_id = mujoco.mj_name2id(sim.mj_model, mujoco.mjtObj.mjOBJ_CAMERA, CAMERA_NAME)
    if camera_id < 0:
        raise ValueError(f"Camera {CAMERA_NAME!r} not found in {SCENE_XML}")
    return mocap_id


def set_camera_pose(sim: Sim, mocap_id: int, t: float) -> None:
    """Move the mocap camera rig and keep the camera aimed at the target."""
    position = camera_position_at(t)
    quat_wxyz = look_at_quat(position, CAMERA_LOOKAT, CAMERA_UP)
    sim.mjx_data = sim.mjx_data.replace(
        mocap_pos=sim.mjx_data.mocap_pos.at[0, mocap_id].set(jnp.asarray(position)),
        mocap_quat=sim.mjx_data.mocap_quat.at[0, mocap_id].set(jnp.asarray(quat_wxyz)),
    )


def build_sim(backend: AppBackend) -> Sim:
    """Create the Crazyflow simulation used for rendering the smoothed spline playback."""
    sim = Sim(
        n_worlds=1,
        n_drones=len(backend.splines),
        drone_model="cf21B_500",
        physics=Physics.first_principles,
        control=Control.state,
        freq=backend.settings["sim_freq"],
        attitude_freq=backend.settings["attitude_freq"],
        state_freq=backend.settings["state_freq"],
        device="cpu",
        xml_path=SCENE_XML,
    )
    sim.max_visual_geom = 100_000

    sim.reset()
    sim.state_control(np.random.random((1, sim.n_drones, 13)))
    sim.step(sim.freq // sim.control_freq)
    sim.reset()

    spline_ids = sorted(backend.splines)
    initial_pos = np.array([backend.splines[i](0.0) for i in spline_ids], dtype=float)[None, ...]
    hover_thrust = -sim.data.params.mass * sim.data.params.gravity_vec[2] / 4
    params = load_params("first_principles", "cf21B_500")
    hover_rpm = motor_force2rotor_vel(hover_thrust, params["rpm2thrust"])
    rotor_vel = jnp.ones_like(sim.data.states.rotor_vel, device=sim.device) * hover_rpm
    sim.data = sim.data.replace(
        states=sim.data.states.replace(
            pos=sim.data.states.pos.at[...].set(initial_pos),
            rotor_vel=sim.data.states.rotor_vel.at[...].set(rotor_vel),
        )
    )
    return sim


def render_preset(
    preset_path: Path = PRESET_PATH,
    output_path: Path = OUTPUT_PATH,
    render_end_time: float | None = None,
    width: int = WIDTH,
    height: int = HEIGHT,
    fps: int = FPS,
) -> Path:
    """Render a saved preset to a video file or frame directory."""
    preset_path = Path(preset_path)
    output_path = Path(output_path)
    if not preset_path.is_dir():
        raise FileNotFoundError(f"Preset directory not found: {preset_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    preset_meta = json.loads((preset_path / "meta.json").read_text())
    backend = AppBackend(
        music_dir=MUSIC_DIR,
        strict_processing=True,
        strict_drone_match=True,
        use_motion_primitives=bool(preset_meta["use_motion_primitives"]),
    )

    logger.info("Loading preset %s", preset_path.name)
    backend.initial_prompt(preset_path.name)
    for _ in backend.simulate(gui=False):
        pass

    if not backend.splines:
        raise RuntimeError("No splines were generated by the simulation pipeline")

    sim = build_sim(backend)
    mocap_id = get_camera_mocap_id(sim)
    spline_ids = sorted(backend.splines)
    pos_splines = [backend.splines[i] for i in spline_ids]
    vel_splines = [spline.derivative() for spline in pos_splines]

    rgbas = np.ones((sim.n_drones, 4), dtype=float)
    rgbas[:, :3] = generate_default_colors(sim.n_drones, limit=1.0)
    drone_ids = np.arange(sim.n_drones)
    trails = [deque(maxlen=TRAIL_LENGTH) for _ in range(sim.n_drones)]
    t_end = float(backend.waypoints["time"][0, -1])
    if render_end_time is not None:
        t_end = min(t_end, float(render_end_time))
    if fps <= 0:
        raise ValueError("fps must be positive")
    if sim.freq % sim.control_freq != 0:
        raise ValueError(
            f"sim_freq {sim.freq} must be divisible by control_freq {sim.control_freq}"
        )

    sim_dt = 1.0 / sim.freq
    control_steps = sim.freq // sim.control_freq
    total_sim_steps = max(0, math.ceil(t_end * sim.freq))
    total_frames = max(1, math.ceil(t_end * fps))

    def apply_control(current_time: float) -> None:
        desired_pos = np.array([spline(current_time) for spline in pos_splines], dtype=float)
        desired_vel = np.array([spline(current_time) for spline in vel_splines], dtype=float)
        controls = np.concatenate(
            (desired_pos, desired_vel, np.zeros((sim.n_drones, 7), dtype=float)), axis=-1
        )[None, ...]
        sim.state_control(controls)

    def render_frame(frame_time: float) -> None:
        positions = np.asarray(sim.data.states.pos[0])
        for i, trail in enumerate(trails):
            trail.append(positions[i])
            if len(trail) > 1:
                draw_line(sim, np.array(trail), rgba=rgbas[i], start_size=2, end_size=5)

        set_camera_pose(sim, mocap_id, frame_time)
        frame = sim.render(mode=RENDER_MODE, camera=CAMERA_NAME, width=width, height=height)
        if frame is None:
            raise RuntimeError("Crazyflow returned no frame in rgb_array mode")
        frame_sink.append_data(frame)

    change_material(
        sim,
        mat_name="led_top",
        drone_ids=drone_ids,
        rgba=rgbas[drone_ids],
        emission=np.ones((sim.n_drones,)),
    )
    change_material(
        sim,
        mat_name="led_bot",
        drone_ids=drone_ids,
        rgba=rgbas[drone_ids],
        emission=np.ones((sim.n_drones,)),
    )

    frame_sink = FrameSink(output_path, fps=fps)
    try:
        apply_control(0.0)
        next_control_step = control_steps
        next_frame_idx = 0
        current_time = 0.0

        with tqdm(total=total_frames, desc="Rendering", unit="frame") as progress:
            while next_frame_idx < total_frames and (next_frame_idx / fps) <= current_time:
                render_frame(next_frame_idx / fps)
                next_frame_idx += 1
                progress.update(1)

            for sim_step in range(1, total_sim_steps + 1):
                sim.step(1)
                current_time = sim_step * sim_dt

                if sim_step == next_control_step and current_time < t_end:
                    apply_control(current_time)
                    next_control_step += control_steps

                while next_frame_idx < total_frames and (next_frame_idx / fps) <= current_time:
                    render_frame(next_frame_idx / fps)
                    next_frame_idx += 1
                    progress.update(1)

            while next_frame_idx < total_frames:
                render_frame(next_frame_idx / fps)
                next_frame_idx += 1
                progress.update(1)
    finally:
        frame_sink.close()
        sim.close()

    logger.info("Saved render to %s", frame_sink.result_path)
    return frame_sink.result_path


def main(
    preset_path: Path = PRESET_PATH,
    output_path: Path = OUTPUT_PATH,
    render_end_time: float | None = None,
    width: int = WIDTH,
    height: int = HEIGHT,
    fps: int = FPS,
) -> Path:
    """Entrypoint for local rendering."""
    return render_preset(
        preset_path=preset_path,
        output_path=output_path,
        render_end_time=render_end_time,
        width=width,
        height=height,
        fps=fps,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
