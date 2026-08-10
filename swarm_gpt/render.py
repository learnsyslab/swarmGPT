"""Render a saved preset with a scripted camera flythrough."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import deque
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING

from drone_models.core import load_params
from drone_models.transform import motor_force2rotor_vel

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
# EGL is typical for headless Linux; macOS needs GLFW for MuJoCo rendering (rgb_array / viewer).
if sys.platform == "darwin":
    os.environ.setdefault("MUJOCO_GL", "glfw")
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
else:
    os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import imageio.v3 as imageio
import jax.numpy as jnp
import mujoco
import numpy as np
from crazyflow.control import Control
from crazyflow.sim import Physics, Sim
from crazyflow.sim.visualize import draw_line
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from swarm_gpt.core import AppBackend
from swarm_gpt.core.sim import TRAIL_RGBA, paint_lighting

if TYPE_CHECKING:
    from scipy.interpolate import BSpline

ROOT = Path(__file__).resolve().parents[1]
MUSIC_DIR = ROOT / "music" / "songs"
SCENE_XML = ROOT / "swarm_gpt/data/scene.xml"

# Pick a preset that matches the drone count in swarm_gpt/data/drones.toml.
PRESET_DIR = ROOT / "swarm_gpt/data/presets"
PRESET_PATH = PRESET_DIR / "The Blue Danube - Op. 314 | 20 | 20260601_004157"
OUTPUT_PATH = ROOT / "renders/the_blue_danube.mp4"

RENDER_MODE = "rgb_array"
CAMERA_BODY_NAME = "render_camera_rig"
CAMERA_NAME = "cinema_cam"

CAMERA_MOVE_START_TIME = 0.0
CAMERA_MOVE_END_TIME = 30.0
# The shape of the move, not its placement: the offsets from CAMERA_LOOKAT set the start and end
# azimuth and elevation and the ratio between the two distances, and the push-in that ratio
# encodes is preserved. Where the camera actually sits comes from the swarm's own extent, so one
# set of constants frames a 20-drone lab show in a 4m box and a 100-drone show in a 20m one.
CAMERA_START_POS = np.array([6.0, 0.0, 6.0], dtype=float)
CAMERA_END_POS = np.array([0.0, -6.00, 3.00], dtype=float)
CAMERA_LOOKAT = np.array([0.0, 0.0, 1.1], dtype=float)
CAMERA_UP = np.array([0.0, 0.0, 1.0], dtype=float)
CAMERA_FIT_MARGIN = 1.15  # Headroom on the exact frame fit, at the move's closest approach.

# Render-only gain on the LEDs' emission, because `lighting.toml` normalizes every hue to a
# constant channel *sum* -- right for the flown LEDs, but it leaves two-channel hues peaking at 0.5
# of the display's range while pure red sits at 1.0. See `paint_lighting` for why it can neither
# blow out nor shift a hue.
RENDER_EMISSION_GAIN = 2.5

WIDTH = 3840
HEIGHT = 2160
FPS = 60
# `--preview` resolution: fast enough to check a change, coarse enough not to be worth keeping.
PREVIEW_WIDTH = 1280
PREVIEW_HEIGHT = 720
PREVIEW_FPS = 30
TRAIL_LENGTH = 120

logger = logging.getLogger(__name__)


def preset_audio_path(preset_meta: dict[str, object]) -> Path:
    """Resolve the audio file declared by a preset's metadata."""
    song = preset_meta["song"]
    if not isinstance(song, str) or not song:
        raise ValueError("Preset metadata must contain a non-empty 'song' field")

    audio_path = MUSIC_DIR / f"{song}.mp3"
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file not found for song {song!r}: {audio_path}")
    return audio_path


def mux_audio(video_path: Path, audio_path: Path, duration: float, audio_start: float) -> Path:
    """Mux a song into an existing video, replacing the original file on success.

    ``audio_start`` is the `song_crops` window start: the trajectory is rebased to 0 while the mp3
    is not, so without the seek the render is 35 s out of sync for `Fearless2`.
    """
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to add audio to rendered videos")
    if duration <= 0.0:
        raise ValueError("duration must be positive")

    temp_file = tempfile.NamedTemporaryFile(
        prefix=f".{video_path.stem}.muxed.",
        suffix=video_path.suffix,
        dir=video_path.parent,
        delete=False,
    )
    muxed_path = Path(temp_file.name)
    temp_file.close()

    command = [
        ffmpeg,
        "-y",
        "-i",
        str(video_path),
        # Before `-i`, so ffmpeg seeks the input rather than decoding and discarding the lead-in.
        "-ss",
        f"{audio_start:.6f}",
        "-i",
        str(audio_path),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-t",
        f"{duration:.6f}",
        "-movflags",
        "+faststart",
        str(muxed_path),
    ]
    try:
        process = subprocess.run(command, capture_output=True, text=True, check=False)
        if process.returncode != 0:
            stderr = process.stderr.strip()
            raise RuntimeError(f"ffmpeg failed to mux audio into {video_path}: {stderr}")
        muxed_path.replace(video_path)
    finally:
        if muxed_path.exists():
            muxed_path.unlink()

    return video_path


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


def swarm_points(
    pos_splines: list[BSpline], t_end: float, samples: int = 512
) -> tuple[np.ndarray, np.ndarray]:
    """Sample where every drone is across the whole flight.

    Returns the centre of the swarm's bounding box and every sampled position as an ``(n, 3)``
    array, both in metres.
    """
    if not pos_splines:
        raise ValueError("At least one position spline is required to frame the camera")
    times = np.linspace(0.0, t_end, samples)
    pos = np.array([spline(times) for spline in pos_splines], dtype=float).reshape(-1, 3)
    centre = (pos.min(axis=0) + pos.max(axis=0)) / 2
    return centre, pos


def camera_fit_distance(
    sim: Sim,
    camera_id: int,
    centre: np.ndarray,
    points: np.ndarray,
    width: int,
    height: int,
    samples: int = 64,
) -> float:
    """Find the move's closest approach to ``centre`` that keeps every point inside the frame.

    The frame is a rectangle, so the half-angles solve separately: a fit satisfying only ``fovy``
    throws away width. Only depth scales with ``distance``, so each point names a fit; take the max.
    """
    fovy = math.radians(float(sim.mj_model.cam_fovy[camera_id]))
    # Half-angles of the actual frame: vertical is `fovy / 2`, horizontal widens it by the aspect.
    half_tans = (math.tan(fovy / 2) * width / height, math.tan(fovy / 2))
    offsets = np.asarray(points, dtype=float) - centre
    distance = 0.0
    for t in np.linspace(CAMERA_MOVE_START_TIME, CAMERA_MOVE_END_TIME, samples):
        # The same move at a fitted distance of one metre, so `span` is the depth each fitted
        # metre buys at this moment -- more than one, wherever the move is not at its closest.
        offset = camera_position_at(float(t), np.zeros(3), 1.0)
        span = float(np.linalg.norm(offset))
        forward = -offset / span
        depth = offsets @ forward
        for axis, half_tan in zip(camera_basis(forward, CAMERA_UP), half_tans, strict=True):
            required = (np.abs(offsets @ axis) / half_tan - depth).max() / span
            distance = max(distance, float(required))
    return distance * CAMERA_FIT_MARGIN


def camera_position_at(t: float, centre: np.ndarray, distance: float) -> np.ndarray:
    """Orbit the camera around the swarm and land at the configured end pose.

    ``distance`` is the move's closest approach to ``centre``; the configured poses rescale around
    it, so the push-in they encode survives at any swarm size.
    """
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

    # Rescale the configured distances so the move keeps its ratio and its closest approach is
    # `distance`. Pinning the closest point rather than the first one means the swarm still fits
    # after the push-in, which is where a move framed only at its start crops.
    scale = distance / min(start_radius, end_radius)
    radius = ((1.0 - alpha) * start_radius + alpha * end_radius) * scale
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
    return centre + orbit_offset


def camera_basis(forward: np.ndarray, up_hint: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build the right and up axes of a camera looking along ``forward``.

    Held apart from `look_at_quat` so the framing fit projects onto exactly the axes the renderer
    uses, fallback included; two copies would disagree about up and crop the swarm sideways.
    """
    up = up_hint / np.linalg.norm(up_hint)
    if abs(np.dot(forward, up)) > 0.99:
        up = np.array([0.0, 1.0, 0.0], dtype=float)

    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    true_up = np.cross(right, forward)
    return right, true_up / np.linalg.norm(true_up)


def look_at_quat(position: np.ndarray, target: np.ndarray, up_hint: np.ndarray) -> np.ndarray:
    """Build a MuJoCo quaternion so the camera points at a fixed target."""
    forward = target - position
    forward_norm = np.linalg.norm(forward)
    if forward_norm == 0.0:
        raise ValueError("Camera position and look-at point must differ")
    forward /= forward_norm

    right, true_up = camera_basis(forward, up_hint)
    rotation = np.column_stack((right, true_up, -forward))
    quat_xyzw = Rotation.from_matrix(rotation).as_quat()
    return np.roll(quat_xyzw, 1)


def get_camera_ids(sim: Sim) -> tuple[int, int]:
    """Resolve the ``(mocap id, camera id)`` of the camera rig and the camera riding on it."""
    body_id = mujoco.mj_name2id(sim.mj_model, mujoco.mjtObj.mjOBJ_BODY, CAMERA_BODY_NAME)
    if body_id < 0:
        raise ValueError(f"Body {CAMERA_BODY_NAME!r} not found in {SCENE_XML}")
    mocap_id = int(sim.mj_model.body_mocapid[body_id])
    if mocap_id < 0:
        raise ValueError(f"Body {CAMERA_BODY_NAME!r} is not configured as a mocap body")
    camera_id = mujoco.mj_name2id(sim.mj_model, mujoco.mjtObj.mjOBJ_CAMERA, CAMERA_NAME)
    if camera_id < 0:
        raise ValueError(f"Camera {CAMERA_NAME!r} not found in {SCENE_XML}")
    return mocap_id, camera_id


def set_camera_pose(sim: Sim, mocap_id: int, t: float, centre: np.ndarray, distance: float) -> None:
    """Move the mocap camera rig and keep the camera aimed at the swarm."""
    position = camera_position_at(t, centre, distance)
    quat_wxyz = look_at_quat(position, centre, CAMERA_UP)
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
    include_audio: bool = True,
) -> Path:
    """Render a saved preset to a video file or frame directory."""
    preset_path = Path(preset_path)
    output_path = Path(output_path)
    if not preset_path.is_dir():
        raise FileNotFoundError(f"Preset directory not found: {preset_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    preset_meta = json.loads((preset_path / "meta.json").read_text())
    audio_path = preset_audio_path(preset_meta) if include_audio else None
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
    mocap_id, camera_id = get_camera_ids(sim)
    spline_ids = sorted(backend.splines)
    pos_splines = [backend.splines[i] for i in spline_ids]
    vel_splines = [spline.derivative() for spline in pos_splines]

    lighting = backend.lighting_timeline()
    trails = [deque(maxlen=TRAIL_LENGTH) for _ in range(sim.n_drones)]
    t_end = float(backend.waypoints["time"][0, -1])
    if render_end_time is not None:
        t_end = min(t_end, float(render_end_time))
    centre, points = swarm_points(pos_splines, t_end)
    camera_distance = camera_fit_distance(sim, camera_id, centre, points, width, height)
    logger.info(
        "Framing %d drones in %dx%d: centre (%.2f, %.2f, %.2f) m, camera %.2f m out",
        sim.n_drones,
        width,
        height,
        *centre,
        camera_distance,
    )
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
        # Per frame, not once before the loop: the lighting timeline is a function of time.
        paint_lighting(sim, lighting, frame_time, emission_gain=RENDER_EMISSION_GAIN)
        positions = np.asarray(sim.data.states.pos[0])
        for i, trail in enumerate(trails):
            trail.append(positions[i])
            if len(trail) > 1:
                draw_line(sim, np.array(trail), rgba=TRAIL_RGBA, start_size=2, end_size=5)
        set_camera_pose(sim, mocap_id, frame_time, centre, camera_distance)
        frame = sim.render(mode=RENDER_MODE, camera=CAMERA_NAME, width=width, height=height)
        if frame is None:
            raise RuntimeError("Crazyflow returned no frame in rgb_array mode")
        frame_sink.append_data(frame)

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

    if audio_path is not None:
        # Same source `normalize_playback` reads for the web player's `audioOffset`.
        crop_start, _crop_end = backend.crop_window(backend.music_manager.song)
        mux_audio(
            frame_sink.result_path, audio_path, duration=total_frames / fps, audio_start=crop_start
        )

    logger.info("Saved render to %s", frame_sink.result_path)
    return frame_sink.result_path


def _resolve_preset(name: str) -> Path:
    """Find a preset directory by exact name, or by a case-insensitive substring matching only one.

    Preset names carry the song, drone count and timestamp, so they are long to type in full.
    """
    if (exact := PRESET_DIR / name).is_dir():
        return exact
    matches = sorted(
        d for d in PRESET_DIR.iterdir() if d.is_dir() and name.lower() in d.name.lower()
    )
    if not matches:
        raise SystemExit(f"No preset matches {name!r}. Use --list to see them.")
    if len(matches) > 1:
        listed = "\n  ".join(d.name for d in matches)
        raise SystemExit(f"{name!r} matches {len(matches)} presets, pick one:\n  {listed}")
    return matches[0]


def _default_output(preset_path: Path) -> Path:
    """Derive the ``renders/<slug>.mp4`` written when ``--out`` is not given."""
    slug = re.sub(r"[^a-z0-9]+", "_", preset_path.name.lower()).strip("_")
    return ROOT / "renders" / f"{slug}.mp4"


def main(argv: list[str] | None = None) -> Path:
    """Render a saved preset from the command line, returning the path written to."""
    parser = argparse.ArgumentParser(description="Render a saved preset to a video.")
    parser.add_argument(
        "preset",
        nargs="?",
        default=PRESET_PATH.name,
        help="preset name, or a substring matching exactly one (default: %(default)s)",
    )
    parser.add_argument(
        "-o", "--out", type=Path, help="output file (default: renders/<preset>.mp4)"
    )
    parser.add_argument("-s", "--seconds", type=float, help="stop after this many seconds of show")
    parser.add_argument(
        "--preview",
        action="store_true",
        help=f"quick pass at {PREVIEW_WIDTH}x{PREVIEW_HEIGHT} @ {PREVIEW_FPS}fps",
    )
    parser.add_argument("--width", type=int, help=f"frame width (default: {WIDTH})")
    parser.add_argument("--height", type=int, help=f"frame height (default: {HEIGHT})")
    parser.add_argument("--fps", type=int, help=f"frames per second (default: {FPS})")
    parser.add_argument("--no-audio", dest="audio", action="store_false", help="skip audio muxing")
    parser.add_argument("--list", action="store_true", help="list available presets and exit")
    args = parser.parse_args(argv)

    if args.list:
        for name in sorted(d.name for d in PRESET_DIR.iterdir() if d.is_dir()):
            print(name)  # stdout is this flag's output, not a diagnostic
        raise SystemExit(0)

    # An explicit --width/--height/--fps wins over --preview, so the two compose.
    preset_path = _resolve_preset(args.preset)
    defaults = (
        (PREVIEW_WIDTH, PREVIEW_HEIGHT, PREVIEW_FPS) if args.preview else (WIDTH, HEIGHT, FPS)
    )
    width, height, fps = (
        args.width or defaults[0],
        args.height or defaults[1],
        args.fps or defaults[2],
    )
    return render_preset(
        preset_path=preset_path,
        output_path=args.out or _default_output(preset_path),
        render_end_time=args.seconds,
        width=width,
        height=height,
        fps=fps,
        include_audio=args.audio,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
