"""Unit tests for the offline renderer's audio muxing, camera framing and LED brightness.

`render_preset` needs a full backend, the axswarm pass and an offscreen MuJoCo context, so the
wiring stays unpinned. The ffmpeg command and the camera geometry are reachable without it.
"""

import math
import subprocess
from pathlib import Path

import numpy as np
import pytest
from scipy.interpolate import BSpline, make_interp_spline
from scipy.spatial.transform import Rotation

import swarm_gpt.render as render
from swarm_gpt.core import sim as sim_module

FOVY_DEG = 39.2  # The `cinema_cam` fovy in swarm_gpt/data/scene.xml.
# Lab scale (a 4 m box) and sim scale (the 20 m box the 100-drone config needs).
SWARM_SCALES = [(20, 2.0, 0.25, 1.7), (100, 10.0, 0.25, 6.0)]


class _StubSim:
    """Stands in for `Sim`, which only `cam_fovy` is read off during framing."""

    def __init__(self, fovy_deg: float = FOVY_DEG):
        self.mj_model = type("_Model", (), {"cam_fovy": np.array([fovy_deg])})()


def _orbit_splines(n_drones: int, radius: float, z_low: float, z_high: float) -> list[BSpline]:
    """Build position splines for a swarm circling `radius` out while climbing z_low -> z_high."""
    times = np.linspace(0.0, 60.0, 64)
    splines = []
    for i in range(n_drones):
        phase = 2 * np.pi * i / n_drones
        angle = phase + np.linspace(0.0, 2 * np.pi, times.size)
        pos = np.stack(
            [
                radius * np.cos(angle),
                radius * np.sin(angle),
                np.linspace(z_low, z_high, times.size),
            ],
            axis=-1,
        )
        splines.append(make_interp_spline(times, pos, k=3))
    return splines


def _worst_frame_fill(
    splines: list[BSpline],
    centre: np.ndarray,
    distance: float,
    width: int = render.WIDTH,
    height: int = render.HEIGHT,
) -> float:
    """How far across the frame the swarm ever reaches, as a fraction of the frame's half-extent.

    Above 1 something was cropped, well below 1 the camera is parked too far out. Axes come from
    `look_at_quat`, not the fit's own basis, which is what catches the two disagreeing.
    """
    tan_v = math.tan(math.radians(FOVY_DEG) / 2)
    tan_h = tan_v * width / height
    worst = 0.0
    for t in np.linspace(render.CAMERA_MOVE_START_TIME, render.CAMERA_MOVE_END_TIME, 127):
        position = render.camera_position_at(float(t), centre, distance)
        quat_wxyz = render.look_at_quat(position, centre, render.CAMERA_UP)
        rotation = Rotation.from_quat(np.roll(quat_wxyz, -1)).as_matrix()
        right, up, forward = rotation[:, 0], rotation[:, 1], -rotation[:, 2]
        for spline in splines:
            rays = np.atleast_2d(spline(np.linspace(0.0, 60.0, 512))) - position
            depth = rays @ forward
            assert np.all(depth > 0.0)  # nothing behind the camera, where the ratios go nonsense
            worst = max(worst, float(np.max(np.abs(rays @ right) / (depth * tan_h))))
            worst = max(worst, float(np.max(np.abs(rays @ up) / (depth * tan_v))))
    return worst


@pytest.mark.parametrize(("n_drones", "radius", "z_low", "z_high"), SWARM_SCALES)
def test_camera_keeps_the_whole_swarm_on_screen(
    n_drones: int, radius: float, z_low: float, z_high: float
):
    """Every drone stays inside the frame for the whole move, at lab scale and at sim scale.

    The camera used to orbit a hardcoded 7.75 m tuned for a 4 m box, so a 20 m show flew out of
    frame. The lower bound catches the opposite: the old bounding sphere filled 0.64 of the frame.
    """
    splines = _orbit_splines(n_drones, radius, z_low, z_high)
    centre, points = render.swarm_points(splines, t_end=60.0)
    distance = render.camera_fit_distance(
        _StubSim(), 0, centre, points, render.WIDTH, render.HEIGHT
    )

    fill = _worst_frame_fill(splines, centre, distance)
    assert fill <= 1.0
    assert fill >= 0.8


@pytest.mark.parametrize(("n_drones", "radius", "z_low", "z_high"), SWARM_SCALES)
def test_camera_fit_is_exact_up_to_the_margin(
    n_drones: int, radius: float, z_low: float, z_high: float, monkeypatch: pytest.MonkeyPatch
):
    """Strip the margin and the outermost drone lands exactly on the frame edge.

    At margin 1.0 the fit is exact, so the shipped margin is the only headroom there is.
    """
    splines = _orbit_splines(n_drones, radius, z_low, z_high)
    centre, points = render.swarm_points(splines, t_end=60.0)
    monkeypatch.setattr(render, "CAMERA_FIT_MARGIN", 1.0)
    distance = render.camera_fit_distance(
        _StubSim(), 0, centre, points, render.WIDTH, render.HEIGHT
    )

    assert _worst_frame_fill(splines, centre, distance) == pytest.approx(1.0, rel=1e-3)


def test_camera_fit_reads_the_frame_aspect_not_just_the_fovy():
    """The horizontal half-angle is the fovy widened by the render's own aspect ratio.

    Two frames of the same shape must agree whatever their pixel count, and a square frame --
    narrower in the axis that was binding -- must push the camera back.
    """
    splines = _orbit_splines(*SWARM_SCALES[1])
    centre, points = render.swarm_points(splines, t_end=60.0)

    def fit(width: int, height: int) -> float:
        return render.camera_fit_distance(_StubSim(), 0, centre, points, width, height)

    wide = fit(render.WIDTH, render.HEIGHT)
    assert fit(render.PREVIEW_WIDTH, render.PREVIEW_HEIGHT) == pytest.approx(wide, rel=1e-9)

    square = fit(render.HEIGHT, render.HEIGHT)
    assert square > wide
    assert _worst_frame_fill(splines, centre, square, render.HEIGHT, render.HEIGHT) <= 1.0


def test_camera_scales_with_the_swarm_but_keeps_its_move():
    """Framing tracks swarm size; the shape of the move does not.

    Scale the show by five and the camera moves out by five, but azimuth, elevation and the
    push-in ratio stay as configured -- a camera that merely backed off would throw the move away.
    """
    scale = 5.0
    small = _orbit_splines(20, 2.0, 0.25, 1.7)
    large = _orbit_splines(20, 2.0 * scale, 0.25 * scale, 1.7 * scale)

    frames = []
    for splines in (small, large):
        centre, points = render.swarm_points(splines, t_end=60.0)
        distance = render.camera_fit_distance(
            _StubSim(), 0, centre, points, render.WIDTH, render.HEIGHT
        )
        start = render.camera_position_at(render.CAMERA_MOVE_START_TIME, centre, distance)
        end = render.camera_position_at(render.CAMERA_MOVE_END_TIME, centre, distance)
        frames.append((start - centre, end - centre))

    (small_start, small_end), (large_start, large_end) = frames
    assert np.linalg.norm(large_start) == pytest.approx(
        np.linalg.norm(small_start) * scale, rel=1e-9
    )

    # Same directions, and the same closing ratio, at both scales.
    for small_off, large_off in ((small_start, large_start), (small_end, large_end)):
        cosine = np.dot(small_off, large_off) / (
            np.linalg.norm(small_off) * np.linalg.norm(large_off)
        )
        assert cosine == pytest.approx(1.0, abs=1e-9)
    small_ratio = np.linalg.norm(small_end) / np.linalg.norm(small_start)
    large_ratio = np.linalg.norm(large_end) / np.linalg.norm(large_start)
    assert small_ratio == pytest.approx(large_ratio, rel=1e-9)
    assert small_ratio < 1.0  # the configured move pushes in, and still does


def test_camera_fits_the_swarm_at_its_closest_approach(monkeypatch: pytest.MonkeyPatch):
    """The whole move has to fit, and it is the push-in that binds.

    Fitting the opening frame leaves the tightest part cropped. `samples=1` is exactly that
    mistake, and the margin is stripped so the crop it causes is not absorbed.
    """
    splines = _orbit_splines(*SWARM_SCALES[1])
    centre, points = render.swarm_points(splines, t_end=60.0)
    monkeypatch.setattr(render, "CAMERA_FIT_MARGIN", 1.0)
    distance = render.camera_fit_distance(
        _StubSim(), 0, centre, points, render.WIDTH, render.HEIGHT
    )

    offsets = [
        np.linalg.norm(render.camera_position_at(float(t), centre, distance) - centre)
        for t in np.linspace(render.CAMERA_MOVE_START_TIME, render.CAMERA_MOVE_END_TIME, 31)
    ]
    assert min(offsets) == pytest.approx(distance, rel=1e-9)
    assert max(offsets) > distance  # the move really does travel, so one moment binds and not all

    start_only = render.camera_fit_distance(
        _StubSim(), 0, centre, points, render.WIDTH, render.HEIGHT, samples=1
    )
    assert start_only < distance
    assert _worst_frame_fill(splines, centre, start_only) > 1.0


def _paint_leds(
    monkeypatch: pytest.MonkeyPatch, wrgb: np.ndarray, **kwargs: float
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Paint one frame of fixed WRGB and return the rgba and emission each LED ring was handed.

    The timeline is stubbed to the one call `paint_lighting` makes, keeping the compile path out.
    """
    painted: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def record(
        _sim: object, mat_name: str, drone_ids: np.ndarray, rgba: np.ndarray, emission: np.ndarray
    ) -> None:
        painted[mat_name] = (
            np.asarray(rgba, dtype=float).copy(),
            np.asarray(emission, dtype=float).copy(),
        )

    monkeypatch.setattr(sim_module, "change_material", record)
    wrgb = np.asarray(wrgb, dtype=float)
    stub_sim = type("_Sim", (), {"n_drones": len(wrgb)})()
    timeline = type("_Timeline", (), {"evaluate": lambda _self, _t: wrgb})()
    render.paint_lighting(stub_sim, timeline, 0.0, **kwargs)
    return painted


def test_render_emission_gain_brightens_the_leds_without_moving_their_hue(
    monkeypatch: pytest.MonkeyPatch,
):
    """The render's LED boost is a scalar on emission, so it can neither clip nor wash out.

    The obvious fix -- multiplying the colour and letting channels clip -- drags mixed hues towards
    white. The three drones are a hue with headroom, one with none, and one that must stay dimmer.
    """
    wrgb = np.array(
        [
            [[0.0, 127.5, 127.5, 0.0]] * 2,  # palette yellow, peaking at half the range
            [[0.0, 255.0, 0.0, 0.0]] * 2,  # palette red, already at the top of it
            [[0.0, 76.5, 0.0, 0.0]] * 2,  # the same red, dimmed to 30%
        ]
    )
    plain = _paint_leds(monkeypatch, wrgb)
    boosted = _paint_leds(monkeypatch, wrgb, emission_gain=render.RENDER_EMISSION_GAIN)

    assert render.RENDER_EMISSION_GAIN > 1.0
    for ring in ("led_top", "led_bot"):
        rgba, emission = plain[ring]
        # The default is the viewer's call, and it paints the show's own brightness, unamplified.
        assert np.allclose(emission, 1.0), ring

        boosted_rgba, boosted_emission = boosted[ring]
        # Only the emission moved. The rgba -- colour and alpha both -- is handed over untouched,
        # so what each LED emits stays a scalar multiple of its own colour and the hue is fixed.
        assert np.allclose(boosted_rgba, rgba), ring
        emitted = boosted_emission[:, None] * boosted_rgba[:, :3]
        assert np.all(emitted <= 1.0), ring  # nothing is driven into saturation
        assert np.allclose(emitted[0, 0], emitted[0, 1]), ring  # yellow's channels stay equal

        # Yellow spends its headroom, red has none to spend and is left exactly alone, and the
        # dimmed drone gains without ever catching up to the one at full brightness.
        assert emitted[0].max() > rgba[0, :3].max()
        assert boosted_emission[1] == pytest.approx(1.0)
        assert rgba[2, :3].max() < emitted[2].max() < emitted[1].max()


def _fake_ffmpeg(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    """Capture the ffmpeg command instead of running it, and report success."""
    commands: list[list[str]] = []

    def run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess:
        commands.append(list(command))
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(render.shutil, "which", lambda _name: "/usr/bin/ffmpeg")
    monkeypatch.setattr(render.subprocess, "run", run)
    return commands


def test_mux_audio_seeks_the_song_to_the_crop_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """The render's audio starts at the crop, not at 0:00.

    The choreography's timeline is rebased to 0 while the mp3 is not -- 35 s out of sync for
    `Fearless2`. `-ss` must sit *before* `-i <audio>`, or ffmpeg trims the muxed output instead.
    """
    video = tmp_path / "show.mp4"
    video.write_bytes(b"video")
    audio = tmp_path / "Fearless2.mp3"
    audio.write_bytes(b"audio")
    commands = _fake_ffmpeg(monkeypatch)

    render.mux_audio(video, audio, duration=39.0, audio_start=35.0)

    assert len(commands) == 1
    command = commands[0]
    audio_input = command.index(str(audio))
    assert command[audio_input - 3 : audio_input] == ["-ss", "35.000000", "-i"]
    # The video is already rebased to 0, so only the audio input is seeked.
    video_input = command.index(str(video))
    assert video_input < audio_input
    assert "-ss" not in command[:video_input]


def test_mux_audio_without_a_crop_still_seeks_explicitly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """A song choreographed from 0:00 passes 0.0, not a dropped flag.

    The negative control: a `-ss` emitted only for non-zero starts would pass the test above.
    """
    video = tmp_path / "show.mp4"
    video.write_bytes(b"video")
    audio = tmp_path / "Harness.mp3"
    audio.write_bytes(b"audio")
    commands = _fake_ffmpeg(monkeypatch)

    render.mux_audio(video, audio, duration=12.0, audio_start=0.0)

    command = commands[0]
    audio_input = command.index(str(audio))
    assert command[audio_input - 3 : audio_input] == ["-ss", "0.000000", "-i"]


def test_the_camera_move_never_leaves_the_audience_arc():
    """The renderer previews what the audience will see, so it stays on the audience's side.

    `stage_axis` in lighting.toml puts audience-right at +y. Swing the camera far enough off that
    eyeline and +y turns into depth, so a `left`/`right` look reads as nothing -- the preview then
    disagrees with the room about the one thing it is there to show.
    """
    centre = np.array([0.0, 0.0, 1.1])
    audience_right = np.array([0.0, 1.0, 0.0])
    worst = 1.0
    for t in np.linspace(render.CAMERA_MOVE_START_TIME, render.CAMERA_MOVE_END_TIME, 127):
        position = render.camera_position_at(float(t), centre, 6.0)
        forward = centre - position
        right, _ = render.camera_basis(forward / np.linalg.norm(forward), render.CAMERA_UP)
        worst = min(worst, float(np.dot(right, audience_right)))
    # cos(60 deg): past that the lateral split is compressed more than it is shown.
    assert worst >= 0.5, (
        f"camera swings to {math.degrees(math.acos(worst)):.0f} deg off the eyeline"
    )


def test_the_camera_never_mirrors_the_audience_left_right():
    """A negative dot would put stage right on the viewer's left -- a preview that lies."""
    centre = np.array([0.0, 0.0, 1.1])
    start = render.camera_position_at(render.CAMERA_MOVE_START_TIME, centre, 6.0)
    forward = centre - start
    right, _ = render.camera_basis(forward / np.linalg.norm(forward), render.CAMERA_UP)
    assert np.dot(right, [0.0, 1.0, 0.0]) > 0.0
