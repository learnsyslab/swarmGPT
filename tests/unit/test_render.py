"""Unit tests for the offline renderer's audio muxing.

Only `mux_audio` is reachable from a unit test: `render_preset` needs a full backend, the axswarm
pass and an offscreen MuJoCo context, so the wiring from `backend.crop_window` into this call stays
unpinned. What is pinned here is the half that was actually wrong -- the ffmpeg command itself.
"""

import subprocess
from pathlib import Path

import pytest

import swarm_gpt.render as render


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

    The choreography is planned against the `song_crops` window and its timeline is rebased to 0,
    while the mp3 is not. Without the seek the song plays from the top under a show written for the
    crop -- 35 s out of sync for `Fearless2`, whose window is [35, 70]. The web player has always
    applied the same offset; the render path never learned it.

    `-ss` must sit **before** `-i <audio>`: after it, ffmpeg applies the seek to the output and
    trims the front of the muxed result instead of skipping the song's lead-in.
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

    The negative control: it is the *value* that carries the crop, so a `-ss` emitted only when the
    start is non-zero would pass the test above while leaving the flag's position untested for the
    songs that need it least.
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
