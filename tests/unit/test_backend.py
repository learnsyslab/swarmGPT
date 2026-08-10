import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest
from conftest import virtual_crazyswarm_config

from swarm_gpt.core.backend import AppBackend, _fold_cues_to_rgb
from swarm_gpt.core.lighting import compile_cues, hue_to_wrgb, load_lighting_config
from swarm_gpt.exception import LLMFormatError
from swarm_gpt.utils import generate_default_colors
from swarm_gpt.utils.music_analyzer import Bar, Beat, Segment, SongStructure


def test_backend_init():
    config_path = virtual_crazyswarm_config(n_drones=4)
    app = AppBackend(config_file=config_path)
    assert app.choreographer.num_drones == 4
    assert app.choreographer.messages == []


def test_songs():
    config_path = virtual_crazyswarm_config(n_drones=4)
    app = AppBackend(config_file=config_path)
    assert isinstance(app.songs, list)
    available_songs = [s.stem for s in app.music_manager.music_dir.glob("*.mp3")]
    for song in app.songs:
        assert isinstance(song, str), f"Song {song} is not a string"
        assert song in available_songs, f"Song {song} is not in the available songs"


def test_presets():
    config_path = virtual_crazyswarm_config(n_drones=4)
    app = AppBackend(config_file=config_path)
    assert isinstance(app.presets, list)
    for preset in app.presets:
        assert isinstance(preset, str), f"Preset {preset} is not a string"


def test_preset_metadata_and_delete(tmp_path: Path):
    config_path = virtual_crazyswarm_config(n_drones=4)
    preset_dir = tmp_path / "presets"
    preset_id = "Example Song | 4 | 20260521_123456"
    (preset_dir / preset_id).mkdir(parents=True)

    app = AppBackend(config_file=config_path, preset_dir=preset_dir)
    metadata = app.preset_metadata(preset_id)

    assert metadata["song"] == "Example Song"
    assert metadata["numDrones"] == 4
    assert metadata["createdAt"] == "2026-05-21T12:34:56"
    assert metadata["createdLabel"] == "2026-05-21 12:34"

    app.delete_preset(preset_id)
    assert not (preset_dir / preset_id).exists()


def test_emergency_stop_active_swarm_stops_live_swarm_and_music() -> None:
    class ActiveSwarm:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def emergency_stop(self) -> None:
            self.calls.append("emergency_stop")

    config_path = virtual_crazyswarm_config(n_drones=1)
    app = AppBackend(config_file=config_path)
    swarm = ActiveSwarm()
    music_calls: list[str] = []
    app._active_swarm = swarm
    app.music_manager.stop = lambda: music_calls.append("stop")

    app.emergency_stop_active_swarm()

    assert swarm.calls == ["emergency_stop"]
    assert music_calls == ["stop"]


LIGHTING_CFG = load_lighting_config()
LIGHTING_N = 4
BPM = 120
# The flight ends 4s after the music: `response2waypoints` appends the return-to-home legs.
SONG_END_S = 16.0
FLIGHT_END_S = SONG_END_S + 4.0

# A look at s1b1t1 (t = 0s): the swarm blue, blinking once a beat.
LIGHTING_RESPONSE = (
    'song_mood: "x"\n'
    "choreography:\n"
    "  s1b1t1: spiral(3, 100)\n"
    "  END\n"
    "lighting:\n"
    "  s1b1t1: light_color(['all', []], 'blue', 'both'); blink(['all', []], 1, 0.5, 'both')\n"
    "  END"
)

# A look that is **deck-asymmetric and spatially varying**: the top deck runs red -> blue along x
# and blinks, the bottom holds a steady green -> amber. Both properties are load-bearing wherever a
# test claims the payload is not permuted. `LIGHTING_RESPONSE` is uniform across drones and
# identical across decks, so against it a reversed drone order and a top/bot swap are both
# byte-identical no-ops -- the assertion passes without constraining anything. The fixture's
# splines put drone i at x = i (`_deploy_backend`), which is what makes `by="x"` vary.
DECK_ASYMMETRIC_RESPONSE = (
    'song_mood: "x"\n'
    "choreography:\n"
    "  s1b1t1: spiral(3, 100)\n"
    "  END\n"
    "lighting:\n"
    "  s1b1t1: gradient(['all', []], 'red', 'blue', 'x', 'top'); "
    "gradient(['all', []], 'green', 'amber', 'x', 'bot'); "
    "blink(['all', []], 1, 0.5, 'top')\n"
    "  END"
)


def _lighting_structure() -> SongStructure:
    """One segment of eight bars at 120 BPM, so s1b1t1 is t = 0s and the song ends at 16s."""
    bars = [
        Bar(
            id=bar + 1,
            start_s=bar * 2.0,
            beats=[
                Beat(id=j + 1, time_s=bar * 2.0 + j * 0.5, position_in_bar=j + 1) for j in range(4)
            ],
        )
        for bar in range(8)
    ]
    segment = Segment(id=1, label="seg", start_s=0.0, end_s=SONG_END_S, bars=bars)
    return SongStructure(
        schema_version=2,
        source_path="t.mp3",
        song_sha256="a",
        analyzer="t",
        bpm=BPM,
        segments=[segment],
    )


class _FakeSwarm:
    """A `DroneSwarm` stand-in that records what `deploy` hands it."""

    instances: list["_FakeSwarm"] = []

    def __init__(self, drones: dict, **kwargs: object) -> None:
        self.drones = drones
        self.kwargs = kwargs
        self.executed: dict | None = None
        _FakeSwarm.instances.append(self)

    def get_obs(self, uri: str) -> dict:
        pos = np.array(next(d["pos"] for d in self.drones.values() if d["uri"] == uri), float)
        # Within the 0.3m pre-flight tolerance of the configured position, and high enough that
        # the post-takeoff check counts the drone as airborne.
        return {"pos": pos + np.array([0.0, 0.0, 0.25]), "quat": np.array([0.0, 0.0, 0.0, 1.0])}

    def is_active(self, uri: str) -> bool:
        return True

    def goto(self, positions: dict, duration: float | None = None) -> None:
        return None

    def execute_choreography(
        self, choreography: dict, t_end: float, color_top: dict, color_bot: dict
    ) -> None:
        self.executed = {"t_end": t_end, "color_top": color_top, "color_bot": color_bot}

    def land(self, duration: float | None = None) -> None:
        return None

    def close(self) -> None:
        return None


def _deploy_backend(monkeypatch: pytest.MonkeyPatch, response: str) -> AppBackend:
    """A backend with the simulation already run, wired to a fake swarm and a fake song."""
    app = AppBackend(config_file=virtual_crazyswarm_config(n_drones=LIGHTING_N))
    app.settings["lighthouse"] = True  # Skips the rclpy import branch.
    app.choreographer.messages = [{"role": "assistant", "content": response}]
    app.waypoints = {"time": np.tile([0.0, FLIGHT_END_S], (LIGHTING_N, 1))}
    app.splines = {i: (lambda t, i=i: np.array([float(i), 0.0, 1.0])) for i in range(LIGHTING_N)}
    monkeypatch.setattr(app, "_load_structure", lambda _song: _lighting_structure())
    monkeypatch.setattr(app.music_manager, "verify_libvlc", lambda: True)
    monkeypatch.setattr(app.music_manager, "play", lambda **_kwargs: True)
    monkeypatch.setattr(app.music_manager, "stop", lambda: None)
    monkeypatch.setattr("swarm_gpt.core.drone_swarm.DroneSwarm", _FakeSwarm)
    _FakeSwarm.instances.clear()
    return app


def test_deploy_builds_colour_cues_from_the_compiled_timeline(monkeypatch: pytest.MonkeyPatch):
    """Deploy's colour dicts come from `compile_cues`, not from a two-cue stub."""
    app = _deploy_backend(monkeypatch, LIGHTING_RESPONSE)

    assert app.deploy() is True

    swarm = _FakeSwarm.instances[-1]
    uris = {d["uri"] for d in app.choreographer.drones.values()}
    assert swarm.executed["t_end"] == FLIGHT_END_S
    for deck in ("color_top", "color_bot"):
        cues = swarm.executed[deck]
        assert set(cues) == uris
        for track in cues.values():
            times = sorted(track)
            # The stub emitted exactly two cues per deck; a blinking look cannot compile to that.
            assert len(times) > 2
            # Never denser than the consumer drains.
            assert min(np.diff(times)) >= 1.0 / LIGHTING_CFG.col_freq - 1e-9
            assert all(np.all(v >= 0) and np.all(v <= 255) for v in track.values())


def test_deploy_passes_one_col_freq_to_the_swarm_and_the_compiler(monkeypatch: pytest.MonkeyPatch):
    """The cue consumer and the cue compiler read the same config field, never a literal.

    `col_freq` is patched off the shipped 10 Hz because a hardcoded 10.0 is otherwise
    indistinguishable from `cfg.col_freq`, and the two diverging is permanent desync.
    """
    cfg = dataclasses.replace(LIGHTING_CFG, col_freq=4.0)
    monkeypatch.setattr("swarm_gpt.core.backend.load_lighting_config", lambda: cfg)
    app = _deploy_backend(monkeypatch, LIGHTING_RESPONSE)

    app.deploy()

    swarm = _FakeSwarm.instances[-1]
    assert swarm.kwargs["col_freq"] == cfg.col_freq
    for deck in ("color_top", "color_bot"):
        for track in swarm.executed[deck].values():
            times = sorted(track)
            assert len(times) > 2
            # The compiler's sample grid follows the same field: never denser than the consumer
            # drains, and every cue but the terminal blackout lands on a tick of that grid.
            assert min(np.diff(times)) >= 1.0 / cfg.col_freq - 1e-9
            ticks = [t * cfg.col_freq for t in times[:-1]]
            assert ticks == pytest.approx([round(tick) for tick in ticks])


def test_deploy_ends_every_drone_and_deck_black(monkeypatch: pytest.MonkeyPatch):
    """The terminal blackout is unconditional, so the drones never land lit."""
    app = _deploy_backend(monkeypatch, LIGHTING_RESPONSE)

    app.deploy()

    swarm = _FakeSwarm.instances[-1]
    for deck in ("color_top", "color_bot"):
        for track in swarm.executed[deck].values():
            last = max(track)
            # Exactly at the end of the flight, not the end of the music: the drones stay lit
            # through the return-to-home legs, as they do today (`backend.py:332`).
            assert last == pytest.approx(FLIGHT_END_S - 0.1)
            assert np.allclose(track[last], 0.0)


def test_deploy_without_a_lighting_track_keeps_todays_static_colours(
    monkeypatch: pytest.MonkeyPatch,
):
    """A preset predating the feature compiles to one colour, then black."""
    response = 'song_mood: "x"\nchoreography:\n  s1b1t1: spiral(3, 100)\n  END'
    app = _deploy_backend(monkeypatch, response)

    app.deploy()

    swarm = _FakeSwarm.instances[-1]
    base = np.round(hue_to_wrgb(np.arange(LIGHTING_N) / LIGHTING_N, LIGHTING_CFG))
    for deck in ("color_top", "color_bot"):
        for i, uri in enumerate(d["uri"] for d in app.choreographer.drones.values()):
            track = swarm.executed[deck][uri]
            times = sorted(track)
            assert len(times) == 2
            assert np.allclose(track[times[0]], base[i])
            assert np.allclose(track[times[1]], 0.0)


def test_the_position_snapshot_is_ordered_by_drone_index(monkeypatch: pytest.MonkeyPatch):
    """The snapshot's row order *is* the drone index, and this is the seam it crosses.

    Nothing position-free notices a wrong order, so this needs a position-dependent primitive:
    `left`, `sweep` and friends would otherwise address the mirror image of the swarm.
    """
    response = (
        'song_mood: "x"\n'
        "choreography:\n  s1b1t1: spiral(3, 100)\n  END\n"
        "lighting:\n  s1b1t1: gradient(['all', []], 'red', 'blue', 'x', 'both')\n  END"
    )
    app = _deploy_backend(monkeypatch, response)

    top = app.lighting_timeline().evaluate(0.0)[:, 0]

    # The fixture's splines put drone i at x = i, so the gradient runs from color_a on drone 0 to
    # color_b on the last one. Reversed, the two ends swap.
    assert np.allclose(top[0], np.round(LIGHTING_CFG.palette["red"]))
    assert np.allclose(top[-1], np.round(LIGHTING_CFG.palette["blue"]))


def test_lighting_compiles_from_the_latest_response_in_the_history(monkeypatch: pytest.MonkeyPatch):
    """After a self-correct round the history holds several responses, not one.

    The lights must come from the last, or the show flies a superseded response's lighting.
    """
    superseded = (
        'song_mood: "x"\n'
        "choreography:\n  s1b1t1: spiral(3, 100)\n  END\n"
        "lighting:\n  s1b1t1: light_color(['all', []], 'green', 'both')\n  END"
    )
    app = _deploy_backend(monkeypatch, LIGHTING_RESPONSE)
    app.choreographer.messages = [
        {"role": "user", "content": "choreograph this"},
        {"role": "assistant", "content": superseded},
        {"role": "user", "content": "the lighting was wrong, try again"},
        {"role": "assistant", "content": LIGHTING_RESPONSE},
    ]

    top = app.lighting_timeline().evaluate(0.0)[:, 0]

    assert np.allclose(top, np.round(LIGHTING_CFG.palette["blue"]))


def test_lighting_refuses_a_history_that_does_not_end_in_a_response(
    monkeypatch: pytest.MonkeyPatch,
):
    """A history ending on a prompt has no response to compile, and must say so rather than
    silently reading the prompt text — which parses as a lighting-less response and compiles to
    the default hue wheel, a plausible-looking show built from the wrong message."""
    app = _deploy_backend(monkeypatch, LIGHTING_RESPONSE)
    app.choreographer.messages.append({"role": "user", "content": "make it bluer"})

    with pytest.raises(AssertionError, match="not a response"):
        app.lighting_timeline()


def test_reprompt_rejects_a_malformed_lighting_emission(monkeypatch: pytest.MonkeyPatch):
    """A reprompt's own lighting track has to be inside the retry loop, not just the first one.

    Otherwise a malformed emission produced while correcting something else surfaces only at
    compile time -- after the axswarm pass, past every retry, with the show about to deploy.
    """
    app = AppBackend(config_file=virtual_crazyswarm_config(n_drones=LIGHTING_N))
    monkeypatch.setattr(app, "_load_structure", lambda _song: _lighting_structure())
    monkeypatch.setattr(
        app.choreographer,
        "response2waypoints",
        lambda *_args, **_kwargs: {"time": np.tile([0.0, FLIGHT_END_S], (LIGHTING_N, 1))},
    )
    bad = (
        'song_mood: "x"\nchoreography:\n  s1b1t1: spiral(3, 100)\n  END\n'
        "lighting:\n  s1b1t1: disco_ball(['all', []], 'both')\n  END"
    )
    attempts = []

    def generate(prompt: list[dict[str, str]], **_kwargs: object) -> str:
        attempts.append(prompt)
        return bad

    monkeypatch.setattr(app.choreographer, "generate_choreography", generate)

    with pytest.raises(LLMFormatError, match="disco_ball"):
        app.reprompt("make it bluer")

    assert len(attempts) > 1, "and the rejection must go back round the self-correct loop"


def test_sim_colours_change_over_a_blink(monkeypatch: pytest.MonkeyPatch):
    """The colour `render.py` and `sim.py` draw is a function of time, sampled per frame.

    A render path keeping its one-shot ``rgbas[:, :3]`` assignment draws these two identically.
    """
    app = _deploy_backend(monkeypatch, LIGHTING_RESPONSE)

    timeline = app.lighting_timeline()

    # One beat is ~0.55s at 120 BPM, duty 0.5, so the swarm is lit on the beat and dark after it.
    lit = timeline.evaluate_rgb01(0.0)
    dark = timeline.evaluate_rgb01(0.4)
    assert not np.allclose(lit, dark)
    assert np.any(lit > 0.0)
    assert np.allclose(dark, 0.0)


def test_sim_colours_without_a_lighting_track_are_the_base_hue_wheel(
    monkeypatch: pytest.MonkeyPatch,
):
    """No lighting is full on, each drone in its own hue — today's colouring, calibrated.

    The one visible change for presets predating the feature: the sim now carries the same
    ``channel_gain`` blue dim the deploy path always has, so the preview matches what flies.
    """
    response = 'song_mood: "x"\nchoreography:\n  s1b1t1: spiral(3, 100)\n  END'
    app = _deploy_backend(monkeypatch, response)

    timeline = app.lighting_timeline()

    base = hue_to_wrgb(np.arange(LIGHTING_N) / LIGHTING_N, LIGHTING_CFG)
    expected = np.round(base)[:, 1:] / 255.0
    for t in (0.0, 0.4, 7.5):
        assert np.allclose(timeline.evaluate_rgb01(t), expected)
    # Same hue wheel as today, and dimmer only in blue -- that difference is the calibration.
    uncalibrated = generate_default_colors(LIGHTING_N, limit=1.0)
    assert not np.allclose(timeline.evaluate_rgb01(0.0), uncalibrated)
    assert np.allclose(timeline.evaluate_rgb01(0.0)[:, :2], uncalibrated[:, :2], atol=2e-3)


def test_browser_cues_are_drone_indexed_and_json_ready(monkeypatch: pytest.MonkeyPatch):
    """`compile_cues` output is not browser-ready, and this is the whole of the adaptation.

    URI keys become drone indices, `{time: NDArray}` dicts become parallel JSON-safe lists, and
    4-channel WRGB becomes 3-channel RGB.
    """
    app = _deploy_backend(monkeypatch, LIGHTING_RESPONSE)

    cues = app.browser_cues()

    assert set(cues) == {"top", "bot"}
    for deck in ("top", "bot"):
        assert len(cues[deck]) == LIGHTING_N
        for entry in cues[deck]:
            assert set(entry) == {"times", "rgb"}
            assert len(entry["times"]) == len(entry["rgb"])
            # "Initial colour from the first cue" is only defined if every list opens at 0.
            assert entry["times"][0] == 0.0
            assert all(b > a for a, b in zip(entry["times"], entry["times"][1:], strict=False))
            # `type(...) is float`, not `isinstance`: `np.float64` subclasses `float`, so an
            # isinstance check passes on exactly the NumPy scalar this is meant to exclude --
            # and `json.dumps` accepts it too, so the round-trip below does not catch it either.
            assert all(type(t) is float for t in entry["times"])
            for rgb in entry["rgb"]:
                assert len(rgb) == 3
                assert all(isinstance(channel, int) for channel in rgb)
                assert all(0 <= channel <= 255 for channel in rgb)

    # Serializable at all -- an NDArray raises here -- and carrying no radio address, which has no
    # business reaching a browser. `np.float64` would survive this step; the `type` check above is
    # what excludes it.
    payload = json.dumps(cues)
    assert all(d["uri"] not in payload for d in app.choreographer.drones.values())


def test_browser_cues_are_the_same_baked_list_the_hardware_gets(monkeypatch: pytest.MonkeyPatch):
    """Browser == hardware, so the preview shows the `col_freq` quantization that will fly.

    Also pins index-vs-URI and deck keying, which need `DECK_ASYMMETRIC_RESPONSE`: under a uniform
    look a reversed order or a deck swap is byte-identical, and so is comparing only `times`.
    """
    app = _deploy_backend(monkeypatch, DECK_ASYMMETRIC_RESPONSE)
    uris = [d["uri"] for d in app.choreographer.drones.values()]

    browser = app.browser_cues()
    hardware = dict(
        zip(
            ("top", "bot"),
            compile_cues(app.lighting_timeline(), uris, LIGHTING_CFG.col_freq, FLIGHT_END_S),
            strict=True,
        )
    )

    for deck, cues in hardware.items():
        for i, uri in enumerate(uris):
            times = sorted(cues[uri])
            # The W fold recomputed from the hardware cues rather than run back through
            # `_fold_cues_to_rgb`, so this compares two independent derivations.
            folded = [
                np.clip(cues[uri][t][1:] + cues[uri][t][0], 0, 255).astype(int).tolist()
                for t in times
            ]
            assert browser[deck][i]["times"] == times
            assert browser[deck][i]["rgb"] == folded
            # The terminal blackout is the last cue, and the browser's zero-order-hold
            # lookup has to be able to reach it.
            assert browser[deck][i]["times"][-1] == pytest.approx(FLIGHT_END_S - 0.1)
            assert browser[deck][i]["rgb"][-1] == [0, 0, 0]

    # The permutation and deck claims above are vacuous unless the fixture actually distinguishes
    # the things being keyed, so pin that here rather than trusting the response string.
    for deck in ("top", "bot"):
        assert len({tuple(entry["rgb"][0]) for entry in browser[deck]}) == LIGHTING_N, deck
    assert [e["rgb"][0] for e in browser["top"]] != [e["rgb"][0] for e in browser["bot"]]


def test_browser_cues_fold_the_white_channel_into_rgb(monkeypatch: pytest.MonkeyPatch):
    """Three.js has no white channel, so W folds into all three, as `evaluate_rgb01` does."""
    response = (
        'song_mood: "x"\n'
        "choreography:\n  s1b1t1: spiral(3, 100)\n  END\n"
        "lighting:\n  s1b1t1: light_color(['all', []], 'white', 'both')\n  END"
    )
    app = _deploy_backend(monkeypatch, response)

    cues = app.browser_cues()

    # `white` is WRGB (255, 0, 0, 0) -- the dedicated white LED. Dropping W renders it black.
    for deck in ("top", "bot"):
        for entry in cues[deck]:
            assert entry["rgb"][0] == [255, 255, 255]


def test_the_white_fold_clips_rather_than_overflowing():
    """The fold is `clip(rgb + w, 0, 255)`, and the clip is not optional.

    Asserted on the fold directly because the shipped palette sums to 255 and cannot reach it. A
    retuned `lighting.toml` can, and an unclipped fold hands three.js a channel above 1.0.
    """
    folded = _fold_cues_to_rgb({0.0: np.array([200.0, 100.0, 0.0, 60.0])})

    assert folded == {"times": [0.0], "rgb": [[255, 200, 255]]}


def test_initial_prompt_rejects_a_malformed_lighting_emission(monkeypatch: pytest.MonkeyPatch):
    """A malformed lighting track must fail at generation time, where `self_correct` can reprompt.

    Otherwise it surfaces only at compile time, long past the retry loop.
    """
    app = AppBackend(config_file=virtual_crazyswarm_config(n_drones=LIGHTING_N))
    monkeypatch.setattr(app, "_load_structure", lambda _song: _lighting_structure())
    monkeypatch.setattr(
        app.choreographer,
        "response2waypoints",
        lambda *_args, **_kwargs: {"time": np.tile([0.0, FLIGHT_END_S], (LIGHTING_N, 1))},
    )
    bad = (
        'song_mood: "x"\nchoreography:\n  s1b1t1: spiral(3, 100)\n  END\n'
        "lighting:\n  s1b1t1: disco_ball(['all', []], 'both')\n  END"
    )

    with pytest.raises(RuntimeError, match="Initial prompt failed") as excinfo:
        app.initial_prompt("Fearless2", response=bad)

    assert isinstance(excinfo.value.__cause__, LLMFormatError)
    assert "disco_ball" in str(excinfo.value.__cause__)


def test_initial_prompt_accepts_a_well_formed_lighting_emission(monkeypatch: pytest.MonkeyPatch):
    """The positive control: a valid track passes the same gate untouched."""
    app = AppBackend(config_file=virtual_crazyswarm_config(n_drones=LIGHTING_N))
    monkeypatch.setattr(app, "_load_structure", lambda _song: _lighting_structure())
    monkeypatch.setattr(
        app.choreographer,
        "response2waypoints",
        lambda *_args, **_kwargs: {"time": np.tile([0.0, FLIGHT_END_S], (LIGHTING_N, 1))},
    )

    app.initial_prompt("Fearless2", response=LIGHTING_RESPONSE)

    assert app.choreographer.messages[-1]["content"] == LIGHTING_RESPONSE


def test_initial_prompt_hands_back_the_prompt_before_the_model_answers(
    monkeypatch: pytest.MonkeyPatch,
):
    """The UI has nothing to show for the whole of a reasoning model's think unless the prompt
    is surfaced up front, so `on_prompt` has to fire ahead of the call, not alongside its result.
    """
    app = AppBackend(config_file=virtual_crazyswarm_config(n_drones=LIGHTING_N))
    monkeypatch.setattr(app, "_load_structure", lambda _song: _lighting_structure())
    monkeypatch.setattr(
        app.choreographer,
        "response2waypoints",
        lambda *_args, **_kwargs: {"time": np.tile([0.0, FLIGHT_END_S], (LIGHTING_N, 1))},
    )
    order: list[str] = []
    seen: list[list[dict[str, str]]] = []

    def generate(_prompt: list[dict[str, str]], **_kwargs: object) -> str:
        order.append("llm")
        return LIGHTING_RESPONSE

    def on_prompt(prompt: list[dict[str, str]]) -> None:
        order.append("prompt")
        seen.append(prompt)

    monkeypatch.setattr(app.choreographer, "generate_choreography", generate)

    app.initial_prompt("Fearless2", on_prompt=on_prompt)

    assert order == ["prompt", "llm"]
    assert any(message["role"] == "user" for message in seen[0])
    assert any("Fearless2" in message["content"] for message in seen[0])
