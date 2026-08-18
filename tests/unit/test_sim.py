"""Unit tests for the sim read-out: what `replay_sim_states` actually draws per frame.

The two read-out mistakes -- sampling once instead of per frame, and painting both rings from one
deck -- survive a green timeline test, so these drive the real loop against a fake `Sim`.
"""

import dataclasses
from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest

import swarm_gpt.core.sim as sim_module
from swarm_gpt.core.lighting import LightingTimeline, build_look, load_lighting_config
from swarm_gpt.core.sim import paint_lighting, replay_sim_states

CFG = load_lighting_config()
BPM = 120.0
N4 = 4
POSITIONS_4 = np.stack([np.arange(4.0), np.zeros(4), np.ones(4)], axis=1)
ALL = ("all", ())

# The replay loop is driven off `time.perf_counter`, so the fake clock's tick sets the playback
# times. Two calls per iteration (frame start, then the sleep budget) means a 0.25s tick puts
# frames half a second apart.
CLOCK_TICK = 0.25
FRAME_TIMES = (0.25, 0.75, 1.25, 1.75, 2.0)

# A 2s replay, and a lighting timeline long enough that the blackout never reaches it.
REPLAY_END_S = 2.0
SHOW_END_S = 10.0

RED = np.array([1.0, 0.0, 0.0])
BLACK = np.zeros(3)


@dataclasses.dataclass(frozen=True)
class _FakeNode:
    """A stand-in for the nested crazyflow sim-data structs, which are all `replace`-able."""

    fields: dict

    def __getattr__(self, name: str) -> Any:
        try:
            return self.fields[name]
        except KeyError as e:
            raise AttributeError(name) from e

    def replace(self, **changes: Any) -> "_FakeNode":
        return _FakeNode(self.fields | changes)


class _FakeSim:
    """A `Sim` stand-in: JAX arrays so `set_state`'s `.at[...].set(...)` works, and no window."""

    def __init__(self, n_drones: int) -> None:
        self.n_drones = n_drones
        self.max_visual_geom = 0
        self.closed = False
        zeros = jnp.zeros((1, n_drones, 3))
        self.data = _FakeNode(
            {
                "states": _FakeNode(
                    {
                        "pos": zeros,
                        "quat": jnp.zeros((1, n_drones, 4)),
                        "vel": zeros,
                        "ang_vel": zeros,
                    }
                ),
                "core": _FakeNode({"mjx_synced": jnp.ones((1,), dtype=bool)}),
            }
        )

    def render(self, cam_config: dict | None = None) -> None:
        return None

    def close(self) -> None:
        self.closed = True


class _FakeClock:
    """A monotonic clock advancing one tick per reading, so the replay loop is deterministic."""

    def __init__(self, tick: float) -> None:
        self.tick = tick
        self.now = 0.0

    def perf_counter(self) -> float:
        now = self.now
        self.now += self.tick
        return now

    def sleep(self, seconds: float) -> None:
        return None


class _FakeMusic:
    """`replay_sim_states` asserts a music manager is present and plays it before the loop."""

    def play(self, wait: bool = False) -> bool:
        return True

    def stop(self) -> None:
        return None


def _timeline(actions: list[dict]) -> LightingTimeline:
    """A one-look timeline over the four-drone fixture, the look starting at t = 0."""
    look = build_look(actions, 0.0, POSITIONS_4, N4, CFG, BPM)
    return LightingTimeline([look], N4, SHOW_END_S, CFG)


def _replay(
    monkeypatch: pytest.MonkeyPatch, timeline: LightingTimeline, trails: list | None = None
) -> dict[str, list]:
    """Run the replay loop against fakes and return the rgba each LED material was handed.

    ``trails`` collects every ``draw_line`` rgba in call order, for the trail-colour tests.
    """
    painted: dict[str, list] = {"led_top": [], "led_bot": []}

    def record(_sim: object, mat_name: str, rgba: np.ndarray, **_kwargs: object) -> None:
        painted[mat_name].append(np.asarray(rgba, dtype=float).copy())

    def record_line(_sim: object, _points: np.ndarray, rgba: np.ndarray, **_kwargs: object) -> None:
        if trails is not None:
            trails.append(np.asarray(rgba, dtype=float).copy())

    monkeypatch.setattr(sim_module, "Sim", lambda **kwargs: _FakeSim(kwargs["n_drones"]))
    monkeypatch.setattr(sim_module, "change_material", record)
    monkeypatch.setattr(sim_module, "draw_line", record_line)
    monkeypatch.setattr(sim_module, "time", _FakeClock(CLOCK_TICK))

    timestamps = np.array([0.0, REPLAY_END_S])
    sim_data = {
        "num_drones": N4,
        "timestamps": timestamps,
        "states": np.zeros((2, N4, 13), dtype=np.float32),
    }
    settings = {"sim_freq": 500, "attitude_freq": 500, "state_freq": 50}
    replay_sim_states(sim_data, settings, timeline, _FakeMusic())
    return painted


def _paint(
    monkeypatch: pytest.MonkeyPatch, timeline: LightingTimeline, t: float
) -> tuple[dict[str, np.ndarray], Any]:
    """Call `paint_lighting` once; return what each material got, and what it handed back.

    The return value is passed through untouched: it has to be `None`, and coercing to an array
    would turn that into a NaN array and hide it.
    """
    painted: dict[str, np.ndarray] = {}

    def record(_sim: object, mat_name: str, rgba: np.ndarray, **_kwargs: object) -> None:
        painted[mat_name] = np.asarray(rgba, dtype=float).copy()

    monkeypatch.setattr(sim_module, "change_material", record)
    return painted, paint_lighting(_FakeSim(N4), timeline, t)


def test_paint_lighting_gives_each_ring_its_own_deck(monkeypatch: pytest.MonkeyPatch):
    """The two decks resolve independently, so one shared array cannot express both.

    Both call sites used to share an array that `evaluate_rgb01` fills from the *top* deck, making
    every `deck="bot"` action invisible. Here the top ring is red and the bottom killed outright.
    """
    timeline = _timeline(
        [
            {"primitive": "light_color", "params": {"sel": ALL, "color": "red", "deck": "top"}},
            {"primitive": "light_off", "params": {"sel": ALL, "deck": "bot"}},
        ]
    )

    painted, _ = _paint(monkeypatch, timeline, 0.0)

    assert set(painted) == {"led_top", "led_bot"}
    assert np.allclose(painted["led_top"][:, :3], RED)
    assert np.allclose(painted["led_bot"][:, :3], BLACK)


def test_paint_lighting_reads_the_timeline_at_the_time_it_is_given(monkeypatch: pytest.MonkeyPatch):
    """The helper takes ``t`` and evaluates from it, so a hoisted read-out cannot go unnoticed.

    This does not pin `render.py`'s own frame loop, which could still pass a stale time:
    `render_preset` has no unit-testable seam. That gap is known and accepted.
    """
    timeline = _timeline(
        [
            {"primitive": "light_color", "params": {"sel": ALL, "color": "red", "deck": "both"}},
            {
                "primitive": "blink",
                "params": {"sel": ALL, "period_beats": 4.0, "duty": 0.5, "deck": "both"},
            },
        ]
    )

    lit, _ = _paint(monkeypatch, timeline, 0.25)
    dark, _ = _paint(monkeypatch, timeline, 1.25)

    for mat_name in ("led_top", "led_bot"):
        assert np.allclose(lit[mat_name][:, :3], RED), mat_name
        assert np.allclose(dark[mat_name][:, :3], BLACK), mat_name


def test_paint_lighting_hands_nothing_back(monkeypatch: pytest.MonkeyPatch):
    """It paints the LED materials and returns nothing — the trails are not a read-out of it.

    Pinned rather than left implicit because it has flipped three times; handing a resolved deck
    back would invite a caller to colour something from it again.
    """
    timeline = _timeline(
        [
            {"primitive": "light_color", "params": {"sel": ALL, "color": "red", "deck": "top"}},
            {"primitive": "light_off", "params": {"sel": ALL, "deck": "bot"}},
        ]
    )

    painted, returned = _paint(monkeypatch, timeline, 0.0)

    assert returned is None
    # ...and it did paint, so the `None` is the contract and not a dead call.
    assert np.allclose(painted["led_top"][:, :3], RED)


def test_replay_paints_each_deck_from_its_own_deck_of_the_timeline(monkeypatch: pytest.MonkeyPatch):
    """A `deck="bot"` action must be visible in the preview.

    Both `change_material` calls used to share a top-deck array, so the bottom ring silently
    mirrored it. Here the top is red and the bottom killed -- no shared array expresses that.
    """
    timeline = _timeline(
        [
            {"primitive": "light_color", "params": {"sel": ALL, "color": "red", "deck": "top"}},
            {"primitive": "light_off", "params": {"sel": ALL, "deck": "bot"}},
        ]
    )

    painted = _replay(monkeypatch, timeline)

    assert len(painted["led_top"]) == len(FRAME_TIMES)
    assert len(painted["led_bot"]) == len(FRAME_TIMES)
    for frame in painted["led_top"]:
        assert np.allclose(frame[:, :3], RED)
    for frame in painted["led_bot"]:
        assert np.allclose(frame[:, :3], BLACK)


def test_replay_resamples_the_timeline_on_every_frame(monkeypatch: pytest.MonkeyPatch):
    """The colour is a function of time, so the loop must resample it, not hoist it.

    The five frames straddle two on-phases and one off-phase of a 2s square wave, so sampling once
    before the loop would paint all five identically.
    """
    timeline = _timeline(
        [
            {"primitive": "light_color", "params": {"sel": ALL, "color": "red", "deck": "both"}},
            {
                "primitive": "blink",
                "params": {"sel": ALL, "period_beats": 4.0, "duty": 0.5, "deck": "both"},
            },
        ]
    )

    painted = _replay(monkeypatch, timeline)

    for deck in ("led_top", "led_bot"):
        drawn = [frame[:, :3] for frame in painted[deck]]
        assert len(drawn) == len(FRAME_TIMES)
        # t = 0.25 / 0.75 are in the on-phase, 1.25 / 1.75 in the off-phase, 2.0 back on.
        assert [bool(np.allclose(f, RED)) for f in drawn] == [True, True, False, False, True], deck
        assert all(np.allclose(f, BLACK) for f in drawn[2:4]), deck


def test_replay_trails_are_one_neutral_grey_whatever_the_lighting_does(
    monkeypatch: pytest.MonkeyPatch,
):
    """Every trail is `TRAIL_RGBA`, for every drone, at every frame, ignoring the lighting.

    Coloured trails oversold the effect: one LED changing repainted a whole streak. The fixture
    varies both decks, and the closing assertions pin that -- else the claim is about the fixture.
    """
    timeline = _timeline(
        [
            {"primitive": "light_color", "params": {"sel": ALL, "color": "red", "deck": "both"}},
            {
                "primitive": "blink",
                "params": {"sel": ALL, "period_beats": 4.0, "duty": 0.5, "deck": "both"},
            },
            {"primitive": "light_off", "params": {"sel": ALL, "deck": "bot"}},
        ]
    )
    trails: list[np.ndarray] = []
    # Alpha is the knob for how present the trail is, and the shipped 0.0 draws nothing at all --
    # pinned by the test below. Made visible here, or this test would pass on an empty list.
    visible = np.array([*sim_module.TRAIL_RGBA[:3], 0.6])
    monkeypatch.setattr(sim_module, "TRAIL_RGBA", visible)

    painted = _replay(monkeypatch, timeline, trails)

    # One `draw_line` per drone per frame, in drone order.
    assert len(trails) == len(FRAME_TIMES) * N4
    drawn = np.stack(trails)
    assert np.allclose(drawn, visible), "every trail is the one colour, always"
    assert visible[0] == visible[1] == visible[2], "neutral: no channel may carry colour"

    # The lighting the trails are ignoring has to actually move, or the test above is vacuous.
    lit = [bool(np.allclose(frame[:, :3], RED)) for frame in painted["led_top"]]
    assert lit == [True, True, False, False, True]
    assert all(np.allclose(frame[:, :3], BLACK) for frame in painted["led_bot"])


def test_a_fully_transparent_trail_draws_nothing_at_all(monkeypatch: pytest.MonkeyPatch):
    """Alpha 0 is the shipped setting, and a trail nobody can see must cost nothing to not see.

    Drawing it anyway is a deque append and a `draw_line` per drone per frame for a line that
    renders as empty, which is the same work the browser player skips for the same reason.
    """
    assert sim_module.TRAIL_RGBA[3] == 0.0, "the shipped trail is transparent"
    trails: list[np.ndarray] = []

    _replay(monkeypatch, _timeline([]), trails)

    assert trails == []
