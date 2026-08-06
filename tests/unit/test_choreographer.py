"""Tests for choreographer orchestration helpers (F3) and form_* motion primitives (F1)."""

import logging
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from conftest import virtual_crazyswarm_config

from swarm_gpt.core.choreographer import (
    Choreographer,
    _form_should_drop_holds,
    _overlapping_drone_set,
)
from swarm_gpt.core.lighting import hue_to_wrgb, load_lighting_config
from swarm_gpt.exception import LLMFormatError
from swarm_gpt.utils.music_analyzer import Bar, Beat, Segment, SongStructure


def test_form_should_drop_holds_when_overlapping_motion_follows():
    """form_star (full swarm) + rotate (full swarm) → overlap, drop holds."""
    action_list = [{"form_star": (100, 60, 80, 1.0)}, {"rotate": (45, "z")}]
    assert _form_should_drop_holds(action_list, 0, num_drones=10) is True


def test_form_should_not_drop_holds_when_motion_targets_disjoint_drones():
    """form_circle drones 1-5 + move_z drones 6-10 → no overlap, keep holds."""
    action_list = [{"form_circle": ([1, 2, 3, 4, 5], 150, 1.0)}, {"move_z": ([6, 7, 8, 9, 10], 50)}]
    assert _form_should_drop_holds(action_list, 0, num_drones=10) is False


def test_form_should_not_drop_holds_when_followed_by_another_form():
    """form_star followed only by form_circle → no motion primitive, keep holds."""
    action_list = [{"form_star": (100, 60, 80, 1.0)}, {"form_circle": ([1, 2, 3, 4, 5], 150, 1.0)}]
    assert _form_should_drop_holds(action_list, 0, num_drones=10) is False


def test_form_should_drop_holds_with_three_deep_stack():
    """form_star; rotate; move_z — both later entries are motion, still drops holds."""
    action_list = [
        {"form_star": (100, 60, 80, 1.0)},
        {"rotate": (45, "z")},
        {"move_z": ([1, 2, 3], 30)},
    ]
    assert _form_should_drop_holds(action_list, 0, num_drones=10) is True


def test_overlapping_drone_set_full_swarm():
    """Primitives without drone subset args touch the full swarm."""
    assert _overlapping_drone_set({"form_star": (100, 60, 80, 1.0)}, num_drones=5) == frozenset(
        {0, 1, 2, 3, 4}
    )
    assert _overlapping_drone_set({"rotate": (45, "z")}, num_drones=3) == frozenset({0, 1, 2})


def test_overlapping_drone_set_subset():
    """form_circle / move_z / center return 0-indexed drone IDs from their first arg."""
    assert _overlapping_drone_set(
        {"form_circle": ([1, 3, 5], 150, 1.0)}, num_drones=10
    ) == frozenset({0, 2, 4})
    assert _overlapping_drone_set({"move_z": ([2, 4], 50)}, num_drones=10) == frozenset({1, 3})
    assert _overlapping_drone_set({"center": ([1, 2, 3],)}, num_drones=10) == frozenset({0, 1, 2})


def test_overlapping_drone_set_swap():
    """swap returns exactly the two drone IDs (0-indexed)."""
    assert _overlapping_drone_set({"swap": (3, 7)}, num_drones=10) == frozenset({2, 6})


def test_overlapping_drone_set_move():
    """move returns the single target drone ID (0-indexed, 4th arg)."""
    assert _overlapping_drone_set({"move": (100, 0, 150, 5)}, num_drones=10) == frozenset({4})


def test_schema_allows_multiple_actions_per_entry():
    """After F3 rollback, action_list must not have maxItems: 1."""
    from swarm_gpt.core.structured_output_schema import build_motion_primitive_response_schema

    schema = build_motion_primitive_response_schema(
        all_keys=[(1, 1, 1)], required_keys=[(1, 1, 1)], num_drones=10
    )
    action_list_schema = schema["$defs"]["action_list"]
    assert "maxItems" not in action_list_schema or action_list_schema["maxItems"] > 1
    assert action_list_schema["minItems"] == 1


def test_form_star_hold_pruning_in_pipeline():
    """form_star + rotate: arrival waypoint kept, holds stripped before merge."""
    config_path = virtual_crazyswarm_config(n_drones=10)
    choreographer = Choreographer(
        config_file=config_path, llm_provider="openai", use_motion_primitives=True
    )
    # 10 drones arranged in a line at z=100 cm
    for i in choreographer.starting_pos:
        choreographer.starting_pos[i] = np.array([(i - 5) * 0.3, 0.0, 1.0])

    t = 0.0
    beats = [Beat(id=j + 1, time_s=t + j * 0.5, position_in_bar=j + 1) for j in range(4)]
    bar = Bar(id=1, start_s=0.0, beats=beats)
    seg = Segment(id=1, label="chorus", start_s=0.0, end_s=20.0, bars=[bar])
    structure = SongStructure(
        schema_version=2,
        source_path="test.mp3",
        song_sha256="abc",
        analyzer="test",
        bpm=120,
        segments=[seg],
    )

    # Build a choreography with form_star followed by rotate at the same key
    choreography = {(1, 1, 1): "form_star(100, 60, 80, 1.0); rotate(45, 'z')"}
    waypoints = choreographer._choreo2waypoints(choreography, structure)

    # The position array must have shape (n_drones, T, 3)
    assert waypoints["pos"].shape[0] == 10
    # Must have more than one timestep (rotate emits dense waypoints)
    assert waypoints["pos"].shape[1] > 2


def test_load_drone_config_uses_active_list(tmp_path: Path) -> None:
    """Loader must respect active list order and build uri from addr and channel."""
    cfg = tmp_path / "drones.toml"
    cfg.write_text(
        'active = ["cf41", "cf31"]\n'
        "[cf31]\naddr = 0x1F\nchannel = 30\npos = [0.0, 0.0, 0.0]\n"
        "[cf41]\naddr = 0x29\nchannel = 40\npos = [1.0, 0.0, 0.0]\n"
    )
    import yaml

    c = Choreographer.__new__(Choreographer)
    c.agents = {}
    c.uris = {}
    c.starting_pos = {}
    c.num_drones = 0
    settings_path = Path(__file__).resolve().parents[2] / "swarm_gpt/data/settings.yaml"
    c.settings = yaml.safe_load(settings_path.read_text())
    c.load_drone_config(config_file=cfg)

    # active = ["cf41", "cf31"] → swarm index 0 = cf41, 1 = cf31
    assert c.num_drones == 2
    assert c.uris[0] == "radio://0/40/2M/E7E7E7E729"  # cf41, channel=40, addr=0x29
    assert c.uris[1] == "radio://0/30/2M/E7E7E7E71F"  # cf31, channel=30, addr=0x1F


# --- lighting track -------------------------------------------------------------------------------

LIGHTING_CFG = load_lighting_config()
LIGHTING_N = 6
# Six drones spread along +x at distinct heights, so the spatial selectors and spreads all resolve
# unambiguously against the frozen snapshot.
LIGHTING_POSITIONS = np.stack([np.arange(6.0), np.zeros(6), np.linspace(1.0, 2.0, 6)], axis=1)
# The music ends at 8s; the flight runs 4s longer, because `response2waypoints` appends the
# Return-to-home legs. The blackout belongs at the end of the *flight*.
SONG_END_S = 8.0
FLIGHT_END_S = 12.0
# The strict-mode selector object: every field present, the unused ones ignored.
ALL_SEL = {"kind": "all", "ids": [], "count": 1}


def _lighting_structure(bpm: int = 120) -> SongStructure:
    """Two one-bar segments of four beats, half a second apart: s1b1t1 = 0s, s2b1t1 = 4s."""
    segments = []
    for seq in (1, 2):
        start = (seq - 1) * 4.0
        beats = [Beat(id=j + 1, time_s=start + j * 0.5, position_in_bar=j + 1) for j in range(4)]
        segments.append(
            Segment(
                id=seq,
                label="seg",
                start_s=start,
                end_s=start + 4.0,
                bars=[Bar(id=1, start_s=start, beats=beats)],
            )
        )
    return SongStructure(
        schema_version=2,
        source_path="t.mp3",
        song_sha256="a",
        analyzer="t",
        bpm=bpm,
        segments=segments,
    )


def _lighting_choreographer() -> Choreographer:
    return Choreographer(
        config_file=virtual_crazyswarm_config(n_drones=LIGHTING_N),
        llm_provider="openai",
        use_motion_primitives=True,
    )


def _payload(lighting: list | None) -> dict:
    """A structured payload with one motion key, plus the given lighting track."""
    payload = {
        "song_mood": "steady",
        "choreography_plan": "one spiral",
        "choreography": [
            {
                "key": "s1b1t1",
                "actions": [{"primitive": "spiral", "params": {"steps": 3, "height_cm": 100}}],
            }
        ],
    }
    if lighting is not None:
        payload["lighting"] = lighting
    return payload


def _blue_then_blink() -> list[dict]:
    """Two lighting keys: the swarm blue from s1b1t1, red and blinking from s2b1t1."""
    return [
        {
            "key": "s1b1t1",
            "actions": [
                {
                    "primitive": "light_color",
                    "params": {"sel": ALL_SEL, "color": "blue", "deck": "both"},
                }
            ],
        },
        {
            "key": "s2b1t1",
            "actions": [
                {
                    "primitive": "light_color",
                    "params": {"sel": ALL_SEL, "color": "red", "deck": "both"},
                },
                {
                    "primitive": "blink",
                    "params": {"sel": ALL_SEL, "period_beats": 1, "duty": 0.5, "deck": "both"},
                },
            ],
        },
    ]


def _spatial_lighting() -> list[dict]:
    """Two keys of `sweep`, whose phase spread reads the frozen position snapshot."""
    return [
        {
            "key": key,
            "actions": [
                {
                    "primitive": "sweep",
                    "params": {"sel": ALL_SEL, "period_beats": 4, "axis": "x", "deck": "both"},
                }
            ],
        }
        for key in ("s1b1t1", "s2b1t1")
    ]


def _recording_position_at(calls: list[float]) -> Callable[[float], np.ndarray]:
    """A `position_at` that records every time it is asked for a snapshot."""

    def position_at(t: float) -> np.ndarray:
        calls.append(t)
        return LIGHTING_POSITIONS

    return position_at


def test_structured_payload_to_text_emits_a_lighting_block():
    """The lighting track renders in the same idiom as `choreography:`, ended by END."""
    text = _lighting_choreographer()._structured_payload_to_text(_payload(_blue_then_blink()))

    assert "\nlighting:\n" in text
    assert "\n  s1b1t1: light_color(['all', []], 'blue', 'both')\n" in text
    assert (
        "\n  s2b1t1: light_color(['all', []], 'red', 'both'); "
        "blink(['all', []], 1, 0.5, 'both')\n" in text
    )
    # One END per block, and the lighting block comes last.
    assert text.count("END") == 2
    assert text.rstrip().endswith("END")
    assert text.index("choreography:") < text.index("\nlighting:")


def test_lighting_block_stays_out_of_the_choreography_slice():
    """The two tracks share an address space; slicing must not mix them."""
    choreographer = _lighting_choreographer()
    text = choreographer._structured_payload_to_text(_payload(_blue_then_blink()))

    assert choreographer._response2choreo(text) == {(1, 1, 1): "spiral(3, 100)"}


def test_lighting_slice_ignores_the_word_lighting_inside_a_multi_line_plan():
    """The ``lighting:`` header is anchored to a line of its own, so prose cannot claim it.

    The structured path keeps the plan on one line -- `json.dumps` escapes the newlines -- but a
    free-text response can wrap it. An unanchored header would then match inside the prose and
    slice from there to the *choreography's* END, handing the motion track's actions to the
    lighting parser and dropping the real lighting block entirely.
    """
    text = (
        'song_mood: "steady"\n'
        "choreography_plan: |\n"
        "  The drop lands at s2b1t1.\n"
        "  lighting: hold blue until then, then blink.\n"
        "choreography:\n"
        "  s1b1t1: spiral(3, 100)\n"
        "  END\n"
        "lighting:\n"
        "  s1b1t1: light_color(['all', []], 'blue', 'both')\n"
        "  END"
    )
    choreographer = _lighting_choreographer()

    assert choreographer.lighting_from_text(text) == {
        (1, 1, 1): "light_color(['all', []], 'blue', 'both')"
    }
    assert choreographer._response2choreo(text) == {(1, 1, 1): "spiral(3, 100)"}


def test_response2lighting_puts_each_look_at_its_resolved_time():
    """Each emitted key becomes a look at `structure.time_of` of that address."""
    choreographer = _lighting_choreographer()
    structure = _lighting_structure()
    text = choreographer._structured_payload_to_text(_payload(_blue_then_blink()))

    timeline = choreographer.response2lighting(
        text, structure, _recording_position_at([]), FLIGHT_END_S
    )

    t_switch = structure.time_of(2, 1, 1)
    assert t_switch == 4.0
    # The first look holds right up to the second one's start, which replaces it outright.
    assert np.allclose(
        timeline.evaluate(t_switch - 0.01)[:, 0], np.round(LIGHTING_CFG.palette["blue"])
    )
    # `blink` is on at phase 0, so the second look reads as its colour at its own start time.
    assert np.allclose(timeline.evaluate(t_switch)[:, 0], np.round(LIGHTING_CFG.palette["red"]))
    # ... and off half a beat later: 1 beat is 0.5s at 120 BPM, duty 0.5.
    assert np.allclose(timeline.evaluate(t_switch + 0.3)[:, 0], 0.0)


def test_response2lighting_converts_period_beats_with_the_songs_own_tempo():
    """`period_beats` is beats, so the song's tempo has to reach every effect it builds.

    Every other fixture in this file is 120 BPM, where a hardcoded 120.0 is indistinguishable from
    the forwarded `structure.bpm` — and a tempo that never arrives runs every
    `period_beats` effect in the show at the wrong rate, for the whole show.
    """
    choreographer = _lighting_choreographer()
    text = (
        "lighting:\n"
        "  s1b1t1: light_color(['all', []], 'red', 'both'); blink(['all', []], 1, 0.5, 'both')\n"
        "  END"
    )

    timeline = choreographer.response2lighting(
        text, _lighting_structure(bpm=90), _recording_position_at([]), FLIGHT_END_S
    )

    # One beat is 2/3 s at 90 BPM, so duty 0.5 holds the swarm lit to t = 1/3 and dark to 2/3. At
    # 120 BPM the beat is 0.5 s and both samples read the other way round.
    assert np.allclose(timeline.evaluate(0.3)[:, 0], np.round(LIGHTING_CFG.palette["red"]))
    assert np.allclose(timeline.evaluate(0.55)[:, 0], 0.0)


def test_response2lighting_reads_positions_at_each_looks_start_and_at_no_other_time():
    """The snapshot is frozen at `t_start`, which is what keeps the timeline pure."""
    choreographer = _lighting_choreographer()
    structure = _lighting_structure()
    text = choreographer._structured_payload_to_text(_payload(_spatial_lighting()))
    calls: list[float] = []

    timeline = choreographer.response2lighting(
        text, structure, _recording_position_at(calls), FLIGHT_END_S
    )

    assert calls == [structure.time_of(1, 1, 1), structure.time_of(2, 1, 1)]
    # Evaluating the timeline must never reach for a position again.
    for t in (0.0, 1.7, 4.0, 6.5):
        timeline.evaluate(t)
    assert calls == [structure.time_of(1, 1, 1), structure.time_of(2, 1, 1)]


@pytest.mark.parametrize("lighting", [None, []])
def test_a_payload_without_lighting_yields_a_full_on_timeline(lighting: list | None):
    """An absent key and an empty array both mean 'no lighting', never an error.

    ``None`` here is the payload predating the feature — this repo's own fixtures — and ``[]`` is
    what strict mode forces a model with nothing to say into emitting.
    """
    choreographer = _lighting_choreographer()
    calls: list[float] = []
    text = choreographer._structured_payload_to_text(_payload(lighting))

    timeline = choreographer.response2lighting(
        text, _lighting_structure(), _recording_position_at(calls), FLIGHT_END_S
    )

    assert calls == []
    # Brightness 1.0 everywhere and the base hue wheel, which is today's colouring exactly.
    base = np.round(hue_to_wrgb(np.arange(LIGHTING_N) / LIGHTING_N, LIGHTING_CFG))
    for deck in range(2):
        assert np.allclose(timeline.evaluate(1.0)[:, deck], base)


def test_a_response_with_no_lighting_block_at_all_yields_a_full_on_timeline():
    """A free-text or preset response that never mentions lighting is not an error."""
    text = 'song_mood: "x"\nchoreography:\n  s1b1t1: spiral(3, 100)\n  END'

    timeline = _lighting_choreographer().response2lighting(
        text, _lighting_structure(), _recording_position_at([]), FLIGHT_END_S
    )

    base = np.round(hue_to_wrgb(np.arange(LIGHTING_N) / LIGHTING_N, LIGHTING_CFG))
    assert np.allclose(timeline.evaluate(1.0)[:, 0], base)


def test_response2lighting_parses_a_hand_written_lighting_block():
    """The free-text path emits the same idiom, so one text parser serves both modes."""
    text = "lighting:\n  s2b1t1: light_color(['ids', [1, 3]], 'green', 'top')\n  END"

    timeline = _lighting_choreographer().response2lighting(
        text, _lighting_structure(), _recording_position_at([]), FLIGHT_END_S
    )

    top = timeline.evaluate(4.0)[:, 0]
    green = np.round(LIGHTING_CFG.palette["green"])
    assert np.allclose(top[[0, 2]], green)
    # `ids` is 1-indexed, and the deck stacks resolve independently.
    assert not np.allclose(top[1], green)
    assert np.allclose(
        timeline.evaluate(4.0)[:, 1],
        np.round(hue_to_wrgb(np.arange(LIGHTING_N) / LIGHTING_N, LIGHTING_CFG)),
    )


def test_response2lighting_reports_an_unknown_primitive_as_a_format_error():
    """A malformed lighting emission is reported the way a malformed motion one is."""
    text = "lighting:\n  s1b1t1: disco_ball(['all', []], 'both')\n  END"

    with pytest.raises(LLMFormatError, match="Unknown lighting primitive 'disco_ball' at s1b1t1"):
        _lighting_choreographer().response2lighting(
            text, _lighting_structure(), _recording_position_at([]), FLIGHT_END_S
        )


def test_response2lighting_reports_an_unknown_selector_as_a_format_error():
    """`build_look` raises a bare KeyError for these; the choreographer must name the key."""
    text = "lighting:\n  s1b1t1: light_on(['nobody', []], 'both')\n  END"

    with pytest.raises(LLMFormatError, match="s1b1t1"):
        _lighting_choreographer().response2lighting(
            text, _lighting_structure(), _recording_position_at([]), FLIGHT_END_S
        )


def test_response2lighting_reports_a_wrong_argument_count_as_a_format_error():
    """Zipping args onto names would silently drop a missing one, so arity is checked first."""
    text = "lighting:\n  s1b1t1: pulse(['all', []], 'both')\n  END"

    with pytest.raises(LLMFormatError, match="pulse at s1b1t1 must have 3 arguments"):
        _lighting_choreographer().response2lighting(
            text, _lighting_structure(), _recording_position_at([]), FLIGHT_END_S
        )


def test_response2lighting_blacks_out_at_the_end_of_the_flight_not_the_music():
    """The blackout is `t_end - 0.1` where `t_end` is the flight, as at `backend.py:332`.

    Deriving it from the song structure instead would darken the swarm for the whole
    return-to-home leg — the drones would fly home and land dark rather than lit.
    """
    choreographer = _lighting_choreographer()
    text = choreographer._structured_payload_to_text(_payload(_blue_then_blink()))

    timeline = choreographer.response2lighting(
        text, _lighting_structure(), _recording_position_at([]), FLIGHT_END_S
    )

    # Still lit through the return-to-home legs, which start after the music ends.
    assert np.any(timeline.evaluate(SONG_END_S + 1.0) > 0.0)
    assert np.allclose(timeline.evaluate(FLIGHT_END_S - 0.05), 0.0)


def test_validate_lighting_rejects_a_malformed_emission_without_positions():
    """A malformed lighting track must reprompt, and cannot wait for the axswarm pass.

    Positions do not exist until `Backend.simulate` has run, so the full compile cannot happen at
    generation time — but the vocabulary and arity can be checked, and that is what catches a
    hallucinated primitive early enough for `self_correct` to act on it.
    """
    choreographer = _lighting_choreographer()

    with pytest.raises(LLMFormatError, match="Unknown lighting primitive 'disco_ball'"):
        choreographer.validate_lighting(
            "lighting:\n  s1b1t1: disco_ball(['all', []], 'both')\n  END"
        )
    with pytest.raises(LLMFormatError, match="must have 3 arguments"):
        choreographer.validate_lighting("lighting:\n  s1b1t1: pulse(['all', []], 'both')\n  END")


@pytest.mark.parametrize(
    ("emission", "match"),
    [
        ("light_color(['all', []], 'chartreuse', 'both')", "chartreuse"),
        ("gradient(['all', []], 'red', 'chartreuse', 'z', 'both')", "chartreuse"),
        ("light_on(['nobody', []], 'both')", "nobody"),
        ("light_on(['all', []], 'middle')", "middle"),
        ("rainbow(['all', []], 4, 'sideways', 'both')", "sideways"),
        ("gradient(['all', []], 'red', 'blue', 'sideways', 'both')", "sideways"),
        ("alternate_blink(['all', []], 2, 'diagonal', 'both')", "diagonal"),
        # The bounds are not the schema's alone: presets and hand-written blocks skip it entirely.
        ("light_on(['ids', [0]], 'both')", r"1\.\.6"),
        ("light_on(['ids', [99]], 'both')", r"1\.\.6"),
        ("light_on(['first', [99]], 'both')", r"1\.\.6"),
    ],
)
def test_validate_lighting_rejects_every_name_the_engine_would_reject(emission: str, match: str):
    """Names, not just primitives: none of these checks needs a position, so all belong here.

    A bad colour, deck, selector kind, spread or gradient axis used to escape this gate and raise
    a bare `KeyError` much later — inside `compile_cues` during `deploy`, or per frame mid-render,
    where nothing reprompts. The colour was the worst of them: the engine resolves palette names
    lazily at read-out, so it did not even fail when the timeline was compiled.
    """
    choreographer = _lighting_choreographer()

    with pytest.raises(LLMFormatError, match=match):
        choreographer.validate_lighting(f"lighting:\n  s1b1t1: {emission}\n  END")


def test_validate_lighting_does_not_warn_about_its_own_dry_run_snapshot(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
):
    """The dry run discards every position-dependent result, so it must not diagnose one.

    Positions do not exist at generation time, so the looks are built against a synthetic
    snapshot. A snapshot with no extent along an axis is exactly what the collapse warnings
    report, so a snapshot of zeros makes every `sweep`, `ripple_light` and `left`/`right` emission
    warn on every generation — noise about the fixture rather than about the show, which trains
    the reader to ignore the one case that is real.
    """
    name = "swarm_gpt.core.lighting"
    monkeypatch.setattr(logging.getLogger(name), "propagate", True)
    caplog.set_level(logging.WARNING, logger=name)
    emission = (
        "sweep(['all', []], 4, 'z', 'both'); ripple_light(['all', []], 4, 'both'); "
        "light_color(['right', []], 'red', 'both')"
    )

    _lighting_choreographer().validate_lighting(f"lighting:\n  s1b1t1: {emission}\n  END")

    assert not [r for r in caplog.records if r.name.startswith("swarm_gpt.core.lighting")]


def test_validate_lighting_accepts_a_good_track_and_a_missing_one():
    """Lighting is optional, so a response without a block must pass validation untouched."""
    choreographer = _lighting_choreographer()

    choreographer.validate_lighting(choreographer._structured_payload_to_text(_payload(None)))
    choreographer.validate_lighting(
        choreographer._structured_payload_to_text(_payload(_blue_then_blink()))
    )
