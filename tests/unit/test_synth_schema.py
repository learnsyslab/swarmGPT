import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest
from conftest import virtual_crazyswarm_config

from swarm_gpt.core import motion_primitives as mp
from swarm_gpt.core.choreographer import Choreographer
from swarm_gpt.core.structured_output_schema import (
    action_to_motion_primitive,
    build_motion_primitive_response_schema,
    clear_synthesized_actions,
    register_synthesized_action,
    synthesized_catalogue,
)
from swarm_gpt.exception import LLMFormatError
from swarm_gpt.synth.manifest import ParamSpec, PrimitiveManifest
from swarm_gpt.utils.music_analyzer import SCHEMA_VERSION, Bar, Beat, Segment, SongStructure

HELIX_SOURCE = """
def double_helix(params, swarm_pos, tstart, tend, limits):
    turns, height_cm = params
    pos = swarm_pos.copy()
    waypoints = {}
    for t in np.linspace(tstart, tend, 3)[1:]:
        pos = pos + np.array([0.0, 0.0, height_cm / 200])
        waypoints[float(t)] = {i: p.copy() for i, p in enumerate(pos)}
    return pos, waypoints
"""

HELIX_CHECK = """
def check(pos, time, params):
    return [("rises", bool(np.all(pos[:, -1, 2] >= pos[:, 0, 2])), "never descends")]
"""

HELIX_PARAMS = (("turns", "int", 1, 4), ("height_cm", "float", 20.0, 150.0))


@pytest.fixture(autouse=True)
def _clean_registries():
    yield
    clear_synthesized_actions()
    for name in list(mp._synthesized):
        mp._synthesized.pop(name)
        mp.motion_primitives.pop(name, None)


def _manifest() -> PrimitiveManifest:
    return PrimitiveManifest(
        name="double_helix",
        intent="two interleaved helices climbing in opposite phase",
        params=tuple(
            ParamSpec(name=n, type=t, minimum=lo, maximum=hi) for n, t, lo, hi in HELIX_PARAMS
        ),
        source=HELIX_SOURCE,
        invariants=HELIX_CHECK,
    )


def _variant(schema: dict, primitive: str) -> dict | None:
    for variant in schema["$defs"]["action"]["anyOf"]:
        if variant["properties"]["primitive"]["enum"] == [primitive]:
            return variant
    return None


def _schema() -> dict:
    return build_motion_primitive_response_schema(
        all_keys=[(1, 1, 1)], required_keys=[(1, 1, 1)], num_drones=4
    )


def _structure() -> SongStructure:
    t = 0.0
    beats = []
    for beat_id in range(1, 5):
        beats.append(Beat(id=beat_id, time_s=t, position_in_bar=beat_id))
        t += 0.5
    bars = [Bar(id=1, start_s=0.0, beats=beats)]
    segments = [Segment(id=1, label="seg1", start_s=0.0, end_s=t, bars=bars)]
    return SongStructure(
        schema_version=SCHEMA_VERSION,
        source_path="music/Test.mp3",
        song_sha256="deadbeef",
        analyzer="allin1@test",
        bpm=120,
        segments=segments,
    )


def test_unregistered_primitive_is_absent_from_the_schema():
    assert _variant(_schema(), "double_helix") is None


def test_registering_adds_an_action_variant():
    register_synthesized_action("double_helix", "two helices", HELIX_PARAMS)
    variant = _variant(_schema(), "double_helix")
    assert variant is not None
    assert variant["required"] == ["primitive", "params"]
    assert variant["properties"]["params"]["required"] == ["turns", "height_cm"]


def test_variant_params_carry_declared_types_and_bounds():
    register_synthesized_action("double_helix", "two helices", HELIX_PARAMS)
    props = _variant(_schema(), "double_helix")["properties"]["params"]["properties"]
    assert props["turns"] == {"type": "integer", "minimum": 1, "maximum": 4}
    assert props["height_cm"] == {"type": "number", "minimum": 20.0, "maximum": 150.0}


def test_hand_written_primitives_survive_registration():
    register_synthesized_action("double_helix", "two helices", HELIX_PARAMS)
    assert _variant(_schema(), "form_circle") is not None


def test_registered_action_renders_a_call():
    register_synthesized_action("double_helix", "two helices", HELIX_PARAMS)
    action = {"primitive": "double_helix", "params": {"turns": 2, "height_cm": 80.0}}
    assert action_to_motion_primitive(action) == "double_helix(2, 80.0)"


def test_argument_order_follows_the_declaration_not_the_payload():
    register_synthesized_action("double_helix", "two helices", HELIX_PARAMS)
    action = {"primitive": "double_helix", "params": {"height_cm": 80.0, "turns": 2}}
    assert action_to_motion_primitive(action) == "double_helix(2, 80.0)"


def test_unknown_primitive_is_still_rejected():
    with pytest.raises(LLMFormatError, match="Unknown motion primitive"):
        action_to_motion_primitive({"primitive": "nope", "params": {}})


def test_wrong_param_names_are_rejected():
    register_synthesized_action("double_helix", "two helices", HELIX_PARAMS)
    action = {"primitive": "double_helix", "params": {"turns": 2, "wrong": 1}}
    with pytest.raises(LLMFormatError, match="must be exactly"):
        action_to_motion_primitive(action)


def test_shadowing_a_hand_written_primitive_is_rejected():
    with pytest.raises(ValueError, match="shadows"):
        register_synthesized_action("form_circle", "impostor", HELIX_PARAMS)


def test_clearing_removes_the_variant():
    register_synthesized_action("double_helix", "two helices", HELIX_PARAMS)
    clear_synthesized_actions()
    assert _variant(_schema(), "double_helix") is None
    with pytest.raises(LLMFormatError):
        action_to_motion_primitive({"primitive": "double_helix", "params": {}})


def test_catalogue_is_empty_when_nothing_is_registered():
    assert synthesized_catalogue() == ""


def test_catalogue_renders_signature_and_intent():
    register_synthesized_action("double_helix", "two interleaved helices", HELIX_PARAMS)
    text = synthesized_catalogue()
    assert "double_helix(turns: int [1, 4], height_cm: float [20.0, 150.0])" in text
    assert "two interleaved helices" in text


def test_manifest_register_wires_both_resolution_and_schema():
    manifest = _manifest()
    fn, _check = manifest.compile()
    manifest.register(fn)
    assert mp.primitive_by_name("double_helix") is fn
    assert mp.motion_primitives["double_helix"]["n_args"] == 2
    assert _variant(_schema(), "double_helix") is not None
    assert "double_helix" in synthesized_catalogue()


def test_registered_primitive_runs_through_the_resolved_callable():
    manifest = _manifest()
    fn, _check = manifest.compile()
    manifest.register(fn)
    limits = {"lower": np.array([-2.0, -2.0, 0.0]), "upper": np.array([2.0, 2.0, 2.0])}
    final_pos, waypoints = mp.primitive_by_name("double_helix")(
        manifest.bind([2, 80.0]), np.zeros((4, 3)), 0.0, 4.0, limits
    )
    assert final_pos.shape == (4, 3)
    assert len(waypoints) == 2


def test_persisted_manifest_round_trips_into_the_library(tmp_path: Path):
    manifest = _manifest()
    entry = {"manifest": dataclasses.asdict(manifest), "args": [2, 80.0]}
    path = tmp_path / "double_helix.json"
    path.write_text(json.dumps(entry, indent=2))

    loaded = PrimitiveManifest.from_payload(json.loads(path.read_text())["manifest"])
    assert loaded == manifest
    fn, _check = loaded.compile()
    loaded.register(fn)
    action = {"primitive": "double_helix", "params": {"turns": 2, "height_cm": 80.0}}
    assert action_to_motion_primitive(action) == "double_helix(2, 80.0)"


def test_prompt_carries_the_catalogue():
    register_synthesized_action("double_helix", "two interleaved helices", HELIX_PARAMS)
    choreographer = Choreographer(
        config_file=virtual_crazyswarm_config(n_drones=4),
        llm_provider="openai",
        use_motion_primitives=True,
    )
    prompt = choreographer._format_initial_user_prompt("song", _structure())
    assert "double_helix(turns: int [1, 4], height_cm: float [20.0, 150.0])" in prompt


def test_prompt_omits_the_block_when_nothing_is_registered():
    choreographer = Choreographer(
        config_file=virtual_crazyswarm_config(n_drones=4),
        llm_provider="openai",
        use_motion_primitives=True,
    )
    prompt = choreographer._format_initial_user_prompt("song", _structure())
    assert "double_helix" not in prompt
    assert "form_circle" in prompt
