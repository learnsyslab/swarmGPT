import json
from types import SimpleNamespace
from typing import Any

import pytest
from conftest import virtual_crazyswarm_config

from swarm_gpt.core.choreographer import Choreographer
from swarm_gpt.core.structured_output_schema import (
    build_motion_primitive_response_schema,
    decode_key,
    encode_key,
)
from swarm_gpt.exception import LLMFormatError
from swarm_gpt.utils.llm_providers import RESPONSES_TEMPERATURE
from swarm_gpt.utils.music_analyzer import SCHEMA_VERSION, Bar, Beat, Segment, SongStructure


def _simple_structure(n_segments: int = 2, n_bars: int = 1, n_beats: int = 4) -> SongStructure:
    """Synthetic SongStructure: ``n_segments`` × ``n_bars`` × ``n_beats`` at 0.5s spacing."""
    segments: list[Segment] = []
    t = 0.0
    for seg_id in range(1, n_segments + 1):
        bars: list[Bar] = []
        seg_start = t
        for bar_id in range(1, n_bars + 1):
            bar_start = t
            beats: list[Beat] = []
            for beat_id in range(1, n_beats + 1):
                beats.append(Beat(id=beat_id, time_s=t, position_in_bar=beat_id))
                t += 0.5
            bars.append(Bar(id=bar_id, start_s=bar_start, beats=beats))
        segments.append(
            Segment(id=seg_id, label=f"seg{seg_id}", start_s=seg_start, end_s=t, bars=bars)
        )
    return SongStructure(
        schema_version=SCHEMA_VERSION,
        source_path="music/Test.mp3",
        source_sha256="deadbeef",
        analyzer="allin1@test",
        bpm=120,
        segments=segments,
    )


def _contains_one_of(node: Any) -> bool:
    if isinstance(node, dict):
        if "oneOf" in node:
            return True
        return any(_contains_one_of(value) for value in node.values())
    if isinstance(node, list):
        return any(_contains_one_of(value) for value in node)
    return False


def test_encode_decode_key_round_trip():
    assert encode_key(2, 4, 1) == "s2b4t1"
    assert decode_key("s2b4t1") == (2, 4, 1)


def test_decode_key_rejects_malformed():
    with pytest.raises(LLMFormatError):
        decode_key("1")


def test_schema_lists_all_keys_in_enum_as_array_of_entries():
    structure = _simple_structure(n_segments=2, n_bars=1, n_beats=4)
    schema = build_motion_primitive_response_schema(
        all_keys=structure.all_keys(), required_keys=structure.required_keys(), num_drones=5
    )
    choreography = schema["properties"]["choreography"]
    # choreography is an array of {key, actions} entries.
    assert choreography["type"] == "array"
    item = choreography["items"]
    # 2 segments × 1 bar × 4 beats = 8 addressable keys, all offered via the key enum.
    assert set(item["properties"]["key"]["enum"]) == {
        "s1b1t1",
        "s1b1t2",
        "s1b1t3",
        "s1b1t4",
        "s2b1t1",
        "s2b1t2",
        "s2b1t3",
        "s2b1t4",
    }
    # Strict mode: both fields of an entry are required.
    assert item["required"] == ["key", "actions"]
    assert item["additionalProperties"] is False


def test_schema_rejects_required_keys_not_in_all_keys():
    with pytest.raises(ValueError, match="required_keys not present"):
        build_motion_primitive_response_schema(
            all_keys=[(1, 1, 1)], required_keys=[(1, 1, 2)], num_drones=4
        )


def test_structured_payload_to_choreography_uses_hierarchical_keys():
    config_path = virtual_crazyswarm_config(n_drones=4)
    choreographer = Choreographer(
        config_file=config_path, llm_provider="openai", use_motion_primitives=True
    )
    payload = {
        "song_mood": "calm",
        "choreography_plan": "simple",
        "choreography": [
            {
                "key": "s1b1t1",
                "actions": [
                    {
                        "primitive": "form_circle",
                        "params": {"drone_ids": [1, 2], "radius_cm": 100, "z_coord_cm": 100, "time_to_finish_s": 1.5},
                    }
                ],
            },
            {
                "key": "s2b1t1",
                "actions": [
                    {"primitive": "rotate", "params": {"angle_deg": 90, "axis": "z"}},
                    {"primitive": "move_z", "params": {"drone_ids": [1], "delta_cm": 10}},
                ],
            },
        ],
    }

    choreography = choreographer._structured_payload_to_choreography(payload)

    assert choreography == {
        (1, 1, 1): "form_circle([1, 2], 100, 100, 1.5)",
        (2, 1, 1): "rotate(90, 'z'); move_z([1], 10)",
    }


def test_structured_payload_to_choreography_rejects_unexpected_named_params():
    config_path = virtual_crazyswarm_config(n_drones=4)
    choreographer = Choreographer(
        config_file=config_path, llm_provider="openai", use_motion_primitives=True
    )
    payload = {
        "song_mood": "calm",
        "choreography_plan": "simple",
        "choreography": [
            {
                "key": "s1b1t1",
                "actions": [
                    {
                        "primitive": "form_cone",
                        "params": {
                            "drone_ids": [1, 2, 3],
                            "delta_height_cm": 60,
                            "spacing_cm": 60,
                            "is_inverted": 0,
                        },
                    }
                ],
            }
        ],
    }
    with pytest.raises(LLMFormatError, match="unexpected \\['drone_ids'\\]"):
        choreographer._structured_payload_to_choreography(payload)


def test_call_responses_structured_includes_json_schema_format():
    config_path = virtual_crazyswarm_config(n_drones=4)
    choreographer = Choreographer(
        config_file=config_path, llm_provider="openai", use_motion_primitives=True
    )
    structure = _simple_structure(n_segments=1, n_bars=1, n_beats=1)
    captured: dict[str, Any] = {}

    class FakeResponses:
        def create(self, **kwargs: Any) -> SimpleNamespace:
            captured.update(kwargs)
            payload = {
                "song_mood": "energetic",
                "choreography_plan": "test",
                "choreography": [
                    {
                        "key": "s1b1t1",
                        "actions": [
                            {"primitive": "rotate", "params": {"angle_deg": 0, "axis": "z"}}
                        ],
                    }
                ],
            }
            return SimpleNamespace(error=None, output_text=json.dumps(payload))

    class FakeClient:
        responses = FakeResponses()

    choreographer._chat_client_for_call = lambda: FakeClient()  # noqa: E731
    messages = [{"role": "user", "content": "hello"}]

    parsed = choreographer._call_responses_structured(messages, structure=structure)

    assert parsed["choreography"][0]["key"] == "s1b1t1"
    assert captured["text"]["format"]["type"] == "json_schema"
    assert captured["text"]["format"]["name"] == "swarmgpt_choreography"
    assert captured["text"]["format"]["strict"] is True
    schema = captured["text"]["format"]["schema"]
    item = schema["properties"]["choreography"]["items"]
    assert item["properties"]["key"]["enum"] == ["s1b1t1"]
    assert item["properties"]["actions"] == {"$ref": "#/$defs/action_list"}
    action_schema = schema["$defs"]["action"]
    variants = action_schema["anyOf"]
    assert any(
        variant["properties"]["primitive"]["enum"] == ["spiral_speed"]
        and variant["properties"]["params"]["required"]
        == ["steps", "height_cm", "degrees", "radius_increase"]
        for variant in variants
    )
    assert all("args" not in json.dumps(variant) for variant in variants)


def test_schema_contains_no_openai_unsupported_keywords():
    structure = _simple_structure(n_segments=2, n_bars=2, n_beats=4)
    schema = build_motion_primitive_response_schema(
        all_keys=structure.all_keys(), required_keys=structure.required_keys(), num_drones=4
    )
    schema_text = json.dumps(schema)
    assert "oneOf" not in schema_text
    assert "uniqueItems" not in schema_text
    assert '"items": false' not in schema_text.lower()
    assert '"args"' not in schema_text


def test_ollama_motion_primitives_uses_structured_outputs():
    config_path = virtual_crazyswarm_config(n_drones=4)
    choreographer = Choreographer(
        config_file=config_path, llm_provider="ollama", use_motion_primitives=True
    )
    assert choreographer._uses_structured_outputs() is True


def test_call_responses_structured_ollama_uses_native_chat(monkeypatch: pytest.MonkeyPatch):
    config_path = virtual_crazyswarm_config(n_drones=4)
    choreographer = Choreographer(
        config_file=config_path, llm_provider="ollama", use_motion_primitives=True
    )
    structure = _simple_structure(n_segments=1, n_bars=1, n_beats=1)
    captured: dict[str, Any] = {}

    def fake_ollama_chat(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        payload = {
            "song_mood": "energetic",
            "choreography_plan": "test",
            "choreography": [
                {
                    "key": "s1b1t1",
                    "actions": [{"primitive": "rotate", "params": {"angle_deg": 0, "axis": "z"}}],
                }
            ],
        }
        return {"message": {"content": json.dumps(payload)}}

    monkeypatch.setattr("swarm_gpt.core.choreographer.cancellable_ollama_chat", fake_ollama_chat)
    parsed = choreographer._call_responses_structured(
        [{"role": "user", "content": "hello"}], structure=structure
    )

    assert parsed["choreography"][0]["actions"][0]["primitive"] == "rotate"
    assert captured["model"] == choreographer.model_id
    assert captured["format"]["properties"]["choreography"]["items"]["properties"]["key"][
        "enum"
    ] == ["s1b1t1"]
    tail = captured["messages"][-1]["content"]
    assert "Return valid JSON only. Match the provided response format exactly." in tail
    assert "never positional args arrays" in tail
    assert captured["options"]["temperature"] == RESPONSES_TEMPERATURE


def test_structured_initial_prompt_renders_hierarchical_keys():
    config_path = virtual_crazyswarm_config(n_drones=4)
    choreographer = Choreographer(
        config_file=config_path, llm_provider="ollama", use_motion_primitives=True
    )
    structure = _simple_structure(n_segments=2, n_bars=1, n_beats=4)

    messages = choreographer.format_initial_prompt("test song", structure)

    user_content = messages[1]["content"]
    assert "test song" in user_content
    assert "120 BPM" in user_content
    # Required keys block must include the segment-opening keys.
    assert "s1b1t1" in user_content
    assert "s2b1t1" in user_content
    # Segment label is rendered.
    assert "seg1" in user_content and "seg2" in user_content


def test_generate_choreography_ollama_raises_on_structured_errors(monkeypatch: pytest.MonkeyPatch):
    config_path = virtual_crazyswarm_config(n_drones=4)
    choreographer = Choreographer(
        config_file=config_path, llm_provider="ollama", use_motion_primitives=True
    )
    structure = _simple_structure(n_segments=1, n_bars=1, n_beats=1)
    monkeypatch.setattr(
        choreographer,
        "_call_responses_structured",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(LLMFormatError("bad json")),
    )
    monkeypatch.setattr(
        choreographer,
        "_call_responses",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("no fallback expected")),
    )
    with pytest.raises(LLMFormatError, match="bad json"):
        choreographer.generate_choreography(
            prompt=[{"role": "user", "content": "hello"}], structure=structure
        )


def test_generate_choreography_ollama_raises_when_structured_payload_incomplete(
    monkeypatch: pytest.MonkeyPatch,
):
    config_path = virtual_crazyswarm_config(n_drones=4)
    choreographer = Choreographer(
        config_file=config_path, llm_provider="ollama", use_motion_primitives=True
    )
    structure = _simple_structure(n_segments=1, n_bars=1, n_beats=1)
    monkeypatch.setattr(
        choreographer,
        "_call_responses_structured",
        lambda *_args, **_kwargs: {
            "choreography": [
                {
                    "key": "s1b1t1",
                    "actions": [{"primitive": "rotate", "params": {"angle_deg": 0, "axis": "z"}}],
                }
            ]
        },
    )
    with pytest.raises(LLMFormatError, match="missing required keys"):
        choreographer.generate_choreography(
            prompt=[{"role": "user", "content": "hello"}], structure=structure
        )


def test_form_star_schema_includes_time_to_finish_s():
    """F1: form_star schema must include time_to_finish_s as a required param."""
    schema = build_motion_primitive_response_schema(
        all_keys=[(1, 1, 1)], required_keys=[(1, 1, 1)], num_drones=5
    )
    variants = schema["$defs"]["action"]["anyOf"]
    form_star_variant = next(
        v for v in variants if v["properties"]["primitive"]["enum"] == ["form_star"]
    )
    assert "time_to_finish_s" in form_star_variant["properties"]["params"]["required"]
    assert "time_to_finish_s" in form_star_variant["properties"]["params"]["properties"]


def test_form_circle_schema_includes_time_to_finish_s():
    """F1: form_circle schema must include time_to_finish_s as a required param."""
    schema = build_motion_primitive_response_schema(
        all_keys=[(1, 1, 1)], required_keys=[(1, 1, 1)], num_drones=5
    )
    variants = schema["$defs"]["action"]["anyOf"]
    form_circle_variant = next(
        v for v in variants if v["properties"]["primitive"]["enum"] == ["form_circle"]
    )
    assert "time_to_finish_s" in form_circle_variant["properties"]["params"]["required"]


def test_form_cone_schema_includes_time_to_finish_s():
    """F1: form_cone schema must include time_to_finish_s as a required param."""
    schema = build_motion_primitive_response_schema(
        all_keys=[(1, 1, 1)], required_keys=[(1, 1, 1)], num_drones=5
    )
    variants = schema["$defs"]["action"]["anyOf"]
    form_cone_variant = next(
        v for v in variants if v["properties"]["primitive"]["enum"] == ["form_cone"]
    )
    assert "time_to_finish_s" in form_cone_variant["properties"]["params"]["required"]
