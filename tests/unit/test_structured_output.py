import ast
import dataclasses
import json
import re
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from conftest import virtual_crazyswarm_config

from swarm_gpt.core.choreographer import Choreographer
from swarm_gpt.core.lighting import Look, load_lighting_config
from swarm_gpt.core.lighting_primitives import LIGHTING_PRIMITIVES, build_look
from swarm_gpt.core.structured_output_schema import (
    LIGHTING_PRIMITIVE_ARG_ORDER,
    action_to_lighting_primitive,
    build_motion_primitive_response_schema,
    decode_key,
    encode_key,
    structured_payload_to_choreography,
    structured_payload_to_lighting,
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
        song_sha256="deadbeef",
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
                        "params": {
                            "drone_ids": [1, 2],
                            "radius_cm": 100,
                            "z_coord_cm": 100,
                            "time_to_finish_s": 1.5,
                        },
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


# --- lighting track (spec §10.1, §10.2) ---------------------------------------------------------

# The §10.2 catalogue, restated deliberately: this table is the pin between the spec's authority on
# parameter names and what the schema actually offers. Deriving it from the code would test nothing.
LIGHTING_CATALOGUE: dict[str, list[str]] = {
    "light_color": ["sel", "color", "deck"],
    "gradient": ["sel", "color_a", "color_b", "by", "deck"],
    "rainbow": ["sel", "period_beats", "spread", "deck"],
    "light_on": ["sel", "deck"],
    "light_off": ["sel", "deck"],
    "pulse": ["sel", "period_beats", "deck"],
    "blink": ["sel", "period_beats", "duty", "deck"],
    "strobe_decay": ["sel", "period_beats", "deck"],
    "chase": ["sel", "period_beats", "length", "group_size", "spread", "deck"],
    "sweep": ["sel", "period_beats", "axis", "deck"],
    "ripple_light": ["sel", "period_beats", "deck"],
    "alternate_blink": ["sel", "period_beats", "by", "deck"],
}

LIGHTING_N_DRONES = 6
# Six drones spread along +x with distinct coordinates, so `left`/`right` and the spatial spreads
# all resolve unambiguously.
LIGHTING_POSITIONS = np.stack([np.arange(6.0), np.zeros(6), np.linspace(1.0, 2.0, 6)], axis=1)
LIGHTING_CFG = load_lighting_config()


def _lighting_schema(num_drones: int = LIGHTING_N_DRONES) -> dict[str, Any]:
    return build_motion_primitive_response_schema(
        all_keys=[(1, 1, 1), (1, 1, 2)], required_keys=[(1, 1, 1)], num_drones=num_drones
    )


def _lighting_variant(primitive: str, schema: dict[str, Any] | None = None) -> dict[str, Any]:
    schema = schema if schema is not None else _lighting_schema()
    return next(
        v
        for v in schema["$defs"]["lighting_action"]["anyOf"]
        if v["properties"]["primitive"]["enum"] == [primitive]
    )


def _sample_value(name: str, param_schema: dict[str, Any]) -> Any:
    """A legal value for one lighting parameter, taken from the schema itself."""
    if name == "sel":
        return {"kind": "all", "ids": [], "count": 1}
    if "enum" in param_schema:
        return param_schema["enum"][0]
    return 1 if param_schema["type"] == "integer" else 1.0


def _sample_action(primitive: str, **overrides: Any) -> dict[str, Any]:
    """An action for ``primitive`` whose every parameter value comes from the built schema."""
    params_schema = _lighting_variant(primitive)["properties"]["params"]["properties"]
    params = {name: _sample_value(name, schema) for name, schema in params_schema.items()}
    params.update(overrides)
    return {"primitive": primitive, "params": params}


def _text_to_action(rendered: str) -> dict[str, Any]:
    """Parse a rendered call back the way the text path does (``choreographer.py:708-719``)."""
    name = rendered.split("(")[0].strip(" -\n")
    args = ast.literal_eval("(" + rendered.split("(")[1].split("#")[0][:-1] + ",)")
    return {"primitive": name, "params": dict(zip(LIGHTING_PRIMITIVE_ARG_ORDER[name], args))}


def _build_rendered(action: dict[str, Any]) -> Look:
    """Render an action to text, parse it back, and compile it into a `Look`."""
    parsed = _text_to_action(action_to_lighting_primitive(action))
    return build_look([parsed], 0.0, LIGHTING_POSITIONS, LIGHTING_N_DRONES, LIGHTING_CFG, bpm=120.0)


def test_schema_carries_a_lighting_track_keyed_to_all_keys():
    """§10.1: lighting is its own top-level array over the same address space as choreography."""
    schema = _lighting_schema()
    lighting = schema["properties"]["lighting"]
    choreography = schema["properties"]["choreography"]
    assert lighting["type"] == "array"
    item = lighting["items"]
    assert item["properties"]["key"]["enum"] == choreography["items"]["properties"]["key"]["enum"]
    assert item["required"] == ["key", "actions"]
    assert item["additionalProperties"] is False
    # Strict mode requires every declared property to be required, so the field must always be
    # emitted -- but an empty array is a legal, and expected, "no lighting" answer.
    assert "lighting" in schema["required"]
    assert "minItems" not in lighting


def test_the_lighting_track_points_at_the_lighting_action_list():
    """§10.1: the two tracks share a shape but not a vocabulary, and the `$ref` is where that lives.

    Pointing `lighting`'s `actions` at `#/$defs/action_list` leaves the lighting action schema
    perfectly valid but orphaned — nothing references it — and tells the model to emit *motion*
    primitives into the lighting array. Every other assertion here reads `$defs` directly, so all
    of them still pass; only the reference itself says which vocabulary the track carries.
    """
    schema = _lighting_schema()
    tracks = schema["properties"]
    assert tracks["lighting"]["items"]["properties"]["actions"] == {
        "$ref": "#/$defs/lighting_action_list"
    }
    assert tracks["choreography"]["items"]["properties"]["actions"] == {
        "$ref": "#/$defs/action_list"
    }


def test_lighting_colour_enum_is_exactly_the_shipped_palette():
    """An invented colour must be unrepresentable: the enum comes from `lighting.toml`."""
    variant = _lighting_variant("light_color")
    colours = variant["properties"]["params"]["properties"]["color"]["enum"]
    assert set(colours) == set(LIGHTING_CFG.palette)
    assert "chartreuse" not in colours
    gradient = _lighting_variant("gradient")["properties"]["params"]["properties"]
    assert gradient["color_a"]["enum"] == colours
    assert gradient["color_b"]["enum"] == colours


def test_lighting_colour_enum_is_resolved_when_the_schema_is_built(monkeypatch: pytest.MonkeyPatch):
    """The palette is read at schema-build time, not at import.

    This module is imported widely, so reading `lighting.toml` from disk at import made a
    malformed calibration file an import error — surfacing far from its cause, and unfixable by
    anything downstream. Patching the loader and seeing the new palette in the enum is what says
    the read happens where it can be reasoned about.
    """
    trimmed = dataclasses.replace(LIGHTING_CFG, palette={"red": LIGHTING_CFG.palette["red"]})
    monkeypatch.setattr(
        "swarm_gpt.core.structured_output_schema.load_lighting_config", lambda: trimmed
    )

    variant = _lighting_variant("light_color")

    assert variant["properties"]["params"]["properties"]["color"]["enum"] == ["red"]


@pytest.mark.parametrize(("primitive", "param_names"), sorted(LIGHTING_CATALOGUE.items()))
def test_lighting_variant_lists_exactly_its_catalogue_parameters(
    primitive: str, param_names: list[str]
):
    """§10.2 is authoritative on each primitive's parameters; the variant must match it exactly."""
    params = _lighting_variant(primitive)["properties"]["params"]
    assert params["required"] == param_names
    assert list(params["properties"]) == param_names
    assert params["additionalProperties"] is False


def test_every_lighting_primitive_has_a_schema_variant_and_vice_versa():
    """No drift: the schema offers exactly the engine's vocabulary, and the catalogue's."""
    schema = _lighting_schema()
    offered = {
        v["properties"]["primitive"]["enum"][0] for v in schema["$defs"]["lighting_action"]["anyOf"]
    }
    assert offered == set(LIGHTING_PRIMITIVES)
    assert offered == set(LIGHTING_CATALOGUE)
    assert set(LIGHTING_PRIMITIVE_ARG_ORDER) == set(LIGHTING_PRIMITIVES)


def test_lighting_selector_carries_every_field_and_stays_strict():
    """Strict mode cannot express a variant by omission, so `sel` carries all three fields."""
    sel = _lighting_variant("light_on")["properties"]["params"]["properties"]["sel"]
    assert sel["type"] == "object"
    assert sel["additionalProperties"] is False
    assert sel["required"] == ["kind", "ids", "count"]
    assert sel["properties"]["kind"]["enum"] == ["all", "ids", "even", "odd", "first", "left"] + [
        "right"
    ]
    # `ids` is ignored unless kind == "ids", so an empty list must be a legal filler.
    assert "minItems" not in sel["properties"]["ids"]
    assert sel["properties"]["ids"]["items"]["maximum"] == LIGHTING_N_DRONES


def test_lighting_keys_are_not_required_keys():
    """§10.1: the point of a separate track -- no required-key or alternation rule applies."""
    schema = _lighting_schema()
    lighting = schema["properties"]["lighting"]
    # Every address is offered, not just the required ones: required_keys is [(1, 1, 1)] here.
    assert lighting["items"]["properties"]["key"]["enum"] == ["s1b1t1", "s1b1t2"]
    # Nothing constrains how many lighting entries there are, nor their order or alternation.
    assert set(lighting) == {"type", "items"}
    # And a lighting track never enters the dict the required-key check downstream polices.
    payload = {
        "song_mood": "calm",
        "choreography_plan": "simple",
        "choreography": [
            {
                "key": "s1b1t1",
                "actions": [{"primitive": "rotate", "params": {"angle_deg": 90, "axis": "z"}}],
            }
        ],
        "lighting": [{"key": "s1b1t2", "actions": [_sample_action("light_on")]}],
    }
    assert set(structured_payload_to_choreography(payload)) == {(1, 1, 1)}


def test_action_to_lighting_primitive_renders_the_selector_as_a_two_sequence():
    """The rendered selector is the ``(kind, args)`` pair `select` consumes, paren-free."""
    assert (
        action_to_lighting_primitive(_sample_action("light_color", color="blue", deck="both"))
        == "light_color(['all', []], 'blue', 'both')"
    )
    assert (
        action_to_lighting_primitive(
            _sample_action(
                "pulse",
                sel={"kind": "ids", "ids": [1, 3, 5], "count": 1},
                period_beats=2,
                deck="top",
            )
        )
        == "pulse(['ids', [1, 3, 5]], 2, 'top')"
    )
    assert (
        action_to_lighting_primitive(
            _sample_action(
                "chase",
                sel={"kind": "first", "ids": [], "count": 4},
                period_beats=4,
                length=2,
                group_size=1,
                spread="neighbour",
                deck="bot",
            )
        )
        == "chase(['first', [4]], 4, 2, 1, 'neighbour', 'bot')"
    )


@pytest.mark.parametrize("primitive", sorted(LIGHTING_CATALOGUE))
def test_rendered_lighting_action_round_trips_into_a_look(primitive: str):
    """Render -> text -> parse -> `build_look` produces exactly one artefact per action."""
    look = _build_rendered(_sample_action(primitive))
    artefacts = len(look.colour_layers) + len(look.brightness_layers) + int(look.off_mask.any())
    assert artefacts == 1


def _enum_cases() -> list[tuple[str, str, Any]]:
    """Every enum value the lighting schema offers, paired with a primitive that takes it."""
    cases: list[tuple[str, str, Any]] = []
    for primitive in LIGHTING_CATALOGUE:
        properties = _lighting_variant(primitive)["properties"]["params"]["properties"]
        for name, param_schema in properties.items():
            for value in param_schema.get("enum", []):
                cases.append((primitive, name, value))
        for kind in properties["sel"]["properties"]["kind"]["enum"]:
            ids = [1, 3] if kind == "ids" else []
            cases.append((primitive, "sel", {"kind": kind, "ids": ids, "count": 2}))
    return cases


@pytest.mark.parametrize(("primitive", "name", "value"), _enum_cases())
def test_every_enum_value_the_schema_offers_is_accepted_by_the_engine(
    primitive: str, name: str, value: Any
):
    """The schema must not offer a colour, selector, spread, axis or deck `build_look` rejects."""
    look = _build_rendered(_sample_action(primitive, **{name: value}))
    artefacts = len(look.colour_layers) + len(look.brightness_layers) + int(look.off_mask.any())
    assert artefacts == 1


def test_structured_payload_to_lighting_rejects_a_duplicate_key():
    """Two entries on one address would silently collapse into whichever came last."""
    entry = {"key": "s1b1t1", "actions": [_sample_action("light_on")]}
    with pytest.raises(LLMFormatError, match="Duplicate lighting key"):
        structured_payload_to_lighting({"lighting": [entry, dict(entry)]})


def test_structured_payload_to_lighting_rejects_an_empty_action_list():
    """An entry with no actions renders to an empty string, which parses back to an empty look —
    a key that claims to change the lights and does nothing, rather than an error."""
    with pytest.raises(LLMFormatError, match="non-empty action list"):
        structured_payload_to_lighting({"lighting": [{"key": "s1b1t1", "actions": []}]})


def test_structured_payload_to_lighting_rejects_a_non_string_key():
    """`decode_key` takes a string; anything else has to be reported as a format error here, not
    left to fail as whatever `re` does with the wrong type."""
    entry = {"key": 111, "actions": [_sample_action("light_on")]}
    with pytest.raises(LLMFormatError, match="'key' must be a string"):
        structured_payload_to_lighting({"lighting": [entry]})


def test_action_to_lighting_primitive_rejects_unknown_primitive():
    with pytest.raises(LLMFormatError, match="Unknown lighting primitive"):
        action_to_lighting_primitive({"primitive": "disco_ball", "params": {}})


def test_action_to_lighting_primitive_rejects_wrong_params():
    action = _sample_action("pulse")
    action["params"]["duty"] = 0.5
    with pytest.raises(LLMFormatError, match="unexpected \\['duty'\\]"):
        action_to_lighting_primitive(action)


# --- lighting prompt (spec §10.2, §10.3) --------------------------------------------------------


def _choreographer() -> Choreographer:
    return Choreographer(
        config_file=virtual_crazyswarm_config(n_drones=4),
        llm_provider="openai",
        use_motion_primitives=True,
    )


def _lighting_prompt_section() -> str:
    """The `<lighting>` block of the *rendered* user prompt, so `.format()` is exercised too."""
    messages = _choreographer().format_initial_prompt("test song", _simple_structure())
    return messages[1]["content"].split("<lighting>")[1].split("</lighting>")[0]


def _json_objects_at(text: str, marker: str) -> list[dict[str, Any]]:
    """Every complete JSON object in ``text`` beginning at ``marker``, found by brace matching.

    Candidates containing the prompt's own ``...`` placeholder are skipped; anything else that
    fails to parse is a real syntax error in the prompt and is allowed to raise.
    """
    objects: list[dict[str, Any]] = []
    for start in (i for i in range(len(text)) if text.startswith(marker, i)):
        depth = 0
        for end in range(start, len(text)):
            depth += {"{": 1, "}": -1}.get(text[end], 0)
            if depth == 0:
                candidate = text[start : end + 1]
                if "..." not in candidate:
                    objects.append(json.loads(candidate))
                break
    return objects


# A documented signature line: ``- name(a, b, c) — what it does``.
_SIGNATURE_RE = re.compile(r"^\s*-\s*(\w+)\(([^)]*)\)")

# The leading run of vocabulary names on a prose line or clause: ``all``, ``even / odd``,
# ``x/y/z``, ``alternate_parity or alternate_side``. It ends where the description begins.
_NAME_RUN_RE = re.compile(r"^([a-z_]+(?:(?:/| / | or )[a-z_]+)*)\s")


def _split_names(run: str) -> list[str]:
    return re.split(r"\s*/\s*|\s+or\s+", run)


def _documented_signatures() -> dict[str, list[str]]:
    """The `name(a, b, c)` signatures the `<lighting>` prose actually shows the model."""
    signatures: dict[str, list[str]] = {}
    for line in _lighting_prompt_section().splitlines():
        if (match := _SIGNATURE_RE.match(line)) and match.group(1) in LIGHTING_PRIMITIVE_ARG_ORDER:
            signatures[match.group(1)] = [arg.strip() for arg in match.group(2).split(",")]
    return signatures


def _prompt_sel_block() -> str:
    """The ``sel —`` selector table, header line dropped."""
    return _lighting_prompt_section().split("sel —")[1].split("deck —")[0]


def _prompt_spread_enumeration() -> str:
    """The clause list the `rainbow` bullet spells the spread vocabulary out in."""
    lines = _lighting_prompt_section().splitlines()
    start = next(i for i, line in enumerate(lines) if line.strip().startswith("- rainbow("))
    bullet = [lines[start]]
    for line in lines[start + 1 :]:
        if not line.strip() or line.strip().startswith("- "):
            break
        bullet.append(line)
    return " ".join(line.strip() for line in bullet).rsplit(" spread ", 1)[1]


def test_lighting_prompt_documents_every_primitive_signature_exactly():
    """The prose signatures are the fourth place a lighting parameter lives (CLAUDE.md §7.1).

    Only the structured JSON examples below were validated; the signature lines the model actually
    reads were merely grepped for `name(`. That is the shape of the `form_circle` bug this repo
    already shipped: dropping `duty` from `blink`, or renaming `sweep`'s `axis` to `direction`,
    leaves the model unable to emit a call the parser will accept and nothing notices.
    """
    assert _documented_signatures() == dict(LIGHTING_PRIMITIVE_ARG_ORDER)


def test_lighting_prompt_offers_exactly_the_selector_vocabulary():
    """A selector the prompt names but `select` does not have is a guaranteed reprompt."""
    kinds = _lighting_variant("light_on")["properties"]["params"]["properties"]["sel"]
    documented: set[str] = set()
    for line in _prompt_sel_block().splitlines()[1:]:
        if match := _NAME_RUN_RE.match(line.strip()):
            documented.update(_split_names(match.group(1)))
    assert documented == set(kinds["properties"]["kind"]["enum"])


def test_lighting_prompt_documents_ids_as_one_indexed():
    """`select` shifts `ids` down by one, and rejects 0 outright.

    Advertising a 0-indexed list is therefore an off-by-one on every `ids` selection the model
    writes, and an outright failure whenever it picks the first drone.
    """
    ids_line = next(
        line for line in _prompt_sel_block().splitlines() if line.strip().startswith("ids")
    )
    assert "1-indexed" in ids_line


def test_lighting_prompt_offers_exactly_the_spread_vocabulary():
    """`rainbow` is the one primitive whose spread the model chooses by name, so the enumeration
    in its bullet is a vocabulary the schema has to be able to represent."""
    spread = _lighting_variant("rainbow")["properties"]["params"]["properties"]["spread"]
    documented: set[str] = set()
    for clause in _prompt_spread_enumeration().split(","):
        if match := _NAME_RUN_RE.match(clause.strip()):
            documented.update(_split_names(match.group(1)))
    assert documented == set(spread["enum"])


def test_lighting_prompt_colour_list_is_exactly_the_shipped_palette():
    """An invented colour is unrepresentable in the schema; the prompt must not imply otherwise."""
    lines = _lighting_prompt_section().splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.strip().startswith("colors ("))
    # The list wraps onto one continuation line.
    listed = " ".join(lines[start : start + 2]).split(":", 1)[1]
    assert {name.strip() for name in listed.split(",")} == set(LIGHTING_CFG.palette)


def _prompt_bullet(name: str) -> str:
    """One primitive's bullet from the `<lighting>` prose, continuation lines folded in."""
    lines = _lighting_prompt_section().splitlines()
    start = next(i for i, line in enumerate(lines) if line.strip().startswith(f"- {name}("))
    bullet = [lines[start]]
    for line in lines[start + 1 :]:
        if not line.strip() or line.strip().startswith("- "):
            break
        bullet.append(line)
    return " ".join(line.strip() for line in bullet)


def test_lighting_prompt_says_which_formations_collapse_a_spatial_spread():
    """A spread with no extent to run along degrades into a synchronised blink (§7.3).

    `sweep(axis="z")` over any `form_circle` or `form_star` output, and `ripple_light` over a ring,
    are both natural things to author and both silently stop travelling. The engine logs it, but
    only the prompt can stop the model writing it in the first place.
    """
    assert "extent" in _prompt_bullet("sweep"), "sweep needs spread along its axis"
    assert "flat" in _prompt_bullet("sweep"), "and z is the axis a formation usually lacks"
    assert "radius" in _prompt_bullet("ripple_light"), "ripple_light needs radial extent"


def test_lighting_prompt_restricts_group_size_to_the_ranked_spreads():
    """`group_size` buckets a rank, and only `neighbour` and `index` produce one (§7.3).

    The engine rejects the other combinations rather than ignoring them, so a prompt that offers
    `group_size` as unconditional spends a reprompt on every emission that pairs it with `x`.
    """
    bullet = _prompt_bullet("chase")
    assert "group_size" in bullet
    assert "neighbour" in bullet.split("group_size", 1)[1].split("spread is", 1)[0]


def test_lighting_prompt_states_what_the_primitive_names_do_not_say():
    """The three facts the LLM cannot infer from the names alone (plan Task 9 step 1)."""
    section = _lighting_prompt_section()
    assert "period_beats is measured in BEATS, not seconds" in section
    assert "Lighting keys are NOT required keys" in section
    assert "light_on DOMINATES every other brightness effect" in section


def test_output_format_structured_lighting_examples_are_valid_actions():
    """A wrong example in the prompt is the `form_circle` bug again; render every one of them."""
    block = _choreographer().prompts["output_format_structured"]
    assert '"lighting" is a SEPARATE array' in block
    examples = [
        action
        for action in _json_objects_at(block, '{"primitive":')
        if action["primitive"] in LIGHTING_PRIMITIVE_ARG_ORDER
    ]
    assert {action["primitive"] for action in examples} >= {"light_color", "pulse", "chase"}
    for action in examples:
        assert action_to_lighting_primitive(action).startswith(action["primitive"] + "(")
