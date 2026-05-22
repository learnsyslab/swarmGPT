import json
from types import SimpleNamespace
from typing import Any

import pytest
from conftest import virtual_crazyswarm_config

from swarm_gpt.core.choreographer import Choreographer
from swarm_gpt.core.structured_output_schema import build_motion_primitive_response_schema
from swarm_gpt.exception import LLMFormatError
from swarm_gpt.utils.llm_providers import RESPONSES_TEMPERATURE


def _contains_one_of(node: Any) -> bool:
    if isinstance(node, dict):
        if "oneOf" in node:
            return True
        return any(_contains_one_of(value) for value in node.values())
    if isinstance(node, list):
        return any(_contains_one_of(value) for value in node)
    return False


def test_build_motion_primitive_response_schema_enforces_exact_num_beats():
    schema = build_motion_primitive_response_schema(num_beats=3, num_drones=5)
    choreography = schema["properties"]["choreography"]

    assert choreography["type"] == "object"
    assert choreography["required"] == ["1", "2", "3"]
    assert set(choreography["properties"]) == {"1", "2", "3"}
    assert choreography["additionalProperties"] is False
    assert not _contains_one_of(schema)


def test_structured_payload_to_choreography_preserves_plan_and_action_order():
    config_path = virtual_crazyswarm_config(n_drones=4)
    choreographer = Choreographer(
        config_file=config_path,
        llm_provider="openai",
        use_motion_primitives=True,
    )
    payload = {
        "song_mood": "calm",
        "cord_analysis": "minor",
        "choreography_plan": "simple",
        "choreography": {
            "1": [
                {"primitive": "form_circle", "args": [[1, 2], 100]},
            ],
            "2": [{"primitive": "PLAN", "args": []}],
            "3": [
                {"primitive": "rotate", "args": [90, "z"]},
                {"primitive": "move_z", "args": [[1], 10]},
            ],
        },
    }

    choreography = choreographer._structured_payload_to_choreography(payload)

    assert choreography == {
        1: "form_circle([1, 2], 100)",
        2: "PLAN",
        3: "rotate(90, 'z'); move_z([1], 10)",
    }


def test_call_responses_structured_includes_json_schema_format():
    config_path = virtual_crazyswarm_config(n_drones=4)
    choreographer = Choreographer(
        config_file=config_path,
        llm_provider="openai",
        use_motion_primitives=True,
    )
    captured: dict[str, Any] = {}

    class FakeResponses:
        def create(self, **kwargs: Any) -> SimpleNamespace:
            captured.update(kwargs)
            payload = {
                "song_mood": "energetic",
                "cord_analysis": "major",
                "choreography_plan": "test",
                "choreography": {"1": [{"primitive": "PLAN", "args": []}]},
            }
            return SimpleNamespace(error=None, output_text=json.dumps(payload))

    class FakeClient:
        responses = FakeResponses()

    choreographer._chat_client_for_call = lambda: FakeClient()  # noqa: E731
    messages = [{"role": "user", "content": "hello"}]

    parsed = choreographer._call_responses_structured(messages, num_beats=1)

    assert "1" in parsed["choreography"]
    assert captured["text"]["format"]["type"] == "json_schema"
    assert captured["text"]["format"]["name"] == "swarmgpt_choreography"
    assert captured["text"]["format"]["strict"] is True
    assert captured["text"]["format"]["schema"]["properties"]["choreography"]["required"] == ["1"]


def test_schema_contains_no_openai_unsupported_keywords():
    schema = build_motion_primitive_response_schema(num_beats=2, num_drones=4)
    schema_text = json.dumps(schema)
    assert "oneOf" not in schema_text
    assert "uniqueItems" not in schema_text
    assert '"items": false' not in schema_text.lower()


def test_ollama_motion_primitives_uses_structured_outputs():
    config_path = virtual_crazyswarm_config(n_drones=4)
    choreographer = Choreographer(
        config_file=config_path,
        llm_provider="ollama",
        use_motion_primitives=True,
    )
    assert choreographer._uses_structured_outputs() is True


def test_call_responses_structured_ollama_uses_native_chat(monkeypatch: pytest.MonkeyPatch):
    config_path = virtual_crazyswarm_config(n_drones=4)
    choreographer = Choreographer(
        config_file=config_path,
        llm_provider="ollama",
        use_motion_primitives=True,
    )
    captured: dict[str, Any] = {}

    def fake_ollama_chat(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        payload = {
            "song_mood": "energetic",
            "cord_analysis": "major",
            "choreography_plan": "test",
            "choreography": {"1": [{"primitive": "PLAN", "args": []}]},
        }
        return {"message": {"content": json.dumps(payload)}}

    monkeypatch.setattr("swarm_gpt.core.choreographer.ollama_chat", fake_ollama_chat)
    parsed = choreographer._call_responses_structured([{"role": "user", "content": "hello"}], 1)

    assert parsed["choreography"]["1"][0]["primitive"] == "PLAN"
    assert captured["model"] == choreographer.model_id
    assert captured["format"]["properties"]["choreography"]["required"] == ["1"]
    assert "JSON schema exactly" in captured["messages"][-1]["content"]
    assert captured["options"] == {"temperature": RESPONSES_TEMPERATURE}


def test_generate_choreography_ollama_raises_on_structured_errors(
    monkeypatch: pytest.MonkeyPatch,
):
    config_path = virtual_crazyswarm_config(n_drones=4)
    choreographer = Choreographer(
        config_file=config_path,
        llm_provider="ollama",
        use_motion_primitives=True,
    )
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
            prompt=[{"role": "user", "content": "hello"}],
            num_beats=1,
        )


def test_generate_choreography_ollama_raises_when_structured_payload_incomplete(
    monkeypatch: pytest.MonkeyPatch,
):
    config_path = virtual_crazyswarm_config(n_drones=4)
    choreographer = Choreographer(
        config_file=config_path,
        llm_provider="ollama",
        use_motion_primitives=True,
    )
    monkeypatch.setattr(
        choreographer,
        "_call_responses_structured",
        lambda *_args, **_kwargs: {"choreography": {"1": [{"primitive": "PLAN", "args": []}]}},
    )
    with pytest.raises(LLMFormatError, match="missing required keys"):
        choreographer.generate_choreography(
            prompt=[{"role": "user", "content": "hello"}],
            num_beats=1,
        )
