"""Structured output schema helpers for OpenAI Responses API."""

from __future__ import annotations

import json
from typing import Any

from swarm_gpt.exception import LLMFormatError

_AXIS_ENUM = ["x", "y", "z"]


def _int_schema(*, minimum: int | None = None, maximum: int | None = None) -> dict[str, Any]:
    schema: dict[str, Any] = {"type": "integer"}
    if minimum is not None:
        schema["minimum"] = minimum
    if maximum is not None:
        schema["maximum"] = maximum
    return schema


def _number_schema() -> dict[str, Any]:
    return {"type": "number"}


def _drone_ids_schema(num_drones: int) -> dict[str, Any]:
    return {
        "type": "array",
        "minItems": 1,
        "items": _int_schema(minimum=1, maximum=num_drones),
    }


def _array_schema(item_schema: dict[str, Any]) -> dict[str, Any]:
    return {"type": "array", "items": item_schema}


def _param_schemas(num_drones: int) -> dict[str, dict[str, Any]]:
    return {
        "x_cm": _number_schema(),
        "y_cm": _number_schema(),
        "z_cm": _number_schema(),
        "drone_id": _int_schema(minimum=1, maximum=num_drones),
        "angle_deg": _number_schema(),
        "axis": {"type": "string", "enum": _AXIS_ENUM},
        "drone_ids": _drone_ids_schema(num_drones),
        "drone_id_1": _int_schema(minimum=1, maximum=num_drones),
        "drone_id_2": _int_schema(minimum=1, maximum=num_drones),
        "delta_cm": _number_schema(),
        "steps": _int_schema(minimum=1),
        "height_cm": _number_schema(),
        "degrees": _number_schema(),
        "radius_increase": _number_schema(),
        "delta_height_cm": _number_schema(),
        "radius_cm": _number_schema(),
        "delta_xy_cm": _number_schema(),
        "delta_z_cm": _number_schema(),
        "mu_pairs": _array_schema(_array_schema(_number_schema())),
        "a_mu": _array_schema(_number_schema()),
        "b_mu": _array_schema(_number_schema()),
        "omega_times_ten": _number_schema(),
        "z_spacing_cm": _number_schema(),
        "min_spacing_cm": _number_schema(),
        "delta_radius_cm": _number_schema(),
        "spacing_cm": _number_schema(),
        "is_inverted": _int_schema(minimum=0, maximum=1),
    }


def _params_schema(num_drones: int, param_names: list[str]) -> dict[str, Any]:
    param_schemas = _param_schemas(num_drones)
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {name: param_schemas[name] for name in param_names},
        "required": param_names,
    }


def _action_variant_schema(primitive: str, num_drones: int) -> dict[str, Any]:
    param_names = [] if primitive == "PLAN" else _PRIMITIVE_ARG_ORDER[primitive]
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "primitive": {
                "type": "string",
                "enum": [primitive],
            },
            "params": _params_schema(num_drones, param_names),
        },
        "required": ["primitive", "params"],
    }


def _action_schema(num_drones: int) -> dict[str, Any]:
    return {
        "anyOf": [
            _action_variant_schema(primitive, num_drones)
            for primitive in ["PLAN", *_PRIMITIVE_ARG_ORDER.keys()]
        ]
    }


def build_motion_primitive_response_schema(*, num_beats: int, num_drones: int) -> dict[str, Any]:
    """Build a strict schema that enforces beat-exact motion-primitive outputs."""
    if num_beats < 1:
        raise ValueError("num_beats must be >= 1")
    if num_drones < 1:
        raise ValueError("num_drones must be >= 1")
    action_schema = {
        "type": "array",
        "minItems": 1,
        "items": _action_schema(num_drones),
    }
    beat_keys = [str(i) for i in range(1, num_beats + 1)]
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "song_mood": {"type": "string"},
            "cord_analysis": {"type": "string"},
            "choreography_plan": {"type": "string"},
            "choreography": {
                "type": "object",
                "additionalProperties": False,
                "properties": {beat_key: action_schema for beat_key in beat_keys},
                "required": beat_keys,
            },
        },
        "required": ["song_mood", "cord_analysis", "choreography_plan", "choreography"],
    }


_PRIMITIVE_ARG_ORDER: dict[str, list[str]] = {
    "move": ["x_cm", "y_cm", "z_cm", "drone_id"],
    "rotate": ["angle_deg", "axis"],
    "center": ["drone_ids"],
    "swap": ["drone_id_1", "drone_id_2"],
    "move_z": ["drone_ids", "delta_cm"],
    "spiral": ["steps", "height_cm"],
    "spiral_speed": ["steps", "height_cm", "degrees", "radius_increase"],
    "helix": ["steps", "delta_height_cm", "height_cm"],
    "form_circle": ["drone_ids", "radius_cm"],
    "zig_zag": ["steps", "delta_xy_cm", "delta_z_cm"],
    "wave": ["steps", "height_cm", "mu_pairs", "a_mu", "b_mu"],
    "twister": ["steps", "omega_times_ten", "z_spacing_cm"],
    "form_star": ["height_cm", "min_spacing_cm", "delta_radius_cm"],
    "form_cone": ["delta_height_cm", "spacing_cm", "is_inverted"],
}


def _python_literal(value: Any) -> str:
    if isinstance(value, bool):
        return str(int(value))
    if isinstance(value, str):
        return repr(value)
    return json.dumps(value)


def _args_from_params(primitive: str, params: Any) -> list[Any]:
    if not isinstance(params, dict):
        raise LLMFormatError(
            f"Params for primitive '{primitive}' must be an object, got {type(params).__name__}"
        )
    ordered_arg_names = _PRIMITIVE_ARG_ORDER[primitive]
    missing = [name for name in ordered_arg_names if name not in params]
    extras = [name for name in params if name not in ordered_arg_names]
    if missing or extras:
        details = []
        if missing:
            details.append(f"missing {missing}")
        if extras:
            details.append(f"unexpected {extras}")
        raise LLMFormatError(
            f"Primitive '{primitive}' params must be exactly {ordered_arg_names}; "
            + ", ".join(details)
        )
    return [params[name] for name in ordered_arg_names]


def action_to_motion_primitive(action: dict[str, Any]) -> str:
    """Convert one structured action object to legacy `primitive(args)` syntax."""
    if not isinstance(action, dict):
        raise LLMFormatError(
            f"Structured choreography action must be an object, got {type(action).__name__}"
        )
    primitive = action.get("primitive")
    if primitive == "PLAN":
        params = action.get("params", {})
        args = action.get("args", [])
        if params or args:
            raise LLMFormatError("PLAN does not accept params or args")
        return "PLAN"
    if primitive not in _PRIMITIVE_ARG_ORDER:
        raise LLMFormatError(f"Unknown motion primitive '{primitive}' in structured output")
    ordered_arg_names = _PRIMITIVE_ARG_ORDER[primitive]  # used for expected arity messaging
    if "params" in action:
        args = _args_from_params(primitive, action["params"])
    else:
        args = action.get("args", [])
        if not isinstance(args, list):
            raise LLMFormatError(f"Args for primitive '{primitive}' must be an array")
    if len(args) != len(ordered_arg_names):
        raise LLMFormatError(
            f"Primitive '{primitive}' expects {len(ordered_arg_names)} args "
            f"({ordered_arg_names}), got {len(args)} args: {args}"
        )
    if primitive in {"center", "move_z", "form_circle"}:
        drone_ids = args[0]
        if not isinstance(drone_ids, list):
            raise LLMFormatError(
                f"Args for primitive '{primitive}' require 'drone_ids' to be a list, got "
                f"{type(drone_ids).__name__}"
            )
        if len(set(drone_ids)) != len(drone_ids):
            raise LLMFormatError(
                f"Args for primitive '{primitive}' must have unique drone_ids, got {drone_ids}"
            )
    try:
        rendered_args = ", ".join(_python_literal(arg) for arg in args)
    except Exception as e:
        raise LLMFormatError(f"Could not serialize args for primitive '{primitive}': {e}") from e
    return f"{primitive}({rendered_args})"


def structured_payload_to_choreography(payload: dict[str, Any]) -> dict[int, str]:
    """Convert structured OpenAI payload to the existing choreography dictionary format."""
    choreography = payload.get("choreography", {})
    if not isinstance(choreography, dict):
        raise LLMFormatError("Structured output field 'choreography' must be an object")
    converted: dict[int, str] = {}
    for beat_text, actions in choreography.items():
        try:
            beat = int(beat_text)
        except (TypeError, ValueError) as e:
            raise LLMFormatError(f"Structured output beat key {beat_text!r} is not an integer") from e
        if not isinstance(actions, list) or len(actions) == 0:
            raise LLMFormatError(f"Structured output beat {beat} must include non-empty 'actions'")
        converted[beat] = "; ".join(action_to_motion_primitive(action) for action in actions)
    return converted
