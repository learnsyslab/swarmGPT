"""Structured output schema helpers for OpenAI Responses API.

Keys take the hierarchical form ``"s{seq}b{bar}t{beat}"`` (e.g. ``"s2b4t1"`` = segment 2,
bar 4, beat 1). The choreographer addresses moments at this granularity; the schema models
``choreography`` as an array of ``{"key", "actions"}`` entries, with ``key`` constrained to
an enum of every addressable beat. The LLM emits only the entries it wants; presence of the
required segment-opening keys is validated downstream.

``lighting`` is a second, independent array over the same address space (spec §10.1). It is
deliberately unconstrained: no required keys and no alternation rule, because LEDs have no
physical continuity constraint the way motion does.
"""

from __future__ import annotations

import json
import re
from typing import Any

from swarm_gpt.core.lighting import load_lighting_config
from swarm_gpt.core.lighting_primitives import LIGHTING_PRIMITIVES
from swarm_gpt.exception import LLMFormatError

_AXIS_ENUM = ["x", "y", "z"]
_KEY_PATTERN = r"s\d+b\d+t\d+"
_KEY_RE = re.compile(r"^s(\d+)b(\d+)t(\d+)$")

# The lighting vocabulary the LLM may say (spec §7.1, §7.3, §8.6, §10.2). Each list is exactly what
# the engine resolves: offering a name that `lighting.select` or `spread_offsets` would reject turns
# a schema-valid emission into a `KeyError` at compile time.
_DECK_ENUM = ["top", "bot", "both"]
_SELECTOR_KINDS = ["all", "ids", "even", "odd", "first", "left", "right"]
_SPREAD_ENUM = [
    "none",
    "neighbour",
    "index",
    "alternate_parity",
    "alternate_side",
    "radius",
    "x",
    "y",
    "z",
]
# `by` is the one parameter name two primitives disagree on: `gradient` interpolates along a
# spatial axis, `alternate_blink` splits the swarm two ways (§10.2). Resolved per primitive.
_LIGHTING_BY_ENUM = {
    "gradient": ["index", "x", "y", "z", "radius"],
    "alternate_blink": ["parity", "side"],
}


def encode_key(seq: int, bar: int, beat: int) -> str:
    """Encode a ``(segment, bar, beat)`` address as a structured-output key string.

    Args:
        seq: 1-indexed segment id.
        bar: 1-indexed bar id within the segment.
        beat: 1-indexed beat id within the bar.

    Returns:
        Key string in the form ``"s{seq}b{bar}t{beat}"``.
    """
    return f"s{seq}b{bar}t{beat}"


def decode_key(key: str) -> tuple[int, int, int]:
    """Decode a structured-output key string into ``(seq, bar, beat)``.

    Args:
        key: A string matching ``s<seq>b<bar>t<beat>``.

    Returns:
        ``(seq, bar, beat)`` as 1-indexed integers.

    Raises:
        LLMFormatError: If ``key`` does not match the expected pattern.
    """
    m = _KEY_RE.match(key)
    if m is None:
        raise LLMFormatError(f"Choreography key {key!r} is not in the form 's<seq>b<bar>t<beat>'")
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


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
    return {"type": "array", "minItems": 1, "items": _int_schema(minimum=1, maximum=num_drones)}


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
        "z_coord_cm": _number_schema(),
        "delta_xy_cm": _number_schema(),
        "delta_z_cm": _number_schema(),
        "omega_times_ten": _number_schema(),
        "z_spacing_cm": _number_schema(),
        "min_spacing_cm": _number_schema(),
        "delta_radius_cm": _number_schema(),
        "spacing_cm": _number_schema(),
        "is_inverted": _int_schema(minimum=0, maximum=1),
        "time_to_finish_s": _number_schema(),
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
    param_names = _PRIMITIVE_ARG_ORDER[primitive]
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "primitive": {"type": "string", "enum": [primitive]},
            "params": _params_schema(num_drones, param_names),
        },
        "required": ["primitive", "params"],
    }


def _action_schema(num_drones: int) -> dict[str, Any]:
    return {
        "anyOf": [
            _action_variant_schema(primitive, num_drones) for primitive in _PRIMITIVE_ARG_ORDER
        ]
    }


def _selector_schema(num_drones: int) -> dict[str, Any]:
    # Strict mode cannot express a variant by omission -- every declared property is required -- so
    # a selector carries all three fields and the unused ones are ignored: `ids` is read only when
    # `kind` is "ids", `count` only when it is "first" (§7.1). `ids` therefore takes no `minItems`,
    # so an empty list is the natural filler everywhere else.
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "kind": {"type": "string", "enum": _SELECTOR_KINDS},
            "ids": _array_schema(_int_schema(minimum=1, maximum=num_drones)),
            "count": _int_schema(minimum=1, maximum=num_drones),
        },
        "required": ["kind", "ids", "count"],
    }


def _lighting_param_schemas(num_drones: int) -> dict[str, dict[str, Any]]:
    # The colour vocabulary is exactly the shipped palette, so an invented colour cannot be
    # expressed (§10.2). Resolved here rather than at import: this module is imported widely, and
    # reading `lighting.toml` from disk at import time turns a malformed calibration file into an
    # import error, surfacing far from its cause and taking every importer down with it.
    palette = list(load_lighting_config().palette)
    return {
        "sel": _selector_schema(num_drones),
        "deck": {"type": "string", "enum": _DECK_ENUM},
        "color": {"type": "string", "enum": palette},
        "color_a": {"type": "string", "enum": palette},
        "color_b": {"type": "string", "enum": palette},
        # Periods are in beats, not seconds (§10.2); `build_look` converts them via the song BPM.
        "period_beats": {"type": "number", "exclusiveMinimum": 0},
        "duty": {"type": "number", "exclusiveMinimum": 0, "maximum": 1},
        "length": _int_schema(minimum=1, maximum=num_drones),
        "group_size": _int_schema(minimum=1, maximum=num_drones),
        "spread": {"type": "string", "enum": _SPREAD_ENUM},
        "axis": {"type": "string", "enum": _AXIS_ENUM},
    }


def _lighting_params_schema(primitive: str, num_drones: int) -> dict[str, Any]:
    param_names = _LIGHTING_PRIMITIVE_ARG_ORDER[primitive]
    param_schemas = _lighting_param_schemas(num_drones)
    if "by" in param_names:
        param_schemas["by"] = {"type": "string", "enum": _LIGHTING_BY_ENUM[primitive]}
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {name: param_schemas[name] for name in param_names},
        "required": param_names,
    }


def _lighting_action_variant_schema(primitive: str, num_drones: int) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "primitive": {"type": "string", "enum": [primitive]},
            "params": _lighting_params_schema(primitive, num_drones),
        },
        "required": ["primitive", "params"],
    }


def _lighting_action_schema(num_drones: int) -> dict[str, Any]:
    # Enumerated from the engine's own table so the schema cannot drift from it: a primitive added
    # to `LIGHTING_PRIMITIVES` without an entry here fails loudly at schema-build time.
    return {
        "anyOf": [
            _lighting_action_variant_schema(primitive, num_drones)
            for primitive in LIGHTING_PRIMITIVES
        ]
    }


def _key_track_schema(encoded_all: list[str], action_list_ref: str) -> dict[str, Any]:
    return {
        "type": "array",
        "items": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "key": {"type": "string", "enum": encoded_all},
                "actions": {"$ref": action_list_ref},
            },
            "required": ["key", "actions"],
        },
    }


def build_motion_primitive_response_schema(
    *,
    all_keys: list[tuple[int, int, int]],
    required_keys: list[tuple[int, int, int]],
    num_drones: int,
) -> dict[str, Any]:
    """Build a strict response schema keyed by hierarchical ``(seq, bar, beat)`` addresses.

    ``choreography`` is an array of ``{"key", "actions"}`` entries; ``key`` is constrained to
    an enum of every beat in ``all_keys``. OpenAI strict mode requires all object properties be
    required, so per-entry both fields are required; the LLM controls sparseness by emitting
    only the entries it wants. Presence of ``required_keys`` is validated downstream, not here.

    ``lighting`` is a second track of the same shape over the same key enum, carrying the §10.2
    lighting primitives instead of the motion ones. ``required_keys`` does not constrain it at
    all -- an empty ``lighting`` array is a complete answer (§10.1).

    Args:
        all_keys: Every addressable ``(seq, bar, beat)`` tuple in the song, in time order.
        required_keys: Subset of ``all_keys`` that the LLM must emit (segment openings). Applies
            to ``choreography`` only.
        num_drones: Number of drones in the swarm (constrains drone-id ranges).

    Returns:
        A JSON-Schema dict suitable for OpenAI Responses API ``response_format``.

    Raises:
        ValueError: If ``all_keys`` or ``num_drones`` is empty / non-positive, or if any
            entry in ``required_keys`` is missing from ``all_keys``.
    """
    if not all_keys:
        raise ValueError("all_keys must contain at least one (seq, bar, beat) entry")
    if num_drones < 1:
        raise ValueError("num_drones must be >= 1")
    encoded_all = [encode_key(*addr) for addr in all_keys]
    encoded_required = [encode_key(*addr) for addr in required_keys]
    all_set = set(encoded_all)
    missing = [k for k in encoded_required if k not in all_set]
    if missing:
        raise ValueError(f"required_keys not present in all_keys: {missing}")
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "song_mood": {"type": "string"},
            "choreography_plan": {"type": "string"},
            "choreography": _key_track_schema(encoded_all, "#/$defs/action_list"),
            "lighting": _key_track_schema(encoded_all, "#/$defs/lighting_action_list"),
        },
        # Strict mode requires every declared property, so `lighting` must be emitted -- but an
        # empty array satisfies it, which is what keeps lighting genuinely optional (§10.1).
        "required": ["song_mood", "choreography_plan", "choreography", "lighting"],
        "$defs": {
            "action": _action_schema(num_drones),
            "action_list": {"type": "array", "minItems": 1, "items": {"$ref": "#/$defs/action"}},
            "lighting_action": _lighting_action_schema(num_drones),
            "lighting_action_list": {
                "type": "array",
                "minItems": 1,
                "items": {"$ref": "#/$defs/lighting_action"},
            },
        },
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
    "form_circle": ["drone_ids", "radius_cm", "z_coord_cm", "time_to_finish_s"],
    "zig_zag": ["steps", "delta_xy_cm", "delta_z_cm"],
    "wave": ["steps", "height_cm"],
    "twister": ["steps", "omega_times_ten", "z_spacing_cm"],
    "form_star": ["height_cm", "min_spacing_cm", "delta_radius_cm", "time_to_finish_s"],
    "form_cone": ["delta_height_cm", "spacing_cm", "is_inverted", "time_to_finish_s"],
}

# The §10.2 catalogue's parameter order, which is also the rendered call's argument order: every
# lighting primitive takes `sel` first and `deck` last.
_LIGHTING_PRIMITIVE_ARG_ORDER: dict[str, list[str]] = {
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


def _python_literal(value: Any) -> str:
    if isinstance(value, bool):
        return str(int(value))
    if isinstance(value, str):
        return repr(value)
    return json.dumps(value)


def _args_from_params(primitive: str, params: Any, arg_order: dict[str, list[str]]) -> list[Any]:
    if not isinstance(params, dict):
        raise LLMFormatError(
            f"Params for primitive '{primitive}' must be an object, got {type(params).__name__}"
        )
    ordered_arg_names = arg_order[primitive]
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
    """Convert one structured action object to legacy ``primitive(args)`` syntax."""
    if not isinstance(action, dict):
        raise LLMFormatError(
            f"Structured choreography action must be an object, got {type(action).__name__}"
        )
    primitive = action.get("primitive")
    if primitive not in _PRIMITIVE_ARG_ORDER:
        raise LLMFormatError(f"Unknown motion primitive '{primitive}' in structured output")
    ordered_arg_names = _PRIMITIVE_ARG_ORDER[primitive]  # used for expected arity messaging
    if "params" in action:
        args = _args_from_params(primitive, action["params"], _PRIMITIVE_ARG_ORDER)
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


def _selector_literal(sel: Any) -> str:
    """Render a structured selector as the ``[kind, args]`` 2-sequence `lighting.select` consumes.

    A list rather than a tuple literal on purpose: the text form is parsed by splitting on the
    first ``(`` (``choreographer.py:719``), so a parenthesised argument would break that parser.
    ``ast.literal_eval`` returns the list, and ``kind, args = sel`` unpacks it unchanged.

    Args:
        sel: The selector object, ``{"kind", "ids", "count"}``. ``ids`` is read only when ``kind``
            is "ids", ``count`` only when it is "first"; the other fields are ignored (§7.1).

    Returns:
        The selector as a Python literal, e.g. ``"['ids', [1, 3, 5]]"``.

    Raises:
        LLMFormatError: If ``sel`` is not an object, or names a selector kind that does not exist.
    """
    if not isinstance(sel, dict):
        raise LLMFormatError(f"Lighting 'sel' must be an object, got {type(sel).__name__}")
    kind = sel["kind"]
    if kind == "ids":
        args = list(sel["ids"])
    elif kind == "first":
        args = [sel["count"]]
    elif kind in _SELECTOR_KINDS:
        args = []
    else:
        raise LLMFormatError(f"Unknown lighting selector kind '{kind}' in structured output")
    return repr([kind, args])


def action_to_lighting_primitive(action: dict[str, Any]) -> str:
    """Convert one structured lighting action to ``primitive(args)`` syntax (§10.2).

    The counterpart of `action_to_motion_primitive` for the lighting track. Arguments are rendered
    in the catalogue's order -- ``sel`` first, ``deck`` last -- so the call reads the way §10.2
    writes it.

    Args:
        action: One entry of a lighting key's ``actions`` array, ``{"primitive", "params"}``.

    Returns:
        The rendered call, e.g. ``"pulse(['ids', [1, 3, 5]], 2, 'both')"``.

    Raises:
        LLMFormatError: If the primitive is unknown, or its params are not exactly the ones §10.2
            gives it.
    """
    primitive = action.get("primitive")
    if primitive not in _LIGHTING_PRIMITIVE_ARG_ORDER:
        raise LLMFormatError(f"Unknown lighting primitive '{primitive}' in structured output")
    arg_names = _LIGHTING_PRIMITIVE_ARG_ORDER[primitive]
    args = _args_from_params(primitive, action.get("params"), _LIGHTING_PRIMITIVE_ARG_ORDER)
    rendered_args = ", ".join(
        _selector_literal(arg) if name == "sel" else _python_literal(arg)
        for name, arg in zip(arg_names, args)
    )
    return f"{primitive}({rendered_args})"


def structured_payload_to_choreography(payload: dict[str, Any]) -> dict[tuple[int, int, int], str]:
    """Convert a structured OpenAI payload to a ``(seq, bar, beat)``-keyed choreography dict.

    Args:
        payload: The structured-output payload from the LLM.

    Returns:
        Dict mapping ``(seq, bar, beat)`` tuples to action strings (one or more primitive
        calls separated by ``"; "``).

    Raises:
        LLMFormatError: If the payload is malformed (wrong field types, unknown keys,
            empty action lists, duplicate keys, etc.).
    """
    choreography = payload.get("choreography", [])
    if not isinstance(choreography, list):
        raise LLMFormatError("Structured output field 'choreography' must be an array")
    converted: dict[tuple[int, int, int], str] = {}
    for entry in choreography:
        if not isinstance(entry, dict):
            raise LLMFormatError(
                "Each choreography entry must be an object with 'key' and 'actions'"
            )
        key = entry.get("key")
        actions = entry.get("actions")
        if not isinstance(key, str):
            raise LLMFormatError("Choreography entry 'key' must be a string")
        addr = decode_key(key)
        if addr in converted:
            raise LLMFormatError(f"Duplicate choreography key {key!r}")
        if not isinstance(actions, list) or len(actions) == 0:
            raise LLMFormatError(
                f"Structured output beat {key!r} must include a non-empty action list"
            )
        converted[addr] = "; ".join(action_to_motion_primitive(action) for action in actions)
    return converted


def structured_payload_to_lighting(payload: dict[str, Any]) -> dict[tuple[int, int, int], str]:
    """Convert a structured payload's lighting track to a ``(seq, bar, beat)``-keyed dict.

    The sibling of `structured_payload_to_choreography` for the second track. A *missing*
    ``lighting`` key is not an error: strict mode forces the field to be required, so a model with
    nothing to say emits ``[]`` rather than omitting it — but payloads predating this feature,
    including this repo's own preset fixtures, carry no key at all and must still load (§10.1).

    Args:
        payload: The structured-output payload from the LLM.

    Returns:
        Dict mapping ``(seq, bar, beat)`` tuples to action strings (one or more lighting
        primitive calls separated by ``"; "``).

    Raises:
        LLMFormatError: If the track is malformed — wrong field types, an unknown primitive,
            duplicate keys, or an empty action list.
    """
    lighting = payload.get("lighting", [])
    if not isinstance(lighting, list):
        raise LLMFormatError("Structured output field 'lighting' must be an array")
    converted: dict[tuple[int, int, int], str] = {}
    for entry in lighting:
        if not isinstance(entry, dict):
            raise LLMFormatError("Each lighting entry must be an object with 'key' and 'actions'")
        key = entry.get("key")
        actions = entry.get("actions")
        if not isinstance(key, str):
            raise LLMFormatError("Lighting entry 'key' must be a string")
        addr = decode_key(key)
        if addr in converted:
            raise LLMFormatError(f"Duplicate lighting key {key!r}")
        if not isinstance(actions, list) or len(actions) == 0:
            raise LLMFormatError(f"Lighting key {key!r} must include a non-empty action list")
        converted[addr] = "; ".join(action_to_lighting_primitive(action) for action in actions)
    return converted


# Re-exported for callers that want to validate raw key strings without decoding.
KEY_PATTERN = _KEY_PATTERN

# Re-exported for the lighting text parser, which zips a parsed `primitive(args)` call back into
# the named `params` dict `build_look` consumes.
LIGHTING_PRIMITIVE_ARG_ORDER = _LIGHTING_PRIMITIVE_ARG_ORDER
