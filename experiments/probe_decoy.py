"""Coverage test by revealed preference: offer primitives that do not exist and see what gets used.

Asking the model which motions it wished for measures its willingness to volunteer dissatisfaction,
and it does not volunteer any. Offering it a menu instead measures what it reaches for.

Two classes of decoy make the result interpretable:

- **gap** decoys do something the real library genuinely cannot (a heart, a two-strand helix, one
  subset orbiting another). Usage is evidence of unmet need.
- **redundant** decoys are renamed duplicates of primitives that already exist (``form_ring`` is
  ``form_circle``). Usage is evidence of nothing but novelty-seeking.

The gap rate is only meaningful against the redundant rate: if both are used equally the model is
just picking new names, and the library is not what binds.

    pixi run python experiments/probe_decoy.py --samples 2
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from swarm_gpt.core.choreographer import Choreographer
from swarm_gpt.core.lighting import load_lighting_config
from swarm_gpt.core.structured_output_schema import build_motion_primitive_response_schema
from swarm_gpt.utils.llm_providers import (
    openai_client_for_provider,
    prepare_responses_messages,
    responses_model_kwargs,
)
from swarm_gpt.utils.music_analyzer import SongStructure

logger = logging.getLogger("decoy")

ROOT = Path(__file__).resolve().parents[1]

# name -> (kind, params, prompt line). `params` are all plain numbers or the drone_ids spec, so a
# decoy needs no backend: nothing is ever executed, only counted.
MOTION_DECOYS: dict[str, tuple[str, list[tuple[str, str]], str]] = {
    "form_heart": (
        "gap",
        [
            ("drone_ids", "ids"),
            ("size_cm", "number"),
            ("z_coord_cm", "number"),
            ("time_to_finish_s", "number"),
        ],
        "form_heart(drone_ids, size_cm, z_coord_cm, time_to_finish_s) — arrange the subset into a "
        "heart outline standing upright in the x-z plane.",
    ),
    "form_letter": (
        "gap",
        [
            ("drone_ids", "ids"),
            ("letter_index", "number"),
            ("size_cm", "number"),
            ("z_coord_cm", "number"),
        ],
        "form_letter(drone_ids, letter_index, size_cm, z_coord_cm) — spell one glyph; "
        "letter_index 1-26 selects A-Z.",
    ),
    "double_helix": (
        "gap",
        [
            ("steps", "number"),
            ("radius_cm", "number"),
            ("delta_height_cm", "number"),
            ("height_cm", "number"),
        ],
        "double_helix(steps, radius_cm, delta_height_cm, height_cm) — two counter-rotating strands "
        "winding upward about a common axis, held opposite each other.",
    ),
    "orbit": (
        "gap",
        [
            ("drone_ids", "ids"),
            ("center_drone_ids", "ids"),
            ("radius_cm", "number"),
            ("degrees", "number"),
        ],
        "orbit(drone_ids, center_drone_ids, radius_cm, degrees) — the first subset circles the "
        "centroid of the second, which holds still.",
    ),
    "ripple": (
        "gap",
        [("source_drone_id", "number"), ("speed_cm_s", "number"), ("amplitude_cm", "number")],
        "ripple(source_drone_id, speed_cm_s, amplitude_cm) — a travelling disturbance spreading "
        "outward from one drone through the rest of the swarm.",
    ),
    "arc_swap": (
        "gap",
        [("drone_id_1", "number"), ("drone_id_2", "number"), ("arc_height_cm", "number")],
        "arc_swap(drone_id_1, drone_id_2, arc_height_cm) — two drones trade places along a visible "
        "arc instead of a straight line.",
    ),
    "split": (
        "gap",
        [
            ("drone_ids_a", "ids"),
            ("drone_ids_b", "ids"),
            ("separation_cm", "number"),
            ("time_to_finish_s", "number"),
        ],
        "split(drone_ids_a, drone_ids_b, separation_cm, time_to_finish_s) — the swarm parts into "
        "two groups drawing away from each other.",
    ),
    "bloom": (
        "gap",
        [
            ("drone_ids", "ids"),
            ("z_coord_cm", "number"),
            ("burst_radius_cm", "number"),
            ("time_to_finish_s", "number"),
        ],
        "bloom(drone_ids, z_coord_cm, burst_radius_cm, time_to_finish_s) — collapse to a point "
        "then burst outward, like a firework.",
    ),
    "form_ring": (
        "redundant",
        [
            ("drone_ids", "ids"),
            ("radius_cm", "number"),
            ("z_coord_cm", "number"),
            ("time_to_finish_s", "number"),
        ],
        "form_ring(drone_ids, radius_cm, z_coord_cm, time_to_finish_s) — place the subset evenly "
        "around a horizontal ring.",
    ),
    "ascend": (
        "redundant",
        [("drone_ids", "ids"), ("delta_cm", "number")],
        "ascend(drone_ids, delta_cm) — shift the subset vertically by delta_cm.",
    ),
    "turn": (
        "redundant",
        [("angle_deg", "number"), ("axis_index", "number")],
        "turn(angle_deg, axis_index) — rotate the whole swarm; axis_index 1/2/3 is x/y/z.",
    ),
    "corkscrew": (
        "redundant",
        [("steps", "number"), ("delta_height_cm", "number"), ("height_cm", "number")],
        "corkscrew(steps, delta_height_cm, height_cm) — the swarm circles a common axis while "
        "climbing.",
    ),
    "gather": (
        "redundant",
        [("drone_ids", "ids")],
        "gather(drone_ids) — bring the subset in around its own centroid.",
    ),
}

LIGHTING_DECOYS: dict[str, tuple[str, list[tuple[str, str]], str]] = {
    "twinkle": (
        "gap",
        [("sel", "sel"), ("density", "number"), ("period_beats", "number"), ("deck", "deck")],
        "twinkle(sel, density, period_beats, deck) — random drones sparkle in and out; density is "
        "the lit fraction.",
    ),
    "fade": (
        "gap",
        [
            ("sel", "sel"),
            ("color_a", "color"),
            ("color_b", "color"),
            ("duration_beats", "number"),
            ("deck", "deck"),
        ],
        "fade(sel, color_a, color_b, duration_beats, deck) — cross-fade the subset from one colour "
        "to another over time.",
    ),
    "color_wave_from": (
        "gap",
        [
            ("sel", "sel"),
            ("source_drone_id", "number"),
            ("color_a", "color"),
            ("color_b", "color"),
            ("period_beats", "number"),
            ("deck", "deck"),
        ],
        "color_wave_from(sel, source_drone_id, color_a, color_b, period_beats, deck) — a colour "
        "front travelling outward from one chosen drone.",
    ),
    "flash": (
        "redundant",
        [("sel", "sel"), ("period_beats", "number"), ("duty", "number"), ("deck", "deck")],
        "flash(sel, period_beats, duty, deck) — hard on/off cycling; duty is the lit fraction.",
    ),
    "hue_cycle": (
        "redundant",
        [("sel", "sel"), ("period_beats", "number"), ("deck", "deck")],
        "hue_cycle(sel, period_beats, deck) — run the subset through the colour wheel.",
    ),
    "breathe": (
        "redundant",
        [("sel", "sel"), ("period_beats", "number"), ("deck", "deck")],
        "breathe(sel, period_beats, deck) — the subset brightens and dims smoothly together.",
    ),
}


def _decoy_param_schema(kind: str, num_drones: int, palette: list[str]) -> dict[str, Any]:
    """Schema for one decoy parameter, mirroring the shapes the real schema uses."""
    if kind == "ids":
        return {"type": "string", "pattern": r"^\d+(-\d+)?(,\d+(-\d+)?)*$"}
    if kind == "color":
        return {"type": "string", "enum": palette}
    if kind == "deck":
        return {"type": "string", "enum": ["top", "bot", "both"]}
    if kind == "sel":
        return {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "kind": {
                    "type": "string",
                    "enum": [
                        "all",
                        "ids",
                        "even",
                        "odd",
                        "first",
                        "left",
                        "right",
                        "upper",
                        "lower",
                    ],
                },
                "ids": {"type": "array", "items": {"type": "integer"}},
                "count": {"type": "integer", "minimum": 1, "maximum": num_drones},
            },
            "required": ["kind", "ids", "count"],
        }
    return {"type": "number"}


def decoy_variant(
    name: str, params: list[tuple[str, str]], num_drones: int, palette: list[str]
) -> dict[str, Any]:
    """Build the ``{primitive, params}`` schema variant for one decoy."""
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "primitive": {"type": "string", "enum": [name]},
            "params": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    pname: _decoy_param_schema(pkind, num_drones, palette)
                    for pname, pkind in params
                },
                "required": [pname for pname, _ in params],
            },
        },
        "required": ["primitive", "params"],
    }


def inject(text: str, closing_tag: str, lines: list[str]) -> str:
    """Insert decoy description lines just before ``closing_tag`` in a prompt message.

    Raises:
        ValueError: If the tag is absent, which means the prompt moved and the probe is stale.
    """
    if closing_tag not in text:
        raise ValueError(f"Prompt no longer contains {closing_tag!r}; decoy probe is out of date")
    return text.replace(closing_tag, "\n".join([*lines, closing_tag]), 1)


def build_probe(
    choreographer: Choreographer, song: str, structure: Any, palette: list[str], seed: int
) -> tuple[list[dict], dict]:
    """Build the messages and schema for one probe, with decoys shuffled into both."""
    messages = [dict(m) for m in choreographer.format_initial_prompt(song, structure)]
    num_drones = choreographer.num_drones

    motion_order = list(MOTION_DECOYS)
    lighting_order = list(LIGHTING_DECOYS)
    # Shuffled so a decoy is not always last in the list; position bias would otherwise be
    # indistinguishable from preference.
    random.Random(seed).shuffle(motion_order)
    random.Random(seed + 1000).shuffle(lighting_order)

    # The YAML block scalar strips its base indentation, so the tags render flush left.
    motion_lines = [f"- {MOTION_DECOYS[n][2]}" for n in motion_order]
    lighting_lines = [f"- {LIGHTING_DECOYS[n][2]}" for n in lighting_order]
    for message in messages:
        if "</primitives>" in message["content"]:
            message["content"] = inject(message["content"], "</primitives>", motion_lines)
        if "</lighting>" in message["content"]:
            message["content"] = inject(message["content"], "</lighting>", lighting_lines)

    schema = build_motion_primitive_response_schema(
        all_keys=structure.all_keys(),
        required_keys=structure.required_keys(4),
        num_drones=num_drones,
    )
    for name in motion_order:
        _, params, _ = MOTION_DECOYS[name]
        schema["$defs"]["action"]["anyOf"].append(decoy_variant(name, params, num_drones, palette))
    for name in lighting_order:
        _, params, _ = LIGHTING_DECOYS[name]
        schema["$defs"]["lighting_action"]["anyOf"].append(
            decoy_variant(name, params, num_drones, palette)
        )
    return messages, schema


def count_usage(payload: dict) -> tuple[Counter, Counter, int, int]:
    """Count primitive usage across both tracks.

    Returns:
        ``(motion_counts, lighting_counts, n_motion_actions, n_lighting_actions)``.
    """
    motion, lighting = Counter(), Counter()
    for entry in payload.get("choreography", []):
        for action in entry.get("actions", []):
            motion[action.get("primitive")] += 1
    for entry in payload.get("lighting", []):
        for action in entry.get("actions", []):
            lighting[action.get("primitive")] += 1
    return motion, lighting, sum(motion.values()), sum(lighting.values())


def load_structure(song: str, settings: dict) -> SongStructure:
    """Load a song's cached analysis, cropped to the choreographed window."""
    crops = settings["song_crops"]
    start_s, end_s = crops.get(song, crops["default"])
    return SongStructure.from_json(ROOT / "music" / "analyzed" / f"{song}.json").crop(
        float(start_s), float(end_s)
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=2, help="Probes per song")
    parser.add_argument("--model", default="gpt-5.6-luna", help="LLM model id")
    parser.add_argument("--songs", nargs="*", help="Restrict to these songs")
    parser.add_argument("--out", type=Path, default=ROOT / "synth_runs", help="Log directory")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> Path:
    """Offer decoy primitives over the song corpus and report which ones get chosen.

    Returns:
        Path of the written report.
    """
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    settings = yaml.safe_load((ROOT / "swarm_gpt/data/settings.yaml").read_text())
    choreographer = Choreographer(model_id=args.model, use_motion_primitives=True)
    client = openai_client_for_provider("openai")
    palette = list(load_lighting_config().palette)

    songs = args.songs or sorted(p.stem for p in (ROOT / "music" / "analyzed").glob("*.json"))
    records = []
    for song in songs:
        structure = load_structure(song, settings)
        for sample in range(args.samples):
            messages, schema = build_probe(choreographer, song, structure, palette, sample)
            input_messages, instructions = prepare_responses_messages(messages)
            response = client.responses.create(
                model=args.model,
                input=input_messages,
                instructions=instructions,
                **responses_model_kwargs(args.model),
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "swarmgpt_choreography",
                        "schema": schema,
                        "strict": True,
                    }
                },
            )
            if response.error is not None:
                raise RuntimeError(f"Model errored: {response.error.message}")
            payload = json.loads(response.output_text)
            motion, lighting, n_m, n_l = count_usage(payload)
            used_gap = [n for n in motion if MOTION_DECOYS.get(n, ("",))[0] == "gap"]
            used_gap += [n for n in lighting if LIGHTING_DECOYS.get(n, ("",))[0] == "gap"]
            records.append(
                {
                    "song": song,
                    "sample": sample,
                    "motion": dict(motion),
                    "lighting": dict(lighting),
                    "n_motion": n_m,
                    "n_lighting": n_l,
                }
            )
            logger.info(
                "%-42s s%d: %d motion / %d lighting actions | gap decoys used: %s",
                song[:42],
                sample,
                n_m,
                n_l,
                ", ".join(sorted(set(used_gap))) or "-",
            )

    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"decoy_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    path.write_text(json.dumps({"model": args.model, "records": records}, indent=2))
    report(records)
    print(f"\nreport: {path}")
    return path


def report(records: list[dict]) -> None:
    """Print decoy usage, split by class, against the share of the menu each class occupies."""
    motion, lighting = Counter(), Counter()
    for r in records:
        motion.update(r["motion"])
        lighting.update(r["lighting"])

    for label, counts, decoys, n_real in (
        ("MOTION", motion, MOTION_DECOYS, 15),
        ("LIGHTING", lighting, LIGHTING_DECOYS, 12),
    ):
        total = sum(counts.values())
        gap = sum(c for n, c in counts.items() if decoys.get(n, ("",))[0] == "gap")
        red = sum(c for n, c in counts.items() if decoys.get(n, ("",))[0] == "redundant")
        n_gap = sum(1 for v in decoys.values() if v[0] == "gap")
        n_red = sum(1 for v in decoys.values() if v[0] == "redundant")
        menu = n_real + n_gap + n_red
        print(f"\n{label}: {total} actions over {len(records)} probes")
        print(
            f"  gap decoys       {gap:>4} ({gap / max(total, 1):>5.1%})  "
            f"menu share {n_gap / menu:>5.1%}  per-name {gap / max(n_gap, 1):>5.1f}"
        )
        print(
            f"  redundant decoys {red:>4} ({red / max(total, 1):>5.1%})  "
            f"menu share {n_red / menu:>5.1%}  per-name {red / max(n_red, 1):>5.1f}"
        )
        print(f"  real primitives  {total - gap - red:>4}")
        used = [(n, c) for n, c in counts.most_common() if n in decoys]
        if used:
            print("  by name:")
            for n, c in used:
                print(f"    {c:>4}  {n} ({decoys[n][0]})")


if __name__ == "__main__":
    main()
