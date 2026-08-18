"""Kill-test: does the 12-primitive library actually bind on real songs?

Runs the normal choreography prompt over the analyzed song corpus, then asks one extra
unconstrained question -- which moments called for motion the library cannot express -- and
clusters the answers to see whether the same gap recurs across songs. Recurrence is what makes a
gap primitive-shaped rather than one-off noise.

    pixi run python experiments/probe_introspective.py --samples 2

The question is leading by construction, so the design pushes back on confabulation: an empty
answer is explicitly allowed and encouraged, every gap must name the nearest primitive it tried,
and each is graded so "I approximated it fine" can be separated from a real shortfall.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from swarm_gpt.core.choreographer import Choreographer
from swarm_gpt.utils.llm_providers import (
    openai_client_for_provider,
    prepare_responses_messages,
    responses_model_kwargs,
)
from swarm_gpt.utils.music_analyzer import SongStructure

logger = logging.getLogger("coverage")

ROOT = Path(__file__).resolve().parents[1]
SHORTFALL = ("none", "minor", "major")

_WISH_PROMPT = """\
Do not write a choreography. Answer one question about the primitive library you were just given.

Go through this song's segments as if you were choreographing it. At each moment, ask whether the
motion the music called for was expressible with the primitives above.

Report ONLY moments where the library genuinely could not express what the music wanted. For each,
name the nearest primitive you would have reached for and say specifically why it falls short, then
grade the shortfall:
  - "none"  -- the nearest primitive covers it; do not report these at all
  - "minor" -- an approximation exists and would read acceptably to an audience
  - "major" -- no combination of the primitives produces the effect

If the library covered everything this song asked for, return an empty list. An empty list is a
perfectly good answer and is expected for many songs. Do not invent gaps to fill the list, and do
not report a gap that is really about parameter ranges rather than a missing motion."""

_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["gaps", "summary"],
    "properties": {
        "summary": {"type": "string"},
        "gaps": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["key", "wanted", "why_not_expressible", "nearest_primitive", "grade"],
                "properties": {
                    "key": {"type": "string"},
                    "wanted": {"type": "string"},
                    "why_not_expressible": {"type": "string"},
                    "nearest_primitive": {"type": "string"},
                    "grade": {"type": "string", "enum": list(SHORTFALL)},
                },
            },
        },
    },
}

_CLUSTER_PROMPT = """\
Below are motion descriptions collected from several different songs, each a moment where a drone
choreographer wanted a motion its primitive library could not express.

Group them into recurring categories of missing motion. A category must describe a motion, not a
mood. Merge descriptions that are the same motion in different words. Leave a description in a
category of its own if nothing else matches it -- do not force merges to make the list tidy.

For each category give a short name, a one-line description of the motion, and the list of input
indices it covers."""

_CLUSTER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["categories"],
    "properties": {
        "categories": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["name", "description", "indices"],
                "properties": {
                    "name": {"type": "string"},
                    "description": {"type": "string"},
                    "indices": {"type": "array", "items": {"type": "integer"}},
                },
            },
        }
    },
}


def call_structured(client: Any, model: str, messages: list[dict], schema: dict, name: str) -> dict:
    """Send ``messages`` under a strict JSON schema and return the parsed payload."""
    input_messages, instructions = prepare_responses_messages(messages)
    response = client.responses.create(
        model=model,
        input=input_messages,
        instructions=instructions,
        **responses_model_kwargs(model),
        text={"format": {"type": "json_schema", "name": name, "schema": schema, "strict": True}},
    )
    if response.error is not None:
        raise RuntimeError(f"Model {model!r} errored: {response.error.message}")
    return json.loads(response.output_text)


def load_structure(song: str, settings: dict) -> SongStructure:
    """Load a song's cached analysis, cropped to the window that is actually choreographed."""
    crops = settings["song_crops"]
    start_s, end_s = crops.get(song, crops["default"])
    path = ROOT / "music" / "analyzed" / f"{song}.json"
    return SongStructure.from_json(path).crop(float(start_s), float(end_s))


def probe_song(client: Any, model: str, choreographer: Choreographer, song: str, structure: Any):
    """Ask one song's worth of the wish question, returning the parsed payload."""
    messages = [
        *choreographer.format_initial_prompt(song, structure),
        {"role": "user", "content": _WISH_PROMPT},
    ]
    return call_structured(client, model, messages, _RESPONSE_SCHEMA, "coverage_gaps")


def cluster_gaps(client: Any, model: str, gaps: list[dict]) -> list[dict]:
    """Group collected gap descriptions into recurring categories of missing motion."""
    if not gaps:
        return []
    listing = "\n".join(
        f"{i}. [{g['song']}] {g['wanted']} (nearest: {g['nearest_primitive']}; {g['grade']})"
        for i, g in enumerate(gaps)
    )
    payload = call_structured(
        client,
        model,
        [{"role": "user", "content": f"{_CLUSTER_PROMPT}\n\n{listing}"}],
        _CLUSTER_SCHEMA,
        "gap_categories",
    )
    return payload["categories"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=2, help="Probes per song")
    parser.add_argument("--model", default="gpt-5.6-luna", help="LLM model id")
    parser.add_argument("--songs", nargs="*", help="Restrict to these songs")
    parser.add_argument("--out", type=Path, default=ROOT / "synth_runs", help="Log directory")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> Path:
    """Probe every analyzed song, cluster the gaps, and report recurrence.

    Returns:
        Path of the written report.
    """
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    settings = yaml.safe_load((ROOT / "swarm_gpt/data/settings.yaml").read_text())
    choreographer = Choreographer(model_id=args.model, use_motion_primitives=True)
    client = openai_client_for_provider("openai")

    songs = args.songs or sorted(p.stem for p in (ROOT / "music" / "analyzed").glob("*.json"))
    records, gaps = [], []
    for song in songs:
        structure = load_structure(song, settings)
        for sample in range(args.samples):
            payload = probe_song(client, args.model, choreographer, song, structure)
            reported = [g for g in payload["gaps"] if g["grade"] != "none"]
            records.append({"song": song, "sample": sample, **payload})
            gaps.extend({"song": song, "sample": sample, **g} for g in reported)
            logger.info(
                "%-42s sample %d: %d gap(s) [%s]",
                song[:42],
                sample,
                len(reported),
                ", ".join(g["grade"] for g in reported) or "-",
            )

    categories = cluster_gaps(client, args.model, gaps)
    for category in categories:
        songs_hit = {gaps[i]["song"] for i in category["indices"] if i < len(gaps)}
        category["n_songs"] = len(songs_hit)
        category["songs"] = sorted(songs_hit)
    categories.sort(key=lambda c: (-c["n_songs"], -len(c["indices"])))

    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"coverage_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    report = {
        "model": args.model,
        "n_songs": len(songs),
        "samples_per_song": args.samples,
        "records": records,
        "gaps": gaps,
        "categories": categories,
    }
    path.write_text(json.dumps(report, indent=2))

    n_probes = len(records)
    empty = sum(1 for r in records if not [g for g in r["gaps"] if g["grade"] != "none"])
    grades = Counter(g["grade"] for g in gaps)
    per_song = defaultdict(int)
    for g in gaps:
        per_song[g["song"]] += 1

    print(f"\n{n_probes} probes over {len(songs)} songs, {args.samples} each")
    print(f"empty answers: {empty}/{n_probes}  ({empty / n_probes:.0%})")
    print(f"gaps: {len(gaps)} total, {dict(grades)}")
    print(f"songs with at least one gap: {len(per_song)}/{len(songs)}")
    print(f"\n{'n_songs':<9}{'n_gaps':<8}category")
    for category in categories:
        print(f"{category['n_songs']:<9}{len(category['indices']):<8}{category['name']}")
    print(f"\nreport: {path}")
    return path


if __name__ == "__main__":
    main()
