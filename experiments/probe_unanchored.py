"""Measure how much showing the primitive vocabulary shrinks the choreography the model asks for.

The decoy probe measures what the model *uses if offered*, which cannot separate need from the
attraction of a menu item. This measures the opposite direction: elicit intent with no vocabulary
in the prompt at all, then have a separate judge decide what of it the real library can express.

Three properties keep it honest:

- **Single-variable A/B.** ``blind`` and ``anchored`` prompts are byte-identical except that
  ``anchored`` carries the production ``<primitives>`` and ``<lighting>`` blocks. The production
  prompt also carries pages of selection guidance; including that would confound vocabulary with
  advice, so it is deliberately left out of both.
- **The planner never judges itself.** Expressibility is labelled by a separate call that sees one
  intent and the library, and never the song, the plan around it, or which condition produced it.
  Asking the planner is what failed in the introspective probe.
- **The headline number is the difference between conditions**, not either level, because an
  absolute "N% inexpressible" figure inherits whatever bias the judge has.

    pixi run python experiments/probe_unanchored.py --songs "Fearless2" "Mortals"
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import toml
import yaml

from swarm_gpt.core.choreographer import _render_segments_table
from swarm_gpt.utils.llm_providers import (
    openai_client_for_provider,
    prepare_responses_messages,
    responses_model_kwargs,
)
from swarm_gpt.utils.music_analyzer import SongStructure

# `experiments` is not a package, so the sibling module is imported by path.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from vocabularies import VARIANTS, vocabulary  # noqa: E402

logger = logging.getLogger("unanchored")

ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = ("blind", "anchored")
VERDICTS = ("expressible", "partial", "not_expressible")

_PLAN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["plan_summary", "intents"],
    "properties": {
        "plan_summary": {"type": "string"},
        "intents": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["key", "motion", "lighting"],
                "properties": {
                    "key": {"type": "string"},
                    "motion": {"type": "string"},
                    "lighting": {"type": "string"},
                },
            },
        },
    },
}

_JUDGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["verdict", "primitives_used", "missing"],
    "properties": {
        "verdict": {"type": "string", "enum": list(VERDICTS)},
        "primitives_used": {"type": "string"},
        "missing": {"type": "string"},
    },
}

_PLAN_PROMPT = """\
You are choreographing a drone light show for "{song}" at {bpm} BPM, with {n_drones} drones.

The flying volume is x and y in [{x_lo:.1f}, {x_hi:.1f}] m and z in [{z_lo:.2f}, {z_hi:.2f}] m.

Song structure, addressed as s<segment>b<bar>t<beat>:
{segments}

Beats you should cover: {keys}

Describe the show you want. For each moment give the motion you want the swarm to make and the
lighting you want on it, in plain language. Describe the effect you are actually after -- do not
water it down toward what you assume is easy to build, and do not describe implementation.
{vocabulary}"""

_VOCAB_PREAMBLE = """
The system that will execute your plan offers exactly these primitives:

{primitives}

{lighting}"""

_JUDGE_PROMPT = """\
Below is a drone-show motion and lighting intent, and the complete list of primitives a drone show
system offers. Decide whether the intent can be realised with those primitives.

Answer "expressible" if the primitives produce the described effect, alone or in combination.
Answer "partial" if the closest achievable version loses something an audience would notice.
Answer "not_expressible" if no combination produces the effect at all.

Judge only what the primitives can do. Do not credit a primitive with behaviour it does not have,
and do not penalise an intent for being ambitious if the primitives cover it.

INTENT
  motion:   {motion}
  lighting: {lighting}

AVAILABLE PRIMITIVES
{primitives}

{lighting_primitives}"""


def vocabulary_blocks(variant: str = "current") -> tuple[str, str]:
    """Return the ``(primitives, lighting)`` prompt blocks for a vocabulary variant.

    Delegates to `scripts.vocabularies`, which derives each variant from the prompt file or from
    ``blocks.py`` on the spline branch rather than restating it, so the vocabulary the probe shows
    and the vocabulary the system has cannot drift apart.
    """
    return vocabulary(variant)


def load_structure(song: str, settings: dict) -> SongStructure:
    """Load a song's cached analysis, cropped to the choreographed window."""
    crops = settings["song_crops"]
    start_s, end_s = crops.get(song, crops["default"])
    return SongStructure.from_json(ROOT / "music" / "analyzed" / f"{song}.json").crop(
        float(start_s), float(end_s)
    )


def call_structured(client: Any, model: str, prompt: str, schema: dict, name: str) -> dict:
    """Send one user-message prompt under a strict schema and return the parsed payload."""
    messages = [{"role": "user", "content": prompt}]
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


def plan_prompt(
    song: str,
    structure: SongStructure,
    settings: dict,
    n_drones: int,
    condition: str,
    variant: str = "current",
) -> str:
    """Build the elicitation prompt for one condition."""
    lo = settings["axswarm"]["pos_min"]
    hi = settings["axswarm"]["pos_max"]
    keys = ", ".join(f"s{s}b{b}t{t}" for s, b, t in structure.required_keys(4))
    vocabulary = ""
    if condition == "anchored":
        motion_block, lighting_block = vocabulary_blocks(variant)
        vocabulary = _VOCAB_PREAMBLE.format(primitives=motion_block, lighting=lighting_block)
    return _PLAN_PROMPT.format(
        song=song,
        bpm=structure.bpm,
        n_drones=n_drones,
        x_lo=lo[0],
        x_hi=hi[0],
        z_lo=lo[2],
        z_hi=hi[2],
        segments=_render_segments_table(structure),
        keys=keys,
        vocabulary=vocabulary,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--songs", nargs="*", help="Restrict to these songs")
    parser.add_argument("--samples", type=int, default=1, help="Plans per song per condition")
    parser.add_argument("--model", default="gpt-5.6-luna", help="Planner model")
    parser.add_argument("--judge-model", default="gpt-5.6-luna", help="Expressibility judge model")
    parser.add_argument("--max-intents", type=int, default=12, help="Intents judged per plan")
    parser.add_argument(
        "--vocabulary",
        choices=VARIANTS,
        default="current",
        help="Which primitive library to judge against",
    )
    parser.add_argument("--out", type=Path, default=ROOT / "synth_runs", help="Log directory")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> Path:
    """Elicit plans in both conditions, judge every intent, and report the anchoring delta.

    Returns:
        Path of the written report.
    """
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    settings = yaml.safe_load((ROOT / "swarm_gpt/data/settings.yaml").read_text())
    n_drones = len(toml.load(ROOT / "swarm_gpt/data/drones.toml")["active"])
    motion_block, lighting_block = vocabulary_blocks(args.vocabulary)
    client = openai_client_for_provider("openai")

    songs = args.songs or sorted(p.stem for p in (ROOT / "music" / "analyzed").glob("*.json"))
    plans, judged = [], []
    for song in songs:
        structure = load_structure(song, settings)
        for condition in CONDITIONS:
            for sample in range(args.samples):
                prompt = plan_prompt(
                    song, structure, settings, n_drones, condition, args.vocabulary
                )
                plan = call_structured(client, args.model, prompt, _PLAN_SCHEMA, "show_plan")
                intents = plan["intents"][: args.max_intents]
                plans.append({"song": song, "condition": condition, "sample": sample, **plan})
                for intent in intents:
                    verdict = call_structured(
                        client,
                        args.judge_model,
                        _JUDGE_PROMPT.format(
                            motion=intent["motion"],
                            lighting=intent["lighting"],
                            primitives=motion_block,
                            lighting_primitives=lighting_block,
                        ),
                        _JUDGE_SCHEMA,
                        "expressibility",
                    )
                    judged.append({"song": song, "condition": condition, **intent, **verdict})
                counts = Counter(
                    j["verdict"]
                    for j in judged
                    if j["song"] == song and j["condition"] == condition
                )
                logger.info(
                    "%-34s %-9s %d intents | expressible %d partial %d NOT %d",
                    song[:34],
                    condition,
                    len(intents),
                    counts["expressible"],
                    counts["partial"],
                    counts["not_expressible"],
                )

    args.out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = args.out / f"unanchored_{args.vocabulary}_{stamp}.json"
    path.write_text(json.dumps({"model": args.model, "plans": plans, "judged": judged}, indent=2))
    report(judged)
    print(f"\nreport: {path}")
    return path


def report(judged: list[dict]) -> None:
    """Print per-condition verdict shares; the delta between them is the headline."""
    print(f"\n{'condition':<11}{'intents':<9}{'expressible':<14}{'partial':<11}not_expressible")
    shares = {}
    for condition in CONDITIONS:
        cell = [j for j in judged if j["condition"] == condition]
        if not cell:
            continue
        counts = Counter(j["verdict"] for j in cell)
        n = len(cell)
        shares[condition] = {v: counts[v] / n for v in VERDICTS}
        print(
            f"{condition:<11}{n:<9}"
            + "".join(f"{counts[v] / n:>6.0%} ({counts[v]:>3})  " for v in VERDICTS)
        )
    if len(shares) == 2:
        # The judge reaches for "partial" far more readily than "not_expressible", so their sum is
        # the measure of the library falling short; the split between them is judge temperament.
        for verdict in ("partial", "not_expressible"):
            delta = shares["blind"][verdict] - shares["anchored"][verdict]
            print(f"anchoring delta on {verdict:<16}: {delta:+.1%}")
        shortfall = {c: shares[c]["partial"] + shares[c]["not_expressible"] for c in CONDITIONS}
        print(
            f"\nSHORTFALL (partial + not_expressible): blind {shortfall['blind']:.0%} vs "
            f"anchored {shortfall['anchored']:.0%}  ->  delta {shortfall['blind'] - shortfall['anchored']:+.1%}"
        )

    print("\nwhat the judge said the library could not deliver, from BLIND plans:")
    for j in judged:
        if j["condition"] == "blind" and j["verdict"] != "expressible":
            print(f"  [{j['verdict']}] {j['motion'][:86]}")
            print(f"      missing: {j['missing'][:86]}")


if __name__ == "__main__":
    main()
