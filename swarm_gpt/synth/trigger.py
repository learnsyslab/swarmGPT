"""Decide whether one refinement request needs a primitive the library does not have.

Asking the choreographer what it is missing does not work -- 26 introspective probes returned 26
empty answers, because it rationalises coverage rather than reporting a gap. This asks a narrower
question about one concrete request against one concrete list, which is a judgement rather than an
act of self-examination.

The catalogue is built from the arity table and the synthesized registry, the same two structures
the response schema is built from, so what the classifier is told exists is what actually exists.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any

import openai

from swarm_gpt.core.structured_output_schema import primitive_exists, primitive_signatures
from swarm_gpt.utils.llm_providers import (
    openai_client_for_provider,
    prepare_responses_messages,
    responses_model_kwargs,
)

logger = logging.getLogger(__name__)

_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]{2,39}$")

# This call sits in front of every refinement and answers one yes/no question, so it is bounded
# tightly: a slow API must not hold a refine open. Observed at a few seconds; seen blocking for
# 25 minutes when the API degraded.
_TIMEOUT_S = 90.0

_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["needs_new", "name", "request", "reasoning"],
    "properties": {
        "needs_new": {"type": "boolean"},
        "name": {"type": "string"},
        "request": {"type": "string"},
        "reasoning": {"type": "string"},
    },
}

_SYSTEM = """\
You decide whether a drone-show choreographer can carry out a request with the primitives it has.

These are every motion primitive that exists. There are no others, and the choreographer can only
emit calls to these:

{catalogue}

The choreographer composes them over time: it can call several at one moment, apply them to
subsets of drones, and follow one with another. Lighting is a separate track and is never a reason
to author a motion primitive.

Answer `needs_new` true only when the request asks for a specific spatial arrangement or motion
that no combination of the listed primitives produces -- a named shape the list does not contain,
for instance. Answer false when the request is about timing, energy, colour, which drones are
involved, or a shape the list already covers under another name.

When it is true, give `name` as a snake_case identifier for the primitive that is missing, and
`request` as one self-contained sentence describing the geometry to build, written for someone who
has not seen the user's message. State the shape, its orientation, and its extent. Do not mention
the song, the moment in the show, or the colour. When it is false, leave `name` and `request` empty.
"""

_USER = """\
The user asked the choreographer for this change:

{message}

Does carrying it out need a motion primitive that is not on the list?"""


@dataclass(frozen=True)
class Gap:
    """A primitive the library lacks, named and specified well enough to synthesize."""

    name: str
    request: str
    reasoning: str


def catalogue() -> str:
    """Render every motion primitive that currently exists, one signature per line."""
    return "\n".join(f"- {line}" for line in primitive_signatures())


def classify_gap(message: str, *, model_id: str, llm_provider: str = "openai") -> Gap | None:
    """Return the missing primitive one refinement request implies, or None if the library covers it.

    Returns:
        A ``Gap`` when a primitive must be authored, otherwise ``None``.
    """
    client = openai_client_for_provider(llm_provider).with_options(max_retries=1)
    messages = [
        {"role": "system", "content": _SYSTEM.format(catalogue=catalogue())},
        {"role": "user", "content": _USER.format(message=message)},
    ]
    input_messages, instructions = prepare_responses_messages(messages)
    try:
        response = client.responses.create(
            model=model_id,
            input=input_messages,
            instructions=instructions,
            timeout=_TIMEOUT_S,
            **responses_model_kwargs(model_id),
            text={
                "format": {
                    "type": "json_schema",
                    "name": "primitive_gap",
                    "schema": _RESPONSE_SCHEMA,
                    "strict": True,
                }
            },
        )
    except openai.APITimeoutError:
        logger.error("Gap classifier timed out after %.0fs; assuming no gap", _TIMEOUT_S)
        return None
    if response.error is not None:
        raise RuntimeError(f"Model {model_id!r} errored: {response.error.message}")
    try:
        verdict = json.loads(response.output_text)
    except json.JSONDecodeError as exc:
        # Observed once in 20 probes: the model ran away and returned ~98 kB of unparseable text.
        # A classifier that cannot answer must not block the refinement it was asked about.
        logger.error("Gap classifier returned unparseable output (%s); assuming no gap", exc)
        return None
    if not verdict["needs_new"]:
        return None
    name = verdict["name"].strip()
    request = verdict["request"].strip()
    if not _NAME_PATTERN.match(name) or not request:
        logger.error("Classifier flagged a gap but named it %r; treating as no gap", name)
        return None
    if primitive_exists(name):
        logger.info("Classifier named %r, which already exists; treating as no gap", name)
        return None
    return Gap(name=name, request=request, reasoning=verdict["reasoning"].strip())
