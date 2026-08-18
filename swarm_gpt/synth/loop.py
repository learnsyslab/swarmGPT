"""The synthesis loop: the LLM authors a primitive, the filter measures it, the LLM decides.

One turn is author-or-revise, compile, run, filter, measure, feed back. The LLM owns the verdict:
it may keep what it wrote, tweak it, or throw it away and write something else. Whether the
feedback it sees carries magnitudes, comparatives, or neither is the experimental variable.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from swarm_gpt.synth import feedback as feedback_arms
from swarm_gpt.synth.manifest import PrimitiveManifest
from swarm_gpt.synth.sandbox import SynthError
from swarm_gpt.synth.verifier import authored_trajectory, check_invariants, measure, solve_only
from swarm_gpt.utils.llm_providers import (
    openai_client_for_provider,
    prepare_responses_messages,
    responses_model_kwargs,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)

VERDICTS = ("author", "keep", "tweak", "rewrite")

_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["verdict", "reasoning", "manifest", "args"],
    "properties": {
        "verdict": {"type": "string", "enum": list(VERDICTS)},
        "reasoning": {"type": "string"},
        "args": {"type": "array", "items": {"type": "number"}},
        "manifest": {
            "type": "object",
            "additionalProperties": False,
            "required": ["name", "intent", "params", "source", "invariants"],
            "properties": {
                "name": {"type": "string"},
                "intent": {"type": "string"},
                "source": {"type": "string"},
                "invariants": {"type": "string"},
                "params": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["name", "type", "minimum", "maximum"],
                        "properties": {
                            "name": {"type": "string"},
                            "type": {"type": "string", "enum": ["int", "float"]},
                            "minimum": {"type": "number"},
                            "maximum": {"type": "number"},
                        },
                    },
                },
            },
        },
    },
}

_SYSTEM = """\
You write new motion primitives for a Crazyflie drone show, in Python, and you judge your own work.

A primitive is one function with exactly this signature:

    def NAME(params, swarm_pos, tstart, tend, limits):
        ...
        return final_pos, waypoints

- `params` is the tuple of your declared parameters, in the order you declare them.
- `swarm_pos` is an (n_drones, 3) array of current positions **in centimetres**.
- `tstart`, `tend` are show times in seconds. Every waypoint time must satisfy tstart < t <= tend.
- `limits` is {{"lower": (3,), "upper": (3,)}} **in metres** -- multiply by 100 to compare against
  positions. The arena is x,y in [{lim_lower[0]:.1f}, {lim_upper[0]:.1f}] m and z in
  [{lim_lower[2]:.2f}, {lim_upper[2]:.2f}] m.
- `final_pos` is the (n_drones, 3) position array in cm after the primitive finishes.
- `waypoints` is {{time_s: {{drone_index: (3,) cm position}}}}, drone indices 0-indexed. A primitive
  may move only some drones; the rest hold.

There are {n_drones} drones. Emit enough waypoints to describe the motion -- a formation that only
snaps to its end pose needs one, a continuous figure needs several per second of travel.

Execution is sandboxed: no imports (numpy is already bound as `np`), no file or system access, no
dunder attributes. Builtins are limited to arithmetic and sequence helpers. A call must finish in
a few seconds.

Both `source` and `invariants` are Python source text, never prose. `source` defines only the
primitive; `invariants` defines only:

    def check(pos, time, params):
        return [(name, ok, detail), ...]

`pos` is (n_drones, n_steps, 3) **in centimetres**, the trajectory that actually flew after the
safety filter, and `time` is (n_steps,) in seconds. Each entry names one geometric property your
primitive is supposed to have, whether it held, and a short human-readable detail. Write checks
that could actually fail -- a check that is true of any trajectory tells you nothing. If you
claim a double helix, check that there are two strands, that each drone stays on one of them, and
that they stay opposed. This is how anyone can tell automatically that your double helix is not
one.

After the primitive runs, a safety filter (an MPC that enforces an ellipsoidal collision envelope)
repairs the trajectory, and you are told what it had to do. You then return a verdict:
  - "keep"    -- this primitive is safe and still looks like what was asked for. Stop.
  - "tweak"   -- same idea, adjusted parameters or code.
  - "rewrite" -- the idea does not work; write a different primitive.
Repeat the manifest you want to stand on with every verdict, including "keep"."""

_USER = """\
Write a motion primitive for: {request}

Return the manifest, a concrete `args` list to test it with (one value per declared parameter, in
order), verdict "author", and your reasoning."""


@dataclass
class Iteration:
    """One turn of the loop, recorded whether it reached the filter or failed before it."""

    index: int
    verdict: str
    reasoning: str
    manifest: dict[str, Any]
    args: list[float]
    stage: str = "proposed"
    error: str | None = None
    metrics: dict[str, Any] | None = None
    checks: list[dict[str, Any]] = field(default_factory=list)
    feedback: str = ""
    # `verdict` judges the *previous* iteration, since the model proposes and judges in one turn.
    # These carry the judgement of this iteration's own result, filled in on the last one.
    closing_verdict: str = ""
    closing_reasoning: str = ""


class SynthesisLoop:
    """Drive an LLM to author, test, and judge one motion primitive."""

    def __init__(
        self,
        *,
        settings: dict,
        start_pos_m: NDArray,
        arm: str,
        model_id: str,
        duration_s: float = 12.0,
        llm_provider: str = "openai",
    ):
        """Configure the loop against a swarm, a solver config, and a feedback arm."""
        if arm not in feedback_arms.ARMS:
            raise ValueError(f"Unknown feedback arm {arm!r}; expected one of {feedback_arms.ARMS}")
        self.settings = settings
        self.start_pos_m = np.asarray(start_pos_m, dtype=float)
        self.arm = arm
        self.model_id = model_id
        self.duration_s = duration_s
        self.limits = {
            "lower": np.asarray(settings["axswarm"]["pos_min"], dtype=float),
            "upper": np.asarray(settings["axswarm"]["pos_max"], dtype=float),
        }
        self.messages: list[dict[str, str]] = []
        self._client = openai_client_for_provider(llm_provider)

    def _call(self) -> dict[str, Any]:
        """Send the accumulated history and parse one structured turn out of the model."""
        input_messages, instructions = prepare_responses_messages(self.messages)
        response = self._client.responses.create(
            model=self.model_id,
            input=input_messages,
            instructions=instructions,
            **responses_model_kwargs(self.model_id),
            text={
                "format": {
                    "type": "json_schema",
                    "name": "synthesized_primitive",
                    "schema": _RESPONSE_SCHEMA,
                    "strict": True,
                }
            },
        )
        if response.error is not None:
            raise RuntimeError(f"Model {self.model_id!r} errored: {response.error.message}")
        if not response.output_text:
            raise RuntimeError(f"Model {self.model_id!r} returned no content.")
        self.messages.append({"role": "assistant", "content": response.output_text})
        return json.loads(response.output_text)

    def _evaluate(self, manifest: PrimitiveManifest, args: list[float]) -> Iteration:
        """Compile, run, filter, and measure one candidate, or record where it fell over."""
        record = Iteration(
            index=0, verdict="", reasoning="", manifest=asdict(manifest), args=list(args)
        )
        try:
            fn, check_fn = manifest.compile()
            bound = manifest.bind(args)
            record.stage = "compiled"
            authored = authored_trajectory(
                fn, bound, self.start_pos_m, 0.0, self.duration_s, self.limits
            )
            record.stage = "executed"
        except SynthError as e:
            record.error = str(e)
            return record

        repaired = solve_only(authored, self.settings)
        record.stage = "filtered"
        window = (0.0, self.duration_s)
        record.metrics = measure(authored, repaired, self.settings, window)
        try:
            record.checks = check_invariants(check_fn, repaired, bound, window)
        except SynthError as e:
            record.checks = [{"name": "check", "ok": False, "detail": f"check failed to run: {e}"}]
        record.feedback = feedback_arms.render(self.arm, record.metrics, record.checks)
        record.stage = "measured"
        return record

    def run(self, request: str, max_iterations: int = 4) -> list[Iteration]:
        """Author and refine a primitive for ``request``, stopping when the model says "keep"."""
        self.messages = [
            {
                "role": "system",
                "content": _SYSTEM.format(
                    n_drones=self.start_pos_m.shape[0],
                    lim_lower=self.limits["lower"],
                    lim_upper=self.limits["upper"],
                ),
            },
            {"role": "user", "content": _USER.format(request=request)},
        ]
        history: list[Iteration] = []
        for index in range(1, max_iterations + 1):
            turn = self._call()
            try:
                manifest = PrimitiveManifest.from_payload(turn["manifest"])
            except SynthError as e:
                record = Iteration(
                    index=index,
                    verdict=turn["verdict"],
                    reasoning=turn["reasoning"],
                    manifest=turn["manifest"],
                    args=turn["args"],
                    error=str(e),
                )
                history.append(record)
                self.messages.append({"role": "user", "content": _next_prompt(str(e))})
                continue

            record = self._evaluate(manifest, turn["args"])
            record.index = index
            record.verdict = turn["verdict"]
            record.reasoning = turn["reasoning"]
            history.append(record)
            logger.info(
                "iteration %d: %s -> %s (%s)", index, manifest.signature(), record.stage, self.arm
            )
            if turn["verdict"] == "keep" and record.error is None:
                record.closing_verdict, record.closing_reasoning = (
                    turn["verdict"],
                    turn["reasoning"],
                )
                break
            self.messages.append(
                {"role": "user", "content": _next_prompt(record.error or record.feedback)}
            )
            if index == max_iterations:
                # The model proposes and judges in one turn, so without this the run would end
                # with nobody having judged the candidate it ended on.
                closing = self._call()
                record.closing_verdict = closing["verdict"]
                record.closing_reasoning = closing["reasoning"]
                break
        return history


def _next_prompt(body: str) -> str:
    """Wrap a verifier report or a compile error as the next turn's user message."""
    return (
        f"Your primitive was run and filtered. Here is what came back:\n\n{body}\n\n"
        "Decide: keep, tweak, or rewrite. Return the manifest you want to stand on, the args to "
        "test it with, your verdict, and your reasoning."
    )
