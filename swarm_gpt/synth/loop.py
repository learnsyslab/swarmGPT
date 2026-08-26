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
from swarm_gpt.synth.verifier import (
    authored_trajectory,
    check_invariants,
    measure,
    screen_authored,
    solve_only,
)
from swarm_gpt.utils.llm_providers import (
    openai_client_for_provider,
    prepare_responses_messages,
    responses_model_kwargs,
)

if TYPE_CHECKING:
    from collections.abc import Callable

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
- These are real drones with real limits: **speed must stay under {vel_max} m/s and acceleration
  under {acc_max} m/s^2**. Consecutive waypoints must be reachable in the time between them, so a
  drone that has to cross 2 m needs at least {crossing_s:.0f} s of waypoints to do it in. Snapping
  the swarm between poses demands speeds no drone can fly; the trajectory is rejected before the
  filter ever sees it, and you are told by how much you exceeded the limit.
- `final_pos` is the (n_drones, 3) position array in cm after the primitive finishes.
- `waypoints` is {{time_s: {{drone_index: (3,) cm position}}}}, drone indices 0-indexed. A primitive
  may move only some drones; the rest hold.

There are {n_drones} drones. Emit enough waypoints to describe the motion -- a formation that only
snaps to its end pose needs one, a continuous figure needs several per second of travel.

Execution is sandboxed: no imports (numpy is already bound as `np`), no file or system access, no
dunder attributes. Builtins are limited to arithmetic and sequence helpers. A call must finish in
a few seconds.

Two helpers are bound for you. They are the ones every hand-written primitive in this library
already uses, and you are expected to use them rather than rewrite them:

    slots = assign(current_pos, target_pos)

  Hungarian assignment. Both arrays are (n_drones, 3) in cm; `slots[i]` is the index of the target
  that drone `i` should fly to. Sending drone `i` to `target_pos[i]` instead, and interpolating
  straight there, is what makes twenty drones cross through each other.

    t_arrive = arrival_time(target_pos, current_pos, tstart, tend)

  The earliest time the whole swarm can reach `target_pos` without exceeding the speed limit,
  sized by the drone with furthest to travel. Emit your formation waypoint at or after this.

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

You have exactly **{duration_s:.1f} seconds**: the primitive is called with
`tend - tstart = {duration_s:.1f}`, both here and in the show it will be used in. Everything it
does must fit in that interval at the speed and acceleration limits above. Do not declare a
duration parameter -- the interval is fixed and is not yours to choose.

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
    # One sentence per limit the pre-solve screen found broken -- separation, speed, or
    # acceleration. Kept verbatim so a reader is never left inferring which one it was.
    violations: list[str] = field(default_factory=list)
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
        screen: bool = False,
    ):
        """Configure the loop against a swarm, a solver config, and a feedback arm.

        ``screen`` skips the solve when the authored geometry is already infeasible. It is off by
        default because it changes what the model is told, and the feedback ablation was measured
        without it.
        """
        if arm not in feedback_arms.ARMS:
            raise ValueError(f"Unknown feedback arm {arm!r}; expected one of {feedback_arms.ARMS}")
        self.settings = settings
        self.start_pos_m = np.asarray(start_pos_m, dtype=float)
        self.arm = arm
        self.model_id = model_id
        self.duration_s = duration_s
        self.screen = screen
        self.limits = {
            "lower": np.asarray(settings["axswarm"]["pos_min"], dtype=float),
            "upper": np.asarray(settings["axswarm"]["pos_max"], dtype=float),
        }
        self.messages: list[dict[str, str]] = []
        # The SDK default stacks a 600 s read timeout with two retries, so one stalled call can
        # hold a browser job for half an hour with nothing to show. A turn legitimately runs into
        # minutes, so the read timeout stays generous and only the retries are cut.
        self._client = openai_client_for_provider(llm_provider).with_options(max_retries=1)

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
        try:
            turn = json.loads(response.output_text)
        except json.JSONDecodeError as e:
            # Seen twice on hard revise turns: the model runs away and returns tens of kB of
            # unparseable text. That is a bad turn, not a broken run, so it becomes feedback like
            # any other failure rather than ending the loop. The runaway text itself is kept out
            # of the history -- replaying it every turn is what makes the next one run away too.
            self.messages.append({"role": "assistant", "content": "(unparseable reply, discarded)"})
            raise SynthError(
                f"Your reply was not valid JSON ({e}). Return one JSON object matching the "
                f"schema, and keep the source short enough to finish."
            ) from e
        self.messages.append({"role": "assistant", "content": response.output_text})
        return turn

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

        window = (0.0, self.duration_s)
        if self.screen:
            screened, violations = screen_authored(authored, self.settings, window)
            if violations:
                record.stage = "screened"
                record.metrics = screened
                record.violations = violations
                record.feedback = _screen_report(violations)
                return record

        repaired = solve_only(authored, self.settings)
        record.stage = "filtered"
        record.metrics = measure(authored, repaired, self.settings, window)
        try:
            record.checks = check_invariants(check_fn, repaired, bound, window)
        except SynthError as e:
            record.checks = [{"name": "check", "ok": False, "detail": f"check failed to run: {e}"}]
        record.feedback = feedback_arms.render(self.arm, record.metrics, record.checks)
        record.stage = "measured"
        return record

    def run(
        self,
        request: str,
        max_iterations: int = 4,
        on_iteration: Callable[[Iteration], None] | None = None,
        on_authoring: Callable[[int], None] | None = None,
    ) -> list[Iteration]:
        """Author and refine a primitive for ``request``, stopping when the model says "keep".

        ``on_iteration`` is called with each finished turn and ``on_authoring`` with the index of
        each turn as it is sent, for a caller streaming progress. A turn is minutes of model time,
        so without the second one a caller has nothing to show between iterations.
        """
        self.messages = [
            {
                "role": "system",
                "content": _SYSTEM.format(
                    n_drones=self.start_pos_m.shape[0],
                    lim_lower=self.limits["lower"],
                    lim_upper=self.limits["upper"],
                    vel_max=self.settings["axswarm"]["vel_max"],
                    acc_max=self.settings["axswarm"]["acc_max"],
                    crossing_s=2.0 / self.settings["axswarm"]["vel_max"],
                ),
            },
            {"role": "user", "content": _USER.format(request=request, duration_s=self.duration_s)},
        ]
        history: list[Iteration] = []
        for index in range(1, max_iterations + 1):
            if on_authoring is not None:
                on_authoring(index)
            try:
                turn = self._call()
            except SynthError as e:
                record = Iteration(
                    index=index, verdict="", reasoning="", manifest={}, args=[], error=str(e)
                )
                history.append(record)
                if on_iteration is not None:
                    on_iteration(record)
                self.messages.append({"role": "user", "content": _next_prompt(str(e))})
                continue
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
                if on_iteration is not None:
                    on_iteration(record)
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
            if on_iteration is not None:
                on_iteration(record)
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


def _screen_report(violations: list[str]) -> str:
    """Render a pre-solve rejection. Deliberately not a feedback arm: arms are the experiment."""
    listed = "\n".join(f"  - {v}" for v in violations)
    return (
        "Your primitive was NOT run through the safety filter. What it authored is already "
        "impossible, so there is nothing for the filter to repair:\n\n"
        f"{listed}\n\n"
        "Two drones cannot occupy the same place at once, and a drone cannot exceed its speed or "
        "acceleration limit. Interpolating each drone straight from where it is to where you want "
        "it crosses paths and demands whatever speed the gap requires. Choose which drone goes to "
        "which target so the paths do not cross, and spread every movement over enough time that "
        "the swarm can actually fly it."
    )


def _next_prompt(body: str) -> str:
    """Wrap a verifier report or a compile error as the next turn's user message."""
    return (
        f"Your primitive was run and filtered. Here is what came back:\n\n{body}\n\n"
        "Decide: keep, tweak, or rewrite. Return the manifest you want to stand on, the args to "
        "test it with, your verdict, and your reasoning."
    )
