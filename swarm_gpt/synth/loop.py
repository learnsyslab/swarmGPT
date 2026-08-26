"""The synthesis loop: the LLM authors a shape, the filter measures it, the LLM decides.

One turn is author-or-revise, compile, screen the geometry, fly it, measure, feed back. The LLM
owns the verdict: it may keep what it wrote, tweak it, or throw it away and write something else.
Whether the feedback it sees carries magnitudes, comparatives, or neither is the experimental
variable.

The model writes the equation of a shape and nothing else. Which drone flies to which point and
how long the arrival takes belong to the library's own formation helpers, which is where every
hand-written primitive already leaves them.
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
from swarm_gpt.synth.shape import screen_shape, targets
from swarm_gpt.synth.verifier import authored_trajectory, measure, screen_authored, solve_only
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
            "required": ["name", "intent", "params", "source"],
            "properties": {
                "name": {"type": "string"},
                "intent": {"type": "string"},
                "source": {"type": "string"},
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
You write new formation primitives for a Crazyflie drone show. A formation primitive says **where
the drones stand**, and nothing else:

    def NAME(params, n_drones):
        ...
        return positions

- `params` is the tuple of your declared parameters, in the order you declare them.
- `n_drones` is how many drones the show has. It is {n_drones} today, but write the equation for
  any number: sample `n_drones` points along the shape.
- `positions` is an (n_drones, 3) array of x, y, z **in centimetres**, one point per drone.

That is the whole job, and it is meant to be short. The library's own `form_circle` is

    angles = np.linspace(0, 2 * np.pi, n_drones, endpoint=False)
    return np.stack([r * np.cos(angles), r * np.sin(angles), np.full(n_drones, z)], axis=-1)

Yours should be about that long. Write the equation of the shape you are asked for and sample it.

**You do not fly the drones anywhere.** Which drone takes which point, how long the flight takes,
and holding the shape for the rest of the interval are handled for you, by the same helpers every
hand-written primitive in this library uses. Do not think about time, waypoints, speed, or
acceleration -- there is no time in your function at all, and no duration parameter.

Three things about the room, and they are the only geometry constraints you have:

- **The arena.** x and y run [{lim_lower[0]:.0f}, {lim_upper[0]:.0f}] cm about a centre at the
  origin, and z runs [{lim_lower[2]:.0f}, {lim_upper[2]:.0f}] cm, giving you {z_span:.0f} cm of
  height. Anything outside is clipped, which deforms your shape -- keep every point inside.
- **Draw the shape standing upright, in the x-z plane.** x runs left to right and z is height, so
  that is the plane an audience standing in front of the swarm sees the shape in. Put every point
  at y = 0 unless the request specifically asks for depth. **Do not tilt, lean, or lay the shape
  flat to make room** -- there is no need, and a leaning shape is not what was asked for.
- **Two drones may not stand too close.** Every pair of points must satisfy
  `(dx/{sep_xy:.0f})^2 + (dy/{sep_xy:.0f})^2 + (dz/{sep_z:.0f})^2 >= 1`, distances in cm. Drones
  may sit directly above one another: {sep_z:.0f} cm of clearance is enough in any direction, and
  downwash is not something you need to think about. If two points do come out too close, scale
  the shape up or sample fewer points along the crowded part of it.

Your shape is measured before anything flies, and you are told in centimetres which two of your
points are too close. Then it is flown through the safety filter and you are told what the filter
had to move.

`source` is Python source text, never prose, and defines only the shape function.

You then return a verdict:
  - "keep"    -- this shape is safe and still looks like what was asked for. Stop.
  - "tweak"   -- same shape, adjusted parameters or code.
  - "rewrite" -- the shape does not work; write a different one.
Repeat the manifest you want to stand on with every verdict, including "keep"."""

_USER = """\
Write a formation primitive for: {request}

Declare the parameters the shape genuinely has -- its size, its height off the floor -- with
ranges, and nothing else. No duration, no speed, no number of drones, and no tilt or lean.

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

    def _system_prompt(self) -> str:
        """Render the authoring contract against this room's arena and collision envelope."""
        envelope = np.asarray(self.settings["axswarm"]["collision_envelope"], dtype=float) * 100
        return _SYSTEM.format(
            n_drones=self.start_pos_m.shape[0],
            lim_lower=self.limits["lower"] * 100,
            lim_upper=self.limits["upper"] * 100,
            sep_xy=envelope[0],
            sep_z=envelope[2],
            z_span=(self.limits["upper"][2] - self.limits["lower"][2]) * 100,
        )

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
            fn, shape_fn = manifest.compile()
            bound = manifest.bind(args)
            record.stage = "compiled"
            # The geometry is judged on its own first: a close pair here is the shape being too
            # dense, which is a different fix from the fly-in crossing.
            des_pos = targets(shape_fn, bound, self.start_pos_m.shape[0])
            record.metrics, record.violations = screen_shape(des_pos, self.settings)
            if record.violations:
                record.stage = "shaped"
                record.feedback = feedback_arms.render_screen(self.arm, record.metrics)
                return record
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
                record.feedback = feedback_arms.render_screen(self.arm, screened)
                return record

        repaired = solve_only(authored, self.settings)
        record.stage = "filtered"
        record.metrics = measure(authored, repaired, self.settings, window)
        record.feedback = feedback_arms.render(self.arm, record.metrics)
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
            {"role": "system", "content": self._system_prompt()},
            {"role": "user", "content": _USER.format(request=request)},
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


def _next_prompt(body: str) -> str:
    """Wrap a verifier report or a compile error as the next turn's user message."""
    return (
        f"Your primitive was run and filtered. Here is what came back:\n\n{body}\n\n"
        "Decide: keep, tweak, or rewrite. Return the manifest you want to stand on, the args to "
        "test it with, your verdict, and your reasoning."
    )
