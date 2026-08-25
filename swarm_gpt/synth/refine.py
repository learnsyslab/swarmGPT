"""Run synthesis as part of one refinement request, and report progress while it happens.

Synthesis is minutes of API time and it can legitimately fail, so every outcome -- gap or no gap,
cleared or refused -- ends with the refinement going ahead against whatever library exists.

**A primitive authored here lives only as long as the choreography that asked for it.** It is
registered in memory and never written to ``results/synthesized/``, and the API drops it when a
new song is selected. Authoring the library on purpose is the CLI's job, not a refinement's, and
a demo that shows the model inventing a shape must be able to show it again tomorrow.

The trigger is the classifier in ``trigger.py`` by default. ``mode`` swaps it: "off" never
synthesizes, "force" skips the classifier and treats the message itself as the request, which is
what a demo should use when it must not depend on a judgement call.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from swarm_gpt.synth.loop import SynthesisLoop
from swarm_gpt.synth.promote import gate, register_entry
from swarm_gpt.synth.run_log import write_run_log
from swarm_gpt.synth.trigger import Gap, classify_gap

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

logger = logging.getLogger(__name__)

SynthesisMode = Literal["auto", "off", "force"]

NO_GAP = -1

_ANNOUNCEMENT = """\
A new motion primitive has just been added to your library and you may now call it: {signature}.
It was authored for exactly this request and has been verified against the safety filter. Use it
where the request asks for it, and keep the rest of the choreography as it is.

"""


@dataclass(frozen=True)
class SynthesisOutcome:
    """What one refine's synthesis attempt did, whether or not it produced a primitive."""

    code: int
    status: str
    gap: Gap | None = None
    name: str = ""
    signature: str = ""

    @property
    def promoted(self) -> bool:
        """Whether a new primitive is now registered and callable."""
        return self.code == 0

    @property
    def failed(self) -> bool:
        """Whether synthesis was attempted and did not produce a usable primitive.

        A refinement that asked for a shape the library cannot express has nothing useful to do
        without it -- the choreographer would approximate it drone by drone with ``move``, which
        is hand-authoring the primitive badly. The caller abandons the refinement instead.
        """
        return self.code not in (0, NO_GAP)

    def prefix(self, message: str) -> str:
        """Prepend the announcement of a new primitive to the user's refinement message."""
        if not self.promoted:
            return message
        return _ANNOUNCEMENT.format(signature=self.signature) + message


def _emit(
    on_event: Callable[[str, dict[str, Any]], None] | None, kind: str, **payload: Any
) -> None:
    if on_event is not None:
        on_event(kind, payload)


def synthesize_for_refine(
    message: str,
    *,
    mode: SynthesisMode,
    settings: dict,
    start_pos_m: NDArray,
    model_id: str,
    llm_provider: str = "openai",
    arm: str = "absolute",
    max_iterations: int = 14,
    duration_s: float = 12.0,
    on_event: Callable[[str, dict[str, Any]], None] | None = None,
) -> SynthesisOutcome:
    """Author, verify, and register the primitive one refinement needs, if it needs one.

    Returns:
        The outcome. ``code`` is 0 promoted, 1 the model never accepted, 2 a gate refused what it
        kept, and ``NO_GAP`` when no synthesis was attempted.
    """
    if mode == "off":
        return SynthesisOutcome(NO_GAP, "synthesis disabled for this refinement")
    if mode == "force":
        gap = Gap(name="", request=message, reasoning="requested explicitly")
    else:
        gap = classify_gap(message, model_id=model_id, llm_provider=llm_provider)
        if gap is None:
            _emit(on_event, "synthesis_skipped", reason="the existing library covers this request")
            return SynthesisOutcome(NO_GAP, "the existing library covers this request")
    _emit(on_event, "synthesis_started", request=gap.request, reasoning=gap.reasoning)

    n_drones = int(start_pos_m.shape[0])
    loop = SynthesisLoop(
        settings=settings,
        start_pos_m=start_pos_m,
        arm=arm,
        model_id=model_id,
        duration_s=duration_s,
        llm_provider=llm_provider,
        screen=True,
    )
    history = loop.run(
        gap.request,
        max_iterations=max_iterations,
        on_authoring=lambda index: _emit(on_event, "synthesis_authoring", index=index),
        on_iteration=lambda record: _emit(
            on_event,
            "synthesis_iteration",
            index=record.index,
            stage=record.stage,
            name=record.manifest.get("name", ""),
            metrics=record.metrics or {},
            violations=record.violations,
            error=record.error,
            verdict=record.verdict,
            reasoning=record.reasoning,
            checks=record.checks,
        ),
    )

    code, status, record = gate(history)
    write_run_log(
        history,
        status,
        request=gap.request,
        arm=arm,
        model=model_id,
        duration_s=duration_s,
        n_drones=n_drones,
        max_iterations=max_iterations,
    )
    if code:
        _emit(on_event, "synthesis_failed", code=code, status=status, request=gap.request)
        return SynthesisOutcome(code, status, gap=gap)

    manifest = register_entry({"manifest": record.manifest})
    signature = manifest.signature()
    _emit(
        on_event,
        "synthesis_promoted",
        name=manifest.name,
        signature=signature,
        metrics=record.metrics,
    )
    return SynthesisOutcome(0, status, gap=gap, name=manifest.name, signature=signature)
