from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pytest

from swarm_gpt.core import motion_primitives as mp
from swarm_gpt.core.structured_output_schema import primitive_exists
from swarm_gpt.synth.loop import Iteration, SynthesisLoop
from swarm_gpt.synth.promote import gate, load_promoted, promote, register_entry, reset_synthesized
from swarm_gpt.synth.refine import NO_GAP, SynthesisOutcome, synthesize_for_refine
from swarm_gpt.synth.trigger import Gap, catalogue

if TYPE_CHECKING:
    from pathlib import Path

RISER_SOURCE = """
def riser(params, swarm_pos, tstart, tend, limits):
    delta_cm, = params
    pos = swarm_pos + np.array([0.0, 0.0, delta_cm])
    return pos, {float(tend): {i: p.copy() for i, p in enumerate(pos)}}
"""

RISER_CHECK = """
def check(pos, time, params):
    return [("rises", bool(np.all(pos[:, -1, 2] >= pos[:, 0, 2])), "never descends")]
"""

MANIFEST = {
    "name": "riser",
    "intent": "lift the whole swarm",
    "params": [{"name": "delta_cm", "type": "float", "minimum": 1.0, "maximum": 50.0}],
    "source": RISER_SOURCE,
    "invariants": RISER_CHECK,
}


@pytest.fixture(autouse=True)
def _clean_registries():
    yield
    reset_synthesized()


def _record(**overrides: object) -> Iteration:
    record = Iteration(
        index=3, verdict="keep", reasoning="looks right", manifest=dict(MANIFEST), args=[10.0]
    )
    record.stage = "measured"
    record.closing_verdict = "keep"
    record.closing_reasoning = "safe"
    record.metrics = {"steps_inside_envelope": 0, "n_steps": 100, "min_sep_norm": 1.4}
    for key, value in overrides.items():
        setattr(record, key, value)
    return record


def _promote(record: Iteration, out_dir: Path) -> tuple[int, str, Path | None]:
    return promote(
        [record],
        request="lift",
        arm="absolute",
        model="m",
        duration_s=12.0,
        n_drones=10,
        out_dir=out_dir,
    )


def test_promote_writes_the_entry_when_both_gates_pass(tmp_path: Path) -> None:
    code, status, path = _promote(_record(), tmp_path)
    assert (code, status) == (0, "promoted")
    entry = json.loads(path.read_text())
    assert entry["manifest"]["name"] == "riser"
    assert entry["provenance"]["n_drones"] == 10


def test_promote_refuses_a_run_the_model_never_accepted(tmp_path: Path) -> None:
    code, status, path = _promote(_record(closing_verdict="tweak"), tmp_path)
    assert code == 1
    assert path is None
    assert "not accepted" in status


def test_promote_refuses_a_kept_primitive_that_flew_inside_the_envelope(tmp_path: Path) -> None:
    metrics = {"steps_inside_envelope": 7, "n_steps": 100, "min_sep_norm": 0.62}
    code, status, path = _promote(_record(metrics=metrics), tmp_path)
    assert code == 2
    assert path is None
    assert "collision envelope" in status


def test_promote_refuses_a_primitive_that_never_reached_the_filter(tmp_path: Path) -> None:
    code, _status, path = _promote(_record(stage="screened"), tmp_path)
    assert code == 2
    assert path is None


def test_register_entry_makes_the_primitive_resolvable_and_emittable() -> None:
    manifest = register_entry({"manifest": MANIFEST})
    assert manifest.name == "riser"
    assert primitive_exists("riser")
    pos, waypoints = mp.primitive_by_name("riser")(
        (10.0,), np.zeros((10, 3)), 0.0, 5.0, {"lower": np.zeros(3), "upper": np.ones(3)}
    )
    assert pos.shape == (10, 3)
    assert waypoints


def test_load_promoted_skips_a_primitive_verified_for_a_different_swarm(tmp_path: Path) -> None:
    entry = {"manifest": MANIFEST, "provenance": {"n_drones": 20}}
    (tmp_path / "riser.json").write_text(json.dumps(entry))
    assert load_promoted(tmp_path, n_drones=10) == []
    assert load_promoted(tmp_path, n_drones=20) == ["riser"]


def test_load_promoted_skips_a_malformed_entry_rather_than_failing(tmp_path: Path) -> None:
    (tmp_path / "broken.json").write_text("{not json")
    (tmp_path / "riser.json").write_text(
        json.dumps({"manifest": MANIFEST, "provenance": {"n_drones": 10}})
    )
    assert load_promoted(tmp_path, n_drones=10) == ["riser"]


def test_load_promoted_is_empty_when_the_directory_does_not_exist(tmp_path: Path) -> None:
    assert load_promoted(tmp_path / "nope") == []


def test_catalogue_lists_hand_written_and_synthesized_primitives() -> None:
    assert "form_circle(drone_ids, radius_cm, z_coord_cm, time_to_finish_s)" in catalogue()
    register_entry({"manifest": MANIFEST})
    assert "riser(delta_cm: float [1.0, 50.0])" in catalogue()


def test_synthesis_is_skipped_entirely_when_the_mode_is_off() -> None:
    outcome = synthesize_for_refine(
        "put a heart at the drop",
        mode="off",
        settings={},
        start_pos_m=np.zeros((10, 3)),
        model_id="unused",
    )
    assert outcome.code == NO_GAP
    assert not outcome.promoted


def test_a_promoted_primitive_is_announced_to_the_choreographer() -> None:
    outcome = SynthesisOutcome(
        0,
        "promoted",
        gap=Gap("riser", "lift", "why"),
        name="riser",
        signature="riser(delta_cm: float [1.0, 50.0])",
    )
    prefixed = outcome.prefix("put a heart at the drop")
    assert "riser(delta_cm: float [1.0, 50.0])" in prefixed
    assert prefixed.endswith("put a heart at the drop")


def test_a_failed_synthesis_leaves_the_refinement_message_untouched() -> None:
    outcome = SynthesisOutcome(1, "not accepted")
    assert outcome.prefix("put a heart at the drop") == "put a heart at the drop"


class _Response:
    def __init__(self, text: str) -> None:
        self.error = None
        self.output_text = text


class _StubClient:
    """Returns a runaway reply first, then nothing the loop can use, so run() must survive both."""

    def __init__(self, texts: list[str]) -> None:
        self.texts = texts
        self.responses = self

    def create(self, **_kwargs: object) -> _Response:
        return _Response(self.texts.pop(0) if self.texts else "{")

    def with_options(self, **_kwargs: object) -> _StubClient:
        return self


def test_an_unparseable_turn_becomes_feedback_instead_of_ending_the_run() -> None:
    loop = SynthesisLoop(
        settings={
            "axswarm": {
                "pos_min": [-2, -2, 0],
                "pos_max": [2, 2, 2],
                "vel_max": 1.73,
                "acc_max": 1.0,
            }
        },
        start_pos_m=np.zeros((10, 3)),
        arm="absolute",
        model_id="stub",
    )
    loop._client = _StubClient(['{"verdict": "author", ', "{ still broken"])
    history = loop.run("a shape", max_iterations=2)
    assert len(history) == 2
    assert all("not valid JSON" in (record.error or "") for record in history)
    assert "(unparseable reply, discarded)" in [m["content"] for m in loop.messages]


def test_reset_synthesized_removes_the_primitive_from_every_registry() -> None:
    register_entry({"manifest": MANIFEST})
    assert primitive_exists("riser")
    assert "riser" in mp.motion_primitives

    reset_synthesized()

    assert not primitive_exists("riser")
    assert "riser" not in mp.motion_primitives
    assert "riser" not in mp._synthesized
    with pytest.raises(KeyError):
        mp.primitive_by_name("riser")


def test_reset_synthesized_leaves_the_hand_written_library_alone() -> None:
    register_entry({"manifest": MANIFEST})
    reset_synthesized()
    assert primitive_exists("form_circle")
    assert mp.primitive_by_name("form_circle") is not None


def test_gate_does_not_write_anything_to_the_library(tmp_path: Path) -> None:
    code, status, record = gate([_record()])
    assert (code, status) == (0, "promoted")
    assert record is not None
    assert list(tmp_path.iterdir()) == []
