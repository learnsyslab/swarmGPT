# Verified primitive synthesis — continuation handoff

Branch `feat/primitive-synthesis`, pushed to `origin`. This document plus that branch is the
whole thread; nothing else is required to pick it up.

Written by **Yiyi Xu** (yiyi.xu@mail.utoronto.ca), LSY Lab TUM, 19 May – 27 August 2026.
Supervisor: **Marcel**. Collaborators: **Martin**, **Alex**.
Last updated **2026-08-26**. Tracked on this branch, so a clone carries it.

---

## 1. What this thread is

The last two weeks of the internship went to a proposal for the ICRA/RAL framing — **the solver as
a verifier for LLM-generated choreography** — and the experiments backing it.

The claim has two halves, and both now have data:

1. The hand-written primitive library **binds** what the LLM can ask for. It is not a neutral
   vocabulary; it silently shrinks the choreography.
2. Feedback from the safety filter, **carrying magnitudes**, lets the LLM author new primitives
   that clear the filter — a loop where the solver teaches rather than merely rejects.

Both halves now have results. Half 1 is settled and written up; half 2 has a working pipeline and
its first verified primitive.

---

## 2. Current state

**It works from the browser.** A refinement that asks for a shape the library cannot express now
authors the primitive, verifies it, registers it, and regenerates the choreography with it — §11
is the whole path and the decisions behind it. The 2026-08-26 run promoted `heart_drop` in six
attempts (authored separation 1.618, flown 1.599, 0 steps inside the envelope) from *"put a heart
shape primitive at the beat drop"*. What has **not** been watched end to end on the current code
is the part after promotion: whether the drones visibly make the shape in playback. That is step 2
of §8 and it is one run.

**The loop works, twice.** `results/synthesized/upright_heart_outline.json` is a heart the
choreographer had no way to express before: 10 drones, tip on the centreline, flaring to a
1.80 m half-span, two lobes with a cleft between them, **y spread 0.000 m**. Seven iterations,
authored separation 1.717, flown 1.645 with 0 steps inside the envelope. It is inserted at the
drop between `spiral_speed` and `helix` in `presets/Fearless2 | 10 | 20260825_170100_after`, solves,
and renders.

`results/synthesized/double_helix.json` is an LLM-authored primitive that
clears both gates: authored separation 1.265, flown 1.235 with **0 steps inside the collision
envelope**, and the filter moving it 0.136 m at most. It converged in 7 iterations. It renders through the shipped `render.py` and flies.

What unblocked it was binding the library's own formation helpers into the sandbox. Before that the
model had to reinvent collision-free formation entry every time and never got authored separation
above 0.484 in 8 iterations; with `assign` available it cleared 1.0 on the first try.

The primitive builds a **static** double helix and holds it — `turns` is the spatial pitch of the
helix, not a spin. Motion comes from composing it in the choreography: `rotate(90, 'z')` on later
keys turns the formed shape and cannot collide, since a z-rotation preserves every radius and
height. Measured over the show, the formation turns 270-288 degrees with radii and height span
unchanged.

**Whether a primitive looks like what was asked for is deliberately not gated.** A hand-written
shape predicate was built and then removed, for three reasons:

- **A primitive already is the geometry.** Look at `form_circle`: it computes the target positions
  and hands off to `_assign_positions` and `_formation_arrival_time`. The LLM's contribution is
  "where does each drone sit for this shape"; the solver moves them. A predicate is the same kind
  of artifact — a geometric characterisation — so grading one with the other is circular.
- **It cannot work for a live request.** A predicate must exist before the shape is asked for. The
  demo case is someone asking for something nobody anticipated, which is exactly what a
  pre-written predicate cannot cover.
- **It was wrong more often than the model was.** It demanded counter-rotation, which makes a
  double helix physically impossible, and it rejected a rational response to a heart request I had
  wrongly measured as infeasible.

Two gates remain, both objective, both from the hardware and the room rather than from anyone's
idea of a shape: the pre-solve screen and the flown collision envelope. Whether it looks right is a
human call. **Report it as a success rate** — the three ablation arms are the baseline, and
`categorical`, what swarmGPT ships today, converged 0/17 against `absolute`'s 8/18.

Run logs stay local — `synth_runs/` is gitignored.

---

## 3. Where things live

| Repo | Holds |
|---|---|
| [`swarmGPT`](https://github.com/learnsyslab/swarmGPT) | This thread, plus the choreographer, primitives, lighting, music analysis, deployment, frontend |
| [`amswarm-continuous`](https://github.com/learnsyslab/amswarm-continuous) | The continuous-time receding-horizon solver |
| [`MAPF_benchmarking`](https://github.com/learnsyslab/MAPF_benchmarking) (Marcel's) | Benchmark harness and solver wrappers; use `two-solver-bench` |

**Branches.** `feat/primitive-synthesis` sits on the lighting line, not the spline one. It holds
`swarm_gpt/synth/` and **all of `experiments/`** — the feedback ablation, three coverage probes,
the vocabulary judge, and their tracked result data. It is the only copy of both, and it has never
been PR'd. **As of 2026-08-26 it is four commits ahead of `origin` (`a7a965e`..`31e4a7e`), which is
all of §11 — push before handing it on.**

The sibling branches matter for merges: `feat/lighting-primitives` (PR #11 still open) has the
current lighting; `feat/swarmgpt2-splines` caught the lighting work one commit before the PR #11
review fixes, so its `lighting.py` is ~50 lines behind. Taking lighting means taking it from
`feat/lighting-primitives`.

**Data.** `results/` is tracked and holds only what a result rests on; raw scratch output goes to
the gitignored `synth_runs/`. The ablation cost ~6 h of API time. Read `results/README.md` before
re-running anything.

---

## 4. The evidence so far

Every figure below is `gpt-5.6-luna`. Coverage work was the 10-drone lab swarm; the synthesis runs
in §6 are 20 drones, so the two are not directly comparable.

### 4.1 Does the library bind? — yes

Three instruments, deliberately biased in different directions. Full write-up in
`results/README.md`; `results/coverage/` holds the data.

- **Introspective probe — broken, kept as a negative result.** Asked which moments the library
  could not express: **26 probes, 26 empty answers**. It rationalises coverage instead of reporting
  a gap. A positive control (asking directly about a double helix) gets a correct "no". *Never ask
  an LLM what capability it is missing.*
- **Decoy menu — revealed preference.** 13 fake motion and 6 fake lighting primitives added to the
  prompt and schema, never executed, only counted: **35 of 39 probes chose a primitive that does
  not exist**. The control that makes it interpretable is that decoys come in two classes — `gap`
  adds capability, `redundant` renames an existing primitive — and no duplicate ever *replaced* its
  twin. Bias: offering an item over-states need.
- **Unanchored elicitation — the primary instrument.** A plan elicited with no vocabulary in the
  prompt, then again with it, and a *separate* judge rating each intent. **88 intents per
  condition, 13 songs, same direction in 13/13** (sign test p ≈ 0.0002). Bias runs the other way,
  which is why the agreement matters.
- **The `move` correction that moved the headline.** The judge had been crediting
  `move(x,y,z,drone_id)` with expressing multi-drone shapes; `move` is emitted **zero times** in 39
  real choreographies, and covering a shape one drone at a time *is* hand-authoring the primitive.
  Disallowing it: blind shortfall 67% → **88%**, anchored unchanged at 24%, delta +43.2 → **+63.6
  pp**. **Judge noise floor ±7 pp** — read every other figure against it.
- **Missing capability vs unwired capability.** The same 88 blind intents against three libraries:
  current 10% expressible, sg2 9%, **sg2_full (all 30 primitives in `blocks.py`) 19%**. Exposing
  the 18 already-written primitives is real (**McNemar p = 0.023**) and worth ~10 points, and it is
  a prompt-and-schema change, not new maths. **81% still falls short with everything the lab has
  already written**, and the residual is **colour palette (47 of 71)**, not motion.

### 4.2 Does the content of feedback matter? — yes, and magnitudes win

`results/feedback-ablation/ablation-54run.jsonl`: 6 requests × 3 arms × 3 repeats. All three arms
read one identical measurements dict, so they differ in wording alone. Primary outcome fixed in the
script docstring before any data existed.

| arm | dev_max median (IQR) | checks pass | model said "keep" |
|---|---|---|---|
| categorical (what swarmGPT ships) | 0.35 (0.19–1.68) | 0.67 | **0/17** |
| **absolute (metres)** | **0.17 (0.13–0.28)** | **1.00** | **8/18** |
| relative (ratios, no units) | 0.46 (0.16–1.58) | 1.00 | 1/17 |

Mann-Whitney one-sided: absolute < categorical **p = 0.017**, absolute < relative **p = 0.015**.
Fisher exact on convergence: **p = 0.0019** and **p = 0.011**. Absolute wins 5 of 6 requests.

Two things matter more than the medians. Absolute's IQR is tight while the others swing past 2.6 —
magnitudes suppress the catastrophic runs rather than shifting the average. And categorical
converged **zero** times: without numbers the model never reaches a state it will accept.

**`relative` is significantly worse than `absolute`.** Marcel's objection was that LLMs are bad
with numbers; stripping the units and substituting "about half the separation they need" made it
worse. That arm existed to test the objection and refuted it.

Caveats: n ≈ 17 per arm, one model, six requests. Two runs of the same arm on the same request once
gave 0.16 and 1.68 — that variance is why a 3-run comparison told us nothing.

### 4.3 The regime that ablation ran in — **state this, but it is not a confound**

The whole ablation ran with the solver failing most of the time. Across the 54 runs, 199 measured
iterations have a **median 50% failed-solve fraction**, and only **9 of 199** were fully clean. The
cause is in §5: the model was never told the drones have kinematic limits, so it authored
trajectories demanding up to 37 m/s.

I first assumed this inverted the primary outcome — that failed solves would hold the swarm still
and make `deviation_max` read flatteringly low. **Checked against the data, it is the opposite:**

| | median `deviation_max` |
|---|---|
| mostly-clean solves (<25% failed, n=64) | **0.16 m** |
| mostly-failed solves (>75% failed, n=47) | **0.98 m** |

Pearson r(failed-solve fraction, `deviation_max`) = **+0.176**. Failing to solve makes the score
*worse*, which is the direction you want: an infeasible authored trajectory should score badly. So
the pre-registered outcome is measuring something real and the arm comparison is sound. Absolute
having both the lowest failure rate (**0.39** vs 0.57 categorical, 0.47 relative) and the lowest
deviation is one coherent story — it authors more feasible trajectories, which both solve better
and track better — not two confounded ones. The convergence result (8/18 vs 0/17) does not touch
deviation at all.

Two things still belong in the write-up:

- **The operating regime.** Every arm was handicapped by a prompt that omitted the kinematic
  limits. Report it as a limitation and note the fix; a reviewer who reads the run logs will see
  the failure rate.
- **Soften "repairs".** Only **16 of 52** runs produced a collision-safe trajectory at all, and
  **2 of the 9** the model said "keep" on were unsafe. The filter did not reliably repair anything.

---

## 5. What this session built

The loop existed before (`swarm_gpt/synth/`: loop, verifier, feedback, sandbox, manifest). What is
new is that nothing the model asserts is taken on trust, and that a promoted primitive is
indistinguishable from a hand-written one at choreography time.

**One manifest, four places.** A primitive's signature lives in the prompt, the structured-output
schema, the backend function, and the offline check — the standing invariant is that changing fewer
than all four causes bugs. `PrimitiveManifest.register()` now writes all of them from one
declaration, so a synthesized primitive is emittable by the choreographer LLM, renderable as a
call, and visible in the prompt catalogue, without any of them able to drift.

**Two gates, neither of which the model may overrule.**

1. **Pre-solve screen** (`screen_authored` in `verifier.py`). The authored waypoints must be
   collision-free *and* flyable. A full axswarm solve is **32 s**; the screen reproducing the same
   `authored_min_sep_norm` is **1.4 ms** — ~23,000× cheaper.
2. **The filter.** Every solve must succeed, and flown separation must clear the envelope.

**The model cannot audit its own geometry.** It passed **5/5** and **4/4** of *its own* invariants
on trajectories that were two flat counter-rotating rings — the third instance of this, after the
introspective probe and after it knowingly kept a trajectory inside the collision envelope. That
finding stands on its own; the fix is a human looking at the render, not an automated predicate.

**Why gate 1 mattered more than expected.** It was built for speed and turned out to change
behaviour. Telling the model *"your own waypoints are infeasible by this much"* moved authored
separation from a 0.484 ceiling to 1.346; post-filter feedback describing a solve that had failed
121/121 times had not. The magnitude finding from §4.2 applies to the model's own geometry, not
just the filter's report.

**The kinematic gap.** The system prompt stated position bounds only — it never mentioned that
drones have a speed or acceleration limit. Authored trajectories demanded up to **37 m/s** against
`vel_max` 1.73 and **370 m/s²** against `acc_max` 1.0. That is the cause of the failed solves in
§4.3. After stating the limits and screening for them, one iteration came back at **5/121** failed
solves.

> This is the same class of bug as the open transitions issue on `feat/swarmgpt2-splines`, where
> the `TRANSITION` keyword lets the LLM schedule a four-metre formation change across half a beat
> and the assembled trajectory bounds at 18.63 m/s against the same 1.73 limit. In both cases a
> layer that picks *timing* was never told what the swarm can physically do. Worth treating as one
> problem.

---

## 6. Environment and commands

None of this is inferable from the repo; this file must stand alone.

- **Two platforms.** Development on Mac (osx-arm64), GPU/deploy on the Linux lab box (linux-64,
  RTX 4090). After any dependency change, confirm `pixi.lock` resolved for linux-64 before pushing.
- **Always `pixi run -e tests` for pytest** — the bare `.venv` lacks scipy.
- **`pixi run -e tests tests` FAILS AT COLLECTION.** A vendored pybind11 tree under `ros_ws` wants a
  `pybind11_tests` module that does not exist. This predates this work (verified by stashing). Use
  the `--ignore` form below; a `norecursedirs` entry in `pyproject.toml` would fix it permanently.
- **OpenAI key** is at `./openai_api_key.sh` (gitignored, untracked, not executable). `source` it;
  nothing reads a `.env`.
- Touching crazyflow: set `SCIPY_ARRAY_API=1` before importing it.
- JAX SIGBUS on the lab box: `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`.

```bash
# The browser demo end to end: pick a song, refine, watch it author a primitive
source ./openai_api_key.sh && pixi run api      # http://127.0.0.1:8000

# Tests (the --ignore is required, see above)
pixi run -e tests pytest tests/unit -q --ignore=tests/unit/ros_ws

# Lint, on every file you touch, before claiming done
pixi run ruff check <files> && pixi run ruff format --check <files>
```

```bash
# One synthesis run. ~10 min, costs API credit.
source ./openai_api_key.sh && pixi run python experiments/synth_to_library.py \
  --request "a double helix: two strands half a turn apart at every height, both winding upward the same way around a common axis" \
  --arm absolute --iters 14
```

Exit `0` promoted, `1` the model never said "keep", `2` a gate refused what it kept. A run log lands
in `synth_runs/promote_<arm>_<stamp>.jsonl` **whatever the outcome**, and a per-iteration table
prints at the end — read those rather than re-running.

---

## 7. Hypotheses ruled out

Each cost real time or API credit. Do not re-investigate.

- **"More iterations will fix it."** 8 unscreened iterations never got authored separation above
  0.484 (needs ≥ 1.0). Not close, and more turns did not help.
- **"The feedback wording is the problem."** Settled by the ablation: absolute beats categorical
  (p = 0.017) and relative (p = 0.015). Wording is not the blocker.
- **"A counter-rotating double helix is buildable."** **Geometrically impossible.** Two strands
  turning opposite ways sweep through each other; a hand-built ideal case measures min separation
  **0.000**. The same geometry with both strands the same handedness, 180° apart, measures
  **1.43–2.15** at identical speed. `REQUESTS[0]` in `ablation_feedback.py` asks for the impossible
  version and three runs were spent failing it. **It was equally impossible for all three arms, so
  the ablation comparison stands** — but say so in the paper before a reader checks the geometry.
- **"The model's own invariants can verify the shape."** 5/5 and 4/4 passed on two flat rings.
- **"`min_sep_norm` alone is a safe promotion gate."** Two iterations reported healthy separation
  (1.062, 1.017) off runs where **95% of solves failed**. Those figures read *better* the more the
  solver gives up.
- **"The assignment helper is unnecessary once the screen exists."** Believed briefly after the
  screen fixed separation on one run. Wrong — assignment is now the sole remaining blocker.
- **Correlation between angle and height as a helix test.** Fails twice over: angle wraps at 2π,
  and in a counter-rotating pair one strand climbs against increasing angle. Replaced by
  monotonicity read around the loop from the extreme, accepting either direction.
- **Gating promotion on zero failed solves.** Rejected a genuine success. axswarm's
  `success=False` means it hit `max_iters` with its **K-step prediction horizon** unsatisfied, but
  only the first step of each horizon is executed and the next tick re-solves. The promoted
  primitive carries 43/121 failures and flies with 0 steps inside the envelope. Gate on
  `steps_inside_envelope`, report `failed_solves`.
- **Pairing tolerance of "half the level spacing".** Drones evenly spaced in a *single* file sit
  exactly on that bound. Replaced by a ratio test: between-level gap ≥ 3× within-pair gap.

---

## 8. Next steps, ranked

1. ~~**Wire synthesis into the frontend.**~~ **Done** — see §11. A refine can now author the
   primitive it needs, and the browser shows the loop failing and fixing itself while it happens.
2. **Verify the demo end to end with `gpt-5.6-terra` authoring.** Everything in §11 was measured on
   `gpt-5.6-luna`, and the last full browser run promoted a primitive but was never watched all the
   way through to playback with the current code. Pick Fearless2, refine with mode `force`, and
   check three things: the panel streams attempts and lands on "flew clear", the choreographer
   emits a call to the new primitive, and the drones visibly make the shape now that the window
   matches. That last one is the open question — the window fix is verified by screening, not by
   eye. **Start here; it is one run and it decides whether anything below matters.**
3. **Close the two name-collision holes in §11** — an in-session clash silently replaces a live
   primitive, and the CLI overwrites tracked evidence files without warning. Both are small, and
   the second risks real result data.
4. **Stop the choreographer crowding a synthesized primitive.** `primitive_window_s()` verifies
   against the narrowest *required*-key gap, but the choreographer may also place actions on
   optional accent beats — which is how a 4.4 s window arose and smeared the shape. The
   announcement asks it to leave room; nothing enforces it. Needs a minimum-interval constraint in
   the schema, or a screen over the composed show.
5. **Synthesize more primitives and re-run the coverage instrument.** This is what turns the
   existence proof into the paper's headline: with the hand library 88% of blind intents fall
   short, so promote N primitives and re-measure the same 88 intents against the extended library.
   The instrument already exists (`experiments/judge_against_vocabulary.py`); the extended library
   is `results/synthesized/` plus the prompt/schema injection, which is already wired.
6. **Write §4.3 into `results/README.md`** — the operating regime and the softened "repairs"
   claim — before the RAL draft is built on the current wording.
7. **Decide what ships for a demo.** A hand-built double helix (radius 1.6 m, 10 levels, 0.75 turns,
   both strands same handedness 180° apart) measures min separation 1.43 and is collision-safe. Shipping that as a *hand-written* primitive decouples "show a double helix on stage"
   from "the LLM authored one unaided" — different claims, very different timelines. **Synthesis is
   library authoring, not a request-time operation; do not run it live for an audience.** Even
   fully working it is minutes per primitive and can legitimately fail.
8. **Rename or delete `results/synthesized/altitude_separated_double_helix.json`.** It is a valid,
   collision-safe primitive that is **two flat counter-rotating rings**, not a double helix. It is
   kept because it is the evidence that the model certifies its own output (5/5 on its own checks),
   but the name will mislead anyone reading the directory.
9. **Cheapest real win outside this loop:** exposing the 30 primitives already in `blocks.py` moves
   expressibility 10% → 19% (McNemar p = 0.023). Prompt and schema only, no new maths. The dominant
   residual after that is **colour palette**, not motion — the lighting family is where the next
   coverage gain is.

---

## 9. Files

`swarm_gpt/synth/`

| file | role |
|---|---|
| `loop.py` | The turn loop. `screen` defaults **off** so the ablation's measured code path is unchanged; `synth_to_library.py` turns it on. |
| `verifier.py` | `authored_trajectory`, `solve_only`, `measure`, `screen_authored` (the pre-solve gate). |
| `manifest.py` | The single declaration; `register()` writes all four places. |
| `sandbox.py` | AST whitelist plus `HELPERS` — the library's own `assign` and `arrival_time`, bound by reference so authored primitives cannot drift from what the hand-written ones use. |
| `feedback.py` | Unchanged: the three ablation arms. |
| `promote.py` | `gate()` (trust) vs `promote()` (trust + persist), `reset_synthesized()`, `load_promoted` for offline tools. |
| `trigger.py` | The per-request gap classifier that decides whether a refine needs synthesis. |
| `refine.py` | Orchestrates classify -> synthesize -> gate -> register for one browser refine. |
| `run_log.py` | The JSONL capture both paths write to the gitignored `synth_runs/`. |

The frontend touches `web/src/{App,api,types}.tsx|ts` and `styles.css`: the synthesis panel,
the mode and authoring-model selects, and the `synthesis_*` / `refine_abandoned` events.

`experiments/synth_to_library.py` is the single entry point (it replaced `synth_single_run.py`,
which logged a run and threw the primitive away). `ablation_feedback.py` and the four coverage
scripts are the experiments behind §4 — `experiments/README.md` indexes them.

`experiments/plot_primitive.py` draws a promoted entry — flown trajectory, the pairing from above,
and the twist against height. `swarm_gpt/data/presets/Fearless2 | 20 | 20260825_160000` is a
minimal preset that renders one through `render.py`; register the manifest before calling
`render_preset` or `primitive_by_name` will not resolve it.

Dock positions in `drones.toml` are a ring at radius 1.5 m. The active swarm is **10 drones**
(`cf11`-`cf15`, `cf21`-`cf25`), 0.927 m apart; the other ten keep their `addr`/`channel` and come
back by adding them to `active` and re-running the ring layout. A ring makes every radial formation a straight in-or-out flight with no crossing paths,
which is the failure that cost five runs. Only `pos` was changed; `addr` and `channel` are as they
were.

Tests: `tests/unit/test_synth_{schema,verifier,sandbox,feedback,promote}.py`, plus the
refine and model-list cases in `test_api.py` and `primitive_window_s` in `test_backend.py`.
843 pass.

Two artifacts are kept deliberately: `results/synthesis-rejected/double_helix-...selfcertified.json`
is outside the load path because it is the *evidence* for the self-certification finding, and
`results/synthesized/altitude_separated_double_helix.json` is the promoted-before-gate-3 case in
step 5 above.

---

## 10. People

**Marcel** is the go-to on all of it. He is leading the next SwarmGPT publication, shaped the
direction of every thread, and has a current picture of where this stands.

**Martin** wrote the original SwarmGPT and knows this area well.

This thread has no second owner. It is the proposal plus the three coverage instruments, the
feedback ablation, and the synthesis loop in §5 — self-contained enough for a student to pick up.
I'm reachable at yiyi.xu@mail.utoronto.ca after I leave.

---

## 11. The frontend path

All on this branch. The lifetime rule below is the load-bearing decision; the three pieces the work
was scoped as -- loading, triggering, progress -- follow from it.

**A refine's primitive lives only as long as the choreography that asked for it.** This is the
rule the rest of the frontend path is built around, and it is a deliberate reversal of the obvious
design. Selecting a song calls `reset_synthesized()`, which clears the runtime registries; a
primitive authored during a refine is registered in memory, used by that choreography, and gone the
moment another song is picked. The running app **never** reads `results/synthesized/` — there is no
startup load — and a refine **never** writes to it.

Two reasons, and the second is the one that matters:

- Persistence quietly kills the demo. With `upright_heart_outline` on disk and loaded, the
  classifier is shown it in the catalogue and correctly answers "covered" for *"put a heart at the
  drop"* — no synthesis, no loop, nothing to watch. Each demo request burned itself out after one
  use. Verified directly: heart → COVERED, butterfly → GAP.
- Authoring the library is a deliberate act, not a side effect of someone refining a show. The CLI
  is where that happens.

So the split is: `gate()` decides whether a run may be trusted, `promote()` is `gate()` plus
persistence. The browser calls `gate()`; only `experiments/synth_to_library.py` calls `promote()`.
Every run, from either path, writes a JSONL record to the gitignored `synth_runs/` via
`synth/run_log.py` — capture is not gated, and that record is not a load path.

`load_promoted` still exists for offline tools that need a promoted entry resolvable (rendering the
tracked preset that contains the heart, for instance). Nothing in the running app calls it. It
skips any entry whose `provenance.n_drones` differs from the active swarm — `double_helix.json`
hardcodes 20 drone indices and would misfire on the 10-drone ring.

**One assumption worth knowing.** The synthesized registries are module-level globals, so clearing
them on song selection assumes one choreography is being worked on at a time. That is what this
local single-user app intends, but two concurrent jobs would tread on each other. Making the
registry per-backend is the real fix if that ever matters.

**The trigger is a per-request classifier**, `synth/trigger.py`. It is shown the signature of every
primitive that exists — built from the same two tables the response schema is built from, so it
cannot drift — plus one refine message, and asked whether that message needs a primitive not on the
list. This is deliberately *not* the introspective question of §4.1: it is a judgement about one
concrete case, and it works where introspection did not. Measured before it was built on, in
`results/coverage/gap-classifier.json`: **15/17 on the unambiguous labels, and 0 false positives.**
It under-fires — it missed "the outline of a cube" and judged "a DNA double helix" covered because
`helix` is on the list, which is the §4.1 failure mode surviving in weaker form. Under-firing is
the right direction, since a false positive spends minutes of synthesis on nothing.

Because it under-fires, the refine box also carries an explicit mode: *auto* (classifier), *force*
(always), *off* (never). **A demo should use `force`, or lean on the fact that a primitive already
promoted needs no synthesis at all.** §8.4 still stands: this is library authoring, and running it
live for an audience is a minutes-long bet that can legitimately fail.

**Progress is streamed, not spun.** `SynthesisLoop.run` takes an `on_iteration` callback; the API
turns each turn into a `synthesis_iteration` event carrying stage, authored and flown separation,
and steps inside the envelope, and the browser renders one line per attempt — "rejected before
flying: the drones reach 0.48 of the 1.00 spacing they need", then "flew clear". Watching it fail
and fix itself is the interesting part for a viewer.

**A failed synthesis abandons the refinement; it does not fall back.** This was tried the other way
first and it is worse: asked for a heart with no heart primitive, the choreographer approximates the
shape one drone at a time with `move`, which is hand-authoring the primitive badly and is exactly
what the coverage work in §4.1 says not to count as expressing it. So when synthesis is attempted
and fails — the model never accepts, a gate refuses what it kept, or it raises — the API emits
`refine_abandoned`, leaves the choreography untouched, and the UI returns to ready with the refine
box open. Nothing is lost: the previous choreography is still playable and deployable. A request
the library genuinely covers (`NO_GAP`) is not a failure and proceeds normally.

**Five bugs the browser path surfaced.** None are frontend bugs; the loop had simply only ever run
on the main thread of a short-lived CLI process, and only ever registered a primitive by reloading
it from JSON.

- **`signal.SIGALRM` cannot be installed off the main thread.** Refine jobs run in a worker, so the
  first browser run died with `ValueError: signal only works in main thread`. `call_guarded` now
  falls back to a join-with-timeout on a daemon thread. A runaway call is abandoned rather than
  interrupted — Python cannot stop another thread — so it leaks a thread until exit. The loop
  staying alive is what that buys; the AST whitelist is the guard that actually matters. **The
  main-thread path is byte-for-byte unchanged, so the ablation's code path is untouched.**
- **A runaway reply killed the whole run.** Twice, on a hard revise turn, the model returned tens of
  kB of unparseable text and `json.loads` raised out of `_call`. It is now a `SynthError`, which the
  loop already knows how to turn into feedback, and the runaway text is kept **out** of the message
  history — replaying it is what makes the next turn run away too.
- **Retries stack.** The SDK default pairs a 600 s read timeout with two retries, so one stalled
  call could hold a browser job for half an hour with nothing to show. Cut to one retry.
- **A turn is minutes, and the panel looked frozen between them.** `run()` also takes an
  `on_authoring` callback, so the UI can say which attempt is currently with the model.
- **A run that cleared both gates was thrown away at the last step.** `Iteration.manifest` is
  `dataclasses.asdict(manifest)`, which keeps `params` a **tuple**, and `from_payload` required a
  `list` — so registering straight off a loop record failed with "Manifest field 'params' must be a
  non-empty array", which reads like the params were missing. The CLI never hit it because
  `promote()` writes with `json.dumps` first, turning the tuple into an array; that is also why
  everything already in `results/synthesized/` reloads fine. `from_payload` now takes either.

**Open, and not to be trusted yet.** The classifier call carries a 90 s timeout so a slow API cannot
hold a refine open. `APITimeoutError` fires correctly in isolation (verified at a 1 ms timeout), but
during a degraded-API window a live classifier call was observed running well past 90 s without
raising, and I could not explain it before the window closed. Check this before relying on it.

**The verification window is the show's window, not an arbitrary 12 s.** This was the bug behind
"the drones do not really make the shape". A primitive plays from its own key until the next
action's, but synthesis verified every one over a fixed `duration_s = 12.0`. Composed into
Fearless2 the same primitive got 4.4 s and demanded 2.06 m/s against the 1.73 limit; at one bar,
4.12 m/s and 11.09 m/s^2 against 1.0. Separation was never the problem -- it stayed at 1.62 -- so
both gates passed and the filter still had to smear the shape away.

`AppBackend.primitive_window_s()` now returns the narrowest gap between required keys for the
selected song (8.18 s for Fearless2), the API passes it, and `synthesize_for_refine` takes
`duration_s` with **no default**: verifying against a window the primitive will not get certifies
nothing. Re-screened at 8.18 s, the `heart_drop` from the 2026-08-26 run needs 1.11 m/s and
passes.

Two supporting changes, because measuring the right window is not enough on its own:

- The synthesis prompt now states the interval and forbids a duration parameter. Not knowing the
  window, the model had been inventing one -- `heart_drop` declared `duration_s: float [30, 60]`
  over a body that does `total = min(duration_s, tend - tstart)`, so it was always clamped away.
- The announcement handed to the choreographer carries the interval the primitive was verified
  over and tells it to leave that much room. The choreographer may place actions on optional
  accent beats between required keys, which is how the 4.4 s window arose in the first place;
  nothing enforces this yet, so it is instruction rather than guarantee.

**Two things this leaves open.** The CLI still takes `--duration`, default 12 s, so a primitive
authored there carries the same mismatch unless the flag is set to the target song's window. And
the choreographer can still crowd a synthesized primitive onto a tighter key; making that
impossible means either a minimum-interval constraint in the schema or a screen over the composed
show, neither of which exists.

**The authoring model is chosen separately from the choreography model.** Writing a primitive is a
harder job than picking calls out of a catalogue, and the working hypothesis is that a model good
enough for one is not automatically good enough for the other. The refine box carries its own
model select beside the synthesis mode; `/api/llm` serves `synthesisModels` alongside the
choreography list, and the refine payload carries `synthesisModelId`. **`gpt-5.6-terra` is offered
for authoring only and is deliberately absent from the choreography list**, and it is currently
the authoring default — change the order of `DEFAULT_SYNTHESIS_MODEL_CHOICES` if that should be
`luna`. Every run so far in §2 and §11 was `gpt-5.6-luna`, so terra is untested here.

**Nothing stops the model reusing a name.** Both registration guards only check hand-written
primitives — `hasattr(motion_primitives, name)` and `name in _PRIMITIVE_ARG_ORDER` — so neither
consults `results/synthesized/`. Investigated, not fixed, in three parts:

- Naming a primitive after an archived one is **harmless in-session**: the archive is not loaded,
  so there is no clash, and the browser never writes.
- A **second refine in the same job** naming its primitive after the first silently replaces it —
  measured going from `n_args` 3 to 1. If the earlier choreography already calls the old one, that
  call now resolves to a different function, raising `LLMFormatError: Wrong number of arguments`,
  or worse, flying different geometry at the same arity.
- **The CLI overwrites archived entries without warning.** `promote()` writes
  `out_dir / f"{name}.json"` unconditionally; a run that picks the name `double_helix` destroys the
  tracked file. §9 keeps `double_helix.json` and `altitude_separated_double_helix.json` as evidence
  for the self-certification finding, so that is real result data. Suggested: refuse to overwrite
  without `--force`, and make an in-session name clash an error the loop feeds back to the model.

**Timing, measured, and worse than §6 implies.** The synthesis runs in §2 were roughly a minute an
iteration. In the browser runs on 2026-08-25 the API was degraded and iterations took several
minutes each, making a 14-iteration `force` run a 45-60 minute bet rather than ten. A crescent-moon
request was also watched failing the pre-solve screen with authored separation *falling* across
attempts (0.65, 0.53, 0.45) — ten drones on one vertical arc may be infeasible the way the
counter-rotating helix is.
