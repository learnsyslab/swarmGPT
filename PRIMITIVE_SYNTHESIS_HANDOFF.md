# Verified primitive synthesis — continuation handoff

Branch `feat/primitive-synthesis`, pushed to `origin`. This document plus that branch is the
whole thread; nothing else is required to pick it up.

Written by **Yiyi Xu** (yiyi.xu@mail.utoronto.ca), LSY Lab TUM, 19 May – 27 August 2026.
Supervisor: **Marcel**. Collaborators: **Martin**, **Alex**.
Last updated **2026-08-26**. Tracked on this branch, so a clone carries it.

> **2026-08-26, late:** the loop was rebuilt. The model now authors **only the equation of a
> shape**; the whole trajectory-authoring path is deleted. Section 2 is the new state, §5 is why,
> and every section below it has been brought in line. If you read an earlier copy of this file,
> the thing that changed is that a primitive is no longer a function of time.

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
its first verified primitives.

**One thing the write-up must say plainly.** For most of this thread the LLM was asked to author a
whole trajectory, and it was bad at it. It is not bad at authoring *geometry*. Separating the two
is the finding, and it is the difference between a loop that converges in four iterations and one
that spends fourteen failing a screen. The claim in half 2 survives the change — the solver still
teaches, and magnitudes still win (§4.2) — but the thing being taught is now the right size.

---

## 2. Current state

**A primitive is a shape, and nothing else.** The model writes

```python
def NAME(params, n_drones):
    ...
    return positions      # (n_drones, 3), centimetres
```

and `synth/shape.py` wraps it in exactly the body `form_circle` already has: Hungarian assignment
onto the target points, then `_formation_waypoints` to schedule the arrival at the library's own
speed budget and hold the pose for the rest of the interval. The model never writes a waypoint,
never picks a time, and never chooses which drone flies where.

**Why this replaced the old design.** The model used to author the whole five-argument primitive,
timing and all. Across the 13 runs in `synth_runs/`, **95 of 141 iterations died at the pre-solve
screen**, and every cause was in the trajectory layer rather than the geometry: acceleration from
hand-rolled interpolation (14), separation *during the fly-in* at t = 2.9, 4.6, 5.9 s (12),
waypoint-contract errors (4), speed (2). The shape was rarely the problem. `form_circle` cannot
fail any of those ways because it does not write a trajectory, and neither can a synthesized
shape now.

**It works from the browser**, on the same path as before: a refine that asks for a shape the
library cannot express authors it, verifies it, registers it, and regenerates the choreography
with it. §11 is that path.

**Measured on the new loop.** `"a heart shape at the beat drop"`, `gpt-5.6-luna`, 8.18 s window,
10 drones: promoted in **4 iterations** — one geometry rejection, then three flights, closing on
flown separation **1.219**, **0 steps inside the envelope**, 3 failed solves. The old path on the
same request took 6–14 iterations, most of them never reaching the solver, and one 14-iteration
run never promoted at all. What it wrote is seven lines:

```python
def form_heart(params, n_drones):
    size_cm, z_cm, lean_deg = params
    t = np.linspace(0, 2 * np.pi, n_drones, endpoint=False)
    x = size_cm * np.sin(t) ** 3
    v = size_cm * (13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t)) / 29
    lean = np.deg2rad(lean_deg)
    return np.stack([x, v * np.sin(lean), z_cm + v * np.cos(lean)], axis=-1)
```

It generalises over swarm size — unmistakable at 20 and 40 drones — which is the property the
whole design is for. It also invented `lean` and converged to **80 degrees**, i.e. a nearly
*horizontal* heart. That is the correct read of the room, not a failure: the collision envelope is
0.6 m deep in z against 0.25 m in x/y, and there are only 1.45 m of usable height, so an upright
shape buys about two levels. **If a demo needs the heart standing up for an audience, that is a
constraint of the arena to design around, not something more iterations will fix.**

**Whether a primitive looks like what was asked for is deliberately not gated.** A hand-written
shape predicate was built and then removed, for three reasons:

- **A primitive already is the geometry.** The LLM's contribution is "where does each drone sit
  for this shape"; the solver moves them. A predicate is the same kind of artifact — a geometric
  characterisation — so grading one with the other is circular.
- **It cannot work for a live request.** A predicate must exist before the shape is asked for. The
  demo case is someone asking for something nobody anticipated.
- **It was wrong more often than the model was.** It demanded counter-rotation, which makes a
  double helix physically impossible, and it rejected a rational response to a heart request.

The model's *own* invariant checks are gone too, for the reason in §7: it passed 5/5 and 4/4 of
them on two flat counter-rotating rings. Nothing is lost by deleting them.

Two gates remain, both objective, both from the hardware and the room rather than from anyone's
idea of a shape, and a third that is cheaper than either:

1. **The shape screen** (`screen_shape`, ~µs). Every pair of target points must clear the
   collision ellipsoid. Feedback names the two points and the gap in centimetres.
2. **The pre-solve screen** (`screen_authored`, 1.4 ms). The assembled trajectory must be flyable
   — in practice this now only fires when the shape is out of reach in the window it gets.
3. **The filter.** Every solve must run, and flown separation must clear the envelope.

Whether it looks right is a human call. Run logs stay local — `synth_runs/` is gitignored.

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

## 5. How the loop is built

**One manifest, four places.** A primitive's signature lives in the prompt, the structured-output
schema, the backend function, and the offline check — the standing invariant is that changing
fewer than all four causes bugs. `PrimitiveManifest.register()` writes all of them from one
declaration, so a synthesized primitive is emittable by the choreographer LLM, renderable as a
call, and visible in the prompt catalogue, without any of them able to drift.

**The model authors geometry; the library does the flying.** This is §2's headline and it is the
load-bearing decision. `shape.py:as_primitive` is the whole wrapper, and it is `form_circle`'s
body. Three classes of failure become structurally impossible rather than merely discouraged:

- **Speed.** `_formation_arrival_time` sizes the arrival to the bottleneck drone at 1.0 m/s with
  headroom. The only way to break the limit now is a shape too far from the docks to reach in the
  window, which is a real fact worth being told.
- **Fly-in collisions.** `_assign_positions` is the library's own Hungarian assignment. Not a
  guarantee, but it is exactly what every hand-written formation relies on.
- **The waypoint contract.** There is no waypoint to get wrong.

**The prompt states the collision envelope, which it never used to.** The old prompt gave arena
bounds and the speed/acceleration limits but never mentioned how far apart two drones must stay.
The shape prompt gives the ellipsoid in centimetres — **25 cm side by side, 60 cm one above the
other** — and says why that makes an upright shape expensive against 145 cm of usable height.
That fact is the one a shape author most needs, and stating it is why `form_heart` reached for a
`lean` parameter unprompted.

**Acceleration is measured but never gated on.** This was a false positive that cost real
iterations. Waypoints are a piecewise-linear reference, so the arrival is a velocity corner: a
heart topping out at 0.77 m/s reads **5.79 m/s²** at exactly the arrival waypoint. The MPC exists
to smooth that corner, the library schedules it rather than the model, and gating on it rejected
good geometry. The figure is still reported.

**The magnitude finding applies to the model's own geometry, not just the filter's report.**
Telling the model *"your own points are this far inside the envelope"* is what moved authored
separation off its old 0.484 ceiling; post-filter feedback describing a solve that had failed
121/121 times had not. This is §4.2 one layer earlier, and it is why the shape screen reports the
gap in centimetres rather than saying "too close".

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
# One synthesis run. A few minutes, costs API credit. Set --duration to the window the target
# song will actually give the primitive (Fearless2 is 8.18 s) -- the default 12 s certifies
# nothing about a show that hands it 4 s.
source ./openai_api_key.sh && pixi run python experiments/synth_to_library.py \
  --request "a heart shape at the beat drop" --arm absolute --iters 6 --duration 8.18

# Look at what it promoted: the flown pose, and the equation at 10, 20 and 40 drones.
pixi run -e tests python experiments/plot_primitive.py results/synthesized/form_heart.json
```

Exit `0` promoted, `1` the model never said "keep", `2` a gate refused what it kept. A run log lands
in `synth_runs/promote_<arm>_<stamp>.jsonl` **whatever the outcome**, and a per-iteration table
prints at the end — read those rather than re-running.

---

## 7. Hypotheses ruled out

Each cost real time or API credit. Do not re-investigate.

- **"The model is bad at authoring primitives."** It is bad at authoring *trajectories*. Given the
  same requests as geometry alone it converges in a handful of turns. This was the expensive one:
  most of this thread was spent tuning feedback for a task that was the wrong shape.
- **"More iterations will fix it."** Under trajectory authoring, 8 unscreened iterations never got
  authored separation above 0.484 (needs ≥ 1.0). Not close, and more turns did not help. This is
  what "the task is the wrong shape" looked like before it was diagnosed.
- **"Gating the authored acceleration catches bad primitives."** It catches the arrival corner of
  every piecewise-linear reference, `form_circle`'s included once the swarm has any real distance
  to travel. Measured: 5.79 m/s² against a 1.0 limit for a heart that never exceeds 0.77 m/s, with
  the peak at exactly the arrival waypoint. Report it; do not gate on it.
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
- **"The assignment helper is unnecessary once the screen exists."** Wrong, and now moot: the
  wrapper always assigns, so the model cannot skip it.
- **Gating promotion on zero failed solves.** Rejected a genuine success. axswarm's
  `success=False` means it hit `max_iters` with its **K-step prediction horizon** unsatisfied, but
  only the first step of each horizon is executed and the next tick re-solves. The promoted
  primitive carries 43/121 failures and flies with 0 steps inside the envelope. Gate on
  `steps_inside_envelope`, report `failed_solves`.

---

## 8. Next steps, ranked

1. **Watch a browser refine all the way to playback.** Everything in §2 was measured through the
   CLI. Pick Fearless2, refine with mode `force`, and check three things: the panel streams
   attempts and lands on "flew clear", the choreographer emits a call to the new primitive, and
   the drones visibly make the shape. **Start here; it is one run and it decides whether anything
   below matters.**
2. **Decide what an upright shape is worth.** `form_heart` chose to lie nearly flat, which is
   right for the envelope and possibly wrong for an audience. Either accept horizontal shapes
   (seen from a raised camera or a balcony), or state uprightness in the request and accept that
   ten drones give you about two levels. This is a staging decision, not a code one.
3. **Stop the choreographer crowding a synthesized primitive.** `primitive_window_s()` verifies
   against the narrowest *required*-key gap, but the choreographer may also place actions on
   optional accent beats — which is how a 4.4 s window arose and smeared a shape. The announcement
   asks it to leave room; nothing enforces this. Needs a minimum-interval constraint in the schema,
   or a screen over the composed show.
4. **Close the two name-collision holes in §11** — an in-session clash silently replaces a live
   primitive, and the CLI overwrites files in `results/synthesized/` without warning.
5. **Synthesize more primitives and re-run the coverage instrument.** This is what turns the
   existence proof into the paper's headline: with the hand library 88% of blind intents fall
   short, so promote N primitives and re-measure the same 88 intents against the extended library.
   The instrument already exists (`experiments/judge_against_vocabulary.py`). **This is much more
   tractable now than it was** — four iterations a primitive rather than fourteen — so N in the
   dozens is a realistic afternoon rather than a week.
6. **Write §4.3 into `results/README.md`** — the operating regime and the softened "repairs"
   claim — before the RAL draft is built on the current wording. Add the §5 diagnosis to it too:
   the ablation ran against trajectory authoring, and that is now part of how its numbers read.
7. **Decide what ships for a demo.** **Synthesis is library authoring, not a request-time
   operation; do not run it live for an audience.** Even at four iterations it is minutes and it
   can legitimately fail. Author ahead of time, ship the JSON.
8. **Cheapest real win outside this loop:** exposing the 30 primitives already in `blocks.py`
   moves expressibility 10% → 19% (McNemar p = 0.023). Prompt and schema only, no new maths. The
   dominant residual after that is **colour palette**, not motion.

## 9. Files

`swarm_gpt/synth/`

| file | role |
|---|---|
| `shape.py` | `as_primitive` (the `form_circle` wrapper), `targets`, and `screen_shape`. The heart of the current design. |
| `loop.py` | The turn loop and the authoring prompt. `screen` defaults **off** so the ablation's measured code path is unchanged; `synth_to_library.py` and the refine path turn it on. |
| `verifier.py` | `authored_trajectory`, `solve_only`, `measure`, `screen_authored`. |
| `manifest.py` | The single declaration; `register()` writes all four places. |
| `sandbox.py` | AST whitelist and `compile_shape`. Nothing but numpy and safe builtins is reachable from authored code. |
| `feedback.py` | The three ablation arms. |
| `promote.py` | `gate()` (trust) vs `promote()` (trust + persist), `reset_synthesized()`, `load_promoted` for offline tools. |
| `trigger.py` | The per-request gap classifier that decides whether a refine needs synthesis. |
| `refine.py` | Orchestrates classify -> synthesize -> gate -> register for one browser refine. |
| `run_log.py` | The JSONL capture both paths write to the gitignored `synth_runs/`. |

The frontend touches `web/src/{App,api,types}.tsx|ts` and `styles.css`: the synthesis panel, the
mode and authoring-model selects, and the `synthesis_*` / `refine_abandoned` events. A rejected
geometry arrives as stage `shaped`, a rejected trajectory as `screened`, a flown one as `measured`.

`experiments/synth_to_library.py` is the single entry point. `experiments/plot_primitive.py` draws
a promoted entry: the pose it flew into over its flight paths, and the equation sampled at 10, 20
and 40 drones. `ablation_feedback.py` and the four coverage scripts are the experiments behind §4;
`experiments/README.md` indexes them.

**`results/synthesized/trajectory-era/` holds four entries the removed path authored** — the first
double helix, the upright heart, the self-certified two-rings case, and one render. They no longer
load, deliberately: `PrimitiveManifest` parses shapes only, and that directory sits outside
`load_promoted`'s glob. They are evidence for §4 and §7, not library entries. Its README says which
is which. `results/synthesis-rejected/double_helix-...selfcertified.json` is kept for the same
reason.

**One casualty worth knowing about.** `ablation_feedback.py` still runs, but it now drives shape
authoring, so it no longer reproduces `results/feedback-ablation/ablation-54run.jsonl` verbatim.
That data stands as recorded; re-running the script measures the current loop, which is a
different (and probably more interesting) experiment. Its self-check secondary outcome is gone
along with the invariants.

Dock positions in `drones.toml` are a ring at radius 1.5 m. The active swarm is **10 drones**
(`cf11`-`cf15`, `cf21`-`cf25`), 0.927 m apart; the other ten keep their `addr`/`channel` and come
back by adding them to `active` and re-running the ring layout. A ring makes every radial formation
a straight in-or-out flight with no crossing paths, which is the failure that cost five runs.

Tests: `tests/unit/test_synth_{shape,schema,verifier,sandbox,feedback,promote}.py`, plus the
refine and model-list cases in `test_api.py` and `primitive_window_s` in `test_backend.py`.
840 pass.

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
it from JSON. All five still apply — they are about the loop's plumbing, not about what it authors.

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
  `promote()` writes with `json.dumps` first, turning the tuple into an array. `from_payload` now
  takes either.

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

- The prompt states the interval and forbids a duration parameter. Not knowing the window, the
  model had been inventing one. Under shape authoring this is close to moot -- there is no time in
  a shape function at all -- but the interval still decides whether the swarm can *reach* the
  shape, so it is still stated and still verified against.
- The announcement handed to the choreographer carries the interval the primitive was verified
  over and tells it to leave that much room. The choreographer may place actions on optional
  accent beats between required keys, which is how the 4.4 s window arose in the first place;
  nothing enforces this yet, so it is instruction rather than guarantee.

**Two things this leaves open.** The CLI still takes `--duration`, default 12 s, so a primitive
authored there carries the same mismatch unless the flag is set to the target song's window. And
the choreographer can still crowd a synthesized primitive onto a tighter key; making that
impossible means either a minimum-interval constraint in the schema or a screen over the composed
show, neither of which exists.

**The authoring model is chosen separately from the choreography model.** Writing the equation of
a shape is a different job from picking calls out of a catalogue, and the working hypothesis is that a model good
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
