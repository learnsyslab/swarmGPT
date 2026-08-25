# Results — verified primitive synthesis

Data behind the 2026-08-17 experiments, kept because the paper needs it and the runs cost hours of
API time to reproduce. Raw scratch output stays in the gitignored `synth_runs/`; this directory
holds only what a result rests on.

Every number below came from `gpt-5.6-luna` against the 10-drone lab swarm.

---

## 1. Does the primitive library actually bind?

Three instruments, deliberately biased in different directions. Two agree; the first one is broken
and is kept because the failure is itself a finding.

### `coverage/introspective-null.json` — the instrument that does not work

Asked the model, holding the full production prompt, which moments the library could not express.
**26 probes, 26 empty answers, zero gaps.** It rationalises coverage instead of reporting a gap
("the library can express this using gentle spirals, helices, rotations").

A positive control settles that this is the instrument's fault, not the library's: asked directly
"can you express a double helix with these primitives?", the same model says no and explains why
(`helix` takes no drone subsets and no rotation direction). **Never ask an LLM what capability it
is missing.**

### `coverage/gap-classifier.json` — asking about one request instead of about itself

The instrument above fails because self-examination is the wrong question. The frontend needs the
same information at refine time, so it asks a narrower one: *here is one user request and the list
of primitives that exist — does this need one that is not on the list?* Judgement about a concrete
case, not introspection.

20 hand-labelled refinement messages, 1 run each, against the hand-written library. On the 17 the
author can label with confidence: **15/17 correct (88%)**, and the breakdown is what matters more
than the total — **0 false positives** and 2 false negatives. It never invented a gap for a
request the library covers, which is the expensive direction: a false positive spends minutes of
synthesis on nothing. It misses gaps instead, twice — "the outline of a cube", and "a DNA double
helix", which it judged covered because `helix` is on the list. That second miss is the
introspective failure mode surviving in weaker form.

One probe in 20 returned ~98 kB of unparseable text instead of a verdict. The classifier treats an
unreadable answer as "no gap" so a runaway response cannot block the refinement it was asked
about.

Because it under-fires rather than over-fires, the refine box also carries an explicit override. A
demo that must show synthesis should use it rather than depend on a judgement call.

The catalogue it is shown is whatever is registered at that moment, which is why a primitive
authored during a refine is discarded when another song is selected: a promoted heart in the
catalogue makes the classifier answer "covered" for a heart request, correctly, and there is then
nothing to demonstrate. Verified both ways -- with the heart registered, "put a heart at the drop"
returns covered while "a butterfly at the bridge" still returns a gap.

### `coverage/decoy.json` — revealed preference

13 fake motion primitives and 6 fake lighting primitives added to the prompt *and* the output
schema, then never executed, only counted. **35 of 39 probes chose a motion primitive that does not
exist.** Recurrence across 13 songs: `double_helix` 12, `color_wave_from` 12, `fade` 10, `split` 9,
`bloom` 7, `form_heart` 6, `twinkle` 5, `ripple` 4.

The control that makes it interpretable: decoys come in two classes. **Gap** decoys add capability;
**redundant** decoys are renamed duplicates (`form_ring`=`form_circle`, `corkscrew`=`helix`,
`breathe`=`pulse`). Raw per-name usage favours the redundant ones, so usage alone proves nothing —
what settles it is whether a duplicate *replaced* its twin. It did not: `turn` was never used
without `rotate`, `form_ring` twice against `form_circle`'s 22, `ascend` once against `move_z`'s 26.
Validity checks: `form_letter` got 0 uses (correct — `move` already places drones arbitrarily) and
`arc_swap` got 0 despite being a real gap, so the model is not grabbing everything offered.

Bias: offering an item makes it more attractive than its absence makes it missed. Over-states need.

### `coverage/unanchored.json` — unanchored elicitation

A plain-language choreography plan elicited with **no primitive list in the prompt** (`blind`), then
again with the list added and nothing else changed (`anchored`). A **separate** judge rates each
intent expressible/partial/not, seeing one intent and the library, never the song, the surrounding
plan, or the condition. The planner never grades itself — that is what failed above.

88 intents per condition, 13 songs. Same direction in **13/13 songs** (sign test p≈0.0002).

Bias: a plan written without knowing what is buildable is unconstrained. Over-states the gap
differently from the decoy menu. Two instruments failing in different directions is why the
conclusion holds.

### `coverage/rejudge-move-rule.json` — the correction that moved the headline

The judge was crediting `move(x,y,z,drone_id)` with expressing multi-drone shapes. Invalid twice
over: `move` is emitted **zero times** across 39 real choreographies, and covering a shape with one
`move` per drone *is* hand-authoring the primitive inline. A paired re-judge disallowing it:

| condition | shortfall before | after |
|---|---|---|
| blind | 67% | **88%** |
| anchored | 24% | **24%** (unchanged) |
| delta | +43.2 pp | **+63.6 pp** |

Validation: of 29 blind "expressible" verdicts, 22 flipped and **20 of those 22 had cited `move`**;
of the 7 that held, 1 had. The rule hit what it aimed at and was inert where it should be.

**Judge noise floor: ±7 pp.** Anchored churned 12/88 verdicts netting exactly zero. The blind net
shift of 20 pp is 3× that. Every other figure here should be read against that floor.

### `coverage/vocabulary-*.json` — how much is missing capability vs unwired capability

The same 88 blind intents judged against three libraries. Blind intents carry no vocabulary, so
this is paired: the library is the only variable.

| library | expressible | shortfall |
|---|---|---|
| `current` — 12, incl. `move`/`swap` | 10% | 90% |
| `sg2` — 12, `move`/`swap` deleted | 9% | 91% |
| `sg2_full` — all 30 in `blocks.py` | **19%** | **81%** |

Exposing the 18 already-written primitives is real (**McNemar p = 0.023**, 11 gained, 2 lost) and
worth ~10 points. **81% still falls short with everything the lab has already written.**

Recurring reasons among the 71 still short under `sg2_full`: **colour palette 47**, colour-over-time
/ fade 12, subset control 11, arbitrary or organic shape 8, easing 3. The dominant residual is
lighting, not motion. Counts come from a keyword regex over the judge's prose, so the ordering is
solid and the exact split is not.

---

## 2. Does the *content* of solver feedback matter?

`feedback-ablation/ablation-54run.jsonl` — 6 requests × 3 arms × 3 repeats. All three arms read one
identical measurements dict, so they differ in wording alone and no arm is handed less information.

- `categorical` — what swarmGPT sends today: who and roughly when, no magnitudes
- `absolute` — the same events in metres
- `relative` — the same magnitudes as ratios and comparatives, no units anywhere

Primary outcome (`deviation_max`) was fixed in the script docstring before any data existed.

| arm | dev_max median (IQR) | checks pass | model said "keep" |
|---|---|---|---|
| categorical | 0.35 (0.19–1.68) | 0.67 | **0/17** |
| **absolute** | **0.17 (0.13–0.28)** | **1.00** | **8/18** |
| relative | 0.46 (0.16–1.58) | 1.00 | 1/17 |

Mann-Whitney one-sided: absolute < categorical **p = 0.017**, absolute < relative **p = 0.015**.
Fisher exact on convergence: vs categorical **p = 0.0019**, vs relative **p = 0.011**. Absolute
wins 5 of 6 requests.

Two things matter more than the medians. Absolute's IQR is tight while the others swing past 2.6 —
magnitudes suppress the catastrophic runs, they do not merely shift the average. And categorical
converged **zero** times: without numbers the model never reaches a state it will accept.

**`relative` is significantly worse than `absolute`.** Stripping units and substituting "about half
the separation they need" hurt. That is the opposite of the "LLMs are bad with numbers" objection
this arm was built to test.

Caveats: n≈17 per arm, one model, six requests. Two runs of the same arm on the same request once
gave 0.16 and 1.68 — the variance is why a 3-run comparison told us nothing.

---

## Reproducing

Scripts live in `experiments/`. Each writes to the gitignored `synth_runs/`; promote anything a
result depends on into this directory by hand.
