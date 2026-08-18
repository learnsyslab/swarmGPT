# Experiments

Research scripts behind `results/`. Not part of the shipped system: nothing in `swarm_gpt/` imports
these. Each writes to the gitignored `synth_runs/`.

## Coverage — does the primitive library bind?

| script | what it measures |
|---|---|
| `probe_introspective.py` | **Kept as a negative result. Do not trust its output.** Asks the model which moments the library could not express. Returns zero gaps, always, because it measures willingness to volunteer a complaint rather than coverage. |
| `probe_decoy.py` | Revealed preference: primitives that do not exist are added to the prompt and schema, never executed, only counted. Two decoy classes — `gap` adds capability, `redundant` renames an existing primitive — because usage alone cannot separate need from novelty. |
| `probe_unanchored.py` | Elicits a plan with no vocabulary in the prompt, then again with it, and has a *separate* judge rate each intent. The planner never grades its own work. |
| `judge_against_vocabulary.py` | Re-judges an existing run's intents against another library (`--vocabulary`) or with per-drone `move` placement disallowed. Blind intents carry no vocabulary, so judging the same ones against several libraries is paired. |
| `vocabularies.py` | The three libraries a probe can be run against: this tree's prompt, `swarmgpt2-spline-foundation`'s prompt, and all 30 primitives in that branch's `blocks.py`. The last is generated from source by AST so it cannot drift from the code. |

## Synthesis — does feedback content matter?

| script | what it does |
|---|---|
| `ablation_feedback.py` | The grid: requests × feedback arms × repeats. Primary outcome fixed in the docstring before the data existed. Appends to JSONL per run so an interrupted sweep keeps what it had. |
| `synth_single_run.py` | One synthesis loop, one arm. For debugging a single request; `ablation_feedback.py` is the experiment. |

## Utilities

| script | what it does |
|---|---|
| `inspect_runs.py` | `status`, `gaps`, `plans`, `decoys`, `ablation` views over the newest result file. Resolves paths against the repo, so it runs from any directory. |
| `report_unanchored.py` | Renders an unanchored run as readable markdown, including every judge verdict in full. |

## Gotchas

- Sibling scripts import each other by `sys.path`, so run them as files, not as modules.
- `probe_decoy.py` and `probe_unanchored.py` write only at the end. A long run that dies loses
  everything; `ablation_feedback.py` flushes per run and does not.
- Glob carefully when comparing vocabulary runs: `rejudge_sg2_*` also matches `rejudge_sg2_full_*`.
