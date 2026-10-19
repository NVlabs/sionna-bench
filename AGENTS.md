# Codex Workflow For Paper-To-Sionna Reimplementation

This repository is a pipeline for reimplementing algorithms from academic
papers as benchmarkable Sionna components. The core workflow uses two roles:

- `implementer`: studies the benchmark harness, reads the paper, writes a
  detailed reviewed plan, implements approved methods, runs smoke tests, and
  reports all deviations.
- `reviewer`: checks the plan and code against the paper, rejects unjustified
  adaptations, accepts or rejects not-implemented claims, and writes the final
  audit report.

## Purpose Of This File

`AGENTS.md` is the repository-level instruction file for agents working here.
It defines the general rules. It must remain independent of the current
wireless/Sionna task. Task-specific rules live in task prompt files.

## Prompt Layers

Agents must combine the relevant prompt layers before acting:

1. `prompts/roles/implementer.md` or `prompts/roles/reviewer.md`: general,
   task-independent role behavior.
2. `prompts/tasks/channel_estimation/implementer.md` or `prompts/tasks/channel_estimation/reviewer.md`:
   task-specific role requirements.
3. `prompts/tasks/channel_estimation/task.md`: current benchmark contract and task interface.

## Contract Review Gate

Before modifying an existing paper implementation, its assets, tests, runner,
or reports, the agent must read:

- the relevant paper section and shared execution rules in `todo.md`;
- the repository `README.md` and every applicable contract under `docs/`;
- the paper-local `README.md` and relevant files under `reports/`;
- the shared interfaces used by the implementation and their tests; and
- the existing paper-specific implementation and tests.

Before editing files, the agent must report a short **Applicable contracts**
list naming the contracts and interface assumptions that govern the work. The
list must include only relevant items, but omission must be deliberate rather
than based on memory. If the work changes a contract or makes its current
classification inaccurate, updating that contract is part of the task.

## Required Per-Paper Layout

Each paper implementation lives under
`src/sionna/bench/mimo_ofdm_chest/papers/<paper_id>/`.

Required code and runner files:

- `paper.pdf`: the source paper as a regular file, if locally available
- `<task-specific-file-name>` such as `estimator.py`, `decoder.py`, or another
  name defined by the task contract
- optional helper modules needed by the implementation
- `run_benchmark.py`

Generated benchmark artifacts, when produced:

- repository-level `outputs/mimo_ofdm_chest/<paper_id>/<scenario_id>/` for canonical results
- repository-level `outputs/mimo_ofdm_chest/<paper_id>/smoke/<scenario_id>/` for smoke results
- optional paper-local temporary artifacts only when documented

Optional package-local inference assets live under the paper package's
`assets/<scenario_id>/<asset_type>/<artifact_id>/` directory. Public
distributions may omit these payloads. Training and regeneration always use an
explicitly supplied external asset root, and missing assets are never generated
or downloaded implicitly.

Required reports under `src/sionna/bench/mimo_ofdm_chest/papers/<paper_id>/reports/`:

- `paper_summary.md`
- `plan.md`
- `plan_review.md`
- `implementation_report.md`
- `final_report.md`


## Workflow

1. Implementer studies the harness and existing implementations in
   `src/sionna/bench/mimo_ofdm_chest/papers/`.
2. Implementer reads the paper and writes `reports/paper_summary.md`.
3. Implementer writes `reports/plan.md`.
4. Reviewer reviews the paper and plan, then writes
   `reports/plan_review.md`. The reviewer step is mandatory and must be
   invoked automatically as described below.
5. Implementer revises `reports/plan.md` if needed.
6. Implementer stops and asks the human for approval, feedback, or changes.
7. Implementation begins only after explicit human approval of the reviewed
   plan.
8. Implementation, asset-preparation, tuning, runner, and focused test code
   are written by the implementer according to the approved plan.
9. Implementer runs inexpensive analytical, compilation, integration, and
   smoke checks.
10. Before starting a full tuning study, implementer performs a preliminary
    tuning probe at a representative operating point. The probe
    exercises a small number of deliberately chosen parameter configurations,
    checks numerical sensitivity and plausible search ranges, and measures the
    dominant execution costs.
11. Implementer reports the preliminary results, proposed final search space,
    expected and conservative upper-bound runtime, storage requirements, and
    extrapolation assumptions. Full tuning must not begin until the human
    explicitly approves it.
12. After approval, implementer generates the fixed datasets or statistics and
    executes the approved algorithmic tuning, fitting, or training protocol.
    Candidate assets remain external, and canonical benchmark outputs remain
    outside parameter selection.
13. Implementer freezes the selected parameters and generates, validates, and
    records and validates any selected canonical inference assets.
14. Detector-facing variance calibration occurs only after estimator behavior,
    algorithmic parameters, and inference assets are frozen.
15. Implementer runs the canonical benchmark and updates results, inventories,
    the paper README, and `reports/implementation_report.md`.
16. Reviewer audits the final implementation and writes
    `reports/final_report.md`.

## Reviewer Invocation

When a user asks to implement a new paper through this workflow, that request
includes the reviewer handoffs required by the workflow. After the implementer
writes `reports/plan.md`, the implementer must automatically invoke a separate
reviewer agent when the environment provides sub-agent tooling and policy
permits its use.

The reviewer agent must use `prompts/roles/reviewer.md`,
`prompts/tasks/channel_estimation/reviewer.md`, and `prompts/tasks/channel_estimation/task.md`, read the paper and plan,
and write `reports/plan_review.md`. The implementer must not ask the human for
plan approval until `reports/plan_review.md` exists and any required plan
revisions have been made.

If a separate reviewer agent cannot be spawned because the environment lacks
sub-agent tooling or a higher-priority tool policy forbids spawning, the
implementer must stop at the reviewer handoff and explicitly report that
`reports/plan_review.md` is blocked pending reviewer execution. The implementer
must not treat its own plan as reviewed.

## Planning Requirements

`reports/plan.md` must inventory every relevant algorithmic variant from the
paper. Each method must be classified as:

- `implement`: faithful enough to implement under the current task contract,
  with permissible adaptations
- `not implemented`: not implemented in the current pass

For every `implement` method, the plan must include a detailed implementation
description close to pseudocode. It must cite equations, algorithms, sections,
tables, or explicit statements from the paper wherever possible. If the method
requires auxiliary assets or resources, the plan must specify their external
generation path, manifest contract, and regeneration procedure. It may also
identify an optional package-local location for internal distributions.

For every `not implemented` method, the plan must state why the method is not
implemented in the current pass and what would be required to implement it
faithfully. Reasons can include benchmark-contract conflicts, unavailable
assets, missing training procedures, missing statistics, missing
infrastructure, or a user-approved scope limit.

## Human Approval Gate

After `reports/plan_review.md` exists and all required plan revisions have
been made, the implementer must stop and ask the human to approve, reject, or
modify the reviewed plan. No implementation code may be written before explicit
human approval. If the environment is noninteractive, the implementer must stop
with status `blocked pending human plan approval`.

Human-requested plan changes must be reflected in `reports/plan.md` before
coding. The implementation report must state which plan version was approved
and summarize any human-requested deviations.


## Surrogate Rule

Do not replace a paper's objective, model, solver, training procedure, or
inference algorithm with a surrogate and still mark the method `implement`.
If the paper method cannot be implemented without such a substitution, classify
it as `not implemented`.

Necessary adaptations to fit the Sionna interface are allowed only when the
paper's actual algorithm remains intact and the adaptation is documented.

## Reporting Requirements

All nontrivial deviations from the paper or approved plan must be documented in
`reports/implementation_report.md`.

Smoke-test failures, baseline sanity-check failures, numerical instability, or
unexpectedly poor results must be reported as red flags. Do not hide them with
silent tuning or undocumented algorithm changes.
