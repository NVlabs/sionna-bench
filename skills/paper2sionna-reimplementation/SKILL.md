---
name: paper2sionna-reimplementation
description: Reimplement, debug, tune, or refactor algorithms from academic papers for Sionna-style benchmarks and paper2sionna workflows. Use when working on paper-to-code implementation plans, estimator/decoder/detector implementations, Sionna benchmark harnesses, underperforming paper methods, hyperparameter tuning, smoke tests, benchmark curve sanity checks, or simplifying overcomplicated research-code architecture.
---

# Paper2sionna Reimplementation

## Scope

Use this skill as an operational reminder for paper-to-Sionna work. The
authoritative workflow and task rules live in:

- `AGENTS.md`
- `prompts/implementer.md` or `prompts/reviewer.md`
- `prompts/task.md`
- `prompts/task_implementer.md` or `prompts/task_reviewer.md`

Follow those files first. This skill should not override their workflow,
human-approval gate, report requirements, or task contract.

## Checklist

1. Study the harness and existing paper implementations before adapting the
   paper.
2. Map the paper to the benchmark explicitly with equation-linked pseudocode.
   If faithful implementation would require a surrogate algorithm, mark the
   method `not implemented` instead of substituting a different method.
3. Generate required assets as part of the implementation when they can
   be produced faithfully. If they cannot, document the acquisition plan and
   mark the affected method `not implemented`.
4. Reuse Sionna abstractions and repository helpers, especially pilot
   extraction, interpolation, result storage, and `_PilotGroupMixin` where
   applicable.
5. Keep the architecture simple. Avoid speculative factories, registries,
   compatibility shims, defensive checks for impossible states, and
   single-use abstractions.
6. Preserve Sionna conventions: constructor-exposed algorithm choices, private
   instance attributes, `precision` and device handling, fixed tensors as
   buffers, and descriptive variable names.
7. For nontrivial math, look for analytical tests: limiting cases,
   pass-through properties, baseline equivalence, transform identities, and
   shape/dtype/device contracts.
8. Use smoke tests for integration and debugging. Use full benchmark runs
   before making performance claims or saving canonical result files.
9. When behavior is surprising, re-read the paper and check dimensions,
   scaling, noise definitions, normalizations, and boundary conventions before
   tuning.
10. Record every meaningful deviation from the paper, including benchmark
    adaptations, formula corrections, missing detector-facing variances, and
    tuned hyperparameters.
