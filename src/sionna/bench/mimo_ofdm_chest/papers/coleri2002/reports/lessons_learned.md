# Lessons Learned

- Do not defer merely because a paper omits an implementation detail. First
  decide whether the core algorithm is specified, then choose a conservative
  rule for the missing detail, document it, and test the impact. Missing edge
  behavior or default settings are often implementation choices, not reasons to
  abandon a method.

- Search for named external procedures before declaring details unavailable.
  Public documentation, reference manuals, and related implementation notes can
  often provide enough information for a faithful first-pass reimplementation,
  even when exact historical behavior remains unverified.

- Treat published formulas as fallible. Check equations against surrounding
  definitions, dimensions, limiting cases, and the stated algorithm before
  transcribing them literally. If a correction is necessary, document why it is
  a correction rather than a benchmark-tuned change.

- Smoke tests are integration checks, not performance evidence. Small runs are
  useful for catching shape errors, invalid values, broken invariants, and
  obvious numerical problems. Use full-size runs before making claims about
  method quality or relative performance.

- Run the real benchmark after meaningful fixes. A change can look ambiguous in
  a small or noisy run but become clearly right or wrong at the target problem
  size. Do not stop after a smoke result when the question is performance.

- Do not tune interface-facing quantities silently. When the benchmark requires
  outputs that the paper does not specify, preserve the benchmark's established
  observation model where possible. If an adaptation is required, make it
  explicit and avoid result-driven scaling.

- Avoid unnecessary device or representation changes. Keep metadata and
  intermediate tensors in the native framework unless there is a clear reason to
  leave it. Extra conversions make accelerator behavior harder to reason about
  and can hide portability issues.

- Prefer explicit adaptations over hidden assumptions. Benchmark-specific time
  handling, boundary rules, default parameters, and formula corrections are
  acceptable only when they are named in the reports and audited against the
  paper.

- Keep exactness claims modest. Following public documentation or a plausible
  reconstruction is not the same as bit-exact agreement with an author's
  original code. State what was verified and what remains unverified.

- Add comments where code encodes research judgment. Mechanical tensor
  operations usually do not need narration, but boundary choices, default
  assumptions, invariant preservation, variance propagation, and benchmark
  adaptations should be explained next to the code.
