# Task Reviewer Prompt: Channel Estimation

Use this together with `prompts/roles/reviewer.md` and
`prompts/tasks/channel_estimation/task.md`.

## Plan Review Checklist

Verify that the implementer first summarized the harness and correctly stated:

- resource-grid and pilot assumptions
- estimator input and output shapes
- built-in baselines
- metric definitions
- smoke-test expectations

For every paper method, check whether the claimed mapping to the benchmark is
valid. Require explicit paper citations for the implementation pseudocode.

## Tuning And Asset Review

Reject the plan if any behavior-relevant parameter lacks an explicit
classification or, where applicable, a canonical tuning procedure.

Reject the plan if a required asset lacks:

- an independent-channel count;
- a validation-channel count where applicable;
- a derived-example count;
- a generation entry point;
- seed information;
- compatibility checks; or
- a proposed storage location.

Verify that all tuning and dataset budgets agree with
`docs/mimo_ofdm_chest/contracts/tuning_protocol.md`. Verify that canonical benchmark outputs
remain outside parameter selection.

Verify that the preliminary tuning probe is used only for feasibility, search-
space refinement, and runtime estimation. Reject plans that treat its 10 dB
result as canonical parameter selection or reuse canonical benchmark outputs.
Confirm that the human approved the reported full-study cost before expensive
tuning began.

Require explicit justification and human approval for any deviation from the
canonical tuning or dataset protocol.

## Channel-Estimation Interface Rules

Verify that the implementer chose one of the approved estimator architectures
from `prompts/tasks/channel_estimation/task.md` and justified the choice against the paper.

The default architecture is Sionna's estimate-at-pilots/interpolate split:
`BasePilotChannelEstimator` with `estimate_at_pilot_locations()`,
`LSChannelEstimator` with an explicit interpolator, or a custom
`BaseChannelInterpolator` used by a Sionna estimator. This is the right choice
when the paper only defines pilot-location estimation, does not discuss
interpolation, assumes pilots on every subcarrier, or already separates pilot
estimation from interpolation.

Accept Sionna's `BaseChannelEstimator` when the
paper explicitly assumes pilots only on selected resource elements and the
proposed algorithm itself reconstructs, interpolates, extrapolates, or
super-resolves the channel across subcarriers and/or OFDM symbols from those
pilot observations. In this case, require direct `call(y, no)` implementation,
Sionna-compatible pilot extraction, and `_PilotGroupMixin` whenever structured
pilot groups are needed.

Do not approve `BaseChannelEstimator` merely because the paper assumes
pilots on all subcarriers and never discusses interpolation. That case should
stay with `estimate_at_pilot_locations()` plus explicit interpolation, with
the observation-model mismatch documented.

Accept interface adaptations only when they preserve the paper estimator. For
example, batching, tensor reshaping, or using the benchmark's pilot extraction
can be acceptable. Replacing the estimator's optimization problem, kernel,
learned model, covariance model, or feedback loop is not an interface
adaptation.

Require the implementer to state the interpolation choice explicitly. Treat
accidental use of Sionna's default nearest-neighbor interpolation as a finding.
Accept built-in `"lin"` or `"lin_time_avg"` interpolation only when it is
planned, justified, and clearly documented as a benchmark adaptation or as the
paper's own interpolation rule.

When the paper only specifies pilot-domain estimation, prefer the conservative
route: faithful pilot-domain estimation plus explicit built-in Sionna
interpolation.

For kernel, sparse-reconstruction, compressed-sensing, or learned
reconstruction papers, do not automatically require the reconstruction to be
marked `not implemented`. If the paper explicitly models sparse pilot
resource-element observations and the reconstruction across the grid is part
of the algorithm, a reviewed `BaseChannelEstimator` can be the faithful
architecture. If the proposed reconstruction is too underspecified,
computationally impractical, or incompatible with the benchmark, require it to
be marked `not implemented` rather than replaced by a surrogate.

## Sionna-Style Review Checklist

Check that the implementation:

- reuses `_PilotGroupMixin` when pilot grouping is needed;
- avoids redundant validation of invariants guaranteed by Sionna or the mixin;
- exposes algorithm choices as constructor arguments;
- stores algorithm state in private attributes and fixed tensors as buffers;
- uses descriptive variable names rather than paper shorthand;
- keeps estimator logic inside estimator/interpolator methods;
- respects `precision` and device placement;
- avoids unnecessary special-case optimization paths;
- implements variants through one configurable class when that is cleaner;
- uses shared aggregated statistics across pilot groups unless the paper
  explicitly requires group-specific statistics.

## Testing Review Checklist

Check whether nontrivial estimator or interpolator logic has focused tests for
analytical properties, limiting cases, or baseline equivalence when such tests
are practical. If tests are absent, verify that the implementation report
explains why smoke checks and benchmark runs are sufficient for the current
pass.

## Sanity-Check Review

Treat these as serious findings:

- smoke failures hidden or minimized
- no comparison against required built-in baselines
- estimator worse than `ls-nn` on NMSE without a paper-based explanation
- undocumented clipping, regularization, fallback interpolation, or parameter
  tuning
- `err_var` fabricated in a way that affects downstream detection without
  disclosure
- detector-facing calibration that does not follow the applicable section of
  `docs/mimo_ofdm_chest/contracts/tuning_protocol.md`

The final report should distinguish whether a failure means the code is wrong,
the plan overclaimed faithfulness, or the paper is not suitable for the current
benchmark contract.
