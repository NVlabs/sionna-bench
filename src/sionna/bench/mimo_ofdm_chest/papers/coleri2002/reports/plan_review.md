# Plan Review

- Paper ID: `coleri2002`
- Reviewer: Codex
- Date: 2026-06-03
- Revision note: supersedes the earlier conservative review that deferred too many interpolation variants.

## Overall Gate Decision

- Can implementation proceed? Yes.
- Approved methods: `comb-ls-second-order`, `comb-ls-low-pass`, `comb-ls-spline`, `comb-ls-time-domain`; `comb-ls-linear` is covered by the existing `ls-lin` baseline.
- Deferred methods: `comb-lms-linear`, `comb-mmse`.
- Incompatible methods: `block-ls`, `block-mmse`, `block-decision-feedback`.
- Required plan changes before coding: none after this revision.

## Harness Understanding

- Did the implementer summarize the harness correctly? Yes.
- Missing or incorrect benchmark-interface details: none.
- Missing built-in baseline or smoke-test details: none.

## Method Reviews

### Method: `comb-ls-second-order`

- Implementer status: `implement`
- Reviewer decision: `accepted`
- Paper coverage complete? Yes: equations (17) and (19).
- Equation-linked pseudocode adequate? Yes.
- Target repository abstraction correct? Yes: `BaseChannelEstimator` plus `BaseChannelInterpolator`.
- Surrogate-rule compliance: compliant.
- Interface adaptations necessary and justified? Offset-aware constant edge extension for second-order interpolation and linear OFDM-time interpolation are documented and acceptable.
- `aux/` plan needed? No.
- Required changes: none.

### Method: `comb-ls-low-pass`

- Implementer status: `implement`
- Reviewer decision: `accepted`
- Paper coverage complete? Yes, with official MATLAB documentation filling in the public `interp` algorithm.
- Equation-linked pseudocode adequate? Yes.
- Target repository abstraction correct? Yes.
- Surrogate-rule compliance: compliant. The implementation must remain zero-insertion plus low-pass FIR interpolation.
- Interface adaptations necessary and justified? Finite filtering with virtual constant edge pilots and linear OFDM-time interpolation are acceptable if reported.
- `aux/` plan needed? No for first pass.
- Required changes: report that the FIR follows public MathWorks documentation and is not claimed to be bit-exact closed MATLAB internals.

### Method: `comb-ls-spline`

- Implementer status: `implement`
- Reviewer decision: `accepted`
- Paper coverage complete? Yes, with official MATLAB `spline` documentation.
- Equation-linked pseudocode adequate? Yes.
- Target repository abstraction correct? Yes.
- Surrogate-rule compliance: compliant.
- Interface adaptations necessary and justified? MATLAB-like not-a-knot spline interpolation with clipped constant edge queries is justified and preferable to deferral or unstated periodic wrap-around.
- `aux/` plan needed? No.
- Required changes: report the clipped constant-edge boundary adaptation.

### Method: `comb-ls-time-domain`

- Implementer status: `implement`
- Reviewer decision: `accepted`
- Paper coverage complete? Yes: equations (17), (20)-(22).
- Equation-linked pseudocode adequate? Yes.
- Target repository abstraction correct? Yes.
- Surrogate-rule compliance: compliant.
- Interface adaptations necessary and justified? FFT scaling and linear OFDM-time interpolation are documented.
- `aux/` plan needed? No.
- Required changes: report pilot pass-through before OFDM-time interpolation.

### Method: `comb-ls-linear`

- Implementer status: `implement` via baseline
- Reviewer decision: `accepted`
- Paper coverage complete? Yes.
- Equation-linked pseudocode adequate? Yes.
- Target repository abstraction correct? Yes.
- Surrogate-rule compliance: compliant.
- Interface adaptations necessary and justified? Yes.
- `aux/` plan needed? No.
- Required changes: do not claim it as new paper-local code.

### Method: `comb-lms-linear`

- Implementer status: `defer`
- Reviewer decision: `accepted`
- Rationale: missing step size plus incompatible independent-frame benchmark state.

### Method: `comb-mmse`

- Implementer status: `defer`
- Reviewer decision: `accepted`
- Rationale: Coleri et al. cite MMSE variants but do not define a self-contained comb-MMSE method.

### Method: `block-ls`, `block-mmse`, `block-decision-feedback`

- Implementer status: `incompatible`
- Reviewer decision: `accepted`
- Rationale: all-subcarrier block pilots and feedback loops violate the fixed Kronecker pilot and estimator-only contract.

## Divergence Ledger

### Divergence: Frequency Boundary Handling

- Source: `interface`
- Reviewer verdict: `justified`
- Rationale: second-order interpolation now uses offset-aware constant edge extension at the first and last pilot intervals, avoiding unstated wrap-around coupling in equation (19). Low-pass uses finite FIR filtering with virtual constant edge pilots, and spline uses offset-aware not-a-knot interpolation with clipped constant edge queries. This avoids coupling the first and last comb pilots through a circular/periodic boundary that the paper does not state.
- Impact on faithfulness: low to moderate.
- Impact on benchmarking fairness: acceptable and explicit.

### Divergence: Linear OFDM-Time Interpolation

- Source: `interface`
- Reviewer verdict: `justified`
- Rationale: the paper assumes comb pilots in each symbol, while the benchmark has two pilot-bearing symbols. Linear interpolation between pilot-bearing OFDM symbols is a documented benchmark adaptation and avoids introducing an extra method-selection knob.
- Impact on faithfulness: moderate.
- Impact on benchmarking fairness: acceptable if reported; it uses the same pilot observations as the baselines.

### Divergence: MATLAB `interp` Approximation

- Source: `implementation`
- Reviewer verdict: `justified`
- Rationale: official MathWorks docs specify zero insertion, default FIR length/settings, and `intfilt`/least-squares design family. A PyTorch/SciPy implementation following those docs is faithful enough for a first pass.
- Impact on faithfulness: low to moderate; not bit-exact to closed MATLAB internals.
- Impact on benchmarking fairness: acceptable.

### Divergence: Detector Error Variance

- Source: `interface`
- Reviewer verdict: `necessary`
- Rationale: the paper does not derive a detector-facing error-variance model,
  but the Sionna detector requires one. The reviewed implementation should use
  Sionna's LS pilot variance `no/|pilot|^2`, matching built-in LS baselines,
  and then propagate it through the paper-local interpolator.
- Impact on faithfulness: low for channel estimates, moderate for BER/BLER.
- Impact on benchmarking fairness: acceptable because it avoids a tuned proxy
  and keeps the LS observation model aligned with built-in baselines.

### Divergence: Printed Equation Typos

- Source: `paper`
- Reviewer verdict: `justified`
- Rationale: equation (19)'s `alpha = l/N` conflicts with the interval definition `k=mL+l` and with equation (18), so the reviewed plan should use `alpha = l/L`. Equation (21)'s printed zero-padding cases have inconsistent index ranges, so the reviewed plan should implement the stated IDFT, middle zero padding, and DFT operation instead of literal transcription.
- Impact on faithfulness: improves faithfulness to the paper's intended interpolation algorithms.
- Impact on benchmarking fairness: acceptable because these are deterministic formula corrections, not result tuning.

## Final Instructions To Implementer

- Methods cleared for implementation: `coleri-second-order`, `coleri-low-pass`, `coleri-spline`, `coleri-time-domain`.
- Constraints: keep code paper-local; use a Sionna channel estimator/interpolator pair; report official MATLAB sources; do not implement LMS or block methods inside the current benchmark contract.
- Smoke/sanity checks that must be reported: commands, separately generated baseline comparisons, finite outputs, nonnegative error variance, frequency pilot pass-through before OFDM-time interpolation, and full default benchmark outputs.
