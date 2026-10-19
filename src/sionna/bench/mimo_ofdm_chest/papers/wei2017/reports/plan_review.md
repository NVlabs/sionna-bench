# Plan Review: Wei et al., 2017

Reviewer: Codex reviewer agent

Reviewed inputs:

- `AGENTS.md`
- `prompts/reviewer.md`
- `prompts/task_reviewer.md`
- `prompts/task.md`
- `src/sionna_bench/mimo_ofdm_chest/papers/wei2017/paper.pdf`
- `src/sionna_bench/mimo_ofdm_chest/papers/wei2017/reports/paper_summary.md`
- `src/sionna_bench/mimo_ofdm_chest/papers/wei2017/reports/plan.md`

## Verdict

The revised plan is accepted for human approval. No further plan revision is
required before the human approval gate, but implementation must not start
until the human explicitly approves this reviewed plan.

The plan now proposes a comb-specialized uplink adaptation of Wei et al.'s
LC-EM-EP estimator. It uses Sionna's current Kronecker pilot combs to divide
out the pilot symbols, transform uniformly spaced pilot-subcarrier
observations into a delay-domain comb basis, and then apply the paper's
conditional Gaussian posterior and common-support EM update across BS receive
antennas.

This is mathematically valid under the stated assumptions: uniform pilot comb
with `Delta * K = fft_size`, correct effective FFT-bin indexing, constant
pilot magnitude within each group, nonzero pilots, and `num_taps <= K`.
Under those conditions, the LS-normalized comb transform converts the linear
Gaussian observation model into independent noisy delay-bin observations, so
the EP linear-message machinery collapses to scalar Gaussian posterior
updates. The shared `alpha_l` update from Equation (21) remains the core
paper idea.

This is not a literal implementation of the paper's downlink superposed-pilot
receiver and not a generic `D = diag(pilots) F` LC-EM-EP solver. The plan is
acceptable because those deviations are explicit, necessary for the current
benchmark, and bounded by fail-fast assumptions.

## Conditions For Implementation

- Use `resource_grid.effective_subcarrier_ind` or the equivalent Sionna
  mapping when converting pilot-group subcarrier coordinates to FFT-bin
  indices.
- Implement the fail-fast checks from the plan: uniform comb, `Delta * K =
  fft_size`, constant nonzero pilot magnitude, and `num_taps <= K`.
- Do not silently fall back to LS, nearest-neighbor interpolation, or a generic
  matrix solver when the comb assumptions fail.
- Keep the estimator in Sionna's pilot-domain split and pass an explicit
  interpolation type, as planned.
- Document in the implementation report and README that the implemented method
  is a comb-specialized uplink adaptation, not a reproduction of the paper's
  downlink receiver or simulation figures.
- Add the planned transform identity and limiting-case tests unless the
  implementation report gives a concrete reason why the limiting-case test is
  numerically unsuitable.
- Treat `posterior_variance_scale` as the only detector-facing confidence
  tuning knob; any non-default canonical value must be bounded and reported.

## Method Review

### Comb-Specialized LC-EM-EP Uplink Common-Support Estimator

Classification: `implement`

Reviewer verdict: accepted with caveats.

The plan covers the relevant paper equations and Algorithm 1. The paper's
general downlink model is

```text
y = D h + w,  D = [P_1 F_Omega, ..., P_M F_Omega].
```

The revised plan does not implement this generic model directly. Instead, for
Sionna's separated uplink Kronecker pilots, it uses

```text
y_{a,q} = p_q H_a[k_q] + w_{a,q}
z_{a,q} = y_{a,q} / p_q
```

and then applies an inverse transform over the uniform comb index. This is an
invertible linear-Gaussian change of coordinates under the plan's comb and
pilot-magnitude assumptions. It therefore does not replace the paper's
probabilistic estimator with an unrelated surrogate; it specializes the paper's
linear Gaussian model to the current benchmark geometry.

The main paper mechanism that must survive implementation is the shared
delay-support EM update:

```text
alpha_l <- mean_over_antennas(|h_hat_{l,a}|^2 + v_hat_{l,a})
```

which corresponds to Equation (21). The plan preserves this by pooling over BS
receive antennas for each UE stream and pilot OFDM symbol.

The main caveat is terminology: the resulting scalar update is closer to the
diagonalized special case of the EP/EM model than to the full LC-EM-EP message
schedule in Equations (23)-(27). The implementation and README should not
overclaim generic LC-EM-EP support.

### Generic Non-Comb Observation-Matrix LC-EM-EP

Classification: `not implemented`

Reviewer verdict: accepted.

The paper's full low-complexity update can be written for a general observation
matrix `D`. The current benchmark does not require this more general path, and
adding it as a speculative fallback would increase complexity without improving
the approved first-pass target. The plan's fail-fast behavior is preferable.

### Full-Grid Delay-Domain Reconstruction

Classification: `not implemented`

Reviewer verdict: accepted.

The paper's delay-domain estimate can be evaluated on all OFDM subcarriers,
but the current task contract requires first-pass estimators to preserve the
pilot-domain estimate plus explicit interpolation split. Delegating full-grid
reconstruction to Sionna interpolation is a reviewed benchmark adaptation.

### Literal Downlink Superposed-Pilot LC-EM-EP

Classification: `not implemented`

Reviewer verdict: accepted.

The literal paper receiver estimates downlink channels at the user from a
single observation vector containing superposed BS transmit-antenna pilots.
The repository benchmark is an uplink BS-side Sionna estimator with separated
Kronecker pilot observations. A literal implementation would require a
benchmark-contract change.

### Block-ISD, BP, And Exact-LS Oracle Comparisons

Classification: `not implemented`

Reviewer verdict: accepted.

These are Section IV comparison baselines, not the proposed Wei et al.
estimator. The exact-LS oracle also assumes known support. They are not
required for this paper folder.

### Paper Simulation Reproduction

Classification: `not implemented`

Reviewer verdict: accepted.

The repository target is the shared Sionna benchmark, not reproduction of the
paper's downlink ITU Vehicular-A simulations with `M=32`, `N=4096`, `L=128`,
and `K=580`. The implementation report and README must avoid claiming
reproduction of Figures 2 or 3.

## Divergence Ledger

### Downlink receiver to uplink estimator

- Divergence: The paper's downlink superposed-pilot receiver is replaced by an
  uplink estimator operating on Sionna's separated BS receive-antenna pilot
  observations.
- Source: `interface`
- Reviewer verdict: `necessary`
- Rationale: The task contract fixes the uplink Sionna estimator interface.
- Impact on paper faithfulness: Moderate. The physical channel direction and
  likelihood structure change, but the common-support sparse Bayesian update
  remains recognizable.
- Impact on benchmarking fairness: Low, because all methods are evaluated on
  the same uplink benchmark.

### Common-support antenna dimension remapping

- Divergence: The paper shares `alpha_l` across BS transmit antennas; the plan
  shares it across BS receive antennas for each UE stream and pilot OFDM
  symbol.
- Source: `interface`
- Reviewer verdict: `justified`
- Rationale: This is the natural co-located antenna dimension available in the
  uplink benchmark.
- Impact on paper faithfulness: Moderate. It changes the antenna role but
  preserves Equation (21)'s shared-support mechanism.
- Impact on benchmarking fairness: Low, provided no information is shared
  across UE streams.

### LS-normalized comb-delay specialization

- Divergence: The implementation divides by pilot symbols and applies a comb
  inverse transform instead of running the general `D = diag(pilots) F`
  LC-EM-EP updates directly.
- Source: `implementation`
- Reviewer verdict: `justified`
- Rationale: With uniform comb pilots and constant pilot magnitude, this is an
  invertible linear-Gaussian coordinate change that diagonalizes the current
  benchmark observation model.
- Impact on paper faithfulness: Moderate. The full message update in
  Equations (23)-(27) collapses to scalar posterior updates, so this must be
  described as a comb-specialized special case rather than a generic solver.
- Impact on benchmarking fairness: Low if the estimator fails outside its
  approved comb assumptions.

### Failure outside comb assumptions

- Divergence: The estimator will reject non-comb or unequal-magnitude pilot
  patterns rather than implementing the paper's generic observation matrix.
- Source: `implementation`
- Reviewer verdict: `justified`
- Rationale: The first-pass benchmark uses comb Kronecker pilots, and the plan
  explicitly classifies the generic solver as not implemented.
- Impact on paper faithfulness: Low for the current benchmark; high for other
  pilot layouts.
- Impact on benchmarking fairness: Low for current results, provided failure
  is explicit.

### DFT scaling and effective FFT-bin convention

- Divergence: The paper does not specify DFT normalization or Sionna
  effective-subcarrier indexing; the plan fixes a unit-magnitude Fourier-atom
  convention and requires effective FFT-bin mapping.
- Source: `paper`
- Reviewer verdict: `necessary`
- Rationale: An implementation must choose a convention, and the selected one
  is compatible with Sionna's frequency-response convention if indexed
  correctly.
- Impact on paper faithfulness: Low if implemented exactly; high if the
  subcarrier-to-FFT-bin mapping is wrong.
- Impact on benchmarking fairness: Potentially high if wrong, so the planned
  transform identity test is important.

### Built-in Sionna interpolation

- Divergence: Full-grid channel estimates are produced by Sionna
  `"lin_time_avg"` interpolation instead of direct delay-domain evaluation over
  all subcarriers.
- Source: `interface`
- Reviewer verdict: `justified`
- Rationale: The task contract requires a pilot-domain estimator plus explicit
  reconstruction/interpolation for first-pass implementations.
- Impact on paper faithfulness: Moderate. The paper naturally permits
  full-grid delay-domain reconstruction, which is deliberately omitted.
- Impact on benchmarking fairness: Low to moderate, because interpolation
  affects NMSE/BER/BLER but is explicit shared infrastructure.

### Detector-facing error variances

- Divergence: Delay-domain posterior variances are propagated to pilot
  frequency locations with a diagonal approximation and then interpolated for
  Sionna's detector.
- Source: `interface`
- Reviewer verdict: `justified`
- Rationale: Equations (14a)/(16a) provide posterior variances, but the paper
  does not define Sionna full-grid detector-facing variances.
- Impact on paper faithfulness: Low to moderate if documented.
- Impact on benchmarking fairness: Moderate because BER/BLER depends on
  confidence scaling. Tuning must be bounded and reported.

### Benchmark default for delay length

- Divergence: Default `num_taps` is `min(cfg.cyclic_prefix_length,
  pilot_group_len)` instead of the paper simulation's `L=128`.
- Source: `interface`
- Reviewer verdict: `justified`
- Rationale: The paper's simulation numerology is not the repository benchmark;
  keeping `num_taps` configurable preserves flexibility.
- Impact on paper faithfulness: Low with documentation.
- Impact on benchmarking fairness: Low if canonical values are reported.

### No sharing across pilot OFDM symbols

- Divergence: The plan processes each fixed pilot OFDM symbol as its own group
  rather than sharing `alpha_l` across pilot symbols.
- Source: `interface`
- Reviewer verdict: `justified`
- Rationale: The paper models one OFDM pilot observation without a time
  dimension, while Sionna's first-pass contract returns pilot-location
  estimates that are later interpolated over time.
- Impact on paper faithfulness: Low to moderate.
- Impact on benchmarking fairness: Low, provided the behavior is documented.

## Final Reviewer Decision

Accepted for human approval. Implementation may proceed only after explicit
human sign-off on this reviewed plan. No estimator code should be written
before that approval.
