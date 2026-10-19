# Plan Re-Review: Li et al., 2020

- Paper ID: `li2020`
- Reviewer: Codex reviewer agent
- Date: 2026-06-23
- Reviewed artifacts: `AGENTS.md`, `prompts/reviewer.md`,
  `prompts/task_reviewer.md`, `prompts/task.md`, `src/sionna_bench/mimo_ofdm_chest/papers/li2020/paper.pdf`,
  revised `src/sionna_bench/mimo_ofdm_chest/papers/li2020/reports/paper_summary.md`, and revised
  `src/sionna_bench/mimo_ofdm_chest/papers/li2020/reports/plan.md`.

## Gate Decision

Implementation planning is approved for human approval.

The revised plan correctly applies the updated estimator contract. Li et al.
explicitly formulate channel estimation as super-resolution from sparse pilot
observations to the full OFDM channel image in Equation (7), and ReEsNet's
upsampling layer is the paper's native full-grid reconstruction step. The
planned `ReEsNetChannelEstimator(FullGridChannelEstimator, PilotGroupMixin)`
with direct `call(y, no)` is therefore a better fit than hiding the network
behind Sionna's `estimate_at_pilot_locations()` plus interpolator pattern.

This approval is only for the human approval gate. No implementation should
start until the human explicitly approves or modifies the reviewed plan.

## Harness And Contract Review

The plan adequately summarizes:

- the benchmark `ResourceGrid`, Kronecker pilot pattern, and output tensor
  contract;
- the distinction between the Sionna pilot mask and the active nonzero pilot
  symbols;
- the built-in baselines: `ls-nn`, `ls-lin`, `lmmse`, and `perfect`;
- the full-grid estimator option introduced by the revised task contract;
- the need for smoke checks, canonical benchmark output, and result storage
  under the paper-local `outputs/` tree.

The selected architecture is contract-compatible because the paper does not
merely estimate channels at pilot positions. It computes LS pilot estimates by
Equation (3), then applies the learned mapping of Equation (7) from the pilot
image to the full channel image. That reconstruction is part of the paper
method and should not be replaced by Sionna's default interpolation.

## Method Review

### Method: `ReEsNetChannelEstimator`

- Implementer status: `implement`
- Reviewer decision: accepted for human approval, with implementation
  conditions below.
- Paper coverage: adequate. The plan covers LS pilot estimation from
  Section II-A/Equation (3), the super-resolution mapping from
  Section III-A/Equation (7), the ReEsNet architecture from
  Section III-C/Figure 2, and the L1 training objective from Equation (8).
- Contract compatibility: accepted. `FullGridChannelEstimator` is justified
  because ReEsNet natively reconstructs the full channel grid from selected
  pilot observations.
- Surrogate-rule compliance: compliant, provided the implementation preserves
  the ReEsNet residual CNN, transposed-convolution upsampling, L1 objective,
  and trained checkpoint rather than substituting a hand-coded interpolator or
  denoiser.
- Aux plan: sufficient. The proposed cache metadata ties weights to the
  scenario, pilot geometry, active nonzero pilot image, architecture, training
  settings, validation settings, and selected `err_var_scale`.

Required implementation conditions:

- Use `PilotGroupMixin(active_only=True)` and build the network input only
  from nonzero pilot symbols.
- Preserve and test the active benchmark mapping `[156, 2] -> [624, 14]`.
- Implement `call(y, no)` directly and return the Sionna full-grid
  `h_hat, err_var` tensors.
- Keep long training behind an explicit runner mode. Normal benchmark mode
  must load a matching asset or fail clearly.
- Report whether canonical weights use paper-style random-speed training
  (`0` to `50` km/h and SNRs `0:5:20` dB) or benchmark-specialized fixed-speed
  training. Fixed-speed training is acceptable only as a documented benchmark
  adaptation.
- Use L1 loss for the ReEsNet checkpoint.
- Run and report a bounded `err_var_scale` validation sweep before canonical
  BER/BLER outputs.
- Compare smoke and canonical outputs against at least `ls-nn`, `ls-lin`, and
  `lmmse`; report unexpectedly poor NMSE or detector behavior as red flags.

### Method: `ChannelNet`

- Implementer status: `not implemented`
- Reviewer decision: accepted.
- Rationale: ChannelNet is prior work from Soltani et al., not Li et al.'s
  proposed method. Li et al. summarize it but do not provide enough SRCNN,
  DnCNN, and separate-training detail for a faithful local implementation.
  A ChannelNet reproduction belongs in a separate paper task.

### Method: Conventional LS With Linear Interpolation

- Implementer status: `not implemented`
- Reviewer decision: accepted.
- Rationale: Section II-A/Equation (3) is a conventional baseline and is
  already represented by the repository `ls-lin` baseline. A paper-local
  duplicate is unnecessary.

### Method: LMMSE

- Implementer status: `not implemented`
- Reviewer decision: accepted.
- Rationale: Section II-B/Equations (4)-(5) describe a conventional baseline
  already represented by the repository `lmmse` baseline. A paper-local
  duplicate is not needed for the Li et al. contribution.

## Divergence Ledger

### Divergence: FullGridChannelEstimator Instead Of Sionna Pilot-Domain Split

- Source: `interface`
- Reviewer verdict: `necessary`
- Rationale: ReEsNet's learned super-resolution from `H_p` to `H` is the
  paper's estimator, not a post-processing interpolation choice. Direct
  `call(y, no)` preserves the paper's full-grid mapping more faithfully than
  `estimate_at_pilot_locations()` followed by a built-in interpolation rule.
- Impact on paper faithfulness: positive; it avoids replacing the learned
  reconstruction with Sionna interpolation.
- Impact on benchmarking fairness: acceptable if the estimator returns the
  standard Sionna full-grid tensors and uses the same benchmark pilots.

### Divergence: Sionna Mask Versus Active Pilot Observations

- Source: `interface`
- Reviewer verdict: `necessary`
- Rationale: Sionna's Kronecker mask marks zero-valued compact entries for
  pilot orthogonalization across streams. Li et al.'s `Y_p, X_p` model
  corresponds to nonzero pilot observations, so feeding only active entries to
  ReEsNet is the faithful interpretation.
- Impact on paper faithfulness: low after documentation; it avoids artificial
  zero placeholders in the pilot image.
- Impact on benchmarking fairness: acceptable if layout tests and checkpoint
  metadata prevent mismatches.

### Divergence: Benchmark Pilot/Image Dimensions

- Source: `interface`
- Reviewer verdict: `necessary`
- Rationale: The paper evaluates 24 or 48 pilots on a `72 x 14` grid. The
  benchmark gives `156 x 2` active nonzero pilot observations per tx/stream and
  requires reconstruction to `624 x 14`.
- Impact on paper faithfulness: moderate; the dimensions differ, but the same
  LS-input, residual CNN, transposed-convolution, and L1-training structure is
  preserved.
- Impact on benchmarking fairness: acceptable because all methods use the same
  benchmark channel and pilot pattern.

### Divergence: Compact Time-Axis Input

- Source: `interface`
- Reviewer verdict: `justified`
- Rationale: The benchmark has two pilot OFDM symbols at absolute indices
  `(2, 11)`, while the ReEsNet input image has a compact time dimension of
  length two. The plan ties this convention to the fixed pilot pattern through
  training and cache metadata.
- Impact on paper faithfulness: moderate; the coordinate convention is
  benchmark-specific.
- Impact on benchmarking fairness: acceptable if documented and tested.

### Divergence: Benchmark-Trained Weights

- Source: `asset`
- Reviewer verdict: `necessary`
- Rationale: The paper does not provide pretrained weights or its Samsung
  simulator dataset. A local aux checkpoint generated from the benchmark
  channel model is required for a runnable benchmark component.
- Impact on paper faithfulness: moderate; results should not be claimed as a
  reproduction of the paper's published curves.
- Impact on benchmarking fairness: acceptable if training data, validation
  data, budget, and seed are reported.

### Divergence: Fixed-Speed Benchmark Training Option

- Source: `asset`
- Reviewer verdict: `justified with disclosure`
- Rationale: The paper samples user speed from 0 to 50 km/h during training.
  The plan also permits benchmark-specialized fixed-speed training for the
  active config. This is valid for benchmark specialization but not for
  claiming reproduction of the paper's scenario-general training mechanism.
- Impact on paper faithfulness: moderate if fixed-speed mode is used.
- Impact on benchmarking fairness: acceptable if the selected mode is recorded
  and baselines are evaluated on the same benchmark config.

### Divergence: Detector Error Variance Fallback

- Source: `interface`
- Reviewer verdict: `necessary`
- Rationale: Li et al. optimize channel reconstruction and do not define
  Sionna detector-facing marginal channel-estimation error variances. A scaled
  LS-variance fallback is acceptable only with a bounded validation sweep for
  `err_var_scale`.
- Impact on paper faithfulness: low for channel estimates; moderate for
  BER/BLER.
- Impact on benchmarking fairness: acceptable with the planned sweep and clear
  reporting.

## Final Plan Review Verdict

No further plan changes are required before asking the human for approval.
Proceed to the human approval gate. If the human approves the revised plan,
implementation may begin for `ReEsNetChannelEstimator` under the conditions
above. `ChannelNet`, paper-local LS, and paper-local LMMSE should remain out
of scope for this implementation pass.
