# Plan Review

- Paper ID: `baur2022`
- Reviewer: Codex reviewer agent
- Date: 2026-06-24
- Reviewed artifacts: `prompts/reviewer.md`, `prompts/task_reviewer.md`,
  `prompts/task.md`, `src/sionna_bench/mimo_ofdm_chest/papers/baur2022/paper.pdf`,
  `src/sionna_bench/mimo_ofdm_chest/papers/baur2022/reports/paper_summary.md`, and
  `src/sionna_bench/mimo_ofdm_chest/papers/baur2022/reports/plan.md`.

## Gate Decision

Implementation may not proceed from the current plan. The proposed estimator
mapping is directionally correct, and `VAE-noisy` and `VAE-real` are
implementable under the benchmark contract, but the plan needs revisions before
human approval:

- specify how per-SNR VAE checkpoints are selected during a benchmark SNR
  sweep;
- make the `aux/` checkpoint acquisition and validation plan concrete;
- define bounded validation protocols for unspecified VAE hyperparameters and
  detector-facing confidence scaling;
- fix or replace the current error-variance mapping from DFT-domain posterior
  variance to Sionna antenna-domain marginal `err_var`;
- add a short inventory of the paper's related/baseline estimators from
  Section III-E and state why they are not paper-local implementations.

After these revisions are made in `reports/plan.md`, the revised plan can be
sent to the human approval gate. No implementation code should be written
before explicit human approval.

## Harness And Architecture Review

The harness summary is adequate. It states the resource-grid assumptions,
Kronecker pilot pattern, input/output tensor shapes, built-in baselines,
metrics, smoke expectations, and output layout. The plan also correctly
identifies that the benchmark estimator receives only `(y, no)` and must return
full-grid `h_hat` and `err_var`.

The selected architecture is the right class of adaptation:

```text
LS at Sionna pilot locations
-> Baur VAE denoising of one receive-array vector
-> explicit Sionna interpolation with "lin_time_avg"
```

This matches the paper's pilot-decorrelated SIMO observation model in
Equation (1), `y = h + n`, and avoids treating the paper as a native sparse
OFDM full-grid reconstruction method. `FullGridChannelEstimator` is therefore
not appropriate for the planned first implementation. The explicit
`lin_time_avg` interpolation is a necessary benchmark adaptation because Baur
et al. do not define interpolation across OFDM symbols or subcarriers.

Using `BaseChannelEstimator` plus `PilotGroupMixin(group_over=())` is consistent
with existing repository style for receive-array pilot denoisers, especially
the spatial GMM pattern in `papers/koller2022`.

## Method Verdicts

### `VAEChannelEstimator (noisy)`

- Implementer status: `implement`
- Reviewer decision: conditionally accepted, pending required plan revisions.
- Paper coverage: adequate. The plan cites Section III-C variant 2,
  Equations (1), (5)-(7), (10), (12), and (13), and Section III-D.
- Faithfulness: acceptable if the implementation preserves the training
  distinction that the encoder input is noisy `x_tilde = F y`, while the
  decoder likelihood target is the clean transformed channel `x = F h`.
- Implementability: yes. The benchmark simulator can generate clean channel
  vectors for the training target and noisy LS observations for the encoder
  input.
- Required conditions:
  - use a unitary DFT convention so the white LS noise variance remains white
    in the transformed domain;
  - train/evaluate with `z = mu(x_tilde)` as in Equation (12), not Monte Carlo
    averaging unless that is added as a separate paper-justified variant;
  - ensure the decoder covariance output is strictly positive with a documented
    parameterization or floor;
  - select and load the SNR-specific checkpoint deterministically during the
    benchmark sweep.

### `VAEChannelEstimator (real)`

- Implementer status: `implement`
- Reviewer decision: conditionally accepted, pending required plan revisions.
- Paper coverage: adequate. The plan cites Section III-C variant 3 and the same
  estimator equations used by the noisy variant.
- Faithfulness: acceptable if the training objective uses the noisy transformed
  observation `x_tilde = F y` both as encoder input and decoder likelihood
  target, with likelihood covariance `diag(c(z)) + sigma2 I` as described in
  Section III-C.
- Implementability: yes. Clean channels are not needed by the training loss,
  although the simulator may still generate them to synthesize noisy
  observations and to report held-out denoising diagnostics.
- Required conditions:
  - do not use clean channels in the VAE-real ELBO or checkpoint selection loss
    unless the plan explicitly labels that metric as diagnostic only;
  - keep the runtime LMMSE update unchanged from Equation (10), using the
    decoder's channel covariance `c(z)`, not the noisy-observation covariance
    `c(z) + sigma2`.

### `VAE-genie`

- Implementer status: `not implemented`
- Reviewer decision: accepted.
- Rationale: Section III-C variant 1 requires the true channel `h` at
  estimation time to form `x = F h` for the encoder. The benchmark estimator
  API exposes only `(y, no)`. Implementing this variant would require an oracle
  evaluation path analogous to the `perfect` baseline and must not be reported
  as a deployable channel estimator.

### Section III-E Related Estimators

The plan should add a short out-of-scope inventory for the related estimators
used in the paper's comparisons:

- LS is already covered by built-in baselines.
- sample-cov and genie-cov require covariance assets or oracle conditional
  covariance not exposed by the current benchmark; if implemented, they belong
  to a separate reviewed scope.
- GMM circ is prior work from Koller et al. and is already represented by the
  repository's Koller implementation rather than this Baur VAE task.

This omission is not a reason to implement those baselines locally, but the
workflow requires the plan to inventory relevant paper methods and baselines.

## Required Plan Revisions

1. Define SNR-specific checkpoint selection.

   The paper trains separate `VAE-noisy` and `VAE-real` models for individual
   SNR values, while the benchmark instantiates an estimator once and sweeps
   `cfg.snr_db_range()`. The plan must state whether checkpoints are keyed by
   benchmark `Eb/N0`, by derived Sionna `no`, or by a paper-style channel SNR.
   At runtime, the estimator must select the matching checkpoint from `no` or
   from an explicitly supplied SNR list. Silent reuse of one checkpoint for all
   SNRs is not acceptable unless it is reclassified as a documented benchmark
   adaptation and validated as such.

2. Make the `aux/` asset plan executable.

   Use the repository convention based on `evaluation.asset_store.aux_slug_for_config(cfg)`.
   The plan should define exact paths, file formats, and metadata fields. At
   minimum, metadata must include variant, config slug, receive-vector length,
   DFT convention, antenna ordering, SNR/no value, pilot-power convention,
   network architecture, `latent_dim`, `free_bits`, optimizer settings,
   train/validation sizes, seed, selected epoch, and loss history summary. The
   loader must reject mismatched assets rather than silently retraining or
   using incompatible weights.

3. Specify bounded training and hyperparameter validation.

   The paper omits `latent_dim`, free-bits threshold, decoder padding/cropping
   details, and positivity parameterization for `c(z)`. The plan only says
   these will be selected by validation, which is too vague. Add a bounded
   candidate grid and selection criterion for at least `latent_dim`,
   `free_bits`, optimizer batch size, and any deviation from the paper's
   learning rate `1e-4`. Checkpoint selection must use an independent
   validation holdout and report both training and validation loss. Smoke
   training can test wiring only; it must not be used as evidence of method
   quality.

4. Correct the detector-facing `err_var` plan.

   The paper does not define Sionna detector-facing marginal channel-estimation
   variances. If the implementation uses the VAE/LMMSE posterior, the
   DFT-domain posterior diagonal

   ```text
   v_x = c(z) * sigma2 / (c(z) + sigma2)
   ```

   must be converted to antenna-domain marginal variances
   `diag(F^H diag(v_x) F)` before returning `err_var`. For a unitary DFT, this
   diagonal is the average of `v_x` repeated across antenna entries; it is not
   the untransformed DFT-domain vector after `h_hat = F^H x_hat`. If the plan
   instead uses the task fallback `err_var_scale * no / |pilot|^2`, make that
   the primary policy. In either case, `err_var_scale` must be selected by a
   bounded validation sweep before canonical BER/BLER outputs, with the
   candidate grid and criterion documented. The current "fallback if unstable"
   wording is too open-ended for the approval gate.

5. Clarify the array-transform adaptation.

   The paper's DFT argument relies on a ULA Toeplitz/circulant approximation
   from Equations (2) and (13), while the benchmark uses a 4x4 planar base
   station array. A unitary DFT over the flattened antenna index is acceptable
   as a benchmark adaptation only if the plan records the exact flattening
   order and states that results are not a reproduction of the paper's ULA
   curves. A 2D planar DFT is also a possible adaptation, but whichever choice
   is made must be fixed before training and encoded in asset metadata.

## Divergence Ledger

| Divergence | Source | Reviewer verdict | Rationale | Impact on paper faithfulness | Impact on benchmarking fairness |
|---|---|---|---|---|---|
| Sionna LS pilot observations used as `y = h + n` | interface | necessary | The paper starts after pilot decorrelation in Equation (1). Sionna's LS pilot estimator provides the closest matching observation under the benchmark. | Low; preserves the paper's estimator domain at pilot locations. | Fair because all estimators use the same pilot observations. |
| `lin_time_avg` interpolation after VAE pilot denoising | interface | necessary | The paper does not define OFDM frequency/time interpolation, but the benchmark requires full-grid estimates. | Moderate for full-grid NMSE/BER; pilot-domain estimator remains intact. | Fair if named and compared against built-ins with the same grid. |
| Shared VAE trained over all active pilot groups | asset | justified | The paper trains on channel vectors from one scenario, not per-pilot specialized distributions. Pooling across pilot groups follows repository style for shared channel statistics. | Low to moderate; benchmark-specific distribution pooling replaces the paper's dataset. | Fair if metadata and reports state the pooled source. |
| 4x4 planar array with flattened DFT instead of ULA DFT | interface | questionable until revised | The paper's diagonal covariance motivation depends on ULA Toeplitz/circulant structure. Flattened DFT preserves a unitary transformed-domain VAE/LMMSE structure but not the exact physical assumption. | Moderate; results cannot be claimed as reproduction of ULA simulations. | Acceptable if fixed, documented, and applied consistently to all variants. |
| Locally generated VAE weights instead of author weights | asset | necessary | No pretrained Baur VAE weights are present. A faithful implementation needs reproducible local training from the benchmark channel model. | Moderate; benchmark results are a reimplementation on a different scenario. | Fair if training budget, validation, and seeds are reported. |
| Per-SNR VAE training mapped to benchmark SNR sweep | asset | questionable until revised | The paper trains separate practical VAEs per SNR, but the current plan does not specify runtime checkpoint selection from `no`/`Eb/N0`. | High if the wrong checkpoint is used. | Could unfairly favor or penalize the method unless deterministic and documented. |
| Missing VAE hyperparameters selected by validation | paper | justified with concrete protocol | Latent dimension, free-bits threshold, padding, and covariance positivity are underspecified in Section III-D. Validation selection is necessary, but the current plan lacks candidate grids and criteria. | Moderate; selected values become part of the reimplementation. | Fair only if selected before final benchmark results and fully reported. |
| DFT-domain posterior variance returned as Sionna `err_var` | interface | incorrect as currently written | Sionna expects antenna-domain marginal error variances. The posterior covariance must be converted through `F^H diag(v_x) F`, or a documented LS fallback must be used. | Low for channel estimates, potentially high for detector BER/BLER. | Unfair if confidence is miscalibrated or tuned silently. |
| Omission of VAE-genie | interface | necessary | VAE-genie needs true `h` at estimator runtime and is oracle-only under the current contract. | None for deployable estimators; the oracle bound is absent. | Fair if not compared as a missing deployable method. |

## Final Verdict

`VAE-noisy` and `VAE-real` are faithful enough in principle for the Sionna
pilot-domain benchmark, and `VAE-genie` is correctly marked not implemented.
The current plan is blocked pending the revisions above, especially the
SNR/checkpoint mapping and `err_var` correction. After the plan is revised, the
implementer should request human approval before writing any implementation
code.
