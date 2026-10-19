# Final Audit Report

- Paper ID: `baur2022`
- Reviewer: Codex reviewer agent
- Date: 2026-06-24
- Verdict: accepted with caveats

## Reviewed Artifacts

- `prompts/reviewer.md`, `prompts/task_reviewer.md`, and `prompts/task.md`
- `src/sionna_bench/mimo_ofdm_chest/papers/baur2022/paper.pdf`
- `src/sionna_bench/mimo_ofdm_chest/papers/baur2022/reports/paper_summary.md`
- `src/sionna_bench/mimo_ofdm_chest/papers/baur2022/reports/plan.md`
- `src/sionna_bench/mimo_ofdm_chest/papers/baur2022/reports/plan_review.md`
- `src/sionna_bench/mimo_ofdm_chest/papers/baur2022/reports/implementation_report.md`
- `src/sionna_bench/mimo_ofdm_chest/papers/baur2022/estimator.py`
- `src/sionna_bench/mimo_ofdm_chest/papers/baur2022/asset.py`
- `src/sionna_bench/mimo_ofdm_chest/papers/baur2022/run_benchmark.py`
- `src/sionna_bench/mimo_ofdm_chest/papers/baur2022/tuning.py`
- `src/sionna_bench/mimo_ofdm_chest/papers/baur2022/README.md`
- `test/mimo_ofdm_chest/papers/baur2022/test_vae.py`

## Summary

The implementation follows the reviewed pilot-domain architecture:

```text
Sionna LS pilot estimates -> Baur VAE/LMMSE denoising per receive-array vector
-> explicit lin_time_avg interpolation
```

This is the right benchmark adaptation for Baur et al.'s post-pilot-decorrelation
SIMO observation model `y = h + n` in Equation (1). Both implemented training
methods preserve the core runtime estimator: DFT-domain VAE moments,
`z = mu(x_tilde)` as in Equation (12), Equation (10)'s conditional LMMSE
update, inverse DFT, and antenna-domain marginal posterior variance for Sionna
`err_var`.

The `VAE-noisy`/`VAE-real` distinction is implemented correctly in the training
objective. For `noisy`, the encoder input is noisy and the decoder target is the
clean transformed channel. For `real`, both encoder input and target are noisy,
and the likelihood variance is `decoder_channel_variance + observation_noise`.
The saved smoke metadata records these distinctions.

The canonical leaderboard now selects only Section III-C variant 2,
`VAE-noisy`, and displays it as `VAEChannelEstimator`. This variant uses noisy
encoder inputs and clean channel targets, matching the label access used by the
repository's other persistently trained neural estimators. `VAE-real` remains
implemented and tested as the paper's noisy-label alternative but is excluded
from canonical comparison. This is a deliberate benchmark-scope adaptation,
not a correction to the paper's method.

`VAE-genie` is correctly omitted because the benchmark estimator interface does
not expose the true channel at runtime. Section III-E comparison baselines are
also reasonably kept out of this paper folder.

## Findings

### 1. Finite clean-channel bank with deterministic per-epoch noise

Paper Section III-D creates new noisy observations in every training epoch by
resampling the additive noise in Equation (1). The accepted `VAE-noisy`
training path stores a finite deterministic bank of clean channel vectors and
pilot powers, then regenerates observation noise deterministically for every
SNR and epoch. It therefore preserves the paper's per-epoch noise resampling
while giving every Optuna trial identical clean-channel exposure. Training and
validation use disjoint banks of 6,400 and 256 channels per SNR.

### 2. Superseded: only smoke settings had been run at original audit time

At original audit time, the existing Baur outputs were smoke-only, one SNR
point at 5 dB, one Monte Carlo iteration, and VAE assets trained with
`train_batches=1`, `train_batch_size=1`, and `epochs=1`. Those smoke outputs
were valid integration checks, not performance evidence.

The calibrated canonical results use full-covariance VAEs with latent
dimension 16, free bits `0.875`, per-SNR weights, 6,400 training and 256
validation channels, at most 200 epochs, patience 20, optimizer batch size
512, learning rate `0.000230`, covariance floor `1e-5`, and
`err_var_scale=0.5405613145`.

### 3. Error variance handling is calibrated

The implementation converts the transformed-domain posterior diagonal variance
to antenna-domain marginal variances by averaging the DFT-domain posterior
variance under the unitary DFT. This addresses the plan-review objection that
Sionna expects antenna-domain marginal `err_var`, not transformed-domain
variance. A ten-trial 10 dB BLER search proposed rounded scale `0.454`, but
canonical-sized paired validation tied the existing `0.5405613145` scale at
BLER `0.0211640212`. The strict-improvement policy retained the existing
scale. Scaling changes detector confidence but not the channel posterior mean
or NMSE.

### 4. Asset metadata are adequate

The VAE assets include method, scenario, receive-vector length, DFT
convention, SNR/no value, pilot-power convention, architecture, latent
dimension, free bits, optimizer settings, covariance mode, train/validation
sizes, seed, selected epoch, and loss history. The loader now checks the
explicit `compatibility` and `generation_recipe` sections before reuse and does not
depend on legacy `key_fields`.

The later full-covariance canonical assets record the selected practical
settings in the artifact identity and metadata. Detector calibration reuses those
frozen assets and therefore does not alter trained-weight identity.

### 5. Post-cleanup tests cover the approved contract

The current Baur suite contains 55 tests. It covers VAE output shapes and
covariance positivity, transform round trips, posterior calculations,
equation-level losses, asset compatibility and malformed checkpoints, tiny
training for both methods, changed system dimensions, eager/compiled
equivalence, and the detector-scale invariance of the posterior mean.

## Verification

Commands rerun during this audit:

```bash
python -m py_compile \
  src/sionna_bench/mimo_ofdm_chest/papers/baur2022/estimator.py \
  src/sionna_bench/mimo_ofdm_chest/papers/baur2022/asset.py \
  src/sionna_bench/mimo_ofdm_chest/papers/baur2022/tuning.py \
  src/sionna_bench/mimo_ofdm_chest/papers/baur2022/run_benchmark.py
```

Result: passed.

```bash
python -m pytest -q test/mimo_ofdm_chest/papers/baur2022/test_vae.py
```

Result after the calibrated v2 asset changes: `55 passed`.

Existing smoke outputs inspected:

| Estimator | Eb/N0 | NMSE | BER | BLER |
|---|---:|---:|---:|---:|
| `VAEChannelEstimator` (`VAE-noisy`) | 5.0 | `1.281847e-02` | `7.335142e-02` | `3.750000e-01` |
| `VAE-real` (noncanonical check) | 5.0 | `1.095399e-02` | `3.960768e-02` | `3.750000e-01` |
| `ls-nn` smoke baseline | 5.0 | `3.762552e-02` | `9.094324e-02` | `5.000000e-01` |
| `ls-lin` smoke baseline | 5.0 | `1.512823e-02` | `3.347245e-02` | `2.500000e-01` |
| `lmmse` smoke baseline | 5.0 | `6.784751e-03` | `9.063720e-03` | `1.250000e-01` |
| `perfect` smoke baseline | 5.0 | `0.000000e+00` | `4.792710e-02` | `3.750000e-01` |

The smoke VAE variants produce finite values and beat `ls-nn` on NMSE in this
single smoke setting. The smoke run is too small for performance conclusions,
and `lmmse` remains stronger in the saved smoke outputs.

## Divergence Ledger

| Divergence | Source | Reviewer verdict | Rationale | Impact on paper faithfulness | Impact on benchmarking fairness |
|---|---|---|---|---|---|
| Sionna LS pilot estimates used as Baur's `y = h + n` input | interface | necessary | The paper starts after pilot decorrelation; Sionna LS pilot estimates are the closest benchmark-domain observation. | Low for pilot-domain estimation. | Fair because all estimators use the same pilot grid and noise. |
| Explicit `lin_time_avg` interpolation after VAE pilot denoising | interface | necessary | The paper does not define OFDM time/frequency interpolation, but the benchmark requires full-grid estimates. | Moderate for full-grid metrics; the paper estimator itself remains pilot-domain. | Fair if reported as a benchmark adaptation. |
| 4x4 planar BS array with flattened 1D receive DFT instead of paper ULA DFT | interface | justified | The paper's Toeplitz/circulant motivation assumes a ULA. The unitary flattened DFT preserves the algebra but not the physical ULA setting. | Moderate; results are not reproductions of the paper's ULA curves. | Fair if both VAE variants use the same transform and metadata records it. |
| Locally trained benchmark-channel VAE weights instead of author weights and paper datasets | asset | necessary | No pretrained Baur assets are present, and the task requires benchmarking on the local Sionna scenario. | Moderate; this is a reimplementation on a different channel distribution. | Fair if training budget, validation, and seeds are reported. |
| Per-SNR checkpoints keyed by Sionna `Eb/N0`-derived noise variance | interface | justified | The paper trains practical VAEs per SNR; the benchmark controls SNR through Sionna `no`. Runtime selection by log noise variance is deterministic. | Low to moderate due to SNR-definition mismatch. | Fair if all requested SNRs have matching checkpoints. |
| `VAE-noisy` clean decoder target with noisy encoder input | paper | justified | Matches Section III-C variant 2. | Low; preserves the intended practical-with-clean-training-data variant. | Fair; uses simulator-clean channels only in training, as documented. |
| `VAE-real` noisy target with likelihood variance `c(z)+sigma2` | paper | justified | Matches Section III-C variant 3 and keeps decoder covariance as channel covariance. | Low; preserves the intended noisy-only training objective. | Fair; no clean channel is used in the `VAE-real` ELBO. |
| `VAE-real` omitted from the canonical leaderboard | benchmark scope | justified | The selected `VAE-noisy` method uses clean channel labels like the repository's other persistently trained neural estimators; retaining both would compare two label-access regimes from one paper. | None for the implemented methods; both training objectives remain available and tested. | Improves comparability, provided the omission and selected method are explicit. |
| Fixed clean-channel bank reused across trials | implementation | justified | The implementation stores 6,400/256 clean channel blocks once and adds deterministic fresh noise for every SNR and epoch, preserving Section III-D's noise resampling while keeping trials paired. | Low; the underlying channel sample is finite, but noise is not frozen across epochs. | Improves reproducibility and gives every hyperparameter trial the same channel exposure. |
| `VAE-genie` omitted | interface | necessary | It requires true `h` at estimator runtime, which the benchmark estimator API does not expose. | No impact on deployable estimator faithfulness. | Fair if not compared as a missing deployable method. |
| Section III-E `sample-cov`, `genie-cov`, LS, and GMM comparison baselines omitted from this folder | implementation | justified | LS is built in, GMM is represented separately, and covariance/genie baselines require separate assets or oracle state. | Low; they are comparison baselines, not the Baur VAE methods. | Fair if reports do not imply those baselines were implemented here. |
| VAE/LMMSE posterior transformed to antenna-domain marginal `err_var` and scaled by calibrated `err_var_scale` | interface | justified | The paper does not define detector-facing variances; this is the approved Sionna adaptation and fixes the DFT-domain variance issue. Paired validation tied a newly proposed scale and retained `0.5405613145`. | Low; the posterior mean is unchanged and the returned confidence remains posterior-derived. | Fair; calibration used independent 10 dB data and the canonical benchmark was not used for selection. |
| Canonical full-covariance assets and outputs added after original smoke audit | asset | justified | The run uses per-SNR assets with non-smoke train/validation settings and records the selected settings. | Low for implementation readiness; moderate because this is still a local benchmark-channel training setup. | Sufficient for leaderboard reporting with the documented caveats. |

## Post-Audit Canonical Update

The canonical output uses `VAEChannelEstimator`, corresponding to the paper's
`VAE-noisy` method. It was trained per SNR on the UMi benchmark configuration
from 6,400 independent training and 256 independent validation channels. Each
channel yields 1,248 receive-transform examples; noise is regenerated
deterministically by SNR and epoch. `VAE-real` remains implemented and tested
but is not a canonical leaderboard entry. The paper's diagonal-covariance
parameterization remains implemented, but is not the benchmark setting.

## Post-Cleanup Faithfulness Recheck

The encoder channel counts, kernel size, stride, Batch Normalization, and ReLU
placement were rechecked against Section III-D. The decoder reverses the paper's
channel progression, while its transposed-convolution kernel size 4, stride 2,
and padding 1 are an explicit exact-doubling boundary convention not specified
by the paper. The full-covariance benchmark extension uses a separate real
Cholesky head and is not presented as the paper's diagonal complex covariance.

Equation-level tests independently reproduce the proper-complex Gaussian
negative log-likelihood in Equation (6), the KL term in Equation (7), the
`VAE-real` observation-noise addition, and the corresponding real Gaussian
loss used by the full-covariance extension. Existing tests retain the unitary
DFT round trip, antenna-domain variance identity, positive-definite covariance,
and tiny training/inference checks for both practical variants. Asset tests
also verify deterministic fresh-noise training, separate validation channels,
best-checkpoint restoration, and strict manifest linkage.

Canonical full-covariance NMSE:

| Estimator | -10 dB | -7.5 dB | -5 dB | -2.5 dB | 0 dB | 2.5 dB | 5 dB | 7.5 dB | 10 dB | 12.5 dB | 15 dB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `VAEChannelEstimator` | `8.317239e-02` | `5.182288e-02` | `3.189962e-02` | `1.933015e-02` | `1.198339e-02` | `7.313015e-03` | `4.461935e-03` | `2.774516e-03` | `1.757453e-03` | `1.152724e-03` | `7.923961e-04` |

The final-bank generation consumed 111,196.71 seconds (30.89 hours). Its
eleven manifests contain 517 finite state tensors and pass the exact audit
with SHA-256
`17c4f440bee3ac63216e71f9bf96bea80e43b2085229b5359bb5845d6c0d3edf`.
The canonical benchmark used 63 batches of four channels per SNR, evaluation
seed 2026, and calibrated Sionna commit
`f16208347246d8d17ec69628dfb856b19287b80c` with explicit UMi v19.2. Its
result SHA-256 is
`8b560cff328d6246f6975265b7653f48f8f9f3b96337ed5fa9825fddaff0fe0a`.
All pointwise NMSE confidence intervals overlap the comparison worktree's
curve; weighted NMSE is 2.63% higher, while BER and BLER improve throughout.
The training study retained ten complete and two failed trial numbers. The
failed numerical trial and interrupted duplicate recovery were excluded from
selection; earlier unseeded and invalid negative-loss early-stopping attempts
were quarantined. The accepted search and replacement consumed 100:05:19 and
17:38:05, and paired validation consumed 14:26:01.

## Final Verdict

Accepted with caveats.

The implementation is acceptable as a Baur 2022 Sionna adaptation with the
caveats documented above. The calibrated full-covariance benchmark resolves
the original smoke-only limitation. Detector-facing variance calibration and
paired validation are complete. The principal remaining deviations are the
flattened planar-array DFT, the full-covariance extension, and explicit OFDM
interpolation, rather than the paper's original ULA and diagonal-covariance
evaluation setting.
