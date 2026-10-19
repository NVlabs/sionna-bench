# Implementation Report

Approved plan: `reports/plan.md` after reviewer-requested revisions on 2026-06-24.

## Implemented Scope

Implemented:

- `VAEChannelEstimator` from Section III-C variant 2 (`VAE-noisy`).
- The `VAE-real` training method from Section III-C variant 3, retained as an
  implemented and tested noncanonical method.
- Shared `VariationalAutoEncoder` model and VAE asset-training pipeline.
- Diagonal and full-covariance decoder parameterizations. The committed
  canonical benchmark outputs use the full-covariance parameterization.
- `asset.py` with strict `load_model()` and explicit-preparation
  `train_model()` handles.
- `run_benchmark.py` with smoke, explicit asset preparation, per-SNR
  checkpointing, and explicit method selection.
- `tuning.py` with the standard `run_tuning()` handle.
- Focused tests for VAE shapes, positive decoder covariance, unitary receive-DFT round trip, and transformed-to-antenna variance mapping.

Not implemented:

- `VAE-genie`, because it requires the true channel at estimator runtime.
- Section III-E comparison baselines, because LS is already built in, Koller GMM is implemented in `papers/koller2022`, and sample/genie covariance baselines require separate covariance/oracle scope.

## Files

- `src/sionna_bench/mimo_ofdm_chest/papers/baur2022/estimator.py`
- `src/sionna_bench/mimo_ofdm_chest/papers/baur2022/asset.py`
- `src/sionna_bench/mimo_ofdm_chest/papers/baur2022/run_benchmark.py`
- `src/sionna_bench/mimo_ofdm_chest/papers/baur2022/tuning.py`
- `src/sionna_bench/mimo_ofdm_chest/papers/baur2022/README.md`
- `test/mimo_ofdm_chest/papers/baur2022/test_asset.py`
- `test/mimo_ofdm_chest/papers/baur2022/test_vae.py`

## Deviations From Paper

The paper's ULA DFT/circulant covariance assumption is adapted to the benchmark's 4x4 planar base-station array by applying a unitary 1D DFT over Sionna's flattened receive-antenna axis. This keeps the LMMSE update algebraically consistent but is not a reproduction of the paper's ULA simulation setting.

The paper estimates a SIMO vector after pilot decorrelation. The benchmark provides sparse OFDM pilots and requires full-grid channel estimates. The implementation therefore applies the VAE estimator at Sionna pilot locations after LS pilot estimation and uses an explicit Sionna interpolator for the remaining resource grid. The default and canonical setting is `lin_time_avg`.

The paper specifies free bits but not the threshold, and does not specify the latent dimension. The implementation exposes these through the estimator and asset-preparation APIs and records the training choices in asset metadata.

Section III-D describes the decoder only as analogous to the encoder and flipped symmetrically. The implementation reverses the stated channel progression but uses transposed-convolution kernel size 4, stride 2, and padding 1 to double each intermediate length exactly. The paper explicitly gives kernel size 7 for the encoder but does not specify the decoder's transposed-convolution boundary convention. This choice is now explicit in constants, tests, and asset compatibility metadata.

The paper resamples additive noise every training epoch. The implementation
reuses a finite, deterministic bank of clean channel vectors but generates
fresh deterministic observation noise for every SNR and training epoch. This
preserves the paper's noise-resampling behavior while keeping channel exposure
paired across Optuna trials, and is recorded as
`noise_resampling="deterministic-per-snr-epoch"` in each manifest's
`generation_recipe`.

The diagonal decoder and loss implement the proper-complex Gaussian likelihood in Equation (6) and the Gaussian KL term in Equation (7). The canonical full-covariance extension instead predicts a positive-definite covariance over the `2M` real-stacked transform coefficients through a separate Cholesky head and uses the corresponding real Gaussian negative log-likelihood. For `VAE-real`, adding half the complex noise variance to each real component preserves the paper's `c(z)+sigma2` likelihood convention.

The paper does not specify detector-facing error variances. The implementation uses the VAE/LMMSE posterior variance transformed to antenna-domain marginal variances and exposes `err_var_scale`. A ten-trial 10 dB BLER calibration proposed rounded scale `0.454`, but paired validation tied the existing `0.5405613145` scale and therefore retained it under the strict-improvement rule. The canonical full-covariance results use the retained value.

The canonical leaderboard intentionally contains only `VAEChannelEstimator`,
trained according to the paper's `VAE-noisy` method with clean transformed
channels as decoder-likelihood targets. Other persistently trained neural
estimators in this repository likewise use simulator-provided clean channel
labels, making this the directly comparable Baur method. `VAE-real`, whose
training target is the noisy observation, remains implemented and tested but is
excluded from the leaderboard. This is a benchmark-scope choice rather than a
claim that the paper's noisy-label method is invalid.

## Auxiliary Assets

VAE assets are stored under:

```text
src/sionna_bench/mimo_ofdm_chest/papers/baur2022/assets/<scenario_id>/vae-<method>/<artifact-id>/vae_params.npz
src/sionna_bench/mimo_ofdm_chest/papers/baur2022/assets/<scenario_id>/vae-<method>/<artifact-id>/manifest.json
```

Both asset types use the global asset-manifest schema version 2. Compatibility
records the method, receive-vector length, DFT convention, Sionna antenna
ordering, pilot-power convention, encoder/decoder architecture, covariance
head, latent dimension, and the exact ordered pilot-group coordinates, complex
values, and powers. The `generation_recipe` records SNR/no values,
noise-resampling policy, optimizer settings, train/validation sizes, and seed;
ordered state keys and training history are nested under `metadata`. The strict
loader validates these fields and every checkpoint array's name, shape, dtype,
size, and checksum.

Asset preparation is explicit: `run_benchmark.py --prepare-assets` and
training-parameter tuning call `asset.train_model()` with an external
writable root. Normal benchmark execution and detector calibration call the
strict `asset.load_model()` handle, which rejects a missing or incompatible
checkpoint rather than training it. Both paths inject the returned state
dictionaries and optional per-SNR noise variances into
`VAEChannelEstimator`; estimator construction therefore performs no persistent
I/O, channel sampling, or VAE training.

The estimator batches all pilot-group receive vectors through the selected VAE
and posterior update. A single mixed-SNR state gives Tier A full-graph
inference. The canonical per-SNR state bank is Tier B: `_select_vae_index()` is
the only explicitly compiler-disabled operation, because it converts the
runtime tensor match into a Python `ModuleList` index. The selected VAE,
diagonal or full-covariance posterior update, and Sionna interpolation remain in
compiled tensor code. `torch._dynamo.explain` reports no break for mixed-SNR
inference and only the explicit state-selection boundary for per-SNR inference.

The selected canonical `VAE-noisy` bank uses latent dimension 16, free bits
`0.875`, per-SNR weights over the UMi benchmark grid, learning rate
`0.000230`, optimizer batch size 512, covariance floor `1e-5`, and seed 42.
Each model uses the same fixed bank of 6,400 independent training and 256
independent validation channels, with deterministic fresh noise for each SNR
and training epoch. Training permits at most 200 epochs, uses patience 20 and
relative minimum validation improvement `1e-4`, and restores the best
checkpoint. `VAE-real` remains implemented and tested but is not canonical.

## Verification

Commands run:

```bash
python -m py_compile src/sionna_bench/mimo_ofdm_chest/papers/baur2022/estimator.py src/sionna_bench/mimo_ofdm_chest/papers/baur2022/asset.py src/sionna_bench/mimo_ofdm_chest/papers/baur2022/tuning.py src/sionna_bench/mimo_ofdm_chest/papers/baur2022/run_benchmark.py
```

Result: passed.

```bash
python -m pytest -q test/mimo_ofdm_chest/papers/baur2022
```

Result after the calibrated v2 asset changes: `55 passed`. The suite verifies the
exact encoder/decoder parameterization and independently reconstructs the
diagonal complex and full real ELBO losses for both `VAE-noisy` and
`VAE-real`, in addition to the transform, covariance, asset, geometry, and
tiny-training checks. It also compares eager and compiled full-covariance
inference for both a Tier A mixed-SNR state and the Tier B per-SNR selection
boundary, and verifies that detector confidence scaling leaves the posterior
channel estimate unchanged.

```bash
python -m sionna_bench.mimo_ofdm_chest.papers.baur2022.run_benchmark --smoke --prepare-assets --asset-root /tmp/baur-assets
python -m sionna_bench.mimo_ofdm_chest.papers.baur2022.run_benchmark --smoke --asset-root /tmp/baur-assets
```

Migration smoke result at 5 dB: BER `9.661937e-02`, BLER `5.000000e-01`,
NMSE `2.836181e-02`. This is an integration check with a deliberately tiny
one-channel, one-epoch model, not performance evidence. The temporary asset
tests separately train and strictly reload both implemented training methods.

```bash
python -m sionna_bench.mimo_ofdm_chest.papers.builtin.run_benchmark --smoke
```

Smoke baseline NMSE at 5 dB:

- `ls-nn`: `3.762552e-02`
- `ls-lin`: `1.512823e-02`
- `lmmse`: `6.784751e-03`
- `perfect`: `0.000000e+00`

Both implemented VAE training methods produce finite values and beat `ls-nn`
on NMSE in the historical smoke run. The smoke run is not final performance
evidence because it uses one tiny training batch and one Monte Carlo iteration;
only the clean-label `VAE-noisy` method is retained for canonical comparison.

Canonical full-covariance asset generation and benchmark commands:

```bash
python -m sionna_bench.mimo_ofdm_chest.papers.baur2022.run_benchmark --prepare-assets --asset-root "$FINAL_ASSET_ROOT" --training-data-manifest "$TRAINING_DATA_MANIFEST" --method noisy --covariance-mode full --latent-dim 16 --train-batches 400 --train-batch-size 16 --validation-batches 16 --validation-batch-size 16 --epochs 200 --optimizer-batch-size 512 --learning-rate 0.000230 --free-bits 0.875 --patience-epochs 20 --relative-min-delta 1e-4 --covariance-floor 1e-5
python -m sionna_bench.mimo_ofdm_chest.papers.baur2022.run_benchmark --asset-root "$FINAL_ASSET_ROOT" --method noisy --err-var-scale 0.5405613145449313
```

Canonical full-covariance NMSE:

| Estimator | -10 dB | -7.5 dB | -5 dB | -2.5 dB | 0 dB | 2.5 dB | 5 dB | 7.5 dB | 10 dB | 12.5 dB | 15 dB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `VAEChannelEstimator` | `8.317239e-02` | `5.182288e-02` | `3.189962e-02` | `1.933015e-02` | `1.198339e-02` | `7.313015e-03` | `4.461935e-03` | `2.774516e-03` | `1.757453e-03` | `1.152724e-03` | `7.923961e-04` |

## Detector-Facing Calibration

The frozen final bank was evaluated in a ten-trial log-scale search over
`err_var_scale` in `[1e-3, 1]`, using codeword BLER at 10 dB, 63 Monte Carlo
batches of size 4, sampler seed 456, and probe seed 271828. The raw winner
`0.4540521216` was rounded to `0.454`. Paired validation with seed 161803 gave
the same BLER, `0.0211640212`, for the rounded candidate and existing
`0.5405613145` scale. The existing scale was retained because the candidate
did not improve it strictly. Search and validation took 20.30 and 9.68
seconds.

## Calibrated V2 Evidence

All accepted work used calibrated Sionna commit
`f16208347246d8d17ec69628dfb856b19287b80c`, including channel-model commit
`ad1609ea1`, explicit UMi TR 38.901 version 19.2, Python 3.12.3, PyTorch
2.11.0+cu126, an NVIDIA L40S, and an AMD EPYC 7R13. The fixed-data manifest
has SHA-256
`35b7728ac44e738ba696df25fdbf5e7f8b69887934268c185c7c8caeee1f9718`.
Ten complete full-effort algorithm trials selected rounded recipe
`(latent_dim=16, learning_rate=0.000230, free_bits=0.875,
optimizer_batch_size=512)`. On 63 paired batches of four channels at each
tuning SNR it achieved weighted NMSE `0.0044366964`, versus `0.0045868521`
for the freshly retrained prior recipe.

The study database retains failed trial 9 and interrupted duplicate recovery
trial 10 alongside the ten complete trials; verified trial 11 supplied the
tenth complete recipe. The initial search segment took 100:05:19, the accepted
replacement took 17:38:05, and paired validation took 14:26:01. Earlier
unseeded and negative-loss early-stopping attempts were quarantined before the
accepted restart and contribute neither selected assets nor accepted runtime.

The final eleven-SNR bank took 111,196.71 seconds (30.89 hours) to train and
contains 517 finite state tensors. Its exact audit has SHA-256
`17c4f440bee3ac63216e71f9bf96bea80e43b2085229b5359bb5845d6c0d3edf`.
The 63-by-4-channel canonical benchmark took 44.66 seconds and produced result
SHA-256
`8b560cff328d6246f6975265b7653f48f8f9f3b96337ed5fa9825fddaff0fe0a`.
Its five-SNR weighted NMSE is `0.0042578149`, 2.63% above the comparison
worktree result; every pointwise NMSE confidence interval overlaps, while BER
and BLER improve over the complete curve.
