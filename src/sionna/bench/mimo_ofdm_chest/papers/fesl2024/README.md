# Implementation of the Estimator from Fesl et al., 2024

Publication: B. Fesl, M. Baur, F. Strasser, M. Joham, and W. Utschick,
“Diffusion-Based Generative Prior for Low-Complexity MIMO Channel Estimation,”
IEEE Wireless Communications Letters, vol. 13, no. 12, pp. 3493–3497, 2024, [DOI: 10.1109/LWC.2024.3474570](https://doi.org/10.1109/LWC.2024.3474570).
[[IEEE](https://ieeexplore.ieee.org/abstract/document/10705115)] [[arXiv](https://arxiv.org/abs/2403.03545v1)]


## Abstract

Fesl et al. learn an angular-domain channel prior with a noise-conditional
convolutional diffusion model. At inference, the estimator normalizes an LS
observation using its actual noise variance, matches it to the diffusion
schedule, and applies deterministic reverse conditional-mean steps. One shared
CNN serves all timesteps and observation SNRs.

This implementation applies the estimator to receive-array vectors at active
Sionna OFDM pilot locations. A physical two-dimensional receive-array Fourier
transform replaces the paper's receive/transmit transform. Sionna interpolation
then reconstructs the full resource grid; the default is `lin_time_avg`.

## Implemented Classes

`DiffusionChannelEstimator` in `estimator.py` is a Sionna
`BasePilotChannelEstimator`. It uses `_PilotGroupMixin` to gather active pilots
and accepts a prepared model state, frozen training channel power, and physical
`receive_array_shape`. Construction performs no file lookup or training.
Benchmark outputs are named `DiffusionChannelEstimator`.

Main methods:

- `estimate_at_pilot_locations()`: computes LS pilot estimates and their
  noise variances, groups receive vectors, denoises them, and returns pilot
  estimates with calibrated empirical variances.
- `_to_angular()` / `_from_angular()`: restore physical antenna ordering and
  apply the unitary 2D FFT or its inverse.
- `_prepare_observations()` / `_reverse_diffusion()`: normalize using actual
  LS noise and execute the matched deterministic mean updates.

The internal CNN preserves the original five 3×3 convolutions with widths
`(2,32,64)` and `(64,43,22,2)`, 16-dimensional sinusoidal conditioning, residual
affine feature modulation, and 55,025 parameters. The 100-step linear VP
schedule and unweighted epsilon-MSE objective follow the released WCL recipe.
The real and imaginary noise components each have variance 1/2.

`asset.py` separates explicit `generate_training_data()` / `train_model()`
from strict `load_training_data()` / `load_model()`. `tuning.py` exposes
`run_tuning()` for optimizer studies or frozen-model variance calibration.
`run_benchmark.py` loads assets, constructs the estimator, evaluates it with
the shared pipeline, and records effective arguments and manifest identities.

Inference is Tier B: runtime matched-timestep bucketing remains outside the
compiled tensor region. Eager and `torch.compile(..., backend="eager",
fullgraph=False)` outputs agree in the tested geometries and precisions.

## Modifications and Omissions

- The paper's 64×16 channel matrix is adapted to a physical receive-array
  snapshot. The selected benchmark uses one single-polarized 4×4 BS array and
  four single-stream UEs. Each active pilot vector contains all 16 receive
  antennas; users and pilot frequencies are not substituted for transmit-array
  coordinates. Other rectangular arrays require compatible trained assets and
  separate variance calibration. Noise must be isotropic within each vector.
- Unitary pilot decorrelation becomes sparse OFDM pilot division. Actual LS
  noise `no / abs(pilot)**2`, divided by frozen training channel power, controls
  normalization and timestep matching. Eb/N0 is not used as pilot SNR.
- Training-only population power implements the paper's unit-average-power
  convention. Test channels and individual observations are not used to
  estimate this normalization.
- Coefficient index `k` denotes paper step `k+1`. Algorithm 1 applies
  `k,...,0` inclusively; the inspected `wcl-2024` release applies `k-1,...,0`
  after selecting `k`. This implementation follows the paper, adding the
  matched transition. Complete-chain equality to the release is not claimed.
  Exact zero noise returns the LS endpoint; finite noise outside the schedule
  matches its nearest endpoint.
- The paper does not define OFDM interpolation. Explicit `lin_time_avg`
  reconstruction fills off-pilot resources; the standard optional custom
  interpolator takes precedence.
- The paper supplies no detector-facing posterior variance. The fallback is
  `err_var_scale * no / abs(pilot)**2` at pilots, propagated by Sionna's
  interpolator. The selected scale is `0.0358`, calibrated by the maintainers
  on the regenerated canonical model; this is empirical confidence
  calibration, not a diffusion posterior covariance.
- The selected UMi model uses 6,400 independent training and 256 validation
  realizations, producing 7,987,200 and 319,488 correlated pilot vectors.
  The same trained model serves the entire SNR range. The released recipe uses
  100,000/10,000 training/validation matrices and up to 500 epochs.
- Spatial-domain ablations, alternative schedules, SNR-mismatch experiments,
  paper comparison estimators, and single-step inference are not implemented.
  The adapted benchmark does not reproduce the paper's original geometry or
  establish a finite-network exact-MMSE guarantee.

### Potential Extensions

Two research directions remain open; neither is implemented or evaluated in
this contribution:

- **Single-pass inference.** Strasser, Bäro, and Utschick (2025),
  [“Enhancements in Score-based Channel Estimation for Real-Time Wireless
  Systems”](https://arxiv.org/abs/2509.07839), derive an SNR-informed single-step
  denoiser in Section IV-B / Algorithm 3. Its variance-exploding score-model
  formulation differs from the VP model used here. A compatible single-pass
  derivation could reuse the separated preprocessing, CNN, schedule, and
  reconstruction components, subject to validating the retained checkpoint's
  noise parameterization, accuracy, latency, and detector performance. The
  paper's single-carrier results do not establish those gains for this UMi
  OFDM adaptation.
- **Direct full-grid channel reconstruction.** A separate extension could
  learn joint spatial, frequency, and time structure and condition on sparse
  OFDM pilots to estimate the complete channel grid directly. This would
  replace independent pilot-vector denoising followed by fixed interpolation.
  It requires a compatible observation model, network/training assets, and
  fresh detector-variance calibration and benchmark evaluation. Potential
  accuracy gains remain unmeasured; this is a proposed OFDM extension, not a
  result demonstrated by the single-pass paper.

## Asset Generation

Public distributions omit data and trained payloads. Generation and training
require an explicit external asset root. Missing or incompatible assets fail
with an actionable error; evaluation never generates, downloads, or trains
implicitly. Schema-2 manifests bind geometry, pilots, transform, normalization,
architecture, schedule, purpose, generation recipe, and payload checksums.

The external layout is:

```text
<asset_root>/3gpp-umi/fesl2024/diffusion-training-data/<id>/
  manifest.json, train.npy, validation.npy, normalization.npz
<asset_root>/3gpp-umi/fesl2024/diffusion-prior/<id>/
  manifest.json, diffusion_params.npz
```

The canonical asset was regenerated by the maintainers with the runner
defaults: Adam, learning rate `1e-4`, batch size 512, max 200 epochs,
patience 20, and relative minimum improvement `1e-4`. Training ran all 200
epochs without early stopping; the minimum validation epsilon-MSE
`0.0807736455` selected epoch 197. Training channel power is
`1.627766664785857`. Training/validation propagation seeds are
24250000/24260000; initialization, epoch/example diffusion noise, and fixed
validation noise use 24270000, 24280000, and 24290000, respectively. Training
took 8,934 s on one NVIDIA L40S (CUDA 12.6, PyTorch 2.11.0, Sionna 2.1.0).
The dataset manifest SHA-256 is
`29563f0993c5bb94939f6f9a5e9b2abcf3f62d204ae68d3f4cd4b4bb6612641b`, the model
manifest SHA-256 is
`b18282e8bfdb466ab248cb134459f46dd29976d7a2892175ed1f28efb3c2fbbd`, and the
inference payload SHA-256 is
`09f1ef549f6f2e6a2ec528020e9791b3582fd304e6ad44fe574514157b7afb04`.

The authors' original reference fit used the same recipe and seeds on their own
host (CPU data generation, MPS training), whose random streams differ from this
CUDA regeneration. It selected epoch 170 with training channel power
`1.6261395394191218` and is not distributed; its record is retained in
[the implementation report](reports/implementation_report.md).

The recipe itself was selected by the authors in their own external study: ten
serial TPE trials over learning rate log `[1e-5,1e-3]` and batch size
`{128,256,512,1024,2048}`, sampler seed 24300000 and startup five, minimizing
linear-SNR-weighted NMSE at `{-10,-5,0,5,10}` dB with three batches of four
realizations per SNR (seed 24310000). No candidate beat the reference; the raw
winner, LR `0.0008461960538786609` with B1024, was worse on independent paired
confirmation (seed 24320000, 63×4 realizations per SNR), so the reference was
retained. The maintainers did not reproduce this study and take the recipe as
reported.

`run_benchmark --prepare-assets --asset-root "$ASSET_ROOT"` is the separate
canonical preparation mode and exits without evaluation. Its defaults are this
recipe, so it regenerates the canonical dataset and model; the training study's
enqueued first trial equals the same defaults. Historical runs require their
frozen sources and environment.

## Detector-Facing Variance Calibration

With the regenerated canonical model frozen, the maintainers ran the
benchmark's calibration: ten TPE trials searched only `err_var_scale`
logarithmically over `[1e-3, 1]`, minimizing BLER at 10 dB on 63 batches of
four complete realizations (calibration seed 24330000, sampler seed 24350000).
Cached observations, estimates, and unscaled variances were reused. The
objective is flat over most of the range: the nine trials between `0.00102`
and `0.623` produced 82 or 83 block errors out of 3,024 codewords, and only
`0.903` was worse with 86.

Raw winner `0.03581037222144431`, the first of four trials tied at 82 errors,
rounded to `0.0358`. A separate paired cohort of another 252 realizations,
seed 24340000, gave BLER `0.026785714286` at the authors' incumbent `0.623`
and `0.026455026455` at `0.0358`, one codeword fewer. The strict-improvement
rule therefore selected `0.0358`, now the constructor default. This is not a
statistical significance claim. Channel estimates and NMSE are independent of
this scale. The authors' own calibration on their reference fit had selected
`0.623` over the uncalibrated scale 1; that record is retained in the
implementation report.

## Benchmark Run

Set `ASSET_ROOT` to the external root containing the canonical assets:

```bash
python -m sionna.bench.mimo_ofdm_chest.papers.fesl2024.run_benchmark \
  --asset-root "$ASSET_ROOT" --device cuda:0
```

Canonical evaluation requires canonical-purpose assets and smoke evaluation
requires smoke-purpose assets; a manifest with a different recorded purpose is
refused. The constructor default is `0.0358`; `--err-var-scale 1` requests the
uncalibrated LS confidence. Smoke preparation and evaluation use the standard
separate `--smoke --prepare-assets` and `--smoke` calls with an external asset
root.

The canonical evaluation used `umi_3gpp_config`: 52 PRBs, 14 OFDM symbols,
pilot symbols `(2,11)`, a 4×4 BS array, four UEs, 63 batches of four
realizations at each of eleven SNRs, setup seed 42, and evaluation seed 2026,
in single precision on one NVIDIA L40S. Like the other canonical entries it
ran on CUDA with the benchmark's common seeds. The results are tracked under
`outputs/mimo_ofdm_chest/fesl2024/3gpp-umi/` and listed in its `results.json`;
the NPZ SHA-256 is
`4d9da3b51cfff3017d16e7cad12baa388969f095aba59ab63f8538ea325694ca`.

| Eb/N0 (dB) | NMSE | BER | BLER |
| -: | -: | -: | -: |
| -10 | 0.098842669 | 0.228839512 | 0.962962963 |
| -7.5 | 0.058598213 | 0.178206347 | 0.871693122 |
| -5 | 0.034749309 | 0.129166667 | 0.729166667 |
| -2.5 | 0.021619389 | 0.085462157 | 0.548941799 |
| 0 | 0.013124813 | 0.049550176 | 0.341931217 |
| 2.5 | 0.008036669 | 0.026328714 | 0.193452381 |
| 5 | 0.004614079 | 0.012635753 | 0.093253968 |
| 7.5 | 0.003116372 | 0.006369412 | 0.047949735 |
| 10 | 0.001778366 | 0.002780704 | 0.021825397 |
| 12.5 | 0.001367779 | 0.001140569 | 0.009920635 |
| 15 | 0.000878081 | 0.000399199 | 0.004298942 |

Against the twenty other tracked entries, diffusion's NMSE ranks 6th to 12th,
with ranks `[12,11,11,11,11,9,8,7,6,7,7]` from -10 dB through 15 dB. Spectral
and spatial GMM, LMMSE, RKHS, and the VAE have lower NMSE throughout; the LS,
Coleri, Edfors-LS, and sparse/diffuse estimators have higher NMSE throughout;
the remaining methods cross it. The authors' original CPU evaluation of their
own fit at scale `0.623` is recorded in the implementation report; its NMSE
agrees with this run within the reported confidence intervals.

Implementation, assets, selection evidence, and verification are recorded in
[implementation_report.md](reports/implementation_report.md); the independent
verdict is in [final_report.md](reports/final_report.md).
