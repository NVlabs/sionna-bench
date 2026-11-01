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
  interpolator. The selected scale is `0.623`; this is empirical confidence
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

The retained reference uses Adam, learning rate `1e-4`, batch size 512,
max 200 epochs, patience 20, and relative minimum improvement `1e-4`.
The absolute minimum validation epsilon-MSE selected epoch 170; training
stopped at epoch 190. Training channel power is `1.6261395394191218`.
Training/validation propagation seeds are 24250000/24260000; initialization,
epoch/example diffusion noise, and fixed validation noise use 24270000,
24280000, and 24290000, respectively.

Ten serial TPE trials searched learning rate log `[1e-5,1e-3]` and batch size
`{128,256,512,1024,2048}`, with sampler seed 24300000 and startup five.
All ten completed, with zero failed/pruned/replacement fits, in 40.40 active
hours. Selection minimized linear-SNR-weighted NMSE at `{-10,-5,0,5,10}` dB,
three batches of four realizations per SNR, seed 24310000. No candidate beat
the reference. The preselected raw winner, LR `0.0008461960538786609`, B1024,
epoch 102, alone received independent paired confirmation: 63×4 realizations
per SNR, seed 24320000. Its weighted NMSE `0.004720078666` was worse than
reference `0.004703574668`, so the original reference was retained. An explicit
approved exception omitted an additional rounded challenger fit.

`run_benchmark --prepare-assets --asset-root "$ASSET_ROOT"` is the separate
canonical preparation mode and exits without evaluation. Its default batch
size is 128, so it does not regenerate the retained B512 reference unchanged.
Use the explicit `asset.train_model()` recipe and recorded seeds in
[the implementation report](reports/implementation_report.md) to reproduce
that fit. Historical runs require their frozen sources and environment.

## Detector-Facing Variance Calibration

With the epoch-170 model frozen, ten TPE trials searched only `err_var_scale`
logarithmically over `[1e-3, 1]`, minimizing BLER at 10 dB on 63 batches of
four complete realizations. Cached observations, estimates, and unscaled
variances were reused. Sampler seed was 24350000, calibration seed 24330000;
unchanged TPE startup 10 means all ten suggestions were startup proposals.

Raw winner `0.6229862696239178` rounded to `0.623`. A separate paired cohort
of another 252 realizations, seed 24340000, gave BLER `0.029761904762` at
scale 1 and `0.028769841270` at 0.623. The strict-improvement rule selected
0.623. The small 0.099206 percentage-point decrease is not a statistical
significance claim. Channel estimates and NMSE are independent of this scale.

## Benchmark Run

Set `ASSET_ROOT` to the external root containing the supplied reference.
Its manifest retains the honest `single-reference` purpose, so evaluation
requires the explicit purpose option:

```bash
python -m sionna.bench.mimo_ofdm_chest.papers.fesl2024.run_benchmark \
  --device cpu --torch-threads 1 \
  --asset-root "$ASSET_ROOT" --artifact-id single-reference \
  --asset-purpose single-reference --err-var-scale 0.623 \
  --output-dir /path/to/new-results
```

Normal omitted-purpose loading requires canonical assets; smoke loading
requires smoke assets. The override does not relabel assets or permit training.
The constructor default is already `0.623`; `--err-var-scale 1` requests the
previous uncalibrated confidence. Smoke preparation and evaluation use the
standard separate `--smoke --prepare-assets` and `--smoke` calls with an
external asset root.

The final evaluation used `umi_3gpp_config`: 52 PRBs, 14 OFDM symbols, pilot
symbols `(2,11)`, a 4×4 BS array, four UEs, 63 batches of four realizations at
each of eleven SNRs, setup seed 42, and evaluation seed 2026. It took
2251.30 seconds on CPU with one Torch thread, including simulation, detection,
decoding, and writing. It is retained externally; official tracked-result
registration and publication remain maintainer follow-up.

| Eb/N0 (dB) | NMSE | BER | BLER |
| -: | -: | -: | -: |
| -10 | 0.098214718 | 0.236066504 | 0.963955026 |
| -7.5 | 0.057984972 | 0.188024132 | 0.897156085 |
| -5 | 0.034439126 | 0.141021356 | 0.768518519 |
| -2.5 | 0.021475698 | 0.097470266 | 0.620039683 |
| 0 | 0.013076603 | 0.058076843 | 0.413029101 |
| 2.5 | 0.008048369 | 0.030869737 | 0.240079365 |
| 5 | 0.004609918 | 0.013452315 | 0.117724868 |
| 7.5 | 0.003086725 | 0.005436972 | 0.047288360 |
| 10 | 0.001736450 | 0.002206113 | 0.016534392 |
| 12.5 | 0.001310283 | 0.001085860 | 0.010582011 |
| 15 | 0.000815907 | 0.000413332 | 0.002976190 |

The complete NMSE and confidence-limit arrays equal the scale-1 evaluation
bit for bit. BER improves at all eleven points; BLER improves at ten and ties
at 15 dB. Against twenty published benchmark entries, diffusion's NMSE ranks
4th–12th, nominally beating VAE/spatial GMM only at 10 dB. Spectral GMM, LMMSE,
and RKHS have lower NMSE throughout. Those comparisons use published aggregates only:
CUDA and CPU runs share the scientific configuration but are not asserted to
use bit-identical channels or support paired cross-estimator significance.

Implementation, assets, selection evidence, and verification are recorded in
[implementation_report.md](reports/implementation_report.md); the independent
verdict is in [final_report.md](reports/final_report.md).
