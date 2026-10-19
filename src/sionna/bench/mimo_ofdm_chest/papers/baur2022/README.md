# Implementation of the Estimators from Baur et al., 2022

Publication: M. Baur, B. Fesl, M. Koller, and W. Utschick, "Variational Autoencoder Leveraged MMSE Channel Estimation," 2022 56th Asilomar Conference on Signals, Systems, and Computers.

## Abstract

Baur et al., 2022 propose to learn conditional channel moments with a variational autoencoder and then use these moments in an LMMSE-style channel estimator. The paper assumes a SIMO uplink observation after pilot decorrelation, `y = h + n`. This implementation adapts that estimator to Sionna's OFDM channel-estimation benchmark by applying the VAE estimator to LS channel estimates at nonzero pilot resource elements.

The VAE operates on a DFT transform of each receive-array channel vector. Its decoder returns a transformed-domain conditional mean and either a diagonal covariance, as in the paper, or a full real-stacked covariance. The estimator then applies the paper's conditional LMMSE update and scatters the denoised pilot estimates back into Sionna's pilot-domain estimator interface. Full-grid channel estimates are produced by the configured Sionna interpolator; the default is `lin_time_avg`.

## Implemented Classes

`VariationalAutoEncoder` in `estimator.py` implements the real-valued convolutional VAE used by the estimator. It follows the paper's Section III-D encoder with three Conv-BN-ReLU blocks, kernel size 7, stride 2, and channel progression 1 -> 8 -> 32 -> 128. The decoder reverses the channel progression with transposed convolutions. It uses kernel size 4, stride 2, and padding 1 to double every intermediate length exactly; the paper describes the decoder only as analogous and symmetric and does not state its transposed-convolution boundary convention. The decoder outputs a real-stacked complex mean and either a positive diagonal covariance or a positive-definite full covariance through a Cholesky factor.

`VAEChannelEstimator` in `estimator.py` is a Sionna
`BasePilotChannelEstimator`. It uses Sionna's internal `_PilotGroupMixin` to
batch the receive-array vectors associated with active pilot resource
elements. Its constructor receives prepared model states and performs no asset
lookup or training. In `asset.py`, `train_model()` explicitly prepares assets
under a caller-supplied external root, while `load_model()` strictly loads
compatible optional package-local or external checkpoints without training or
writing. Public distributions omit checkpoint payloads.

Main methods:

- `estimate_at_pilot_locations()`: computes Sionna LS pilot estimates, selects the SNR-specific VAE checkpoint, denoises each receive-array pilot vector, and returns compact pilot-domain estimates.
- `_estimate_group()`: applies the unitary receive-DFT, evaluates `z = mu(x_tilde)` as in Equation (12), applies the LMMSE update from Equation (10), transforms back to the antenna domain, and returns antenna-domain marginal error variances.
- `_build_unitary_dft()`, `_dft()`, and `_idft()`: implement the benchmark's DFT-domain adaptation of Equation (13).

Implemented methods:

- `VAEChannelEstimator`: Section III-C variant 2, called `VAE-noisy` in the paper. The encoder is trained on noisy transformed observations, while the decoder likelihood target is the clean transformed channel.
- The paper's `VAE-real` method remains implemented and tested. Its encoder and decoder likelihood target are both noisy transformed observations, with the observation-noise variance added in the likelihood so the decoder covariance remains a channel covariance.

Only `VAEChannelEstimator`, using the paper's `VAE-noisy` training method, is
included in the canonical leaderboard. This is a deliberate benchmark-scope
adaptation: the other persistently trained neural estimators in this repository
use simulator-provided clean channels as training labels, so the clean-label
VAE method is the directly comparable Baur representative. `VAE-real` remains
available for studying the paper's noisy-label setting, but it is not presented
as a separate leaderboard estimator.

The diagonal covariance parameterization remains available through `covariance_mode="diagonal"`, but the committed canonical benchmark results use `covariance_mode="full"`.

The diagonal mode implements the paper's proper-complex Gaussian likelihood from Equation (6) directly. The full mode is a benchmark extension: it uses a full covariance over the `2M` real-stacked transform coefficients, a real multivariate-Gaussian likelihood, and a separate Cholesky head driven by the decoder's initial linear features. For `VAE-real`, the complex observation variance is added as `noise_variance/2` to each real component, which is equivalent to adding the complex noise variance in the paper's diagonal formulation.

`run_tuning()` in `tuning.py` uses the shared Optuna evaluation path either to
tune latent dimension, learning rate, and free bits while preparing
trial-specific states through `asset.py`, or to calibrate the detector-facing
variance scale with the frozen canonical model bank.

## Modifications and Omissions

The paper assumes a ULA and uses the Toeplitz/circulant approximation to justify DFT-domain diagonal channel covariances. The benchmark uses a 4x4 planar base-station array. This implementation uses a unitary 1D DFT over Sionna's flattened `num_rx_ant` axis. This preserves the estimator algebra and white-noise variance under the transform, but it is not a reproduction of the paper's ULA curves.

The paper does not address OFDM subcarrier or OFDM-symbol interpolation. The implementation therefore uses the pilot-domain Sionna estimator contract and explicit `lin_time_avg` interpolation after VAE denoising.

`VAE-genie` is not implemented because it requires the true channel at estimator runtime. That would be an oracle path rather than a deployable channel estimator under the current benchmark contract.

The paper omits several training details, including latent dimension and
free-bits threshold. These are explicit asset-preparation arguments recorded
in each manifest's `generation_recipe`. The calibrated study selected latent
dimension 16, free bits `0.875`, learning rate `0.000230`, and optimizer batch
size 512. Every SNR-specific model used 6,400 training and 256 validation
channels, at most 200 epochs, patience 20, and a relative minimum validation
improvement of `1e-4`, with restoration of the best checkpoint.

Section III-D resamples additive noise every training epoch. The reusable data
bank stores clean transformed channels and pilot powers, while training adds
deterministic SNR- and epoch-specific noise. It therefore preserves per-epoch
noise resampling without regenerating the underlying channels for every
Optuna trial. The channel/data seed is 42 and the independent validation seed
is 100045.

The paper does not define Sionna detector-facing error variances. In diagonal covariance mode, this implementation converts the transformed-domain posterior diagonal variance to antenna-domain marginal variance via `diag(F^H diag(v) F)`. In full covariance mode, it applies the corresponding real-stacked inverse-DFT covariance transform and returns the antenna-domain complex marginal variance. The posterior mean is independent of `err_var_scale`; the scale only calibrates the confidence passed to Sionna's detector.

A ten-trial log-scale search over `[1e-3, 1]` at 10 dB proposed raw scale
`0.4540521216`, rounded to `0.454`. On 63 paired batches of four channels, the
candidate and the existing `0.5405613145` scale both produced BLER
`0.0211640212`. The strict-improvement rule therefore retained the existing
scale. This calibration is detector-facing only; channel estimates and NMSE
are unchanged by the scale.

### Compilation

VAE training, checkpoint discovery, validation, and loading happen in
`asset.py` before estimator construction and are not reachable from online
inference. Pilot groups are processed as one tensor batch. With one mixed-SNR
state, the complete estimator is Tier A and compiles as one full graph. The
canonical per-SNR configuration is Tier B: selecting one prepared VAE from the
Python `ModuleList` is explicitly compiler-disabled, while the selected
encoder/decoder, posterior channel estimation, and Sionna interpolation remain
compiled. Eager and `torch.compile(..., fullgraph=False)` outputs agree for the
canonical full-covariance path.

## Asset Generation

The accepted study used calibrated Sionna commit
`f16208347246d8d17ec69628dfb856b19287b80c`, including calibration commit
`ad1609ea1`, and explicit UMi TR 38.901 version 19.2. It ran on an NVIDIA L40S
with driver `580.126.09` and AMD EPYC 7R13 using Python 3.12.3, PyTorch
`2.11.0+cu126`, CUDA 12.6, and cuDNN 91002. The dependency snapshot SHA-256 is
`b981919b35253baa1f4cbc244763e09292be6a503d42480d3f9003e397884082`.
Dataset/model seed is 42, validation seed 100045, Optuna sampler seed 123,
NMSE probe seed 4242, paired-validation seed 314159, variance sampler seed
456, variance probe seed 271828, detector-validation seed 161803, and
canonical evaluation seed 2026.

The fixed bank is generated once under an external asset root with
`generate_training_data()` from `asset.py`: 400 batches of 16 training
channels, 16 batches of 16 validation channels, all eleven canonical SNRs,
and seed 42. Its manifest SHA-256 is
`35b7728ac44e738ba696df25fdbf5e7f8b69887934268c185c7c8caeee1f9718`.
Training and validation channels are unique within each split and have no
exact overlap.

The algorithm study produced ten complete trials. Each trial trained fresh models
at `{-10,-5,0,5,10}` dB and minimized weighted NMSE on independent probe
channels. It searched latent dimension `{4,8,16}`, learning rate
`[1e-5,1e-3]`, free bits `[0,1]`, and optimizer batch size
`{256,512,1024,2048}`. The raw winner was latent dimension 16, learning rate
`0.0002304632`, free bits `0.8754568`, and batch size 512. The rounded recipe
`(16, 0.000230, 0.875, 512)` achieved paired weighted NMSE
`0.0044366964`, versus `0.0045868521` for the freshly retrained prior recipe,
and was selected.

The accepted database records ten complete and two failed trial numbers. Trial
9 failed numerically and remains excluded from selection; trial 10 was an
interrupted duplicate recovery attempt. Trial 11 supplied the verified tenth
complete recipe. The initial search segment took 100:05:19 and the accepted
replacement 17:38:05. Paired validation took 14:26:01. Two earlier invalid
pipeline attempts, one unseeded and one using an incorrect negative-loss early
stop, were quarantined and are not part of the accepted study, assets, or
runtime total.

Generate the final eleven-SNR bank with:

```bash
python -m sionna.bench.mimo_ofdm_chest.papers.baur2022.run_benchmark \
  --prepare-assets --asset-root "$FINAL_ASSET_ROOT" \
  --training-data-manifest "$TRAINING_DATA_MANIFEST" \
  --train-batches 400 --train-batch-size 16 \
  --validation-batches 16 --validation-batch-size 16 \
  --epochs 200 --optimizer-batch-size 512 \
  --learning-rate 0.000230 --free-bits 0.875 --latent-dim 16 \
  --patience-epochs 20 --relative-min-delta 1e-4
```

The eleven model fits consumed 111,196.71 seconds (30.89 hours). Every one of
the 517 state tensors is finite and strictly loadable; the aggregate final-bank
audit has SHA-256
`17c4f440bee3ac63216e71f9bf96bea80e43b2085229b5359bb5845d6c0d3edf`.

## Benchmark Run

Detector-facing variance calibration and its paired validation are reproduced
with:

```bash
python -m sionna.bench.mimo_ofdm_chest.papers.baur2022.tuning \
  --study variance --asset-root "$FINAL_ASSET_ROOT"
python -m sionna.bench.mimo_ofdm_chest.papers.baur2022.tuning \
  --study variance-validation --asset-root "$FINAL_ASSET_ROOT"
```

Training-parameter tuning is invoked with `--study training`, followed by
`--study training-validation`; both commands receive the fixed training-data
manifest and an external trial-asset root.

Smoke commands used for integration first prepare an external asset and then
strictly load it for evaluation:

```bash
python -m sionna.bench.mimo_ofdm_chest.papers.baur2022.run_benchmark \
  --smoke --prepare-assets --asset-root /tmp/baur-assets
python -m sionna.bench.mimo_ofdm_chest.papers.baur2022.run_benchmark \
  --smoke --asset-root /tmp/baur-assets
```

The temporary asset tests independently train and strictly reload both the
`VAE-noisy` and `VAE-real` paths. Only the selected `VAE-noisy` method is used
by the smoke benchmark and canonical leaderboard.

After the selected checkpoints have been generated under an external asset
root, the canonical benchmark loads them without training or writing assets:

```bash
python -m sionna.bench.mimo_ofdm_chest.papers.baur2022.run_benchmark \
  --method noisy --asset-root "$FINAL_ASSET_ROOT" \
  --err-var-scale 0.5405613145449313
```

The canonical benchmark uses 63 batches of four channels at every SNR and
evaluation seed 2026. Its NMSE curve is:

| -10 dB | -7.5 dB | -5 dB | -2.5 dB | 0 dB | 2.5 dB | 5 dB | 7.5 dB | 10 dB | 12.5 dB | 15 dB |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `8.317239e-02` | `5.182288e-02` | `3.189962e-02` | `1.933015e-02` | `1.198339e-02` | `7.313015e-03` | `4.461935e-03` | `2.774516e-03` | `1.757453e-03` | `1.152724e-03` | `7.923961e-04` |

The five-SNR weighted NMSE is `0.0042578149`, 2.63% above the comparison
worktree result. All pointwise NMSE confidence intervals overlap, while BER
and BLER improve across the full curve.
