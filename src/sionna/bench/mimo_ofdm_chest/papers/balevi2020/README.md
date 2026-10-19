# Implementation of the Estimator from Balevi et al., 2020

Publication: E. Balevi, A. Doshi, and J. G. Andrews, "Massive MIMO Channel Estimation With an Untrained Deep Neural Network," IEEE Transactions on Wireless Communications, vol. 19, no. 3, pp. 2079-2090, March 2020.

## Abstract

Balevi et al., 2020 propose an untrained deep-prior channel estimator for
massive MIMO-OFDM. The method fits a small neural network at inference time,
uses the fitted output as a denoised observation, and then applies an LS pilot
operation.

This implementation adapts the idea to the repository's Sionna uplink MU-MIMO
benchmark by first forming LS channel observations at active pilots and then
fitting the untrained network to their frequency-time grids. The paper instead
fits the complete received coherence grid before applying LS. The paper also
does not define interpolation for Sionna's sparse Kronecker pilot pattern. The
estimator therefore exposes Sionna's standard `interpolation_type` and custom
`interpolator` interface. The canonical run uses `lin_time_avg`.

## Implemented Classes

`DeepPriorChannelEstimator` in `estimator.py` is a Sionna
`BasePilotChannelEstimator`. It uses Sionna's internal `_PilotGroupMixin` to
organize active pilot resource elements without exposing pilot grouping as
part of the estimator's public API.

Main methods:

- `estimate_at_pilot_locations()`: computes compact LS pilot observations,
  denoises the active pilot observations, and returns compact pilot estimates
  for Sionna interpolation.
- `_fit_ls_pilot_denoiser()`: groups active LS channel observations into
  rectangular frequency-time grids and fits one untrained denoiser per
  batch/rx/stream image.
- `_fit_real_channel_images()`: performs the independent runtime Adam
  optimizations analogous to Equation (16), but executes all networks in a
  batch with separate parameters and optimizer states.

`_DeepPriorDenoiser` implements the bias-free 1x1-convolutional deep decoder
from Equations (11)-(14), with deterministic bilinear upsampling sizes. The
paper's published parameter counts are reproduced exactly: 496 parameters for
`k=8`, `M=1`, and 3776 parameters for `k=16`, `M=64`.

The independent per-image networks are vectorized with `torch.func`. This does
not share weights, normalization statistics, or Adam-state values between
channel realizations. The fits use the same seeded uniform fixed-input values,
which do not couple their independently optimized parameters. Vectorization
changes the floating-point execution order and can therefore produce a
slightly different trajectory for the nonconvex fit, but it leaves the
underlying collection of optimization problems unchanged.

Benchmark outputs are named `DeepPriorChannelEstimator`.

The paper-local implementation follows the shared module contract:

- `estimator.py` contains the network architecture and observation-specific
  online fitting;
- `tuning.py` exposes `run_tuning()` and delegates Optuna evaluation to
  `evaluation.tune_channel_estimator()`; and
- `run_benchmark.py` exposes `run_benchmark()` and delegates canonical metric
  evaluation and result writing to `evaluation.evaluate_channel_estimator()`.

No persistent asset module is used. The fitted network parameters are specific
to the current observations and are not reusable model weights.

## Supported System Geometry

For every transmit stream, active pilots must form a complete rectangular grid
over their distinct OFDM subcarriers and OFDM symbols. All streams must have
the same active pilot-subcarrier count and pilot-symbol count so their
observation-specific networks can be processed together. Subject to these
conditions, the estimator derives its network image dimensions from the
resource grid and supports changed FFT sizes, OFDM-symbol counts, UE counts,
and BS receive-antenna counts. Incomplete rectangles and inconsistent stream
dimensions are rejected during construction with explicit messages.

The tensor-only `_DeepPriorDenoiser.forward()` compiles as one full graph and
agrees with eager execution. The complete estimator is Tier C: dynamic network
construction, RNG isolation, `torch.func` state stacking, autograd, and Adam
updates are the observation-specific estimation algorithm and remain in an
explicitly compiler-disabled eager region. `torch.compile(fullgraph=False)` can
execute the complete call around that boundary with eager-equivalent outputs.

## Modifications and Omissions

The paper's received-tensor fitting objective is not applied directly to the
full coded Sionna resource grid. A raw-grid implementation was tested and found
to destroy the pilot observations because the benchmark grid contains many
data-bearing resource elements with random QAM symbols. The current
implementation therefore applies LS first and fits the untrained network to LS
channel-observation grids. This reverses the paper's denoising-before-LS order
and is a material, human-approved benchmark adaptation rather than an exact
reproduction of the paper's observation model.

The paper assumes pilot observations on all subcarriers and does not define
interpolation for Sionna's sparse Kronecker pilot pattern. The implementation
uses Sionna's internal `_PilotGroupMixin` to group active Kronecker pilot
resource elements and then delegates full-grid completion to the configured
Sionna interpolator through `BasePilotChannelEstimator`. The canonical run
selects `lin_time_avg`.

The paper's simulations use `64 x 64` OFDM grids. The benchmark pilot images
have the active pilot-subcarrier and pilot-symbol dimensions of the configured
Sionna resource grid. The implementation keeps the paper's bilinear-upsampling
idea but resizes deterministically to this LS pilot-grid shape. Exact factor-two
upsampling is retained when dimensions permit it; intermediate dimensions are
rounded upward otherwise.

The fixed input `Z0` is sampled uniformly once from
`[0, input_noise_scale)`, registered as a buffer, and held fixed throughout
every optimization. Hidden-layer normalization matches training-mode
batch normalization with batch size one: each hidden channel is normalized
over its frequency-time grid and then receives a learned scale and offset.
The implementation minimizes mean squared error per fitted grid and sums those
independent losses. This is a size-normalized form of Equation (16) with the
same minimizers as its squared Euclidean norm.

The paper selects early-stopping epochs using true-channel NMSE. That is not
available to a deployable Sionna estimator, so the implementation exposes a
fixed `num_steps` runtime fitting budget. The runtime fitting parameters were
selected with Optuna on validation probes rather than by using final benchmark
metrics.

The paper does not define detector-facing Sionna error variances. The
implementation returns an `err_var_scale`-scaled LS observation-variance
fallback before interpolation. The calibrated default is documented below.

Pilot-contamination/inpainting experiments from the paper are not implemented
because the current benchmark does not generate neighboring-cell pilot reuse or
explicit interference patches.

Paper MMSE and genie-aided MMSE baselines are not implemented because they are
reference baselines, not Balevi estimator variants, and exact reproduction
would require separate covariance/oracle-statistics assets.

## Hyperparameter Tuning And Benchmark

The final v2 study used calibrated Sionna source commit
`f16208347246d8d17ec69628dfb856b19287b80c`, including the channel-model
correction in `ad1609ea1`, and the UMi implementation from TR 38.901 v19.2.
The 30-trial seeded TPE study minimized weighted NMSE at
`[-10, -5, 0, 5, 10]` dB. It used three Monte Carlo batches of four channels
per SNR and weights `10**(Eb/N0/10)`, for 60 SNR/channel pairs per trial. All
trials reused the same observations generated with probe seed `4242`; sampler
seed `123` controlled Optuna. The study fixed `hidden_channels=32`,
`num_hidden_layers=6`, seed `42`, and `interpolation_type="lin_time_avg"`.
It ran 30 requested trials (30 completed, zero failed, and zero pruned) and
optimized:

```text
learning_rate:     log-uniform from 1e-4 to 3e-2
input_noise_scale: log-uniform from 0.1 to 3.0
num_steps:         log-uniform integer from 100 to 2000
```

Trial 6 selected raw `learning_rate=0.0020735628628380286`,
`input_noise_scale=0.6103428098763377`, and `num_steps=668`, with weighted
NMSE `0.005091621349032229`. Continuous values were rounded to three
significant digits before selection. Independent paired validation used seed
`314159`, 63 batches of four channels at every tuning SNR, and the same
channels for both candidates. The incumbent scored `0.0062212779057432685`;
the rounded winner `(0.00207, 0.61, 668)` scored
`0.006240309213763271`. The incumbent was therefore retained:

```text
hidden_channels=32
num_hidden_layers=6
num_steps=1597
learning_rate=0.000698
input_noise_scale=1.70
interpolation_type="lin_time_avg"
```

The search and paired validation took 47 minutes 22 seconds and 1 hour 25
minutes 56 seconds, respectively, on one NVIDIA L40S. They are reproduced by:

```bash
python -m sionna.bench.mimo_ofdm_chest.papers.balevi2020.tuning \
  --study optimization \
  --output-dir tuning/mimo_ofdm_chest/balevi2020/optimization_v2
python -m sionna.bench.mimo_ofdm_chest.papers.balevi2020.tuning \
  --study optimization-validation \
  --output-dir tuning/mimo_ofdm_chest/balevi2020/optimization_v2
```

### Detector-Facing Error Variance

After freezing fitting behavior, ten seeded TPE trials calibrated
`err_var_scale` over the logarithmic range `[1e-3, 1]` against codeword BLER
at 10 dB. The study used 63 batches of four channels and probe seed `271828`.
Sampler seed `456` controlled Optuna. The calibration ran 10 requested trials:
10 completed, zero failed, and zero were pruned.
Each observation was fitted once; channel estimates and unscaled variances
were cached across scalar candidates. The raw winner was
`0.4540521216492649`, with BLER `0.027777777777777776`.

Independent paired validation used seed `161803` and another 63 batches of
four channels. Incumbent `0.334` scored `0.03505291005291005`, while rounded
`0.454` scored `0.034722222222222224`. The strict selection rule therefore
accepted `err_var_scale=0.454`. The search and validation took 11 minutes 51
seconds and 12 minutes 24 seconds. Scaling changes detector confidence and
BER/BLER but cannot change `h_hat` or NMSE.

The calibration can be reproduced with:

```bash
python -m sionna.bench.mimo_ofdm_chest.papers.balevi2020.tuning \
  --study variance \
  --algorithm-output-dir tuning/mimo_ofdm_chest/balevi2020/optimization_v2 \
  --output-dir tuning/mimo_ofdm_chest/balevi2020/variance_scale_v2
python -m sionna.bench.mimo_ofdm_chest.papers.balevi2020.tuning \
  --study variance-validation \
  --algorithm-output-dir tuning/mimo_ofdm_chest/balevi2020/optimization_v2 \
  --output-dir tuning/mimo_ofdm_chest/balevi2020/variance_scale_v2
```

The canonical benchmark uses 63 Monte Carlo batches of size four at each of
the eleven SNR points and is reproduced with:

```bash
python -m sionna.bench.mimo_ofdm_chest.papers.balevi2020.run_benchmark
```

The selected operating point is `hidden_channels=32`, `num_hidden_layers=6`,
`num_steps=1597`, `learning_rate=0.000698`, `input_noise_scale=1.70`,
`interpolation_type="lin_time_avg"`, and `err_var_scale=0.454`. The isolated
run took 2 hours 11 minutes 19 seconds on one NVIDIA L40S. Its NMSE curve is:

```text
Eb/N0 [dB]  -10       -7.5      -5        -2.5      0         2.5
NMSE         1.3307e-1 7.0263e-2 3.7201e-2 1.9862e-2 1.1276e-2 6.8445e-3

Eb/N0 [dB]   5         7.5       10        12.5      15
NMSE         5.0895e-3 3.6920e-3 2.8465e-3 2.4686e-3 2.5658e-3
```

The five-point tuning-grid weighted NMSE is `0.0055496595`, compared with
`0.0054665398` for the preserved worktree result. All NMSE, BER, and BLER 95%
confidence intervals overlap at every SNR, so the 1.52% weighted-NMSE increase
is statistically consistent. The final means are NMSE `0.02683422`, BER
`0.06391128`, and BLER `0.33946609`. The complete v2 pipeline took 4 hours 48
minutes 52 seconds, excluding only preflight and dependency capture.

The retained preflight records Sionna Bench commit
`5012974748d4b8aa6426e563d8fed1380eba3abc` and tracked-diff SHA-256
`813d526cd41372f19a4713eb9d276a6c9dd9fe8fcbfcf9b186c0c9cb13b610ec`.
The environment was Python 3.12.3, PyTorch `2.11.0+cu126`, CUDA 12.6,
driver `580.126.09`, NumPy 2.4.4, SciPy 1.17.1, and Optuna 4.9.0 on an
NVIDIA L40S and AMD EPYC 7R13. The retained `pip-freeze.txt` has SHA-256
`b981919b35253baa1f4cbc244763e09292be6a503d42480d3f9003e397884082`.
Canonical evaluation used seed `2026`.
