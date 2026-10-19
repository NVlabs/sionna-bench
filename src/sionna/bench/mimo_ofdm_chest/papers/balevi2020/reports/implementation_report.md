# Implementation Report: Balevi et al., 2020

Plan status: implemented after reviewer conditional approval in
`reports/plan_review.md`, human approval on 2026-06-24, and a later
human-requested revision from raw-grid fitting to pilot-observation fitting.

V2 calibrated-channel addendum, 2026-08-14: the complete study used Sionna
commit `f1620834`, calibrated-channel ancestor `ad1609ea1`, and UMi TR 38.901
v19.2. The 30-trial five-SNR search selected raw fitting parameters
`learning_rate=0.0020735628628380286`, `input_noise_scale=0.6103428098763377`,
and `num_steps=668`. On independent canonical-sized paired validation, the
rounded candidate scored weighted NMSE `0.006240309213763271`, versus
`0.0062212779057432685` for the incumbent; the incumbent fitting recipe was
retained. Ten-trial 10 dB BLER calibration selected raw scale
`0.4540521216492649`; independent paired validation selected rounded `0.454`
over incumbent `0.334` by `0.034722222222222224` versus
`0.03505291005291005`. The isolated canonical benchmark took 2 hours 11
minutes 19 seconds. Its five-SNR weighted NMSE is `0.0055496595`, and all
NMSE, BER, and BLER 95% intervals overlap the preserved worktree result at all
eleven SNRs. This addendum supersedes the tuning and calibration values below.

RNG-correction note, 2026-08-05: the 30-trial fitting study was repeated with
the shared post-construction probe reset and probe seed `314159`. The raw
winner did not retain its improvement after rounding to three significant
digits on the independent seed-`161803` holdout, so the existing
`num_steps=1597`, `learning_rate=0.000698`, and `input_noise_scale=1.70`
operating point remains selected. Detector-variance calibration and the
canonical benchmark were then repeated against that unchanged fitting point.
The corrected detector study produced raw `err_var_scale=0.38285407090201085`,
but rounded `0.383` did not improve over incumbent `0.334` on the independent
holdout, so `0.334` remains selected. The corrected canonical benchmark is
complete: its weighted NMSE over -10 dB through 10 dB is `5.963163e-3`, and
its 2-hour, 37-minute, 15-second wall-clock runtime includes concurrent Melgar
training on the same NVIDIA L40S.

Revision note, 2026-08-04: after reselecting the runtime fitting parameters
under Sionna 2.0.1, the detector-facing variance scale was recalibrated with
Optuna and an independent canonical-size holdout. The selected
three-significant-digit value `0.334` is now the estimator and runner default.
This supersedes the 2026-07-17 calibration of `0.38415771061572224`.

## Implemented Files

- `src/sionna_bench/mimo_ofdm_chest/papers/balevi2020/estimator.py`
- `src/sionna_bench/mimo_ofdm_chest/papers/balevi2020/run_benchmark.py`
- `src/sionna_bench/mimo_ofdm_chest/papers/balevi2020/tuning.py`
- `test/mimo_ofdm_chest/papers/balevi2020/test_estimator.py`
- `src/sionna_bench/mimo_ofdm_chest/papers/balevi2020/README.md`

`estimator.py` owns the network architecture and observation-specific online
optimization. `tuning.py` exposes the standard `run_tuning()` handle and
delegates to `evaluation.tune_channel_estimator()`. `run_benchmark.py` exposes
the standard `run_benchmark()` handle and delegates evaluation and persistence
to `evaluation.evaluate_channel_estimator()`. No persistent `asset.py` is
needed because the per-observation fitted parameters are not reusable assets.

## Implemented Estimator

`DeepPriorChannelEstimator` implements an LS pilot-grid adaptation of Balevi et
al.'s untrained deep-prior channel estimator.

The estimator:

1. extracts compact Sionna pilot observations through `BasePilotChannelEstimator`;
2. applies LS division at compact pilot positions;
3. groups active Kronecker pilot estimates into rectangular
   pilot-subcarrier/pilot-symbol images with `_PilotGroupMixin`;
4. stacks real and imaginary parts with the BS receive-antenna axis, matching
   the paper's `Ns = 2M` representation in spirit;
5. fits an untrained bias-free 1x1-convolutional deep decoder to each LS pilot
   grid by Adam minimization of a size-normalized Equation (16);
6. scatters denoised active pilot estimates back to Sionna's compact pilot
   layout;
7. lets `BasePilotChannelEstimator` apply the configured built-in or custom Sionna
   interpolator; the canonical run selects `lin_time_avg`;
8. returns a scaled LS-error-variance fallback because the paper does not
   define detector-facing Sionna error variances.

The bias-free hidden and output convolutions reproduce the paper's published
parameter counts exactly: 496 parameters for `k=8`, `M=1`, and 3776 parameters
for `k=16`, `M=64`. The hidden layers follow Equations (12)-(13): 1x1 linear
mapping, bilinear upsampling except in the last hidden layer, ReLU, and
batch-size-one channel normalization. The final 1x1 linear map follows
Equation (14).

The per-image DNNs retain independent parameters, normalization statistics,
loss contributions, and Adam-state slices. They use the same fixed random
input values, which do not create coupling because the fitted parameters and
optimizer states remain separate. The networks are executed together through
stacked module state and `torch.func.vmap`. A direct two-channel comparison at
250 optimizer steps reduced fitting time from approximately 15.4 seconds to
2.5 seconds. Batched kernels change floating-point operation ordering and can
therefore lead to different trajectories for this nonconvex fit, but no model
state is shared between observations.

## Deviations And Adaptations

- The original reviewed implementation fitted the DNN to the full received
  coded Sionna resource grid before pilot extraction. Smoke diagnostics showed
  that this destroyed the pilot observations and gave near-random performance.
  The human-approved revision applies LS first and fits only LS channel-
  observation grids instead. This reverses the paper's DNN-before-LS order and
  is a material benchmark adaptation.
- The paper assumes pilots on all subcarriers in its system model. The
  benchmark uses sparse Sionna Kronecker pilots. The implementation denoises
  only active pilot observations and then uses the configured Sionna
  interpolator. The canonical run selects `lin_time_avg`.
- The paper tunes early stopping using true-channel NMSE. The implementation
  exposes `num_steps` as a fixed runtime fitting budget.
- The benchmark pilot image shape is determined by the configured active pilot
  subcarriers and pilot OFDM symbols, not by the paper's `64 x 64` simulation
  grid. For multiple UEs, neighboring entries along the fitted frequency axis
  correspond to neighboring pilots in that UE's decimated comb, not adjacent
  physical subcarriers.
- The paper's repeated factor-two bilinear upsampling is exact when dimensions
  permit it. For other configured pilot-grid dimensions, intermediate sizes
  are rounded upward deterministically.
- The fixed uniform input is registered as a buffer and reused across the
  independent per-observation fits. Equation (16)'s squared Euclidean norm is
  normalized by each grid's element count; this preserves its minimizers while
  making the learning-rate interpretation less dependent on grid size.
- No persistent asset is used. Balevi et al.'s method is untrained in the
  offline sense and fits weights at estimator runtime.

## Verification

Focused tests:

```bash
python -m pytest -q test/mimo_ofdm_chest/papers/balevi2020/test_estimator.py
```

Result: passed, `21 passed`.

The tests cover:

- deterministic DNN upsampling output shape;
- exact paper parameter counts for the SISO `k=8` and massive-MIMO `k=16`
  architectures;
- batch-size-one hidden-channel normalization;
- deterministic, registered, global-RNG-neutral uniform fixed input;
- descent of the normalized Equation (16) objective;
- `num_steps=0` LS-equivalence against Sionna `LSChannelEstimator` with
  nearest-neighbor, linear, and linear-with-time-averaging interpolation;
- custom `BaseChannelInterpolator` precedence over `interpolation_type`;
- identity-denoiser equivalence of the fitted LS-first pilot path;
- fitted-path output shapes, complex/real dtypes, configured CPU placement,
  finite values, and non-negative error variances;
- explicit rejection of incomplete frequency-time pilot rectangles;
- finite, nonzero one-step fitted inference on 48-, 72-, and 96-subcarrier
  grids with one, three, and four UEs; two, four, and twelve BS antennas; and
  one, two, and three pilot-bearing OFDM symbols, respectively, without a
  canonical-performance threshold; and
- batched-versus-separate optimization equivalence over multiple image and
  channel dimensions, including isolation when only one target is perturbed.

The online Adam loop explicitly enables autograd so the estimator continues to
work when invoked by an otherwise inference-only caller under
`torch.no_grad()`.

Compilation checks use `torch.compile(..., backend="eager")`. The tensor-only
deep decoder compiles with `fullgraph=True`, produces one graph with no graph
breaks in the diagnostic run, and matches eager output. The complete estimator
runs with `fullgraph=False` and matches eager output while
`_fit_real_channel_images()` remains explicitly compiler-disabled. Dynamo
reported six graphs and five breaks, all attributed to skipping that fitting
region; these counts are diagnostic and intentionally not asserted in tests.

Compile check:

```bash
python -m py_compile src/sionna_bench/mimo_ofdm_chest/papers/balevi2020/estimator.py src/sionna_bench/mimo_ofdm_chest/papers/balevi2020/run_benchmark.py src/sionna_bench/mimo_ofdm_chest/papers/balevi2020/tuning.py
```

Result: passed.

Smoke command:

```bash
python -u -m sionna_bench.mimo_ofdm_chest.papers.balevi2020.run_benchmark --smoke --hidden-channels 32 --num-steps 1063 --learning-rate 0.0022718381741860987 --input-noise-scale 2.0156915698697073
```

Result: executed and wrote
`outputs/balevi2020/smoke/3gpp-umi/DeepPriorChannelEstimator.npz`.

Smoke metrics at 5 dB:

```text
BER  = 7.860323e-03
BLER = 1.250000e-01
NMSE = 3.309466e-03
```

The raw-grid diagnostic results from the earlier implementation attempt were
removed from the smoke output directory and manifest to avoid mixing
incompatible experiments.

## Hyperparameter Selection And Canonical Benchmark

The runtime fitting parameters were reselected under Sionna 2.0.1 with 30
Optuna TPE trials over `learning_rate`, `input_noise_scale`, and `num_steps`,
while fixing the paper architecture width to 32 hidden channels and six hidden
layers. Each trial used weighted NMSE over the nine SNR points from -10 dB
through 10 dB, 10 Monte Carlo batches per SNR, batch size 4, and weights
`10**(Eb/N0/10)`, for 360 independently sampled channels per trial. Sampler
seed `123` and probe seed `314159` were fixed. The shared RNG reset was applied
after estimator construction so hyperparameter-dependent initialization could
not change the probe channels. The canonical operating point was evaluated as
trial zero and scored `5.952883959875279e-3`.

Trial 22 selected raw values `learning_rate=0.001083708466655142`,
`input_noise_scale=0.3852956689041489`, and `num_steps=1587`, with objective
`5.85286041544616e-3`. On an independent seed-`161803` holdout with 63 batches
per SNR, the incumbent, raw winner, and three-significant-digit candidate
scored `5.817681719894735e-3`, `5.816172850453034e-3`, and
`5.844327044435728e-3`, respectively. The raw value was diagnostic only;
selection compared the incumbent with the rounded candidate, so the existing
`learning_rate=0.000698`, `input_noise_scale=1.70`, and `num_steps=1597`
operating point was retained. The corrected study took 6 hours 30 minutes and
41 seconds, and the three-candidate holdout took 7 hours 3 minutes and 13
seconds. The selected fitting parameters remain:

```text
hidden_channels=32
num_hidden_layers=6
num_steps=1597
learning_rate=0.000698
input_noise_scale=1.70
interpolation_type="lin_time_avg"
```

The tracked pre-migration canonical benchmark used all 11 SNR points, 63 Monte
Carlo batches per SNR, batch size 4, and the superseded fitting parameters. Its
NMSE curve is retained below as the regression reference for the Sionna 2.0.1
rerun:

```text
Eb/N0 [dB]  -10      -7.5     -5       -2.5     0        2.5
NMSE         1.1743e-1 6.1830e-2 3.3007e-2 1.8251e-2 1.1044e-2 7.2047e-3

Eb/N0 [dB]   5        7.5      10       12.5     15
NMSE         5.4561e-3 4.4767e-3 4.2098e-3 3.6602e-3 3.5654e-3
```

This reference was superseded by the Sionna 2.0.1 parameter selection. The
final canonical run uses `num_steps=1597`, `learning_rate=0.000698`,
`input_noise_scale=1.70`, and `err_var_scale=0.334`. Its weighted NMSE over the
tuning SNR region is `5.963163e-3`, compared with `6.110896e-3` for the
pre-RNG-correction Sionna 2.0.1 run and `6.682170e-3` for the pre-migration
reference. All three metric curves have overlapping 95% intervals with the
pre-correction Sionna 2.0.1 run at every SNR. Relative to the older reference,
NMSE is worse through 0 dB and better from 2.5 dB onward; at 15 dB it improves
from `3.565445e-3` to `2.447565e-3`. BER and BLER improve at every SNR. The
full run took 2 hours 37 minutes 15 seconds on one NVIDIA L40S while a Melgar
training study shared the GPU.

The canonical result was regenerated with bootstrap confidence intervals and
stored at
`outputs/balevi2020/3gpp-umi/DeepPriorChannelEstimator.npz`.

The paper does not define detector-facing Sionna error variances. After the
RNG-corrected fitting selection, a fresh Sionna 2.0.1 cached-estimate study
selected raw `0.38285407090201085`, with objective
`0.22256858020943995`, compared with `0.22286092350887846` for incumbent
`0.334`. On the independent seed-`314159` canonical-size holdout, incumbent
`0.334`, the raw winner, and rounded `0.383` scored `0.21055273600140045`,
`0.21070107626868895`, and `0.21070088304541565`, respectively. The incumbent
therefore remains selected. Regression tests verify that the scale changes
only `err_var`, not `h_hat`. The corrected canonical benchmark reports final
mean metrics NMSE `2.678719e-2`, BER `6.384782e-2`, and BLER `3.384139e-1`.
