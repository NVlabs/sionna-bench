# Optuna Study Protocol

This document defines the protocol used for hyperparameter tuning of the
canonical 3GPP UMi benchmark. It is normative for the retained studies and
their selected parameters.

The purpose of the protocol is to give every method a transparent and serious
but bounded optimization effort, remove arbitrary differences in channel
counts, and keep final benchmark outputs outside the parameter-selection loop.

## Preliminary Feasibility Probe

Before freezing and starting a canonical algorithmic study, the implementer
performs a small deterministic feasibility probe for every method requiring
nontrivial hyperparameter tuning.

Unless the reviewed plan specifies a more representative operating point, the
probe evaluates a small number of deliberately selected parameter
configurations at 10 dB. Its purposes are to:

- identify invalid, unstable, or clearly unproductive regions of the proposed
  search space;
- detect gross numerical or performance problems before committing substantial
  resources;
- measure the dominant inference, fitting, training, and data-generation
  costs; and
- estimate the expected full-study runtime, conservative
  no-early-termination upper bound, and storage requirements.

The feasibility probe uses a dedicated development seed and a bounded sample
budget. Its channels, observations, fitted models, and trained weights are
development artifacts. They are not reused for canonical training,
algorithmic probes, paired validation, detector calibration, or canonical
benchmark evaluation.

For learned or fitted methods, runtime estimates measure representative
full-sized operations, such as optimizer-batch throughput, epoch throughput
extrapolated from representative batches, or one representative full-quality
fit. A reduced smoke model is not the sole basis for extrapolation when its
scaling behavior is unknown.

The feasibility probe may refine the proposed search space, but it does not
select canonical parameters. The implementer records the tested
configurations, preliminary seed, sample budget, measured behavior, hardware,
runtime measurements, extrapolation method, proposed final search space, and
estimated full-study cost in the implementation plan or report.

The implementer updates `reports/plan.md` with the final study design and stops
for explicit human approval before generating canonical study data or starting
the full study. A material change to the reviewed study design also requires
the corresponding plan review to be updated.

## Protocol Freeze

The canonical study protocol and its implementation are frozen after the
preliminary feasibility gate and before canonical training or validation data,
probe observations, paired-validation observations, or study trials are
generated. Once frozen, the study design does not change in response to
canonical study outcomes. The frozen record includes:

- this protocol and the exact tuning and asset-generation source code in a
  common commit;
- fixed training, validation, model-initialization, Optuna sampler,
  algorithm-probe, paired-validation, detector-calibration, and
  canonical-evaluation seeds;
- the source commit and complete Python dependency versions; and
- the hardware information captured by every run, including GPU, CPU, CUDA,
  driver, and PyTorch versions.

Once a fixed dataset has been generated, the SHA-256 hashes of its manifest
and every payload file are recorded before the dataset is used in a study. A
hash change invalidates the study. The generated study summary records the
source commit, dependency snapshot, dataset hashes, seeds, and actual hardware
alongside its timing and objective evidence.

## Shared Algorithm-Study Protocol

Unless a row in the study table states otherwise, every algorithmic study
uses the following rules.

- Each study minimizes weighted NMSE at five representative Eb/N0 points:
  -10, -5, 0, 5, and 10 dB.
- The NMSE at Eb/N0 `s` is weighted by `10**(s/10)` and normalized by the sum
  of the weights. This approximately balances estimators whose NMSE decreases
  in inverse proportion to linear SNR while penalizing high-SNR error floors.
- Each trial uses three Monte Carlo batches of four channels at every SNR and
  therefore evaluates 60 SNR/channel pairs. Consecutive deterministic channel
  draws are consumed across SNRs, and every trial reuses the same probe
  observations.
- Probe observations are generated and cached once per study. Estimator
  construction, fitting, and inference are repeated when required by the
  candidate parameters.
- Continuous and mixed search spaces use a seeded Optuna TPE sampler. The four
  GMM component counts use an exhaustive grid.
- Canonical benchmark outputs are excluded from optimization. Training data,
  validation data, Optuna probes, paired validation, and canonical evaluation
  use distinct deterministic random streams.
- Every candidate receives the specified dataset size, epoch limit, solver
  iterations, and fitting quality.
- A failed or non-finite trial is recorded as failed and does not count toward
  the requested number of completed trials.

## Fixed Statistical And Training Data

Expensive channel generation is performed once before a study. The resulting
clean channel data are immutable and reused by every candidate.

| Asset or model family | Required channel data |
| :- | :- |
| Built-in and Edfors covariance statistics | 64,000 complete system-channel realizations per covariance asset. |
| Spatial and spectral GMMs | 6,400 complete system-channel realizations. Both component-count studies use the same generated channels and their method-specific derived vectors. |
| Each SNR-specific neural model | 6,400 training and 256 validation system-channel realizations. One fixed data bank covers all 11 canonical SNR points; Optuna trials train only the five tuning-SNR models. |

For neural methods, one deterministic dataset bank is generated before the
study. Each SNR-specific model receives its own recorded 6,400/256 split, and
all trials reuse those splits. If noisy inputs are regenerated across epochs,
the noise uses deterministic SNR/epoch seeds shared across trials; the costly
propagation channels are never regenerated inside a trial.

Derived vectors from different antennas, users, pilot groups, or frequency
blocks of one realization are not counted as independent channels. Dataset
manifests and paper READMEs record both the independent-channel count and the
number of derived optimizer examples.

## Neural Training Protocol

The Melgar Model 2 and Model 3 estimators, Li ReEsNet, and the selected Baur
VAE-noisy estimator use the same outer training policy:

- Every Optuna trial trains one independent model for each of the five tuning
  SNRs.
- Training is limited to 200 epochs.
- Early stopping has a patience of 20 complete epochs.
- A validation loss counts as improved only when its relative decrease is at
  least `1e-4`.
- A tuned `optimizer_batch_size` uses the categorical search space
  `{256, 512, 1024, 2048}`.
- Validation monitors the method's paper-appropriate objective.
- Training restores the checkpoint with the best validation objective.
- Every trial model starts from a fresh, deterministically seeded
  initialization.

The common policy does not replace a paper's loss, optimizer, architecture,
or learning-rate schedule. Those remain method-specific. Early stopping may
shorten a trial, while the configured epoch limit and dataset size remain the
same for every candidate. After the recipe is selected, the final canonical
bank of 11 SNR-specific models is trained with the same full-effort policy.

## Algorithm Study Table

Detector-facing error-variance scales are deliberately excluded from this
table. Their separate calibration protocol is defined below.

| Paper and method | Parameters and search space | Completed trials | Trial-specific fitting or training |
| :- | :- | -: | :- |
| Coleri 2002, `LSCombEstimator` (low-pass) | `lowpass_filter_half_length`: integer `[1, 16]`; `lowpass_cutoff`: uniform `[0.1, 1.0]` | 50 | None. |
| Edfors 1996, `TruncatedChannelEstimator` (LS) | `num_taps`: integer `[1, N_p]`, where `N_p` is the number of frequency pilots in one pilot-bearing OFDM symbol; canonical maximum 156. The selected value is reused by LS, diagonal LMMSE, and LMMSE. | 50 | The common 64,000-channel delay-covariance asset is generated once before tuning. |
| Delfeld 2025, `RKHSChannelEstimator` | `rank`: integer `[1, 4]`; `sparse_num_iter`: integer `[5, 20]`; `debias_num_iter`: integer `[0, 20]`; `step_size`: log `[0.05, 2]`; `lambda_scale`: log `[1, 300]`; `debias_scale`: log `[1e-4, 1]` | 50 | None. |
| Koller 2022, `SpatialGMMChannelEstimator` | `num_components`: categorical `{5, 10, 15, 20}` | 4, exhaustive | Each component count is fitted to the common 6,400-channel dataset using 15 EM iterations and covariance regularization `1e-3`. |
| Koller 2022, `SpectralGMMChannelEstimator` | `num_components`: categorical `{5, 10, 15, 20}` | 4, exhaustive | Each component count is fitted to the common 6,400-channel dataset using 15 EM iterations and covariance regularization `1e-3`. |
| Wei 2017, `SBLChannelEstimator` | `num_iter`: integer `[1, 60]`; `alpha_init`: log `[1e-3, 10]`; `num_taps`: integer `[1, N_p]`, canonical maximum 156 | 50 | None. |
| Lyu 2026, `SparseDiffuseChannelEstimator` | `diffuse_num_taps`: integer `[1, 24]`; `tau_noise_scale`: log `[1e-2, 1e2]`; `diffuse_energy`: log `[1e-3, 1e2]`; `admm_rho`: log `[1e-2, 3]`; `admm_num_iters`: integer `[5, 20]` | 80 | None; every trial runs the complete Torch ADMM estimator. |
| Balevi 2020, `DeepPriorChannelEstimator` | `learning_rate`: log `[1e-4, 3e-2]`; `input_noise_scale`: log `[0.1, 3]`; `num_steps`: log-integer `[100, 2000]` | 30 | Observation-specific online fitting is performed for every probe observation. `hidden_channels=32` and six hidden layers remain fixed. |
| Melgar 2022, `DNNChannelEstimator` (Model 2) | `learning_rate`: log `[1e-5, 1e-3]`; `optimizer_batch_size`: categorical `{256, 512, 1024, 2048}` | 10 | Every trial trains a full five-SNR model bank with the common neural dataset and training protocol. |
| Melgar 2022, `InterpolatedDNNChannelEstimator` (Model 3) | `learning_rate`: log `[1e-5, 1e-3]`; `optimizer_batch_size`: categorical `{256, 512, 1024, 2048}` | 10 | Every trial trains a full five-SNR model bank with the common neural dataset and training protocol. |
| Li 2020, `ReEsNetChannelEstimator` | `learning_rate`: log `[1e-5, 1e-3]`; `optimizer_batch_size`: categorical `{256, 512, 1024, 2048}` | 10 | Every trial trains a full five-SNR model bank with the common neural dataset and training protocol. |
| Baur 2022, `VAEChannelEstimator` (`VAE-noisy`) | `latent_dim`: categorical `{4, 8, 16}`; `learning_rate`: log `[1e-5, 1e-3]`; `free_bits`: uniform `[0, 1]`; `optimizer_batch_size`: categorical `{256, 512, 1024, 2048}` | 10 | Every trial trains a full five-SNR model bank with the common neural dataset and training protocol. Only the selected clean-label `VAE-noisy` method is tuned for canonical use. |

The built-in LS/LMMSE references and the non-low-pass Coleri methods have no
algorithmic Optuna parameters. The Edfors covariance-generation budget is not
a hyperparameter. GMM component counts outside `{5, 10, 15, 20}` are not
considered, and previous Koller component counts are not injected as control
trials.

## Selection And Rounding

The study database and summary retain the raw Optuna winner. Each selected
continuous constructor value is rounded to at most three significant digits;
integer and categorical values remain unchanged.

Each non-GMM study compares the rounded winner with the currently selected
configuration on an independent, paired validation stream. The comparison
processes complete five-SNR sweeps so every intermediate result measures the
weighted tuning-range objective. It uses the canonical Monte Carlo budget of
63 batches of four channels at every tuning SNR, giving 252 channels per SNR
and 1,260 SNR/channel pairs per candidate. The rounded winner is selected if
and only if its paired weighted NMSE is strictly lower. A numerically tiny
improvement still counts; an exact tie retains the current configuration.

The GMM studies select the component count with the smallest study objective
from the four exhaustive candidates. They do not compare against or retain a
component count outside the defined grid.

The canonical benchmark is run only after parameters and assets are frozen.
Its results may be compared with the previous canonical reference, but that
comparison does not influence the selected parameters.

## Detector-Facing Variance Calibration

Variance calibration is separate from the algorithm-study table and starts
only after estimator parameters and inference assets are frozen.

- Calibration searches only the scalar variance multiplier over the
  logarithmic interval `[1e-3, 1]`.
- Each calibration uses 10 TPE trials and caches observations, channel
  estimates, and unscaled variances whenever the scalar cannot affect
  `h_hat`.
- The objective is codeword BLER at 10 dB.
- The objective uses the canonical benchmark budget at that SNR: 63 batches
  of four channels, or 252 channels in total.
- The rounded winner is compared with the current scale on a separate paired
  set of another 252 channels at 10 dB. It is selected if and only if its BLER
  is strictly lower; a tie retains the current scale.
- A shared scale used by several methods is calibrated against their mean BLER
  with equal method weight.
- Derived/reference variances without an empirical scalar are not calibrated.

Error-variance calibration changes neither the algorithm-study objective nor
the selected channel-estimator parameters and is documented separately in
`docs/mimo_ofdm_chest/inventories/error_variance_inventory.md`.

## Required Study Record

Each completed study records the following information in its summary, paper
README, and generation manifest where applicable:

- the exact parameter space and fixed constructor arguments;
- sampler type and seed;
- training, validation, probe, and paired-validation seeds;
- requested, completed, failed, and pruned trial counts;
- independent-channel counts and derived-example counts;
- SNR points, weights, batch size, and Monte Carlo batches;
- training epoch limit, early-stopping settings, and observed stopping epochs;
- raw winner, rounded candidate, paired comparison, and final selection;
- wall-clock runtime and GPU, CPU, CUDA, driver, and PyTorch details;
- the exact command, source commit, and Python dependency snapshot; and
- manifest and payload hashes for every fixed dataset or selected asset.
