# Historical Sionna 2.0.1 Retuning Work Order

This work order records the completed retuning effort. Its source-commit gate,
commands, and document paths describe that historical run; new studies follow
`docs/mimo_ofdm_chest/contracts/tuning_protocol.md` and the current paper
entry points.

This checklist applies the protocol in
`docs/optuna_studies_v2.md`. Sections are ordered by estimated serial runtime
on one NVIDIA L40S so that short work can be completed locally while access to
a GB300 is arranged for the neural studies.

Runtime estimates include search, candidate selection, asset preparation where
applicable, detector-facing variance calibration, and one canonical benchmark.
They are planning estimates rather than deadlines. A single GB300 helps the
Torch workloads substantially but does little for the current NumPy/CPU GMM
fit.

| Order | Paper or reference | L40S best estimate | L40S conservative bound | One GB300 best estimate |
| -: | :- | :-: | :-: | :-: |
| 1 | Built-in references | 35-40 min | 45 min | 20-35 min |
| 2 | Coleri 2002 | 6-10 min | 15 min | 3-6 min |
| 3 | Edfors 1996 | 8-13 min | 20 min | 4-8 min |
| 4 | Wei 2017 | 7-11 min | 18 min | 3-7 min |
| 5 | Delfeld 2025 | 9-14 min | 25 min | 4-8 min |
| 6 | Balevi 2020 | 2.4-3.2 h | 5 h | 0.8-1.6 h |
| 7 | Lyu 2026 | 2.1-2.8 h | 4 h | 0.8-1.5 h |
| 8 | Koller 2022 | 28-38 h | 55 h | 27-43 h |
| 9 | Li 2020 | 38-50 h | 70 h | 10-22 h |
| 10 | Baur 2022 | 45-75 h | 190 h | 13-32 h |
| 11 | Melgar 2022 | 75-105 h | 180 h | 23-45 h |

The five-point tuning grid and canonical-sized paired validation give a
realistic complete serial estimate of roughly 10-13 days on one L40S, or 4-7
days on one GB300. The corresponding conservative L40S bound is about 22 days.
Koller changes little because GMM fitting, rather than the Optuna NMSE probe,
dominates its runtime.

## Protocol Freeze

### Mandatory Calibrated-Sionna Gate

No asset generation, Optuna study, paired validation, variance calibration,
or canonical benchmark may start unless a recorded preflight check confirms
all of the following:

- Sionna is imported from the calibrated source checkout under
  `/home/gmarti/sionna/src`, not from a previously installed wheel or
  `site-packages` copy;
- the active Sionna source commit is
  `f16208347246d8d17ec69628dfb856b19287b80c`;
- calibrated-channel-model commit
  `ad1609ea1` is an ancestor of the active Sionna commit; and
- the UMi implementation supports and uses TR 38.901 `spec_version="19.2"`.

Run commands must put `/home/gmarti/sionna/src` before the sionna_bench
`src` directory on `PYTHONPATH`. The preflight import path, Sionna commit,
model version, and command must be stored with each study's evidence. A run
that does not satisfy this gate is invalid regardless of its reported Sionna
version string and must not be promoted, compared as a v2 candidate, or used
to select parameters. This gate exists because the older installed package
and calibrated source checkout both report Sionna 2.0.1 despite implementing
different TR 38.901 channel models.

- [x] Freeze `docs/optuna_studies_v2.md`, this work order, every paper tuning
  entry point, and every asset/data-generation implementation before starting
  a v2 run. The user subsequently required the repository diff to remain
  uncommitted pending review; each accepted run therefore records the exact
  governed diff and reconstructable source hash instead of a protocol commit.
- [x] Record the exact seed assigned to training data, validation data, model
  initialization, Optuna sampling, algorithm probes, paired validation,
  detector calibration, and canonical evaluation. Ensure the streams are
  deterministic and distinct.
- [x] Record the source commit and a complete Python dependency snapshot.
- [x] Define one run-metadata record containing GPU, CPU, CUDA, driver,
  PyTorch, and wall-clock information, and require it from every stage.
- [x] After generating each fixed dataset, record its manifest and payload
  hashes before allowing any study to consume it.
- [x] Do not modify the frozen protocol, search spaces, trial counts, seeds,
  datasets, or selection rule during execution. Any necessary change starts a
  separately versioned protocol rather than silently altering v2.

### Actual Runtime Record

Populate this table from measured wall-clock times. Use `N/A` when a phase
does not apply, and include failed or repeated work in the notes rather than
hiding it from the total.

| Paper or reference | Data generation | Algorithm study | Paired validation | Final asset | Variance calibration | Benchmark | Total | Hardware and notes |
| :- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :- |
| Built-in references | 52.64 s | N/A | N/A | Included in data generation | N/A | 32 min 56.67 s | 33 min 49.31 s | NVIDIA L40S. The accepted rerun used calibrated Sionna commit `f1620834`, UMi v19.2, and produced results bit-identical to the comparison worktree. An earlier installed-package run was invalid and is excluded from the total. |
| Coleri 2002 | N/A | 21.25 s | 27.19 s | N/A | 78.64 s search + 30.90 s validation | 3 min 44.30 s | 6 min 22.28 s | NVIDIA L40S; calibrated Sionna commit `f1620834`, UMi v19.2. All canonical curves are bit-identical to the comparison worktree. |
| Edfors 1996 | 1 min 19.34 s | 15.88 s | 23.09 s | Included in data generation | 56.45 s search + 20.40 s validation | 2 min 4.71 s | 5 min 19.87 s | NVIDIA L40S; calibrated Sionna commit `f1620834`, UMi v19.2. The accepted 64,000-channel covariance produced curves statistically consistent with the comparison worktree. |
| Wei 2017 | N/A | 17.95 s | 25.01 s | N/A | 25.16 s search + 12.16 s validation | 58.16 s | 2 min 18.44 s | NVIDIA L40S; calibrated Sionna commit `f1620834`, UMi v19.2. The selected algorithm trades modest low-SNR NMSE for improved high-SNR NMSE under the frozen weighted objective; BER and BLER remain statistically consistent with the comparison worktree. |
| Delfeld 2025 | N/A | 35.22 s | 45.51 s | N/A | 27.83 s search + 14.80 s validation | 1 min 25.76 s | 3 min 29.12 s | NVIDIA L40S; calibrated Sionna commit `f1620834`, UMi v19.2. Paired validation rejected the algorithm winner and tied the variance candidate, retaining all current parameters. The canonical output is bit-identical to the comparison worktree. |
| Balevi 2020 | N/A | 47 min 22.05 s | 1 h 25 min 56 s | N/A | 11 min 51.46 s search + 12 min 24.39 s validation | 2 h 11 min 19 s | 4 h 48 min 52.90 s | NVIDIA L40S; calibrated Sionna commit `f1620834`, UMi v19.2. Paired validation retained the incumbent fitting recipe; variance validation selected `0.454`. All canonical 95% intervals overlap the comparison worktree. |
| Lyu 2026 | N/A | 1 h 32 min 12 s | 41 min 7.55 s | N/A | 5 min 21.10 s search + 5 min 10.89 s validation | 57 min 40.57 s | 3 h 21 min 32.11 s | NVIDIA L40S; calibrated Sionna commit `f1620834`, UMi v19.2. Paired validation selected rounded ADMM parameters `(3, 6.55, 0.807, 0.08, 18)`; variance validation tied and retained scale `1.0`. The canonical run improves BER/BLER throughout but has 10.6% worse five-SNR weighted NMSE than the comparison worktree. |
| Koller 2022 | 57.15 s | Spatial 1 h 18 min 25 s; spectral 3 min 44.09 s | N/A | Included in fitting | N/A | Spatial 42.50 s; spectral 11 min 45.39 s | 1 h 35 min 34.13 s | NVIDIA L40S; calibrated Sionna commit `f1620834`, UMi v19.2. Exhaustive grids selected spatial K20 and spectral K15 from one 6,400-channel bank. |
| Li 2020 | 3 min 28.85 s | 29 h 15 min 10 s | 3 h 42 min 24 s | 4 h 8 min 14 s | 19.31 s search + 9.25 s validation | 42.45 s | 37 h 10 min 27.86 s | NVIDIA L40S; calibrated Sionna commit `f1620834`, UMi v19.2. Ten trials selected learning rate `0.000186` and batch size 256; variance validation retained the incumbent. |
| Baur 2022 | bounded 1 min 33.01-33.73 s | 100 h 5 min 19 s + 17 h 38 min 5 s accepted recovery | 14 h 26 min 1 s | 30 h 53 min 22 s | 20.30 s search + 9.68 s validation | 44.66 s | about 163 h 5 min 35 s | NVIDIA L40S; calibrated Sionna commit `f1620834`, UMi v19.2. Ten complete and two failed trial numbers are retained. Invalid unseeded and negative-loss attempts are quarantined and excluded. |
| Melgar 2022 Model 3 | 7.94 s | 11 h 1 min 56 s | 2 h 11 min 48 s | 3 h 48 min 10 s | 19.74 s search + 9.79 s validation | 44.11 s | 17 h 3 min 15.58 s | NVIDIA L40S; calibrated Sionna commit `f1620834`, UMi v19.2. Ten trials selected learning rate `3.17e-5` and batch size 512; variance validation retained exact incumbent scale. |
| Melgar 2022 Model 2 | 1 min 34.93 s | 14 h 7 min 6 s | 3 h 41 min 2 s | 6 h 12 min 29 s | 24.96 s search + 13.67 s validation | 59.47 s | 24 h 3 min 50.03 s | NVIDIA L40S; calibrated Sionna commit `f1620834`, UMi v19.2. Ten trials selected learning rate `7.02e-5` and batch size 256; variance validation selected scale `0.454`. All 11 payloads exactly reproduced their recorded best-validation metrics after manifest bookkeeping was corrected. |

The comparison worktree is `/home/gmarti/sionna_bench`. Candidate work remains
in `/home/gmarti/sionna_bench-cleanup`. Never overwrite the comparison
worktree's assets, tuning evidence, or outputs.

## 1. Built-In References

- [x] Rerun the mandatory calibrated-Sionna preflight and retain its import
  path, source commits, and explicit UMi model version with the evidence.
- [x] Regenerate the canonical frequency, time, and spatial covariance
  statistics once from 64,000 channels under an external asset root.
- [x] Validate the asset manifest and load the covariance through the strict
  bundled/external asset API.
- [x] Run LS nearest-neighbor, LS linear, and LMMSE canonical benchmarks into
  a candidate output directory.
- [x] Compare all NMSE, BER, and BLER curves with the corresponding outputs in
  `/home/gmarti/sionna_bench`; explain statistically material differences.
- [x] Promote the accepted covariance and outputs, then document the exact
  generation command, seed, runtime, hardware, and manifest identity.

The earlier built-in v2 asset and outputs generated from the installed Sionna
2.0.1 package were invalid and are excluded from the evidence. The checked-off
items above refer only to their calibrated-source replacements.

## 2. Coleri 2002

- [x] Update the low-pass tuning entry point to the v2 five-SNR, 50-trial,
  three-batch protocol and remove historical trial injection.
- [x] Run the low-pass study and paired validation of the rounded winner.
- [x] Freeze the common interpolation parameters, then calibrate the shared
  detector-facing variance scale at 10 dB with the v2 calibration protocol.
- [x] Run all four canonical Coleri methods into a candidate output directory.
- [x] Compare all NMSE, BER, and BLER curves with the corresponding outputs in
  `/home/gmarti/sionna_bench`; explain statistically material differences.
- [x] Promote accepted results and update the README, reports, study summary,
  and tuning command/runtime record.

## 3. Edfors 1996

- [x] Generate one 64,000-channel delay-covariance asset before tuning and
  reuse it for every tap-count candidate and all three methods.
- [x] Update and run the 50-trial LS `num_taps` study under the v2 weighted-NMSE
  protocol; reuse the selected tap count for LS, diagonal LMMSE, and LMMSE.
- [x] Paired-validate the rounded/integer winner against the current tap count.
- [x] Freeze the covariance and tap count, then calibrate the shared
  detector-facing scale against the equal-method mean 10 dB BLER.
- [x] Run all three canonical methods into a candidate output directory.
- [x] Compare all NMSE, BER, and BLER curves with the corresponding outputs in
  `/home/gmarti/sionna_bench`; explain statistically material differences.
- [x] Promote accepted assets/results and document covariance generation,
  study evidence, calibration, runtime, hardware, and manifest identity.

## 4. Wei 2017

- [x] Update and run the 50-trial SBL study over the full v2 SNR grid and probe
  budget without historical trial injection.
- [x] Round and paired-validate `alpha_init` together with the selected integer
  iteration and tap counts.
- [x] Freeze the SBL algorithm, then calibrate `posterior_variance_scale` at
  10 dB under the v2 detector protocol.
- [x] Run the canonical benchmark into a candidate output directory.
- [x] Compare all NMSE, BER, and BLER curves with the corresponding output in
  `/home/gmarti/sionna_bench`; explain statistically material differences.
- [x] Promote accepted results and update the README, reports, and complete
  tuning/calibration evidence.

## 5. Delfeld 2025

- [x] Update and run the 50-trial RKHS study over the full v2 SNR grid and
  fixed probe observations without historical trial injection.
- [x] Round all continuous parameters to three significant digits and run the
  paired five-SNR validation.
- [x] Freeze the RKHS parameters, then calibrate `err_var_scale` at 10 dB with
  cached channel estimates and unscaled variances.
- [x] Run the canonical benchmark into a candidate output directory.
- [x] Compare all NMSE, BER, and BLER curves with the corresponding output in
  `/home/gmarti/sionna_bench`; explain statistically material differences.
- [x] Promote accepted results and update the README, reports, and complete
  tuning/calibration evidence.

## 6. Balevi 2020

- [x] Update and run the 30-trial online-fitting study with the exact v2 probe
  observations and full candidate `num_steps`; do not use a low-fidelity fit.
- [x] Round the continuous parameters and run paired five-SNR validation.
- [x] Freeze fitting behavior, then run the 10 dB variance calibration. Fit
  each calibration observation once and cache its estimate and unscaled
  variance across scalar trials.
- [x] Run the canonical benchmark into a candidate output directory.
- [x] Compare all NMSE, BER, and BLER curves with the corresponding output in
  `/home/gmarti/sionna_bench`; explain statistically material differences.
- [x] Promote accepted results and update the README, reports, study summary,
  and measured online-fitting runtime.

## 7. Lyu 2026

- [x] Update and run the 80-completed-trial ADMM study with the v2 SNR grid,
  probe budget, and failure accounting.
- [x] Round continuous solver/model parameters and run paired five-SNR
  validation with complete ADMM inference.
- [x] Freeze the ADMM parameters, then calibrate `err_var_scale` at 10 dB with
  cached estimates and unscaled variances.
- [x] Run the canonical benchmark into a candidate output directory.
- [x] Compare all NMSE, BER, and BLER curves with the corresponding output in
  `/home/gmarti/sionna_bench`; explain statistically material differences.
- [x] Promote accepted results and update the README, reports, failure counts,
  selected parameters, commands, hardware, and runtime.

## 8. Koller 2022

- [x] Generate and retain one deterministic 6,400-channel raw fitting dataset;
  derive both spatial and spectral training vectors from it.
- [x] Replace arbitrary integer component suggestions with exhaustive Optuna
  grids over `{5, 10, 15, 20}` for each method.
- [x] Fit all eight GMM candidates with 15 EM iterations and covariance
  regularization `1e-3`, reusing the cached generated channels and never
  regenerating an existing candidate.
- [x] Evaluate every candidate on the common v2 weighted-NMSE probe and select
  the lowest-objective component count independently for each method. Do not
  inject or retain an old component count outside the grid.
- [x] Validate both selected manifests and run spatial and spectral canonical
  benchmarks into a candidate output directory.
- [x] Compare both NMSE, BER, and BLER curves with the corresponding outputs in
  `/home/gmarti/sionna_bench`; treat this only as regression analysis, not as a
  reason to switch to an excluded component count.
- [x] Promote accepted assets/results and document independent-channel and
  derived-vector counts, commands, seeds, fit runtime, hardware, and payload
  manifests.

## 9. Li 2020

- [x] Implement a reusable deterministic data bank containing 6,400 training
  and 256 validation channels for each canonical SNR; generate it once.
- [x] Update ReEsNet training to the common 200-epoch, patience-20,
  relative-`1e-4` early-stopping policy with best-checkpoint restoration.
- [x] Run ten full training trials over learning rate and optimizer batch size;
  every trial trains all five tuning-SNR models on the fixed data bank.
- [x] Round the learning rate and paired-validate the selected recipe against
  the current recipe using independently trained full-effort model banks.
- [x] Train the final 11-SNR checkpoint bank with the selected recipe, then
  calibrate `err_var_scale` at 10 dB with cached estimates and unscaled
  variances.
- [x] Run the canonical benchmark into a candidate output directory.
- [x] Compare all NMSE, BER, and BLER curves with the corresponding output in
  `/home/gmarti/sionna_bench`; explain statistically material differences.
- [x] Promote accepted assets/results and document per-SNR channel counts,
  derived grids, stopping epochs, commands, seeds, runtime, hardware, and all
  selected manifest identities.

## 10. Baur 2022

- [x] Implement a reusable deterministic data bank containing 6,400 training
  and 256 validation channels for each canonical SNR; generate it once.
- [x] Update VAE-noisy training to the common 200-epoch, patience-20,
  relative-`1e-4` early-stopping policy with best-checkpoint restoration.
- [x] Run ten full training trials over latent dimension, learning rate, free
  bits, and optimizer batch size; every trial trains all five tuning-SNR
  models on the fixed data bank.
- [x] Round continuous parameters and paired-validate the selected recipe
  against the current recipe using independently trained full-effort banks.
- [x] Train the final 11-SNR VAE-noisy checkpoint bank with the selected
  recipe, then calibrate its detector-facing variance scale at 10 dB with
  cached estimates and unscaled variances.
- [x] Run the canonical benchmark into a candidate output directory.
- [x] Compare all NMSE, BER, and BLER curves with the corresponding output in
  `/home/gmarti/sionna_bench`; explain statistically material differences.
- [x] Promote accepted assets/results and document per-SNR channel counts,
  derived receive vectors, stopping epochs, commands, seeds, runtime,
  hardware, and all selected manifest identities.

## 11. Melgar 2022

Both Model 2 and Model 3 have completed the calibrated-Sionna training,
validation, asset-generation, detector-calibration, and benchmark protocol.

- [x] Implement reusable deterministic Model 2 and Model 3 data banks with
  6,400 training and 256 validation channels for every canonical SNR and
  expose their derived examples without regenerating channels.
- [x] Apply the common 200-epoch, patience-20, relative-`1e-4` early-stopping
  policy with best-checkpoint restoration to Model 3.
- [x] Run ten full Model 3 trials over learning rate and optimizer batch size;
  every trial trains all five tuning-SNR models.
- [x] Round the selected learning rate and paired-validate the selected Model 3
  recipe against its current recipe using independently trained full-effort
  model banks.
- [x] Train the final 11-SNR Model 3 checkpoint bank with the selected recipe,
  then calibrate its detector-facing variance scale at 10 dB with cached
  estimates and unscaled variances.
- [x] Run Model 3 into a candidate output directory.
- [x] Compare its NMSE, BER, and BLER curves with the corresponding output in
  `/home/gmarti/sionna_bench`; explain statistically material differences.
- [x] Promote the accepted Model 3 asset/result and document per-SNR channel counts,
  derived examples, stopping epochs, commands, seeds, runtime, hardware, and
  all selected manifest identities.
- [x] Run ten full Model 2 trials over learning rate and optimizer batch size;
  every trial trains all five tuning-SNR models from the fixed data bank.
- [x] Pair-validate the rounded Model 2 winner, train and strictly validate the
  final 11-SNR bank, and independently verify that every saved payload
  reproduces its recorded best-validation metric.
- [x] Calibrate Model 2 detector-facing variance at 10 dB, run the canonical
  benchmark, compare it with the previous reference, and promote the selected
  assets, result, metadata, and evidence.

## Final Cross-Paper Checks

- [x] Confirm that every selected algorithm parameter and asset was chosen
  before its canonical benchmark was inspected.
- [x] Confirm that all continuous selected values use at most three
  significant digits and that raw Optuna winners remain in study evidence.
- [x] Confirm that every paper README reports the v2 dataset, search,
  validation, calibration, hardware, runtime, and exact commands.
- [x] Regenerate the leaderboard only after every accepted result has been
  promoted together with its matching configuration and asset references.
