# Implementation of the Estimators from Lyu et al., 2026

Publication: L. Lyu, M. Ferreira Da Costa, and U. Mitra, "Atomic Hybrid Sparse/Diffuse Channel Estimation and Cramer-Rao Bounds Analysis," arXiv:2605.01825v1, 2026

## Abstract

Lyu et al., 2026 propose a hybrid sparse/diffuse OFDM channel model and a
Toeplitz semidefinite-program estimator derived from the atomic norm. This
implementation applies the P2 Toeplitz SDP to LS-normalized pilot-comb vectors,
treats the comb as a reduced OFDM grid, and delegates off-pilot reconstruction
to explicit Sionna interpolation.

This is a diagnostic adaptation, not the paper's exact full-grid HALS/P2
estimator.

## Implemented Class

`SparseDiffuseChannelEstimator`

The class is a Sionna `BasePilotChannelEstimator` and uses Sionna's internal
`_PilotGroupMixin` to form uniform pilot-comb observations. The base estimator
extracts pilots and applies the configured interpolator after the paper-local
pilot-domain estimate has been computed.

- `estimate_at_pilot_locations()` forms LS pilot observations and returns
  reduced-comb estimates plus detector-facing error variances.
- `_average_and_solve_pilot_groups()` averages repeated pilot-symbol
  observations for each stream and receive antenna as in Equation (12), then
  reuses the resulting estimate at the corresponding pilot symbols.
- `_solve_admm()` implements the iterative ADMM approximation as batched
  Torch solves on the estimator device.
- `_build_diffuse_dictionary()` constructs the reduced-grid diffuse steering
  dictionary corresponding to the paper's matrix `D`.

The paper-local module layout follows `docs/paper_architecture.md`:
`estimator.py` owns the online Torch solver, `tuning.py` exposes `run_tuning()`,
and `run_benchmark.py` exposes `run_benchmark()`. No persistent asset is
required. The independent CVXPY formulation exists only in the test module.

## Modifications and Omissions

- Equations (5)-(6) stack observations, transmitted symbols, and channel
  coefficients over all `N` subcarriers. Equation (12) then averages complete
  `N`-subcarrier LS observations; the paper defines neither sparse pilot
  sampling nor interpolation. This implementation therefore remains a
  pilot-domain estimator: it replaces the paper's complete observation by a
  Sionna pilot-comb LS vector, treats that comb as a reduced OFDM grid with a
  larger effective subcarrier spacing, and applies the configured Sionna
  interpolator explicitly afterward.
- Repeated pilot-symbol observations are averaged independently for every
  receive antenna. The implementation does not impose shared support across
  antennas.
- The Toeplitz SDP structure is retained; this is not a finite-grid `l1`
  replacement.
- The estimator returns the biased `h_s + D gamma` reconstruction. Continuous
  support extraction from Equation (16) and projection debiasing from Equation
  (17) are not implemented.
- The paper does not define Sionna detector-facing marginal error variances.
  For the Equation (12) average with weights `w_g`, this implementation uses
  `err_var_scale * no * sum_g w_g^2/|pilot_g|^2`, followed by the configured
  Sionna interpolation. With `G` equal-power pilots, the unscaled term is
  `no/(G*|pilot|^2)`.
- Torch ADMM is approximate and is regression-tested directly against a
  test-local CVXPY/Clarabel oracle on 50 small
  randomized reduced-SDP cases spanning `(K, L) = (6, 2), (8, 4), (10, 7)`.
- The ADMM estimate is returned directly. Numerical failures are not hidden by
  substituting LS observations or applying a reconstruction-gain threshold.
- The complete estimator, including ADMM and Sionna interpolation, compiles as
  one full graph with no graph breaks and is classified as Tier A.
- End-to-end tests cover uniform comb spacings 1, 3, and 4; one, two, and three
  pilot-bearing OFDM symbols; 1, 3, and 4 UEs; and 2, 4, and 12 BS antennas.
  These tests use a two-iteration ADMM budget and verify interface behavior
  rather than canonical performance.
- The calibrated-source v2 regeneration completed 80 ADMM trials, paired
  algorithm validation, detector-facing variance calibration, and a canonical
  benchmark. The validated rounded ADMM configuration is now canonical.

## Hyperparameter Tuning And Calibration

The final study used the calibrated Sionna source checkout at commit
`f16208347246d8d17ec69628dfb856b19287b80c`, including calibration commit
`ad1609ea1`, and the explicit UMi TR 38.901 version `19.2`. The seeded TPE
study minimized weighted NMSE at `[-10, -5, 0, 5, 10]` dB with weights
`10**(Eb/N0/10)`. Each of its 80 completed trials used three batches of four
channels per SNR, or 60 SNR/channel evaluations. The study ran 80 requested
trials: 80 completed, zero failed, and zero were pruned.
The search space was:

- integer `diffuse_num_taps` from 1 through 24;
- logarithmic `tau_noise_scale` from `1e-2` through `1e2`;
- logarithmic `diffuse_energy` from `1e-3` through `1e2`;
- logarithmic `admm_rho` from `1e-2` through `3`; and
- integer `admm_num_iters` from 5 through 20.

Sampler seed `123` and probe seed `4242` produced the raw winner
`(3, 6.5484840637, 0.8068198786, 0.07996883845, 18)` with weighted NMSE
`0.0058468221`. The study took 1 h 32 min 12 s on one NVIDIA L40S. Continuous
parameters were rounded to three significant digits before validation.

Paired validation used seed `314159`, all 63 canonical Monte Carlo batches of
size four, and the same five SNRs. The former canonical configuration scored
`0.0085787708`; rounded `(3, 6.55, 0.807, 0.08, 18)` scored `0.0061135085`
and was selected. Validation took 41 min 8 s.

```bash
python -m sionna.bench.mimo_ofdm_chest.papers.lyu2026.tuning \
  --study admm --trials 80 --num-mc-batches 3 --batch-size 4 \
  --output-dir tuning/mimo_ofdm_chest/lyu2026/admm_v2
python -m sionna.bench.mimo_ofdm_chest.papers.lyu2026.tuning \
  --study admm-validation --num-mc-batches 63 --batch-size 4 \
  --output-dir tuning/mimo_ofdm_chest/lyu2026/admm_v2
```

### Detector-Facing Error Variance

With the rounded ADMM configuration frozen, ten logarithmic trials searched
`err_var_scale` over `[1e-3, 1]` using codeword BLER at 10 dB. Each trial
reused cached estimates and unscaled variances for 63 batches of four
channels. Sampler seed `456` and probe seed `271828` selected raw
`0.005575293412686262`, with raw objective `0.02513227513227513`; cache
preparation and the search took 5 min 21 s, while the
cached Optuna evaluations took 11.5 s.
The calibration ran 10 requested trials: 10 completed, zero failed, and zero
were pruned.

Paired validation with seed `161803` and another 63 batches of four channels
gave identical BLER `0.0201719577` for incumbent `1.0` and rounded `0.00558`.
The strict tie rule therefore retained `err_var_scale=1.0`. Validation took
5 min 11 s.

```bash
python -m sionna.bench.mimo_ofdm_chest.papers.lyu2026.tuning \
  --study variance --algorithm-output-dir tuning/mimo_ofdm_chest/lyu2026/admm_v2 \
  --output-dir tuning/mimo_ofdm_chest/lyu2026/variance_scale_v2
python -m sionna.bench.mimo_ofdm_chest.papers.lyu2026.tuning \
  --study variance-validation \
  --algorithm-output-dir tuning/mimo_ofdm_chest/lyu2026/admm_v2 \
  --output-dir tuning/mimo_ofdm_chest/lyu2026/variance_scale_v2
```

## Benchmark Run

The canonical command is:

```bash
python -m sionna.bench.mimo_ofdm_chest.papers.lyu2026.run_benchmark \
  --diffuse-num-taps 3 --tau-noise-scale 6.55 \
  --diffuse-energy 0.807 --err-var-scale 1.0 \
  --admm-rho 0.08 --admm-num-iters 18
```

The remaining defaults are `tau=3.0`, `tau_min=1e-6`, `tau_max=3.0`, and
`interpolation_type=lin_time_avg`. The canonical run took 57 min 41 s.

Compared with the former canonical output, weighted NMSE over the five tuning
SNRs changed from `0.0053374035` to `0.0059030533` (+10.6%). NMSE improved
from -10 through -2.5 dB and regressed from 0 through 15 dB. BER and BLER,
however, improved at every SNR; at 10 dB BER fell by 33.2% and BLER by 43.9%.
The benchmark was used only as a final test, not to revise the parameter
selection made on independent paired validation.

The retained preflight records Sionna Bench commit
`5012974748d4b8aa6426e563d8fed1380eba3abc` and tracked-diff SHA-256
`5b3059817c65b81e80165c3ce6b8f562678b0d8d47aed3e3ff91c234eb45309f`.
The environment was Python 3.12.3, PyTorch `2.11.0+cu126`, CUDA 12.6,
driver `580.126.09`, NumPy 2.4.4, SciPy 1.17.1, and Optuna 4.9.0 on an
NVIDIA L40S and AMD EPYC 7R13. The retained `pip-freeze.txt` has SHA-256
`b981919b35253baa1f4cbc244763e09292be6a503d42480d3f9003e397884082`.
Canonical evaluation used seed `2026`.
