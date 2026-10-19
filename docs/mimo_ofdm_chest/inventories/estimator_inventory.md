# Estimator Validation And Inference Inventory

This document is the cross-paper inventory of supported system geometries,
focused validation evidence, preparation and online-inference boundaries, and
`torch.compile` classifications. Detailed constructor contracts remain in the
repository README and paper READMEs. Asset generation and dataset sizes are
documented separately in
[`asset_inventory.md`](asset_inventory.md) and
[`dataset_inventory.md`](dataset_inventory.md).

Supported variants test only for interface correctness and numerical plausibility. They
do not claim that canonical hyperparameters are optimal for every listed
geometry. Restrictions outside a paper's supported contract are covered by
intentional rejection tests where applicable.

## Shared System Variants

| Variant | Effective subcarriers | OFDM symbols | Pilot symbols | UEs | BS antennas |
| :- | -: | -: | :- | -: | -: |
| `baseline` | 48 | 8 | `(1,)` | 1 | 2 (`1x2`) |
| `alternate` | 72 | 10 | `(2, 7)` | 3 | 4 (`2x2`) |
| `wide` | 96 | 12 | `(2, 5, 9)` | 4 | 12 (`2x6`) |

## Compilation Tiers

- **Tier A:** the complete online inference core compiles as a full graph.
- **Tier B:** the complete estimator runs through `torch.compile` with a small,
  documented compiler-disabled boundary or graph break.
- **Tier C:** an algorithm-required region such as observation-specific
  optimization cannot be compiled, while its tensor-only inference subroutine
  remains compilable.

Asset discovery, file I/O, dataset generation, offline fitting, and offline
training are preparation work and are excluded from all compiled inference
regions.

## Validation Coverage

| Implementation | Supported variant coverage | Intentional restrictions | Test coverage | Current CPU suite |
| :- | :- | :- | :- | :- |
| Built-in LS and LMMSE | All three variants exercise changed grids, pilot layouts, UE counts, and receive arrays. | Uses geometry accepted by Sionna. LMMSE covariance dimensions must match frequency, OFDM time, and the receive array. | End-to-end NMSE plausibility, covariance compatibility and malformed-asset rejection, precision, batch chunking, result linkage, and eager/compiled agreement. | 29 tests, 8.78 s |
| Coleri et al., 2002 | All three variants and all four interpolation methods. | Uniform frequency combs with common spacing, complete comb coverage, and equal pilot-bearing-symbol counts per stream. | Pilot preservation, boundaries, analytical polynomial/delay-limited cases, noise-gain bounds, result recording, and eager/compiled agreement. | 49 tests, 9.17 s |
| Edfors et al., 1996 | All three variants; LS, diagonal LMMSE, and LMMSE share one tap count. | Nonempty equal-length frequency groups; the DFT interpretation is validated for regular combs. Covariance identity must match the ordered pilot geometry. | Transform round trips, equivalence to Sionna LMMSE without truncation, delay-limited recovery, covariance ordering, truncation, asset validation, and eager/compiled agreement. | 27 tests, 9.43 s |
| Delfeld et al., 2025 | All three variants, including changed comb spacing and BS-array shape. | Equal-length uniform combs with at least two pilots, equal pilot-symbol counts per stream, and `num_rows*num_cols == num_rx_ant`. | Operator/adjoint identity, sparse thresholding, debiasing, low-rank reconstruction, geometry rejection, result recording, and eager/compiled agreement. | 18 tests, 7.09 s |
| Wei et al., 2017 | All three variants; compilation is checked on `baseline` and `alternate`. | Uniform constant-power combs satisfying `spacing*group_len == fft_size`. | Comb-transform reconstruction, one-update posterior equations, final variances, high-SNR limiting behavior, geometry rejection, result recording, and eager/compiled agreement. | 21 tests, 7.48 s |
| Koller et al., 2022 | All three variants for spatial and spectral inference; temporary fitting assets cover `baseline` and `alternate`. | Spatial mode requires equal-power active pilots. Spectral mode requires equal-length frequency groups and equal pilot-symbol counts. Both require an exact-geometry GMM. | Real/complex conventions, mixture responsibilities and conditional moments, vectorized-versus-loop processing, temporary fitting, malformed assets, result linkage, and eager/compiled agreement. | 55 tests, 10.37 s |
| Lyu et al., 2026 | All three variants, including changed comb spacing, pilot-symbol count, and antenna count. | Uniform nonzero frequency combs and compatible repeated pilot-symbol groups. Off-pilot values use explicit interpolation. | Repeated-observation averaging, diffuse-only recovery, Torch ADMM agreement with independent CVXPY P2 oracles, result recording, and eager/compiled agreement. | 16 tests, 115.70 s |
| Balevi et al., 2020 | All three variants with one-step temporary fitting. | Every stream's active pilots must form an equally sized rectangular frequency-time grid. | Architecture and parameter count, deterministic fixed input, batched-versus-independent optimization, objective reduction, interpolation behavior, rejection cases, result recording, and compiled/eager agreement across the fitting boundary. | 23 tests, 8.12 s |
| Melgar et al., 2022 | All three variants for Models 2 and 3 with temporary checkpoints. | Model 2 requires equal-length frequency groups. Model 3 requires a compatible LS-interpolation layout and `block_len` dividing the effective-subcarrier count. Checkpoints are exact-geometry assets. | Architecture, state selection, fixed-data reuse, best-checkpoint restoration, temporary training, malformed checkpoints, ordered manifest linkage, and mixed/per-SNR compiled inference. | 27 tests, 8.35 s |
| Li et al., 2020 | All three variants; temporary training covers `baseline` and `alternate`. | Active pilots must form rectangular grids, and full-grid dimensions must be integer multiples of pilot-grid dimensions. | Architecture, residual and long skips, super-resolution dimensions, pilot layout, link independence, temporary training, malformed checkpoints, and mixed/per-SNR compiled inference. | 23 tests, 8.92 s |
| Baur et al., 2022 | All three variants for noisy and real training paths with temporary checkpoints. | Pilot positions must form compatible receive-array vectors. Checkpoint method, architecture, pilot geometry, and receive dimension must match exactly. | Encoder/decoder parameterization, ELBO expressions, covariance properties, transform round trips, reproducible temporary training, malformed checkpoints, and mixed/per-SNR compiled inference. | 72 tests, 11.52 s |

## Preparation And Inference Boundaries

| Implementation | Construction and preparation | Online inference | Classification |
| :- | :- | :- | :- |
| Built-in LS | Creates the Sionna estimator and selected interpolator; no persistent asset is required. | LS estimation and Sionna interpolation. | Tier A. |
| Built-in LMMSE | `asset.py` loads or estimates covariance tensors before estimator construction. | Sionna full-grid LMMSE channel estimation. | Tier B at Sionna's compiler-disabled `LMMSEChannelEstimator.call()`. |
| Coleri et al., 2002 | Derives pilot geometry and fixed interpolation quantities. Cubic-spline weights are generated with SciPy. | LS estimation and configured interpolation use Torch operations and registered buffers. | Tier A for all four methods; constructor-time SciPy is outside inference. |
| Edfors et al., 1996 | `asset.py` loads or estimates the delay covariance; the estimator receives and registers the prepared matrix. | LS/LMMSE filtering, delay truncation, and configured Sionna interpolation. | Tier A for all three methods. |
| Delfeld et al., 2025 | Derives pilot geometry, registers RKHS quantities, and precomputes group-to-output indices. | Vectorized pilot-group processing, fixed-iteration sparse optimization and debiasing, full-frequency RKHS evaluation, and OFDM-time completion. | Tier A. |
| Wei et al., 2017 | Derives comb geometry and registers fixed transform quantities. | Joint pilot-group FFT processing, scalar Gaussian posterior and common-support EM updates, then Sionna interpolation. | Tier A. The implemented diagonalized uplink adaptation does not execute EP message passing. |
| Koller et al., 2022 | The runner strictly loads the selected GMM; tuning explicitly fits trial assets through `asset.py`. Estimators receive and register mixture tensors. | Spatial or full-spectral GMM conditional-mean and posterior-variance estimation, with interpolation where required. | Tier A for both methods. |
| Lyu et al., 2026 | Derives pilot geometry, averaging indices, and ADMM constants. | Fixed-iteration batched Torch ADMM followed by Sionna interpolation. | Tier A. CVXPY is a test-only oracle. |
| Balevi et al., 2020 | Stores architecture and optimization controls; no persistent model is loaded. | Each received LS pilot grid triggers construction and Adam fitting of an independent untrained network. | Tier C. The denoiser forward is Tier A; dynamic module construction, autograd, and optimizer mutation remain in an explicit compiler-disabled region. |
| Melgar et al., 2022, Model 2 | `asset.py` loads or trains DNN states; the estimator receives prepared states and builds the denoisers. | DNN pilot denoising followed by configured Sionna interpolation. | One mixed-SNR state is Tier A. A per-SNR bank is Tier B at the explicit prepared-state selection boundary. |
| Melgar et al., 2022, Model 3 | `asset.py` loads or trains DNN states; the estimator receives prepared states and builds the denoisers. | LS interpolation followed by block-DNN inference. | One mixed-SNR state is Tier A. The canonical per-SNR bank is Tier B at the prepared-state selection boundary. |
| Li et al., 2020 | `asset.py` loads or trains ReEsNet states; the estimator receives prepared states and optional per-SNR noise variances. | LS extraction, selected ReEsNet inference, and full-grid assembly. | One mixed-SNR state is Tier A. The canonical per-SNR bank is Tier B at the prepared-state selection boundary. |
| Baur et al., 2022 | `asset.py` loads or trains VAE states; the estimator receives prepared states and optional per-SNR noise variances. | Batched pilot-group encoder/decoder inference, VAE-LMMSE posterior estimation, and Sionna interpolation. | One mixed-SNR state is Tier A. The canonical per-SNR bank is Tier B at the prepared-state selection boundary. |

No estimator performs persistent file I/O from `call()` or
`estimate_at_pilot_locations()`. No pretrained estimator generates a dataset,
fits statistics, or trains weights from its online inference method. Balevi is
the deliberate exception for optimization: fitting an untrained network to
the current observation is the estimator itself and therefore remains an
online Tier C region.

Compilation tests instantiate estimators and resolve assets before calling
`torch.compile`. They compare eager and compiled outputs with the stable eager
backend; exact graph-break counts are diagnostic rather than brittle test
assertions.

## Static Implementation Audit

A current source scan found no online inference reads
from Sionna's global device or precision, and no tensor attributes created by
direct `torch.tensor`, `torch.arange`, `torch.zeros`, `torch.ones`, or
`torch.eye` assignment instead of buffer or parameter registration. The
built-in integration reads `config.precision` only during construction when no
explicit precision is supplied, matching Sionna's object fallback contract.

Fixed inference tensors are registered buffers or belong to registered model
modules. Neural-network weights remain `torch.float32`, matching their trained
checkpoint contracts; estimator outputs return to the configured Sionna
precision.

## Test Methodology

The focused suites above were recollected and run on 2026-08-26
with `CUDA_VISIBLE_DEVICES=''` on an AMD EPYC 7R13 host. The
environment used Python 3.12.3, calibrated Sionna 2.0.1 source commit
`f16208347246d8d17ec69628dfb856b19287b80c`, PyTorch 2.11.0+cu126,
NumPy 2.4.4, and SciPy 1.17.1. Each runtime is `/usr/bin/time` wall time for one fresh
`python -m pytest -q <suite>` process, including Python and pytest startup.
These suites currently collect **360 tests**. All were included in a full
repository run in which **424 tests passed**. The per-suite times in the table
remain the latest fresh-process measurements; they characterize development-test
cost, not estimator inference latency. Lyu's runtime is dominated by
independent CVXPY oracle comparisons.

The focused commands cover:

- `test/mimo_ofdm_chest/builtin/`; and
- every directory under `test/mimo_ofdm_chest/papers/`.

Other persistence, package, benchmark-infrastructure, and leaderboard tests
under `test/mimo_ofdm_chest/infrastructure/` are outside these per-estimator counts.

On 2026-09-09, the complete repository suite was also run in a clean CPU-only
environment installed from PyPI with Sionna 2.1.0 and Sionna RT 2.1.0. All
**425 tests passed**. Separate smoke checks covered built-in LS, analytical
Coleri interpolation, and Edfors inference using a freshly generated temporary
covariance asset.
