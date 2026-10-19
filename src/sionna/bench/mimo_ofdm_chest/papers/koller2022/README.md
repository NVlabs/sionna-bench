# Implementation of the Estimators from Koller et al., 2022

Publication: M. Koller, B. Fesl, N. Turan, and W. Utschick, "An Asymptotically MSE-Optimal Estimator Based on Gaussian Mixture Models," IEEE Transactions on Signal Processing, vol. 70, pp. 4109-4123, 2022.

## Abstract

This implementation covers GMM conditional-mean estimators from Koller et al.,
2022. The paper models the channel vector with a Gaussian mixture model fitted
from channel samples and then computes the conditional mean given noisy
observations.

Two benchmarkable estimators are implemented:

- `SpectralGMMChannelEstimator`: uses a GMM fitted over full
  effective-subcarrier channel vectors and treats the benchmark pilot
  subcarriers as the observed coordinates of that vector.
- `SpatialGMMChannelEstimator`: uses a GMM fitted over receive-array vectors
  at each nonzero pilot resource element, estimates only pilot-domain values,
  and delegates off-pilot resource-grid reconstruction to a configured Sionna
  interpolator. Its canonical default is `lin_time_avg`.

## Implemented Classes

### `SpatialGMMChannelEstimator`

`SpatialGMMChannelEstimator` is a Sionna `BasePilotChannelEstimator` for the
spatial pilot-domain setting. It uses Sionna's internal `_PilotGroupMixin` to
group nonzero pilot resource elements, computes LS pilot estimates, applies
the paper's GMM conditional-mean estimator to each receive-array vector, and
then relies on Sionna interpolation for off-pilot resource elements. It
exposes the standard `interpolation_type` and custom `interpolator` constructor
arguments; the canonical default remains `lin_time_avg`.

Its main methods are:

- `__init__`: initializes Sionna's `BasePilotChannelEstimator`, validates the
  injected GMM weights, means, and covariances, and registers them as buffers.
- `estimate_at_pilot_locations`: computes Sionna LS pilot estimates, gathers
  them into pilot groups, jointly evaluates the groups sharing GMM and noise
  statistics, and scatters the denoised estimates back to Sionna's layout.
- `_estimate_groups`: vectorizes the online GMM conditional mean and posterior
  variance computation over pilot groups. This corresponds to equations (13),
  (15), and (16) with the pilot-domain observation matrix specialized to
  `A = I`.
- `_compute_gmm_term_quantities`: builds the noise-dependent Cholesky factors,
  component gains, posterior diagonals, and log normalizers used by
  `_estimate_groups`.

Runner outputs for this estimator are named `SpatialGMMChannelEstimator`.

### `SpectralGMMChannelEstimator`

`SpectralGMMChannelEstimator` is a Sionna `BaseChannelEstimator` that implements
the full-grid `call()` contract directly. Its prepared GMM models full
effective-subcarrier vectors for each receive antenna, transmit stream, and
pilot-bearing OFDM symbol. At runtime it uses Sionna's known pilots to form LS
observations on the pilot subcarriers, then evaluates the GMM conditional mean
for the full subcarrier vector using the corresponding selection observation
model from the paper. Sionna's internal `_PilotGroupMixin` supplies the pilot
coordinates used by this observation model.

Its main methods are:

- `__init__`: initializes pilot grouping, validates and registers the injected
  full-spectral GMM parameters, and configures OFDM-time interpolation.
- `call`: gathers Sionna pilot observations from the received resource grid,
  estimates all effective subcarriers on pilot-bearing OFDM symbols, and calls
  `TimeDomainInterpolator`.
- `_estimate_full_spectral_group`: implements the full-vector GMM conditional
  mean and posterior variance for one pilot group observed through a
  subcarrier-selection matrix.

Runner outputs for this estimator are named `SpectralGMMChannelEstimator`.

### `asset.py`

`asset.py` implements the offline GMM fitting step corresponding to Algorithm
1 line 1. Its public `fit_model()` handle derives the selected method's pilot
groups, samples true channels, and fits a full-covariance real GMM by EM under
an explicit external asset root. Normal benchmark execution calls the strict
`load_model()` handle, which never samples channels, fits a model, or writes
files. Both functions return prepared parameters for direct injection into an
estimator together with the selected manifest path. Tuning first attempts a
strict load and explicitly fits only when the trial-specific asset is absent.

Strict loading without an explicit asset root checks for optional package-local
K15 spectral and K20 spatial assets. Public distributions omit these payloads,
so public evaluation uses an explicitly generated asset under
`<asset-root>/<scenario_id>/koller2022/`. Versioned manifests record the
estimator mode, real-vector convention, vector axis and dimensions, and exact
ordered pilot-group coordinates, values, and powers through their shapes,
dtypes, encodings, and canonical SHA-256 identities. Compatibility is not
inferred from the artifact directory name or a group count.

Both GMM asset types use the repository-wide asset-manifest schema 2. Spatial
GMMs store `weights`, `means`, and `covs` together in `gmm_params.npz`.
Full-spectral GMMs store `weights` and `means` in that archive and write each
component covariance to
`gmm_covariance_<component-index>.npy`. Loading stacks those files back into
the same float32 covariance tensor used by the estimator. This persistence-only
split keeps each distributable payload comfortably below common Git hosting
file-size limits; it does not change fitting or inference. The `spatial-gmm`
and `spectral-gmm` asset types and their nested payload metadata identify these
different layouts; the loader validates each layout explicitly.

Asset-generation tests fit one- or two-component spatial and spectral GMMs on
two reduced noncanonical system geometries using temporary asset roots. They
also verify that compatible generated parameters reload without fitting again.
Loader rejection tests cover changed ordered pilot coordinates, changed
spatial receive-vector and spectral full-frequency dimensions, missing GMM
parameter or covariance payloads, and incorrect weight, mean, or covariance
shapes and dtypes.
Loading uses the manifest's explicit `compatibility` and `generation_recipe`
sections. Newly generated manifests omit the redundant legacy `key_fields`
entry.
Constructor tests also verify that spatial joint processing rejects unequal
pilot powers and that spectral reconstruction rejects unequal pilot-bearing
OFDM-symbol counts between streams.
System-level tests then run both estimators with geometry-matched temporary
GMMs for 1, 3, and 4 UEs and 2, 4, and 12 BS antennas. The complete OFDM/UMi
paths satisfy the output contract and outperform a zero channel estimate at
5 dB with the small test fitting budget.

Independent analytical tests verify the real-then-imaginary stacking order,
component Gaussian likelihoods, normalized responsibilities, conditional
mixture means, and non-negative posterior variances. These checks use
nonuniform per-entry noise and explicit subcarrier selection so matrix-ordering
errors are not hidden by scalar-noise special cases. A separate parametrized
check verifies that vectorized spatial processing agrees with independent
group processing for noncanonical batch, receiver, group, and antenna shapes.
The spectral checks cover every UE comb offset in the three-UE test geometry.

For the spatial estimator, LS normalization turns the observation model into
`A = I`. For the spectral estimator, `A` selects the active pilot-subcarrier
coordinates from the full spectral vector. In the real-stacked implementation,
those coordinates are `[q, q+N]` for subcarrier indices `q` and full-vector
length `N`. A proper complex LS error variance `v` becomes `v/2` on both the
real and imaginary noise-covariance diagonals. The component gains,
likelihood-based responsibilities, and weighted conditional mean then implement
equations (13), (15), and (16), respectively.

## Compilation

GMM fitting, manifest validation, and NPZ loading are confined to `asset.py`
and complete before estimator construction. With prepared GMM tensors, both
the spatial and spectral estimator calls compile with
`torch.compile(..., backend="eager", fullgraph=True)` and match eager outputs.
`torch._dynamo.explain` reports one graph and no graph breaks for each method,
so both online conditional-mean paths are classified Tier A.

## Modifications And Omissions

The following modifications have been made compared to the full paper method:

- `SpatialGMMChannelEstimator` is pilot-domain only and uses `A = I` because
  Sionna has already converted received pilots into LS pilot-domain channel
  observations.
- `SpectralGMMChannelEstimator` implements the spectral full-grid observation
  model with the benchmark pilot subcarriers as the observed coordinates, but
  it still models one receive-antenna spectral vector at a time rather than the
  complete spatial-frequency MIMO resource grid.
- The spectral estimator uses one pooled GMM over full effective-subcarrier
  vectors from all pilot groups and receive antennas, rather than one separate
  GMM per pilot group.
- The spatial estimator uses one pooled GMM over receive-array vectors from
  all nonzero pilot resource elements. It captures array-domain coupling at
  pilot positions but does not model frequency correlation.
- Pooling gives every extracted vector equal weight and assumes a common
  marginal distribution across the pooled groups. Spatial training pools UEs,
  pilot symbols, and pilot subcarriers; spectral training additionally pools
  receive antennas while retaining each complete subcarrier vector. This is a
  documented stationarity/exchangeability adaptation, not a group-specific
  statistical model.
- EM initialization, stopping, and covariance diagonal loading are numerical
  implementation choices because the paper specifies EM but not these local
  details.
- The paper defines the conditional mean, but Sionna's detector also needs
  channel-estimation error variances. The implementation returns the posterior
  variances implied by the same GMM model.
- The spatial LMMSE gain is evaluated as `C(C+R)^-1`. Earlier code used the
  reversed product, which is equivalent for the benchmark's scalar noise but
  not for a general diagonal per-antenna noise covariance.

The following paper variants are not implemented here:

- The circulant-covariance simplification is not implemented because the
  benchmark uses a rectangular BS array and the current vectorizations do not
  directly match the paper's ULA covariance assumption.
- The Kronecker-covariance simplification is not implemented because the
  current implementation does not model a full transmit-receive MIMO channel
  vector.
- A full spatial-frequency GMM over the complete MIMO resource-grid channel is
  not implemented. It would require much larger assets and a reviewed
  vectorization/modeling choice.

## Asset Generation

The canonical GMMs were selected under calibrated Sionna commit
`f16208347246d8d17ec69628dfb856b19287b80c` with UMi TR 38.901 version
19.2. One deterministic bank of 100 batches of 64 independent complete
four-UE channel realizations (6,400 channels) was generated with seed `22022`
and reused for every fit.
It yields 7,987,200 spatial vectors of length 32 and 819,200 spectral vectors
of length 1,248. Its manifest SHA-256 is
`02d6593b6f0aa736ee28b3edea2e44e28699173e4140565e76f578ea372170d1`.

The retained fitting payload records are:

| File | Size (bytes) | SHA-256 |
| :- | -: | :- |
| `pilot_symbol_channels.npy` | 4,089,446,528 | `6f8d127941e85656991f81a6d05a30c7432854ca398c0abd3552fef2664563c8` |
| `spatial_vectors.npy` | 1,022,361,728 | `df11514a935f56cc897e45413a89be8e59ec98329ffb296635d3e486c8d7ac7b` |
| `spectral_vectors.npy` | 4,089,446,528 | `407945f7952e3aeb5579b61a65f54bc27f4625045a809cfdabee3315b286cfe5` |

Generate the shared fitting bank once:

```bash
python -m sionna.bench.mimo_ofdm_chest.papers.koller2022.tuning \
  --prepare-fitting-data \
  --asset-root "$ASSET_ROOT"
```

The printed `manifest.json` path is supplied as `FITTING_DATA_MANIFEST` to
both exhaustive component-count studies:

```bash
python -m sionna.bench.mimo_ofdm_chest.papers.koller2022.tuning \
  --method spectral \
  --asset-root "$ASSET_ROOT" \
  --fitting-data-manifest "$FITTING_DATA_MANIFEST" \
  --output-dir "$SPECTRAL_STUDY_DIR"

python -m sionna.bench.mimo_ofdm_chest.papers.koller2022.tuning \
  --method spatial \
  --asset-root "$ASSET_ROOT" \
  --fitting-data-manifest "$FITTING_DATA_MANIFEST" \
  --output-dir "$SPATIAL_STUDY_DIR"
```

Both studies exhaustively evaluated `K in {5, 10, 15, 20}` with 15 EM
iterations and covariance regularization `0.001`. Selection used weighted
NMSE at `{-10, -5, 0, 5, 10}` dB, three batches of four channels per SNR,
exact weights `[0.1, 0.31622776601683794, 1.0, 3.1622776601683795, 10.0]`,
`GridSampler` with seed `123`, and probe seed `4242`. Each method ran four
requested trials: four completed, zero failed, and zero were pruned. Because
this is an exhaustive four-point choice with no continuous parameter to round,
rounded-candidate paired validation is inapplicable; no separate holdout was
used.

The spectral objectives for increasing `K` were `0.00183402`, `0.00168584`,
`0.00157878`, and `0.00161731`, selecting `K=15`. The spatial objectives were
`0.00475302`, `0.00443746`, `0.00441224`, and `0.00432931`, selecting `K=20`.
The selected manifest SHA-256 values are respectively
`07cc19084a9e294e182fd0705429fcafccda7bfaeb4a27006ed0122e4fcaad92`
and `d7be491c1caa51c302b1e68743145254e40276277c9b00a96ff6a90c2cd2f2a2`.

The selected spatial payload is `gmm_params.npz` (85,310 bytes, SHA-256
`668bb025823d9043795cb60cf1469620a3ac0aa8560f292fce85d1703d87fecd`).
The selected spectral payload comprises `gmm_params.npz` (75,450 bytes,
SHA-256 `c5fd94e72ac6a81e8f9c8073197266ba9b04f22355672e90f804353d89bfb637`)
and fifteen 6,230,144-byte covariance files. Their ordered SHA-256 values are:

```text
640b9a7613de89cf1879115d9915ef05845d90a589444a7612cfdafa3bd6abe3
feb62873539a1036c01a1a0dbab2853020bbacac77e048a3a2129a22ebaf47fb
23cf8cc0eca004ff06498385e60c426bd655443c6ab59f8074e95cf84dedb742
e89a4fbd15497745c047cbc1fd39bf336f8e5fc5cbd199760cc012a64ce264d1
260a201c42b0b04a05d4180e938a1e0a5329903d6c7eda919a7b10d02a54c859
81eb7e43f1b469903932cc6436f6ff4e3d120b0897aa9563aa3bc04a7cf9400a
e1384b40a57703c135c88a7759008c3a49028d0877bebac541b2c3057eb8dc04
09018a354f007faab36ddf95f8affca700ea2db2c365ffd20d94dc1cd2d28ec9
ac75dddf17e376b0655b5570eef675d72331b228828bf1438c414828d1ebaca2
d4791f6ba46a3e06c05685839f444df8e3e3c0fd0f928756c5dfa503ede59256
887d170fb7f90f1073ae8e32b65ca5ac7323d7785f5284aea5c1499df427ad1c
f81efeb84d1cee2f743d4848c33fdde190aa9a755b89e2412e0e9f97c9fdd03b
7cc0d8a181de481cd2d088446a584fb61430f81167cf87e1fc2c41ac91b3ab27
f15b5fd5a8d532971b0021d7f7c1f9a8c6fd4f1ff1329bb4ee6ce9c3d363a6b6
158d8007007c182ec7aa8352c83b70fa7189ac03e116091a97ee02ec2dd6cd4e
```

Accepted stage runtimes on an NVIDIA L40S were 57.15 seconds for data
generation, 224.09 seconds for the spectral study, and 4,705 seconds for the
spatial study.

## Canonical Benchmark Run

After generating the K15 spectral and K20 spatial assets from the shared
6,400-channel bank with seed `22022`, evaluate them from the same external
root:

```bash
python -m sionna.bench.mimo_ofdm_chest.papers.koller2022.run_benchmark \
  --method spectral --asset-root "$ASSET_ROOT"
python -m sionna.bench.mimo_ofdm_chest.papers.koller2022.run_benchmark \
  --method spatial --asset-root "$ASSET_ROOT"
```

The spectral and spatial benchmarks took 705.39 and 42.50 seconds. Their
weighted NMSE values were `0.00158260` and `0.00423842`, respectively, versus
`0.00174428` and `0.00429603` in the calibrated comparison worktree. The
historical outputs contain no confidence intervals, so these differences are
reported as point-estimate comparisons rather than formal significance
claims. Benchmark results were not used for component-count selection.

The retained preflight records Sionna Bench commit
`5012974748d4b8aa6426e563d8fed1380eba3abc` and tracked-diff SHA-256
`03b172aa0890b355ee413c788d448c54319aacc7eda53d502683a3e9395c27e2`.
The environment was Python 3.12.3, PyTorch `2.11.0+cu126`, CUDA 12.6,
driver `580.126.09`, NumPy 2.4.4, SciPy 1.17.1, and Optuna 4.9.0 on an
NVIDIA L40S and AMD EPYC 7R13. The retained `pip-freeze.txt` has SHA-256
`b981919b35253baa1f4cbc244763e09292be6a503d42480d3f9003e397884082`.
Canonical evaluation used seed `2026`.
