# Implementation of the Estimators from Edfors et al., 1996

Publication: O. Edfors, M. Sandell, J.-J. van de Beek, S. K. Wilson, and P. O. Börjesson, "Analysis of DFT-based channel estimators for OFDM," Research Report, 1996.

## Abstract

This implementation covers the delay-truncated DFT-domain channel estimators
from Edfors et al., 1996. The paper assumes known transmitted symbols on all
OFDM subcarriers and analyzes low-rank filtering in the IDFT domain. The
benchmark exposes only pilot-subcarrier observations, so the implementation
applies the same delay-domain truncation and filtering ideas to each compact
pilot-subcarrier vector.

The estimator first forms LS pilot estimates, maps each pilot-subcarrier vector
to the delay domain with a unitary IDFT, keeps the strongest delay taps, applies
one of the paper's selected-tap filters, zeroes excluded taps, and maps back to
the pilot-subcarrier domain with a unitary DFT. Full resource-grid
reconstruction is then delegated to a configured Sionna interpolator. The
canonical default is linear frequency interpolation with OFDM-time averaging.

Three variants are implemented:

- `LMMSE`: full selected-tap LMMSE filtering, corresponding to Estimator A;
- `diag. LMMSE`: diagonal selected-tap LMMSE filtering, corresponding to
  Estimator B;
- `LS`: delay truncation without LMMSE shrinkage, corresponding to Estimator C.

## Implemented Classes

### `TruncatedChannelEstimator`

`TruncatedChannelEstimator` is a Sionna `BasePilotChannelEstimator`. It uses
Sionna's internal `_PilotGroupMixin` to group nonzero pilot subcarriers by
transmit stream and pilot-bearing OFDM symbol, applies a DFT-domain Edfors
method to each group, and uses Sionna's configured interpolator to reconstruct
the resulting pilot-domain estimates over the full resource grid. Its standard
`interpolation_type` and `interpolator` arguments follow
`BasePilotChannelEstimator`; the canonical default is `lin_time_avg`.

Its main methods are:

- `__init__`: validates the selected method and tap count, initializes pilot
  grouping, accepts an injected empirical delay-domain covariance matrix,
  selects the strongest delay taps, and registers the covariance and tap order
  as buffers.
- `_selected_tap_indices`: returns the strongest delay-tap indices retained by
  the estimator.
- `_weight`: builds the selected-tap filtering matrix. For `LMMSE`, this is the
  full Appendix A Estimator A LMMSE matrix. For `diagonal LMMSE`, this is the
  scalar shrinkage from Appendix A Estimator B. For `LS`, this is the identity
  on the selected taps.
- `estimate_at_pilot_locations`: computes Sionna LS pilot estimates, applies
  the unitary IDFT, selected-tap filtering, tap zeroing, and unitary DFT for
  each pilot group, then returns the filtered pilot-domain estimates and the
  scaled LS-derived detector-confidence fallback.

Benchmark outputs are named:

- `TruncatedChannelEstimator (LMMSE)`;
- `TruncatedChannelEstimator (diag. LMMSE)`;
- `TruncatedChannelEstimator (LS)`.

## Supported System Geometry

The estimator groups active pilot resource elements by fixed transmit index,
stream index, and OFDM symbol, with subcarrier varying inside each group. It
supports different resource-grid sizes, pilot-bearing OFDM-symbol positions,
UE counts, and BS antenna counts under the following conditions:

- at least one nonzero pilot group exists;
- every active `(tx, stream, pilot OFDM symbol)` group contains the same number
  of nonzero pilot subcarriers;
- `num_taps` is between one and that common group length; and
- the injected delay covariance is a square matrix whose dimension equals the
  common group length.

Ragged active pilot groups are rejected during construction with an explicit
equal-length-group error. A persisted covariance asset is also accepted only
when its recorded ordered pilot coordinates and receive-array geometry match
the active system configuration. Receive antennas are otherwise processed as
independent leading dimensions by the online estimator.

## Auxiliary Covariance Asset

The LMMSE variants require delay-domain second-order statistics. Their complete
lifecycle is handled by `src/sionna/bench/mimo_ofdm_chest/papers/edfors1996/asset.py`.
`estimate_covariance()` generates an asset only beneath an explicit external
asset root. Normal benchmark execution calls strict `load_covariance()`, which
never samples channels or writes files. Both functions return the covariance
matrix and selected manifest path; the runner injects the matrix into
`TruncatedChannelEstimator` and records the manifest for every method. The
estimator performs no file access or covariance estimation.

The covariance helper samples true channels from the active benchmark channel,
gathers the same pilot subcarriers used by the estimator, maps the
pilot-subcarrier vectors to the delay domain with the same unitary IDFT, and
accumulates one pooled empirical covariance matrix. Public distributions omit
the selected matrix. Regeneration writes it beneath
`<asset_root>/<scenario_id>/edfors1996/delay-covariance/...`; an internal
distribution may optionally provide the same complete artifact under the
paper's package-local `assets/` tree. Each artifact records the covariance
budget, seed, pilot group layout, and IDFT normalization.

The current implementation uses one shared covariance matrix for all transmit
streams and pilot-bearing OFDM symbols. This matches the benchmark assumption
that the relevant delay-domain second-order statistics should not depend on
stream index or pilot symbol under the sampled channel law.

The canonical covariance was generated in an empty staging asset root with:

```bash
python -m sionna.bench.mimo_ofdm_chest.papers.edfors1996.run_benchmark \
  --prepare-assets \
  --asset-root "${RUN_ROOT}/assets" \
  --cov-num-iters 1000 \
  --cov-batch-size 64
```

The preparation command generates the asset, strictly reloads it for
validation, and exits without writing benchmark results. Evaluate the prepared
asset by passing the same root:

```bash
python -m sionna.bench.mimo_ofdm_chest.papers.edfors1996.run_benchmark \
  --asset-root "${RUN_ROOT}/assets"
```

This recipe samples 64,000 independent complete four-UE channel realizations
and pools the delay-domain statistics over all pilot groups. With 16 receive
antennas, four UEs, fourteen OFDM symbols, and eight groups, the implemented
reshape contributes `64,000 * 16 * 4 * 14 * 8 = 458,752,000` delay-domain
vectors to the shared covariance. It ran in 79.34 seconds on an NVIDIA L40S
with Sionna source commit
`f16208347246d8d17ec69628dfb856b19287b80c` and the explicit UMi TR 38.901
v19.2 channel model. The selected manifest has scenario `3gpp-umi`, asset type
`delay-covariance`, artifact ID `it1000-bs64-seed42`, and SHA-256
`88cce34a7c00fb023410ba4aea3fe2df16919a2adcb3ffbf7618dfd484c5955e`.
Its payload is
`time_covariances.npz` (195,850 bytes, SHA-256
`eea2a270eb004f475b3a6753efde8845ddee88bd3802131fbf3e52a0d4eb426f`).
The covariance and tap-power arrays have shapes `[156, 156]` (`complex64`) and
`[156]` (`float32`), respectively.

Asset generation resets Python, NumPy, Torch, CUDA, and Sionna's explicit RNGs
to seed `42`. Resetting Sionna's RNG is required for reproducible Sionna 2.0.1
channel and topology samples.

## Modifications And Omissions

The following modifications have been made compared to the paper's literal
receiver:

- The paper assumes an LS vector over all OFDM subcarriers. The benchmark
  provides only sparse pilot-subcarrier observations, so the implementation
  applies the DFT-domain estimators to compact pilot-subcarrier vectors.
- The paper does not define how sparse-pilot estimates should be reconstructed
  over the whole time-frequency resource grid. The implementation exposes
  Sionna's standard interpolation interface and uses
  `interpolation_type="lin_time_avg"` by default, matching the common policy
  for repository paper estimators. This policy is a Sionna adaptation rather
  than a choice specified by Edfors et al.
- The paper derives or assumes covariance from its analytical channel model.
  This implementation estimates a matching empirical covariance from the active
  Sionna benchmark channel in an explicit external asset root. The public
  distribution records its recipe and identity but does not include the
  covariance payload.
- If `num_taps=None` is passed to the estimator class, the number of retained
  taps falls back to the configured cyclic-prefix length, capped by the
  pilot-vector length. The saved benchmark results do not use this fallback:
  they use `num_taps=64`, selected by the LS-only Optuna tuning procedure
  described below and then reused for all variants. The all-tap setting remains
  available through `num_taps`.
- The paper derives estimator MSEs, but it does not define the per-resource
  detector-facing variance required after this implementation's sparse-pilot
  adaptation and full-grid interpolation. The implementation therefore uses
  `chest_err_const*no/|pilot|^2` at pilot positions and lets Sionna linearly
  interpolate those confidence values. This is not the posterior variance of
  Estimators A, B, or C. A shared aggregate detector-facing calibration across
  all three methods selected the three-significant-digit value
  `chest_err_const=0.308`.

The following paper setting is not implemented as a separate benchmark method:

- The literal all-subcarrier known-symbol receiver analyzed in the paper is
  incompatible with the current pilot-pattern channel-estimation benchmark
  unless the harness adds an all-pilot training-symbol mode or another
  compatible receiver contract.

## Compilation

Covariance loading and estimation happen in `asset.py` before estimator
construction and are not reachable from online inference. With the covariance
injected as a registered buffer, all three complete estimator methods compile
with `torch.compile(..., fullgraph=True)`. On the alternate test geometry,
`torch._dynamo.explain` reports one graph and no graph breaks, classifying the
online estimator as Tier A.

## Hyperparameter Tuning

The canonical benchmark uses one shared tap count for all three variants. This
shared value was selected with Optuna by minimizing the weighted NMSE of the
`LS` variant only. The selected value is then reused unchanged for `LMMSE` and
`diag. LMMSE`, so the LMMSE variants do not receive their own separately tuned
tap count.

The v2 objective used Eb/N0 points `(-10, -5, 0, 5, 10)` dB and linear-SNR
weights `10**(Eb/N0/10)`. Each trial reused three cached batches of four
channels at each SNR, for 60 SNR/channel pairs. The 50-trial TPE study used
sampler seed `123` and probe seed `4242`; it did not inject a historical
configuration. It nominated `num_taps=61` with weighted NMSE
`0.008155232729752004`. Independent paired validation used seed `314159` and
63 batches of four channels per SNR. The 61-tap candidate scored
`0.009036960406574467`, while the current 64-tap setting scored
`0.00889008178325196`. The canonical runner therefore retains
`num_taps=64` for all three variants. Search and validation took 15.88 and
23.09 seconds, respectively, on an NVIDIA L40S.

The tuning summary and trial history are stored in:

```text
tuning/mimo_ofdm_chest/edfors1996/taps_v2/best_params.json
tuning/mimo_ofdm_chest/edfors1996/taps_v2/paired_validation.json
tuning/mimo_ofdm_chest/edfors1996/taps_v2/trials.csv
```

Run the tap-count study with:

```bash
python -m sionna.bench.mimo_ofdm_chest.papers.edfors1996.tuning \
  --study taps \
  --asset-root "${RUN_ROOT}/assets" \
  --output-dir tuning/mimo_ofdm_chest/edfors1996/taps_v2
python -m sionna.bench.mimo_ofdm_chest.papers.edfors1996.tuning \
  --study taps-validation \
  --asset-root "${RUN_ROOT}/assets" \
  --output-dir tuning/mimo_ofdm_chest/edfors1996/taps_v2
```

### Detector-Facing Error Variance

The shared `chest_err_const` changes only the detector-facing error variance;
it cannot change `h_hat` or NMSE. After the 64-tap algorithm was frozen, one
aggregate 10-trial TPE study searched `[1e-3, 1]` logarithmically. It minimized
the equal-method mean codeword BLER of all three methods at 10 dB, using 63
cached batches of four channels. The sampler seed was `456` and calibration
seed was `271828`; no historical scale was injected.

Optuna selected raw `0.4540521216492649` with mean BLER
`0.03196649029982363`. Independent paired validation used seed `161803` and
another 63 batches of four channels. Rounded `0.454` scored
`0.0333994708994709`, while current `0.308` scored
`0.033289241622574954`; current `0.308` was therefore retained. Search and
validation took 56.45 and 20.40 seconds, respectively, on an NVIDIA L40S.

Run the calibration with:

```bash
python -m sionna.bench.mimo_ofdm_chest.papers.edfors1996.tuning \
  --study variance \
  --asset-root "${RUN_ROOT}/assets" \
  --tap-output-dir tuning/mimo_ofdm_chest/edfors1996/taps_v2 \
  --output-dir tuning/mimo_ofdm_chest/edfors1996/variance_scale_v2
python -m sionna.bench.mimo_ofdm_chest.papers.edfors1996.tuning \
  --study variance-validation \
  --asset-root "${RUN_ROOT}/assets" \
  --tap-output-dir tuning/mimo_ofdm_chest/edfors1996/taps_v2 \
  --output-dir tuning/mimo_ofdm_chest/edfors1996/variance_scale_v2
```

The study summary and trial history are stored in:

```text
tuning/mimo_ofdm_chest/edfors1996/variance_scale_v2/best_params.json
tuning/mimo_ofdm_chest/edfors1996/variance_scale_v2/paired_validation.json
tuning/mimo_ofdm_chest/edfors1996/variance_scale_v2/trials.csv
```

All requested trials completed: tap tuning recorded
`50 requested / 50 complete / 0 failed / 0 pruned`, and variance calibration
recorded `10 requested / 10 complete / 0 failed / 0 pruned`. The studies used
Sionna Bench commit `5012974748d4b8aa6426e563d8fed1380eba3abc` with
tracked-diff SHA-256
`a65022bbb9df7a64f16a77a0ace028ef4cf10ba84ff7766bdbf051d0f7457549`.
The host used an AMD EPYC 7R13 CPU, NVIDIA L40S GPU, driver 580.126.09,
Python 3.12.3, PyTorch 2.11.0+cu126, CUDA 12.6, Sionna 2.0.1, Optuna 4.9.0,
NumPy 2.4.4, and SciPy 1.17.1. The shared retained dependency snapshot is
`tuning/mimo_ofdm_chest/evidence/environments/v2-l40s-20260814/pip-freeze.txt`, SHA-256
`b981919b35253baa1f4cbc244763e09292be6a503d42480d3f9003e397884082`.

## Benchmark Run

The saved benchmark results were generated with the default 3GPP UMi
configuration:

```bash
python -m sionna.bench.mimo_ofdm_chest.papers.edfors1996.run_benchmark \
  --asset-root "${RUN_ROOT}/assets"
```

No CLI overrides were used. The run used `configs.umi_3gpp_config`, i.e. the
Eb/N0 grid from `-10` dB to `15` dB in `2.5` dB steps, `batch_size=4`,
`num_mc_batches=63`, four UEs, a `4x4` BS antenna array, 52 PRBs, and pilot-bearing
OFDM symbols `(2, 11)`. The canonical evaluation seed was `2026`.

The runner evaluated all three variants:

- `method="LMMSE"`;
- `method="diagonal LMMSE"`;
- `method="LS"`.

The saved run used the runner default `num_taps=64`, which is the LS-optimized
tap count reused for all three variants. It used the empirical covariance
asset generated with the explicit runner defaults `cov_num_iters=1000`,
`cov_batch_size=64`, and the configured setup seed. These values identify the
asset generation recipe and are not fields of `SystemConfig`. The run used the
aggregate-calibrated task-contract fallback scale `chest_err_const=0.308`.

Results were written to
`outputs/mimo_ofdm_chest/edfors1996/3gpp-umi/`.

The three-method canonical benchmark took 124.71 seconds on the L40S. Relative
to the calibrated comparison worktree, median NMSE changed by less than 0.7%
for every method. Every new NMSE, BER, and BLER point lies inside the
corresponding comparison 95% confidence interval.
