# Final Report

- Paper ID: `koller2022`
- Reviewer: Codex
- Date: 2026-04-29
- Revision note: updated on 2026-07-10 after separating offline GMM fitting
  and persistence from estimator construction, verifying Tier A inference,
  and completing asset-loader and unsupported-geometry validation. Updated on
  2026-08-06 to audit the finalized Sionna 2.0.1 spatial selection evidence,
  and on 2026-08-07 for the completed spectral study and paired candidate
  benchmark evidence.
- Reviewed artifacts: `paper.pdf`, reports, `estimator.py`, `asset.py`,
  `tuning.py`, `run_benchmark.py`, benchmark outputs, and GMM asset metadata.

## Findings

No blocking findings.

The implementation applies Koller et al.'s GMM conditional mean to
receive-array vectors at pilot positions and to full-subcarrier spectral
vectors observed on pilot subcarriers. It does not claim to model the complete
spatial-frequency MIMO channel jointly.

## Code Audit

- `SpatialGMMChannelEstimator` models receive-array vectors at pilot
  positions, while `SpectralGMMChannelEstimator` models full-subcarrier
  vectors per receive antenna and reconstructs the full frequency grid.
- Both estimators accept and register prepared GMM parameters; neither
  performs repository I/O or offline fitting.
- `asset.py` implements Algorithm 1 line 1 for the selected vectorization and
  exposes external-only `fit_model()` plus strict `load_model()` for reusable
  pooled GMM assets.
- Asset compatibility is explicit about the estimator mode, real-vector
  convention, vector dimensions, and exact ordered pilot coordinates, values,
  and powers; it is not inferred from the directory name or group count.
- Both asset types use global asset-manifest schema 2. Spatial assets use one
  parameter archive; full-spectral assets split component covariances into
  ordered NPY payloads and reconstruct the same float32 covariance tensor on
  load, avoiding an oversized release file without changing the estimator.
- Online estimation uses LS-divided pilot observations, component Gaussian
  likelihoods, component LMMSE estimates, responsibilities, and mixture
  averaging corresponding to equations (13), (15), and (16). The spatial
  method uses `A = I`; the spectral method uses pilot-subcarrier selection.
- The posterior update uses a diagonal real-valued noise covariance derived
  from Sionna's LS error variances. Spatial groups share these terms when pilot
  power is equal; spectral groups use their selected pilot variances.
- The runner prepares and injects the GMM, uses
  `evaluation.evaluate_channel_estimator`, writes the selected paper result,
  and exposes `--method spectral|spatial`.
- GMM fitting and persistent I/O are outside estimator construction and
  inference. Both complete online estimators compile as one full graph and are
  classified Tier A.

## Divergence Ledger

| Divergence | Source | Reviewer verdict | Rationale | Impact on paper faithfulness | Impact on benchmarking fairness |
|---|---|---|---|---|---|
| Spatial pilot-domain `A = I` | interface | justified | The spatial method observes complete receive-array vectors at each pilot resource element. | Limits this method to pilot-domain spatial denoising. | Fair because it is named separately from the spectral method. |
| Sionna `lin_time_avg` interpolation | interface | necessary | Required to return full-grid estimates under the pilot-domain contract. | Does not reproduce Section VI-C full-grid GMM reconstruction. | Can affect NMSE/BER; documented. |
| Pooled spectral GMM over full-subcarrier vectors rather than a joint spatial-frequency GMM | implementation | justified | Keeps equation (16) intact for each modeled vector and uses shared statistics across pilot groups, matching the intended stationary-statistics assumption. | Drops spatial coupling while retaining native frequency reconstruction. | Conservative; likely not an unfair advantage. |
| Pooled spatial-vector GMM over 16-antenna pilot vectors rather than full spatial-frequency GMM | implementation | justified | Applies the same equations to receive-array pilot vectors while sharing statistics across pilot groups. | Captures receive-array coupling only at pilot subcarriers. | Fair if named separately and not overclaimed. |
| EM initialization and covariance loading | implementation | necessary | The paper does not define local EM numerical details. | Minor numerical implementation detail. | Fair; fixed before smoke results. |
| GMM posterior variance used as `err_var` | interface | justified | Derived from the same posterior model and required by the detector. | Adds a quantity not reported by the paper. | Needs future calibration checks. |

## Verification Review

Smoke command:

```text
python -m sionna_bench.mimo_ofdm_chest.papers.koller2022.run_benchmark --smoke --method spectral --prepare-assets --asset-root /tmp/koller-assets
python -m sionna_bench.mimo_ofdm_chest.papers.koller2022.run_benchmark --smoke --method spectral --asset-root /tmp/koller-assets
```

Result: passed.

At 5 dB in the final fresh-asset smoke run,
`SpectralGMMChannelEstimator` achieved NMSE `1.000382e-02`, BER
`9.903311e-02`, and BLER `3.750000e-01`.

Spatial smoke command:

```text
python -m sionna_bench.mimo_ofdm_chest.papers.koller2022.run_benchmark --smoke --method spatial --prepare-assets --asset-root /tmp/koller-assets
python -m sionna_bench.mimo_ofdm_chest.papers.koller2022.run_benchmark --smoke --method spatial --asset-root /tmp/koller-assets
```

Result: passed. At 5 dB in the final fresh-asset smoke run,
`SpatialGMMChannelEstimator` achieved NMSE `8.417397e-03`, BER
`9.984697e-02`, and BLER `3.750000e-01`.

Additional checks:

- `py_compile` passed for the new code.
- `compileall src/sionna_bench/mimo_ofdm_chest/papers/koller2022/estimator.py` and
  `pytest test/mimo_ofdm_chest/infrastructure/test_pilot_groups.py` passed after the `_PilotGroupMixin`
  refactor.
- Tiny temporary spectral and spatial evaluations passed with one-component,
  one-batch pooled GMM assets through the current runner; the temporary assets
  and outputs were removed.
- Strict default loading confirmed that the selected canonical K15 spectral
  and K20 spatial assets match the current group order and estimator geometry.
  Every payload checksum and scenario-local result reference was verified.
- A parametrized eager-versus-compiled test passes for both complete estimator
  calls with `fullgraph=True`; `torch._dynamo.explain` reports one graph and no
  graph breaks for each method.
- Temporary one- and two-component fitting checks pass for spatial and spectral
  GMMs on two reduced noncanonical geometries, including load-without-refit
  verification.
- Both GMM loaders reject changed ordered pilot coordinates, incompatible
  spatial or spectral vector dimensions, missing parameter or covariance
  payloads, and invalid weight, mean, or covariance shapes and dtypes.
- Both loaders operate without legacy `key_fields`; they select assets through
  exact `compatibility` and direct `generation_recipe` validation.
- Spatial construction rejects unequal pilot powers, and spectral construction
  rejects unequal pilot-bearing OFDM-symbol counts between streams, using the
  documented estimator-specific messages.
- Full UMi integration probes pass for both estimators with 1/3/4 UEs and
  2/4/12 BS antennas; all six geometry/method cases satisfy the output contract
  and beat zero-estimate NMSE at 5 dB.
- Independent Gaussian-conditioning checks verify stacking, likelihoods,
  responsibilities, conditional means, and posterior variances for both
  vectorizations. They also identified and corrected the spatial LMMSE gain
  orientation for nonuniform diagonal noise; canonical scalar-noise behavior
  is unchanged.
- Vectorized spatial group processing agrees with a direct group loop for
  several noncanonical batch, receiver, group, and antenna shapes, including
  nonuniform per-antenna noise variances.
- An equation-level audit confirms the spatial `A=I` and spectral selection
  observation models, the `v/2` real/imaginary noise covariance, and the code's
  correspondence to equations (13), (15), and (16). Spectral selection is
  checked for every comb offset in the alternate geometry. Both priors pool
  groups with equal weight and intentionally assume common group statistics.
- Saved smoke metrics are finite.
- The canonical K15/K20 selections were made by exhaustive four-point studies
  on one common 6,400-channel fitting bank before either canonical benchmark.
  The spatial online CME was vectorized over pilot groups before both studies.

## Final Calibrated-Sionna Addendum

The release selection uses K15 spectral and K20 spatial GMMs fitted from one
shared deterministic bank of 6,400 channels. The exhaustive four-point studies
used the same fitting data, 15 EM iterations, covariance regularization
`0.001`, and the five-SNR weighted-NMSE objective. K15 and K20 minimized their
respective objectives at `0.00157878` and `0.00432931`.

Both schema-2 assets have explicit compatibility, generation-recipe, payload,
size, and checksum metadata and load under the canonical runner defaults. The
canonical benchmarks are finite and achieve weighted NMSE `0.00158260` and
`0.00423842`, respectively. Their exact selected manifests are linked from
the scenario-local `results.json`. The historical comparison lacks confidence
intervals, so the reported improvements remain point-estimate evidence rather
than formal significance claims.

## Verdict

Accepted with caveats for benchmark use as separate spatial and spectral GMM
implementations. A joint spatial-frequency GMM remains outside the implemented
scope.
