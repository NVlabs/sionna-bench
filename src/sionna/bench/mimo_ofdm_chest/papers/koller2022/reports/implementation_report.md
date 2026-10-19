# Implementation Report

- Paper ID: `koller2022`
- Implementer: Codex
- Date: 2026-04-29
- Implemented methods: `SpectralGMMChannelEstimator`,
  `SpatialGMMChannelEstimator`
- Revision note: updated on 2026-07-09 to separate offline GMM fitting and
  persistence into `asset.py`. The runner and tuner now prepare GMM parameters
  outside inference-only estimator constructors. Updated on 2026-07-10 after
  final asset-loader and unsupported-geometry validation. Updated on 2026-08-04
  so normal benchmark runs use strict `load_model()` while explicit asset
  preparation uses `fit_model()`. Updated on 2026-08-18 with the final
  calibrated-Sionna K15/K20 exhaustive selections, canonical manifests,
  canonical outputs, and strict result-linkage verification.

## Code Delivered

- `src/sionna_bench/mimo_ofdm_chest/papers/koller2022/estimator.py`
  - Adds the pilot-domain `SpatialGMMChannelEstimator` and full-subcarrier
    `SpectralGMMChannelEstimator`.
  - Accepts prepared GMM weights, means, and covariances and registers them as
    buffers without performing fitting or file I/O.
  - Exposes Sionna's standard `interpolation_type` and custom `interpolator`
    interface for the pilot-domain spatial estimator, with `lin_time_avg` as
    the canonical default.
  - Applies the paper's GMM conditional mean from equations (13), (15), and
    (16) to the selected spatial or spectral vectors.
- `src/sionna_bench/mimo_ofdm_chest/papers/koller2022/asset.py`
  - Exposes strict `load_model()` for normal benchmark execution and
    external-only `fit_model()` for explicit preparation. The preparation
    path derives training groups with `_PilotGroupMixin`, pools channel samples,
    fits a full-covariance GMM by EM, and loads/saves reusable assets.
  - Uses global asset-manifest schema 2 for both GMM types. Spatial GMMs retain
    the compact single-NPZ payload; spectral GMMs store weights and means in
    `gmm_params.npz` and one full covariance per component in an ordered NPY
    payload, then reconstruct the same covariance tensor at load time.
- `src/sionna_bench/mimo_ofdm_chest/papers/koller2022/run_benchmark.py`
  - Exposes `run_benchmark()`, strictly loads and injects the selected GMM for
    canonical evaluation, and calls `evaluation.evaluate_channel_estimator`.
    `--prepare-assets` explicitly fits the requested asset and exits
    before evaluation.
  - Adds `--method spectral|spatial` and saves results under
    `outputs/koller2022/`.
- `src/sionna_bench/mimo_ofdm_chest/papers/koller2022/tuning.py`
  - Exposes `run_tuning()`, strictly reuses an existing trial-specific GMM, or
    explicitly calls `fit_model()` before the shared Optuna probe.

## Deviations From Paper Or Plan

- The spatial method is the reviewed pilot-domain adaptation. The spectral
  method reconstructs full effective-subcarrier vectors but does not model the
  complete spatial-frequency MIMO channel jointly.
- The spectral method fits one GMM over pooled full-subcarrier vectors from
  all pilot OFDM symbol groups and receive-antenna links. This preserves
  equation (16) for the modeled vector but assumes the same spectral
  statistics across pilot groups.
- The spatial method fits one GMM over pooled 16-antenna vectors from all
  nonzero pilot resource elements. It captures receive-array coupling but
  assumes the same spatial statistics across pilot groups and intentionally
  does not model frequency correlation.
- Spectral smoke defaults use `K=2`, 4 training channel batches, train batch size 8, and 6 EM iterations. Spatial smoke defaults use `K=2`, 8 training channel batches, train batch size 16, and 6 EM iterations because the spatial vector has real dimension 32. These settings are runtime-oriented and are not the paper's high-sample numerical setup.
- EM uses deterministic sample initialization and covariance diagonal loading `1e-3 * mean(sample variance)`, because the paper names EM but does not specify those numerical details.
- The returned `err_var` is the GMM posterior variance implied by the same mixture model. The paper only defines the conditional mean; `err_var` is required by the benchmark detector.
- The GMM posterior update uses the LS error variances as a diagonal
  real-valued noise covariance. Spatial groups share the noise-dependent terms
  because equal pilot power makes their LS noise covariance identical;
  spectral groups compute terms from their own selected subcarriers and LS
  variances.

## Auxiliary Assets

- The canonical spectral asset is
  `src/sionna_bench/mimo_ofdm_chest/papers/koller2022/assets/3gpp-umi/spectral-gmm/k15-tb100-bs64-em15-reg0p001-seed22022/`.
  Its arrays have shapes `weights=(15,)`, `means=(15, 1248)`, and
  `covs=(15, 1248, 1248)`.
- The canonical spatial asset is
  `src/sionna_bench/mimo_ofdm_chest/papers/koller2022/assets/3gpp-umi/spatial-gmm/k20-tb100-bs64-em15-reg0p001-seed22022/`.
  Its arrays have shapes `weights=(20,)`, `means=(20, 32)`, and
  `covs=(20, 32, 32)`.
- Both models were fitted from one deterministic 6,400-channel bank generated
  with seed `22022`. Their selected manifest SHA-256 values are
  `07cc19084a9e294e182fd0705429fcafccda7bfaeb4a27006ed0122e4fcaad92`
  and `d7be491c1caa51c302b1e68743145254e40276277c9b00a96ff6a90c2cd2f2a2`.
- GMM compatibility now records the estimator mode, real-then-imaginary vector
  convention, vector axis and dimensions, and exact ordered pilot-group
  coordinates, values, and powers. Canonical manifests were migrated without
  changing either GMM payload.
- Full-spectral assets use schema 2 so no single covariance payload grows
  with the number of mixture components. This changes persistence only; the
  fitted component parameters and online conditional-mean equations are
  unchanged.

## Verification Commands

```text
python -m py_compile src/sionna_bench/mimo_ofdm_chest/papers/koller2022/estimator.py src/sionna_bench/mimo_ofdm_chest/papers/koller2022/asset.py src/sionna_bench/mimo_ofdm_chest/papers/koller2022/run_benchmark.py src/sionna_bench/mimo_ofdm_chest/papers/koller2022/tuning.py
```

Result: passed.

Current refactor checks:

```text
python -m compileall src/sionna_bench/mimo_ofdm_chest/papers/koller2022/estimator.py
python -m pytest test/mimo_ofdm_chest/infrastructure/test_pilot_groups.py
python -m pytest test/mimo_ofdm_chest/papers/koller2022/test_asset.py
python -m pytest test/mimo_ofdm_chest/papers/koller2022/test_estimator.py
```

Result: passed. Tiny temporary spectral and spatial runner evaluations with
one-component, one-batch pooled GMM assets also passed through
`evaluate_channel_estimator`; their temporary assets and outputs were removed.
Both complete estimator calls also match eager execution with
`torch.compile(..., backend="eager", fullgraph=True)`. Diagnostic
`torch._dynamo.explain` runs report one graph and no graph breaks for each
method, establishing Tier A online inference. GMM fitting and cache I/O occur
only before estimator construction and are not reachable from either call.
On the alternate three-UE, two-pilot-symbol test geometry, the stacked spectral
group assembly also matches the former coordinate-indexed assembly for both
channel estimates and posterior variances.
Tiny one- and two-component GMM fitting also passes for both methods on the
baseline and alternate shared system geometries. These checks use temporary
asset roots, validate parameter shapes and dtypes, and confirm that a second
load does not refit the model.
Both loaders reject assets with changed ordered pilot coordinates. The spatial
loader additionally rejects a changed receive-vector dimension, and the
spectral loader rejects a changed full-frequency vector dimension. Consistent
temporary manifests also exercise missing NPZ keys or spectral covariance
payloads and incorrect shapes and dtypes for every weight, mean, and
covariance array.
New spatial and spectral manifests omit the legacy `key_fields` entry. Exact
compatibility is checked through the common schema-2 manifest envelope, and
the requested fitting recipe is compared directly with `generation_recipe`.
The schema-2 writer and loader were also exercised transiently on the preserved
historical K17 full-spectral payload. All arrays round-tripped exactly; its former
101.09 MiB archive became one parameter archive and 17 covariance files, with
each covariance file 5.94 MiB and no change to the total tensor contents.
A transient wheel build containing representative schema-2 files also
included the manifest, parameter NPZ, and nested covariance NPY at the expected
installed-package path. The final asset-bearing wheel remains subject to the
repository-wide clean-clone audit after the selected GMM has been regenerated.
A transient CPU smoke through `run_benchmark()` then fitted a two-component
schema-2 spectral asset, executed full system inference, and wrote a result
manifest whose asset reference matched all three authoritative payload sizes
and SHA-256 checksums. The run completed in about 24 seconds with finite NMSE,
BER, and BLER; its temporary asset and output roots were outside the repository.
Constructor checks verify the documented unsupported cases: spatial joint
processing rejects unequal pilot powers, and spectral reconstruction rejects
different pilot-bearing OFDM-symbol counts between streams.
Both estimators additionally pass system-level UMi probes on the baseline,
alternate, and wide test configurations, covering 1/3/4 UEs and 2/4/12 BS
antennas. Every case satisfies the estimator output contract and has lower
NMSE than a zero channel estimate at 5 dB.
Independent double-precision checks compare both implementations with direct
Gaussian-conditioning formulas. They verify real/complex stacking, component
likelihoods, responsibilities, posterior means, and non-negative posterior
variances. The nonuniform-noise spatial check exposed a reversed LMMSE gain
product; it was corrected from `(C+R)^-1 C` to `C(C+R)^-1`, together with the
corresponding posterior-covariance expression. This does not change canonical
scalar-noise results because `R` is then proportional to the identity.
An equation-level reread confirmed that spatial LS normalization gives the
paper's model with `A=I`, while spectral inference uses the active pilot
subcarriers as a selection matrix. In real-stacked coordinates, complex LS
variance `v` contributes `v/2` to each real and imaginary diagonal. The code's
component gain, Gaussian responsibility, and mixture mean therefore implement
equations (13), (15), and (16). Spectral analytical tests cover all three UE
comb offsets of the alternate geometry. Both fitted priors deliberately pool
all applicable groups with equal sample weight and reuse the resulting GMM for
every group; this common-statistics assumption is not mandated by the paper
and remains an explicit benchmark adaptation.

```text
python -m sionna_bench.mimo_ofdm_chest.papers.koller2022.run_benchmark --smoke --method spectral --prepare-assets --asset-root /tmp/koller-assets
python -m sionna_bench.mimo_ofdm_chest.papers.koller2022.run_benchmark --smoke --method spectral --asset-root /tmp/koller-assets
```

Result: passed with the current `evaluate_channel_estimator` runner. The
current runner persists only the selected Koller paper result.

Smoke metrics at 5 dB:

| Estimator | NMSE | BER | BLER |
|---|---:|---:|---:|
| `SpectralGMMChannelEstimator` | 1.000382e-02 | 9.903311e-02 | 3.750000e-01 |

Spatial smoke command:

```text
python -m sionna_bench.mimo_ofdm_chest.papers.koller2022.run_benchmark --smoke --method spatial --prepare-assets --asset-root /tmp/koller-assets
python -m sionna_bench.mimo_ofdm_chest.papers.koller2022.run_benchmark --smoke --method spatial --asset-root /tmp/koller-assets
```

Result: passed.

Spatial smoke metrics at 5 dB:

| Estimator | NMSE | BER | BLER |
|---|---:|---:|---:|
| `SpatialGMMChannelEstimator` | 8.417397e-03 | 9.984697e-02 | 3.750000e-01 |

```text
python - <<'PY'
... finite metric assertion over smoke paper npz files ...
PY
```

Result: passed; saved NMSE, BER, and BLER arrays are finite.

Cached smoke asset check:

```text
python - <<'PY'
... inject prepared spectral GMM parameters into
SpectralGMMChannelEstimator and run one NMSE batch ...
PY
```

Result: passed. The estimator loaded the pooled smoke GMM asset and returned
finite full-grid channel estimates with shape `[1, 1, 16, 4, 1, 14, 624]`.

Final canonical benchmark commands:

```text
python -m sionna_bench.mimo_ofdm_chest.papers.koller2022.run_benchmark --method spectral
python -m sionna_bench.mimo_ofdm_chest.papers.koller2022.run_benchmark --method spatial
python leaderboard/generate_data.py
```

Result: passed with the selected K15 spectral and K20 spatial assets. The
scenario-local results and exact manifest references are saved under
`outputs/koller2022/3gpp-umi/`. A direct comparison of vectorized spatial
processing against the previous group loop remains parametrized over multiple
antenna, group, and batch shapes.

## Final Calibrated-Sionna Selection

One deterministic 6,400-channel fitting bank generated with seed `22022` was
reused for all spatial and spectral candidates. Both methods exhaustively
evaluated K5, K10, K15, and K20 with 15 EM iterations and covariance
regularization `0.001`. Selection used weighted NMSE at
`{-10, -5, 0, 5, 10}` dB, three batches of four channels per SNR, sampler seed
`123`, and probe seed `4242`. No incumbent was injected and no separate
holdout was used.

The spectral objectives selected K15 at `0.00157878`; the spatial objectives
selected K20 at `0.00432931`. Their selected manifest SHA-256 values are
`07cc19084a9e294e182fd0705429fcafccda7bfaeb4a27006ed0122e4fcaad92`
and `d7be491c1caa51c302b1e68743145254e40276277c9b00a96ff6a90c2cd2f2a2`.
Strict default loading, payload validation, and exact result-manifest linkage
passed for both canonical assets.

The canonical weighted NMSE values are `0.00158260` for spectral and
`0.00423842` for spatial, versus `0.00174428` and `0.00429603` in the
calibrated comparison worktree. Historical results lack confidence intervals,
so these are point-estimate comparisons only. The benchmark was not used to
revise component-count selection.

## Sanity Assessment

- Shapes match the task contract.
- `h_hat` and `err_var` are finite in the constructor/call check and saved smoke run.
- Saved smoke and full benchmark metrics for both implemented methods are
  finite.

## Remaining Risks

- A joint spatial-frequency GMM is not implemented.
- Larger full-covariance GMM training is computationally expensive with the
  current NumPy EM implementation.
- The spectral method still loops over the small set of pilot-symbol groups
  during inference.
