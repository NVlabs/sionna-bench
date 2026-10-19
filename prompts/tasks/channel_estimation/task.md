# Task Contract: Channel Estimation

The current task is reimplementing academic channel estimators as benchmarkable
Sionna OFDM channel-estimation components.

## Harness Files

- `src/sionna_bench/mimo_ofdm_chest/configs/umi_3gpp_config.py`: full 3GPP UMi configuration
- `src/sionna_bench/mimo_ofdm_chest/configs/smoke_config.py`: lightweight smoke configuration
- `src/sionna_bench/mimo_ofdm_chest/benchmarking/communication_system.py`: executable OFDM system
- `src/sionna_bench/mimo_ofdm_chest/benchmarking/channel_estimation.py`: metrics and result writes
- `src/sionna_bench/mimo_ofdm_chest/benchmarking/result_store.py`: output-directory helpers
- `src/sionna_bench/mimo_ofdm_chest/asset_store.py`: asset paths and manifests
- `src/sionna_bench/mimo_ofdm_chest/papers/builtin/`: built-in Sionna baselines
- `src/sionna_bench/mimo_ofdm_chest/papers/<paper_id>/`: paper reimplementations
- `sionna.phy.ofdm.channel_estimation._PilotGroupMixin`: Sionna's shared,
  developer-facing pilot-grouping helper
- `leaderboard/mimo_ofdm_chest/`: browser-based visual leaderboard

## Benchmark Contract

The benchmark uses a fixed OFDM multi-user channel-estimation pipeline:

- Sionna `ResourceGrid`
- Kronecker pilot pattern
- received grid `y`
- estimator call
- LMMSE detector
- LDPC decode
- BER, BLER, and NMSE metrics

Implemented estimators must return:

```text
h_hat   : [B, num_rx, num_rx_ant, num_tx, num_streams_per_tx,
           num_ofdm_symbols, num_effective_subcarriers] complex
err_var : same shape as h_hat, float
```

The estimator input is:

```text
y  : [B, num_rx, num_rx_ant, num_ofdm_symbols, fft_size] complex
no : scalar or broadcastable noise variance
```

## Required Estimator Architecture

The binding paper-module and dependency contract is defined in
`docs/mimo_ofdm_chest/contracts/paper_architecture.md`. The code
surface is `estimator.py`, optional `asset.py`, optional `tuning.py`, and
`run_benchmark.py`. Additional modules require a concrete reason.

In particular, `estimator.py` must not import `asset.py`. Offline asset lookup,
validation, generation, fitting, and training belong in `asset.py` and are
orchestrated by `run_benchmark.py`. Estimators receive prepared inference
assets through constructor arguments. Online optimization that is itself the
paper's estimation algorithm remains in `estimator.py`.

The implementation plan must choose one of the approved estimator
architectures and justify the choice against the paper.

### Pilot-Domain Estimator With Explicit Interpolation

The default architecture is Sionna's pilot-domain pattern:

```text
pilot observations
-> estimates at pilot locations
-> explicit interpolation/reconstruction
-> full resource-grid channel estimate
```

Use this architecture for papers that:

- only define how to estimate the channel at pilot locations;
- do not consider interpolation or assume pilots on all OFDM subcarriers;
- explicitly use the same estimate-at-pilots followed by interpolation split;
- can be implemented as an estimator at pilot locations followed by a clean
  Sionna or paper-specific interpolator.

Use one of:

- Sionna `BasePilotChannelEstimator`, implementing
  `estimate_at_pilot_locations()` and passing `interpolation_type` or
  `interpolator` explicitly
- Sionna `LSChannelEstimator` with a custom `BaseChannelInterpolator`
- a custom `BaseChannelInterpolator` used by a Sionna estimator

If a paper does not consider interpolation and assumes pilot observations on
all OFDM subcarriers, do not turn that assumption into a
`BaseChannelEstimator`. Implement the estimator at Sionna pilot locations
with `estimate_at_pilot_locations()` and use explicit interpolation afterward.
Document the full-subcarrier pilot assumption as a mismatch between the paper
and benchmark observation model.

### Full-Grid Estimator

Use Sionna's `BaseChannelEstimator` when the paper explicitly assumes pilots
only at selected resource elements and its algorithm natively performs
interpolation, extrapolation, super-resolution, or full-grid reconstruction
from those pilot observations.

A full-grid estimator implements `call(y, no)` directly and returns the
benchmark output tensors. It does not need to expose or accept
`interpolation_type` unless that is meaningful for the paper method. It should
still use Sionna-compatible pilot extraction and `_PilotGroupMixin` whenever
the method needs structured pilot observations.

If the paper reconstructs all subcarriers only on pilot-bearing OFDM symbols
and does not define interpolation across OFDM time, use a documented
time-domain interpolation policy, such as `TimeDomainInterpolator`, for the
remaining OFDM symbols.

If neither approved architecture can implement the method without replacing the
paper's objective, solver, training procedure, or inference algorithm, classify
the method as `not implemented`.

## Interpolation Policy

Interpolation or reconstruction is part of the approved implementation plan.
The implementer must choose it explicitly; relying on Sionna's constructor
default is not allowed.

When the paper only specifies pilot-domain estimation, prefer:

```text
faithful pilot-domain estimator + explicit Sionna built-in interpolation
```

Allowed interpolation choices for the pilot-domain architecture are:

- a paper-specified interpolation/reconstruction implemented as a
  `BaseChannelInterpolator`
- a Sionna built-in interpolation type, such as `"lin"` or `"lin_time_avg"`,
  when the paper only specifies the pilot-domain estimator or when the paper's
  reconstruction is intentionally not implemented in this pass
- `"nn"` only when the paper specifies nearest-neighbor behavior or the
  reviewer approves it as a deliberate benchmark adaptation

For kernel, sparse-reconstruction, compressed-sensing, or learned
reconstruction papers, do not automatically collapse the method to a
pilot-domain estimator with built-in interpolation. If the paper explicitly
models pilot subcarrier sampling and the reconstruction across subcarriers is
part of the algorithm, prefer a reviewed `BaseChannelEstimator`. If the
paper assumes observations on all subcarriers and does not address
interpolation, use the pilot-domain architecture with explicit interpolation
and document the observation-model mismatch.

## Error Variance Policy

Implemented estimators must return:

```text
err_var : same shape as h_hat, float
```

When the paper or implemented statistical model provides channel-estimation
error variances, the implementation derives err_var from that model and
documents any adaptation required to produce Sionna-compatible full-grid
marginal variances.
When no suitable detector-facing variance is available, the default fallback
is a scaled LS observation variance,

```text
err_var_scale * no / |pilot|^2
```

propagated to the full resource grid consistently with the channel estimate.
The empirical scale is exposed as a behavior-relevant constructor argument.
Every empirical detector-facing variance scale is calibrated according to the
Detector-Facing Variance Calibration section of
`docs/mimo_ofdm_chest/contracts/tuning_protocol.md`. Calibration occurs only after the
estimator implementation, algorithmic hyperparameters, interpolation policy,
and inference assets are frozen. Canonical benchmark outputs are not used for
scale selection.
The implementation report and paper README record the unscaled variance
expression, the raw study winner, the rounded candidate, the paired-validation
result, and the final selected constructor value. Estimators whose variances
are derived directly from a posterior or covariance model require no empirical
calibration unless an additional scalar multiplier is introduced.

## Built-In Baselines

The current built-ins are:

- `ls-nn`
- `ls-lin`
- `lmmse`
- `perfect`

## Outputs

Built-in baseline outputs and paper-specific outputs live under the
repository-level output root:

```text
outputs/
  mimo_ofdm_chest/
    builtin/
      <scenario_id>/
        config.json
        results.json
        <result_name>.npz

    <paper_id>/
      <scenario_id>/
        config.json
        results.json
        <result_name>.npz
```

Smoke outputs use a dedicated subtree:

```text
outputs/
  mimo_ofdm_chest/
    builtin/
      smoke/<scenario_id>/
        config.json
        results.json
        <result_name>.npz

    <paper_id>/
      smoke/<scenario_id>/
        config.json
        results.json
        <result_name>.npz
```

Output method names are the leaderboard display names. A method implemented by
one estimator class uses ``EstimatorClass.__name__`` directly. If multiple
algorithmic variants share that class, the runner uses
``EstimatorClass (variant label)`` and keeps any machine-oriented CLI or
constructor token separate from the display label.

The runner must use the exact same display name as the result ``.npz`` filename
stem, returned result-dictionary key, and key under ``results`` in the
scenario-local ``results.json`` file. After the evaluation function has written
the NPZ payload, the runner calls
``sionna_bench.mimo_ofdm_chest.benchmarking.result_store.record_result()`` with that display
name, the concrete estimator class, the effective behavior-relevant constructor
arguments, and any selected asset-manifest paths.
Estimator classes must not duplicate this runner-owned identity in ``NAME`` or
``REFERENCE`` attributes. Publication references belong in the paper README
and the leaderboard's paper-level source metadata.

## Auxiliary Assets

Record selected reusable inference assets under the manifest contract in
`docs/mimo_ofdm_chest/contracts/asset_manifests.md` and `docs/mimo_ofdm_chest/inventories/asset_inventory.md`.
Generate or train every candidate under an explicitly supplied external asset
root. A complete package-local copy under
`src/sionna_bench/mimo_ofdm_chest/papers/<paper_id>/assets/<scenario_id>/<asset_type>/<artifact_id>/`
is optional for internal distributions and must not be assumed to exist.
Examples include:

- covariance/statistics files required by a paper method
- pretrained weights
- calibration tables
- reference simulation settings

Training datasets and rejected trial assets remain external unless they are
themselves required for inference and approved for distribution. If a method
requires an inference asset, the default expectation is that it is generated
faithfully as part of the implementation and reused by the estimator or runner.
Write a concrete acquisition plan before generating it.
The plan must describe the source or generation procedure, validation checks,
storage path, and how the estimator verifies that the asset matches the active
scenario.

Statistics estimated from the benchmark channel model should be reusable
scenario assets. The default for pilot-group methods is to pool
samples across all pilot groups when estimating shared channel statistics, then
reuse the shared statistics for all groups during channel estimation. Only use
per-group statistics when the paper explicitly models group-dependent
statistics or when a reviewer approves the distinction as necessary.

If a required asset cannot be faithfully generated in the current pass, classify
the corresponding method as `not implemented` and keep the acquisition plan in
`reports/plan.md`.

## Testing Discipline

Use focused tests under `test/` for mathematical components when the expected
property is independent of a full Monte Carlo benchmark. Good candidates are:

- interpolation pass-through at pilot positions;
- known limiting cases, such as zero noise or disabled regularization;
- equivalence to a built-in Sionna baseline for a special parameter setting;
- shape, dtype, and device behavior for custom estimators/interpolators;
- covariance, projection, kernel, or filtering identities implied by the
  paper.

Smoke runs remain required integration checks. Full benchmark runs are still
needed before reporting performance claims or canonical result files.

## README Generation

Every finished paper reimplementation should be accompanied by a corresponding README in
`src/sionna_bench/mimo_ofdm_chest/papers/<paper_id>/README.md`. The README should have the
following shape:

```text
# Implementation of the Estimators from <Author> et al., <Year>

Publication: <full citation with venue/report>

## Abstract

Short description of what was implemented, using Sionna terminology where
possible.

## Implemented Classes

List public classes, main methods, variants, and correspondence to the paper.

## Modifications and Omissions

Document benchmark adaptations, omitted variants, error-variance fallbacks,
training/statistics generation, and departures from the paper.

## Benchmark Run

Show the `python -m sionna_bench.mimo_ofdm_chest.papers.<paper_id>.run_benchmark ...` command and the
important parameters used for the canonical result.
```
