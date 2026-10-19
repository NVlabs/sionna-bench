# Task Implementer Prompt: Channel Estimation

Use this together with `prompts/roles/implementer.md` and
`prompts/tasks/channel_estimation/task.md`.

## Required Contracts

Before writing the plan, read:

- `docs/mimo_ofdm_chest/contracts/paper_architecture.md`;
- `docs/mimo_ofdm_chest/contracts/asset_manifests.md`, if the method requires persistent
  assets;
- `docs/mimo_ofdm_chest/contracts/tuning_protocol.md`, if any behavior-relevant parameter
  requires selection or calibration; and
- `docs/mimo_ofdm_chest/contracts/canonical_benchmark_references.md` before producing or
  replacing canonical outputs.

Do not begin implementation until the plan identifies and applies every
relevant contract.

## Harness Familiarization Checklist

Before implementing a paper, inspect and summarize:

- `src/sionna_bench/mimo_ofdm_chest/configs/umi_3gpp_config.py`
- `src/sionna_bench/mimo_ofdm_chest/configs/smoke_config.py`
- `src/sionna_bench/mimo_ofdm_chest/benchmarking/communication_system.py`
- `src/sionna_bench/mimo_ofdm_chest/benchmarking/channel_estimation.py`
- `src/sionna_bench/mimo_ofdm_chest/benchmarking/result_store.py`
- `src/sionna_bench/mimo_ofdm_chest/asset_store.py`
- `sionna.phy.ofdm.channel_estimation._PilotGroupMixin`
- `src/sionna_bench/mimo_ofdm_chest/papers/builtin/`
- existing paper-local `estimator.py` and `run_benchmark.py` examples

Your harness summary must state the tensor shapes, pilot pattern, built-in
baselines, output metrics, and output directory layout.

## Channel-Estimation Planning Requirements

For each channel estimator in the paper, explicitly answer:

- What are the paper's pilot observations or training measurements?
- What is estimated at pilot or training locations?
- What interpolation, extrapolation, reconstruction, filtering, or inverse
  problem produces the full channel estimate?
- Does the paper treat interpolation or reconstruction across OFDM subcarriers
  or OFDM symbols as part of the estimator, or does it only define estimation
  at observed pilot locations?
- Should the implementation use Sionna `BasePilotChannelEstimator`,
  `LSChannelEstimator` plus a custom `BaseChannelInterpolator`,
  a standalone custom `BaseChannelInterpolator`, or
  Sionna `BaseChannelEstimator` with a direct full-grid `call()`?
- Does the method require detector feedback, data-symbol decisions, persistent
  state across frames, raw-grid joint inference, learned weights, or external
  statistics?

The implementation pseudocode must cite paper equations next to the matching
code-level operation.

For interpolation, extrapolation, filtering, or reconstruction methods, state
the frequency/time boundary convention explicitly. If the paper omits it, choose
a conservative convention only after checking whether the named method or
standard procedure has documented semantics. Missing boundary details alone do
not justify marking a method `not implemented` when the estimator itself is
otherwise specified.

For formulas that map pilot-domain estimates to a full resource grid, verify
index ranges, interval coordinates, transform scaling, and pass-through
invariants before coding. If a literal equation violates these checks, document
the correction in the plan and implementation report.

## Estimator Architecture Decision

The default implementation is Sionna's estimate-at-pilots/interpolate
separation. Use it when the paper only defines pilot-location estimation, does
not discuss interpolation, assumes pilot observations on every subcarrier, or
already separates pilot estimation from interpolation.

For this default architecture, implement the paper method as one of:

- a Sionna `BasePilotChannelEstimator` that implements
  `estimate_at_pilot_locations()` and passes `interpolation_type` or
  `interpolator` explicitly
- an `LSChannelEstimator` with an explicit interpolator
- a custom `BaseChannelInterpolator` used by a Sionna estimator

Use Sionna's `BaseChannelEstimator` when the paper
explicitly assumes pilots only on selected resource elements and the proposed
algorithm itself reconstructs, interpolates, extrapolates, or super-resolves
the channel across subcarriers and/or OFDM symbols from those pilot observations.
In that case, implement `call(y, no)` directly, still use Sionna-compatible
pilot extraction and `_PilotGroupMixin` when structured pilot groups are needed,
and return full-grid `h_hat` and `err_var` with the benchmark shapes.

Do not use `BaseChannelEstimator` merely because the paper assumes pilots
on all subcarriers and never discusses interpolation. That case should use
`BasePilotChannelEstimator` with `estimate_at_pilot_locations()` and explicit
interpolation, with the all-subcarrier pilot assumption documented as an
observation-model mismatch.

Never rely on Sionna's default interpolation. Pass `interpolation_type` or
`interpolator` explicitly for the pilot-domain architecture and justify the
choice in `reports/plan.md`. For direct `BaseChannelEstimator` implementations,
document the
paper-specific subcarrier reconstruction and any separate OFDM-time
interpolation policy.

When the paper only specifies pilot-domain estimation, default to the
conservative route:

```text
faithful pilot-domain estimator + explicit Sionna built-in interpolation
```

If the paper's native subcarrier reconstruction is too underspecified,
computationally impractical, or incompatible with the benchmark contract, do
not replace it with an unrelated surrogate. Either implement the faithful
pilot-domain part with a documented interpolation adaptation, or mark the
method `not implemented`, depending on what remains of the paper algorithm.

## Hyperparameter Planning

Classify every behavior-relevant parameter as one of:

- fixed by the paper;
- derived from the system configuration;
- fixed by an explicitly documented implementation decision; or
- selected through hyperparameter tuning.

For tuned parameters, specify in the plan:

- parameter names and search spaces;
- objective and SNR points;
- sampler and trial count;
- probe and paired-validation budgets;
- fixed parameters;
- whether each trial requires asset fitting or model training;
- rounding and final selection procedure; and
- the stage at which tuning occurs.

Follow `docs/mimo_ofdm_chest/contracts/tuning_protocol.md` for every canonical tuning study.

Do not replace a paper-prescribed value with a tuned value without a reviewed
justification. Typical exemptions are parameters that depend on the scale
of vectors, such as regularization constants.
Do not assign an arbitrary canonical default to an unspecified
or adapted parameter when the tuning protocol applies.

Complete algorithmic tuning only after the implementation and relevant tests
are stable. Complete detector-facing variance calibration only after
algorithmic parameters and inference assets are frozen. Do not use canonical
benchmark outputs for parameter selection.

## Asset Planning

For every persistent asset required by an implemented method, specify in the
plan:

- asset type, external generation path, and optional package-local path;
- generation, training, fitting, or estimation entry point;
- number of independent training or statistical channels;
- number of independent validation channels;
- number of derived optimizer examples;
- SNR coverage and whether clean channels are shared across SNRs;
- generation, training, and validation seeds;
- fitted or trained parameters;
- payload files, tensor shapes, and dtypes;
- expected storage size;
- compatibility and validation checks;
- estimated generation runtime; and
- required hardware.

Follow the dataset sizes in `docs/mimo_ofdm_chest/contracts/tuning_protocol.md`:

- use 64,000 independent channels for covariance statistics;
- use 6,400 independent channels for GMM fitting; and
- use 6,400 training and 256 validation channels for each SNR-specific neural
  model.

For a new asset family not covered by the protocol, propose and justify an
independent-channel budget in the plan and obtain reviewer approval before
generation.

Distinguish independent channel realizations, derived optimizer examples, and
serialized file size. Do not report derived examples as independent channels.

## Preliminary Tuning Gate

Before a full hyperparameter study, run a small deterministic feasibility probe
at 10 dB unless the approved plan explains why another operating point is more
representative.

The preliminary probe must:

- use a dedicated seed not reused for Optuna probes, paired validation, or
  canonical evaluation;
- evaluate a small number of deliberately selected configurations spanning the
  proposed search space;
- use enough channels to reveal gross numerical or performance differences,
  without being treated as final statistical evidence;
- verify finite outputs and expected qualitative parameter sensitivity;
- identify invalid, unstable, or clearly unproductive regions;
- measure estimator, fitting, training, and data-generation costs separately
  where applicable; and
- leave every generated model or fitted candidate under a temporary or external
  asset root.

For learned or fitted methods, measure enough representative work to estimate
the full study. For example, time full-dataset epoch throughput, one
representative full-quality fit, or another stage that dominates trial runtime.
Do not estimate a full study solely from a reduced smoke model when its scaling
behavior is unknown.

The preliminary probe may refine the proposed search space, but it does not
select canonical parameters. Record any resulting plan change explicitly.

## Post-Approval Execution Order

1. Implement estimator, runner, asset API, tuning entry point, and focused
   tests without starting expensive studies.
2. Verify analytical behavior, supported geometries, eager/compiled agreement,
   and temporary smoke execution.
3. Run the preliminary 10 dB tuning probe.
4. Report to the human:
   - tested configurations and preliminary seed;
   - sample and training/fitting budgets;
   - measured runtimes and hardware;
   - observed numerical and performance behavior;
   - proposed final search space;
   - expected runtime and conservative no-early-termination upper bound;
   - estimated storage requirements; and
   - assumptions used for extrapolation.
5. Stop and request explicit approval before starting full tuning.
6. After approval, generate and hash the immutable full-size datasets or
   statistics required by the study.
7. Run canonical algorithmic tuning according to
   `docs/mimo_ofdm_chest/contracts/tuning_protocol.md`.
8. Round and independently validate the selected candidate, then freeze the
   estimator defaults and generation recipe.
9. Generate, strictly validate, and promote the final canonical inference
   assets.
10. Calibrate any empirical detector-facing variance scale.
11. Run the canonical benchmark exactly once after selection is complete.
12. Update reports, READMEs, inventories, result metadata, and leaderboard data.

## Sionna-Style Implementation Rules

- Use Sionna's private `_PilotGroupMixin` whenever the method needs structured
  pilot groups.
  Do not reimplement pilot grouping, pilot coordinates, or pilot-value
  gathering functionality already provided by the mixin.
- Do not check properties that are guaranteed by Sionna, the active benchmark
  contract, or `_PilotGroupMixin`. Avoid defensive code for impossible states.
- Expose algorithm choices as constructor arguments and store them as private
  instance attributes.
- Use descriptive variable names. Do not copy cryptic paper notation directly
  into code when a semantic name is available.
- Do not create module-level private helper functions for estimator logic.
  Prefer private methods on the estimator or interpolator class. Module-level
  public helpers are acceptable for asset generation, training entry points, or
  reusable non-estimator workflows.
- Maintain torch-compilability where practical. Avoid unnecessary Python-side
  data-dependent branching in hot tensor paths.
- Respect Sionna's `precision` and `device` conventions. Do not hard-code
  dtype/device except at external boundaries such as serialization.
- Register fixed tensors as buffers.
- Favor simplicity over maximum performance. Prefer one uniform code path over
  special-case branches that only save small amounts of computation.
- Implement closely related estimator variants through configuration of one
  shared estimator class when this keeps the code simpler and preserves the
  paper's distinctions.
- If pilot-group processing depends on estimated channel statistics, aggregate
  those statistics over all pilot groups and reuse the aggregate across pilot
  groups during estimation, unless the paper explicitly requires
  group-specific statistics.
- If a paper proceeds from noisy channel measurements directly, consider the
  use of `LSChannelEstimator.estimate_at_pilot_locations()` inside your
  implementation to avoid code duplication.

## No Surrogate Estimators

Do not replace a paper estimator with a different estimator that merely shares
the same intuition, sparsity domain, regularizer family, or interpolation
category. If the exact estimator requires unavailable weights, statistics,
training data, or a benchmark interface change, make an `assets/` acquisition
plan or mark it `not implemented`.

Examples that must use an `assets/` acquisition plan or be marked
`not implemented`:

- learned estimators without trained weights or a reproducible training recipe
- MMSE methods without the required covariance/statistical model

Examples that must be marked `not implemented`:
- iterative detector-feedback methods without detector feedback in the harness

## Learned-Method Training Discipline

Treat smoke training only as an integration test. Select canonical training
hyperparameters using `docs/mimo_ofdm_chest/contracts/tuning_protocol.md`.

Use the prescribed fixed training and validation datasets, trial count,
SNR tuning bank, paired validation, epoch limit, early-stopping rule, and
rounding policy. After selecting the recipe, train the canonical SNR
model bank with the same full-effort training policy.

Do not select hyperparameters from smoke results, training loss alone, or
canonical benchmark outputs.

For learned methods, `implementation_report.md` must include:

- training-set size and validation-set size
- optimizer, learning rate, optimizer batch size, and any scheduler
- early-stopping rule and selected epoch/checkpoint
- final training loss and validation loss
- evidence that selected hyperparameters were sanity-checked, especially batch
  size and learning rate

## Analytical Tests

For nontrivial estimator or interpolator logic, consider whether the paper or
the adapted implementation implies a lightweight property that can be tested in
`test/`. Prefer tests that do not require a full benchmark run, such as pilot
pass-through, baseline equivalence under a special setting, known transform
identities, shape/dtype/device contracts, and limiting cases such as zero
noise or disabled regularization.

If such a test is appropriate, add it and record the command in
`reports/implementation_report.md`. If no such property is available or the
test would duplicate a full benchmark, state that explicitly in the report.

## Smoke Sanity Checks

The smoke run is primarily an integration check, but it must still produce
credible channel estimates.

Treat these as red flags and report them in `implementation_report.md`:

- shape or dtype mismatches
- NaN or Inf in `h_hat`, `err_var`, BER, BLER, or NMSE
- paper estimator worse than `ls-nn` on NMSE at the smoke SNR, unless the plan
  explains why that is expected for this specific method
- negative error variances
- output that is independent of received pilots or noise level when the paper
  method should depend on them

Do not tune parameters after seeing smoke failures unless the tuning rule is
from the paper or is reviewed and documented as an interface adaptation.

Calibrate every empirical detector-facing variance scale according to the
detector-calibration section of `docs/mimo_ofdm_chest/contracts/tuning_protocol.md`. Perform
this calibration only after estimator parameters and inference assets are
frozen. Record the raw winner, rounded candidate, paired comparison, and final
selected value.

Smoke underperformance is a red flag for debugging, not a final performance
verdict. When integration checks pass but performance is ambiguous, run the
default-size benchmark before accepting or rejecting the channel-estimation
method.
