# Implementation Report

Approved basis: reviewed plan v2, 2026-10-25, and subsequent explicitly
approved reference training, bounded studies, calibration and final evaluation.
The README and five required reports are the maintained documentation.
Original stage reports and full numerical evidence remain externally archived.

## Implemented Scope

- `DiffusionChannelEstimator`: original iterative angular-domain prior adapted
  to physical receive-array vectors and Sionna's pilot-domain interface.
- Original 55,025-parameter CNN, 100-step VP schedule, epsilon-MSE training,
  actual-noise normalization, and inclusive deterministic reverse means.
- `asset.py`: explicit data generation/training and strict prepared-state
  loading; schema-2 dataset and inference manifests.
- `tuning.py`: shared optimizer-study and frozen-model variance entry points.
- `run_benchmark.py`: separate preparation/evaluation, effective constructor
  arguments, standard result storage and selected manifest linkage.

Spatial-domain ablations, alternate schedules/targets, SNR-mismatch studies,
single-step inference, and reproduction of paper comparison estimators are
not implemented. Their requirements are inventoried in [plan.md](plan.md).
The source qualification is in [paper_summary.md](paper_summary.md).

## Deviations From Paper and Approved Plan

The paper's 64×16 receive/transmit matrix is adapted to physical receive-array
snapshots. The selected single-polarized 4×4 BS array is reordered using
`antenna_index = row + rows*column`, followed by an orthonormal 2D FFT.
Users and pilot frequencies are not artificial transmit-array coordinates.
Other rectangular arrays require compatible training and their own calibration;
noise must be isotropic within each receive vector.

Sparse OFDM pilot division replaces unitary pilot decorrelation. Actual pilot
variance is `no/abs(pilot)**2`; training channel power supplies the further
normalization. Training-only population power follows the paper's unit-average
convention, without per-observation/test-channel normalization. Explicit
`lin_time_avg` reconstruction fills off-pilot resources; a custom interpolator
has standard precedence.

The human designated arXiv:2403.03545v1 authoritative. Algorithm 1 includes the
matched transition: zero-based coefficient index `k` executes `k,...,0`.
The inspected `wcl-2024` release executes `k-1,...,0`. The implementation
intentionally follows the paper, not complete-chain release equality.
A correlated Gaussian conditioning check supports the inclusive convention.
Final IEEE indexed text showed no production-algorithm change, but the binary
final PDF was unavailable and complete final-PDF inspection is not claimed.

The original architecture, target and solver remain intact. The new UMi bank,
6,400/256 independent realizations, 200-epoch ceiling, and validation policy
are approved benchmark adaptations. One shared conditional model replaces
the named supervised families' per-SNR bank. The approved optimizer-study
amendments retain the actual B512 incumbent outside trial accounting, include
B128, use TPE startup five, cap at ten new fits/zero replacements, and confirm
the fixed raw winner without an additional rounded challenger refit.

The detector variance is empirical scaled LS, not diffusion posterior
covariance. Explicit runner/variance purpose options strictly load the retained
`single-reference` manifest without relabeling it. Normal defaults still require
canonical assets; purpose overrides cannot prepare or train assets.

## Auxiliary Assets and Selected Training

All writable assets are external. The layout is
`<asset_root>/3gpp-umi/fesl2024/<asset_type>/<artifact_id>/`, with types
`diffusion-training-data` and `diffusion-prior`. Datasets store `train.npy`,
`validation.npy`, and `normalization.npz`; inference stores
`diffusion_params.npz`. Manifests bind physical geometry, exact active pilots,
transform/order, normalization, schedule, architecture, purpose, RNG roles,
independent/derived counts, source/runtime and payload metadata/checksums.

Strict loaders validate schema, compatibility, ordered state keys, shapes,
dtypes, finite values and hashes, then return prepared tensors and manifest
paths. Construction and inference perform no asset I/O or preparation. Public
distributions omit payloads; missing assets never trigger implicit generation
or download. Standard result metadata links the exact selected manifest.

The retained reference uses 6,400 training/256 validation complete realizations,
7,987,200/319,488 correlated active-pilot vectors, Adam default moments,
zero weight decay, LR `1e-4`, B512, and no learning-rate decay. Each epoch
resamples deterministic timesteps/noise using stable example addresses;
validation noise/timesteps are fixed. Training-only power is
`1.6261395394191218`. Max 200 epochs, patience 20, relative improvement `1e-4`
control stopping; absolute minimum validation epsilon-MSE selects weights.
Selected epoch 170 had training loss `0.08046158737192552` and validation loss
`0.08196578150352415`. Training stopped at epoch 190; total training time was
13,356.49 s (3.71 h), with native MPS training and CPU preprocessing.

The explicit regeneration recipe uses the fixed dataset and:

```python
state, power, manifest = train_model(
    resource_grid, cfg,
    training_data_manifest=training_data_manifest,
    asset_root=external_root, artifact_id="reference-reproduction",
    epochs=200, optimizer_batch_size=512, learning_rate=1e-4,
    patience_epochs=20, relative_min_delta=1e-4,
    initialization_seed=24270000, noise_seed=24280000,
    validation_noise_seed=24290000, training_device="mps",
)
```

Here `train_model` is imported from the paper's `asset.py`, and the resource
grid/config match `umi_3gpp_config` on CPU. Data generation uses
`generate_training_data()` with 6,400/256 channels, propagation seeds
24250000/24260000 and an explicit external root/purpose. Preserve the original
`single-reference` purpose when reproducing that reference. Canonical runner
preparation uses a canonical-purpose dataset and default B128; it is not an
unchanged regeneration of the selected B512 fit. Exact historical reproduction
also requires the recorded frozen sources/environment and immutable dataset.

| Protected artifact | SHA-256 |
| :-- | :-- |
| Dataset manifest | `335bae0356cc46b62b7c9f4859f49ceca330cf6305743484681bab83d2ff79f1` |
| Training array | `8324daf1e7f9dfe6263c75680a184fa2335dd6509e8f2d1b8d67734bbf3b1868` |
| Validation array | `cc0ae306a4d35486325c201fe67c719298167f3b12137e0e47b214ebb8882565` |
| Normalization array | `93c9bc4078e5858b81927422daf8a66bedb55e808c0a8f529b502c569db76cb0` |
| Selected model manifest | `ef8d88197d860f9942d085e9e2eb7af3f14088341d40bdac35bfbe3eb2c9668b` |
| Selected inference payload | `60b2d5e8d4c801d40a20354336e396981965583bada545c375f3eff96204ad1d` |
| Actual epoch-170 checkpoint | `1e32560f9ed70138df5a29af497885c0bad4b985157c4b7df59d6bc94425d98e` |

## Optimizer Study and Independent Selection

The ten-trial optimizer-only study searched LR log `[1e-5,1e-3]` and batches
`{128,256,512,1024,2048}`. TPE seed 24300000, startup five: one enqueued
LR1e-4/B128, four additional startup proposals, then five guided suggestions.
Every trial started from fresh seed 24270000 with shared noise/shuffle seed
24280000 and fixed validation seed 24290000, on the same immutable bank and
full training policy. All ten completed; zero failed/pruned/replacement fits.

Selection used cached observations at `{-10,-5,0,5,10}` dB, three batches of
four complete realizations per SNR, seed 24310000. The objective pools
SSE/channel energy within each SNR, then averages with normalized weights
proportional to `10**(SNR/10)`. None of the ten candidates beat the reference.
Raw winner trial 6 was LR `0.0008461960538786609`, B1024, epoch 102, selection
objective `0.00479760613839` versus reference `0.00477079707088`.

Only that preselected raw winner received independent confirmation, seed
24320000, 63×4 complete realizations at each of the five SNRs. Weighted
full-grid NMSE was `0.004720078666` versus reference `0.004703574668`:
0.3509% worse, nominal paired 95% interval [0.1872%, 0.5148%] worse.
The strict-lower rule retained the original epoch-170 model. Bootstrap seed
24321000 used 20,000 complete-realization draws. No candidate reranking,
rounded fit, or retraining followed. The earlier six-checkpoint diagnostic
also retained epoch 170; a compact-CNN ablation was not adopted.

The study used 1,618 completed epochs and 40.4045 active hours; independent
confirmation/finalization brought metered scientific work to 40.7419 h,
below the approved 216 h ceiling. The study had one authorized same-trial
interruption/resume, not an extra fresh fit. Retained trial files occupied
about 1.378 GiB and selection observations 0.313 GiB, excluding the original
bank. This bounded one-initialization study does not establish global
optimality or between-training-seed robustness.

## Detector-Facing Calibration

The frozen estimator returns `err_var_scale * no/abs(pilot)**2` at pilots,
propagated through Sionna interpolation. Ten TPE trials searched only this
scale on log `[1e-3,1]`, minimizing codeword BLER at 10 dB with 63×4 complete
realizations. Observations, estimates and unscaled variances were cached.
Calibration seed 24330000 and independent validation seed 24340000 reserve
separate `base+[0,62]` streams. Sampler seed 24350000 retained startup ten,
so all ten suggestions were startup proposals. All trials completed, with
zero failed/pruned/replacement trials.

Trial 8 won at raw scale `0.6229862696239178`, calibration BLER
`0.027777777777777776`. The three-significant-digit candidate `0.623` was
validated against incumbent 1 on another paired 252-realization cohort:

| Scale | Independent validation BLER |
| :-- | --: |
| 1 | 0.02976190476190476 |
| 0.623 | 0.028769841269841268 |

Strict improvement selects **0.623**, now the constructor default. The absolute
gain is 0.099206 percentage points, 3.333% relative. Saved aggregate counts
do not establish paired significance. Canonical observations did not choose
the scale. Calibration took 1078.894 s and validation 244.457 s.

## Final Benchmark Results

The selected-scale evaluation used the shared 3GPP UMi configuration,
single precision, CPU/one Torch thread, setup seed 42, evaluation seed 2026,
and 63×4 complete realizations per SNR over −10:2.5:15 dB. Its scientific
fingerprint is
`4f774132bba964e4d31fe64a10374b020db5abd5c13a5e564b4e62bf258a4efc`.
The complete NMSE/BER/BLER curve and command are in [the README](../README.md).
Final NPZ SHA-256:
`10396d3b84973308571fdfd3a78bb1419b8849712615ea2a5c972ae7ed075876`.

All SNR/NMSE/both NMSE interval arrays equal the scale-1 evaluation in shape,
dtype and bytes. Channel estimates remain unchanged. BER improves at all
11 points; BLER improves at ten and ties at 15 dB. At 10 dB, NMSE is
`0.0017364497734659434`, BER `0.0022061129218892157`, and BLER
`0.016534391534391533` versus prior BLER `0.018849206349206348`.
The final evaluator took 2251.301 s; the full calibration/validation/evaluation
sequence took 3575.759 s (59.60 min), excluding no relevant phase.

Comparison read twenty published entries without executing competitors.
All published leaderboard SNR/NMSE/BER/BLER arrays matched their NPZs;
the audit reproduced 220 relative comparisons and 231 ranks. Diffusion's NMSE
ranks across ascending SNR are `[12,11,11,11,11,9,8,7,4,7,7]` of 21.
It nominally beats VAE/spatial GMM only at 10 dB, by 1.1655%/1.8878%, and
trails spectral GMM/LMMSE/RKHS at every point. Published LS interpolation is
nn/lin, not matched `lin_time_avg`. Published CUDA and current CPU evaluations
share scientific metadata but are not asserted to have identical realized
channels; aggregates do not support paired cross-estimator significance.

The result is external canonical-configuration evidence, not an officially
tracked canonical reference or leaderboard entry. Maintainers own global
inventory integration, result registration and public asset distribution.
The retained manifest is never relabeled. Source, shared inventories,
published competitor results and unrelated submodule state were not changed
by documentation consolidation.

## Verification and Known Limitations

Final code cleanup passed 35 focused tests without training or propagation-data
generation: 29 analytical/reference/compile/mocked-integrity checks (5.54 s)
and six fixed-asset/default/purpose/result checks (5.63 s). They cover physical
FFT/order, complex noise, inclusive reverse/Gaussian conditioning, author CNN
and scheduler parity, single/double precision, rectangular geometry,
eager/compiled agreement, protected-state loading and purpose rejection before
output creation. Fixed existing weights/observations give bitwise equality
between cleaned and executed classes at scale 0.623; changing scale leaves
h_hat exact and scales only variance.

The five executable syntax trees match the executed snapshots after removing
docstrings and normalizing only the approved variance default 1→0.623.
Temporary Black 26.10.1 formatting changed no algorithm; the exact 82-line
launch dependency snapshot remained unchanged. Inference is Tier B for the
tested eager compile backend; runtime timestep bucketing is outside compiled
tensor execution. CUDA/Inductor performance is not claimed.

Historical failures remain disclosed: tiny-fit underperformance was not final
selection evidence; interrupted full fits resumed the same trial; a prelaunch
filename defect was fixed before launch. Post-study JSON publication failed
on a NumPy Boolean after all observations were saved; a bounded serializer
recovery changed no statistics or inference. Calibration v1 stopped before
any data/trial due solely to the launch-PYTHONPATH pip-freeze representation;
v2 froze the exact worker environment and completed. None was hidden by
retraining or replacing an outcome. Generic partial stochastic-study restart
is not exactly reproducible because `_bounded_study` does not save sampler RNG;
the reviewed external controllers enforce their separate restart rules.

## Reproduction Evidence

Measured environment: macOS 27.0.1 arm64, Python 3.12.13, PyTorch 2.11.0,
NumPy 2.4.4, Sionna 2.1.0; native MPS training, CPU evaluation, one CPU Torch
thread, no CUDA. The calibration cache process reported about 3.46 GiB RSS
at startup; this is not a universal peak-memory bound.
Source HEAD was `e1d2fdbeed3186201de1929e88169446583b54a4`; the paper was
uncommitted, so frozen source hashes/snapshots, not HEAD alone, identify it.

External roots are sibling directories of the checkout on the original host:

| Root | Retained evidence |
| :-- | :-- |
| `fesl2024-single-trial` | Fixed bank, reference checkpoints, selected manifest/payload, recipe and training logs. |
| `fesl2024-optuna-study` | Frozen sources/dependencies, study DB/summary, ten fits, probes, `poststudy/` raw-winner confirmation and audit. |
| `fesl2024-canonical-evaluation-v1` | Initial scale-1 evaluation and `comparison.json` with all baseline arrays/ranks. |
| `fesl2024-variance-calibration-v2` | Exact worker/source/dependency freeze, study DB, paired record, `selected_scale.json`, separate logs and final `evaluate/results/`. |
| `fesl2024-finalization-v1` | Full pre-consolidation source/reports, hashes, verification scripts/logs and independent audit snapshots. |

The finalization archive is
`/Users/bene/projects/channel-estimation/fesl2024-finalization-v1/`.
Its `paper_after_cleanup/` snapshot and `final_hashes.json` preserve all
48 pre-consolidation files, including the 37 removed stage reports. Current
reports retain the necessary scientific evidence; task prompts, launch
checklists, development histories, inline test source and PR draft are external.
Historical runs must use their archived sources/environment rather than
substitute current documentation or source hashes into frozen manifests.

The actual no-training test files and logs are in that root's `verification/`:

```bash
MPLCONFIGDIR=/private/tmp/fesl2024-mpl PYTHONPATH=src .venv/bin/python -m pytest -q "$FINALIZATION_ROOT/verification/test_legacy_no_training.py"
MPLCONFIGDIR=/private/tmp/fesl2024-mpl PYTHONPATH=src .venv/bin/python -m pytest -q "$FINALIZATION_ROOT/verification/test_final_package.py"
```

Set `FINALIZATION_ROOT` to the archive above. These are retained verification
commands, not new runs during documentation consolidation. Asset-dependent
checks require the protected reference/development observations; parity checks
use the recorded author-reference checkouts as test-only dependencies.
Independent acceptance and qualifications are in
[final_report.md](final_report.md).
