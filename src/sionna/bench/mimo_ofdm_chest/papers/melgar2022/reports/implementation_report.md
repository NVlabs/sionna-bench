# Implementation Report

- Paper ID: `melgar2022`
- Implementer: Codex
- Date: 2026-05-05
- Revision note: updated after user guidance to adapt Model 2 directly to the benchmark's 156-pilot input/output dimension.
- Revision note 2: updated after user-requested diagnostic removing the final 8-neuron hidden layer, making the bottleneck 16 neurons.
- Revision note 3: updated after user-requested h64x3 diagnostic with hidden layers `1024, 512, 256, 128, 64, 64, 64`.
- Revision note 4: updated after same-batch pilot/full-grid diagnostic showing the h64x3 model does not beat LS on fresh/live pilot samples.
- Revision note 5: updated after a one-hour serious training run using 10000 training channels and 2000 independent validation channels.
- Revision note 6: updated after the full default benchmark sweep for the serious h64 asset.
- Revision note 7: updated after fixed-5 dB training with 100000 training channels, explicit per-SNR asset support, and early stopping.
- Revision note 8: updated after the full default benchmark sweep for the fixed-5 dB 100000-channel asset.
- Revision note 9: updated after mixed-SNR training with 100000 training channels and early stopping.
- Revision note 10: updated after the full default benchmark sweep for the mixed-SNR 100000-channel asset.
- Revision note 11: updated after refactoring the implementation to use Sionna's pilot-domain base class with shared pilot grouping, switching training-pair generation to the shared `evaluation` channel model, moving the paper runner to `evaluate_channel_estimator`, and folding the paper-local DNN/cache helper modules into `estimator.py` and `dnn_training.py`.
- Revision note 12: updated after adding SNR-specific DNN weight support. The
  estimator can now load one Model 2 DNN per configured benchmark SNR and
  select the matching denoiser from the runtime noise variance. The existing
  mixed-SNR cache remains available through `--mixed-snr-weights`.
- Revision note 13: updated after the local channel-estimator contract gained
  full-grid estimator support. Model 3 is now implemented as
  `InterpolatedDNNChannelEstimator`, which applies the paper's LS-linear
  interpolation stage before a learned full-grid frequency-block DNN
  post-filter.
- Revision note 14: aligned the implementation with the paper-module
  architecture contract. Dataset generation, training, manifests, and model
  persistence now live in `asset.py`; `run_benchmark.py` and `tuning.py`
  prepare model states and inject them into estimators whose constructors no
  longer perform repository I/O or training.
- Revision note 15: added isolated one-SNR training fixtures for Model 2 and
  Model 3 across the shared baseline and alternate systems. Each fixture
  trains a one-epoch CPU asset beneath `tmp_path`, injects the resulting state
  into its estimator, and verifies finite inference without writing to
  canonical asset or output directories. The systems cover 48 and 72
  subcarriers, 8 and 10 OFDM symbols, 1 and 3 UEs, 1 and 2 pilot-bearing
  symbols, and 2 and 4 BS receive antennas. Model 2 derives endpoint sizes 48
  and 24 from the active pilot groups; Model 3 applies its configured
  12-subcarrier block to both full-grid sizes.
- Revision note 16: verified strict temporary asset loading and geometry
  collision handling. An exactly matching request reloads identical states
  without entering dataset generation, while an occupied
  Model 2 path rejects shifted pilot-symbol positions and an occupied Model 3
  path rejects a changed effective-subcarrier count.
- Revision note 17: strengthened model compatibility without regenerating DNN
  payloads. Model 2 manifests now hash the exact ordered active pilot-group
  coordinates, complex values, and powers. Model 3 manifests now hash the
  complete LS-linear layout, including resource-grid dimensions, effective
  subcarrier indices, pilot mask, compact complex values, and powers. All 39
  retained manifests were migrated while the aggregate payload digest remained
  `a6845112da7d6baa6983db07f5336ca6c5498d2aac03d0d34ca91b759c5a9c1e`.
- Revision note 18: strengthened both checkpoint loaders to validate the exact
  ordered DNN state keys plus every tensor's architecture-derived shape and
  `float32` dtype without coercion. Temporary tests cover changed pilot masks,
  changed Model 3 grid sizes, missing state keys, wrong shapes, and wrong
  dtypes for both model types. All 39 retained checkpoints pass the stricter
  state validation.
- Revision note 19: both checkpoint loaders now select and validate assets
  exclusively through the manifest's explicit `compatibility` and
  `generation_recipe` sections; newly generated manifests omit legacy
  `key_fields`. Temporary exact-load and malformed-checkpoint tests exercise
  that structured contract. Model 2 compatibility also drops the redundant `groups`,
  `group_lengths`, and `num_groups` summaries because the exact `pilot_groups`
  identity supersedes them and avoids device-dependent rounding in
  `pilot_abs_mean`. Representative retained Model 2 and Model 3 checkpoints
  load without retraining after the metadata-only migration.
- Revision note 20: reread Sections III-B through III-D and the retained
  training histories. The Model 2 and Model 3 processing order, separate
  real/imaginary treatment, shared DNN, MSE loss, RMSprop optimizer, and
  per-SNR state selection are implemented as intended. The audit records the
  current `1024, 512, 256, 128, 64, 64, 64` architecture against the paper's
  `1000, 500, 256, 128, 32, 16, 8`, distinguishes full system-channel
  realizations from real-valued DNN examples, and documents that the current
  per-SNR assets specialize only by SNR over the UMi distribution. Model 2's
  runtime interpolation was removed from checkpoint compatibility because
  training occurs entirely at pilot positions. Historical reports that marked
  Model 3 not implemented or claimed Model 2 always used `"lin"` were
  corrected.
- Revision note 21: established the trained-model compilation boundary for
  both estimators. Single-state mixed-SNR inference compiles as one full graph.
  Canonical per-SNR inference is Tier B: `_select_denoiser_index()` is the
  explicitly compiler-disabled Python `ModuleList` selection boundary, while
  the selected DNN and the remaining estimator path compile. Training,
  checkpoint validation, and cache loading remain in `asset.py` and are not
  reachable from estimator inference.
- Revision note 22: added a runtime-only `err_var_scale` to Model 2 and
  calibrated both models after freezing their code and canonical checkpoints.
  Separate 30-trial detector studies and independent canonical-size holdouts
  selected `0.2731792705783113` for Model 2 and `0.3405776426701012` for
  Model 3. These are historical studies superseded for Model 3 by revision 26;
  the scales remain excluded from checkpoint compatibility and identity.
- Revision note 23: separated checkpoint preparation from canonical loading.
  `run_benchmark.py --prepare-assets` uses the method-specific training function with an external
  writable root; normal benchmark execution and detector calibration use
  strict method-specific loader and fail rather than training a missing checkpoint.
  Training-parameter tuning explicitly loads a matching trial asset or calls
  corresponding training function when it is absent. Both dataset collectors now pass the
  configured device and precision to Sionna's topology, channel, interpolation,
  and Eb/N0 conversion helpers. Isolated CPU preparation and evaluation checks
  passed for Model 2 and Model 3.
- Revision note 24: completed the superseded initial Sionna 2.0.1 Model 3 training-parameter
  Optuna study. Twenty valid trials selected rounded candidate settings
  `learning_rate=1.1e-5` and `optimizer_batch_size=256`. The independent
  holdout retained those settings, all eleven per-SNR assets passed strict
  validation, the refreshed detector holdout retained rounded scale `0.341`,
  and the isolated candidate benchmark improved weighted NMSE through 10 dB
  by 9.7% relative to the preserved result.
- Revision note 25: migrated Model 2 and Model 3 persistence to the global
  asset-manifest schema version 2. Manifests now use `generation_recipe` and
  nested `metadata`, and strict loading returns one ordered manifest path per
  prepared state. The benchmark runner records these exact paths together with
  JSON-only architecture, interpolation or block, weight-mode, and
  detector-facing variance settings in the scenario-local result registry.
- Revision note 26: completed the calibrated v2 Model 3 study. A deterministic
  6,400/256-channel bank, ten full five-SNR trials, canonical-sized paired
  validation, eleven final checkpoints, 10 dB detector calibration, and the
  canonical benchmark all used calibrated Sionna commit
  `f16208347246d8d17ec69628dfb856b19287b80c` with UMi TR 38.901 version 19.2.
  The selected recipe is `learning_rate=3.17e-5` and
  `optimizer_batch_size=512`; the detector study retained exact incumbent
  `err_var_scale=0.3405776426701012`. Model 2 remained postponed at this stage.
- Revision note 27: completed the corresponding calibrated Model 2 study.
  Ten full five-SNR trials and canonical-sized paired validation selected
  `learning_rate=7.02e-5` and `optimizer_batch_size=256`. Eleven final
  SNR-specific checkpoints were trained from the deterministic 6,400/256
  channel bank and independently verified against their restored best-state
  validation metrics. Detector calibration selected `err_var_scale=0.454`,
  and the canonical benchmark and exact result-to-asset linkage passed.

## Paper Fidelity Recheck

- **Architecture and latent dimension:** Both methods use one real-valued,
  fully connected ReLU DNN with a linear output, shared between real and
  imaginary inputs. The current hidden widths are
  `1024, 512, 256, 128, 64, 64, 64`, including a 64-dimensional bottleneck.
  Section III-D instead specifies `1000, 500, 256, 128, 32, 16, 8`. This is a
  deliberate, material architecture adaptation, not an exact reproduction.
- **Model 2 placement:** LS pilot vectors are denoised before off-pilot
  interpolation, matching Section III-B2. The paper specifies 2D linear
  interpolation. The implementation exposes `"lin"`, but its canonical
  repository-wide default is `"lin_time_avg"`; interpolation is runtime-only
  and is not part of checkpoint compatibility.
- **Model 3 placement:** Sionna `"lin"` interpolation is applied over the full
  grid before the DNN processes each OFDM symbol in contiguous frequency
  blocks, matching Section III-B3. The block length is adapted from 72 to 78.
- **Per-SNR selection:** `_DNNBank` selects the state whose recorded AWGN
  variance matches the runtime variance. Canonical assets provide one state
  for each benchmark SNR. Unlike the paper's specialized mode, they do not
  additionally specialize by delay spread and Doppler; those vary only through
  the configured UMi distribution.
- **Loss and optimization:** Separate real and imaginary MSE is equivalent,
  up to a constant factor, to complex squared-error minimization. ReLU hidden
  activations, linear output, RMSprop, and the paper's default learning rate
  `0.0005` and optimizer batch size `500` are supported. Best-validation
  checkpointing, early stopping, and the selected Model 2 and Model 3
  optimizer settings are implementation adaptations.
- **Sample accounting:** A paper data point is one real-valued DNN input/target
  vector, not one full OFDM channel realization. Both calibrated banks use
  6,400 training and 256 validation channel realizations per SNR. Model 2
  produces 1,638,400 training and 65,536 validation pilot-vector examples;
  Model 3 produces 3,276,800 retained training and 131,072 validation block
  examples.
- **Compilation:** Prepared model states are injected during construction.
  Complete Model 2 and Model 3 inference with one mixed-SNR state passes
  `torch.compile(..., fullgraph=True)`. Per-SNR banks pass with
  `fullgraph=False`; `torch._dynamo.explain` reports only the explicit
  state-selector boundary as the graph-break cause. Regression tests compare
  eager and compiled channel estimates and error variances in both modes.
- **Detector-facing variance:** The paper supplies no Sionna-compatible
  uncertainty. Model 2 scales LS pilot variance before interpolation; Model 3
  scales LS-linear full-grid variance. Regression tests verify that changing
  either scale leaves `h_hat` unchanged and scales only `err_var`.

## Implemented Methods

### Method: `melgar-model2-dnn-pilot`

- Status after implementation: implemented, integration-tested, and evaluated
  with a calibrated canonical per-SNR asset set.
- Files changed:
  - `src/sionna_bench/mimo_ofdm_chest/papers/melgar2022/__init__.py`
  - `src/sionna_bench/mimo_ofdm_chest/papers/melgar2022/asset.py`
  - `src/sionna_bench/mimo_ofdm_chest/papers/melgar2022/tuning.py`
  - `src/sionna_bench/mimo_ofdm_chest/papers/melgar2022/estimator.py`
  - `src/sionna_bench/mimo_ofdm_chest/papers/melgar2022/run_benchmark.py`
  - `evaluation/__init__.py`
  - `evaluation/channel_estimation.py`
  - `evaluation/system.py`
  - `src/sionna_bench/mimo_ofdm_chest/papers/melgar2022/reports/paper_summary.md`
  - `src/sionna_bench/mimo_ofdm_chest/papers/melgar2022/reports/plan.md`
  - `src/sionna_bench/mimo_ofdm_chest/papers/melgar2022/reports/plan_review.md`
  - `src/sionna_bench/mimo_ofdm_chest/papers/melgar2022/reports/implementation_report.md`
  - `src/sionna_bench/mimo_ofdm_chest/papers/melgar2022/reports/final_report.md`
- Paper equations/statements implemented:
  - Eq. (6): LS pilot estimates via Sionna `LSChannelEstimator.estimate_at_pilot_locations`.
  - Eq. (11): learned DNN map `f_theta`.
  - Section III-B2: real/imaginary Model 2 DNN denoising of pilot-resource CSI followed by linear interpolation.
  - Section III-D baseline: hidden layers `1000, 500, 256, 128, 32, 16, 8`, ReLU hidden activations, linear output, MSE loss, RMSprop, learning rate `0.0005`, and mini-batch size `500`.
- Interface adaptations actually made:
  - Implemented as a Sionna `BasePilotChannelEstimator` with `_PilotGroupMixin`;
    Sionna handles pilot extraction and the configured interpolation, while
    the mixin handles pilot-vector grouping.
  - Exposes Sionna's `interpolation_type` and `interpolator` arguments and
    defaults to the repository-wide `interpolation_type="lin_time_avg"`.
  - Adapts the paper's 48-input/48-output endpoint to the active benchmark's 156-pilot groups.
  - Uses the user-requested h64x3 diagnostic architecture, so the current DNN is `156 -> 1024 -> 512 -> 256 -> 128 -> 64 -> 64 -> 64 -> 156`.
  - Trains reusable DNN assets from the active Sionna UMi benchmark channel instead of the paper's MATLAB CDL-A dataset.
  - Synthesizes LS training observations with variance `no / |pilot|^2`, matching Sionna pilot division.
  - Uses calibrated `0.454 *` LS pilot error variance as
    `err_var` because the paper does not define detector-facing uncertainty.
  - `src/sionna_bench/mimo_ofdm_chest/papers/melgar2022/run_benchmark.py` evaluates the selected Melgar
    estimator through `evaluate_channel_estimator`; it does not run built-in
    baselines or plot results.
- Parameters used:
  - Default smoke and runner training: `train_batches=8`, `train_batch_size=8`, `epochs=100`, optimizer batch size `500`, learning rate `0.0005`, seed `42`.
  - Additional long-training diagnostic: `train_batches=8`, `train_batch_size=8`, `epochs=1000`, optimizer batch size `500`, learning rate `0.0005`, seed `42`, estimator suffix `-ep1000`.
  - User-requested h16 diagnostic: same as above, estimator suffix `-h16-ep1000`, hidden layers `1000, 500, 256, 128, 32, 16`.
  - User-requested h64x3 diagnostic: same as above, estimator suffix `-h64x3-ep1000`, hidden layers `1024, 512, 256, 128, 64, 64, 64`.
  - Serious training run: full benchmark config, `train_batches=100`, `train_batch_size=100`, `validation_batches=20`, `validation_batch_size=100`, `epochs=10000`, optimizer batch size `500`, learning rate `0.0005`, `min_train_seconds=3600`, `patience_epochs=10`, `min_delta=1e-5`, seed `42`. This corresponds to 10000 training channels and 2000 independent validation channels.
  - Fixed-5 dB large training run: full benchmark config, `train_batches=1000`, `train_batch_size=100`, `validation_batches=125`, `validation_batch_size=16`, `epochs=100`, optimizer batch size `500`, learning rate `0.0005`, `patience_epochs=5`, `min_delta=1e-5`, explicit `training_snr_dbs=[5.0]`, seed `42`. This corresponds to 100000 training channels and 2000 independent validation channels.
  - Mixed-SNR large training run: full benchmark config, `train_batches=1000`, `train_batch_size=100`, `validation_batches=125`, `validation_batch_size=16`, `epochs=100`, optimizer batch size `500`, learning rate `0.0005`, `patience_epochs=5`, `min_delta=1e-5`, implicit cycling over the benchmark SNR grid, seed `42`. This corresponds to 100000 training channels and 2000 independent validation channels.
  - The canonical per-SNR Model 2 assets use `800 x 8` training and `32 x 8`
    validation channels independently at every SNR, at most 200 epochs,
    patience 20, learning rate `7.02e-5`, and optimizer batch size 256.
    Each complete four-UE, 16-receive-antenna system realization contributes
    256 real-valued DNN pairs, yielding 1,638,400 training and 65,536 validation
    examples per SNR.
- Deviations from approved plan:
  - Earlier exploratory code tried preserving 48-pilot windows. User guidance clarified that direct 156-dimensional endpoints are preferable for this numerology. The active implementation and plan now use direct 156 input/output dimensions.
  - The runner supports explicit fixed-SNR training assets via repeated
    `--training-snr-db`; its default asset preparation loads or trains one
    singleton-SNR DNN cache per configured SNR. Historical mixed-SNR assets
    remain available through `--mixed-snr-weights`.
  - This per-SNR mode specializes only by SNR while sampling the configured
    UMi distribution. It does not reproduce the paper's separate model for
    every SNR, delay-spread, and Doppler combination. Mixed-SNR mode likewise
    mixes SNR within one UMi scenario rather than all paper scattering models.
  - SNR-specific model selection is an inference concern, but cache loading
    and training are not: `asset.py` prepares the states and aligned noise
    variances before the runner injects them into the estimator.

### Method: `melgar-model3-dnn-full-grid`

- Status after implementation: implemented, integration-tested, and evaluated
  with a canonical per-SNR asset set.
- Files changed:
  - `src/sionna_bench/mimo_ofdm_chest/papers/melgar2022/asset.py`
  - `src/sionna_bench/mimo_ofdm_chest/papers/melgar2022/tuning.py`
  - `src/sionna_bench/mimo_ofdm_chest/papers/melgar2022/estimator.py`
  - `src/sionna_bench/mimo_ofdm_chest/papers/melgar2022/run_benchmark.py`
  - `src/sionna_bench/mimo_ofdm_chest/papers/melgar2022/README.md`
  - `src/sionna_bench/mimo_ofdm_chest/papers/melgar2022/reports/implementation_report.md`
- Paper equations/statements implemented:
  - Eq. (6): LS pilot estimates through Sionna's LS estimator.
  - Eq. (11): learned DNN map `f_theta`.
  - Section III-B3: LS at pilot resources, linear interpolation over all
    channel resources, then DNN improvement of all resources.
- Interface adaptations actually made:
  - Implemented as `InterpolatedDNNChannelEstimator`, a
    Sionna `BaseChannelEstimator` that owns `call(y, no)` directly.
  - Uses an internal Sionna `LSChannelEstimator(interpolation_type="lin")` to
    create the full-grid input to Model 3.
  - Applies the same real-valued `DNN` separately to real and imaginary parts
    of full-grid frequency blocks.
  - Adapts the paper's 72-subcarrier block length to default `block_len=78`
    because the benchmark has 624 effective subcarriers and `8 * 78 = 624`.
  - Stores Model 3 assets separately under the documented
    `src/sionna_bench/mimo_ofdm_chest/papers/melgar2022/assets/<scenario_id>/melgar-model3/` hierarchy.
  - Uses LS-linear detector-facing error variances with calibrated v2
    `err_var_scale=0.3405776426701012`.
- Parameters used:
  - Default smoke Model 3 settings: `train_batches=2`,
    `train_batch_size=2`, `epochs=5`, `block_len=78`,
    `max_examples_per_batch=4096`.
  - Default non-smoke Model 3 settings: `train_batches=8`,
    `train_batch_size=8`, `epochs=50`, `block_len=78`,
    `max_examples_per_batch=65536`.
  - Superseded initial Sionna 2.0.1 Model 3 assets: `train_batches=128`,
    `train_batch_size=8`, `validation_batches=32`,
    `validation_batch_size=8`, `epochs=1000`, optimizer batch size `256`,
    learning rate `1.1e-5`, patience `50`, minimum delta `1e-6`,
    `block_len=78`, and `max_examples_per_batch=4096`. Each asset uses 1024
    training and 256 validation system-channel realizations. Capping each
    generated batch at 4096 real-valued frequency-block pairs yields 524288
    training and 131072 validation examples per SNR.
  - Calibrated v2 canonical Model 3 assets: `train_batches=800`,
    `train_batch_size=8`, `validation_batches=32`,
    `validation_batch_size=8`, `epochs=200`, optimizer batch size `512`,
    learning rate `3.17e-5`, patience `20`, relative minimum improvement
    `1e-4`, `block_len=78`, and `max_examples_per_batch=4096`. Each asset uses
    6,400 training and 256 validation channels, yielding 3,276,800 training
    and 131,072 validation examples per SNR.
- Verification:
  - `py_compile` passed for `estimator.py`, `asset.py`, `tuning.py`, and
    `run_benchmark.py`.
  - A tiny smoke run with `--method model3 --mixed-snr-weights
    --train-batches 1 --train-batch-size 1 --validation-batches 1
    --validation-batch-size 1 --epochs 1 --optimizer-batch-size 128
    --max-examples-per-batch 512 --estimator-suffix=-tmp-model3` completed
    end-to-end. The temporary output and cache were removed because the
    one-epoch asset is not a meaningful benchmark result.
  - A tiny SNR-specific `--method model3 --prepare-assets` run with the same
    training budget and an explicit temporary asset root completed and loaded
    the per-SNR asset path. Its temporary cache was also removed.

## Not Implemented Methods

### Method: `melgar-model1-preliminary-dnn`

- Final status: `not implemented`
- Reason: learned full-grid per-resource reconstruction requiring target
  coordinates and trained weights. This is a separate estimator design from
  the implemented Model 2 pilot denoiser and Model 3 full-grid post-filter.
- Asset acquisition plan needed: implement the target-coordinate DNN inference
  path, then train Model 1 weights into an explicit external asset root and
  promote only the selected checkpoints into the paper package.

## Deviation Log

### Deviation: Benchmark-Trained DNN Asset

- Stage: `planning`
- Source: `asset`
- Trigger: no paper weights or MATLAB CDL-A training data are available.
- Decision: train DNN assets from the active Sionna UMi benchmark channel.
- Alternatives considered: not implementing any DNN method or using
  random/untrained weights.
- Why the chosen decision is technically justified: the paper provides a supervised training recipe, and benchmark-domain training preserves the Model 2 inference algorithm.
- Faithfulness impact: moderate; results are not comparable to the paper's CDL-A curves.
- Benchmarking impact: acceptable if the training data and generation settings are disclosed.
- Resolution: the canonical per-SNR assets use independent validation channels
  and best-validation checkpoint selection. Both canonical estimator outputs
  have been regenerated, and their detector-facing variance scales were
  accepted only after independent canonical-size holdout evaluation.

### Deviation: Endpoint Dimension Adaptation

- Stage: `implementation`
- Source: `interface`
- Trigger: the active benchmark has 156 pilot subcarriers per stream and pilot OFDM symbol, while the paper's example block has 48.
- Decision: adapt both input and output layers to 156 while keeping the paper's hidden-layer structure.
- Alternatives considered: 48-pilot tiling; 156-dimensional endpoint with LS blending; fallback to LS.
- Why the chosen decision is technically justified: Model 2's DNN consumes the full pilot vector for the pilot block. Under this numerology, the corresponding block has 156 pilot values.
- Faithfulness impact: moderate; endpoint dimensions follow benchmark numerology.
- Benchmarking impact: interface-compatible. The initial small-training result
  was not credible, but it was superseded by the canonical per-SNR assets and
  independent validation described above.
- Resolution: larger training and default-size validation were completed.
  Input normalization was considered but was not introduced into the approved
  estimator architecture.

### Deviation: Remove 8-Neuron Hidden Layer

- Stage: `implementation`
- Source: `implementation`
- Trigger: 1000-epoch training with the paper's 8-neuron bottleneck plateaued far above LS, and a PCA probe suggested rank 8 is too narrow for 156-dimensional pilot vectors.
- Decision: remove the final 8-neuron hidden layer so the bottleneck is 16 neurons.
- Alternatives considered: keep h8; jump directly to a 32-neuron bottleneck.
- Why the chosen decision is technically justified: it is the smallest architecture relaxation that follows the user's request and remains close to the paper's decreasing pyramid.
- Faithfulness impact: moderate; the hidden architecture no longer exactly matches Section III-D.
- Benchmarking impact: positive but insufficient; h16 improves substantially but still does not beat LS.
- Resolution: this diagnostic was superseded by the reviewed h64x3
  architecture and is retained only as historical evidence about bottleneck
  capacity.

### Deviation: h64x3 Hidden Architecture

- Stage: `implementation`
- Source: `implementation`
- Trigger: h16 improved substantially but still did not beat LS validation MSE or smoke NMSE.
- Decision: use user-requested hidden layers `1024, 512, 256, 128, 64, 64, 64`.
- Alternatives considered: h32 bottleneck; paper h8; h16.
- Why the chosen decision is technically justified: PCA indicated rank 64 has enough representational capacity for very low target-vector reconstruction error, and the requested stack remains a monotone-to-plateau fully connected DNN denoiser.
- Faithfulness impact: high relative to the exact Section III-D architecture; the method remains Model 2-style learned pilot denoising.
- Benchmarking impact: the final per-SNR assets improve over LS-linear in parts
  of the SNR range but retain a documented high-SNR weakness.
- Resolution: best-validation checkpointing and averaged pilot/full-grid
  diagnostics were completed. Residual output parameterization was not added
  because it would change the reviewed architecture.

### Deviation: LS Noise Scaling Fix

- Stage: `implementation`
- Source: `implementation`
- Trigger: Sionna Kronecker pilots have magnitude `2`.
- Decision: synthesize LS training noise with variance `no / |pilot|^2`.
- Alternatives considered: use variance `no`.
- Why the chosen decision is technically justified: Sionna LS pilot division uses the actual pilot symbols; Eq. (6) is the post-division LS estimate.
- Faithfulness impact: positive.
- Benchmarking impact: positive; removes a training/benchmark mismatch.
- Follow-up needed: none.

## Calibrated V2 Model 3 Training Study

The accepted study searched learning rate logarithmically over `[1e-5,1e-3]`
and optimizer batch size over `{256,512,1024,2048}` with
`TPESampler(seed=123)`. All ten requested trials completed; none failed or was
pruned. Every trial trained fresh models at `{-10,-5,0,5,10}` dB using the same
immutable data bank, at most 200 epochs, patience 20, relative minimum
improvement `1e-4`, and best-checkpoint restoration. The weighted-NMSE probe
used three batches of four channels per SNR and seed 4242.

The fixed bank contains 6,400 training and 256 validation channel realizations
at every canonical SNR. With the 4,096-example batch cap, each model sees
3,276,800 training and 131,072 validation frequency-block examples. The data
manifest SHA-256 is
`da52a76d051517397c86730a870b2466c897daaa456453a7278af6b2cb500a31`.
The search took 11:01:56. Trial 9 won with raw learning rate
`3.168916025578695e-05`, batch size 512, and objective
`0.004829016614663942`. Canonical-sized paired validation used seed 314159 and
selected rounded learning rate `3.17e-5` and batch size 512: its objective was
`0.004890910391028793`, versus `0.005637904627207314` for the freshly trained
prior recipe. Validation took 2:11:48.

The final eleven-SNR bank trained in 3:48:10 and passed its exact manifest,
payload, state-key, shape, dtype, and finite-value audit with SHA-256
`40595b56e5c74b9ca5f8a8b151576a53d01f2558b40003e57fd1f5018d438691`.
The first attempted v2 search was rejected because its writer saved the best
state but recorded metrics from the unrestored last state. The corrected
search restores `best_state` before computing metadata; no rejected asset or
runtime contributes to the accepted study.

With the bank frozen, ten detector trials at 10 dB proposed raw scale
`0.005575293412686262`, rounded to `0.00558`. On an independent 63-by-4-channel
paired set, that candidate produced BLER `0.020833333333333332`, while exact
incumbent `0.3405776426701012` produced `0.02017195767195767`; the incumbent
was retained. The 44.11-second canonical benchmark achieved five-SNR weighted
NMSE `0.004788725857026059`, 13.81% below the calibrated comparison worktree,
with lower BER and BLER at every SNR. The result is promoted and linked to the
exact eleven canonical manifest identities.

## Verification

- Commands run:
  - `python -m pytest -q test/mimo_ofdm_chest/papers/melgar2022/test_asset.py`
  - `python -m py_compile src/sionna_bench/mimo_ofdm_chest/papers/melgar2022/estimator.py src/sionna_bench/mimo_ofdm_chest/papers/melgar2022/asset.py src/sionna_bench/mimo_ofdm_chest/papers/melgar2022/tuning.py src/sionna_bench/mimo_ofdm_chest/papers/melgar2022/run_benchmark.py`
  - `python -m sionna_bench.mimo_ofdm_chest.papers.melgar2022.run_benchmark --smoke`
  - metadata inspection for the saved `n156` DNN asset.
  - long-training diagnostic command: `python -m sionna_bench.mimo_ofdm_chest.papers.melgar2022.run_benchmark --smoke --epochs 1000 --estimator-suffix=-ep1000`
  - h16 diagnostic command: `python -m sionna_bench.mimo_ofdm_chest.papers.melgar2022.run_benchmark --smoke --epochs 1000 --estimator-suffix=-h16-ep1000`
  - h64x3 diagnostic command: `python -u -m sionna_bench.mimo_ofdm_chest.papers.melgar2022.run_benchmark --smoke --epochs 1000 --estimator-suffix=-h64x3-ep1000`
  - serious asset-preparation command: `python -u -m sionna_bench.mimo_ofdm_chest.papers.melgar2022.run_benchmark --prepare-assets --asset-root assets/generated --train-batches 100 --train-batch-size 100 --validation-batches 20 --validation-batch-size 100 --epochs 10000 --optimizer-batch-size 500 --learning-rate 0.0005 --min-train-seconds 3600 --patience-epochs 10 --min-delta 1e-5`
  - full default benchmark command: `python -u -m sionna_bench.mimo_ofdm_chest.papers.melgar2022.run_benchmark --asset-root assets/generated --train-batches 100 --train-batch-size 100 --validation-batches 20 --validation-batch-size 100 --epochs 10000 --optimizer-batch-size 500 --learning-rate 0.0005 --min-train-seconds 3600 --patience-epochs 10 --min-delta 1e-5 --estimator-suffix=-serious-h64`
  - fixed-SNR smoke command: `python -u -m sionna_bench.mimo_ofdm_chest.papers.melgar2022.run_benchmark --smoke --prepare-assets --asset-root assets/smoke --train-batches 1 --train-batch-size 2 --validation-batches 1 --validation-batch-size 2 --epochs 2 --optimizer-batch-size 64 --training-snr-db 5 --patience-epochs 1 --min-delta 1e-6`
  - fixed-5 dB large asset-preparation command: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -u -m sionna_bench.mimo_ofdm_chest.papers.melgar2022.run_benchmark --prepare-assets --asset-root assets/generated --train-batches 1000 --train-batch-size 100 --validation-batches 125 --validation-batch-size 16 --epochs 100 --optimizer-batch-size 500 --learning-rate 0.0005 --patience-epochs 5 --min-delta 1e-5 --training-snr-db 5`
  - fixed-5 dB full default benchmark command: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -u -m sionna_bench.mimo_ofdm_chest.papers.melgar2022.run_benchmark --asset-root assets/generated --train-batches 1000 --train-batch-size 100 --validation-batches 125 --validation-batch-size 16 --epochs 100 --optimizer-batch-size 500 --learning-rate 0.0005 --patience-epochs 5 --min-delta 1e-5 --training-snr-db 5 --estimator-suffix=-trsnr5-100k`
  - mixed-SNR large asset-preparation command: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -u -m sionna_bench.mimo_ofdm_chest.papers.melgar2022.run_benchmark --prepare-assets --asset-root assets/generated --mixed-snr-weights --train-batches 1000 --train-batch-size 100 --validation-batches 125 --validation-batch-size 16 --epochs 100 --optimizer-batch-size 500 --learning-rate 0.0005 --patience-epochs 5 --min-delta 1e-5`
  - mixed-SNR 100000-channel full default benchmark command: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -u -m sionna_bench.mimo_ofdm_chest.papers.melgar2022.run_benchmark --asset-root assets/generated --mixed-snr-weights --train-batches 1000 --train-batch-size 100 --validation-batches 125 --validation-batch-size 16 --epochs 100 --optimizer-batch-size 500 --learning-rate 0.0005 --patience-epochs 5 --min-delta 1e-5 --estimator-suffix=-mixsnr100k`
  - SNR-specific 100000-channel asset-preparation command: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -u -m sionna_bench.mimo_ofdm_chest.papers.melgar2022.run_benchmark --prepare-assets --asset-root assets/generated --train-batches 1000 --train-batch-size 100 --validation-batches 125 --validation-batch-size 16 --epochs 100 --optimizer-batch-size 500 --learning-rate 0.0005 --patience-epochs 5 --min-delta 1e-5`
  - SNR-specific 100000-channel full default benchmark command: `python -u -m sionna_bench.mimo_ofdm_chest.papers.melgar2022.run_benchmark --asset-root assets/generated --train-batches 1000 --train-batch-size 100 --validation-batches 125 --validation-batch-size 16 --epochs 100 --optimizer-batch-size 500 --learning-rate 0.0005 --patience-epochs 5 --min-delta 1e-5`
  - refactor compile check: `python -m compileall src/sionna_bench/mimo_ofdm_chest/benchmarking src/sionna_bench/mimo_ofdm_chest/papers/melgar2022`
  - refactor PilotGroup regression check: `python -m pytest test/mimo_ofdm_chest/infrastructure/test_pilot_groups.py -q`
  - refactor cached smoke check through the runner: `run_benchmark(smoke_config, smoke=True)`
  - refactor cache-key checks confirmed that the default smoke cache and the current full mixed-SNR cache still match the regenerated key fields.
  - Model 2 detector calibration: `python -m sionna_bench.mimo_ofdm_chest.papers.melgar2022.tuning --study variance --method model2`.
  - Model 3 detector calibration: `python -m sionna_bench.mimo_ofdm_chest.papers.melgar2022.tuning --study variance --method model3`.
  - independent canonical-size detector holdouts for the Optuna candidates and
    the prior scale `1.0`, using seed `314159`.
  - the calibrated Sionna 2.0.1 Model 2 and Model 3 commands documented in
    `src/sionna_bench/mimo_ofdm_chest/papers/melgar2022/README.md`.
  - final regression suite: `python -m pytest -q test/mimo_ofdm_chest/papers/melgar2022`
    (`18 passed`).
  - same-batch pilot/full-grid diagnostic using Sionna's `BasePilotChannelEstimator` pilot gather path and active Melgar pilot groups.
  - fresh synthetic validation diagnostic using the same pair generator as DNN training with a new seed.
  - PCA bottleneck probe over held-out 156-dimensional real pilot vectors.
  - finite metric inspection over saved smoke `.npz` files.
- Smoke-test result:
  - The documented smoke command passed and saved `melgar-model2-dnn-pilot.npz`.
  - The post-refactor cached smoke check loaded `DNNChannelEstimator` through `evaluate_channel_estimator` and saved a temporary `DNNChannelEstimator.npz`, which was deleted after verification.
  - Saved one-batch smoke metrics at 5 dB: NMSE `1.539393e-01`, BER `1.485740e-01`, BLER `1.000000e+00`.
- Built-in baselines compared:
  - `ls-nn`: NMSE `2.623171e-02`, BER `8.752783e-02`, BLER `5.000000e-01`.
  - `ls-lin`: NMSE `2.190418e-02`, BER `8.166388e-02`, BLER `5.000000e-01`.
  - `lmmse`: NMSE `4.635212e-03`, BER `4.024763e-02`, BLER `3.750000e-01`.
- Sanity checks:
  - Saved smoke metrics are finite.
  - `err_var` is inherited from Sionna LS and is nonnegative by construction.
  - Smoke asset metadata records input dimension `156`, group length `156`, Sionna UMi data source, optimizer settings, and `ls_noise_scaling=no_over_abs_pilot_squared`.
  - Smoke asset validation MSE: `1.912407e-01`; LS validation MSE: `2.064986e-02`.
- Numerical credibility:
  - The current 156-dimensional smoke asset is much worse than LS. This is a red flag, not a validated benchmark result.
  - Long training to 1000 epochs did not fix the issue. Validation MSE fell from `3.131233e-01` at epoch 25 to about `1.5e-01` after epoch 725, ending at `1.538492e-01`, while the LS validation MSE stayed `2.064986e-02`.
  - The 1000-epoch saved smoke output at 5 dB was NMSE `1.531599e-01`, BER `3.1854e-01`, BLER `1.0`.
  - The h16 1000-epoch trace improved much more: validation MSE fell from `1.661083e-01` at epoch 25 to best `4.493884e-02` at epoch 975, ending at `4.667528e-02`. Full final train MSE was `4.022346e-02`.
  - The h16 saved smoke output at 5 dB was NMSE `3.494113e-02`, BER `1.157763e-01`, BLER `6.25e-01`.
  - The h64x3 1000-epoch trace crossed below LS validation MSE after about epoch 725 on the cached train/validation split. Best cached validation MSE was `1.642788e-02` at epoch 975; final cached validation MSE was `1.945443e-02`, and final full-train MSE was `1.399372e-02`.
  - The h64x3 saved one-batch smoke output at 5 dB was NMSE `7.034373e-02`, BER `1.6226e-01`, BLER `7.5e-01`.
  - A 16-batch NMSE-only check with the final h64x3 checkpoint gave `melgar-h64=1.081921e-01`, `ls-nn=2.987393e-02`, `ls-lin=1.469803e-02`, and `lmmse=9.859192e-03`.
  - Same-batch active-pilot/full-grid diagnostic at 5 dB over 64 smoke batches gave active-pilot NMSE `2.548100e-02` for LS and `9.726845e-02` for Melgar h64x3. The corresponding full-grid NMSE after Sionna linear interpolation was `1.608316e-02` for LS-linear and `9.676811e-02` for Melgar h64x3.
  - Fresh synthetic validation with a new channel seed gave real-coordinate MSE `2.056142e-02` for LS and `8.588159e-02` for Melgar h64x3, confirming that the cached validation improvement was not reproduced on fresh samples.
  - Serious one-hour training generated 2560000 real 156-dimensional training examples from 10000 channels and 512000 real validation examples from 2000 independent holdout channels. Holdout LS MSE over the full SNR mixture was `1.480641e-01`. Best validation MSE was `1.878637e-02` at epoch 163, with best-epoch training MSE `1.560615e-02`. The final epoch was 418 after `3603.45` seconds; final training MSE was `1.556807e-02` and final validation MSE was `1.963983e-02`, so the saved asset uses the best-validation checkpoint rather than the last epoch.
  - Same-batch live diagnostic for the serious asset at 5 dB over 32 default-config batches gave active-pilot NMSE `2.545337e-02` for LS and `1.035078e-02` for Melgar. The corresponding full-grid NMSE after Sionna linear interpolation was `1.612470e-02` for LS-linear and `1.011368e-02` for Melgar.
  - Historical full default benchmark metrics for the serious h64 asset were recorded before the output cleanup.
  - Full default NMSE for Melgar serious h64 across SNR `[-10, -7.5, -5, -2.5, 0, 2.5, 5, 7.5, 10, 12.5, 15]` dB was `[7.988612e-02, 4.631064e-02, 2.811853e-02, 1.837109e-02, 1.400929e-02, 1.306235e-02, 7.245636e-03, 9.180505e-03, 5.141742e-03, 8.026105e-03, 8.314212e-03]`.
  - At 5 dB in the full default sweep, Melgar serious h64 had NMSE `7.245636e-03`, BER `3.571390e-02`, and BLER `2.172619e-01`; LS-linear had NMSE `1.614313e-02`, BER `5.085857e-02`, and BLER `3.134921e-01`; LMMSE had NMSE `3.767772e-03`, BER `2.379197e-02`, and BLER `1.656746e-01`.
  - Fixed-5 dB large training generated 25600000 real 156-dimensional training examples from 100000 channels and 512000 real validation examples from 2000 independent holdout channels. Holdout LS MSE was `2.057016e-02`. Best validation MSE was `6.866017e-03` at epoch 20, with best-epoch training MSE `6.748488e-03`. Training stopped early at epoch 25 after `2210.4` optimizer seconds because validation did not improve by `1e-5` for 5 epochs. The final epoch's training MSE was `7.405620e-03` and validation MSE was `7.564394e-03`, so the saved asset uses the best-validation checkpoint rather than the last epoch.
  - Historical full default benchmark metrics for the fixed-5 dB asset were recorded before the output cleanup.
  - Full default NMSE for the fixed-5 dB asset across SNR `[-10, -7.5, -5, -2.5, 0, 2.5, 5, 7.5, 10, 12.5, 15]` dB was `[1.838504e-01, 1.040325e-01, 5.249816e-02, 3.239417e-02, 1.708205e-02, 9.488896e-03, 1.053208e-02, 7.279683e-03, 5.489341e-03, 4.746144e-03, 6.793848e-03]`.
  - At 5 dB in the fixed-5 dB full sweep, Melgar had NMSE `1.053208e-02`, BER `3.8931e-02`, and BLER `2.3413e-01`; LS-linear had NMSE `1.614313e-02`, BER `5.085857e-02`, and BLER `3.134921e-01`; LMMSE had NMSE `3.767772e-03`, BER `2.379197e-02`, and BLER `1.656746e-01`. The fixed-5 dB asset improves over LS-linear at 5 dB but is worse than the earlier mixed-SNR serious asset at 5 dB (`7.245636e-03` NMSE).
  - Mixed-SNR large training generated 25600000 real 156-dimensional training examples from 100000 channels and 512000 real validation examples from 2000 independent holdout channels. Holdout LS MSE was `1.413071e-01`. Best validation MSE was `1.661705e-02` at epoch 30, with best-epoch training MSE `1.622887e-02`. Training stopped early at epoch 35 after `3111.4` optimizer seconds because validation did not improve by `1e-5` for 5 epochs. The final epoch's training MSE was `1.663354e-02` and validation MSE was `1.701652e-02`, so the saved asset uses the best-validation checkpoint rather than the last epoch. Relative to LS, the best validation ratio was `0.1176`, better than the earlier 10000-channel mixed run (`0.1269`) and the fixed-5 dB run (`0.3338`).
  - The post-refactor full default benchmark for the mixed-SNR 100000-channel asset was saved to `outputs/melgar2022/3gpp-umi/DNNChannelEstimator.npz` before the SNR-specific rerun; the old method-named `.npz` result and generated plot were removed from the output directory.
  - Post-refactor full default NMSE for the mixed-SNR 100000-channel asset across SNR `[-10, -7.5, -5, -2.5, 0, 2.5, 5, 7.5, 10, 12.5, 15]` dB was `[6.171739e-02, 4.155864e-02, 2.816607e-02, 2.019856e-02, 1.271033e-02, 8.059776e-03, 1.139673e-02, 8.785467e-03, 5.524380e-03, 7.081898e-03, 9.234877e-03]`.
  - At 5 dB in the post-refactor mixed-SNR 100000-channel full sweep, Melgar had NMSE `1.139673e-02`, BER `2.954146e-02`, and BLER `2.043651e-01`; LS-linear had NMSE `1.614313e-02`, BER `5.085857e-02`, and BLER `3.134921e-01`; LMMSE had NMSE `3.767772e-03`, BER `2.379197e-02`, and BLER `1.656746e-01`. The mixed-SNR 100000-channel asset improves over LS-linear at 5 dB but is worse than the earlier 10000-channel mixed asset at 5 dB NMSE (`7.245636e-03`).
  - The pre-promotion archive preserves the pre-Sionna-2.0.1 canonical
    benchmark, which used one prepared DNN
    asset per SNR and the schema-3 configuration with evaluation seed `2026`.
    The retained Model 2 output is
    `DNNChannelEstimator.npz`; its NMSE over
    SNR `[-10, -7.5, -5, -2.5, 0, 2.5, 5, 7.5, 10, 12.5, 15]` dB is
    `[4.698305e-02, 3.261523e-02, 2.204486e-02, 1.595047e-02, 1.275509e-02, 1.100557e-02, 9.467702e-03, 9.811638e-03, 1.124631e-02, 7.851594e-03, 9.495686e-03]`.
    Mean NMSE, BER, and BLER are `1.720247e-02`, `7.402138e-02`, and
    `4.028078e-01`, respectively. This payload is retained only as the
    pre-regeneration comparison reference.
  - The superseded initial Sionna 2.0.1 Model 3 candidate was stored under
    `benchmark_output/sionna-2.0.1-seedfix/melgar2022/3gpp-umi/`;
    its NMSE is `[9.411422e-02, 6.050508e-02, 3.829770e-02, 2.369156e-02,
    1.452431e-02, 8.957506e-03, 5.520702e-03, 3.238458e-03, 1.965614e-03,
    1.211988e-03, 8.121850e-04]`. Mean NMSE, BER, and BLER are
    `2.298539e-02`, `6.365112e-02`, and `3.411797e-01`, respectively.
  - The promoted calibrated v2 Model 3 result has NMSE
    `[9.207146e-02, 5.896127e-02, 3.686300e-02, 2.295963e-02, 1.406740e-02,
    8.612731e-03, 5.168823e-03, 3.137101e-03, 1.853556e-03, 1.124965e-03,
    7.306132e-04]`. Its five-SNR weighted NMSE is `0.004788725857026059`,
    13.81% below the calibrated comparison result.
  - The promoted calibrated Model 2 result has NMSE
    `[4.849086e-02, 3.257865e-02, 2.343215e-02, 1.666506e-02, 1.332787e-02,
    1.031226e-02, 9.751628e-03, 8.078198e-03, 7.200751e-03, 7.612416e-03,
    5.452624e-03]`. Its five-SNR weighted NMSE is `0.008809663267045341`,
    versus `0.011443368547024409` for the comparison reference. All NMSE
    confidence intervals overlap, while BER and BLER improve materially over
    most of the curve.
  - Variance scaling does not change `h_hat` or NMSE. Differences from the
    preceding legacy result block therefore reflect the current deterministic
    evaluation contract, while the calibrated scales affect BER and BLER.
  - PCA probe on held-out real pilot vectors showed rank-8 reconstruction MSE `1.112830e-01`, rank-16 `3.873865e-02`, rank-32 `8.370255e-03`, rank-64 `2.328784e-04`. This suggests the paper's 8-neuron bottleneck is too small for the benchmark's 156-dimensional UMi pilot vectors.
- Red flags:
  - The small h64x3 run only beats LS on the cached train/validation split; on fresh synthetic samples and live same-batch active pilots it is worse than LS.
  - For the serious run, validation saturated around epoch 163 and later epochs did not improve the holdout; the trainer now saves the best-validation checkpoint.
  - The serious asset beats LS-linear NMSE through 10 dB, but degrades relative to LS-linear at 12.5 and 15 dB, likely because the single network was trained across the full SNR mixture without SNR conditioning.
  - The serious asset remains worse than LMMSE across the full default sweep.
  - An initial fixed-5 dB large run with validation `20 x 100` hit CUDA OOM during validation channel generation after the 100000-channel training set had been collected. The rerun kept 2000 validation channels but generated them as `125 x 16` to reduce temporary OFDM channel memory.
  - The fixed-5 dB asset has strong pilot-domain holdout MSE but does not translate to the best full-grid 5 dB NMSE after interpolation; the previous mixed-SNR serious asset remains better at 5 dB full-grid NMSE.
  - The mixed-SNR 100000-channel asset has the best pilot-domain validation ratio so far, but that does not translate uniformly to the full-grid benchmark: it improves the old mixed asset at several SNR points, but is worse at 5 dB and 10 dB.
- Unverified items:
  - Training with the paper's exact 800000 data points, 20-channel CDL-A
    dataset, and 1000 epochs.
  - Input/output normalization or architecture changes that would require revised review.
