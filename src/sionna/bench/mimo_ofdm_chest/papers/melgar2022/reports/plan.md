# Implementation Plan

- Paper ID: `melgar2022`
- Full citation: Antonio Melgar, Alejandro de la Fuente, Leopoldo Carro-Calvo, Oscar Barquero-Perez, and Eduardo Morgado, "Deep Neural Network: An Alternative to Traditional Channel Estimators in Massive MIMO Systems," IEEE Transactions on Cognitive Communications and Networking, vol. 8, no. 2, pp. 657-671, June 2022.
- Task: channel estimation
- Implementer: Codex
- Date: 2026-05-05

## Harness Summary

- Benchmark config: `configs/benchmark_config.py` fixes a 4-UE uplink OFDM MU-MIMO benchmark with 14 OFDM symbols, 624 subcarriers, 15 kHz spacing, cyclic prefix length 44, 4x4 single-polarized BS array for 16 receive antennas, single-antenna UEs, UMi channel at 2.6 GHz, pathloss/shadowing disabled, UE speed 3 km/h, 16-QAM rate 0.6, and Eb/N0 sweep `[-10, 15]` dB in 2.5 dB steps.
- Smoke config: `configs/smoke_config.py` uses batch size 2, one Eb/N0 point at 5 dB, one BER/BLER Monte Carlo batch, one NMSE batch, and reduced covariance-estimation settings. It is an integration and sanity check, not final performance evidence.
- Resource grid and pilots: `evaluation.system.build_resource_grid` creates a Sionna `ResourceGrid` with `pilot_pattern="kronecker"` and pilot OFDM symbols `[2, 11]`. With the default config each stream has comb pilots on those symbols.
- Task base class: Sionna `BaseChannelEstimator`; first-pass paper estimators implement `estimate_at_pilot_locations(y_pilots, no)` and pass an explicit interpolation choice or interpolator. Shared pilot-vector grouping is provided by `src.pilot_groups.PilotGroupMixin`.
- Benchmark input contract: estimator call receives `y : [B, num_rx, num_rx_ant, num_ofdm_symbols, fft_size]` complex and scalar/broadcastable noise variance `no`.
- Benchmark output contract: estimator returns `h_hat : [B, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm_symbols, fft_size]` complex and `err_var` with the same shape as float.
- Built-in baselines: `src/sionna_bench/mimo_ofdm_chest/papers/builtin/` exposes `ls-nn`, `ls-lin`, `lmmse`, and `perfect`. `ls-nn` and `ls-lin` use Sionna `LSChannelEstimator` with explicit nearest-neighbor or linear interpolation. `lmmse` uses LS at pilots plus Sionna `LMMSEInterpolator` with empirical covariance assets from `src/sionna_bench/mimo_ofdm_chest/papers/builtin/asset.py`.
- Metrics: `evaluation.evaluate_channel_estimator` computes BER, BLER, and NMSE from `OFDMChannelEstimationSystem` batches by comparing transmitted bits, decoded bits, true channel, and estimated channel.
- Output layout: built-in outputs go under `outputs/builtin/<config-slug>/` or `outputs/builtin/smoke/<config-slug>/`. Paper outputs go under `outputs/<paper_id>/<config-slug>/` or `outputs/<paper_id>/smoke/<config-slug>/`.

## Paper Method Inventory

### Method: `melgar-model1-preliminary-dnn`

- Paper location: Section III-B1, Fig. 5; model assessed in Section IV-A.
- Paper equations / algorithms / statements: uses LS pilot estimates from Eq. (6); DNN mapping follows Eq. (11); paper states the model estimates one target resource from a 66-dimensional vector of 32 complex LS pilot values plus target coordinates.
- Paper inputs: 32 LS pilot CSI values from a 24 subcarrier by 28 OFDM-symbol block, plus `(f_i, t_i)` target coordinates.
- Paper outputs: real and imaginary parts of one target channel resource.
- Required side information: trained DNN weights from the paper's supervised dataset.
- Target repository abstraction: deferred; would need a learned `BaseChannelInterpolator` or reviewed full-grid wrapper carrying a DNN from pilot estimates and coordinates to arbitrary grid locations.
- Status: `defer`

#### Faithful Implementation Detail

No first-pass code will be written.

```text
1. Extract LS pilot estimates from Sionna pilot locations, corresponding to Eq. (6).
2. For each target resource, assemble paper-style features: real LS pilots, imaginary LS pilots, and resource coordinates as in Section III-B1.
3. Apply the trained DNN mapping f_theta from Eq. (11).
4. Write the resulting complex scalar into the full resource-grid estimate.
```

#### Interface Adaptations

- The paper block spans 28 OFDM symbols, while the benchmark frame has 14 symbols. A faithful implementation would require a reviewed block definition for this benchmark.
- A direct `AbstractEstimator` full-grid implementation is not allowed by the task contract. This should be implemented as a custom interpolator if pursued later.

#### Auxiliary Asset Plan

- Required asset: trained Model 1 DNN weights and training metadata.
- Proposed path: `aux/<aux_slug>/melgar2022_model1_dnn/`.
- Why the paper requires it: Model 1 is a learned estimator; inference is undefined without learned `theta`.
- Source or generation procedure: generate supervised examples from the active benchmark channel, extract LS pilot estimates and true channel labels, train the paper's ReLU DNN with RMSprop and MSE.
- Validation checks: finite weights, validation MSE below LS-only MSE on held-out channel samples, target-coordinate invariance checks, and smoke comparison against `ls-nn`.
- Future-agent resume steps: design a task-compatible `BaseChannelInterpolator`, train/store weights, then add the method to the runner.

#### Risks

- Missing paper details: exact coordinate scaling, random seeds, data normalization, and model checkpoints.
- Numerical risks: expensive per-resource inference and possible overfitting.
- Faithfulness risks: changing the 28-symbol block shape could change the learned estimator.
- Benchmarking fairness risks: full-grid learned interpolation could gain or lose from benchmark-specific retraining.

### Method: `melgar-model2-dnn-pilot`

- Paper location: Section III-B2, Fig. 6; Section III-C; Section III-D; Section IV.
- Paper equations / algorithms / statements: Eq. (6) defines LS pilot CSI; Eq. (11) defines the learned DNN map; Section III-B2 says the DNN improves pilot-resource CSI and then 2D linear interpolation estimates remaining resources; Section III-B2 uses a 48-input/48-output DNN for the paper's pilot block, while Section III-D specifies hidden layers `1000, 500, 256, 128, 32, 16, 8`, ReLU hidden activation, linear output, MSE loss, RMSprop optimizer, learning rate `0.0005`, batch size `500`, and 1000 epochs.
- Paper inputs: real or imaginary parts of LS pilot CSI over one frequency block dedicated to pilot signaling.
- Paper outputs: improved real or imaginary parts of pilot-resource CSI for the same block.
- Required side information: trained DNN weights. No covariance matrix is needed at inference time.
- Target repository abstraction: Sionna `BaseChannelEstimator` with `PilotGroupMixin`, `LSChannelEstimator.estimate_at_pilot_locations` for Eq. (6), a paper DNN denoiser at pilot locations, and explicit Sionna linear interpolation (`interpolation_type="lin"`) to produce full-grid CSI.
- Status: `implement`

#### Faithful Implementation Detail

```text
Construction:
1. Initialize `BaseChannelEstimator(resource_grid, interpolation_type="lin")` to preserve Sionna pilot extraction and paper-stated linear interpolation.
2. Use `PilotGroupMixin(group_over=("subcarrier",))` to group each `(tx, stream, pilot OFDM symbol)` into a sorted pilot-frequency vector.
3. Let `Np` be the active benchmark pilot-vector length for one such group. In the current config this is 156 rather than the paper's 48 because the resource grid and pilot pattern differ.
4. Build the paper DNN with input dimension `Np`, hidden layers `[1000, 500, 256, 128, 32, 16, 8]`, ReLU activations, and output dimension `Np`. This keeps the paper's hidden-layer architecture and changes only the endpoint dimensions needed by the benchmark numerology. Use one real-valued network shared for real and imaginary parts, matching Section III-B2's statement that the parts are estimated separately and can share the same model.
5. Load a matching `aux/<aux_slug>/melgar2022_model2_dnn/...` checkpoint. If absent, generate a finite supervised training asset from the active benchmark channel:
   a. Sample true OFDM channel realizations from Sionna UMi using the benchmark resource grid.
   b. Extract true pilot-frequency vectors for all `(tx, stream, pilot symbol)` groups.
   c. Draw training SNRs from the active config's Eb/N0 range and form noisy LS pilot vectors `h_LS = h + n_LS`, with variance `no / |pilot|^2` after Sionna pilot division, corresponding to Eq. (6).
   d. Add real-part pairs `(real(h_LS), real(h))` and imaginary-part pairs `(imag(h_LS), imag(h))`.
   e. Train with MSE loss, RMSprop, learning rate `0.0005`, and mini-batch size `500`, matching Section III-D. The number of examples and epochs are reduced for repository smoke/default runtime and recorded as an asset metadata field.

Inference:
1. Call `LSChannelEstimator.estimate_at_pilot_locations(self, y_pilots, no)` to obtain `h_LS` and LS pilot error variance, implementing Eq. (6) through Sionna's pilot division.
2. For each `(tx, stream, pilot symbol)` group:
   a. Gather the compact pilot-vector entries for that group.
   b. Flatten leading batch/receiver/antenna dimensions.
   c. Run the DNN once on the real part and once on the imaginary part, as in Section III-B2.
   d. Recombine the outputs into complex pilot CSI.
   e. Scatter the denoised pilot CSI back into Sionna's compact pilot-vector layout.
3. Return the denoised pilot CSI and LS-derived pilot error variance. Sionna then applies explicit linear interpolation to the full resource grid.
```

#### Interface Adaptations

- The paper's Model 2 endpoint dimensions are adapted from 48 to the active benchmark pilot group length, currently 156. The first hidden layer remains 1000 neurons and the output layer uses the same adapted endpoint dimension.
- The paper's 2D linear interpolation is mapped to Sionna's explicit `interpolation_type="lin"`.
- The paper's MATLAB CDL-A training data are replaced by Sionna UMi samples from the active benchmark. This is necessary to train a benchmark-compatible estimator and will be recorded as a benchmark-domain adaptation.
- The paper's 800000 training points and 1000 epochs are too large for mandatory smoke checks. The runner will expose training budget flags, and the saved asset metadata will state the actual finite budget used.
- The paper does not define `err_var`; the implementation will use the LS pilot error variance, equivalent to the task-contract fallback with `chest_err_const=1.0`.

#### Surrogate Check

- The plan does not replace the paper's learned pilot-denoising objective, DNN architecture family, optimizer, loss, or inference algorithm.
- It adapts endpoint dimensions and data source to the benchmark and reduces training budget for runtime. These are documented interface/runtime adaptations, not a different estimator.

#### Auxiliary Asset Plan

- Required asset: trained Model 2 DNN checkpoint.
- Proposed path: `aux/<aux_slug>/melgar2022_model2_dnn/n<Np>_tb<train_batches>_bs<train_batch_size>_ep<epochs>_mb<optimizer_batch_size>_lr<learning_rate>_seed<seed>/`.
- Why the paper requires it: Eq. (11) depends on learned parameters `theta`; no pretrained weights are provided.
- Source or generation procedure: generate true channel samples from Sionna's benchmark UMi model, synthesize LS pilot observations at random training SNRs, and train the paper DNN with MSE/RMSprop.
- Validation checks: metadata match to config/pilot layout and active input/output dimension, finite weights, training and validation MSE logged, denoised pilot MSE compared with LS pilot MSE on a held-out split, finite `h_hat`/`err_var`, and smoke NMSE compared with `ls-nn`.
- Future-agent resume steps: increase training budget toward the paper's 800000 examples and 1000 epochs, add held-out default benchmark validation, and save canonical assets under the same path convention.

#### Risks

- Missing paper details: exact data normalization and MATLAB channel sampling are not provided.
- Numerical risks: the 8-neuron bottleneck may underfit the active UMi pilot vectors when training budget is small.
- Faithfulness risks: training on UMi instead of CDL-A and adapting Model 2 endpoint dimensions limit direct comparability to the paper's reported figures.
- Benchmarking fairness risks: if trained on the exact benchmark distribution, this learned estimator has distribution-specific knowledge similar to empirical covariance baselines. This is acceptable only if the training data and generation settings are reported.

### Method: `melgar-model3-dnn-full-grid`

- Paper location: Section III-B3, Fig. 7; Section IV-B; Fig. 13.
- Paper equations / algorithms / statements: uses LS from Eq. (6), applies 2D linear interpolation, then DNN mapping from Eq. (11) over all resources in a 72-subcarrier block.
- Paper inputs: real or imaginary parts of LS-plus-linear-interpolation CSI over one full frequency block.
- Paper outputs: improved real or imaginary parts of all channel resources in the block.
- Required side information: trained Model 3 DNN weights.
- Target repository abstraction: deferred; would require a custom learned `BaseChannelInterpolator` or reviewed equivalent that can run the DNN after Sionna linear interpolation.
- Status: `defer`

#### Faithful Implementation Detail

No first-pass code will be written.

```text
1. Compute LS pilot estimates using Eq. (6).
2. Apply 2D linear interpolation to all channel resources as in Section III-B3.
3. Split each OFDM symbol into frequency blocks.
4. Apply the trained DNN f_theta from Eq. (11) independently to real and imaginary interpolated CSI blocks.
5. Return the DNN-refined full-grid channel estimate.
```

#### Interface Adaptations

- The method is naturally a learned full-grid post-filter. The current task allows first-pass custom interpolation only when the scope is simple and reviewed. This learned full-grid DNN should be deferred until a clean `BaseChannelInterpolator` design exists.

#### Auxiliary Asset Plan

- Required asset: trained Model 3 DNN checkpoint and metadata.
- Proposed path: `aux/<aux_slug>/melgar2022_model3_dnn/`.
- Why the paper requires it: inference requires learned parameters `theta`.
- Source or generation procedure: generate full-grid LS-plus-linear-interpolation inputs and true channel labels from the active benchmark, train the Section III-D DNN with Model 3 dimensions.
- Validation checks: full-grid validation MSE below LS-linear interpolation, finite full-grid output, no hidden default interpolation, smoke comparison against `ls-lin`.
- Future-agent resume steps: implement a learned interpolator carrying Model 3 weights, review it, then add method to runner.

#### Risks

- Missing paper details: exact handling of frequency-block edges and arbitrary pilot patterns.
- Numerical risks: full-grid DNN inference can be expensive.
- Faithfulness risks: block-size adaptation from 72 subcarriers may be needed for the 624-subcarrier benchmark.
- Benchmarking fairness risks: learned full-grid refinement may be hard to compare with pilot-only methods unless asset generation is standardized.

## Cross-Cutting Decisions

- Shared assumptions: paper DNN weights are unavailable, so the implementation will train reusable `aux/` assets from the active benchmark channel distribution.
- Shared helper modules: add a paper-local training helper only for DNN asset generation and loading.
- Parameters taken from the paper: hidden layer sizes, ReLU hidden activation, linear output, MSE loss, RMSprop optimizer, learning rate `0.0005`, batch size default `500`.
- Parameters requiring reviewer approval: adapting Model 2 input/output dimensions to the active pilot group length, training on Sionna UMi rather than MATLAB CDL-A, and using reduced training budgets for smoke/default runs.
- Planned smoke command: `python -m sionna_bench.mimo_ofdm_chest.papers.melgar2022.run_benchmark --smoke`.
- Required sanity checks: `py_compile`; finite metrics; nonnegative `err_var`; smoke comparison against `ls-nn`, `ls-lin`, and `lmmse`; inspect saved `.npz` output.

## Questions For Reviewer

- Can Model 2 proceed as a benchmark-domain learned pilot denoiser with explicit linear interpolation, while Model 1 and Model 3 remain deferred pending learned full-grid interpolator designs and larger trained assets?
