# Final Review Report

- Paper ID: `melgar2022`
- Reviewer: Codex
- Date: 2026-05-05
- Revision note: updated after the implementation was refactored from `PilotDomainEstimator` plus local pilot-group records to Sionna's pilot-domain base class plus shared pilot grouping, and after `run_benchmark.py` was moved to `evaluate_channel_estimator`.
- Post-review revision, 2026-07-16: updated the factual audit after Model 3 was
  implemented, checkpoint preparation moved to `asset.py`, SNR-specific assets
  became the canonical path, and the paper architecture and training histories
  were reread.
- Post-review compilation revision, 2026-07-16: verified Tier A complete
  inference for either model with one mixed-SNR state and Tier B inference for
  canonical per-SNR banks at an explicit model-selection boundary. Training
  and checkpoint I/O remain outside estimator construction and inference.
- Post-review variance revision, 2026-07-16: calibrated separate Model 2 and
  Model 3 detector-facing scales with the shared Optuna and independent
  holdout protocol, then regenerated both canonical outputs.
- Sionna 2.0.1 Model 3 revision, 2026-08-06: a fresh 20-trial training study
  and independent holdout selected learning rate `1.1e-5` and optimizer batch
  size `256`; all eleven checkpoints passed strict validation. A refreshed
  detector holdout retained scale `0.341`, and the isolated candidate improved
  weighted NMSE through 10 dB by 9.7%. Model 2 regeneration and joint result
  promotion were postponed at that stage.
- Calibrated v2 revision, 2026-08-24: the preceding Model 3 study was
  superseded by the frozen v2 protocol. Ten full five-SNR trials on one
  deterministic 6,400/256-channel bank selected learning rate `3.17e-5` and
  optimizer batch size 512. The final eleven-SNR bank, detector calibration,
  canonical benchmark, comparison, and exact result-to-asset linkage passed
  all completion audits under calibrated Sionna commit
  `f16208347246d8d17ec69628dfb856b19287b80c` and explicit UMi v19.2.
- Calibrated Model 2 revision, 2026-09-02: ten full five-SNR trials and
  canonical-sized paired validation selected learning rate `7.02e-5` and
  optimizer batch size 256. The final eleven-SNR bank passed restored-state,
  manifest, payload, and strict-loader validation. Detector calibration
  selected `err_var_scale=0.454`; the canonical benchmark and exact
  result-to-asset linkage passed.

## Final Verdict

- Verdict: `accepted with caveats`
- Faithful enough for benchmarking? Structurally yes for the implemented Model
  2 pilot denoiser and Model 3 full-grid post-filter, subject to the documented
  architecture, channel-model, interpolation, and training adaptations.
- Main caveats:
  - The implementation is trained on Sionna UMi benchmark samples, not the paper's MATLAB CDL-A dataset.
  - Model 2 endpoint dimensions are adapted from the paper's 48 to the benchmark's 156 pilot values per stream and pilot OFDM symbol.
  - The active h64x3 architecture uses user-requested hidden layers
    `1024, 512, 256, 128, 64, 64, 64`, which is not the exact paper
    architecture.
  - Both canonical methods use SNR-specific model banks trained on 6,400
    channels and validated on 256 independent channels per SNR. Model 2 still
    exhibits a high-SNR NMSE floor despite improving substantially over its
    previous canonical reference.
- Required fixes: none for the calibrated Model 2 and Model 3 benchmark paths.

## Findings

### Finding: Model 2 Preserves The Required Pilot-Domain Split

- Severity: low
- Description: `DNNChannelEstimator` inherits Sionna `BasePilotChannelEstimator`,
  uses `_PilotGroupMixin` for compact pilot-vector grouping, calls Sionna LS
  pilot estimation, and denoises grouped pilot estimates before configured
  Sionna interpolation. The paper's 2D linear interpolation is available as
  `interpolation_type="lin"`; the repository-standard canonical default is
  `"lin_time_avg"`.
- Evidence: `src/sionna_bench/mimo_ofdm_chest/papers/melgar2022/estimator.py`.
- Recommended disposition: accept.

### Finding: Runner Now Uses The Shared Evaluation Path

- Severity: low
- Description: `src/sionna_bench/mimo_ofdm_chest/papers/melgar2022/run_benchmark.py` evaluates the selected
  `DNNChannelEstimator` or `InterpolatedDNNChannelEstimator` through
  `evaluate_channel_estimator`, saves class-named `.npz` results, writes the
  config snapshot, and updates the paper manifest.
- Evidence: `src/sionna_bench/mimo_ofdm_chest/papers/melgar2022/run_benchmark.py`.
- Recommended disposition: accept.

### Finding: Endpoint Dimensions Are Now Numerology-Compatible

- Severity: low
- Description: the DNN input and output layers both use the active pilot-group
  length. For the current benchmark this is 156, connected to the adapted
  1024-neuron first hidden layer and 156 output neurons.
- Evidence: `src/sionna_bench/mimo_ofdm_chest/papers/melgar2022/estimator.py` and `src/sionna_bench/mimo_ofdm_chest/papers/melgar2022/asset.py`.
- Recommended disposition: accept as an interface adaptation.

### Finding: Serious Holdout Training Fixes The Small-Data Failure

- Severity: high
- Description: the h64x3 1000-epoch small-data diagnostic reaches best cached validation MSE `1.642788e-02`, below LS validation MSE `2.064986e-02`, but fails fresh/live pilot checks. The subsequent serious run used 10000 training channels and 2000 independent validation channels. It reached best validation MSE `1.878637e-02` versus holdout LS MSE `1.480641e-01`. In the full default benchmark, Melgar serious h64 beats LS-linear NMSE through 10 dB, e.g. at 5 dB NMSE `7.245636e-03` versus LS-linear `1.614313e-02`; BER `3.571390e-02` versus `5.085857e-02`; BLER `2.172619e-01` versus `3.134921e-01`.
- Evidence: `reports/implementation_report.md`.
- Recommended disposition: accept as a much more credible trained asset, but report high-SNR degradation and LMMSE gap.

### Finding: Model 2 Retains A High-SNR Error Floor

- Severity: medium
- Description: the regenerated benchmark uses one calibrated DNN asset per
  SNR, but its high-SNR NMSE remains `7.612416e-03` and `5.452624e-03` at
  12.5 and 15 dB. This is substantially better than the previous 15 dB result
  but remains an error floor relative to estimators with stronger structural
  priors.
- Evidence: `reports/implementation_report.md`.
- Recommended disposition: future work should try residual denoising,
  identity-preserving high-SNR regularization, or stronger validation criteria
  tied to full-grid NMSE rather than only pilot-domain MSE.

### Finding: Fixed-SNR Training Is Now Supported

- Severity: low
- Description: the runner can train explicit fixed-SNR assets via
  `--training-snr-db` and prepares one asset per configured benchmark SNR by
  default. `--mixed-snr-weights` instead requests one shared model. The current
  saved benchmark uses per-SNR assets.
  Its full-grid NMSE across SNR
  `[-10, -7.5, -5, -2.5, 0, 2.5, 5, 7.5, 10, 12.5, 15]` dB is
  `[6.034027e-02, 3.770162e-02, 2.457133e-02, 1.706145e-02, 1.265474e-02, 9.145125e-03, 7.506953e-03, 7.515958e-03, 6.211712e-03, 5.968683e-03, 8.359977e-03]`.
- Evidence: `src/sionna_bench/mimo_ofdm_chest/papers/melgar2022/run_benchmark.py` and `reports/implementation_report.md`.
- Recommended disposition: accept for benchmark use with the documented
  high-SNR caveat.

### Finding: Larger Mixed-SNR Training Improves Pilot Validation

- Severity: low
- Description: the 100000-channel mixed-SNR run cycles over the full benchmark SNR grid and stopped at epoch 35, keeping epoch 30. Best validation MSE was `1.661705e-02` versus holdout LS MSE `1.413071e-01`, a relative ratio of `0.1176`. This is better than the previous 10000-channel mixed run's relative ratio `0.1269` and much better than the fixed-5 dB run's relative ratio `0.3338`. In the full-grid benchmark, its 5 dB NMSE was `9.345172e-03`, better than LS-linear `1.614313e-02` but worse than the previous 10000-channel mixed asset `7.245636e-03`.
- Evidence: `reports/implementation_report.md`.
- Recommended disposition: do not assume pilot-domain validation MSE selects the best full-grid checkpoint. Keep the full benchmark result separate from the older mixed asset and compare by leaderboard metric.

### Finding: 8-Neuron Bottleneck Likely Does Not Scale To 156 Inputs

- Severity: high
- Description: a PCA probe over held-out 156-dimensional real pilot vectors gives rank-8 reconstruction MSE `1.112830e-01` and rank-16 `3.873865e-02`, still above LS input MSE. Rank 32 is the first tested bottleneck that gets clearly below LS.
- Evidence: `reports/implementation_report.md`.
- Recommended disposition: future work should review a wider bottleneck or normalized/residual training before running default benchmarks.

### Finding: Model 1 Remains Not Implemented And Model 3 Is Implemented

- Severity: low
- Description: Model 1 remains not implemented. Model 3 is now implemented
  under the later full-grid estimator contract as LS-linear interpolation
  followed by independent real/imaginary DNN processing of frequency blocks.
- Evidence: `src/sionna_bench/mimo_ofdm_chest/papers/melgar2022/estimator.py` and
  `reports/implementation_report.md`.
- Recommended disposition: accept Model 3 with the documented 78-subcarrier
  block and benchmark-trained-weight adaptations.

### Finding: Calibrated Training Counts Are Internally Consistent

- Severity: low
- Description: both calibrated model banks use 6,400 training and 256
  validation channel realizations per SNR. Model 2 derives 1,638,400 training
  and 65,536 validation pilot-vector examples per model. Model 3 caps each
  generated batch at 4,096 examples, producing 3,276,800 training and 131,072
  validation examples per model. These are DNN data-point counts, distinct
  from the paper's 20 full channel realizations.
- Evidence: retained asset `generation_recipe` and `metadata.history` sections.
- Recommended disposition: accept; keep both units explicit in documentation.

## Method Audit

### Method: `melgar-model2-dnn-pilot`

- Paper method faithfully implemented? Partially. The pilot denoising structure, loss, optimizer, real/imaginary split, and linear interpolation are represented. Endpoint dimensions and hidden-layer widths are benchmark/user-adapted.
- Approved plan followed? Yes after the plan revision to direct 156-dimensional endpoints.
- Surrogate-rule compliance: compliant. No LS fallback, blending, or alternate denoiser was introduced to hide underperformance.
- Interface adaptations documented? Yes.
- Smoke/sanity checks acceptable? Yes. The final assets pass strict loading,
  best-state validation, and the canonical benchmark.
- Remaining risk: moderate for paper-to-benchmark faithfulness due to the
  benchmark-domain channel and adapted architecture.

### Method: `melgar-model1-preliminary-dnn`

- Paper method faithfully implemented? No; not implemented.
- Approved plan followed? Yes.
- Surrogate-rule compliance: compliant.
- Interface adaptations documented? Yes.
- Smoke/sanity checks acceptable? Not applicable.
- Remaining risk: requires custom learned interpolator and trained asset.

### Method: `melgar-model3-dnn-full-grid`

- Paper method faithfully implemented? Partially. The LS-linear-before-DNN
  order, frequency blocking, shared real-valued DNN, and separate real/imaginary
  inference match Section III-B3. Block size, architecture, training channel,
  and training budget are adapted.
- Approved plan followed? Implemented after the repository gained the reviewed
  full-grid estimator contract.
- Surrogate-rule compliance: compliant.
- Interface adaptations documented? Yes.
- Smoke/sanity checks acceptable? Yes for integration; canonical performance is
  represented by a retained per-SNR asset set.
- Remaining risk: benchmark-domain training and the adapted architecture limit
  direct comparison with the paper's CDL-A curves.

## Divergence Ledger

### Divergence: Sionna UMi Training Instead Of MATLAB CDL-A

- Source: `asset`
- Reviewer verdict: `justified`
- Rationale: the paper weights and datasets are absent, while the repository benchmark evaluates Sionna UMi.
- Impact on faithfulness: moderate.
- Impact on benchmarking fairness: acceptable with metadata and caveats.

### Divergence: 156-Dimensional Endpoint Adaptation

- Source: `interface`
- Reviewer verdict: `necessary`
- Rationale: the active benchmark presents 156 pilot observations per stream and pilot OFDM symbol. The DNN endpoint dimensions must match the vector being denoised.
- Impact on faithfulness: moderate.
- Impact on benchmarking fairness: acceptable structurally, but numerically weak with current training.

### Divergence: Reduced Training Budget

- Source: `asset`
- Reviewer verdict: `justified`
- Rationale: smoke tests use deliberately small budgets, while canonical
  assets use the shared full-effort 6,400/256-channel, 200-epoch protocol.
- Impact on faithfulness: low for the canonical numerical results.
- Impact on benchmarking fairness: acceptable under the shared tuning and
  paired-validation protocol.

### Divergence: 16-Neuron Bottleneck Diagnostic

- Source: `implementation`
- Reviewer verdict: `questionable`
- Rationale: user-requested diagnostic; it is a modest architecture relaxation but no longer exactly follows Section III-D.
- Impact on faithfulness: moderate.
- Impact on benchmarking fairness: improves performance, but still not enough for benchmark claims.

### Divergence: h64x3 Hidden Architecture

- Source: `implementation`
- Reviewer verdict: `questionable`
- Rationale: user-requested diagnostic; it improves default NMSE over LS-linear across most SNR points after serious holdout training, but deviates from Section III-D.
- Impact on faithfulness: high relative to exact paper architecture.
- Impact on benchmarking fairness: acceptable as an explicitly documented,
  identically trained benchmark-domain architecture adaptation.

### Divergence: LS Error Variance Used As `err_var`

- Source: `interface`
- Reviewer verdict: `necessary`
- Rationale: the paper does not provide detector-facing error variance, and
  Sionna requires it. Independent detector calibrations selected
  `0.454` times LS pilot variance for Model 2 and
  `0.3405776426701012` times LS-linear variance for Model 3. In the final v2
  Model 3 calibration, rounded candidate `0.00558` had paired BLER
  `0.0208333333`, while that incumbent had `0.0201719577`, so the incumbent
  was retained.
- Impact on faithfulness: low.
- Impact on benchmarking fairness: acceptable because selection used an
  independent canonical-size 10 dB paired comparison and the canonical
  benchmark was not used to revise the scale.

## Suitability Assessment

- Strengths of the paper for this task: Model 2 naturally decomposes into
  pilot-domain denoising followed by explicit interpolation, while Model 3
  naturally maps to the repository's full-grid estimator contract.
- Weaknesses of the paper for this task: no weights are provided, the published
  setup differs substantially from the benchmark, and the exact 8-neuron
  bottleneck does not scale well to the adapted vectors.
- Required assets or benchmark changes for the remaining method: Model 1 would
  require its target-coordinate DNN path and separately trained weights.

## Calibrated V2 Canonical Update

The final Model 2 study used the same 6,400/256-channel policy. Ten completed
trials selected raw learning rate `7.01799283113844e-05` and optimizer batch
size 256 with objective `0.004693918396679923`. Canonical-sized paired
validation selected rounded `7.02e-5`/256 with objective
`0.008802547101945238`, versus `0.012489951946848734` for the prior recipe.
The final bank trained in 6:12:29 and all 11 saved payloads reproduced their
best-validation metrics exactly. Detector validation selected scale `0.454`.
The canonical NMSE curve is
`[4.849086e-02, 3.257865e-02, 2.343215e-02, 1.666506e-02, 1.332787e-02,
1.031226e-02, 9.751628e-03, 8.078198e-03, 7.200751e-03, 7.612416e-03,
5.452624e-03]`; five-SNR weighted NMSE is `0.008809663267045341`, versus
`0.011443368547024409` for the comparison reference.

The final Model 3 study used one immutable bank containing 6,400 training and
256 validation channels at every canonical SNR. Ten completed trials searched
learning rate `[1e-5,1e-3]` and batch size `{256,512,1024,2048}` at
`{-10,-5,0,5,10}` dB with the common 200-epoch, patience-20 protocol. No trial
failed or was pruned. Raw winner `3.168916025578695e-05`/512 had objective
`0.004829016614663942`; canonical-sized paired validation selected rounded
`3.17e-5`/512 with objective `0.004890910391028793`, versus
`0.005637904627207314` for the freshly trained prior recipe.

The final eleven-model bank trained in 3:48:10 and passes exact asset audit
SHA-256 `40595b56e5c74b9ca5f8a8b151576a53d01f2558b40003e57fd1f5018d438691`.
Its canonical NMSE curve is
`[9.207146e-02, 5.896127e-02, 3.686300e-02, 2.295963e-02, 1.406740e-02,
8.612731e-03, 5.168823e-03, 3.137101e-03, 1.853556e-03, 1.124965e-03,
7.306132e-04]`. Weighted NMSE is `0.004788725857026059`, 13.81% below the
calibrated comparison worktree, with lower BER and BLER at every SNR. The
44.11-second benchmark result is linked one-to-one to the eleven promoted
manifest hashes. The earlier 20-trial/1,024-channel Model 3 study remains
historical evidence only; its assets and result are not canonical.

Final verdict: accepted with caveats for both implemented methods. The
canonical assets are benchmark-domain adaptations, not reproductions of the
paper's published CDL-A curves.
