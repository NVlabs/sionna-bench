# Plan Review

- Paper ID: `melgar2022`
- Reviewer: Codex
- Date: 2026-05-05
- Reviewed artifacts: `src/sionna_bench/mimo_ofdm_chest/papers/melgar2022/paper.pdf`, `reports/paper_summary.md`, `reports/plan.md`, `AGENTS.md`, `prompts/reviewer.md`, `prompts/task_reviewer.md`, and `tasks/channel_estimation.md`.

## Summary Verdict

Implementation may proceed for `melgar-model2-dnn-pilot` only.

The plan correctly identifies Model 2 as the only first-pass method that preserves the repository's mandatory pilot-domain estimator plus explicit interpolation split. Model 1 and Model 3 are learned full-grid reconstruction/post-filtering methods and are correctly deferred pending trained assets and a reviewed learned interpolator design.

## Harness Review

The plan adequately summarizes:

- The fixed resource grid, UMi channel, Kronecker comb pilots, and two pilot-bearing symbols.
- The estimator input and output tensor shapes.
- The built-in baselines (`ls-nn`, `ls-lin`, `lmmse`, `perfect`).
- BER, BLER, and NMSE metric definitions.
- Smoke-test expectations and output layout.

No harness-blocking issue found.

## Method Review

### Method: `melgar-model1-preliminary-dnn`

- Plan status: `defer`
- Reviewer verdict: accepted.
- Rationale: Section III-B1 estimates one target resource from LS pilot values plus coordinates. This can be made task-compatible only through a learned interpolation/reconstruction component; a direct full-grid raw-observation estimator would violate the channel-estimation task contract. The paper itself presents Model 1 as preliminary and computationally inefficient.
- May implementation proceed now? No.

### Method: `melgar-model2-dnn-pilot`

- Plan status: `implement`
- Reviewer verdict: accepted with caveats.
- Rationale: Section III-B2 explicitly denoises LS estimates at pilot resources and then uses linear interpolation. This matches a Sionna `BaseChannelEstimator` with `PilotGroupMixin`: Sionna extracts pilots, the paper DNN refines grouped pilot estimates, and Sionna performs explicit linear interpolation. The plan preserves Eq. (6), the DNN mapping of Eq. (11), real/imaginary separated inference, the Section III-D DNN hidden structure, MSE/RMSprop training, and the linear interpolation stage.
- Required caveats:
  - The implementation must name the method as a benchmark-trained pilot-domain Model 2 adaptation, not as a reproduction of the paper's published CDL-A curves.
  - The checkpoint metadata must record active pilot length, endpoint dimension adaptation, Sionna UMi data source, training SNRs, training budget, and optimizer settings.
  - Adapting Model 2's input/output layer from 48 to the active pilot-vector length is acceptable because the benchmark numerology changes the number of pilot observations processed by one DNN call. The first hidden layer remains 1000 neurons and the output layer must use the same adapted dimension as the input layer.
  - Reduced training budgets are acceptable for smoke/default execution, but the report must avoid claiming paper-level convergence.
- May implementation proceed now? Yes.

### Method: `melgar-model3-dnn-full-grid`

- Plan status: `defer`
- Reviewer verdict: accepted.
- Rationale: Section III-B3 applies the DNN after LS plus full-grid linear interpolation, so the method is a learned full-grid post-filter. It should not be hidden inside `estimate_at_pilot_locations`; it needs a reviewed custom interpolator and trained assets.
- May implementation proceed now? No.

## Divergence Ledger

### Divergence: Model 2 Endpoint Dimensions

- Source: `interface`
- Reviewer verdict: `necessary`
- Rationale: The paper uses 48 pilot resources in a 288-subcarrier pilot block, while the active Sionna Kronecker pilot pattern gives 156 pilots per stream and pilot symbol. The DNN endpoint dimensions must match the active pilot observation vector. The hidden architecture and learned pilot-denoising objective are preserved.
- Impact on faithfulness: Moderate; endpoint dimensions change with numerology, but this is the direct Model 2 analogue for the active benchmark.
- Impact on benchmarking fairness: Acceptable if metadata records the benchmark-specific pilot layout.

### Divergence: Training Data Source

- Source: `asset`
- Reviewer verdict: `justified`
- Rationale: The paper's trained weights and MATLAB CDL-A datasets are absent. Training on the active benchmark UMi channel is necessary for a benchmarkable estimator and is analogous to generating covariance or learned auxiliary assets for this repository.
- Impact on faithfulness: Moderate; results cannot be compared directly to the paper's CDL-A numerical curves.
- Impact on benchmarking fairness: Acceptable if the method is reported as benchmark-trained and compared against baselines generated on the same config.

### Divergence: Reduced Training Budget

- Source: `asset`
- Reviewer verdict: `justified`
- Rationale: The paper's 800000 examples and 1000 epochs are not reasonable for mandatory smoke checks. A smaller checkpoint can validate the implementation path; larger assets can be produced later.
- Impact on faithfulness: Moderate to high for numerical performance, low for algorithmic structure.
- Impact on benchmarking fairness: Acceptable for smoke only; full benchmark claims require reporting the exact budget used.

### Divergence: Error Variance Fallback

- Source: `interface`
- Reviewer verdict: `necessary`
- Rationale: The paper optimizes channel MSE and does not define detector-facing error variance. The task contract requires `err_var`. Reusing the LS pilot error variance is the least invasive fallback and corresponds to `chest_err_const=1.0`.
- Impact on faithfulness: Low for channel-estimation output, because `err_var` is not part of the paper method.
- Impact on benchmarking fairness: Medium for BER/BLER; must be documented because the detector consumes it.

### Divergence: Sionna Linear Interpolation

- Source: `interface`
- Reviewer verdict: `justified`
- Rationale: The paper states that Model 2 uses 2D linear interpolation after pilot denoising. Passing `interpolation_type="lin"` explicitly is consistent with the task contract.
- Impact on faithfulness: Low.
- Impact on benchmarking fairness: Low; comparable to the `ls-lin` baseline.

## Required Implementation Conditions

- Use Sionna `BaseChannelEstimator` with `PilotGroupMixin`, not direct `AbstractEstimator`.
- Call Sionna LS pilot estimation rather than manually bypassing pilot extraction.
- Pass explicit `interpolation_type="lin"`.
- Save DNN weights under `aux/` with metadata sufficient to reproduce the training asset.
- Include a paper-local runner that evaluates the Melgar estimator through `evaluate_channel_estimator` and leaves built-in baselines/dashboard generation to the shared tooling.
- Report smoke metrics honestly, especially if the small trained DNN is worse than `ls-nn`.

## Final Plan Review Verdict

Proceed with `melgar-model2-dnn-pilot` under the caveats above. Keep `melgar-model1-preliminary-dnn` and `melgar-model3-dnn-full-grid` deferred.
