# Final Review Report: Li et al., 2020

- Paper ID: `li2020`
- Reviewer: Codex reviewer
- Date: 2026-06-23
- Reviewed artifacts: `AGENTS.md`, `prompts/reviewer.md`,
  `prompts/task_reviewer.md`, `prompts/task.md`, `src/sionna_bench/mimo_ofdm_chest/papers/li2020/paper.pdf`,
  `reports/paper_summary.md`, `reports/plan.md`,
  `reports/plan_review.md`, `estimator.py`, `asset.py`, `tuning.py`,
  `run_benchmark.py`, `README.md`, `reports/implementation_report.md`,
  `test/mimo_ofdm_chest/papers/li2020/test_asset.py`, and
  `test/mimo_ofdm_chest/papers/li2020/test_estimator.py`.
- Follow-up review: re-read the updated `asset.py`, `README.md`, and
  `implementation_report.md` after the SNR augmentation fix.
- Post-review compilation revision, 2026-07-17: verified Tier A complete
  inference with one mixed-SNR state and Tier B inference with the canonical
  per-SNR bank. The only observed graph break is the explicit prepared-state
  selector; eager and compiled outputs agree.
- Post-review canonical revision, 2026-07-17: validated the frozen per-SNR
  assets, calibrated `err_var_scale`, and regenerated the canonical benchmark
  output and leaderboard entry.

## Summary

The implementation follows the approved architecture: `ReEsNetChannelEstimator`
is a Sionna `BaseChannelEstimator` using `_PilotGroupMixin` and direct
`call(y, no)`. This is the right contract for Li et al.'s ReEsNet because the
paper's proposed method maps LS pilot estimates to a full OFDM channel image,
rather than only estimating at pilot locations.

The core model is faithful to the paper's Section III-C architecture: a
two-channel real/imaginary LS pilot image, 16-filter input convolution, four
residual blocks, a post-block convolution, transposed-convolution upsampling,
and a final two-channel convolution. The implementation also uses L1 training
loss, matching Equation (8).

The implementation now has a validated canonical per-SNR asset bank, an
independently confirmed detector-facing variance calibration, and a current
canonical benchmark output.

The previously identified multi-SNR training issue is resolved. The training
data path now loops over every requested training SNR for each sampled channel
batch, matching the paper's statement that multiple noisy SNR versions of a
sample are generated for training. The cache metadata records this with
`snr_augmentation=each_channel_batch_all_training_snrs`.

## Findings

### Finding 1: Canonical Training, Benchmark, And Calibration

- Status: resolved by post-review canonical revision
- Source: `asset`, `interface`
- Evidence: 11 frozen per-SNR checkpoints trained from the common 6,400/256
  data bank are present under `assets/`. Ten full five-SNR trials and paired
  validation selected optimizer batch size 256 and rounded learning rate
  `0.000186`. The canonical output links all 11 exact manifests.

### Finding 2: Detector-Facing `err_var_scale`

- Status: resolved by post-review canonical revision
- Source: `interface`
- Evidence: the estimator applies the calibrated default
  `0.23432889192166198` to its bilinearly upsampled LS error-variance image.
  Ten logarithmic BLER trials at 10 dB proposed rounded scale `0.224`; paired
  validation tied it with the incumbent at BLER `0.03042328`, so the
  strict-better rule retained the incumbent.
- Impact on faithfulness: low for channel estimates, because Li et al. do not
  define detector-facing variances. Impact on BER/BLER can be significant.
- Impact on benchmarking fairness: acceptable; the completed protocol and
  results are reported.

### Finding 3: Focused Tests Cover Layout And Shape, But Not Cache/Training Semantics

- Status: resolved by follow-up tests
- Source: `implementation`
- Evidence: the expanded estimator suite now checks residual blocks, the long
  skip, super-resolution dimensions, ordered pilot layouts, exact
  linear/pilot consistency, and stream/receive-antenna isolation over three
  system geometries. `test_asset.py` trains and reloads tiny one-SNR models for
  two geometries under temporary asset roots.
- Remaining scope: malformed checkpoint contents and incompatible manifest
  metadata are covered by separate open todos and are not claimed here.

## Divergence Ledger

### Divergence: Direct Full-Grid Estimator Instead Of Sionna Pilot-Domain Split

- Source: `interface`
- Reviewer verdict: `necessary`
- Rationale: ReEsNet's Equation (7) is a full-grid learned reconstruction from
  pilot LS estimates. Direct `call(y, no)` preserves that mapping.
- Impact on paper faithfulness: positive.
- Impact on benchmarking fairness: acceptable; output tensors match the
  Sionna benchmark contract.

### Divergence: Active Nonzero Pilot Observations Instead Of Full Sionna Mask

- Source: `interface`
- Reviewer verdict: `necessary`
- Rationale: Sionna's Kronecker mask includes zero-valued pilot placeholders
  for stream orthogonality. Li et al.'s `Y_p, X_p` model corresponds to
  nonzero pilot observations.
- Impact on paper faithfulness: low and justified.
- Impact on benchmarking fairness: acceptable; the implementation tests the
  active `156 x 2` image layout.

### Divergence: Benchmark Dimensions `156 x 2 -> 624 x 14`

- Source: `interface`
- Reviewer verdict: `necessary`
- Rationale: the paper uses a `72 x 14` single-antenna downlink grid with 24
  or 48 pilots, while the repository benchmark uses the UMi MU-MIMO grid.
- Impact on paper faithfulness: moderate, but the same estimator structure is
  preserved.
- Impact on benchmarking fairness: acceptable because all methods use the same
  benchmark grid and pilot pattern.

### Divergence: Shared Network Across UE Streams And BS Antennas

- Source: `interface`
- Reviewer verdict: `justified`
- Rationale: the paper evaluates single-antenna downlink channels. The
  benchmark is uplink MU-MIMO, so each stream/receive-antenna channel image is
  treated as one ReEsNet example.
- Impact on paper faithfulness: moderate.
- Impact on benchmarking fairness: acceptable with disclosure.

### Divergence: Local Benchmark-Trained Weights

- Source: `asset`
- Reviewer verdict: `necessary`
- Rationale: no pretrained weights or original simulator dataset are provided.
  A local `assets/` artifact is required.
- Impact on paper faithfulness: moderate; results are benchmark adaptations,
  not reproductions of the paper's published curves.
- Impact on benchmarking fairness: acceptable after training metadata and
  validation results are reported.

### Divergence: Compact Finite-Dataset Storage

- Source: `implementation`
- Reviewer verdict: `justified`
- Rationale: the enlarged Sionna 2.0.1 replacement corpus cannot be retained
  in available memory as `float32` full-grid targets. Generated inputs and
  targets are stored in `float16`, and every minibatch is converted to
  `float32` before model inference and loss computation.
- Impact on paper faithfulness: low; this changes finite-dataset storage
  precision, not the network, loss, or optimizer computation.
- Impact on benchmarking fairness: acceptable with explicit manifest and
  README disclosure.

### Divergence: Multi-SNR Training Augmentation

- Source: `implementation`
- Reviewer verdict: `justified`
- Rationale: Li et al. state that multiple SNR versions of each training
  sample are generated. The updated implementation now samples one channel
  batch and generates one noisy LS pilot-image version for every requested
  training SNR before moving to the next channel batch. Cache metadata records
  the rule as `each_channel_batch_all_training_snrs`.
- Impact on paper faithfulness: positive relative to the previous version;
  this now matches the paper's multi-SNR training mechanism at the benchmark
  data-generation level.
- Impact on benchmarking fairness: acceptable with the existing cache metadata
  and README disclosure of the selected SNR grid.

### Divergence: Bilinear LS Error-Variance Fallback

- Source: `interface`
- Reviewer verdict: `necessary`
- Rationale: the paper does not define detector-facing marginal error
  variances. A fallback is required by the Sionna benchmark.
- Impact on paper faithfulness: low for NMSE, moderate for BER/BLER.
- Impact on benchmarking fairness: acceptable after the completed
  `err_var_scale` validation.

## Verification

I ran:

```bash
python -m py_compile src/sionna_bench/mimo_ofdm_chest/papers/li2020/estimator.py src/sionna_bench/mimo_ofdm_chest/papers/li2020/asset.py src/sionna_bench/mimo_ofdm_chest/papers/li2020/tuning.py src/sionna_bench/mimo_ofdm_chest/papers/li2020/run_benchmark.py test/mimo_ofdm_chest/papers/li2020/test_asset.py test/mimo_ofdm_chest/papers/li2020/test_estimator.py
python -m pytest -q test/mimo_ofdm_chest/papers/li2020
python -m sionna_bench.mimo_ofdm_chest.papers.li2020.tuning --study training
python -m sionna_bench.mimo_ofdm_chest.papers.li2020.tuning --study training-validation
python -m sionna_bench.mimo_ofdm_chest.papers.li2020.run_benchmark
```

Result:

```text
py_compile passed
focused promotion tests passed
```

The canonical weighted NMSE is `0.00427391`, versus `0.00498370` in the
calibrated comparison worktree. All per-SNR NMSE confidence intervals overlap,
and the comparison was not used to revise the frozen recipe.

## Final Verdict

Accepted for the repository's canonical adapted benchmark, with the documented
paper-to-benchmark deviations and detector-facing variance fallback.
