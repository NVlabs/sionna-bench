# Final Review Report

- Paper ID: `edfors1996`
- Reviewer: Codex
- Date: 2026-08-04
- Revision note: audit updated after the `_PilotGroupMixin` and shared
  `evaluate_channel_estimator` runner refactor. Updated again after
  LS-only Optuna tuning and independent holdout validation retained
  `num_taps=64`, which is reused for all three variants. Updated on 2026-07-09
  after the transform-domain and confidence
  propagation audit and aggregate detector-facing calibration. Updated again
  after adopting the repository-wide `lin_time_avg` interpolation default and
  after Sionna 2.0.1 asset regeneration, tuning, and benchmark validation.
- Reviewed sources: paper PDF, `reports/paper_summary.md`, `reports/plan.md`,
  `reports/plan_review.md`, `reports/implementation_report.md`,
  `estimator.py`, `asset.py`, `run_benchmark.py`, and `tuning.py`.

## Code Audit

The implementation follows the reviewed plan as a pilot-domain adaptation.
`TruncatedChannelEstimator` now inherits from Sionna's
`BasePilotChannelEstimator` and uses Sionna's private `_PilotGroupMixin` for
compact pilot-vector grouping. It uses Sionna LS pilot extraction, implements the paper's
IDFT-domain transformations for the `LMMSE`, `diagonal LMMSE`, and `LS`
variants, and exposes Sionna's standard interpolation interface with
`interpolation_type="lin_time_avg"` as the default.

The DFT ordering and covariance application match Chapter 3 and Appendix A.
The implementation uses a unitary IDFT/DFT pair, orders transformed
coefficients by decreasing covariance diagonal, applies the selected full,
diagonal, or identity matrix, and zeroes excluded coefficients. The full LMMSE
solve is algebraically `R_L(R_L+sigma2 I)^(-1)`.

The runner now follows the shared evaluation path: it calls
`evaluate_channel_estimator`, writes one `.npz` file per requested variant,
saves the config snapshot, and records each result in the scenario-local
`results.json`. It no longer runs built-in baselines or creates combined plots.

The asset helper owns the reusable empirical time-domain covariance lifecycle.
It pools all pilot groups into one shared covariance, while metadata records
the active configuration, covariance budget, seed, group layout, covariance
scope, and IDFT normalization. Explicit preparation calls
`estimate_covariance()` and inference calls strict `load_covariance()`; both
return the selected manifest path alongside the injected matrix. Estimator
construction performs no file I/O or covariance estimation.

## Verification Review

Verified commands:

- `compileall` passed for `src/sionna_bench/mimo_ofdm_chest/papers/edfors1996/estimator.py` and
  `src/sionna_bench/mimo_ofdm_chest/papers/edfors1996/run_benchmark.py`.
- `pytest test/mimo_ofdm_chest/infrastructure/test_pilot_groups.py` passed.
- A direct smoke-sized `evaluate_channel_estimator` run for
  `method="diagonal LMMSE"` passed and loaded the existing smoke covariance
  asset.
- A `run_benchmark(..., smoke=True)` runner check passed and wrote
  `TruncatedChannelEstimator (diag. LMMSE).npz` beneath the smoke output tree.
- A final calibrated `python -m sionna_bench.mimo_ofdm_chest.papers.edfors1996.run_benchmark --smoke` run
  passed for all three methods; its transient results and covariance asset were
  removed after validation.
- `python -m sionna_bench.mimo_ofdm_chest.papers.edfors1996.run_benchmark` passed with the regenerated validated
  covariance asset and calibrated `chest_err_const=0.308`, and
  saved
  `TruncatedChannelEstimator (LMMSE).npz`,
  `TruncatedChannelEstimator (diag. LMMSE).npz`, and
  `TruncatedChannelEstimator (LS).npz`.
- The regenerated default covariance asset has shape `(156, 156)`, all entries
  are finite, Hermitian residual is `0.0`, and metadata records
  `covariance_scope=pooled_all_pilot_groups`,
  `array_format=single_matrix_2d`, `num_groups=8`, and `subcarrier_groups`.
- A targeted same-batch check confirmed that explicit all-tap Estimator C
  (`--num-taps 156`) matches `ls-lin` to numerical precision, which validates
  the DFT/IDFT normalization and scatter/gather path.
- Permanent transform-limit tests confirm all-tap LS equivalence, all-tap
  LMMSE equivalence to Sionna's `LMMSEChannelEstimator` for the corresponding
  covariance, correct application of a complex non-diagonal covariance,
  strongest-tap truncation, and exact recovery of a delay-limited channel.
- A permanent geometry test confirms that unequal active pilot-frequency group
  lengths fail during construction with the documented specific message.
- A confidence-propagation test confirms that the returned full-grid
  `err_var` is exactly `chest_err_const` times linearly interpolated LS
  observation variance. It is not a paper-derived posterior covariance.
- All three methods match eager output under
  `torch.compile(..., fullgraph=True)`. `torch._dynamo.explain` reports one
  graph and no graph breaks on the alternate geometry; covariance asset work
  is completed before estimator construction and is not compiled.
- `python -m sionna_bench.mimo_ofdm_chest.papers.edfors1996.tuning --study taps` completed 50
  Optuna trials for one shared tap count. Each trial evaluated the `LS` variant
  only and minimized weighted NMSE over five SNRs. Optuna selected 61 taps,
  but paired canonical-size validation favored `num_taps=64`; this value is
  reused unchanged for `LMMSE` and `diagonal LMMSE`.
- `python -m sionna_bench.mimo_ofdm_chest.papers.edfors1996.tuning --study variance` completed
  the shared 10-trial detector calibration over all three methods at 10 dB.
  Optuna selected raw `chest_err_const=0.4540521216492649`, but paired
  canonical-size validation favored the existing `0.308`, which remains the
  canonical value.
- `python -m sionna_bench.mimo_ofdm_chest.papers.edfors1996.run_benchmark` passed and overwrote the
  canonical `.npz` files for all three variants. The paper runner now uses
  `64` as its default tap count and the calibrated variance scale.

Smoke results are numerically credible for an integration run: all variants
produce finite metrics, nonnegative propagated error variances, and NMSE better
than `ls-nn` at the smoke SNR. No smoke failure was hidden.

LS-tuned `L=64` benchmark results show that `LMMSE` is the strongest
Edfors variant by NMSE over most of the Eb/N0 sweep, while
`diagonal LMMSE` is not the best tradeoff under this pilot-domain
adaptation. The 64-tap setting still shows a truncation-related floor at higher
SNRs.

Compared with the previous canonical benchmark, median NMSE ratios remain
within roughly one percent for every method. Every regenerated NMSE, BER, and
BLER point lies inside the corresponding previous 95% confidence interval, so
the calibrated Sionna 2.0.1 rerun is statistically consistent with the prior
result.

## Divergence Ledger

| Divergence | Source | Reviewer verdict | Rationale | Impact on paper faithfulness | Impact on benchmarking fairness |
| --- | --- | --- | --- | --- | --- |
| Pilot-domain vectors replace the paper's all-tone known-symbol LS vector | interface | necessary | The current benchmark exposes pilots only and requires the pilot/interpolation split. | Material: this is a benchmark adaptation, not the literal all-tone receiver. | Fair if reported; comparisons should be against pilot-domain baselines. |
| Empirical pooled Sionna UMi covariance replaces analytical paper covariance | asset | justified | The active benchmark channel differs from the paper's analytical model; the estimator needs matching second-order statistics. Pooling all pilot groups is appropriate because these statistics should not depend on transmit index, stream index, or pilot OFDM symbol under the benchmark channel law. | Moderate: covariance source changes, but the estimator structure is preserved. | Fair with cached metadata and shared config. |
| Explicit Sionna `"lin_time_avg"` interpolation fills off-pilot resources | interface | necessary | The paper estimates a known-tone vector and does not define sparse-pilot full-grid interpolation. The common repository default adds OFDM-time averaging. | Moderate: full-grid behavior is partly benchmark interpolation. | Fair because the choice is explicit and shared by pilot-domain paper estimators. |
| Task-contract fallback error variance is returned for detector use | interface | justified | The paper provides MSE analysis but not the Sionna `err_var` tensor after sparse-pilot adaptation and interpolation. The implementation uses `chest_err_const*no/|pilot|^2`; a calibrated Sionna 2.0.1 aggregate detector study over all three methods produced raw winner `0.4540521216492649`, while paired canonical-size validation retained canonical value `0.308`. | Low for channel estimates; affects BER/BLER through detector uncertainty. | Fair because the objective, statistical budget, three-digit selection, and paired validation are disclosed. |
| LS-tuned shared tap count, `L=64` | implementation | justified | The paper treats `L` as a complexity/performance parameter. The canonical benchmark selects `L` by optimizing the `LS` variant and then reuses the same value for `LMMSE` and `diagonal LMMSE`, avoiding separate per-variant tuning. The all-tap setting remains available with `--num-taps 156`. | Moderate: the benchmark uses a tuned pilot-domain truncation level rather than the paper's analytical simulation examples. | Fair because the same `L` is shared by all three variants and the tuning protocol is disclosed. |

## Findings

No blocking findings.

Residual caveats:

- The implementation should be described as "Edfors DFT pilot-domain
  adaptation" rather than an exact reproduction of the paper's all-tone
  receiver.
- The paper's conclusion favoring Estimator B does not carry over directly to
  this benchmark adaptation; full-grid interpolation and pilot-only observation
  geometry materially affect the ordering.
- The LS-tuned `L=64` outputs are not the no-error-floor all-tap setting; high
  SNR comparisons should account for truncation.
- Smoke results are transient and are removed after validation; the canonical
  output directory and manifest contain the calibrated
  `TruncatedChannelEstimator (...)` curves.
- BER/BLER interpretation should mention that `err_var` is an empirically
  calibrated task-contract adaptation rather than the paper's analytical MSE.

## Verdict

Accepted with caveats for benchmark use as a first-pass pilot-domain
reimplementation of the paper's DFT estimators A, B, and C. The literal
all-tone known-symbol receiver remains incompatible with the current benchmark
contract unless the harness adds an all-pilot/training-symbol or
decision-directed estimator mode.
