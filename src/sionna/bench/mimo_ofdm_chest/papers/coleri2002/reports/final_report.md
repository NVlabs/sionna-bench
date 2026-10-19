# Final Review Report

- Paper ID: `coleri2002`
- Reviewer: Codex
- Date: 2026-06-03
- Revision note: audit after refactoring the Coleri implementation to use `_PilotGroupMixin`, renaming the public classes to `CombInterpolator` and `LSCombEstimator`, fixing OFDM-time handling to linear interpolation between pilot-bearing symbols, switching the paper runner to `evaluate_channel_estimator`, and rerunning the full benchmark with renamed output files.

## Final Verdict

- Verdict: `accepted with caveats`
- Faithful enough for benchmarking? Yes, for the four implemented comb-LS interpolation variants.
- Main caveats: frequency-boundary rules and linear OFDM-time interpolation are benchmark adaptations; `LSCombEstimator (low-pass)` follows the public MathWorks interpolation algorithm family but its canonical FIR length/cutoff are Optuna-tuned benchmark parameters, not MATLAB defaults; LMS/MMSE/block methods are not implemented.
- Required fixes before final numerical publication: none specific to the
  Sionna 2.0.1 Coleri rerun; `leaderboard/data.json` was regenerated.

## Findings

### Finding: Coleri Estimator Structure Is Now Symmetric

- Severity: low
- Description: all four Coleri variants are now selected through `LSCombEstimator(method=...)`; there is no one-off time-domain estimator class.
- Evidence: `src/sionna_bench/mimo_ofdm_chest/papers/coleri2002/estimator.py` defines `CombInterpolator` and `LSCombEstimator`; `src/sionna_bench/mimo_ofdm_chest/papers/coleri2002/run_benchmark.py` imports `LSCombEstimator`.
- Recommended disposition: accepted.

### Finding: Paper Runner Uses Shared Evaluation Entry Point

- Severity: low
- Description: `run_benchmark.py` now follows the Delfeld-style runner pattern and delegates metrics/output writing to `evaluate_channel_estimator`.
- Evidence: the runner calls `evaluate_channel_estimator` for each Coleri method, writes `LSCombEstimator (<method>).npz` files, saves the config snapshot, and updates the paper manifest without loading built-in baselines or plotting.
- Recommended disposition: accepted.

### Finding: Pilot Grouping Uses Shared Infrastructure

- Severity: low
- Description: the comb interpolator now uses `_PilotGroupMixin` instead of duplicating Sionna pilot-vector grouping logic.
- Evidence: `CombInterpolator` inherits `BaseChannelInterpolator, _PilotGroupMixin` and calls `_init_pilot_groups(pilot_pattern, group_over=("subcarrier",))`.
- Recommended disposition: accepted.

### Finding: Linear OFDM-Time Interpolation Is The Fixed Policy

- Severity: medium
- Description: the implementation no longer exposes a time-averaging option. Reconstructed pilot-bearing OFDM symbols are mapped to all OFDM symbols by linear interpolation.
- Evidence: `_interpolate_time` takes `pilot_symbol_indices` and returns all `num_ofdm_symbols` in one call. The exact synthetic-input comparison against the previous linear-time branch had max channel/error differences of `0.000e+00` for all four variants.
- Recommended disposition: accepted as the benchmark interface adaptation.

### Finding: Low-Pass Is Documentation-Faithful, Not MATLAB-Bit-Exact

- Severity: low
- Description: `LSCombEstimator (low-pass)` implements the public zero-insertion/FIR algorithm family from MathWorks docs, but exact closed MATLAB coefficients were not validated. The class defaults and canonical benchmark use the retained settings `lowpass_filter_half_length=16` and `lowpass_cutoff=0.105`; the documented MATLAB-style settings `4` and `0.5` remain available as explicit arguments.
- Evidence: implementation uses least-squares low-pass FIR design, finite filtering with virtual constant edge pilots, forced pilot pass-through, and the tuning and paired-validation evidence recorded in `tuning/coleri2002/lowpass_v2/`.
- Recommended disposition: acceptable for benchmark reporting if the tuned FIR parameters are disclosed; future MATLAB reference vectors would improve confidence.

### Finding: Low-Pass Hyperparameters Were Benchmark Tuned

- Severity: medium
- Description: The canonical low-pass configuration uses a 50-trial Optuna search over `lowpass_filter_half_length` and `lowpass_cutoff`. The objective was weighted NMSE over Eb/N0 points `{-10, -5, 0, 5, 10}` dB with linear-SNR weights, `num_mc_batches=3`, and `batch_size=4`.
- Evidence: `src/sionna_bench/mimo_ofdm_chest/papers/coleri2002/tuning.py` and the `best_params.json`, `trials.csv`, and `paired_validation.json` evidence under `tuning/coleri2002/lowpass_v2/`.
- Recommended disposition: accept as a benchmark calibration, not as a paper-specified parameter choice.

### Finding: Detector-Facing Variance Scale Was Calibrated

- Severity: low
- Description: all four variants expose one shared `err_var_scale`. In the calibrated-source v2 10-trial detector-objective study, raw winner `0.4540521216492649` was rounded to `0.454`. On independent canonical-size paired validation, the rounded candidate scored mean BLER `0.057953042328042326`, while incumbent `0.868` scored `0.055803571428571425`. The incumbent is therefore retained.
- Evidence: `src/sionna_bench/mimo_ofdm_chest/papers/coleri2002/tuning.py` and the `best_params.json`, `trials.csv`, and `paired_validation.json` evidence under `tuning/coleri2002/variance_scale_v2/`.
- Recommended disposition: accepted; reuse `0.868` for every Coleri method unless the estimator or receiver contract changes.

### Finding: Printed Equation Corrections Remain Justified

- Severity: medium
- Description: equation (19) prints `alpha = l/N`, and equation (21) has inconsistent zero-padding index ranges. The implementation treats both as paper typos rather than literal formulas.
- Evidence: second-order interpolation uses `alpha = l/L`, consistent with equation (18) and `k=mL+l`; time-domain interpolation uses pilot-rate IDFT, middle zero padding, and DFT reconstruction.
- Recommended disposition: accepted; keep these corrections documented and do not revert to the literal printed formulas.

### Finding: Full Default Results Were Rerun

- Severity: medium
- Description: The Coleri full default benchmark was rerun after the shared
  tuning RNG correction. All NMSE, BER, and BLER samples for all four methods
  exactly match the preserved pre-fix Sionna 2.0.1 outputs, confirming that
  retaining the canonical tuning parameters leaves benchmark behavior
  unchanged.
- Evidence: the default output directory contains the four canonical
  `LSCombEstimator (<method>).npz` files regenerated with Sionna 2.0.1; the
  prior payloads are preserved under
  `tuning/coleri2002/pre-seed-fix/canonical-output/`.
- Recommended disposition: accepted; the dashboard data reflects the
  regenerated canonical files.

## Method Audit

### Method: `LSCombEstimator (second-order)`

- Paper method faithfully implemented? Yes, equation (19) is implemented with offset-aware constant edge extension at the first and last pilot intervals.
- Surrogate-rule compliance: compliant.
- Interface adaptations documented? Yes: frequency boundary handling and linear OFDM-time interpolation.
- Verification status: smoke passed; exact equivalence to the previous linear-time branch passed; full default run NMSE at 5 dB / 15 dB was `0.01789558` / `0.00188881`.
- Remaining risk: offset-aware constant edge extension is plausible but not explicitly stated by the paper.

### Method: `LSCombEstimator (low-pass)`

- Paper method faithfully implemented? Yes as a documented MATLAB-style zero-insertion plus low-pass FIR interpolation method.
- Surrogate-rule compliance: compliant.
- Interface adaptations documented? Yes: finite filtering at the grid edges and linear OFDM-time interpolation.
- Verification status: smoke passed; exact equivalence to the previous linear-time branch passed; tuned full default run NMSE at 5 dB / 15 dB was `0.01324415` / `0.00363038`.
- Remaining risk: exact MATLAB `interp` coefficients were not validated, and the canonical benchmark uses tuned FIR hyperparameters rather than MATLAB defaults.

### Method: `LSCombEstimator (cubic spline)`

- Paper method faithfully implemented? Yes as MATLAB-like not-a-knot cubic spline interpolation with clipped constant edge queries.
- Surrogate-rule compliance: compliant.
- Interface adaptations documented? Yes: clipped edge behavior and linear OFDM-time interpolation.
- Verification status: smoke passed; exact equivalence to the previous linear-time branch passed; full default run NMSE at 5 dB / 15 dB was `0.01917050` / `0.00195503`.
- Remaining risk: clipped constant-edge behavior differs from default MATLAB extrapolation outside the provided coordinate range, but avoids an unstated periodic wrap-around.

### Method: `LSCombEstimator (time-domain)`

- Paper method faithfully implemented? Yes for equations (20)-(22), with the stated zero-padding correction.
- Surrogate-rule compliance: compliant.
- Interface adaptations documented? Yes: FFT scaling, stream comb-offset handling, and linear OFDM-time interpolation.
- Verification status: smoke passed; exact equivalence to the previous linear-time branch passed; full default run NMSE at 5 dB / 15 dB was `0.02390011` / `0.00429512`.
- Remaining risk: FFT/zero-padding convention may not exactly match the original simulation.

## Divergence Ledger

### Divergence: Frequency Boundary Handling

- Source: `interface`
- Reviewer verdict: `justified`
- Rationale: second-order interpolation uses offset-aware constant edge extension at the first and last pilot intervals; low-pass uses finite filtering with virtual constant edge pilots; spline uses offset-aware not-a-knot interpolation with clipped constant edge queries. These choices avoid unstated circular/periodic coupling of the first and last comb pilots.
- Impact on faithfulness: low to moderate.
- Impact on benchmarking fairness: acceptable.

### Divergence: Linear OFDM-Time Interpolation

- Source: `interface`
- Reviewer verdict: `justified`
- Rationale: the paper assumes comb pilots in each OFDM symbol, while the benchmark has sparse pilot-bearing symbols. Linear interpolation between pilot-bearing OFDM symbols is deterministic, simple, and avoids exposing a non-paper time-averaging method knob.
- Impact on faithfulness: moderate.
- Impact on benchmarking fairness: acceptable if disclosed.

### Divergence: MATLAB `interp` Approximation

- Source: `implementation`
- Reviewer verdict: `justified`
- Rationale: public MathWorks documentation is sufficient for a first-pass equivalent, though exact closed implementation validation remains open.
- Impact on faithfulness: low to moderate.
- Impact on benchmarking fairness: acceptable.

### Divergence: Tuned Low-Pass FIR Parameters

- Source: `implementation`
- Reviewer verdict: `justified`
- Rationale: `lowpass_filter_half_length` and `lowpass_cutoff` are low-pass FIR design parameters exposed by the estimator. Tuning them improved weighted benchmark NMSE without replacing the paper's zero-insertion plus FIR interpolation algorithm.
- Impact on faithfulness: moderate because MATLAB defaults are no longer the canonical benchmark setting.
- Impact on benchmarking fairness: acceptable if reported together with the tuning objective and selected parameters.

### Divergence: Detector Error Variance

- Source: `interface`
- Reviewer verdict: `necessary`
- Rationale: the paper lacks a detector-facing error-variance model and the Sionna detector requires one. The implementation uses `err_var_scale` times Sionna's propagated LS pilot variance. The calibrated-source v2 calibration produced raw winner `0.4540521216492649`; on independent canonical-size paired validation, rounded `0.454` scored mean BLER `0.057953042328042326` versus `0.055803571428571425` for incumbent `0.868`, so the incumbent is retained.
- Impact on faithfulness: low for channel NMSE, moderate for BER/BLER.
- Impact on benchmarking fairness: acceptable because the objective, statistical budget, and holdout selection are disclosed.

### Divergence: Printed Equation Typos

- Source: `paper`
- Reviewer verdict: `justified`
- Rationale: literal equation (19) would make the interpolation coordinate `l/N` instead of the interval coordinate `l/L`, and literal equation (21) has inconsistent zero-padding cases. The corrected forms match the surrounding definitions, equation (18), and the paper's stated DFT/IDFT zero-padding method.
- Impact on faithfulness: improves faithfulness to the intended algorithms.
- Impact on benchmarking fairness: acceptable because the corrections are disclosed and are not benchmark-tuned.

### Divergence: LMS And Block Methods Still Not Implemented

- Source: `interface`
- Reviewer verdict: `necessary`
- Rationale: LMS needs ordered temporal state and a step-size rule; block methods need all-subcarrier block pilots and decision-feedback receiver plumbing.
- Impact on faithfulness: limits paper coverage.
- Impact on benchmarking fairness: avoids misleading substitutions.

## Suitability Assessment

- Strengths of the paper for this task: comb LS estimators map cleanly to Sionna pilot extraction plus custom interpolation.
- Weaknesses of the paper for this task: LMS and block-pilot methods are outside the current independent-frame comb-pilot benchmark; BER/BLER interpretation remains sensitive to detector-facing variance propagation, which the paper does not specify.
- Required assets or benchmark changes for methods not implemented here: temporal benchmark/state for LMS, separate source-paper implementation for comb-MMSE, and block-pilot/decision-feedback receiver support for block methods.
