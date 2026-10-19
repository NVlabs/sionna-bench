# Implementation Report

- Paper ID: `coleri2002`
- Implementer: Codex
- Date: 2026-06-03
- Revision note: seventh pass after Optuna tuning of the low-pass interpolation FIR parameters and rerunning the low-pass benchmark output.

## Implemented Methods

### Method: `LSCombEstimator (second-order)`

- Status after implementation: implemented, smoke-tested, and included in the
  completed Sionna 2.0.1 canonical benchmark rerun.
- Files changed:
  - `src/sionna_bench/mimo_ofdm_chest/papers/coleri2002/estimator.py`
  - `src/sionna_bench/mimo_ofdm_chest/papers/coleri2002/run_benchmark.py`
  - `src/sionna_bench/mimo_ofdm_chest/papers/coleri2002/reports/paper_summary.md`
  - `src/sionna_bench/mimo_ofdm_chest/papers/coleri2002/reports/plan.md`
  - `src/sionna_bench/mimo_ofdm_chest/papers/coleri2002/reports/plan_review.md`
  - `src/sionna_bench/mimo_ofdm_chest/papers/coleri2002/reports/implementation_report.md`
  - `src/sionna_bench/mimo_ofdm_chest/papers/coleri2002/reports/final_report.md`
- Paper equations/statements implemented:
  - Equation (17): LS pilot estimate.
  - Equation (19): second-order interpolation coefficients over adjacent pilots.
- Interface adaptations actually made:
  - Offset-aware constant edge extension for first/last pilot intervals.
  - Independent processing per Sionna stream and receive antenna.
  - Linear OFDM-time interpolation between the benchmark's pilot-bearing symbols.
  - Sionna LS pilot error variance `no/|pilot|^2`, propagated by the paper-local interpolator.
- Parameters used: shared calibrated detector-facing
  `err_var_scale=0.868`.
- Deviations from approved plan: revised from the earlier constant-variance proxy after BER/BLER diagnosis; later revised second-order edge handling from circular wrapping to offset-aware constant edge extension.

### Method: `LSCombEstimator (low-pass)`

- Status after implementation: implemented, smoke-tested, and included in the
  completed Sionna 2.0.1 canonical benchmark rerun.
- Paper equations/statements implemented:
  - Equation (17): LS pilot estimate.
  - Section V low-pass interpolation paragraph: zero insertion plus low-pass FIR interpolation.
  - Official MathWorks `interp`/`intfilt` documentation: default `n=4`, `cutoff=0.5`, symmetric low-pass FIR, original samples pass through unchanged.
- Interface adaptations actually made:
  - FIR designed from public MathWorks algorithm description using a least-squares low-pass design; not claimed bit-exact to closed MATLAB internals.
  - Finite FIR filtering at actual stream-offset grid coordinates with virtual constant edge pilots outside the observed pilot span.
  - Forced pilot pass-through after filtering.
  - Linear OFDM-time interpolation between pilot-bearing symbols.
  - Sionna LS pilot error variance `no/|pilot|^2`, propagated by the paper-local interpolator.
- Canonical benchmark parameters used after Sionna 2.0.1 retuning and
  validation: interpolation factor `L=4`,
  `lowpass_filter_half_length=16`, `lowpass_cutoff=0.105`, filter length
  `129`, passband gain `4`, and shared detector-facing
  `err_var_scale=0.868`.
- Deviations from approved plan: revised from circular convolution to finite constant-edge filtering after boundary review; canonical low-pass FIR length/cutoff were later selected by Optuna benchmark tuning rather than MATLAB defaults.

### Method: `LSCombEstimator (cubic spline)`

- Status after implementation: implemented, smoke-tested, and included in the
  completed Sionna 2.0.1 canonical benchmark rerun.
- Paper equations/statements implemented:
  - Equation (17): LS pilot estimate.
  - Section V spline paragraph: cubic spline interpolation.
  - Official MathWorks `spline` documentation: cubic spline interpolation via piecewise polynomials from a tridiagonal solve.
- Interface adaptations actually made:
  - Precomputed offset-specific cubic-spline linear interpolation matrices for fixed pilot coordinates.
  - Used MATLAB-like not-a-knot cubic splines inside the observed pilot span and clipped constant edge queries outside that span.
  - Linear OFDM-time interpolation between pilot-bearing symbols.
  - Sionna LS pilot error variance `no/|pilot|^2`, propagated by the paper-local interpolator.
- Parameters used: actual stream pilot offsets, not-a-knot spline boundary
  conditions, constant edge clipping outside the first/last pilot, and shared
  detector-facing `err_var_scale=0.868`.
- Deviations from approved plan: revised from periodic endpoint handling to offset-aware constant-edge clipping after boundary review.

### Method: `LSCombEstimator (time-domain)`

- Status after implementation: uses the same linear OFDM-time interpolation
  adaptation as the other Coleri methods; smoke-tested and included in the
  completed Sionna 2.0.1 canonical benchmark rerun.
- Paper equations/statements implemented:
  - Equation (17): LS pilot estimate.
  - Equations (20)-(22): IDFT, middle zero padding, and DFT reconstruction.
- Interface adaptations actually made:
  - Stream-offset handling for Sionna's multi-user comb pattern.
  - Linear OFDM-time interpolation between pilot-bearing symbols.
  - Sionna LS pilot error variance `no/|pilot|^2`, propagated by the paper-local interpolator.
- Parameters used: PyTorch FFT scaling selected so the frequency interpolator
  passes through pilot samples before OFDM-time interpolation, and shared
  detector-facing `err_var_scale=0.868`.
- Deviations from approved plan:
  - The implementation now uses fixed linear OFDM-time interpolation and no constructor option for time averaging.

## Not Implemented Methods

### Method: `comb-ls-linear`

- Final status: covered by built-in baseline.
- Reason: Existing `ls-lin` implements LS pilot estimates plus explicit linear interpolation equivalent to equations (17)-(18) under the benchmark pilot pattern.
- Required asset or benchmark-contract change: none.

### Method: `comb-lms-linear`

- Final status: `not implemented`
- Reason: missing LMS step size and no defined sequential update state in the
  benchmark's independent Monte-Carlo frames.
- Required asset or benchmark-contract change: sequential temporal-channel
  benchmark mode and reviewed LMS step-size rule.

### Method: `comb-mmse`

- Final status: `not implemented`
- Reason: Coleri et al. cite comb-MMSE variants but do not specify a self-contained method.
- Required asset or benchmark-contract change: implement the cited MMSE source
  paper separately or provide accepted covariance/statistical details.

### Method: `block-ls`, `block-mmse`, `block-decision-feedback`

- Final status: `not implemented`
- Reason: current benchmark fixes a Kronecker comb pilot grid and estimator-only receiver interface; the paper's block methods require all-subcarrier block pilots, a block schedule, and for decision feedback a detector/remapper loop.
- Required asset or benchmark-contract change: new block-pilot resource grid
  and sequential receiver contract.

## Deviation Log

### Deviation: Frequency Boundary Handling

- Stage: `planning`
- Source: `interface`
- Trigger: second-order, low-pass, and spline interpolation need values at or across the frequency-grid boundary.
- Decision: use offset-aware constant edge extension for second-order interpolation, finite low-pass filtering with virtual constant edge pilots, and offset-aware spline interpolation with clipped constant edge queries.
- Alternatives considered: not implementing the method, one-sided edge extrapolation, zero padding, circular wrapping, and periodic spline endpoints.
- Why the chosen decision is technically justified: equation (19) only needs adjacent pilot estimates, and offset-aware constant extension avoids coupling the first and last pilot estimates through an unstated boundary assumption, including for Sionna streams whose comb pilots do not start at subcarrier zero. The same concern applies to low-pass and spline: the paper names interpolation methods but does not state a circular frequency boundary, so the implemented low-pass/spline edge handling uses constant extension rather than wrapping the first and last comb pilots together.
- Faithfulness impact: low to moderate.
- Benchmarking impact: acceptable and explicit.
- Follow-up needed: none for first pass.

### Deviation: Linear OFDM-Time Interpolation

- Stage: `implementation`
- Source: `interface`
- Trigger: the paper assumes comb pilots in every OFDM symbol, while the benchmark has pilots on symbols 2 and 11 only.
- Decision: linearly interpolate frequency-reconstructed estimates between the pilot-bearing symbols across OFDM time.
- Alternatives considered: no OFDM-time interpolation and averaging across pilot-bearing OFDM symbols.
- Why the chosen decision is technically justified: linear interpolation is the simplest deterministic adaptation from the benchmark's sparse pilot-bearing OFDM symbols to the paper's per-symbol comb-pilot assumption, and it avoids exposing a benchmark-specific algorithm knob.
- Faithfulness impact: moderate.
- Benchmarking impact: affects full-grid channel estimates between pilot-bearing symbols and must be reported as an adaptation.
- Follow-up needed: rerun the full benchmark outputs after this refactor.

### Deviation: MATLAB `interp` Reimplementation

- Stage: `implementation`
- Source: `implementation`
- Trigger: the paper names MATLAB `interp`, not FIR coefficients.
- Decision: implement the documented zero-insertion plus symmetric low-pass FIR design using public MathWorks defaults and least-squares FIR design.
- Alternatives considered: leave the method unimplemented, require MATLAB runtime, or use arbitrary `firwin` coefficients.
- Why the chosen decision is technically justified: official MathWorks docs specify the algorithm family, defaults, and pass-through behavior well enough for a first-pass benchmark implementation.
- Faithfulness impact: low to moderate; not bit-exact to closed MATLAB internals.
- Benchmarking impact: acceptable.
- Follow-up needed: exact MATLAB reference vectors would further validate it.

### Deviation: Tuned Low-Pass FIR Hyperparameters

- Stage: `benchmarking`
- Source: `implementation`
- Trigger: the MATLAB-style defaults produced substantially worse NMSE than the other interpolation variants under the Sionna UMi benchmark.
- Decision: tune only `lowpass_filter_half_length` and `lowpass_cutoff` for `method="low-pass"` with Optuna, keeping the zero-insertion plus FIR interpolation algorithm unchanged.
- Alternatives considered: keep MATLAB defaults; tune the interpolation method categorically; add detector-facing variance scaling.
- Why the chosen decision is technically justified: these are exposed low-pass FIR design parameters, and tuning them does not replace the paper's low-pass interpolation algorithm. The v2 study used weighted NMSE at `[-10, -5, 0, 5, 10]` dB, 50 trials, and three cached batches of four channels per SNR. Its raw winner was `(15, 0.1309620268)`. Independent 63-batch-per-SNR paired validation rejected rounded `(15, 0.131)` (`0.0157995612`) in favor of current `(16, 0.105)` (`0.0145468554`).
- Faithfulness impact: moderate because the canonical benchmark no longer uses the documented MATLAB defaults.
- Benchmarking impact: improves low-pass NMSE substantially under the active scenario; BER/BLER improvements are not uniform at the highest SNRs.
- Follow-up needed: if exact MATLAB behavior is later validated, rerun this comparison against MATLAB coefficients.

### Deviation: Equation (19) Alpha Typo

- Stage: `planning`
- Source: `paper`
- Trigger: equation (19) prints `alpha = l/N`, but the interpolation interval is defined by `k=mL+l` with `0 <= l < L`.
- Decision: implement `alpha = l/L`.
- Alternatives considered: literal `l/N`.
- Why the chosen decision is technically justified: equation (18) already uses `l/L`, and second-order interpolation coefficients require a normalized coordinate that runs from `0` toward `1` across one pilot interval. With the printed `l/N`, `alpha` would remain near zero for all in-interval data subcarriers.
- Faithfulness impact: improves faithfulness to the intended algorithm.
- Benchmarking impact: prevents a degenerate near-nearest-pilot interpolation.
- Follow-up needed: none.

### Deviation: Equation (21) Zero-Padding Typos

- Stage: `planning`
- Source: `paper`
- Trigger: equation (21) has inconsistent case labels and index ranges for forming the zero-padded time-domain vector.
- Decision: implement the standard DFT-interpolation operation described by the surrounding prose and Fig. 3: split the `N_p` IDFT output at `N_p/2+1`, insert `N-N_p` middle zeros, and DFT back to `N` samples.
- Alternatives considered: literal transcription of equation (21) or not implementing time-domain interpolation.
- Why the chosen decision is technically justified: the printed cases appear to omit `n` in one condition, exclude the last output index, and use an inconsistent wrapped index. The implemented operation is the standard multirate zero-padding method and preserves pilot samples.
- Faithfulness impact: improves faithfulness to the stated zero-padding/DFT method, but not to the flawed printed formula.
- Benchmarking impact: stable and sanity-checkable implementation.
- Follow-up needed: exact reference vectors from the authors' MATLAB code would be the strongest validation.

### Deviation: Detector Error Variance

- Stage: `implementation`
- Source: `interface`
- Trigger: paper does not provide error variance, but the benchmark detector requires it.
- Decision: use Sionna's LS pilot variance `no/|pilot|^2`, propagate it through the paper-local interpolator, and apply one shared calibrated `err_var_scale` to all four variants.
- Alternatives considered: zero error variance, constant `err_var = no`, and tuned scaling.
- Why the chosen decision is technically justified: it matches the built-in LS observation model and avoids making the paper estimator appear less reliable solely because the Kronecker pilots have magnitude `2`.
- Faithfulness impact: low for NMSE, moderate for BER/BLER.
- Benchmarking impact: detector-facing metrics are now comparable to `ls-lin` without a tuned variance proxy.
- Follow-up needed: none; detector-facing calibration was completed after the estimator refactor was frozen.

## Formula Reaudit (2026-07-08)

Sections IV-V and equations (16)-(22) were reread against the current
implementation. No behavioral correction was required.

- Comb offsets are handled in actual resource-grid coordinates. Subtracting a
  stream's first pilot subcarrier before computing its pilot interval, and
  rolling the transform-domain reconstruction forward by the same offset,
  both place reconstructed pilot samples at the original subcarriers.
- The second-order weights match equation (19):
  `alpha*(alpha-1)/2`, `1-alpha**2`, and
  `alpha*(alpha+1)/2` multiply the previous, current, and next pilots. The
  implementation deliberately uses `alpha=l/L`; the printed `l/N` conflicts
  with equation (18) and does not traverse a pilot interval.
- The low-pass path inserts zeros on the expanded subcarrier grid and designs
  a length `2*n*L+1` FIR with passband gain `L`. That gain compensates for
  zero insertion. The interpolation polyphase is explicitly normalized to
  pass pilots unchanged; constant virtual edge pilots provide finite-filter
  context without circular endpoint coupling.
- Cubic-spline construction uses the pilot-coordinate dimension as SciPy's
  interpolation axis. Its precomputed matrix has shape
  `[num_subcarriers, num_pilots_per_symbol]`, so right multiplication applies
  the same real not-a-knot spline independently to the real and imaginary
  parts of every channel vector.
- The time-domain method treats the pilot sequence as periodic frequency
  samples. Its IDFT is an FFT-ordered circular-delay vector; middle zero
  padding separates the nonnegative- and negative-delay bins before the
  full-size DFT. For even `N_p`, the implementation follows the paper's
  `M=N_p/2+1` convention and retains bin `N_p/2` in the leading segment. The
  printed ranges in equation (21) remain internally inconsistent, so exact
  agreement with unpublished author code cannot be claimed.

The analytical tests cover pilot pass-through for every method, interior
quadratic and cubic recovery, low-frequency preservation and normalization,
delay-limited transform recovery for every comb offset, bounded interpolation
noise gain, and linear OFDM-time interpolation. Explicit boundary tests verify
the second-order virtual-neighbor clamping, low-pass constant preservation
through both finite-grid edges, and spline clipping outside the observed pilot
span.

The same invariants were subsequently parameterized over the shared test
systems with 48, 72, and 96 subcarriers; 8, 10, and 12 OFDM symbols; one, two,
and three pilot-bearing symbols; and UE comb spacings 1, 3, and 4. This exposed
the degenerate spacing-one low-pass case, where no interpolation is needed.
That case now uses the identity FIR rather than asking SciPy to design a filter
with a zero-width stopband.

Complete `LSCombEstimator` calls for all four methods now run on all three
shared systems and validate output shape, dtype, device, finiteness, and
non-negative error variance. Focused rejection tests cover nonuniform comb
spacing, uniform combs that do not span the full grid, and unequal active
pilot-symbol counts per stream. The latter check counts groups per stream
directly rather than relying only on divisibility of the total group count.

## Compilation Audit (2026-07-08)

All four complete `LSCombEstimator` inference calls compile with
`torch.compile(..., backend="eager", fullgraph=True)` and match eager outputs.
`torch._dynamo.explain` reports one graph and zero graph breaks for
second-order, low-pass, cubic-spline, and time-domain interpolation on the
alternate shared system configuration. They are therefore classified Tier A.

The refactor replaced inference-time `.item()` extraction from fixed pilot
metadata with immutable Python index tuples, allowing Dynamo to specialize the
fixed group loops. Numerical interpolation quantities remain registered Torch
buffers. SciPy is used only in the cubic-spline constructor to generate the
fixed weight matrix and is not reachable from the compiled call.

## Detector-Facing Variance Calibration (2026-08-14)

After interpolation parameters were frozen, the v2 study calibrated the one
shared scale against the equal-method mean of codeword BLER at 10 dB. It used
10 TPE trials over `[1e-3, 1]`, 63 cached batches of size four, sampler seed
`456`, and probe seed `271828`. The raw winner was `0.4540521216`, with mean
BLER `0.0602678571`.

Independent paired validation used another 63 batches of four channels with
seed `161803`. Rounded `0.454` scored `0.0579530423`, while current `0.868`
scored `0.0558035714`; the current scale is retained. The scale affects only
`err_var`; direct tests confirm identical `h_hat` and exactly scaled variances
for every method.

The final canonical rerun took 224.30 seconds on an NVIDIA L40S. Every NMSE,
BER, and BLER value for all four methods is bit-identical to the comparison
worktree. The scenario-local `results.json` records each method's constructor
arguments alongside its result archive.

## Verification

- Current source checks after the `_PilotGroupMixin` and linear-time refactor:
  - `python -m compileall src/sionna_bench/mimo_ofdm_chest/papers/coleri2002/estimator.py src/sionna_bench/mimo_ofdm_chest/papers/coleri2002/run_benchmark.py`
  - `python -m pytest test/mimo_ofdm_chest/infrastructure/test_pilot_groups.py`
  - exact synthetic-input equivalence against the previous linear-time implementation branch for all four variants; max channel/error differences were `0.000e+00`.
  - `python -m sionna_bench.mimo_ofdm_chest.papers.coleri2002.run_benchmark --smoke --output-dir /tmp/sionna_bench_coleri_renamed_smoke`
  - `python -m sionna_bench.mimo_ofdm_chest.papers.coleri2002.run_benchmark`
- Runner status: `sionna_bench.mimo_ofdm_chest.papers.coleri2002.run_benchmark` calls
  `evaluate_channel_estimator` once per Coleri method and writes
  `LSCombEstimator (<method>).npz` files; it does not run built-in baselines or
  generate plots.
- Smoke status: passed after the refactor. One-batch smoke BER/BLER/NMSE are noisy and are not treated as canonical benchmark values.
- Full default benchmark status: passed after the shared RNG correction; all
  metric samples exactly match the preserved pre-fix Sionna 2.0.1 run.
- Full default benchmark output files:
  - `outputs/coleri2002/3gpp-umi/LSCombEstimator (second-order).npz`
  - `outputs/coleri2002/3gpp-umi/LSCombEstimator (low-pass).npz`
  - `outputs/coleri2002/3gpp-umi/LSCombEstimator (cubic spline).npz`
  - `outputs/coleri2002/3gpp-umi/LSCombEstimator (time-domain).npz`
- Full default benchmark NMSE at 5 dB / 15 dB:
  - `LSCombEstimator (second-order)`: `0.01789558` / `0.00188881`
  - `LSCombEstimator (low-pass)`: `0.01324415` / `0.00363038`
  - `LSCombEstimator (cubic spline)`: `0.01917050` / `0.00195503`
  - `LSCombEstimator (time-domain)`: `0.02390011` / `0.00429512`
- Full default benchmark BER / BLER at 5 dB and 15 dB:
  - `LSCombEstimator (second-order)`: `0.02619324 / 0.19113757` and `0.00086420 / 0.00793651`
  - `LSCombEstimator (low-pass)`: `0.02112300 / 0.15476190` and `0.00326685 / 0.02910053`
  - `LSCombEstimator (cubic spline)`: `0.02797696 / 0.19940476` and `0.00089076 / 0.00925926`
  - `LSCombEstimator (time-domain)`: `0.03224907 / 0.22883598` and `0.00128946 / 0.01256614`
- Shape/dtype check: covered by the direct equivalence probe and smoke runner.
- NaN/Inf check: covered by smoke and full benchmark execution.
- Numerical credibility:
  - The current implementation is exactly equivalent to the earlier linear-time branch on synthetic inputs.
  - Full benchmark outputs use `LSCombEstimator (<method>)` filenames and include bootstrap confidence intervals from the current evaluation pipeline.
- Red flags:
  - Smoke BER/BLER remain noisy because the smoke benchmark uses only one Monte-Carlo batch.
  - `LSCombEstimator (time-domain)` still has the highest high-SNR NMSE among the four Coleri variants.
  - The tuned low-pass benchmark improves NMSE versus the MATLAB-default low-pass output, but its BER/BLER at 12.5 dB and 15 dB are slightly worse than the preceding low-pass output.
- Unverified items:
  - exact MATLAB `interp` bit-for-bit agreement.
  - LMS, comb-MMSE, and block-pilot methods.
