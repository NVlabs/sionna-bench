# Plan Review

- Paper ID: `edfors1996`
- Reviewer: Codex
- Date: 2026-05-06
- Reviewed sources: `src/sionna_bench/mimo_ofdm_chest/papers/edfors1996/paper.pdf`, `reports/paper_summary.md`,
  `reports/plan.md`, `tasks/channel_estimation.md`,
  `prompts/reviewer.md`, and `prompts/task_reviewer.md`.

## Harness Understanding

The plan correctly identifies the current channel-estimation contract:
first-pass estimators must use Sionna pilot extraction through
`PilotDomainEstimator` or an equivalent LS wrapper, must return full-grid
`h_hat` and `err_var`, and must choose interpolation explicitly. The plan also
records the Kronecker comb-pilot layout, built-in baselines, metrics, and output
directories.

## Method Verdicts

### `edfors-dft-a`

Verdict: accepted for implementation with caveats.

The planned implementation preserves the paper's Estimator A core operation:
LS observation, unitary IDFT, selection of largest-power time coefficients,
full LMMSE weighting on the selected coefficient covariance, zeroing of excluded
coefficients, and DFT back. Applying it to the benchmark pilot vector instead of
the paper's all-tone vector is a significant but necessary interface adaptation
because the current task does not expose known data tones.

Required conditions:

- The implementation must use the same DFT/IDFT normalization for covariance
  estimation and inference.
- The selected taps must be based on empirical time-domain channel power, not
  hard-coded low-delay indices.
- The empirical covariance should be shared across pilot groups unless a
  concrete benchmark asymmetry is identified; the channel second-order
  statistics should not depend on transmit index, stream index, or pilot OFDM
  symbol under the active setup.
- The off-pilot interpolation choice must be explicit.
- The implementation report must state that this is a pilot-domain adaptation,
  not the literal all-tone receiver.

### `edfors-dft-b`

Verdict: accepted for implementation with caveats.

The plan matches Appendix A's diagonal approximation: use selected coefficient
powers and scalar weights `sigma_i2 / (sigma_i2 + noise)`. A cyclic-prefix
default tap count is acceptable for the benchmark numerology, while the
all-coefficient setting remains useful as an explicit no-truncation check
because the paper notes that Estimator B can remove the error floor when all
coefficients are retained.

Required conditions:

- Noise variance must come from Sionna's LS pilot observation variance after
  known-pilot division.
- The method must not silently use full covariance terms.

### `edfors-dft-c`

Verdict: accepted for implementation with caveats.

The plan matches the paper's final restriction: selected time-domain LS
coefficients are passed through directly and excluded coefficients are zeroed.
The limiting case `L=M_p` should reduce to LS on the pilot vector, which is a
useful sanity check.

### all-tone data-aided DFT estimators

Verdict: accepted as incompatible under the current benchmark contract.

The paper's literal receiver assumes known transmitted symbols on all tones of
the OFDM symbol. The current benchmark supplies only pilots to the estimator and
does not pass data decisions or an all-pilot training symbol. A faithful all-tone
implementation would require a benchmark contract change.

## Divergence Ledger

| Divergence | Source | Reviewer verdict | Rationale | Impact on paper faithfulness | Impact on benchmarking fairness |
| --- | --- | --- | --- | --- | --- |
| Use Sionna nonzero pilot-frequency vectors instead of all known OFDM tones | interface | necessary | The task contract exposes pilot observations and requires the pilot/interpolation split. | Material: results are for a pilot-domain adaptation, not the exact paper receiver. | Fair if named and reported; should be compared against pilot-domain baselines. |
| Empirical pooled UMi covariance asset instead of analytical paper channel covariance | asset | justified | The benchmark channel differs from the paper's analytical PDP; the estimator still needs covariance from the active channel law. Pooling pilot groups avoids baking finite-sample group noise into what should be shared statistics. | Moderate: covariance source changes to match the benchmark scenario. | Fair if cached metadata records the sampling budget, config, and covariance scope. |
| Explicit Sionna `"lin"` interpolation for off-pilot reconstruction | interface | necessary | The paper does not define sparse-pilot full-grid interpolation; the benchmark requires full-grid output. | Moderate: off-pilot behavior is partly Sionna interpolation. | Fair if explicit and compared to LS interpolation baselines. |
| Return LS pilot error variance | interface | justified | The paper gives MSE analysis, not the detector-facing full-grid variance tensor needed by Sionna. Avoiding tuned confidence values is conservative. | Low for channel estimates; affects detector uncertainty. | Fair if disclosed and not tuned to BER/BLER. |
| Default `L=min(cyclic_prefix_length, M_p)` rather than paper figure values | implementation | justified | The active numerology has a 44-sample cyclic prefix and 156 pilot-domain coefficients; selecting 44 taps by empirical power matches the cyclic support assumption better than an all-tap default. The all-tap setting remains available with `--num-taps 156`. | Moderate: the default intentionally reintroduces truncation behavior the paper discusses. | Fair because output names include `-l44` and the tap-count knob is exposed. |

## Reviewer Decision

Implementation may proceed for `edfors-dft-a`, `edfors-dft-b`, and
`edfors-dft-c` under the conditions above. The literal all-tone data-aided
receiver remains incompatible unless the benchmark contract changes.
