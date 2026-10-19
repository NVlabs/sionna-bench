# Plan Review

- Paper ID: `koller2022`
- Reviewer: Codex
- Date: 2026-04-29
- Reviewed files: `src/sionna_bench/mimo_ofdm_chest/papers/koller2022/paper.pdf`, `src/sionna_bench/mimo_ofdm_chest/papers/koller2022/reports/paper_summary.md`, `src/sionna_bench/mimo_ofdm_chest/papers/koller2022/reports/plan.md`, `tasks/channel_estimation.md`

## Overall Assessment

The plan correctly identifies the paper's main estimator: an offline EM-fitted GMM prior followed by the closed-form conditional mean estimator in equations (13), (15), and (16). It also correctly identifies that the paper's full wideband estimator does not naturally match the repository's first-pass pilot-domain contract.

Implementation may proceed for `koller-gmm-pilot-cme` and the later requested `koller-spatial-gmm-pilot-cme` variant only. Both are accepted as benchmark-compatible pilot-domain implementations of the paper's estimator, not as the full wideband GMM estimator from Section VI-C.

## Method Verdicts

### `koller-gmm-pilot-cme`

- Classification: `implement`
- Verdict: accepted with caveats
- Rationale: Applying equation (16) to LS-divided pilot observations is a valid instance of equation (1) with `A = I`. The method preserves the paper's GMM objective, EM-trained mixture prior, component likelihoods, responsibilities, LMMSE component estimates, and mixture conditional mean. The pilot-domain restriction and Sionna interpolation are explicit interface adaptations required by the task contract.
- Required implementation constraints:
  - Use `PilotDomainEstimator`; do not bypass Sionna pilot extraction.
  - Pass `interpolation_type="lin_time_avg"` explicitly.
  - Document EM initialization, covariance loading, training sample count, and component count in the implementation report.
  - Do not claim this is the full wideband estimator.

### `koller-spatial-gmm-pilot-cme`

- Classification: `implement`
- Verdict: accepted with caveats
- Rationale: The variant applies the same EM-fitted GMM prior and conditional mean equations (13), (15), and (16) to a receive-array pilot vector with `A = I`. This preserves the paper's estimator while changing the modeled vector from a per-antenna frequency vector to a per-subcarrier 16-antenna spatial vector.
- Required implementation constraints:
  - Use `PilotDomainEstimator`; do not bypass Sionna pilot extraction.
  - Pass `interpolation_type="lin_time_avg"` explicitly.
  - Store assets under a separate `koller2022_spatial_pilot_gmm` path so they cannot be confused with frequency-vector assets.
  - Document the larger number of per-resource-element GMMs and do not claim this is the full spatial-frequency wideband estimator.

### `koller-gmm-wideband-full`

- Classification: `defer`
- Verdict: accepted
- Rationale: A faithful implementation would estimate the full resource-grid channel vector from a pilot-selection observation matrix and would require a high-dimensional GMM asset or a reviewed structural simplification. This is too configuration-dependent for the first pass and cannot be implemented by directly subclassing `AbstractEstimator` under the current contract.

### `koller-circulant-gmm`

- Classification: `defer`
- Verdict: accepted
- Rationale: The paper motivates this simplification through ULA spatial Toeplitz/circulant covariance. The benchmark geometry and pilot-domain vectorization do not provide the same assumption without additional modeling work.

### `koller-kronecker-gmm`

- Classification: `defer`
- Verdict: accepted
- Rationale: The paper's Kronecker construction targets MIMO transmit/receive covariance. The current benchmark has single-antenna UEs and would need a reviewed receive-geometry mapping plus constrained EM assets.

## Divergence Ledger

| Divergence | Source | Reviewer verdict | Rationale | Impact on paper faithfulness | Impact on benchmarking fairness |
|---|---|---|---|---|---|
| Pilot-domain `A = I` model instead of full wideband selection matrix | interface | justified | Required by the task's pilot-domain split and still an exact instance of equation (1). | Limits scope to pilot denoising. | Fair if named and compared as a pilot-domain method. |
| `lin_time_avg` off-pilot reconstruction | interface | necessary | The implemented paper method returns pilot estimates; the benchmark requires full-grid estimates. | Does not reproduce Section VI-C full reconstruction. | Fair if reported; interpolation can influence NMSE and BER. |
| Per receive-antenna scalar-link GMM rather than full multi-antenna GMM | implementation | justified | Keeps EM and online CME tractable while preserving equation (16) for each modeled vector. | Loses spatial coupling from a larger possible GMM. | May underperform stronger GMMs; not an unfair advantage. |
| Per-subcarrier spatial-vector GMM rather than full spatial-frequency GMM | implementation | justified | Applies the same paper equations to the requested 16-antenna pilot vector while keeping the pilot-domain contract. | Captures receive-array coupling only at pilot subcarriers. | Fair if named separately and not overclaimed. |
| Deterministic EM initialization and covariance diagonal loading | implementation | necessary | The paper names EM but omits local numerical details; loading prevents singular covariances. | Minor, standard EM implementation detail. | Fair if fixed before smoke results and documented. |
| GMM posterior variance used as `err_var` | interface | justified | The paper does not define `err_var`, but the posterior variance is derived from the same model rather than tuned. | Adds detector-facing information not reported in the paper. | Fair if not tuned for BER/BLER. |

## Required Follow-Up In Final Audit

- Verify saved GMM assets match the active config and pilot-group dimensions.
- Verify smoke outputs are finite and `err_var` is nonnegative.
- Compare smoke NMSE against built-in baselines and report any underperformance as a red flag.
- Confirm code does not tune training settings or error variance after seeing smoke results.

## Verdict

Implementation may proceed for `koller-gmm-pilot-cme` and `koller-spatial-gmm-pilot-cme`. All other paper variants remain deferred.
