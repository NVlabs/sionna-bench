# Implementation Plan

- Paper ID: `koller2022`
- Full citation: Michael Koller, Benedikt Fesl, Nurettin Turan, and Wolfgang Utschick, "An Asymptotically MSE-Optimal Estimator Based on Gaussian Mixture Models," IEEE Transactions on Signal Processing, vol. 70, pp. 4109-4123, 2022. DOI: 10.1109/TSP.2022.3194348.
- Task: Channel estimation
- Implementer: Codex
- Date: 2026-04-29

## Harness Summary

- Benchmark config: `configs/benchmark_config.py`; default OFDM grid has 14 OFDM symbols, 624 subcarriers, Kronecker pilots on OFDM symbols `(2, 11)`, four single-stream UEs, and 16 BS receive antennas.
- Smoke config: `configs/smoke_config.py`; one SNR at 5 dB, `batch_size=2`, one BER/BLER Monte-Carlo iteration, one NMSE batch, and reduced covariance iterations for the LMMSE baseline.
- Built-in baselines: `ls-nn`, `ls-lin`, `lmmse`, and `perfect` from `src/sionna_bench/mimo_ofdm_chest/papers/builtin/`.
- Base class: first-pass paper estimators must use Sionna pilot extraction through `PilotDomainEstimator` or equivalent `LSChannelEstimator` plus explicit interpolation.
- Estimator input contract: `y : [B, num_rx, num_rx_ant, num_ofdm_symbols, fft_size]` and scalar or broadcastable `no`.
- Estimator output contract: `h_hat` and `err_var` with shape `[B, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm_symbols, fft_size]`.
- Pilot pattern: Sionna `KroneckerPilotPattern`; each stream has 1248 gathered pilot-vector entries, of which 312 are nonzero pilots across OFDM symbols 2 and 11.
- Metrics: BER, BLER, and NMSE from `evaluation/channel_estimation.py`.
- Output layout: built-ins under `outputs/builtin/<config_slug>/`; paper outputs under `outputs/koller2022/<config_slug>/`; smoke runs use the dedicated `outputs/smoke/` tree.
- Smoke meaning: integration and numerical sanity check. NaN/Inf, shape failures, negative error variance, output independent of pilots/noise, or NMSE worse than `ls-nn` are red flags unless explained.

## Paper Method Inventory

### Method: `koller-gmm-pilot-cme`

- Paper location: Section IV-A, equations (4), (11), (13), (15), (16), and Algorithm 1.
- Paper inputs: training channel samples `{h_m}`, number of components `K`, observation `y`, observation matrix `A`, and noise covariance `Sigma`.
- Paper outputs: conditional-mean estimate `h_hat`.
- Target repository abstraction: `PilotDomainEstimator` with explicit `interpolation_type="lin_time_avg"`.
- Status: `implement`

#### Faithful Implementation Detail

This first-pass method applies the paper's GMM conditional-mean estimator to the pilot-domain channel vector required by the repository. For each transmit stream and each pilot-bearing OFDM symbol, define `h` as the vector of scalar-link channel values over that symbol's nonzero comb-pilot subcarriers for one receive antenna. After Sionna divides by the known pilot symbols, the pilot-domain observation model is

```text
z = h + w,
```

which is equation (1) with `A = I`. The complex model is implemented by stacking real and imaginary parts, consistent with the paper's discussion before Theorem 1.

```text
Offline training:
1. Generate UMi channel samples from the benchmark scenario.
2. For each stream and pilot OFDM symbol:
   a. Gather the true channel values at the nonzero Sionna pilot positions.
   b. Treat every generated frame and receive antenna as one channel sample
      h_m over that symbol's comb-pilot subcarriers.
   c. Stack real and imaginary parts.
   d. Fit a K-component full-covariance real GMM by EM, matching Algorithm 1
      line 1 and equation (4).
   e. Store weights p(k), means mu_k, and covariances C_k under
      aux/<config_slug>/koller2022_pilot_gmm/.

Online estimation:
1. Use Sionna's pilot extraction through `PilotDomainEstimator`.
2. Compute LS pilot observations z = y_p / x_p, matching the equation (1)
   identity-observation case after known pilot division.
3. For each stream and pilot OFDM symbol:
   a. Gather the nonzero pilot entries in increasing subcarrier order.
   b. Stack real and imaginary parts into observation y.
   c. Let Sigma = (err_var / 2) I in the real-valued representation, where
      `err_var = no / |pilot|^2` is Sionna's LS pilot variance.
   d. For each GMM component, compute the component likelihood in
      equation (15) with `A = I`.
   e. Compute the component LMMSE estimate in equation (13) with `A = I`:
      h_tilde_k = C_k (C_k + Sigma)^-1 (y - mu_k) + mu_k.
   f. Return the mixture CME from equation (16):
      h_hat = sum_k p(k | y) h_tilde_k.
4. Place the denoised estimates back into Sionna's pilot-vector layout.
5. Return a detector-facing pilot error variance from the GMM posterior
   variance implied by the same responsibilities and Gaussian posterior
   covariances. The paper does not require this output; this is a benchmark
   interface adaptation.
6. Let Sionna's explicit `lin_time_avg` interpolator produce the full
   resource-grid channel estimate.
```

#### Interface Adaptations

- The full wideband selection-matrix estimator is not implemented directly because the repository requires the pilot-domain split.
- The GMM is applied per scalar receive-antenna link rather than to the full multi-antenna resource grid. This keeps the paper's estimator intact for the modeled vector while avoiding a first-pass high-dimensional full-grid GMM.
- `lin_time_avg` is used explicitly for off-pilot reconstruction because the paper method being implemented here only denoises the pilot-domain vector.
- EM uses deterministic initialization and diagonal loading for numerical stability. The paper names EM but does not specify these local implementation choices.
- Posterior variance is returned only to satisfy the Sionna detector interface.

#### Auxiliary Assets

- Asset path: `aux/<config_slug>/koller2022_pilot_gmm/`
- Files: `gmm_params.npz` and `manifest.json`
- Generation procedure: sample UMi channels with the benchmark topology generator, gather true pilot-domain channel vectors, fit the GMM by EM, and store the parameters plus training metadata.
- Validation checks: component weights sum to one, covariances are finite positive definite after diagonal loading, and the saved dimensions match the current pilot groups.

#### Risks

- With small smoke-time `K` and training sample counts, this is an executable first-pass estimator rather than a reproduction of the paper's high-sample numerical curves.
- Per-link pilot-domain training discards spatial coupling; implementing the full spatial/frequency GMM is deferred.
- `lin_time_avg` can dominate off-pilot NMSE behavior, so benchmark results must not be interpreted as the full paper wideband estimator.

### Method: `koller-spatial-gmm-pilot-cme`

- Paper location: Section IV-A, equations (4), (11), (13), (15), (16), and Algorithm 1.
- Paper inputs: training channel samples `{h_m}`, number of components `K`, observation `y`, observation matrix `A`, and noise covariance `Sigma`.
- Paper outputs: conditional-mean estimate `h_hat`.
- Target repository abstraction: `PilotDomainEstimator` with explicit `interpolation_type="lin_time_avg"`.
- Status: `implement`

#### Faithful Implementation Detail

This variant applies the same GMM conditional-mean estimator to a different pilot-domain vectorization. For each transmit stream, pilot OFDM symbol, and nonzero pilot subcarrier, define `h` as the 16-dimensional complex receive-array channel vector at that single pilot resource element. After Sionna's known-pilot division, the observation model is again

```text
z = h + w,
```

which is equation (1) with `A = I`. The GMM is fitted and evaluated in the real-valued representation obtained by stacking the real and imaginary parts of the receive-array vector.

```text
Offline training:
1. Generate UMi channel samples from the benchmark scenario.
2. For each stream, pilot OFDM symbol, and nonzero pilot subcarrier:
   a. Gather the true complex channel across all 16 BS receive antennas.
   b. Treat every generated frame and receiver index as one spatial channel
      sample h_m for that pilot resource element.
   c. Stack real and imaginary parts.
   d. Fit a K-component full-covariance real GMM by EM, matching Algorithm 1
      line 1 and equation (4).
   e. Store weights p(k), means mu_k, and covariances C_k under
      aux/<config_slug>/koller2022_spatial_pilot_gmm/.

Online estimation:
1. Use Sionna's pilot extraction through `PilotDomainEstimator`.
2. Compute LS pilot observations z = y_p / x_p.
3. For each stream, pilot OFDM symbol, and nonzero pilot subcarrier:
   a. Gather the 16 receive-antenna LS values for that pilot resource element.
   b. Stack real and imaginary parts into observation y.
   c. Let Sigma = (err_var / 2) I in the real-valued representation.
   d. For each GMM component, compute the likelihood in equation (15) with
      `A = I`.
   e. Compute the component LMMSE estimate in equation (13) with `A = I`.
   f. Return the mixture CME from equation (16).
4. Place the denoised 16-antenna vector back into Sionna's pilot-vector layout
   at the same pilot index.
5. Return the corresponding posterior variance as the detector-facing
   `err_var`, then let explicit `lin_time_avg` interpolation produce the full
   resource-grid channel estimate.
```

#### Interface Adaptations

- This is a pilot-domain spatial-vector application of the paper's estimator, not the full wideband Section VI-C estimator.
- The GMM couples the 16 receive antennas at one pilot subcarrier, but it intentionally does not couple different subcarriers.
- The method uses one small full-covariance GMM per nonzero pilot resource element. This keeps the modeled vector close to the user's requested variant and avoids a full spatial-frequency GMM.
- Posterior variance is returned only to satisfy the Sionna detector interface.

#### Auxiliary Assets

- Asset path: `aux/<config_slug>/koller2022_spatial_pilot_gmm/`
- Files: `gmm_params.npz` and `manifest.json`
- Generation procedure: sample UMi channels with the benchmark topology generator, gather true 16-antenna spatial channel vectors at each nonzero pilot resource element, fit each GMM by EM, and store parameters plus metadata.
- Validation checks: component weights sum to one, covariances are finite positive definite after diagonal loading, saved dimensions match the number of spatial pilot groups, and each group has real dimension `2 * cfg.num_rx_ant`.

#### Risks

- The number of GMM groups is much larger than the frequency-vector variant, so asset generation is more expensive.
- Per-subcarrier spatial training ignores frequency correlation and may be sample hungry at large `K`.
- `lin_time_avg` still controls off-pilot reconstruction, so this remains a pilot-domain estimator rather than the full paper wideband estimator.

### Method: `koller-gmm-wideband-full`

- Paper location: Sections II-C, IV-A, and VI-C.
- Paper inputs: full time-frequency channel samples, pilot-position selection matrix `A`, noise covariance `Sigma`, and fitted GMM parameters.
- Paper outputs: full time-frequency conditional-mean channel estimate.
- Target repository abstraction: future `BaseChannelInterpolator` or reviewed equivalent wrapper.
- Status: `defer`

#### If `defer`

- Exact blocker: a faithful full-grid GMM would model the full resource-grid channel vector and evaluate equation (16) with the pilot selection matrix `A`. Under the current benchmark dimensions, this requires a large trained GMM asset and a custom reconstruction path. A direct `AbstractEstimator` bypass is not allowed by the task contract.
- Proposed future asset path: `aux/<config_slug>/koller2022_full_grid_gmm/`
- Acquisition plan: generate full-grid channel samples from the benchmark UMi scenario, design a reviewed covariance constraint or dimensionality reduction that is justified by the paper, fit and validate the GMM, then implement a `BaseChannelInterpolator` that maps Sionna pilot estimates to the full grid using equation (16).

### Method: `koller-circulant-gmm`

- Paper location: Section IV-B1, equations (18) and (19).
- Paper inputs: ULA-like channel samples whose covariance is well approximated by circulant matrices.
- Paper outputs: lower-complexity GMM conditional-mean estimates.
- Target repository abstraction: future `PilotDomainEstimator` or custom interpolator depending on modeled vector.
- Status: `defer`

#### If `defer`

- Exact blocker: the paper's circulant simplification is motivated by ULA spatial covariance. The benchmark uses a 4x4 UPA and pilot-domain frequency vectors; applying the simplification without a reviewed model mapping would be a surrogate.
- Proposed future asset path: `aux/<config_slug>/koller2022_circulant_gmm/`
- Acquisition plan: define a paper-justified transform domain for the benchmark geometry, fit transformed diagonal-covariance GMMs, and compare against the full-covariance pilot-domain implementation.

### Method: `koller-kronecker-gmm`

- Paper location: Section IV-B2.
- Paper inputs: MIMO channel matrices and separable transmit/receive covariance structure.
- Paper outputs: GMM conditional-mean estimates with Kronecker covariance constraints.
- Target repository abstraction: future `PilotDomainEstimator` or custom interpolator.
- Status: `defer`

#### If `defer`

- Exact blocker: the benchmark has single-antenna UEs and a UPA receiver. The paper's transmit/receive Kronecker construction would need a reviewed mapping to this geometry and a separate constrained EM implementation.
- Proposed future asset path: `aux/<config_slug>/koller2022_kronecker_gmm/`
- Acquisition plan: define the receive-side row/column factorization, generate training samples, fit constrained GMM factors, and validate against the full pilot-domain GMM.

## Cross-Cutting Decisions

- New code stays in `src/sionna_bench/mimo_ofdm_chest/papers/koller2022/estimator.py` and `src/sionna_bench/mimo_ofdm_chest/papers/koller2022/run_benchmark.py`.
- Planned estimator name: `koller-gmm-pilot-cme`.
- Planned smoke command: `python -m sionna_bench.mimo_ofdm_chest.papers.koller2022.run_benchmark --smoke`
- Required sanity checks: finite `h_hat`, finite and nonnegative `err_var`, paper estimator output changes with received pilots and noise level, and smoke NMSE compared against `ls-nn`, `ls-lin`, and `lmmse`.

## Questions For Reviewer

- Is the `A = I` pilot-domain application of equation (16) acceptable as the first-pass implementation while deferring the full wideband selection-matrix reconstruction?
- Is `lin_time_avg` acceptable as the explicit benchmark interpolation choice for this pilot-domain GMM?
- Is returning the GMM posterior variance as `err_var` acceptable even though the paper only defines the conditional mean?
