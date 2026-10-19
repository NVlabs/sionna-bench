# Implementation Plan

- Paper ID: `edfors1996`
- Citation: Ove Edfors, Magnus Sandell, Jan-Jaap van de Beek, Sarah Kate Wilson, and Per Ola Borjesson, "Analysis of DFT Based Channel Estimators for OFDM," 1996 report version of work presented in part at IEEE VTC 1995.
- Task: channel estimation
- Implementer: Codex
- Date: 2026-05-06

## Harness Summary

- Benchmark config: `configs/benchmark_config.py` fixes a 4-UE uplink OFDM
  MU-MIMO benchmark with 14 OFDM symbols, 624 subcarriers, 15 kHz subcarrier
  spacing, cyclic prefix length 44, a 16-antenna single-polarized 4x4 BS array,
  single-antenna UEs, UMi channel at 2.6 GHz, disabled pathloss/shadowing, UE
  speed 3 km/h, 16-QAM rate 0.6, and Eb/N0 sweep from -10 dB to 15 dB.
- Smoke config: `configs/smoke_config.py` uses batch size 2, one Eb/N0 point
  at 5 dB, one BER/BLER Monte Carlo batch, one NMSE batch, and reduced
  covariance iterations. It is an integration check, not final performance
  evidence.
- Resource grid and pilots: `evaluation.system.build_resource_grid` creates a
  Sionna `ResourceGrid` with `pilot_pattern="kronecker"` and pilot OFDM symbols
  `[2, 11]`. In the default config each stream has comb pilots every 4
  subcarriers on those two symbols, i.e. 156 nonzero pilots per pilot-bearing
  symbol.
- Base class: `src/base.py::PilotDomainEstimator` delegates pilot extraction and
  interpolation to Sionna `BaseChannelEstimator`; paper estimators implement
  `estimate_at_pilot_locations(y_pilots, no)` and pass an explicit interpolation
  choice.
- Input contract: estimator calls receive
  `y : [B, num_rx, num_rx_ant, num_ofdm_symbols, fft_size]` complex and scalar
  or broadcastable noise variance `no`.
- Output contract: estimators return
  `h_hat : [B, num_rx, num_rx_ant, num_tx, num_streams_per_tx,
  num_ofdm_symbols, fft_size]` complex and same-shaped float `err_var`.
- Built-in baselines: `src/sionna_bench/mimo_ofdm_chest/papers/builtin/` exposes `ls-nn`, `ls-lin`, `lmmse`, and
  `perfect`. The LMMSE baseline uses empirical covariance assets from
  `src/sionna_bench/mimo_ofdm_chest/papers/builtin/asset.py` and Sionna `LMMSEInterpolator`.
- Metrics: `evaluation/channel_estimation.py` computes BER, BLER, and NMSE in
  the same Monte Carlo loop by comparing decoded bits and estimator output
  against the true OFDM channel.
- Output layout: built-ins are stored under `outputs/builtin/<config-slug>/`
  or `outputs/builtin/smoke/<config-slug>/`; paper results are stored under
  `outputs/edfors1996/<config-slug>/` or
  `outputs/edfors1996/smoke/<config-slug>/`.

## Paper Method Inventory

### Method: `edfors-dft-a`

- Paper location: Chapter 3, Fig. 3.2 and Table 3.1; Appendix A, "Estimator A".
- Paper description: form the LS frequency-domain observation, transform it to
  the cyclic time domain by an IDFT, keep the `L` coefficients with largest
  channel power, apply the full `L x L` LMMSE map among those coefficients, zero
  the excluded coefficients, and DFT back to frequency.
- Paper inputs: full known-tone LS vector `g_ls`, time-domain covariance of
  `f = IDFT(g)`, observation noise variance, and tap count `L`.
- Paper output: frequency-domain channel estimate for the same `M` tones.
- Target repository abstraction: `PilotDomainEstimator` operating on each
  compact Sionna nonzero pilot-frequency vector, followed by explicit
  `interpolation_type="lin"` for full-grid reconstruction.
- Status: `implement`.

#### Implementation Detail

```text
Construction:
1. Initialize `PilotDomainEstimator(resource_grid, interpolation_type="lin")`.
2. Build pilot groups from Sionna `_pilot_ind` and `pilot_pattern.pilots`:
   one group per `(tx, stream, pilot OFDM symbol)` containing only nonzero
   pilot-vector entries sorted by subcarrier.
3. Load or estimate one auxiliary time-domain covariance matrix `R_f`, pooled
   across all pilot groups:
   a. sample true UMi channels from the active benchmark channel model;
   b. gather each group's true channel values at its comb-pilot subcarriers;
   c. apply the same unitary IDFT used by the estimator;
   d. accumulate `E[f f^H]`, pooling pilot groups, frames, receiver index, and
      receive antennas;
   e. cache the matrix under `aux/<aux_slug>/edfors1996_dft_time_cov/...`.
4. Compute one channel-power ordering from `diag(R_f)` and keep the first `L`
   indices for every group; if `L` is omitted, use
   `min(cfg.cyclic_prefix_length, M_p)`, which is 44 of 156 pilot-domain
   coefficients in the active benchmark configuration.

Inference:
1. Use `LSChannelEstimator.estimate_at_pilot_locations` to form the LS pilot
   observations, matching Chapter 3's back-rotated observation.
2. For each group:
   a. gather active pilot LS values in sorted subcarrier order;
   b. apply `torch.fft.ifft(..., norm="ortho")` to obtain the cyclic
      time-domain LS vector, matching the IDFT in Fig. 3.2;
   c. obtain the scalar LS observation noise variance from Sionna's
      `no / |pilot|^2`;
   d. set `R_L` to the shared covariance submatrix on the selected tap indices;
   e. compute the Estimator A LMMSE matrix from Appendix A:
      `P_L = R_L (R_L + sigma2 I)^(-1)`;
   f. write `P_L f_LS,selected` into the selected coefficients and zeros
      elsewhere;
   g. apply `torch.fft.fft(..., norm="ortho")`;
   h. scatter the denoised pilot-frequency estimates back into Sionna's compact
      pilot layout.
3. Return the denoised pilot estimates and the original LS pilot error
   variance as a conservative detector-facing variance.
4. Let Sionna's explicit linear interpolator produce the full-grid output.
```

#### Interface Adaptations

- The paper's `M` known tones are replaced by the benchmark's `M_p` nonzero
  pilot tones per pilot-bearing symbol. This preserves the DFT-domain linear
  estimator on the available LS observation vector but is not the exact all-tone
  receiver assumed in the paper.
- The paper does not specify time interpolation across OFDM symbols; Sionna
  `interpolation_type="lin"` is used explicitly to satisfy the benchmark output
  shape.
- The paper derives covariance from its analytical channel model. The
  implementation estimates the needed covariance empirically from the active
  Sionna UMi configuration, pools all statistically equivalent pilot groups,
  and stores it under `aux/`.
- Detector-facing `err_var` is not provided by the paper. The implementation
  returns the LS pilot variance rather than tuning a confidence value.

#### Risks

- With `L < M_p`, the paper predicts an error floor from excluded channel
  energy; smoke underperformance at high SNR may be expected.
- Estimator A can become close to a pilot-grid LMMSE estimator when `L=M_p`, so
  it may be higher complexity than the paper's preferred method.

### Method: `edfors-dft-b`

- Paper location: Chapter 3, Table 3.1; Appendix A, "Estimator B"; Chapter 5
  conclusion.
- Paper description: same DFT-domain structure and selected coefficients as
  Estimator A, but ignore cross-correlations and use only scalar diagonal
  LMMSE weights.
- Paper inputs: LS vector, time-domain coefficient powers, noise variance, and
  tap count `L`.
- Paper output: frequency-domain channel estimate for the same tones.
- Target repository abstraction: same `PilotDomainEstimator` adaptation as
  `edfors-dft-a`.
- Status: `implement`.

#### Implementation Detail

```text
1. Perform the same LS pilot extraction, grouping, unitary IDFT, and selected
   tap ordering used for Estimator A.
2. For each selected tap `i`, take its empirical time-domain variance
   `sigma_i2 = R_f[i, i]`.
3. Apply the Appendix A diagonal weight
   `p_i = sigma_i2 / (sigma_i2 + sigma2_noise)`.
4. Zero excluded taps, DFT back to the pilot-frequency vector, scatter into
   Sionna's compact pilot layout, and use explicit Sionna linear interpolation.
```

#### Interface Adaptations

Same as Estimator A. The default runner uses the active cyclic-prefix length as
the tap count, while the all-pilot-domain setting highlighted by the paper for
removing the DFT truncation floor remains available with `--num-taps 156`.

#### Risks

- Ignoring cross-correlation is the paper's deliberate approximation, but the
  benchmark's comb-pilot grid can make the empirical time-domain covariance
  different from the full-tone covariance analyzed in the paper.

### Method: `edfors-dft-c`

- Paper location: Chapter 3, Table 3.1; Appendix A, "Estimator C".
- Paper description: keep the selected time-domain coefficients directly and
  zero all excluded coefficients. No LMMSE weighting is applied.
- Paper inputs: LS vector, time-domain coefficient powers, and tap count `L`.
- Paper output: frequency-domain channel estimate for the same tones.
- Target repository abstraction: same `PilotDomainEstimator` adaptation as
  `edfors-dft-a`.
- Status: `implement`.

#### Implementation Detail

```text
1. Perform the same LS pilot extraction, grouping, unitary IDFT, and selected
   tap ordering used for Estimators A and B.
2. Copy selected time-domain LS coefficients unchanged.
3. Set all excluded coefficients to zero.
4. DFT back, scatter into the pilot layout, return LS pilot error variance, and
   use explicit Sionna linear interpolation.
```

#### Interface Adaptations

Same as Estimator A. With an explicit all-tap override `L=M_p`, Estimator C
reduces to the LS pilot estimate followed by the explicit Sionna interpolation.
This limiting case is useful as a sanity check.

#### Risks

- For non-sample-spaced channels and `L < M_p`, the paper explicitly predicts an
  error floor. This method may be weaker than LS at high SNR.

### Method: all-tone data-aided DFT estimators

- Paper location: Chapter 3 system and estimator derivation.
- Paper description: the literal paper receiver uses known transmitted symbols
  on every subcarrier of the OFDM symbol to form a full `M`-tone LS vector
  before the DFT-domain estimator.
- Target repository abstraction: none under the current first-pass channel
  estimation contract.
- Status: `incompatible`.

#### Incompatibility

The benchmark exposes pilot observations through Sionna's pilot extraction and
does not provide known data symbols or detector feedback to the channel
estimator. A literal all-tone implementation would need either an all-pilot
resource grid, a preamble/training-symbol benchmark mode, or a reviewed
decision-directed contract that passes data-symbol decisions back to the
estimator. None of these are present in the current task contract.

## Auxiliary Assets

- Asset name: `edfors1996_dft_time_cov`
- Proposed path:
  `aux/<aux_slug>/edfors1996_dft_time_cov/it<cov_num_iters>_bs<cov_batch_size>_seed<seed>/`
- Files: `time_covariances.npz` and `manifest.json`
- Stored array format: `time_covariances` is a single 2D complex matrix with
  shape `(M_p, M_p)` and `tap_powers` is a 1D float vector with shape `(M_p,)`.
- Generation procedure: sample benchmark UMi channels, gather true channels at
  each pilot group, apply the estimator's unitary IDFT, and estimate one
  complex covariance matrix pooled across all pilot groups.
- Validation checks: finite Hermitian matrices, nonnegative diagonal powers,
  a single covariance matrix has the active pilot-vector length, group count
  and group lengths match the active pilot layout, and metadata matches the
  active config/covariance budget.

## Cross-Cutting Decisions

- New code will stay under `src/sionna_bench/mimo_ofdm_chest/papers/edfors1996/`, with implementation in
  `estimator.py`, covariance/cache helper code in a paper-local helper module,
  and the CLI in `run_benchmark.py`.
- Method names: `edfors-dft-a`, `edfors-dft-b`, and `edfors-dft-c`, with a
  `-l<N>` suffix when the tap count is smaller than the full pilot-domain
  length. The default benchmark names are therefore `edfors-dft-a-l44`,
  `edfors-dft-b-l44`, and `edfors-dft-c-l44`.
- Default tap count: `min(cfg.cyclic_prefix_length, M_p)`, which is 44 for the
  active benchmark's 156-entry pilot-domain vector. The 44 retained taps are
  chosen by empirical IDFT-domain power ordering, not by taking the first 44
  delay indices. The all-tap no-error-floor setting remains available with
  `--num-taps 156`.
- Default covariance budget: use `cfg.cov_num_iters` and `cfg.cov_batch_size`;
  smoke runs inherit the smaller smoke values.
- Planned smoke command:
  `python -m sionna_bench.mimo_ofdm_chest.papers.edfors1996.run_benchmark --smoke`
- Required checks: `py_compile`, finite outputs, nonnegative `err_var`, smoke
  comparison against `ls-nn`, `ls-lin`, and `lmmse`, and inspection of saved
  `.npz` metrics.

## Questions For Reviewer

- Is the pilot-domain application of the paper's DFT-domain A/B/C transforms an
  acceptable first-pass implementation, with the literal all-tone receiver
  marked incompatible under the current benchmark contract?
- Is explicit Sionna `"lin"` interpolation acceptable for the off-pilot
  reconstruction, given that the paper does not specify time/frequency
  interpolation from sparse pilots?
