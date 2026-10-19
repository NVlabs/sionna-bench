# Implementation Plan

- Paper ID: `coleri2002`
- Full citation: Sinem Coleri, Mustafa Ergen, Anuj Puri, and Ahmad Bahai, "Channel Estimation Techniques Based on Pilot Arrangement in OFDM Systems," IEEE Transactions on Broadcasting, vol. 48, no. 3, pp. 223-229, Sept. 2002. DOI: 10.1109/TBC.2002.804034.
- Task: Channel estimation
- Implementer: Codex
- Date: 2026-06-03
- Revision note: revised after review feedback to implement the comb interpolation variants that were previously deferred too conservatively.

## Harness Summary

- Benchmark config: `configs/benchmark_config.py`; default OFDM grid has 14 symbols, 624 subcarriers, Kronecker pilots on OFDM symbols `(2, 11)`, four single-stream UEs, and 16 BS antennas.
- Smoke config: `configs/smoke_config.py`; one SNR at 5 dB, `batch_size=2`, and one Monte-Carlo batch.
- Built-in baselines: `ls-nn`, `ls-lin`, `lmmse`, and `perfect` from `src/sionna_bench/mimo_ofdm_chest/papers/builtin/`; these are generated separately from the Coleri paper runner.
- Task base class or template: first-pass paper implementations must use Sionna pilot extraction and explicit interpolation through a Sionna `BaseChannelEstimator` plus a `BaseChannelInterpolator`.
- Benchmark input contract: estimator call receives `y : [B, num_rx, num_rx_ant, num_ofdm_symbols, fft_size]` and scalar or broadcastable noise variance `no`.
- Benchmark output contract: `(h_hat, err_var)` with shape `[B, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm_symbols, fft_size]`.
- Pilot pattern: Sionna `KroneckerPilotPattern`; each stream has 312 nonzero pilots split over OFDM symbols 2 and 11 with comb spacing 4 and stream-dependent subcarrier offsets.
- Metrics: BER, BLER, and NMSE from `evaluation/channel_estimation.py`.
- Output layout: built-ins under `outputs/builtin/<config_slug>/`; paper outputs under `outputs/coleri2002/<config_slug>/`; smoke runs use the dedicated `outputs/smoke/` tree.
- Smoke meaning: integration and numerical sanity check. NaN/Inf, shape failures, negative error variance, or NMSE worse than `ls-nn` are red flags unless explained.

## External Documentation Used

- MATLAB `interp`: MathWorks documents `interp(x,r)` as zero insertion followed by a symmetric low-pass FIR interpolating filter; defaults include `n=4`, `cutoff=0.5`, and filter length `2*n*r+1`. The same page says the algorithm is intended to pass original samples unchanged and minimize interpolation error. URL: https://www.mathworks.com/help/signal/ref/interp.html
- MATLAB `intfilt`: MathWorks documents `intfilt(l,p,alpha)` as the interpolation FIR design used by `interp`, with bandlimited interpolation and `firls`-based filter design. URL: https://www.mathworks.com/help/signal/ref/intfilt.html
- MATLAB `spline`: MathWorks documents `spline(x,y,xq)` as cubic spline interpolation implemented through piecewise polynomials from a tridiagonal solve; `x` must be strictly increasing. URL: https://www.mathworks.com/help/matlab/ref/spline.html

## Paper Method Inventory

### Method: `comb-ls-second-order`

- Paper location: Sections IV and V; equation (17) for LS pilot estimates; equation (19) for second-order interpolation.
- Paper equations / algorithms / statements:
  - Equation (17): `H_e(k) = Y_p(k) / X_p(k)` for pilot subcarriers.
  - Equation (19): estimate `H_e(mL+l)` from `H_p(m-1)`, `H_p(m)`, and `H_p(m+1)` using quadratic coefficients.
  - Equation (19) appears to contain a typo in `alpha = l/N`; the implemented coordinate is `alpha = l/L`, consistent with equation (18), the definition `k=mL+l`, and second-order interpolation over one pilot interval.
- Paper inputs: uniformly spaced pilot observations and pilot values.
- Paper outputs: channel estimates at all data subcarriers for the OFDM symbol.
- Required side information: pilot spacing and a boundary convention.
- Target repository abstraction: `LSCombEstimator` with custom `CombInterpolator`.
- Status: `implement`

#### Faithful Implementation Detail

```text
1. Use Sionna pilot extraction through `LSCombEstimator`.
2. Compute LS pilot estimates `h_p = y_p / x_p`, matching equation (17).
3. For each transmit stream and pilot-bearing OFDM symbol:
   a. Gather nonzero pilots in increasing comb order, equation (16).
   b. For each target subcarrier `k` in actual resource-grid coordinates:
      rel = k - pilot_offset.
      If rel < 0, clamp to the first pilot with alpha = 0.
      Otherwise use m = floor(rel/L), alpha = (rel mod L)/L.
   c. Use constant edge extension for the first and last pilot intervals:
      h_m1 = H_p(max(m-1, 0)),
      h_0  = H_p(m),
      h_p1 = H_p(min(m+1, N_p-1)).
   d. Apply equation (19):
      c1 = alpha(alpha-1)/2,
      c0 = -(alpha-1)(alpha+1),
      c_1 = alpha(alpha+1)/2,
      H(k)=c1*h_m1 + c0*h_0 + c_1*h_p1.
   e. Return actual-grid estimates directly. Unlike the DFT/IDFT time-domain
      method, second-order interpolation does not circularly roll across the
      resource-grid edge.
4. The paper assumes comb pilots in every OFDM symbol; the benchmark has only
   symbols 2 and 11. Linearly interpolate the reconstructed pilot-symbol
   estimates over OFDM time to fill all resource-grid symbols. This is an
   explicit benchmark adaptation analogous to Sionna's time interpolation.
5. Return full-grid `h_hat` and the Sionna LS pilot error variance
   `no/|pilot|^2` propagated by the paper-local interpolator.
```

#### Interface Adaptations

- Multi-user Sionna streams are handled independently.
- Boundary samples use offset-aware constant edge extension instead of
  circular wrapping for second-order interpolation. This follows the reviewer
  request to avoid coupling the first and last pilot estimates through the
  equation (19) neighbor terms.
- Linear OFDM-time interpolation is not in the paper, but is needed to map the benchmark's two pilot-bearing symbols to all OFDM symbols without treating one pilot symbol as authoritative.

#### Surrogate Check

- This keeps the paper's LS objective and equation (19) interpolation formula. The boundary and time handling are interface adaptations, not a replacement estimator.

### Method: `comb-ls-low-pass`

- Paper location: Section V low-pass interpolation paragraph; Section VI-B simulation discussion and conclusion.
- Paper equations / algorithms / statements: LS pilot estimates from equation (17), zero insertion, and MATLAB `interp` low-pass FIR interpolation.
- Paper inputs: LS pilot estimates from equation (17).
- Paper outputs: channel estimates at data subcarriers.
- Required side information: interpolation factor; MATLAB defaults for `interp`.
- Target repository abstraction: `LSCombEstimator` with custom `CombInterpolator`.
- Status: `implement`

#### Faithful Implementation Detail

```text
1. Compute LS pilot estimates at Sionna-extracted pilots, equation (17).
2. For each stream and pilot OFDM symbol, arrange the `N_p` estimates in
   stream-relative comb order.
3. Insert `L-1` zeros between pilot estimates, as described by MATLAB `interp`.
4. Apply a symmetric real FIR low-pass interpolation filter:
   a. Use the documented MATLAB defaults `n=4`, `cutoff=0.5`.
   b. Use length `2*n*L+1`.
   c. Design a bandlimited least-squares low-pass filter with passband gain `L`,
      matching the documented `interp`/`intfilt` design family.
   d. Force the Nyquist interpolation samples to pass original pilots unchanged,
      matching the paper's and MATLAB's stated pass-through behavior.
5. Place pilots at their actual stream-offset resource-grid coordinates and
   apply finite FIR filtering with virtual constant edge pilots outside the
   first and last observed pilot. This avoids wrapping the first and last comb
   pilots together through an unstated boundary assumption.
6. Force the original pilot positions to pass through unchanged after
   filtering.
7. Linearly interpolate reconstructed estimates between pilot OFDM symbols.
8. Return full-grid `h_hat` and the Sionna LS pilot error variance
   `no/|pilot|^2` propagated by the paper-local interpolator.
```

#### Interface Adaptations

- The implementation follows MathWorks' public algorithm description but is not a bit-exact copy of closed MATLAB internals.
- Finite filtering with virtual constant edge pilots is a benchmark boundary adaptation.
- Linear OFDM-time interpolation is a benchmark adaptation.

#### Surrogate Check

- This is still the paper's low-pass interpolation method: zero insertion plus low-pass FIR interpolation. The filter design follows official MATLAB documentation rather than an unrelated filter family.

### Method: `comb-ls-spline`

- Paper location: Section V spline interpolation paragraph.
- Paper equations / algorithms / statements: LS pilot estimates from equation (17), then MATLAB `spline` cubic interpolation.
- Paper inputs: LS pilot estimates and pilot subcarrier coordinates.
- Paper outputs: channel estimates at data subcarriers.
- Required side information: cubic spline semantics and boundary convention.
- Target repository abstraction: `LSCombEstimator` with custom `CombInterpolator`.
- Status: `implement`

#### Faithful Implementation Detail

```text
1. Compute LS pilot estimates at Sionna-extracted pilots, equation (17).
2. Arrange pilot estimates in stream-relative comb order.
3. Precompute the linear interpolation matrix from pilot values to all
   subcarriers using cubic spline interpolation:
   a. Use strictly increasing actual-grid sample points
      `offset, offset+L, ..., offset+(N_p-1)L`.
   b. Use MATLAB-like not-a-knot cubic spline interpolation over the observed
      pilot span.
   c. Clip target points outside the first/last pilot coordinate to the nearest
      edge coordinate, giving constant edge extension instead of extrapolation
      or wrap-around.
4. Apply the fixed spline matrix to complex pilot estimates.
5. Linearly interpolate between pilot OFDM symbols.
6. Return full-grid `h_hat` and the Sionna LS pilot error variance
   `no/|pilot|^2` propagated by the paper-local interpolator.
```

#### Interface Adaptations

- Clipped constant edge handling is a benchmark boundary adaptation. It avoids arbitrary spline extrapolation and avoids coupling the first and last comb pilots through a periodic endpoint that the paper does not state.
- Real and imaginary parts are handled through the same real linear spline weights, which is equivalent to applying the spline to each component.

#### Surrogate Check

- This implements cubic spline interpolation, the paper method. The boundary rule is documented rather than used as a reason to defer.

### Method: `comb-ls-time-domain`

- Paper location: Sections IV and V; equations (17), (20)-(22), and Fig. 3.
- Paper equations / algorithms / statements: LS pilot estimates, IDFT to pilot-rate time domain, zero padding, and DFT back to all frequencies.
- Paper inputs: uniformly spaced pilot observations, pilot values, `N_p`, and `N`.
- Paper outputs: channel estimate at all frequencies.
- Required side information: DFT scaling and zero-padding convention.
- Target repository abstraction: `LSCombEstimator` with custom `CombInterpolator`.
- Status: `implement`

#### Paper Equation Correction

Equation (21) is treated as a flawed printed version of the standard
DFT-interpolation operation described in the surrounding text and Fig. 3. The
implementation follows the prose: IDFT the `N_p` pilot-frequency estimates,
insert `N-N_p` zeros in the middle of the delay-domain vector, and DFT back to
`N` frequency samples. The result must pass through the original pilot samples.

#### Faithful Implementation Detail

```text
1. Compute LS pilot estimates, equation (17).
2. For each stream and pilot OFDM symbol:
   a. Gather pilot estimates in comb order.
   b. Compute a pilot-rate inverse DFT, equation (20).
   c. Zero-pad the high-delay middle to expand from `N_p` to `N`, equation (21).
   d. Compute a DFT to obtain full frequency estimates, equation (22).
   e. Shift by stream comb offset.
3. Linearly interpolate reconstructed estimates between pilot OFDM symbols.
4. Return full-grid `h_hat` and the Sionna LS pilot error variance
   `no/|pilot|^2` propagated by the paper-local interpolator.
```

#### Interface Adaptations

- PyTorch FFT scaling is chosen so reconstructed values pass through pilot samples before OFDM-time interpolation.
- Linear OFDM-time interpolation is the same benchmark adaptation used for the other paper-local methods.

#### Surrogate Check

- This preserves equations (20)-(22); no alternate transform-domain estimator is substituted.

### Method: `comb-ls-linear`

- Paper location: Sections IV and V; equations (17)-(18).
- Paper equations / algorithms / statements: LS at pilot frequencies followed by linear interpolation.
- Paper inputs: uniformly spaced pilot observations and pilot values.
- Paper outputs: channel estimates at data subcarriers.
- Required side information: pilot locations and pilot values.
- Target repository abstraction: existing built-in `LSEstimatorWrapper(interpolation_type="lin")`.
- Status: `implement`

#### Faithful Implementation Detail

This method is already implemented by the shared `ls-lin` baseline. The Coleri paper runner does not duplicate it; comparison against `ls-lin` is done from separately generated baseline results.

### Method: `comb-lms-linear`

- Paper location: Section VI-A.4 and Fig. 4.
- Paper equations / algorithms / statements: one-tap LMS adaptive filter at each pilot frequency; first value from LS and following values from previous estimate plus current channel output; linear interpolation follows.
- Paper inputs: ordered time sequence of pilot observations and transmitted pilot values.
- Paper outputs: tracked pilot-frequency channel estimates and linearly interpolated data-subcarrier estimates.
- Required side information: LMS step size, state-reset policy, and continuous frame ordering.
- Target repository abstraction: would require a stateful estimator and a sequential benchmark mode.
- Status: `defer`

#### If `defer`

- Exact blocker: the LMS step size is not specified, and the benchmark samples independent Monte-Carlo frames with newly sampled topology rather than a continuous ordered channel track.
- What must exist before implementation can resume: a benchmark mode that preserves ordered frames for a UE/channel realization and a reviewed LMS step-size rule.
- Proposed future task: add a temporal-channel benchmark contract and implement stateful per-pilot LMS tracking under that contract.

### Method: `block-ls`, `block-mmse`, `block-decision-feedback`

- Paper location: Section III, equations (11)-(15), and Section VI-A.3.
- Paper equations / algorithms / statements: all-subcarrier block pilots, optional MMSE, and optional decision-feedback updates.
- Paper inputs: all-subcarrier pilot OFDM symbols, a block schedule, and for feedback the demapper/mapper loop.
- Paper outputs: reused or decision-updated channel estimates for following data symbols.
- Required side information: block timing, all-subcarrier pilots, covariance for MMSE, and detector decisions for feedback.
- Target repository abstraction: none under the current fixed Kronecker pilot benchmark.
- Status: `incompatible`

#### If `incompatible`

- Violated benchmark assumption: the benchmark uses a fixed Kronecker comb pilot pattern and an estimator-only interface.
- Why implementation inside the current interface would be misleading: remapping block pilots to the existing comb grid would change the paper's pilot arrangement, and decision feedback would require data decisions unavailable to the estimator.
- Benchmark-contract change required: add a block-pilot resource grid and sequential receiver evaluation with optional decision feedback.

### Method: `comb-mmse`

- Paper location: Introduction and Section IV discussion of MMSE/simplified MMSE/low-rank MMSE for comb pilots.
- Paper equations / algorithms / statements: the paper states MMSE can estimate pilot frequencies but does not provide a complete comb-pilot MMSE algorithm; it points to references [4] and [6].
- Paper inputs: comb pilot observations, noise variance, and channel statistics.
- Paper outputs: pilot-frequency MMSE estimates before interpolation.
- Required side information: exact covariance/statistical model and low-rank truncation rule.
- Target repository abstraction: future Sionna channel estimator/interpolator pair.
- Status: `defer`

#### If `defer`

- Exact blocker: Coleri et al. do not specify a self-contained comb-pilot MMSE estimator; implementing one from Sionna covariance helpers would be an external-method surrogate.
- What must exist before implementation can resume: a decision to implement reference [4] or [6] as its own paper task, or a concrete covariance model and formula accepted as Coleri's intended comb-MMSE variant.
- Proposed future task: create a separate paper folder for Hsieh and Wei 1998 or Reimers/DVB MMSE if that estimator is desired.

## Cross-Cutting Decisions

- Shared assumptions: Sionna pilot extraction is authoritative; nonzero pilots in Sionna's Kronecker pattern are treated as the paper's uniformly spaced comb pilots.
- Shared helper modules: all paper-local implementations live in `src/sionna_bench/mimo_ofdm_chest/papers/coleri2002/estimator.py`; the paper runner lives in `src/sionna_bench/mimo_ofdm_chest/papers/coleri2002/run_benchmark.py`.
- Parameters taken from the paper/docs: LS equation (17); second-order equation (19); MATLAB `interp` defaults `n=4`, `cutoff=0.5`; cubic spline interpolation; DFT/IDFT equations (20)-(22).
- Parameters requiring reviewer approval: offset-aware constant/finite edge handling for second-order, low-pass, and spline interpolation, and linear OFDM-time interpolation between pilot symbols.
- Planned smoke command: `python -m sionna_bench.mimo_ofdm_chest.papers.coleri2002.run_benchmark --smoke`
- Planned benchmark command: `python -m sionna_bench.mimo_ofdm_chest.papers.coleri2002.run_benchmark`.
- Required sanity checks: finite outputs, nonnegative error variance, frequency-interpolator pilot pass-through before OFDM-time interpolation, separately comparing against `ls-nn`, `ls-lin`, and `lmmse`, and reporting any noisy or nonmonotonic benchmark curves.
