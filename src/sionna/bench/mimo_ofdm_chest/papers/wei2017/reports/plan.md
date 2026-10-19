# Implementation Plan: Wei et al., 2017

This is the implementer plan for `papers/wei2017`. No estimator code should be
written until this plan has been reviewed in `reports/plan_review.md` and the
human has approved proceeding.

## Historical Post-Approval Human-Requested Update

After the reviewed pilot-domain plan was implemented, the human requested that
Wei et al., 2017 be adapted to the repository's full-grid channel-estimator
interface. The approved update is:

- `ExpectationPropagationChannelEstimator` inherits from
  `FullGridChannelEstimator` and overrides `call()` directly.
- The Wei EP/EM step is still applied to LS-normalized uniform pilot-comb
  observations after a comb IFFT.
- The estimator reconstructs all effective subcarriers on pilot-bearing OFDM
  symbols by DFT-ordered middle-zero-padding of the denoised comb-delay
  coefficients followed by a full-rate FFT. This replaces the initially tried
  causal/end-padded reconstruction, which produced a large off-pilot high-SNR
  floor for Sionna's continuous-delay UMi channel.
- OFDM-time reconstruction is delegated to `TimeDomainInterpolator` with
  `time_avg=True` for the canonical benchmark.
- The tuned canonical parameters are `num_iter=40`, `alpha_init=0.3`,
  `posterior_variance_scale=0.1`, and `num_taps=None`.

This intermediate full-grid design was implemented and evaluated, but was
later superseded by the final human-requested revision below. It is retained
here as implementation history rather than as the active design.

## Final Human-Requested Revision

After comparing the full-grid delay reconstruction with explicit interpolation,
the human approved the following final design:

- The public class is `SBLChannelEstimator`. The name reflects that the
  comb-specialized uplink likelihood admits exact scalar Gaussian posterior
  moments and an EM common-support update without EP message passing.
- The estimator inherits from Sionna's `BasePilotChannelEstimator`, estimates
  denoised values only at the observed comb pilots, and exposes Sionna's
  standard `interpolation_type` and `interpolator` interface. The canonical
  default is `lin_time_avg`.
- Causal/end-padded and centered/middle-padded full-frequency reconstructions
  were evaluated but rejected because both produced substantially worse
  high-SNR NMSE for Sionna's continuous-delay UMi channels. This benchmark
  adaptation is documented in the README and implementation report.
- Pilot groups are processed as a tensor dimension so the complete online
  estimator is Tier A compilable.
- The selected Sionna 2.0.1 parameters are `num_iter=27`,
  `alpha_init=0.00507`, `num_taps=None` (the complete 156-coefficient comb
  representation), and `posterior_variance_scale=0.755`.

The original reviewed plan below remains unchanged as historical context for
the first implementation pass. The final implementation choices and benchmark
numbers are documented in `reports/implementation_report.md` and independently
audited in `reports/final_report.md`.

## Harness Summary

The current task is channel estimation in a fixed Sionna OFDM MU-MIMO
benchmark. The relevant harness files are:

- `configs/umi_3gpp_config.py`: canonical 3GPP UMi uplink configuration.
- `configs/smoke_config.py`: one-SNR, one-iteration smoke configuration.
- `evaluation/system.py`: builds the `ResourceGrid`, channel source, mapper,
  detector, LDPC chain, and calls the injected estimator.
- `evaluation/channel_estimation.py`: runs the SNR sweep, computes BER, BLER,
  and NMSE, and writes one `.npz` result file.
- `evaluation/result_store.py`: stores paper-local outputs and manifests.
- `src/pilot_groups.py`: provides `PilotGroupMixin` for grouping Sionna compact
  pilot vectors by physical pilot coordinates.

The benchmark estimator input is

```text
y  : [B, num_rx, num_rx_ant, num_ofdm_symbols, fft_size] complex
no : scalar or broadcastable noise variance
```

and the estimator output must be

```text
h_hat   : [B, num_rx, num_rx_ant, num_tx, num_streams_per_tx,
           num_ofdm_symbols, fft_size] complex
err_var : same shape as h_hat, float
```

The active resource grid uses a Kronecker pilot pattern. In the default UMi
configuration, the physical scenario is uplink, with 4 single-stream UEs,
16 BS receive antennas, 14 OFDM symbols, 624 subcarriers, pilot OFDM symbols
2 and 11, and cyclic-prefix length 44. Built-in baseline outputs are produced
under `outputs/builtin/<config-slug>/`; paper outputs live under
`outputs/<paper_id>/<config-slug>/`. Smoke outputs use the dedicated
`outputs/smoke/` tree and check integration, shapes, finite metrics, and rough sanity,
not final performance.

Existing paper implementations use one paper-local estimator class in
`estimator.py`, a paper-local `run_benchmark.py` that calls
`evaluate_channel_estimator`, explicit Sionna interpolation, Sionna-style
constructor arguments, and README/report documentation of benchmark
adaptations. Pilot-structured methods reuse `PilotGroupMixin`.

## Paper-To-Benchmark Mapping

The paper is a downlink estimator. Its literal model has one user observing a
superposition of pilot contributions from `M` BS transmit antennas:

```text
y_k = sum_m sum_l D_{k,(m-1)L+l} h_{l,m} + w_k              (1)-(2),(5)
```

The current benchmark is uplink. For one UE stream and one pilot OFDM symbol,
Sionna exposes separated pilot observations for each BS receive antenna:

```text
y_{a,k} = p_k sum_l F_{k,l} h_{a,l} + w_{a,k}
```

where `a` indexes BS receive antennas. The proposed adaptation treats `a` as
the co-located antenna dimension that shares sparse delay support. The
likelihood becomes block diagonal across receive antennas, while the EM update
continues to share one `alpha_l` across antennas, matching the common-support
role of Equations (19)-(21).

For the current Kronecker comb pilot groups, the block-diagonal uplink
likelihood is further simplified by LS-normalizing the pilot observations and
applying an inverse transform across the uniformly spaced pilot subcarriers.
This is the planned implementation path. If a future configuration does not
satisfy the stated comb assumptions, the estimator should fail rather than
silently use an unreviewed generic linear-model solver.

This is an interface adaptation, not a literal reproduction of the downlink
pilot model. It must be reviewed before implementation.

## Method Inventory

### Comb-Specialized LC-EM-EP Uplink Common-Support Estimator

Status: `implement`

Paper locations:

- Equations (1)-(2): pilot observation model.
- Equations (3)-(5): conditional complex Gaussian prior and likelihood.
- Equations (13)-(16): posterior mean and variance updates for one tap.
- Equations (19)-(21): EM update of the shared sparsity variance.
- Equations (22)-(27): low-complexity approximations used by LC-EM-EP.
- Algorithm 1: initialization and iteration order.

Method description:

Implement the paper's common-support sparse delay-domain EP/EM idea on Sionna
pilot groups, specialized to the current Kronecker comb pilot layout. For
every fixed `(tx, stream, pilot OFDM symbol)` group, gather the raw pilot
observations over subcarriers, divide by the known pilot symbols, and apply a
`K`-point inverse transform over the uniform comb index. Under the assumptions
listed below, this produces direct noisy observations of delay-domain
coefficients modulo `K`, up to a deterministic unit-magnitude phase. The
Gaussian posterior and EM common-support update from Equations (13)-(16) and
(19)-(21) are then applied across all BS receive antennas. The estimated
delay-domain coefficients are transformed back to pilot-subcarrier channel
estimates and returned to Sionna's `BaseChannelEstimator`, which applies
explicit built-in interpolation.

This is a reviewed benchmark specialization of the paper's linear Gaussian
model, not a generic `D = diag(pilots) F` solver for arbitrary pilot sets.

Target repository abstraction:

- Public class: `ExpectationPropagationChannelEstimator`.
- File: `src/sionna_bench/mimo_ofdm_chest/papers/wei2017/estimator.py`.
- Base class: `sionna.phy.ofdm.BaseChannelEstimator`.
- Mixin: `src.pilot_groups.PilotGroupMixin`.
- Pilot grouping: `_init_pilot_groups(group_over=("subcarrier",))`.
- Interpolation: explicit Sionna built-in `"lin_time_avg"` by default.

Planned constructor:

```text
ExpectationPropagationChannelEstimator(
    resource_grid,
    *,
    num_taps=None,
    num_iter=10,
    alpha_init=1.0,
    posterior_variance_scale=0.1,
    interpolation_type="lin_time_avg",
    precision=None,
    device=None,
)
```

Constructor choices:

- `num_taps` is the paper's maximum delay length `L`. If `None`, use the full
  pilot-group length. This is a benchmark adaptation, not a paper-specified
  default; the paper uses `L=128` in its own simulation. The full comb default
  avoids an artificial high-SNR floor from truncating the comb-delay response
  of Sionna's continuous-delay channel model. The constructor argument remains
  available to impose a shorter delay model explicitly.
- `num_iter` is the predefined number of EM iterations. In the comb-specialized
  observation model, the EP linear-message machinery collapses to scalar
  Gaussian posterior updates, but the iteration still follows Algorithm 1's
  initialize/update structure. A bounded validation sweep over
  `{1, 2, 5, 10, 20, 40}` selected `num_iter=10` by mean validation NMSE.
  It must be a positive integer.
- `alpha_init` implements Algorithm 1's sparsity-variance initialization
  (`alpha_l^1 = 1` by default). The paper's `v_hat^1` initialization is not a
  separate constructor parameter in this comb-specialized path because the
  posterior variance is computed directly from `alpha` and the transformed
  noise variance at each iteration. It must be positive.
- `posterior_variance_scale` exposes a documented detector-facing confidence
  scale on top of the paper-derived posterior variances. A bounded validation
  sweep over `{0.03, 0.1, 0.3, 1, 3, 10, 30}` selected `0.1` by mean
  validation BLER. It must be non-negative.
- `interpolation_type` is explicit and must be one of Sionna's supported
  interpolation types.

Fixed tensors to register as buffers:

- pilot subcarrier coordinates per group, from
  `_pilot_group_varying_coord("subcarrier")`;
- gathered pilot symbols per group, from the Sionna pilot pattern;
- FFT-bin indices per group, obtained by applying the resource-grid
  effective-subcarrier mapping to the pilot-group subcarrier coordinates;
- the first FFT-bin index `k0`, comb spacing `Delta`, and pilot power for each
  group;
- optional phase factors `exp(-j 2 pi k0 l / fft_size)` for diagnostics. The
  first-pass estimator can work in the equivalent comb-delay basis because the
  phase has unit magnitude and the output is transformed back only to the same
  pilot comb.

Comb-transform convention and required assumptions:

Use the OFDM frequency-response convention
`H[k] = sum_l h[l] exp(-j 2 pi k l / N)` with unit-magnitude Fourier atoms.
The paper does not specify the DFT normalization. For a pilot group with
`K` subcarriers and FFT-bin indices

```text
k_q = k0 + q Delta, q = 0, ..., K-1,
```

the implementation requires `Delta * K = N`. Then

```text
H[k_q] = sum_l h[l] exp(-j 2 pi k0 l / N) exp(-j 2 pi q l / K).
```

After LS normalization,

```text
z_q = y_q / p_q = H[k_q] + w_q / p_q,
```

and `torch.fft.ifft(z, n=K)` gives direct observations of
`d_l = h[l] exp(-j 2 pi k0 l / N)` modulo `K`. If `num_taps <= K` and the
modeled channel has no energy outside the first `num_taps` taps, there is no
aliasing inside the modeled support. For constant pilot magnitude
`|p_q|^2 = c`, the delay-domain noise variance is `no / (K c)`.

These assumptions are part of the implementation contract. The implementation
must fail explicitly during construction if the mapped FFT-bin indices are not
a uniform comb with `Delta * K = fft_size`, if pilot magnitudes are not
constant within a group, or if `num_taps > K`. It must not silently fall back
to a different estimator.

Detailed pseudocode:

```text
estimate_at_pilot_locations(y_pilots, no):
    # y_pilots is raw received pilot data from BaseChannelEstimator with shape
    # [B, num_rx, num_rx_ant, num_tx, num_streams, num_pilots].
    grouped_y = gather pilot groups from y_pilots
    grouped_h_hat = zeros_like(grouped_y)
    grouped_err_var = zeros_like(real(grouped_y))
    noise_var = expand no to [B, num_rx, num_rx_ant]

    for each pilot group g:
        observations = grouped_y[..., g, :]
            # shape [B, num_rx, num_rx_ant, K]
        pilot_values = pilot_values_by_group[g]
            # shape [K], nonzero with constant magnitude
        pilot_power = pilot_power_by_group[g]
            # scalar c = |p_q|^2 for the group

        # LS-normalized comb observations:
        # z_q = y_q / p_q = H[k_q] + w_q / p_q.
        normalized_observations = observations / pilot_values

        # Delay observations in the comb basis:
        # r_l = d_l + eta_l, eta_l ~ CN(0, no / (K * pilot_power)).
        all_delay_observations = ifft(normalized_observations, n=K, dim=-1)
        delay_observations = all_delay_observations[..., :num_taps]
        delay_noise_var = noise_var / (K * pilot_power)

        delay_mean, delay_var = _estimate_delay_observations(
            delay_observations,
            delay_noise_var,
        )
            # delay_mean: [B, num_rx, num_rx_ant, L] in comb-delay basis
            # delay_var : [B, num_rx, num_rx_ant, L]

        padded_delay_mean = zero-pad delay_mean to length K
        grouped_h_hat[..., g, :] = fft(padded_delay_mean, n=K, dim=-1)

        # Diagonal posterior approximation. FFT atoms have unit magnitude.
        grouped_err_var[..., g, :] =
            posterior_variance_scale * sum_l delay_var
            # shape [B, num_rx, num_rx_ant, K], constant across q

    scatter grouped_h_hat and grouped_err_var back to compact pilot layout
    return h_hat_at_pilots, err_var_at_pilots
```

The group estimator follows the posterior and EM updates from Equations
(13)-(16) and (19)-(21) after the comb transform diagonalizes the observation
model:

```text
_estimate_delay_observations(delay_observations, delay_noise_var):
    # delay_observations: [B, num_rx, num_rx_ant, L]
    # delay_noise_var   : [B, num_rx, num_rx_ant]

    alpha = alpha_init * ones([B, num_rx, L])     # Algorithm 1 alpha_l^1 = 1

    for iteration in range(num_iter):
        # Eqs. (14)/(16) for the diagonal Gaussian observation r_l = d_l + eta_l.
        delay_var = 1 / (1 / alpha[..., None, :] + 1 / delay_noise_var[..., None])
        delay_mean = delay_var * delay_observations / delay_noise_var[..., None]

        # Eq. (21): common-support EM update. Average over BS receive antennas.
        alpha = mean_over_receive_antennas(abs(delay_mean)**2 + delay_var)

    return delay_mean, delay_var
```

Important formula adaptation:

In the paper's downlink model, Equations (23) and (26) are needed because each
scalar observation `y_k` contains a Fourier-domain superposition from all BS
transmit antennas. In the benchmark uplink comb-pilot model, each BS receive
antenna has its own uniformly sampled frequency response. LS normalization and
the `K`-point inverse transform convert that observation to independent noisy
delay-bin observations under the assumptions above. Therefore the
low-complexity EP message updates collapse to scalar Gaussian posterior
updates, and common support is imposed through the shared `alpha_l` update in
Equation (21).

Error-variance handling:

The paper provides posterior delay-domain variances through Equations
(14a)/(16a). The implementation will propagate these variances to pilot
frequency locations with the diagonal approximation
`var(H[k_q]) = sum_l var(d_l)` because the comb FFT atoms have unit magnitude.
This ignores posterior tap correlations. Sionna interpolation will then produce
full-grid detector-facing variances, which is a benchmark adaptation rather
than a quantity defined by the paper. `posterior_variance_scale` is the only
permitted confidence-tuning knob; the canonical value must be documented before
result generation.

Auxiliary assets:

No reusable `aux/` asset is required for this method. All parameters are
computed from the active resource grid, pilot pattern, and runtime noise
variance.

Testing plan:

- Add a lightweight transform-identity test: for a synthetic delay vector with
  `num_taps <= K`, evaluate the corresponding comb-subcarrier frequency
  response, apply the planned `ifft`, and check that the first `num_taps`
  entries match the expected comb-delay coefficients.
- Add a shape/dtype test for `_estimate_delay_observations` on a small
  synthetic tensor, checking finite complex estimates and non-negative real
  variances.
- Add a deterministic no-noise or high-SNR limiting-case test with a small
  sparse common-support delay channel, unit pilots, and uniform comb spacing.
  After enough EM iterations, the reconstructed pilot-domain response should
  match the generated pilot response below a loose threshold. If this proves
  numerically brittle, record why in `implementation_report.md` and keep the
  deterministic transform and shape tests.
- Run the smoke benchmark and compare against `ls-nn`/`ls-lin` sanity results.

Faithfulness risks:

- The method is adapted from downlink superposed pilots to uplink separated
  receive-antenna observations.
- The implemented route is a comb-pilot specialization of the paper's linear
  Gaussian EP/EM estimator. It is not a generic solver for arbitrary pilot
  subcarrier sets or arbitrary pilot magnitudes.
- The paper's full delay-domain reconstruction over all subcarriers is not
  used as the full-grid output in this first pass. Full-grid reconstruction is
  delegated to explicit Sionna interpolation.
- The DFT scaling convention is selected for Sionna compatibility because the
  paper does not specify a normalization.
- Using all comb-delay bins by default departs from the paper's shorter
  finite-delay simulation model, but avoids a delay-truncation error floor in
  the current Sionna benchmark. Iteration count and posterior variance scale
  may need benchmark-specific calibration; any calibration must be bounded and
  documented.

Benchmarking risks:

- The benchmark pilot density and channel model differ from the paper's
  simulation setup, so paper-level performance claims should not be expected.
- Sparse delay support may be less exact for the 3GPP UMi frequency-domain
  channel than in the paper's ITU Vehicular-A simulation.
- The downstream BER/BLER depends on detector-facing variances whose semantics
  are only approximately connected to the paper's posterior variances.

### Generic Non-Comb Observation-Matrix LC-EM-EP

Status: `not implemented`

Paper locations: Equations (1)-(2), (22)-(27), and Algorithm 1.

The paper's low-complexity EP updates can be written for a general observation
matrix `D = diag(pilots) F_Omega`. The first-pass implementation above uses
the current benchmark's uniform comb pilots to diagonalize this model before
running the posterior and EM updates. It will explicitly fail if those comb
assumptions are not met.

Reason for not implementing:

The generic matrix-`D` solver is not needed for the current Sionna benchmark
configuration and is more configuration-dependent. A future pass could add it
as a reviewed fallback for non-comb pilot patterns, but the current
implementation should stay simple and fail fast outside its approved
assumptions.

### Full-Grid Delay-Domain Reconstruction

Status: `not implemented`

Paper locations: Equations (1)-(2) and the delay-domain channel model in
Section II.

Once delay-domain coefficients are estimated, the paper's model can evaluate
the channel frequency response on any OFDM subcarrier. The first-pass
repository contract requires a pilot-domain estimator plus explicit
interpolation, so the planned estimator reconstructs only the pilot comb and
delegates full-grid estimates and variances to Sionna interpolation.

Reason for not implementing:

This is a task-contract adaptation. A full-grid delay-domain interpolator could
be added later as a custom `BaseChannelInterpolator` if the reviewer approves a
configuration-independent implementation.

### Literal Downlink Superposed-Pilot LC-EM-EP

Status: `not implemented`

Paper locations: Equations (1)-(2), (5), and Algorithm 1.

The literal paper method assumes one user observes a single pilot vector that
superposes all BS transmit antenna pilot sequences on the same pilot
subcarriers. The current benchmark provides uplink receive-grid observations
with Kronecker pilots from UEs to the BS. Implementing the literal downlink
estimator would require a different channel direction, a different pilot
design, and a different receiver-side benchmark contract.

Reason for not implementing:

The literal method violates the current benchmark's uplink observation model.
The implemented method above is an approved-if-reviewed uplink adaptation of
the common-support EP/EM algorithm, not the literal downlink pilot receiver.

### Block-ISD, BP, And Exact-LS Oracle Comparisons

Status: `not implemented`

Paper locations: Section IV and Figure 3.

The paper compares LC-EM-EP against block-ISD, BP, and an exact-LS oracle with
known support. These are external comparison methods, not the paper's proposed
estimator. The benchmark already has built-in Sionna baselines, and the exact
LS oracle assumes unavailable true support information.

Reason for not implementing:

These methods are not necessary to implement the paper's contribution under
the current paper-to-Sionna workflow. A future comparison study could add them
as separate baseline papers or explicitly reviewed benchmark baselines.

### Paper Simulation Reproduction

Status: `not implemented`

Paper locations: Section IV and Figures 2-3.

The paper's simulation uses `M=32`, `N=4096`, ITU Vehicular-A, maximum channel
length `L=128`, sparsity level 9, and pilot length `K=580`. The repository's
benchmark uses the shared Sionna 3GPP UMi uplink configuration and leaderboard
metrics.

Reason for not implementing:

The goal is a benchmarkable Sionna channel estimator under the repository
contract, not a standalone reproduction of the paper's downlink simulation.
The README and implementation report must avoid claiming paper-figure
reproduction.

## Proposed Files After Approval

After plan review and human approval, implementation should add:

- `src/sionna_bench/mimo_ofdm_chest/papers/wei2017/estimator.py`
- `src/sionna_bench/mimo_ofdm_chest/papers/wei2017/run_benchmark.py`
- `src/sionna_bench/mimo_ofdm_chest/papers/wei2017/README.md`
- focused tests under `tests/` if the test suite structure remains appropriate
- `reports/implementation_report.md` after implementation and smoke checks

No shared harness change is currently planned.
