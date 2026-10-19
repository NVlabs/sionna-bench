# Implementation Report: Wei et al., 2017

## Approved Plan

Implementation followed the revised `reports/plan.md` accepted in
`reports/plan_review.md`. The human approved the reviewed plan before coding.

The implemented method is a comb-specialized uplink SBL adaptation of Wei et
al.'s common-support conditional Gaussian prior and EM update. It does not
implement EP message passing, the literal downlink superposed-pilot receiver,
or a generic observation-matrix LC-EM-EP fallback.

## Implemented Files

- `src/sionna_bench/mimo_ofdm_chest/papers/wei2017/estimator.py`
- `src/sionna_bench/mimo_ofdm_chest/papers/wei2017/run_benchmark.py`
- `src/sionna_bench/mimo_ofdm_chest/papers/wei2017/tuning.py`
- `src/sionna_bench/mimo_ofdm_chest/papers/wei2017/README.md`
- `test/mimo_ofdm_chest/papers/wei2017/test_estimator.py`

The paper-local Python layout follows `docs/paper_architecture.md`:
`estimator.py`, `tuning.py`, and `run_benchmark.py` are the only implementation
modules. The runner and tuner expose the standard `run_benchmark()` and
`run_tuning()` handles, respectively.

## Implementation Notes

`SBLChannelEstimator` inherits from
Sionna's `BasePilotChannelEstimator` and `_PilotGroupMixin`. It initializes pilot groups
over subcarriers and validates the approved comb assumptions during
construction:

- pilot FFT-bin indices must be uniformly spaced;
- `spacing * group_len == fft_size`;
- pilot magnitudes must be constant and nonzero within each group;
- `num_taps <= group_len`.

After the first full benchmark exposed a delay-truncation error floor, the
human approved changing the default `num_taps=None` from the cyclic-prefix
length to the full pilot-comb length. This is recorded as a human-requested
plan change and avoids discarding comb-delay energy from Sionna's
continuous-delay channel model. A shorter `num_taps` can still be requested
explicitly.

For the earlier pilot-only implementation after the full-comb change, two
bounded validation sweeps were run with
`num_mc_batches=16`, `batch_size=4`, and validation SNRs `[0, 5, 10, 15]`:

- `num_iter` candidates `{1, 2, 5, 10, 20, 40}` with
  `posterior_variance_scale=1.0`. Mean NMSEs were
  `{1: 9.7670e-03, 2: 6.5675e-03, 5: 4.8633e-03, 10: 4.6813e-03,
  20: 5.1125e-03, 40: 4.8566e-03}`. The selected value for that earlier
  pilot-only version was `num_iter=10`.
- `posterior_variance_scale` candidates `{0.03, 0.1, 0.3, 1, 3, 10, 30}`
  with `num_iter=10`. Mean BLERs were
  `{0.03: 1.5723e-01, 0.1: 1.4258e-01, 0.3: 1.4648e-01,
  1: 1.6797e-01, 3: 1.8750e-01, 10: 2.5293e-01, 30: 4.4336e-01}`.
  The selected value, retained through the later reconstruction variants as
  the precalibration value, was `posterior_variance_scale=0.1`.

After moving the estimator to the full-grid channel-estimator interface, the
initial causal/end-padded full-subcarrier reconstruction caused a large
off-pilot high-SNR floor. The reconstruction was changed to DFT-ordered
middle-zero-padding of the denoised comb-delay coefficients followed by a
full-rate FFT. A bounded retuning sweep for this variant used
`batch_size=4`, `num_mc_batches=12`, validation SNRs `[0, 5, 10, 15]`,
`posterior_variance_scale=0.1`, and
`interpolation_type="lin_time_avg"`. The best candidates were
then validated with `num_mc_batches=24` on the full benchmark SNR grid. The
selected values are `num_iter=40` and `alpha_init=0.3`.

An equation-level reread subsequently found that the former loop updated
`alpha` at the end of every iteration but returned the posterior from before
the final update. Algorithm 1 and Equation (16) instead return posterior
moments evaluated with the updated `alpha`. The corrected implementation does
that explicitly. Historical `num_iter=40` therefore maps exactly to 39
effective EM updates; direct single-precision tests confirm bitwise-identical
posterior means and variances under this mapping. A subsequent full benchmark
compared corrected 39 and 40 updates. Corrected 40 produced identical BLER, a
detector objective of `0.218968668520004` versus `0.218971509153062`, slightly
better low-SNR NMSE, and slightly worse high-SNR NMSE. The human selected
corrected `num_iter=40` as the canonical setting.

The middle-zero-padding reconstruction was then compared directly with Sionna
linear frequency interpolation while keeping the SBL pilot estimates and
`interpolation_type="lin_time_avg"` fixed. A paired probe with 16 batches of
size 4 per SNR gave
linear/middle-padding NMSEs of `9.974182e-3/1.084779e-2` at 0 dB,
`5.192339e-3/6.028906e-3` at 5 dB,
`2.515874e-3/3.704476e-3` at 10 dB, and
`1.480056e-3/2.637060e-3` at 15 dB. The implementation was therefore returned
to the pilot-domain Sionna contract with explicit linear interpolation.

For each pilot group, the estimator divides raw Sionna pilot observations by
the known pilot symbols, applies a `group_len`-point IFFT over the comb index,
keeps the first `num_taps` delay coefficients, and applies scalar Gaussian
posterior updates with the common-support EM update across BS receive
antennas. The denoised coefficients are transformed back only to the observed
pilot subcarriers. Sionna then applies linear frequency interpolation and,
with the benchmark's `lin_time_avg` setting, averages the two resulting pilot
symbols over OFDM time.

All pilot groups are processed as one tensor with layout
`[batch, receiver, group, antenna, tap]`. The group dimension remains
independent while Equation (21) averages only over the antenna dimension.
This removes the former Python group loop and all tensor-to-Python extraction
from online inference without adding a special-case execution path.

The implementation uses `resource_grid.effective_subcarrier_ind` when
validating FFT-bin spacing.

### Paper Equation Recheck

The final implementation was reread against Section III-B through III-D and
Algorithm 1. After pilot division and the comb IFFT, each retained delay tap
has a direct observation `z = h + n` with variance
`nu = no/(group_len*pilot_power)`. The paper's Gaussian likelihood quantities
therefore reduce to `R=z` and `Sigma=nu`. Equations (14a)/(16a) become
`v=(1/alpha + 1/nu)^-1`, and Equations (14b)/(16b) become `h_hat=v*z/nu`,
exactly as evaluated by `_gaussian_posterior_moments()`.

Equation (21) averages `abs(h_hat)^2 + v` only over the co-located antenna
axis. Batch, receiver, pilot-group, and delay-tap axes remain independent.
The direct equation-level test covers two antennas and verifies the initial
posterior, Equation (21), and the final Equation (16) posterior. The generic
factor-node residual and cavity-message updates in Equations (23)-(27) are not
executed because the approved comb specialization diagonalizes the likelihood;
the implementation therefore remains an SBL special case rather than generic
LC-EM-EP.

Algorithm 1 explicitly permits stopping after a predefined iteration count.
The implementation uses exactly `num_iter` EM updates and then returns the
posterior evaluated with the resulting `alpha`; it does not implement an
adaptive convergence criterion. The `torch.finfo` noise floor is a numerical
safeguard for zero-noise tests, not a model change at benchmark noise levels.

Because the scalar delay posteriors are independent and each unnormalized FFT
atom has unit magnitude, the marginal pilot-subcarrier variance is the sum of
the retained delay variances. Frequency-domain estimation errors are correlated
across subcarriers, but Sionna's detector interface accepts only marginal
`err_var` values and interpolation does not propagate those cross-covariances.
The documented and independently validated `posterior_variance_scale` handles
this detector-facing approximation.

## Deviations From The Paper

- The paper's downlink transmit-antenna common support is adapted to uplink BS
  receive-antenna common support.
- The generic downlink observation matrix `D = [P_1 F, ..., P_M F]` is not
  implemented. The current Sionna comb pilot geometry is diagonalized by LS
  normalization and an IFFT.
- Because the transformed observation and prior covariances are diagonal, the
  implementation uses exact scalar Gaussian posterior moments rather than EP
  cavity distributions or factor-to-variable messages. The public estimator
  is consequently named `SBLChannelEstimator`.
- The paper's discrete-tap model implies causal end-padding followed by an
  `N`-point FFT. That is the most faithful reconstruction for its model-matched
  simulations, but it produced a large high-SNR floor for Sionna's
  continuous-delay UMi channels. Centered middle-padding reduced that mismatch
  but remained substantially worse than linear interpolation. The implemented
  full-grid completion is therefore explicit Sionna linear interpolation, not
  a reconstruction step from the paper.
- Detector-facing variances are exact pilot-subcarrier marginals under the
  diagonalized retained-tap model, but omit frequency cross-covariances when
  propagated by Sionna interpolation.

## Verification

Commands run:

```bash
python -m pytest test/mimo_ofdm_chest/papers/wei2017/test_estimator.py -q
python -m py_compile src/sionna_bench/mimo_ofdm_chest/papers/wei2017/estimator.py src/sionna_bench/mimo_ofdm_chest/papers/wei2017/run_benchmark.py test/mimo_ofdm_chest/papers/wei2017/test_estimator.py
python -m sionna_bench.mimo_ofdm_chest.papers.wei2017.run_benchmark --smoke
python -m pytest test/mimo_ofdm_chest/infrastructure/test_pilot_groups.py test/mimo_ofdm_chest/papers/coleri2002/test_interpolation.py test/mimo_ofdm_chest/papers/wei2017/test_estimator.py -q
python -m sionna_bench.mimo_ofdm_chest.papers.wei2017.run_benchmark
```

Results:

- Wei focused tests before the full-comb default change: `3 passed`.
- Wei focused tests after the full-comb default change: `4 passed`.
- Python compilation with the project Sionna environment: passed.
- Current parameterized Wei suite after group vectorization and variance-scale
  regression coverage: `20 passed`.
- Eager and `torch.compile(..., backend="eager", fullgraph=True)` outputs agree
  on the baseline and alternate system geometries. `torch._dynamo.explain` on
  the alternate geometry reports one graph, zero graph breaks, and no break
  reasons. A separate full-graph probe also passed with the canonical
  selected 10-update loop and scalar noise variance. The complete estimator is
  therefore Tier A.
- Current smoke run at 5 dB with linear interpolation:
  - BER: `6.250000e-02`
  - BLER: `3.750000e-01`
  - NMSE: `8.851104e-03`
- Pilot-group, Coleri interpolation, and Wei tests together after the
  full-comb default change: `18 passed`.
- Current Wei and PilotGroup suites after the equation-level correction:
  `30 passed`.
- The calibrated Sionna 2.0.1 linear-interpolation UMi benchmark was run with
  `num_iter=10`,
  `alpha_init=0.00259`, `posterior_variance_scale=0.755`,
  `num_taps=156`, and
  `interpolation_type="lin_time_avg"`. The canonical Wei benchmark file was overwritten at
  `outputs/wei2017/3gpp-umi/SBLChannelEstimator.npz`.
  Its NMSE over the SNR grid
  `[-10, -7.5, -5, -2.5, 0, 2.5, 5, 7.5, 10, 12.5, 15]` is
  `[4.313531e-02, 3.054678e-02, 2.117557e-02, 1.459569e-02,
  1.011717e-02, 7.072674e-03, 4.970072e-03, 3.486605e-03,
  2.423694e-03, 1.663590e-03, 1.134483e-03]`.
- The independent final review accepted the implementation with caveats as a
  comb-specialized uplink SBL adaptation; see `reports/final_report.md`.

Temporary smoke sanity comparisons against built-in LS estimators were run to
`/tmp`:

- `ls-nn` at 5 dB: BER `2.722593e-02`, BLER `3.750000e-01`, NMSE
  `2.012864e-02`.
- `ls-lin` at 5 dB: BER `3.805648e-02`, BLER `3.750000e-01`, NMSE
  `2.192022e-02`.

The Wei smoke NMSE is better than both LS smoke baselines for this one smoke
batch. This is an integration sanity check, not final performance evidence.

## Sionna 2.0.1 Optuna Retuning

A calibrated-Sionna v2 50-trial TPE study evaluated `num_iter`, `alpha_init`,
and `num_taps` with the final linear-interpolation implementation. Each trial
used weighted NMSE over `-10`, `-5`, `0`, `5`, and `10` dB, weights
`10**(Eb/N0/10)`, three batches of four channels per SNR, sampler seed `123`,
and probe seed `4242`. No historical trial was injected.

The search ranges were `num_iter` from 1 through 60, logarithmic `alpha_init`
from `1e-3` through `10`, and `num_taps` from 1 through 156. The raw winner
scored `0.004178389339985936` with 10 iterations,
`alpha_init=0.002592776428399033`, and 156 taps. The search ran for 17.95
seconds on an NVIDIA L40S.

Paired validation used independent seed `314159` and 63 batches of four
channels at every tuning SNR. The existing configuration scored
`0.004325937694561677`; the three-significant-digit candidate scored
`0.0042844525264313345` and was selected. Validation took 25.01 seconds. The
selected canonical algorithm parameters are therefore:

```text
num_iter=10
alpha_init=0.00259
num_taps=156  # equivalent to runner default None on this geometry
```

The full external evidence is stored under `sbl/` in the retained run tree.

## Sionna 2.0.1 Detector-Facing Variance Calibration

After the SBL implementation, interpolation choice, selected 10-update
algorithmic parameters, and compile refactor were frozen, a clean 10-trial TPE
study varied only `posterior_variance_scale` over logarithmic `[1e-3, 1]` and
minimized codeword BLER at 10 dB. Each trial used 63 batches of four channels,
sampler seed `456`, and probe seed `271828`.

Optuna selected raw `0.06491437795831466` with objective
`0.022486772486772486`. Paired validation with independent seed `161803`
scored incumbent `0.755` at `0.01951058201058201` and rounded candidate
`0.0649` at `0.01984126984126984`. The candidate did not reproduce its gain,
so `0.755` remains the estimator and runner default. Search and validation took
25.16 and 12.16 seconds.

A direct regression test confirms identical channel estimates and scaled
error variances.
The full external evidence is stored under `variance/` in the retained run
tree; the SQLite database is not part of the repository.

The calibrated canonical rerun completed in 58.16 seconds. Against the previous
canonical curve, NMSE is 1.1-4.7% higher from -10 through 0 dB and 1.1-3.1%
lower from 2.5 through 15 dB. The -7.5 and -5 dB NMSE points fall outside the
previous 95% intervals; every BER and BLER point remains inside its previous
interval. This low-SNR trade is disclosed and follows the frozen
linear-SNR-weighted selection rule rather than selection on canonical output.
The output SHA-256 is
`9ca84a263f13787cfc31e6bd7271a4d0ce8c64d098b6577c6e79a9b482b59ebb`.

## Remaining Risks

- The implementation is only approved for the current comb-pilot geometry.
- Sionna interpolation does not propagate frequency-domain error
  cross-covariances.
- Smoke metrics are noisy and should not be used as canonical benchmark
  results.
