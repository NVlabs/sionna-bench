# Implementation of the SBL Estimator Adapted from Wei et al., 2017

Publication: C. Wei, Z. Zhang, and H. Liu, "Expectation-propagation based low-complexity channel estimation for massive MIMO systems," IEEE International Conference on Communications (ICC), 2017.

## Abstract

Wei et al., 2017 present an LC-EM-EP channel estimator for sparse delay-domain
massive MIMO channels with common support across co-located antennas. Under the
diagonalized uplink observation model used here, the Gaussian posterior moments
are available exactly as scalar updates. The implemented estimator is therefore
sparse Bayesian learning with a common-support expectation-maximization update;
it does not execute expectation-propagation message passing.

The implemented estimator uses Sionna's Kronecker comb pilots. It divides the received pilot observations by the known pilot symbols, transforms each uniform pilot comb to a delay-domain basis, and applies the paper's Gaussian posterior update with the shared support variance update from Equation (21). It transforms the denoised coefficients back to the observed pilot subcarriers, after which a configured Sionna interpolator completes the resource grid. The canonical default, `lin_time_avg`, applies linear frequency interpolation and averages the resulting pilot-symbol estimates over OFDM time.

## Implemented Classes

`SBLChannelEstimator` in `estimator.py` is a Sionna
`BasePilotChannelEstimator`. It uses Sionna's internal `_PilotGroupMixin` to
form the uniform pilot-comb vectors processed by the sparse Bayesian update.

Main methods:

- `estimate_at_pilot_locations()`: gathers pilot groups, performs LS normalization, maps uniform comb pilots to delay-domain observations, estimates the delay coefficients, and transforms them back to denoised pilot-subcarrier estimates.
- `_delay_observations()`: applies the comb IFFT transform jointly to all pilot groups.
- `_estimate_delay_observations()`: applies the scalar Gaussian posterior update and common-support EM update corresponding to Equations (13)-(16) and (19)-(21).

`run_benchmark.py` exposes `run_benchmark()`, which constructs and evaluates
the estimator through the shared evaluation pipeline. `tuning.py` exposes
`run_tuning()`, which uses the shared Optuna tuning lifecycle.

The complete online estimator, including Sionna interpolation, compiles as one
graph with `torch.compile(..., fullgraph=True)` and is classified as Tier A.

## Modifications and Omissions

The paper is a downlink estimator with superposed BS transmit-antenna pilots. The benchmark is uplink and exposes separated BS receive-antenna pilot observations. The implementation therefore shares the support variance across BS receive antennas for each UE stream and pilot OFDM symbol.

After pilot division and the comb IFFT, the adapted observation model is
diagonal with white delay-domain noise. For fixed `alpha`, the implementation
therefore computes each delay coefficient's Gaussian posterior mean and
variance exactly and uses those moments in the scalar EM update from Equation
(21). It does not implement EP cavity distributions, factor-to-variable
messages, or the paper's `Z_k` and `V_k` approximations. The SBL name reflects
the algorithm that is actually executed.

The implementation is specialized to uniform comb pilots with constant nonzero pilot power and `spacing * group_len == fft_size`. It fails explicitly outside these assumptions. The generic observation-matrix LC-EM-EP solver for arbitrary `D = diag(pilots) F` is not implemented.

The paper's discrete-tap model implies placing the estimated causal taps at the beginning of an `N`-sample delay vector, zero-padding its end, and taking an `N`-point FFT. That is the most faithful full-frequency reconstruction for the paper's model-matched discrete-tap simulations. In the benchmark, however, both this causal reconstruction and a later centered/middle-zero-padding adaptation produced substantially worse high-SNR NMSE than linear interpolation because Sionna's UMi channels use continuous, off-grid path delays. The implementation therefore retains the paper-derived SBL estimator only at the observed pilot subcarriers and exposes Sionna's standard interpolation interface, with `lin_time_avg` as the default. This interpolation is a benchmark adaptation, not a step proposed by Wei et al.

The paper provides posterior delay-domain variances, but it does not define Sionna detector-facing full-grid error variances. In the diagonalized comb model, delay taps have independent scalar posteriors, so this implementation returns their summed marginal variance at each pilot subcarrier. Frequency-domain errors produced from those taps are correlated across subcarriers, but Sionna's `err_var` interface carries only marginal variances and its interpolation does not propagate cross-covariances. `posterior_variance_scale` is the only exposed confidence scale; its calibrated benchmark value is `0.755`.

The paper's block-ISD, BP, and exact-LS oracle comparisons are not implemented because they are external baselines rather than the proposed estimator.

## Benchmark Run

Canonical benchmark command:

```bash
python -m sionna.bench.mimo_ofdm_chest.papers.wei2017.run_benchmark
```

The calibrated Sionna 2.0.1 default uses source commit
`f16208347246d8d17ec69628dfb856b19287b80c` and the explicit UMi TR 38.901
v19.2 channel model. It uses `num_iter=10`, `alpha_init=0.00259`,
`num_taps=None` meaning the full pilot-comb length, and Sionna `lin_time_avg`
interpolation. The detector-facing variance scale is documented below.

The v2 50-trial Optuna TPE study jointly searched `num_iter` from 1 through 60,
`alpha_init` from `1e-3` through `10` logarithmically, and `num_taps` from 1
through the full 156-tap pilot-comb length. It minimized linear-SNR-weighted
NMSE over `-10`, `-5`, `0`, `5`, and `10` dB. Every trial used three batches of
four channels per SNR, sampler seed `123`, and probe seed `4242`; no historical
configuration was injected. The exact weights in increasing-SNR order were
`[0.1, 0.31622776601683794, 1.0, 3.1622776601683795, 10.0]`. The study ran
50 requested trials: 50 completed, zero failed, and zero were pruned.

The raw winner and independently selected setting were:

```text
num_iter=10
raw alpha_init=0.002592776428399033
num_taps=156
raw Optuna objective=0.004178389339985936
selected rounded alpha_init=0.00259
paired rounded-winner objective=0.0042844525264313345
```

`num_taps=156` is equivalent to the runner's canonical `num_taps=None` for
this pilot geometry. Paired validation used independent seed `314159` and 63
batches of four channels at every tuning SNR. The existing
`num_iter=27`, `alpha_init=0.00507` configuration scored
`0.004325937694561677`; the rounded winner scored
`0.0042844525264313345` and was selected. Search and validation took 17.95 and
25.01 seconds, respectively, on an NVIDIA L40S.

Run the study with:

```bash
python -m sionna.bench.mimo_ofdm_chest.papers.wei2017.tuning \
  --study sbl \
  --output-dir /path/to/tuning/mimo_ofdm_chest/wei2017/sbl_v2

python -m sionna.bench.mimo_ofdm_chest.papers.wei2017.tuning \
  --study sbl-validation \
  --output-dir /path/to/tuning/mimo_ofdm_chest/wei2017/sbl_v2
```

The study summary, paired comparison, and trials are stored under the selected
external output directory.

## Detector-Facing Variance Calibration

The v2 calibration held `num_iter=10`, `alpha_init=0.00259`, `num_taps=156`,
`interpolation_type="lin_time_avg"`, and all estimate-producing code fixed.
Ten TPE trials searched `posterior_variance_scale` logarithmically over
`[1e-3, 1]` and minimized codeword BLER at 10 dB. Each trial used the canonical
63 batches of four channels, sampler seed `456`, and probe seed `271828`.
The calibration ran 10 requested trials: 10 completed, zero failed, and zero
were pruned.

Optuna selected raw `0.06491437795831466` with objective
`0.022486772486772486`. Paired validation with independent seed `161803`
scored the existing `0.755` at `0.01951058201058201` and the rounded candidate
`0.0649` at `0.01984126984126984`. The candidate did not reproduce its gain,
so `0.755` remains canonical. Search and validation took 25.16 and 12.16
seconds. The scale changes only detector-facing error variance and cannot
change channel estimates or NMSE.

Run the calibration with:

```bash
python -m sionna.bench.mimo_ofdm_chest.papers.wei2017.tuning \
  --study variance \
  --algorithm-output-dir /path/to/tuning/mimo_ofdm_chest/wei2017/sbl_v2 \
  --output-dir /path/to/tuning/mimo_ofdm_chest/wei2017/variance_scale_v2

python -m sionna.bench.mimo_ofdm_chest.papers.wei2017.tuning \
  --study variance-validation \
  --algorithm-output-dir /path/to/tuning/mimo_ofdm_chest/wei2017/sbl_v2 \
  --output-dir /path/to/tuning/mimo_ofdm_chest/wei2017/variance_scale_v2
```

Study summaries are stored under the selected external output directory.

The retained preflight records Sionna Bench commit
`5012974748d4b8aa6426e563d8fed1380eba3abc` and tracked-diff SHA-256
`5331b65bac8e86d4b2b1342b10251dcb12aeaedbc387dddeae5480aec9baae7b`.
The environment was Python 3.12.3, PyTorch `2.11.0+cu126`, CUDA 12.6,
driver `580.126.09`, NumPy 2.4.4, SciPy 1.17.1, and Optuna 4.9.0 on an
NVIDIA L40S and AMD EPYC 7R13. The retained `pip-freeze.txt` has SHA-256
`b981919b35253baa1f4cbc244763e09292be6a503d42480d3f9003e397884082`.
Canonical evaluation used seed `2026`.

The calibrated Sionna 2.0.1 canonical benchmark completed in 58.16 seconds.
Against the previous canonical curve, NMSE is 1.1-4.7% higher from -10 through
0 dB and 1.1-3.1% lower from 2.5 through 15 dB. The -7.5 and -5 dB NMSE points
fall outside the previous 95% intervals, while every BER and BLER point remains
inside its corresponding interval. This disclosed low-SNR trade is consistent
with the frozen linear-SNR-weighted selection objective. The complete result is stored in
`outputs/mimo_ofdm_chest/wei2017/3gpp-umi/SBLChannelEstimator.npz` with SHA-256
`9ca84a263f13787cfc31e6bd7271a4d0ce8c64d098b6577c6e79a9b482b59ebb`.

Smoke command:

```bash
python -m sionna.bench.mimo_ofdm_chest.papers.wei2017.run_benchmark --smoke
```
