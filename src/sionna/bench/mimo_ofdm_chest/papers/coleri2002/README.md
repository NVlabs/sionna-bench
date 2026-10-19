# Implementation of the Estimators from Coleri et al., 2002

Publication: S. Coleri, M. Ergen, A. Puri, and A. Bahai, "Channel Estimation Techniques Based on Pilot Arrangement in OFDM Systems," IEEE Transactions on Broadcasting, vol. 48, no. 3, pp. 223-229, 2002.

## Abstract

This implementation covers the LS interpolation methods from Coleri et al.,
2002 for uniformly spaced pilot subcarriers. The implemented contribution is
not a new way to obtain channel estimates at pilot resource elements: those
pilot estimates are standard least-squares estimates. The paper-specific part
is how those pilot-domain estimates are interpolated across OFDM subcarriers.

Four subcarrier-domain interpolation approaches are implemented:

- second-order polynomial interpolation;
- low-pass interpolation by zero insertion followed by FIR filtering;
- cubic spline interpolation;
- time-domain interpolation by pilot-rate IDFT, zero padding, and DFT back to
  the full frequency grid.

The benchmark resource grid has nonzero pilots only on selected OFDM symbols,
so all four methods reconstruct the subcarrier response on pilot-bearing
symbols first and then use fixed linear interpolation over OFDM time.

## Implemented Classes

### `CombInterpolator`

`CombInterpolator` is the paper-specific Sionna
`BaseChannelInterpolator`. It expects uniformly spaced pilot subcarriers and
uses Sionna's developer-facing `_PilotGroupMixin` to gather the pilot estimates
belonging to each transmit stream and pilot-bearing OFDM symbol.

Its main methods are:

- `__call__`: Sionna interpolator entry point. It gathers pilot groups,
  applies subcarrier interpolation to each group, and then linearly
  interpolates the reconstructed pilot-bearing symbols over OFDM time.
- `_interpolate_frequency`: dispatches one pilot group to the selected Coleri
  subcarrier interpolation method.
- `_interpolate_with_second_order_polynomial`: implements the second-order
  interpolation formula corresponding to equation (19).
- `_interpolate_with_lowpass_filter`: implements the low-pass interpolation
  method described in Section V as MATLAB `interp`-style zero insertion plus a
  low-pass FIR filter.
- `_interpolate_with_cubic_spline`: implements the spline interpolation method
  described in Section V using precomputed cubic-spline interpolation weights.
- `_interpolate_with_time_domain_transform`: implements the DFT/IDFT
  interpolation corresponding to equations (20)-(22).
- `_interpolate_time`: applies the benchmark adaptation of linear interpolation
  between pilot-bearing OFDM symbols.
- `_interpolate_error_variance`: propagates Sionna LS pilot error variances to
  the full frequency grid for detector compatibility.

### `LSCombEstimator`

`LSCombEstimator` is the public channel estimator used for benchmarking. It is
a subclass of Sionna's `LSChannelEstimator` that supplies
`CombInterpolator` as its interpolation strategy. Pilot extraction and LS
estimation are therefore inherited directly from Sionna.

Its main methods are:

- `__init__`: constructs the selected `CombInterpolator` variant.
- inherited `estimate_at_pilot_locations`: computes Sionna's standard
  pilot-domain LS estimate, corresponding to equation (17).

Benchmark outputs are named `LSCombEstimator (<method>)`, where `<method>` is
one of `second-order`, `low-pass`, `cubic spline`, or `time-domain`.

## Supported Geometry And Compilation

The implementation supports uniformly spaced frequency-comb pilots with one
common spacing, full-grid comb coverage, and the same number of active
pilot-bearing OFDM symbols for every transmit stream. Different stream comb
offsets are supported. A spacing of one is treated as the identity case for
low-pass interpolation. Nonuniform, incomplete, or unequal active comb layouts
are rejected with explicit messages.

All four complete estimator inference paths compile as Tier A with
`torch.compile(..., fullgraph=True)`: eager and compiled outputs agree, and
`torch._dynamo.explain` reports one graph with no graph breaks on the alternate
shared system configuration. Cubic-spline weight matrices are generated once
with SciPy during construction and registered as Torch buffers. SciPy is not
called from compiled inference.

## Modifications And Omissions

The following modifications have been made compared to how the methods were
presented in the paper:

- The benchmark has pilots on only a subset of OFDM symbols, while the paper's
  interpolation methods are formulated per OFDM symbol. The implementation
  therefore linearly interpolates over OFDM time after subcarrier
  interpolation.
- Frequency-edge handling is not fully specified by the paper. The
  implementation uses offset-aware constant edge extension for second-order
  interpolation, finite low-pass filtering with virtual constant edge pilots,
  and clipped constant-edge queries for spline interpolation.
- The low-pass method follows public MathWorks documentation for `interp` and
  `intfilt`, but is not claimed to be bit-exact to MATLAB internals.
- The low-pass class defaults are the canonical Optuna-selected settings
  `lowpass_filter_half_length=16` and `lowpass_cutoff=0.105`. The documented
  MATLAB-style settings `4` and `0.5` remain available as explicit arguments.
- Equation (19) is implemented with the interval coordinate `l/L`, consistent
  with equation (18) and `k=mL+l`, rather than the printed `l/N`.
- Equation (21) is treated as a flawed printed zero-padding formula. The
  implementation follows the surrounding text and Fig. 3: IDFT, middle zero
  padding, and DFT back to all subcarriers.
- The paper does not define detector-facing error variances for the channel
  estimates. The implementation propagates Sionna LS pilot variances and
  multiplies them by a shared, calibrated `err_var_scale` so that BER and BLER
  evaluation can use the existing Sionna receiver chain.

The following paper variants are not implemented here:

- Linear subcarrier interpolation is already covered by Sionna's built-in
  `LinearInterpolator`.
- LMS interpolation is not implemented because the current benchmark uses independent
  Monte-Carlo frames and does not provide a reviewed sequential update rule or
  step-size choice.
- MMSE variants for pilot-subcarrier estimation are not implemented because the paper
  does not provide a self-contained method specification.
- All-subcarrier pilot-symbol and decision-feedback methods are not implemented
  because they require a different resource grid and receiver loop from the
  current pilot-pattern estimator contract.

## Hyperparameter Tuning And Calibration

### Low-Pass Interpolation

The v2 algorithm study tuned only the two exposed low-pass FIR parameters.
It minimized
`sum_s 10**(s/10) * NMSE_s / sum_s 10**(s/10)` at Eb/N0 values
`(-10, -5, 0, 5, 10)` dB. Each of 50 seeded TPE trials reused the same
three batches of four channels per SNR, for 60 SNR/channel pairs per trial.
The sampler seed was `123`, the cached-probe seed was `4242`, and no
historical configuration was injected into the study. The search spaces were
integer `lowpass_filter_half_length` in `[1, 16]` and uniform
`lowpass_cutoff` in `[0.1, 1.0]`.

The raw winner was half-length `15` and cutoff
`0.13096202677999685`, with weighted NMSE `0.013767726086134014` on the
search probe. Its rounded candidate `(15, 0.131)` was compared with the
current `(16, 0.105)` on an independent paired stream using seed `314159`,
63 batches of four channels at each of the five SNRs, and the same weighted
objective. The candidate scored `0.015799561211600647`; the current setting
scored `0.014546855397130133`. The current half-length `16` and cutoff
`0.105` were therefore retained.

The search took 21.25 seconds and paired validation took 27.19 seconds on an
NVIDIA L40S. They were run as:

```bash
python -m sionna.bench.mimo_ofdm_chest.papers.coleri2002.tuning \
    --study lowpass \
    --output-dir tuning/mimo_ofdm_chest/coleri2002/lowpass_v2
python -m sionna.bench.mimo_ofdm_chest.papers.coleri2002.tuning \
    --study lowpass-validation \
    --output-dir tuning/mimo_ofdm_chest/coleri2002/lowpass_v2
```

### Detector-Facing Error Variance

All four variants use one shared `err_var_scale`. After the interpolation
parameters were frozen, a separate 10-trial seeded TPE study searched the
logarithmic interval `[1e-3, 1]`. Its objective was the equal-method mean of
actual codeword BLER for the four methods at 10 dB. It used 63 batches of four
channels, sampler seed `456`, and cached calibration seed `271828`; cached
channel estimates and unscaled variances were reused across scalar trials.

The raw winner was `0.4540521216492649`, with mean BLER
`0.060267857142857144`. Independent paired validation used seed `161803` and
another 63 batches of four channels. The rounded candidate `0.454` scored
`0.057953042328042326`, while the current `0.868` scored
`0.055803571428571425`. The shared scale `0.868` was therefore retained.

Calibration took 78.64 seconds and paired validation took 30.90 seconds on
the same L40S. They were run as:

```bash
python -m sionna.bench.mimo_ofdm_chest.papers.coleri2002.tuning \
    --study variance \
    --output-dir tuning/mimo_ofdm_chest/coleri2002/variance_scale_v2
python -m sionna.bench.mimo_ofdm_chest.papers.coleri2002.tuning \
    --study variance-validation \
    --output-dir tuning/mimo_ofdm_chest/coleri2002/variance_scale_v2
```

The studies used Sionna Bench commit
`5012974748d4b8aa6426e563d8fed1380eba3abc`, tracked-diff SHA-256
`7271ee02fc968085f4eeb2bb4d92326aae7336f04c912fc6642ee31319172ce2`,
and Sionna source commit
`f16208347246d8d17ec69628dfb856b19287b80c`, including calibrated channel-model
commit `ad1609ea1` and the explicit UMi TR 38.901 v19.2 model. The principal
runtime versions were Python 3.12.3, Sionna 2.0.1, PyTorch 2.11.0+cu126,
CUDA 12.6, Optuna 4.9.0, NumPy 2.4.4, and SciPy 1.17.1. The host used an
AMD EPYC 7R13 CPU, NVIDIA driver 580.126.09, and an NVIDIA L40S.

All requested trials completed: the low-pass study recorded 50 complete
trials with no failures or pruning, and variance calibration recorded ten
complete trials with no failures or pruning. The paper-specific environment
check is retained in `tuning/mimo_ofdm_chest/coleri2002/preflight.txt`. The complete dependency
snapshot is shared at
`tuning/mimo_ofdm_chest/evidence/environments/v2-l40s-20260814/pip-freeze.txt`, with SHA-256
`b981919b35253baa1f4cbc244763e09292be6a503d42480d3f9003e397884082`.
Structured study evidence is retained under `tuning/mimo_ofdm_chest/coleri2002/lowpass_v2/`
and `tuning/mimo_ofdm_chest/coleri2002/variance_scale_v2/`; raw databases, logs, and timing
records remain external evidence. Coleri has no persistent inference asset.

## Benchmark Run

The canonical runner uses the default 3GPP UMi configuration:

```bash
python -m sionna.bench.mimo_ofdm_chest.papers.coleri2002.run_benchmark
```

No CLI overrides were used. The run evaluated all four methods with the
retained low-pass settings and shared `err_var_scale=0.868` over the canonical
11-point Eb/N0 grid, 63 Monte Carlo batches of size four per SNR, four UEs,
the `4x4` BS array, 52 PRBs, pilot-bearing OFDM symbols `(2, 11)`, and
canonical evaluation seed `2026`.

The Sionna 2.0.1 canonical rerun took 3 minutes 44.30 seconds on the L40S.
Every NMSE, BER, and BLER value for every method is bit-identical to the
corresponding result in the comparison worktree. Results and their
constructor arguments are recorded under
`outputs/mimo_ofdm_chest/coleri2002/3gpp-umi/` in the scenario-local `results.json`.
