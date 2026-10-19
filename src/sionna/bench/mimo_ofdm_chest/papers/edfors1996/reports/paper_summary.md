# Paper Summary

- Paper ID: `edfors1996`
- Source file: `src/sionna_bench/mimo_ofdm_chest/papers/edfors1996/paper.pdf`
- Citation used here: Ove Edfors, Magnus Sandell, Jan-Jaap van de Beek, Sarah Kate Wilson, and Per Ola Borjesson, "Analysis of DFT Based Channel Estimators for OFDM," 1996 report version of work presented in part at IEEE VTC 1995.
- Task: channel estimation
- Implementer: Codex
- Date: 2026-05-06

## Problem

The paper studies low-complexity linear OFDM channel estimators. The receiver is
assumed to know the transmitted symbols on all tones of one OFDM symbol, so it
can form a least-squares frequency-domain observation of the whole channel
attenuation vector. The goal is to smooth that noisy vector using channel
correlation while avoiding the full `M x M` LMMSE matrix multiplication.

## System Model

Chapter 2 models one OFDM symbol as parallel Gaussian channels with correlated
channel attenuations:

```text
x = W g + n
```

where `W` is diagonal with transmitted symbols, `g` is the frequency-domain
channel attenuation vector, and `n` is white complex Gaussian noise. The cyclic
prefix is assumed long enough to avoid ISI and preserve subcarrier
orthogonality. The channel is slowly Rayleigh fading and constant during one
OFDM symbol.

Chapter 3 starts from the LS/back-rotated observation

```text
g_ls = W^{-1} x = g + n_tilde
```

and then derives a fixed LMMSE smoother for `g_ls` based on the covariance
matrix `R = E[g g^H]`. The paper then moves the smoothing into the cyclic
time-domain impulse-response domain by applying an IDFT, a time-domain linear
map, and a DFT back to frequency.

## Algorithmic Variants

The paper analyzes five relevant estimators:

- LS reference: no smoothing after back-rotation.
- Full LMMSE reference: dense frequency-domain smoothing using `R`.
- Estimator A: DFT-domain approximation that keeps the `L` time-domain
  coefficients with largest channel power and applies a full `L x L` LMMSE
  matrix among those coefficients.
- Estimator B: same selected `L` coefficients, but ignores cross-correlation and
  applies only scalar diagonal LMMSE weights.
- Estimator C: same selected `L` coefficients, but uses them directly and zeros
  all excluded coefficients.

The cumulative restrictions are summarized in Chapter 3, Table 3.1, and the
matrix forms and MSE expressions are given in Appendix A. The paper concludes
that Estimator B with all `M` coefficients is the best complexity/performance
tradeoff among the DFT-based estimators because it removes the high-SNR error
floor without using the dense LMMSE matrix.

## Required Inputs And Side Information

The paper estimators need:

- LS frequency-domain channel observations over a vector of known tones.
- A channel covariance matrix, or equivalently the covariance/variances of the
  IDFT-domain channel coefficients.
- A noise variance for the LS observation.
- A tap budget `L`, and an ordering of IDFT-domain coefficients by decreasing
  channel power.

The paper obtains covariance from its assumed channel model and power-delay
profile. In this repository, the corresponding covariance must be estimated as
an auxiliary asset from the active Sionna UMi benchmark channel.

## Outputs

The paper outputs an estimated frequency-domain channel vector for one OFDM
symbol. The repository benchmark requires a full resource-grid estimate and
error variance tensor, so a benchmark implementation must additionally map from
pilot-domain estimates to the full grid through explicit interpolation.

## Ambiguities And Risks

- The paper assumes known transmitted symbols on all tones, but the benchmark
  exposes only pilot observations. Applying the DFT-domain estimator to the
  compact pilot-frequency vector is a necessary benchmark adaptation, not the
  exact all-tone receiver in the paper.
- The paper does not define inter-OFDM-symbol interpolation. The benchmark has
  two pilot-bearing symbols and data on the remaining symbols, so this must be
  delegated to an explicit Sionna interpolator.
- The paper's covariance comes from an analytical channel model; the benchmark
  covariance must be empirically estimated from Sionna UMi samples and cached in
  `assets/`.
- The paper gives MSE expressions, but not the detector-facing per-resource
  `err_var` tensor required after sparse-pilot adaptation and interpolation.
  The implementation therefore returns a scalar multiple of the LS observation
  variance as a detector-confidence fallback, not as a paper-derived posterior.
- Estimators with `L < M` intentionally drop channel power and can show a
  high-SNR error floor. This is expected behavior, not automatically a bug.
