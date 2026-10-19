# Paper Summary

- Paper slug: `coleri2002`
- Full citation: Sinem Coleri, Mustafa Ergen, Anuj Puri, and Ahmad Bahai, "Channel Estimation Techniques Based on Pilot Arrangement in OFDM Systems," IEEE Transactions on Broadcasting, vol. 48, no. 3, pp. 223-229, Sept. 2002. DOI: 10.1109/TBC.2002.804034.
- Task: Channel estimation
- Target repository interface: Sionna OFDM channel estimator returning full resource-grid `h_hat` and `err_var`
- Implementer: Codex
- Date: 2026-06-03

## Core Problem

The paper compares OFDM channel-estimation schemes built around pilot placement. It separates block-type pilot estimation, where all subcarriers are pilots on periodic OFDM symbols, from comb-type pilot estimation, where pilots are uniformly spaced in frequency and must be interpolated to data subcarriers. For comb pilots, the paper estimates the channel at pilot frequencies using LS or LMS and compares linear, second-order, low-pass, spline-cubic, and DFT/IDFT time-domain interpolation. The reported simulations conclude that comb-type LS with low-pass interpolation gives the best robustness for the fast fading cases studied, while decision-feedback block pilots can be attractive only for slow fading.

## Algorithms Or Variants In Scope

- Name: `comb-ls-linear`
  Short description: LS estimate at pilot frequencies using equation (17), followed by linear interpolation using equation (18).
- Name: `comb-ls-second-order`
  Short description: LS pilot estimates from equation (17), followed by the quadratic interpolation formula in equation (19).
- Name: `comb-ls-low-pass`
  Short description: LS pilot estimates from equation (17), zero insertion, and MATLAB `interp` low-pass FIR interpolation.
- Name: `comb-ls-spline`
  Short description: LS pilot estimates from equation (17), followed by MATLAB `spline` cubic interpolation.
- Name: `comb-ls-time-domain`
  Short description: LS pilot estimates from equation (17), IDFT to a pilot-rate time-domain sequence, zero padding, and DFT back to all subcarriers using equations (20)-(22).
- Name: `comb-lms-linear`
  Short description: one-tap LMS adaptation at each pilot frequency, initialized by LS and followed by linear interpolation.
- Name: `block-ls`
  Short description: all-subcarrier block pilot symbol estimated by LS using equation (13), reused over a block of following OFDM symbols.
- Name: `block-mmse`
  Short description: all-subcarrier block pilot symbol estimated by the frequency-domain MMSE estimator in equations (11)-(12).
- Name: `block-decision-feedback`
  Short description: block LS initialization followed by detected-symbol remapping and channel updates using equations (14)-(15).

## Assumptions

- Signal or system model: single-user OFDM with AWGN, possible ICI from Doppler, and frequency-selective time-varying multipath; the simulation uses FFT size 1024, 128 active carriers, pilot ratio 1/8, and Rayleigh or AR fading channels.
- Training, pilot, or input structure: comb pilots are uniformly inserted in frequency; block pilots occupy every subcarrier in a periodic pilot OFDM symbol.
- Required side information: pilot positions and pilot values for LS; covariance matrices and noise variance for MMSE; LMS step-size/state for LMS; decided and remapped data symbols for decision feedback.
- Output quantity: channel transfer estimate at data subcarriers or all subcarriers, then equalized symbols by dividing received values by the estimated channel.

## Required Assets

- Datasets: none for the implemented comb-LS interpolation variants.
- Learned weights: none.
- Statistics or covariance files: required for faithful block/comb MMSE variants, not provided by the paper for the current benchmark channel.
- Other `aux` candidates: optional MATLAB reference vectors for validating `coleri-low-pass` against exact MATLAB `interp` behavior.

## Potential Reimplementation Risks

- Missing details: LMS step size is not specified; second-order/low-pass/spline boundary behavior is not stated by the paper. The implementation uses offset-aware constant edge extension for second-order interpolation, finite low-pass filtering with virtual constant edge pilots, and MATLAB-like not-a-knot spline interpolation with clipped constant edge queries.
- Benchmark mismatch: the benchmark uses a multi-user Sionna Kronecker pilot pattern on OFDM symbols 2 and 11, not pilots in every OFDM symbol or all-subcarrier block pilots.
- Surrogate risk: replacing LMS tracking with per-frame smoothing would not be the paper algorithm; low-pass interpolation should remain zero-insertion plus low-pass FIR interpolation following public MATLAB documentation.
- Numerical concerns: equation (19) prints `alpha = l/N`, but the interpolation coordinate must be `alpha = l/L`; otherwise `alpha` never reaches the next pilot in a pilot interval. Equation (21) also appears to have indexing and normalization typos, so the implementation uses the stated DFT/IDFT zero-padding method with a pilot pass-through sanity check instead of literal transcription.
- Other: the paper does not provide a detector-facing error-variance model, so the implementation uses Sionna's LS pilot error variance `no/|pilot|^2` and propagates it through the paper-local interpolator. The benchmark has pilots only on OFDM symbols 2 and 11, so the implementation uses fixed linear OFDM-time interpolation between pilot-bearing symbols; this is an interface adaptation to the benchmark, not part of the paper's per-symbol comb-pilot assumption.
