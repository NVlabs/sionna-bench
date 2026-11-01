# Paper Summary

Publication: B. Fesl, M. Baur, F. Strasser, M. Joham, and W. Utschick,
“Diffusion-Based Generative Prior for Low-Complexity MIMO Channel Estimation,”
IEEE Wireless Communications Letters, vol. 13, no. 12, pp. 3493–3497, 2024.
[DOI: 10.1109/LWC.2024.3474570](https://doi.org/10.1109/LWC.2024.3474570).

## Problem

The paper estimates a narrowband MIMO channel after unitary pilot
decorrelation, yielding `y = h + n` with known white circular complex noise.
A learned angular-domain prior replaces a fixed parametric channel prior.
The paper assumes unit average complex entry power and illustrates a 64×16
receive/transmit matrix. Its output is a channel estimate, not an OFDM
interpolator or a detector-facing posterior variance.

## Algorithm

An unconditional, timestep-conditioned CNN predicts diffusion noise. Training
uses a variance-preserving forward process and unweighted epsilon-MSE. The
released recipe uses 100 linear beta steps from `1 - 10000/10001` to `0.1`,
100,000 training matrices and 10,000 validation matrices. Real and imaginary
components occupy separate channels; each real noise component has variance
1/2. One shared model covers all timesteps and observation SNRs.

At inference, Algorithm 1 normalizes the angular LS observation by its actual
noise variance and matches its linear SNR to the diffusion schedule. It then
composes the remaining reverse conditional means, including the matched
transition, without adding random noise, and inverts the angular transform.
A finite trained network approximates the conditional mean; the paper's
asymptotic result has additional learning and schedule assumptions.

The original CNN has two 3×3 pre-convolutions `(2,32,64)`, a 16-dimensional
sinusoidal embedding with a 16-to-128 affine layer, residual feature modulation
`features*(1+scale)+shift`, and three 3×3 post-convolutions `(64,43,22,2)`.
ReLU follows the first pre-convolution and first two post-convolutions.
There is no pooling, batch normalization, or output clipping. The network has
55,025 parameters.

## Relevant Variants

| Method or experiment | Implementation status |
| :-- | :-- |
| Angular deterministic diffusion, released 100-step recipe | Implemented with the approved receive-array/OFDM adaptation. |
| Spatial-domain ablation | Not implemented; needs a separately trained identity-transform prior. |
| SNR-side-information mismatch ±3 dB | Not implemented; requires a separately controlled robustness experiment. |
| Alternative total diffusion-step schedules | Not implemented; requires schedule-specific training and assets. |
| Generic stochastic sampling or alternative prediction targets in the author library | Not implemented; outside the selected WCL algorithm and training recipe. |
| Paper comparison estimators | Not reproduced; corresponding methods need their own models, statistics, or oracle inputs. |
| Later single-step extension | Outside this contribution; no single-step mode is implemented. |

## Benchmark Mapping and Assets

The implementation denoises one complete receive-array vector per nonzero
OFDM pilot. Its two FFT/CNN axes are physical receive rows and columns,
replacing the paper's receive/transmit axes. Sionna's pilot-domain estimator
then applies explicit `lin_time_avg` interpolation to the remaining grid.
Actual pilot noise is `no/abs(pilot)**2`; Eb/N0 is not the diffusion SNR.

New scenario-matched data and weights are needed. Original 64×16 checkpoints
are not valid calibrated 4×4 UMi assets. The selected implementation uses one
shared model, training-only population-power normalization, and separate
schema-2 dataset/model manifests. Generation and training are explicit and
external. An empirical scaled-LS variance supplies the detector interface;
its independent calibration is recorded in
[implementation_report.md](implementation_report.md).

## Source Qualification and Risks

The human designated [arXiv:2403.03545v1](https://arxiv.org/abs/2403.03545v1)
authoritative. The complete preprint, including Algorithm 1 and its
architecture figure, was inspected. Its reviewed PDF SHA-256 was
`8b99d0a8cf36d61378cb737fd98aa015e81693aa62cf65a95f3cf845e421ddd4`.
The final IEEE indexed text was compared; no production-algorithm change was
identified. The final binary PDF could not be retrieved, so complete final-PDF
inspection is not claimed. The source paper is excluded from the contribution.

Operational details were checked against author release `wcl-2024`, commit
`fe43f0aa7d10da06f3f37217168625bf6c197cbb`, and scheduler reference
`diffusers-dmse`, commit `63dfa20e0003a47295eb22cdb7120a04e4903aec`.
These are reference checks, not runtime dependencies.

The release loops over `k-1,...,0` after matching coefficient index `k`;
Algorithm 1 includes `k,...,0`. The approved implementation follows the paper
and explicitly documents this correction. Physical antenna ordering,
training-only normalization, complex noise factors, and pilot powers require
careful checks. Correlated pilot vectors are not independent channel samples.
The 4×4 prior and OFDM interpolation need not reproduce the original paper's
accuracy, and the empirical detector variance is not posterior covariance.
