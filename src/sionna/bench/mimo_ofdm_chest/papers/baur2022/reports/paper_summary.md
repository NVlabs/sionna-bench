# Paper Summary

Paper: M. Baur, B. Fesl, M. Koller, and W. Utschick, "Variational Autoencoder Leveraged MMSE Channel Estimation," 2022 56th Asilomar Conference on Signals, Systems, and Computers.

## Problem

Baur et al., 2022 propose a data-driven channel estimator for a SIMO uplink training model. After pilot decorrelation, the base station observes

```text
y = h + n
```

as in Equation (1), where `h` is the receive-array channel vector and `n` is complex Gaussian noise with covariance `Sigma`. The goal is to estimate `h` from one noisy snapshot.

The paper assumes a ULA and uses the fact that ULA channel covariances are Toeplitz and approximately diagonalized by a DFT matrix, Equations (2) and (13). The VAE is trained on DFT-domain channel vectors so that the decoder can output a diagonal conditional covariance.

## Algorithmic Idea

The VAE models the channel distribution as conditionally Gaussian:

```text
h | z ~ CN(mu_{h|z}, C_{h|z})
```

as in Equation (8). For a latent value `z`, the inner conditional mean in the total-expectation decomposition, Equations (9)-(11), has the LMMSE form

```text
E[h | z, y] = mu_{h|z} + C_{h|z}(C_{h|z}+Sigma)^(-1)(y - mu_{h|z})
```

from Equation (10). The decoder produces a DFT-domain mean `m(z)` and diagonal covariance `c(z)`, which correspond to antenna-domain moments via Equation (13). The paper uses one latent sample, namely the encoder mean `z = mu(x)`, giving the approximation in Equation (12).

In the DFT domain, and with white observation noise, the estimator can be evaluated elementwise:

```text
x_hat = m(z) + c(z) / (c(z) + sigma2) * (x_obs - m(z))
h_hat = F^H x_hat
```

where `x_obs = F y`.

## Variants

The paper defines three VAE channel estimators in Section III-C:

- `VAE-genie`: The encoder input during training and estimation is the true transformed channel `x = F h`. This is an oracle estimator and is not practically deployable.
- `VAE-noisy`: During training, the encoder input is noisy `x_tilde = F y`, while the decoder likelihood is evaluated against the true transformed channel `x = F h`. At estimation time only noisy observations are used.
- `VAE-real`: During training and estimation, the encoder input is noisy `x_tilde = F y`. The decoder likelihood is also evaluated against `x_tilde`, but the covariance term used in the likelihood is `diag(c(z)) + sigma2 I`, forcing the decoder output `c(z)` to represent channel covariance rather than observation covariance.

All variants use `z = mu(x)` during estimation rather than Monte Carlo averaging over Equation (11). Section III-D specifies the neural-network architecture at a high level: real and imaginary parts are stacked, the encoder uses three Conv-BN-ReLU blocks with kernel size 7, stride 2, and channel counts 1 -> 8 -> 32 -> 128, followed by linear latent mean and variance heads. The decoder is symmetric. The paper uses Adam, learning rate `1e-4`, and free bits.

## Inputs, Assets, And Assumptions

Required training data are channel vectors drawn from the target radio scenario. For `VAE-noisy`, noiseless channels are needed during training because the decoder likelihood uses `F h`. For `VAE-real`, only noisy observations are used by the training objective, although the simulator still generates the underlying channels to synthesize those observations.

The paper trains separate VAEs for individual SNR values, except for `VAE-genie`. Therefore, benchmark integration should use one trained asset per configured SNR for the practical variants.

Important underspecified details:

- The latent dimension `L` is not given in the paper text.
- The free-bits threshold is not specified.
- Padding/cropping details of the convolutional decoder are not specified.
- The paper uses ULA receive arrays; the current benchmark uses a 4x4 planar base-station array. A DFT over the flattened receive-antenna index is an approximation rather than the paper's exact ULA/circulant setting.
- The paper does not define detector-facing marginal channel-estimation error variances for Sionna's detector.

## Benchmark Mapping

The closest local mapping is a pilot-domain estimator. Sionna's LS estimator converts each nonzero pilot resource element into a noisy channel vector over the receive antennas. The Baur VAE estimator can then denoise each receive-array vector independently. Full OFDM-grid values should be produced by explicit Sionna interpolation afterward, because the paper does not define interpolation across OFDM subcarriers or OFDM symbols.

This mapping is close to the existing spatial GMM implementation in `papers/koller2022`: aggregate training statistics across all pilot groups, reuse one shared learned/statistical model for all pilot groups, and apply the estimator to each pilot resource element.
