# Implementation Plan

Paper: M. Baur, B. Fesl, M. Koller, and W. Utschick, "Variational Autoencoder Leveraged MMSE Channel Estimation," 2022 56th Asilomar Conference on Signals, Systems, and Computers.

## Harness Summary

The channel-estimation benchmark builds a Sionna `ResourceGrid` from `configs/umi_3gpp_config.py` or `configs/smoke_config.py`. The default scenario has 52 PRBs, 14 OFDM symbols, Kronecker pilots on OFDM symbols 2 and 11, four UEs with one stream each, a 4x4 single-polarized base-station array, 16-QAM, LDPC coding, and UMi uplink channels without path loss or shadow fading. The smoke config keeps the same physical layout but uses one SNR point, smaller batches, and one Monte Carlo iteration.

`evaluation.channel_estimation.evaluate_channel_estimator()` builds the resource grid, instantiates an estimator as `estimator_class(resource_grid, **estimator_kwargs)`, wraps it in `OFDMChannelEstimationSystem`, sweeps the configured SNR grid, and writes `ebno_dbs`, `nmse`, `ber`, and `bler` to a `.npz` file. Estimator inputs are

```text
y  : [B, num_rx, num_rx_ant, num_ofdm_symbols, fft_size] complex
no : scalar or broadcastable noise variance
```

and estimator outputs must be

```text
h_hat   : [B, num_rx, num_rx_ant, num_tx, num_streams_per_tx,
           num_ofdm_symbols, fft_size] complex
err_var : same shape, real-valued
```

The built-in baselines are `ls-nn`, `ls-lin`, `lmmse`, and `perfect`. Paper outputs live in `outputs/<paper_id>/<config-slug>/`, while smoke outputs live under `outputs/<paper_id>/smoke/<config-slug>/`. Reusable learned assets belong under `aux/<config-slug>/`.

`src.pilot_groups.PilotGroupMixin` provides pilot coordinate grouping, gathering, and scattering. For this paper the relevant grouping is `group_over=()`: one group per nonzero pilot resource element, with the receive-antenna dimension treated as the vector estimated by the paper method.

## Architecture Decision

Use Sionna's pilot-domain estimator architecture:

```text
LS at pilot locations -> Baur VAE denoising per receive-array vector
-> explicit Sionna interpolation over the resource grid
```

This is the correct default because the paper's observation model, Equation (1), is the pilot-decorrelated vector `y = h + n` and the paper does not consider OFDM interpolation. It assumes the estimator receives one noisy SIMO channel vector, not a sparse OFDM pilot grid. Therefore, `FullGridChannelEstimator` is not appropriate for the first faithful implementation.

The estimator should subclass `BaseChannelEstimator` and `PilotGroupMixin`, use `LSChannelEstimator.estimate_at_pilot_locations()` to obtain noisy channel vectors and LS observation variances, apply the VAE estimator at pilot locations, and pass `interpolation_type="lin_time_avg"` explicitly. The interpolation choice is a benchmark adaptation, matching the existing spatial GMM estimator for a receive-array statistical estimator.

The paper's DFT diagonalization argument is derived for a ULA with Toeplitz/circulant covariance, Equations (2) and (13). The benchmark uses Sionna's flattened receive-antenna dimension for a 4x4 planar base-station array. This implementation will use a unitary 1D DFT over the receive-antenna order exactly as Sionna exposes it in the `num_rx_ant` tensor axis. This keeps the estimator algebra unitary and preserves white-noise variance under the transform, but it is a benchmark adaptation rather than a reproduction of the paper's ULA simulation curves. The transform convention and receive-vector length must be stored in every VAE asset metadata file.

## Method: VAEChannelEstimator (noisy)

Status: `implement`

Paper location:

- Section III-C, variant 2
- Equations (1), (5)-(7), (10), (12), and (13)
- Section III-D for the VAE architecture and training setup

Paper method:

During training, the encoder receives the noisy transformed observation
`x_tilde = F y`, while the decoder likelihood is evaluated against the clean
transformed channel `x = F h`. During estimation, only `x_tilde` is used. The latent point is `z = mu(x_tilde)` as in Equation (12). The decoder outputs `m(z)` and diagonal channel covariance `c(z)`, which are used in the LMMSE expression from Equation (10).

Target abstraction:

One shared class `VAEChannelEstimator` in `src/sionna_bench/mimo_ofdm_chest/papers/baur2022/estimator.py`, configured with `variant="noisy"`. The leaderboard/result name should be `VAEChannelEstimator (noisy)`.

Implementation pseudocode:

```text
__init__(resource_grid, cfg, variant="noisy", ...):
    BaseChannelEstimator.__init__(interpolation_type="lin_time_avg")
    PilotGroupMixin._init_pilot_groups(group_over=())
    collect fixed pilot-group coordinates and pilot values
    load or train one VAE asset per configured SNR
    register the Sionna noise variances corresponding to those SNRs
    register unitary DFT matrix over flattened receive antennas as a buffer
    store trained VAE modules in a ModuleList

estimate_at_pilot_locations(y_pilots, no):
    h_ls, ls_err = LSChannelEstimator.estimate_at_pilot_locations(y_pilots, no)
    select VAE checkpoint whose registered Sionna noise variance matches no
    grouped_h_ls = _gather_pilot_groups(h_ls)
    grouped_ls_err = _gather_pilot_groups(ls_err)
    for each pilot group:
        values = grouped_h_ls[..., group, 0]       # receive-array vector
        error_variance = grouped_ls_err[..., group, 0]
        estimates, variances = _estimate_group(values, error_variance)
        scatter estimates and variances back to pilot order
    return h_out, err_out

_estimate_group(values, error_variance):
    x_obs = F @ values                              # Equation (13) domain
    real_stacked = stack(real(x_obs), imag(x_obs))
    z = encoder_mean(real_stacked)                  # Equation (12)
    mean_real, variance = decoder(z)
    mean = unstack_complex(mean_real)
    gain = variance / (variance + error_variance)   # Equation (10), DFT domain
    x_hat = mean + gain * (x_obs - mean)
    h_hat = F^H @ x_hat
    posterior_variance_x = variance * error_variance / (variance + error_variance)
    err_var = diag(F^H diag(posterior_variance_x) F)
    return h_hat, err_var * err_var_scale
```

For a unitary DFT, `diag(F^H diag(posterior_variance_x) F)` is the average of
the DFT-domain posterior variances repeated over the antenna entries of the
same receive-array vector. Returning the untransformed DFT-domain posterior
variance would be incorrect because Sionna's detector expects antenna-domain
marginal error variances.

SNR-specific checkpoint selection:

```text
training_snr_dbs defaults to cfg.snr_db_range()
for each snr_db:
    training_no = ebnodb2no(snr_db, cfg.num_bits_per_symbol, cfg.coderate, resource_grid)
    train/load one VAE checkpoint keyed by training_no and snr_db
at runtime:
    runtime_no = mean(no) cast to estimator dtype
    select checkpoint minimizing |log(runtime_no) - log(training_no)|
    assert the selected log-distance is below 1e-3
```

Assets are therefore keyed by the actual Sionna noise variance used by the
benchmark, not by the paper's per-sample channel SNR definition. This is a
documented benchmark adaptation because the Sionna harness controls `no`
through `Eb/N0`.

Training asset:

```text
aux/<config-slug>/baur2022_vae/noisy/<cache-key>/vae_params.npz
aux/<config-slug>/baur2022_vae/noisy/<cache-key>/manifest.json
```

The path must be built from `evaluation.asset_store.aux_slug_for_config(cfg)`.
The cache key will include at least:

```text
variant, receive-vector length, latent_dim, train_batches, train_batch_size,
validation_batches, validation_batch_size, epochs, optimizer_batch_size,
learning_rate, free_bits, training SNR/no token, err_var policy, seed
```

`vae_params.npz` stores the PyTorch state dictionary tensors by key order, as in
the Melgar DNN cache. `manifest.json` must include:

- variant and config slug;
- receive-vector length and Sionna antenna flattening convention;
- DFT convention: unitary 1D DFT over the Sionna `num_rx_ant` axis;
- training `Eb/N0` and derived Sionna `no`;
- pilot-power convention `no / |pilot|^2`;
- network architecture, including convolution channels, kernel size, stride,
  padding/cropping convention, latent dimension, and covariance positivity
  parameterization;
- optimizer, learning rate, optimizer batch size, free-bits value, train and
  validation sizes, seed, selected epoch, and loss-history summary;
- validation criterion and selected `err_var_scale` when canonical results are
  generated.

The loader must reject incompatible metadata. If a path exists with mismatched
metadata, the implementation should raise or train under a distinct cache key;
it must not silently reuse incompatible weights.

Training pseudocode:

```text
for each requested training SNR:
    sample channel batches from the configured channel source
    for every nonzero pilot group:
        collect clean receive-array vector h
        draw complex Gaussian LS noise with variance no / |pilot|^2
        y = h + noise
        encoder input = F y
        decoder target = F h
    optimize negative ELBO:
        reconstruction term from Equation (6) with complex diagonal covariance
        KL term from Equation (7)
        optional free-bits threshold exposed as free_bits
    select checkpoint by validation ELBO/reconstruction loss
```

Training and validation protocol:

- The paper learning rate `1e-4` is the default.
- Before canonical outputs, run a bounded architecture/training validation grid
  separately for `variant="noisy"` and `variant="real"`:

```text
latent_dim in {4, 8, 16}
free_bits in {0.0, 0.5}
optimizer_batch_size in {4096, 16384}
learning_rate in {1e-4, 3e-4}
```

- Select by best held-out validation negative ELBO for the corresponding
  variant. For `VAE-noisy`, the validation objective uses noisy encoder input
  and clean decoder target. For `VAE-real`, the validation objective uses only
  noisy observations in the ELBO; clean channels may be reported as diagnostic
  denoising metrics but must not decide the checkpoint.
- Each final per-SNR model uses an independent validation holdout and stores
  best epoch, training loss, validation loss, and the selected checkpoint.
- Smoke training may use tiny values only to test execution and cache wiring.
  Smoke metrics must not be reported as evidence of method quality.

Detector-facing confidence validation:

- Primary `err_var` policy: transform the VAE/LMMSE posterior diagonal to
  antenna-domain marginal variances by the unitary-DFT rule above.
- Tune only the scalar `err_var_scale` with a bounded validation sweep:

```text
err_var_scale in {0.03, 0.1, 0.3, 1.0, 3.0}
```

- Select by average validation BLER over a small independent validation Monte
  Carlo run using the trained weights and the configured SNR grid; use BER as
  the tie-breaker and then choose the value closest to `1.0`.
- This validation run must be separate from the canonical benchmark result
  files and documented in `implementation_report.md` and the README.

Required constructor/training arguments:

- `variant`
- `latent_dim`
- `train_batches`, `train_batch_size`
- `validation_batches`, `validation_batch_size`
- `epochs`, `optimizer_batch_size`, `learning_rate`
- `free_bits`
- `snr_specific_weights`, `training_snr_dbs`
- `err_var_scale`
- `seed`, `precision`, `device`

Interface adaptations and risks:

- The paper assumes a ULA and DFT diagonalization of receive covariance. The benchmark has a 4x4 planar array. The implementation will use a unitary DFT over the flattened receive-antenna index, which preserves the VAE/LMMSE structure but is not the exact ULA/circulant assumption.
- The paper does not provide `latent_dim` or `free_bits`. These must be exposed and selected through the bounded validation run above.
- The paper does not define detector-facing error variances. The implementation will use the antenna-domain marginal posterior variance implied by the VAE/LMMSE update, scaled by a validated `err_var_scale`.

## Method: VAEChannelEstimator (real)

Status: `implement`

Paper location:

- Section III-C, variant 3
- Equations (1), (5)-(7), (10), (12), and (13)

Paper method:

Both encoder input and decoder likelihood target are the noisy transformed observation `x_tilde = F y`. In the likelihood, the covariance is `diag(c(z)) + sigma2 I`, so that the decoder learns the channel covariance `c(z)` rather than the observation covariance. At estimation time the same LMMSE expression as above is used with decoder outputs `m(z)` and `c(z)`.

Target abstraction:

Use the same `VAEChannelEstimator` class with `variant="real"`. The leaderboard/result name should be `VAEChannelEstimator (real)`.

Implementation differences from `variant="noisy"`:

```text
training target = F y
reconstruction variance in Equation (6) = decoder_channel_variance + noise_variance
runtime estimator = unchanged LMMSE update with decoder_channel_variance
```

Training asset:

```text
aux/<config-slug>/baur2022_vae/real/<cache-key>/vae_params.npz
aux/<config-slug>/baur2022_vae/real/<cache-key>/manifest.json
```

The file format, metadata requirements, checkpoint selection rule, and
validation protocol are the same as for `variant="noisy"`, except that
metadata must record `decoder_target="noisy"` and
`likelihood_variance="decoder_channel_variance_plus_observation_noise"`.

Interface adaptations and risks:

- This is the most practical paper variant because it trains on noisy observations only, but in simulation we still generate clean channels internally to synthesize observations. The training loss must not use the clean channels.
- At low SNR the paper itself expects degradation because `z = mu(x)` is a one-sample approximation of Equation (11).

## Method: VAE-genie

Status: `not implemented`

Paper location:

- Section III-C, variant 1

Reason:

The method requires the true channel `h` at estimation time to compute the encoder input `x = F h`. The benchmark estimator API receives only `(y, no)` and must estimate `h` from observations. Implementing VAE-genie would require changing the evaluation system into an oracle benchmark that passes `h_true` into the estimator. That would not be a deployable channel estimator under the current contract.

What would be required:

- A separate oracle evaluation path, analogous in spirit to the `perfect` baseline, where the system explicitly exposes `h_true` to the estimator.
- A report labeling the result as an oracle bound, not a comparable deployable estimator.

## Related Estimators And Baselines From Section III-E

Status: `not implemented` in this paper folder

The paper compares the VAE estimators to several related estimators:

- `LS`: already covered by the repository's built-in LS baselines.
- `sample-cov`: Equation (14) with a sample covariance matrix. This is a
  covariance-estimation baseline rather than a Baur VAE method. It would
  require a separate covariance asset and a separate reviewed scope.
- `genie-cov`: Equation (14) with oracle conditional covariance `C_delta`.
  The current benchmark does not expose `delta` or the conditional covariance
  of each channel realization to the estimator. It is an oracle baseline, not
  a deployable paper-local VAE method.
- `GMM circ`: prior work from Koller et al. and already represented by
  `papers/koller2022`.

These baselines should not be reimplemented inside `papers/baur2022` in the
first pass. The Baur folder should focus on the VAE estimators from
Section III-C.

## Tests

Planned focused tests under `tests/`:

- A DFT round-trip test for the estimator's fixed transform buffer.
- A limiting-case test for `_estimate_group`: if decoder variance is very large, the LMMSE update approaches the LS input; if observation noise tends to zero, the output approaches the LS input.
- A shape/dtype test for `VAEChannelEstimator.estimate_at_pilot_locations()` on a tiny synthetic resource grid or smoke resource grid.

Smoke testing after implementation:

```bash
python -m sionna_bench.mimo_ofdm_chest.papers.baur2022.run_benchmark --smoke --variant noisy
python -m sionna_bench.mimo_ofdm_chest.papers.baur2022.run_benchmark --smoke --variant real
```

Canonical benchmark commands, after validation/tuning:

```bash
python -m sionna_bench.mimo_ofdm_chest.papers.baur2022.run_benchmark --variant noisy ...
python -m sionna_bench.mimo_ofdm_chest.papers.baur2022.run_benchmark --variant real ...
```

The final command must include the selected training and confidence-scale parameters.
