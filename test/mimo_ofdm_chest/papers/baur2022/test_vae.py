# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from sionna.bench.mimo_ofdm_chest.benchmarking import build_resource_grid
from sionna.bench.mimo_ofdm_chest.papers.baur2022.asset import (
    METHOD_NOISY,
    METHOD_REAL,
    _vae_loss,
)
from sionna.bench.mimo_ofdm_chest.papers.baur2022.estimator import (
    COVARIANCE_DIAGONAL,
    VariationalAutoEncoder,
    COVARIANCE_FULL,
    DECODER_CHANNELS,
    DECODER_KERNEL_SIZE,
    DECODER_PADDING,
    DECODER_STRIDE,
    ENCODER_CHANNELS,
    ENCODER_KERNEL_SIZE,
    ENCODER_PADDING,
    ENCODER_STRIDE,
    VAEChannelEstimator,
)
from test.mimo_ofdm_chest.system_variants import (
    assert_compiled_estimator_matches_eager,
    system_variant_config,
)


class _DftBuilder:
    dtype = torch.float32
    cdtype = torch.complex64
    device = "cpu"


class _FixedVAE:
    """Return fixed encoder and decoder values for ELBO checks."""

    def __init__(
        self,
        covariance_mode,
        latent_mean,
        latent_log_variance,
        decoder_mean,
        decoder_covariance,
    ):
        self.covariance_mode = covariance_mode
        self._outputs = (
            latent_mean,
            latent_log_variance,
            decoder_mean,
            decoder_covariance,
        )

    def __call__(self, x, *, sample=True):
        assert not sample
        return self._outputs


def _constant_vae_state(
    input_dim: int,
    latent_dim: int,
    value: float,
) -> dict[str, torch.Tensor]:
    """Return a valid full-covariance VAE state with a constant mean head."""
    model = VariationalAutoEncoder(
        input_dim,
        latent_dim,
        covariance_mode=COVARIANCE_FULL,
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.decoder_mean.bias.fill_(value)
    return model.state_dict()


@pytest.mark.compile
@pytest.mark.parametrize(
    ("num_states", "fullgraph"),
    [(1, True), (2, False)],
    ids=["mixed-snr-fullgraph", "per-snr-selector-boundary"],
)
def test_estimator_matches_compiled_inference(
    num_states: int,
    fullgraph: bool,
) -> None:
    """Full-covariance VAE inference agrees in eager and compiled execution."""
    cfg = system_variant_config("baseline")
    resource_grid = build_resource_grid(cfg)
    latent_dim = 2
    model_states = [
        _constant_vae_state(
            2 * cfg.num_rx_ant,
            latent_dim,
            float(index + 1),
        )
        for index in range(num_states)
    ]
    model_noise_variances = (
        None if num_states == 1 else torch.tensor([0.2, 0.1])
    )
    estimator = VAEChannelEstimator(
        resource_grid,
        model_states,
        cfg.num_rx_ant,
        model_noise_variances,
        latent_dim=latent_dim,
        covariance_mode=COVARIANCE_FULL,
        precision=cfg.precision,
        device=cfg.device,
    )
    received_grid = torch.randn(
        1,
        1,
        cfg.num_rx_ant,
        cfg.num_ofdm_symbols,
        cfg.fft_size,
        dtype=torch.complex64,
    )

    assert_compiled_estimator_matches_eager(
        estimator,
        received_grid,
        torch.tensor(0.1, dtype=torch.float32),
        fullgraph=fullgraph,
    )


def test_error_variance_scale_does_not_change_channel_estimate() -> None:
    """Detector confidence scaling leaves the VAE posterior mean unchanged."""
    cfg = system_variant_config("baseline")
    resource_grid = build_resource_grid(cfg)
    latent_dim = 2
    model_state = _constant_vae_state(
        2 * cfg.num_rx_ant,
        latent_dim,
        1.0,
    )
    common_kwargs = {
        "resource_grid": resource_grid,
        "model_states": [model_state],
        "num_rx_ant": cfg.num_rx_ant,
        "latent_dim": latent_dim,
        "covariance_mode": COVARIANCE_FULL,
        "precision": cfg.precision,
        "device": cfg.device,
    }
    unscaled_estimator = VAEChannelEstimator(
        err_var_scale=1.0,
        **common_kwargs,
    )
    calibrated_scale = 0.25
    calibrated_estimator = VAEChannelEstimator(
        err_var_scale=calibrated_scale,
        **common_kwargs,
    )
    received_grid = torch.randn(
        1,
        1,
        cfg.num_rx_ant,
        cfg.num_ofdm_symbols,
        cfg.fft_size,
        dtype=torch.complex64,
    )
    noise_variance = torch.tensor(0.1, dtype=torch.float32)

    with torch.no_grad():
        unscaled_channel, unscaled_variance = unscaled_estimator(
            received_grid,
            noise_variance,
        )
        calibrated_channel, calibrated_variance = calibrated_estimator(
            received_grid,
            noise_variance,
        )

    torch.testing.assert_close(calibrated_channel, unscaled_channel)
    torch.testing.assert_close(
        calibrated_variance,
        calibrated_scale * unscaled_variance,
    )


def test_baur_vae_encoder_and_decoder_parameterization():
    """Section III-D channel counts and convolution settings are explicit."""
    model = VariationalAutoEncoder(input_dim=32, latent_dim=8)

    encoder_convolutions = [block[0] for block in model.encoder]
    assert [layer.in_channels for layer in encoder_convolutions] == list(
        ENCODER_CHANNELS[:-1]
    )
    assert [layer.out_channels for layer in encoder_convolutions] == list(
        ENCODER_CHANNELS[1:]
    )
    for layer in encoder_convolutions:
        assert layer.kernel_size == (ENCODER_KERNEL_SIZE,)
        assert layer.stride == (ENCODER_STRIDE,)
        assert layer.padding == (ENCODER_PADDING,)
    assert all(
        isinstance(block[1], torch.nn.BatchNorm1d)
        and isinstance(block[2], torch.nn.ReLU)
        for block in model.encoder
    )

    decoder_convolutions = [block[0] for block in model.decoder_trunk]
    decoder_convolutions.append(model.decoder_mean)
    assert [layer.in_channels for layer in decoder_convolutions] == list(
        DECODER_CHANNELS[:-1]
    )
    assert [layer.out_channels for layer in decoder_convolutions] == list(
        DECODER_CHANNELS[1:]
    )
    for layer in decoder_convolutions:
        assert layer.kernel_size == (DECODER_KERNEL_SIZE,)
        assert layer.stride == (DECODER_STRIDE,)
        assert layer.padding == (DECODER_PADDING,)
    assert all(
        isinstance(block[1], torch.nn.BatchNorm1d)
        and isinstance(block[2], torch.nn.ReLU)
        for block in model.decoder_trunk
    )


@pytest.mark.parametrize("method", (METHOD_NOISY, METHOD_REAL))
def test_diagonal_vae_loss_matches_complex_elbo(method):
    """The diagonal loss matches Equations (6) and (7) for both methods."""
    latent_mean = torch.tensor([[0.2, -0.3]])
    latent_log_variance = torch.tensor([[0.1, -0.2]])
    decoder_mean = torch.tensor([[0.3, -0.4, 0.2, 0.1]])
    decoder_covariance = torch.tensor([[0.7, 1.2]])
    target = torch.tensor([[0.8, -0.1, -0.2, 0.5]])
    noise_variance = torch.tensor([[0.4]])
    model = _FixedVAE(
        COVARIANCE_DIAGONAL,
        latent_mean,
        latent_log_variance,
        decoder_mean,
        decoder_covariance,
    )

    loss = _vae_loss(
        model,
        torch.zeros_like(target),
        target,
        noise_variance,
        method=method,
        free_bits=0.0,
        sample=False,
    )

    target_complex = torch.complex(target[:, :2], target[:, 2:])
    mean_complex = torch.complex(decoder_mean[:, :2], decoder_mean[:, 2:])
    likelihood_variance = decoder_covariance
    if method == METHOD_REAL:
        likelihood_variance = likelihood_variance + noise_variance
    reconstruction = (
        torch.log(torch.tensor(torch.pi))
        + torch.log(likelihood_variance)
        + (target_complex - mean_complex).abs().square()
        / likelihood_variance
    ).sum(dim=-1)
    kl = 0.5 * (
        -latent_log_variance
        + latent_mean.square()
        + torch.exp(latent_log_variance)
        - 1.0
    ).sum(dim=-1)
    torch.testing.assert_close(loss, (reconstruction + kl).mean())


@pytest.mark.parametrize("method", (METHOD_NOISY, METHOD_REAL))
def test_full_covariance_vae_loss_matches_real_gaussian_elbo(method):
    """The full-covariance extension uses the exact real Gaussian NLL."""
    latent_mean = torch.tensor([[0.2, -0.3]])
    latent_log_variance = torch.tensor([[0.1, -0.2]])
    decoder_mean = torch.tensor([[0.3, -0.4, 0.2, 0.1]])
    covariance_factor = torch.tensor(
        [[
            [1.0, 0.0, 0.0, 0.0],
            [0.2, 0.8, 0.0, 0.0],
            [0.1, -0.1, 0.9, 0.0],
            [0.0, 0.2, 0.1, 0.7],
        ]]
    )
    decoder_covariance = (
        covariance_factor @ covariance_factor.transpose(-2, -1)
    )
    target = torch.tensor([[0.8, -0.1, -0.2, 0.5]])
    noise_variance = torch.tensor([[0.4]])
    model = _FixedVAE(
        COVARIANCE_FULL,
        latent_mean,
        latent_log_variance,
        decoder_mean,
        decoder_covariance,
    )

    loss = _vae_loss(
        model,
        torch.zeros_like(target),
        target,
        noise_variance,
        method=method,
        free_bits=0.0,
        sample=False,
    )

    likelihood_covariance = decoder_covariance
    if method == METHOD_REAL:
        likelihood_covariance = likelihood_covariance + 0.5 * (
            noise_variance.reshape(-1, 1, 1) * torch.eye(target.shape[-1])
        )
    residual = target - decoder_mean
    sign, log_det = torch.linalg.slogdet(likelihood_covariance)
    assert bool(torch.all(sign > 0.0))
    mahalanobis = (
        residual
        * torch.linalg.solve(likelihood_covariance, residual.unsqueeze(-1))
        .squeeze(-1)
    ).sum(dim=-1)
    reconstruction = 0.5 * (
        target.shape[-1] * torch.log(torch.tensor(2.0 * torch.pi))
        + log_det
        + mahalanobis
    )
    kl = 0.5 * (
        -latent_log_variance
        + latent_mean.square()
        + torch.exp(latent_log_variance)
        - 1.0
    ).sum(dim=-1)
    torch.testing.assert_close(loss, (reconstruction + kl).mean())


def test_vae_loss_applies_free_bits_per_latent_dimension():
    """Free bits clamp each Equation (7) latent contribution independently."""
    latent_mean = torch.zeros(1, 3)
    latent_log_variance = torch.zeros(1, 3)
    decoder_mean = torch.zeros(1, 4)
    decoder_covariance = torch.ones(1, 2)
    target = torch.zeros(1, 4)
    noise_variance = torch.ones(1, 1)
    model = _FixedVAE(
        COVARIANCE_DIAGONAL,
        latent_mean,
        latent_log_variance,
        decoder_mean,
        decoder_covariance,
    )

    loss_without_free_bits = _vae_loss(
        model,
        target,
        target,
        noise_variance,
        method=METHOD_NOISY,
        free_bits=0.0,
        sample=False,
    )
    loss_with_free_bits = _vae_loss(
        model,
        target,
        target,
        noise_variance,
        method=METHOD_NOISY,
        free_bits=0.25,
        sample=False,
    )
    torch.testing.assert_close(
        loss_with_free_bits - loss_without_free_bits,
        torch.tensor(3 * 0.25),
    )


def test_baur_vae_shapes_and_positive_covariance():
    model = VariationalAutoEncoder(input_dim=32, latent_dim=8, covariance_floor=1e-5)
    x = torch.randn(5, 32)

    latent_mean, latent_log_variance, decoder_mean, decoder_covariance = model(
        x,
        sample=False,
    )

    assert latent_mean.shape == (5, 8)
    assert latent_log_variance.shape == (5, 8)
    assert decoder_mean.shape == (5, 32)
    assert decoder_covariance.shape == (5, 16)
    assert bool(torch.all(decoder_covariance > 0.0))


def test_baur_vae_full_covariance_is_positive_definite():
    model = VariationalAutoEncoder(
        input_dim=32,
        latent_dim=8,
        covariance_floor=1e-5,
        covariance_mode=COVARIANCE_FULL,
    )
    x = torch.randn(5, 32)

    _, _, decoder_mean, decoder_covariance = model(x, sample=False)

    assert decoder_mean.shape == (5, 32)
    assert decoder_covariance.shape == (5, 32, 32)
    assert model.decoder_cholesky.out_features == 32 * 33 // 2
    torch.linalg.cholesky(decoder_covariance)


def test_full_covariance_output_matches_cholesky_parameterization():
    """The full covariance head returns its documented positive-definite form."""
    floor = 1e-4
    model = VariationalAutoEncoder(
        input_dim=4,
        latent_dim=2,
        covariance_floor=floor,
        covariance_mode=COVARIANCE_FULL,
    )
    params = torch.linspace(-0.5, 0.5, 10).reshape(1, -1)

    covariance = model._full_covariance_from_params(params)

    rows, cols = torch.tril_indices(4, 4)
    factor = torch.zeros(1, 4, 4)
    factor[:, rows, cols] = params
    diagonal = torch.arange(4)
    factor[:, diagonal, diagonal] = (
        torch.nn.functional.softplus(factor[:, diagonal, diagonal])
        + floor ** 0.5
    )
    expected = factor @ factor.transpose(-2, -1) + floor * torch.eye(4)
    torch.testing.assert_close(covariance, expected)


def test_receive_dft_round_trip():
    dft = VAEChannelEstimator._build_unitary_dft(_DftBuilder(), 16)
    values = torch.randn(7, 16, dtype=torch.complex64)

    spectral = values @ dft.transpose(0, 1)
    reconstructed = spectral @ dft.conj()

    torch.testing.assert_close(reconstructed, values, rtol=1e-5, atol=1e-5)


def test_real_stack_round_trip_preserves_complex_transform_order():
    """The VAE representation stores all real parts before imaginary parts."""
    values = torch.randn(7, 4, dtype=torch.complex64)

    stacked = VAEChannelEstimator._stack_complex(_DftBuilder(), values)
    reconstructed = VAEChannelEstimator._unstack_complex(
        _DftBuilder(),
        stacked,
    )

    torch.testing.assert_close(reconstructed, values)


def test_unitary_dft_posterior_marginal_variance_is_average():
    dft = VAEChannelEstimator._build_unitary_dft(_DftBuilder(), 4)
    spectral_variance = torch.tensor([1.0, 2.0, 4.0, 8.0])
    covariance = (
        dft.conj().transpose(0, 1)
        @ torch.diag(spectral_variance).to(torch.complex64)
        @ dft
    )
    antenna_marginals = torch.diagonal(covariance).real

    expected = spectral_variance.mean().expand(4)
    torch.testing.assert_close(antenna_marginals, expected, rtol=1e-6, atol=1e-6)
