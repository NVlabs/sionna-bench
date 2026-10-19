# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Inference tests for the Koller 2022 GMM estimators."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from sionna.phy.ofdm import NearestNeighborInterpolator, PilotPattern

from sionna.bench.mimo_ofdm_chest.benchmarking import build_resource_grid
from sionna.bench.mimo_ofdm_chest.papers.koller2022 import asset as koller_asset
from sionna.bench.mimo_ofdm_chest.papers.koller2022 import estimator as koller_estimator
from sionna.bench.mimo_ofdm_chest.papers.koller2022.asset import (
    METHOD_SPATIAL,
    METHOD_SPECTRAL,
    fit_model,
)
from sionna.bench.mimo_ofdm_chest.papers.koller2022.estimator import (
    SpatialGMMChannelEstimator,
    SpectralGMMChannelEstimator,
)
from test.mimo_ofdm_chest.system_variants import (
    assert_compiled_estimator_matches_eager,
    build_test_system,
    run_nmse_probe,
    system_variant_config,
)


def _gmm_parameters(
    real_vector_length: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return a deterministic two-component full-covariance GMM."""
    weights = torch.tensor([0.35, 0.65], dtype=torch.float64)
    coordinate = torch.linspace(-0.4, 0.6, real_vector_length, dtype=torch.float64)
    means = torch.stack([0.2 * coordinate, -0.3 * coordinate + 0.1])
    diagonal = torch.linspace(0.6, 1.4, real_vector_length, dtype=torch.float64)
    direction = torch.linspace(-0.2, 0.3, real_vector_length, dtype=torch.float64)
    covariance_0 = torch.diag(diagonal) + 0.4 * torch.outer(direction, direction)
    covariance_1 = (
        torch.diag(diagonal.flip(0))
        + 0.25 * torch.outer(direction.flip(0), direction.flip(0))
    )
    return weights, means, torch.stack([covariance_0, covariance_1])


def test_spectral_interpolator_uses_estimator_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The derived pilot pattern must not inherit Sionna's global device."""
    cfg = system_variant_config("alternate", precision="double", device="cpu")
    resource_grid = build_resource_grid(cfg)
    real_vector_length = 2 * int(resource_grid.num_effective_subcarriers)
    weights, means, covariances = _gmm_parameters(real_vector_length)

    pilot_pattern_devices = []

    def tracked_pilot_pattern(*args, **kwargs):
        pilot_pattern_devices.append(kwargs.get("device"))
        return PilotPattern(*args, **kwargs)

    monkeypatch.setattr(koller_estimator, "PilotPattern", tracked_pilot_pattern)
    estimator = SpectralGMMChannelEstimator(
        resource_grid,
        weights,
        means,
        covariances,
        precision=cfg.precision,
        device=cfg.device,
    )

    assert pilot_pattern_devices == [estimator.device]
    assert estimator._time_interpolator.device == estimator.device


def _conditional_mixture_reference(
    weights: torch.Tensor,
    means: torch.Tensor,
    covariances: torch.Tensor,
    observations: torch.Tensor,
    observed_indices: torch.Tensor,
    noise_variance: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Evaluate Gaussian likelihoods and conditional moments directly."""
    component_log_likelihoods = []
    component_posterior_means = []
    component_posterior_diagonals = []
    noise_covariance = torch.diag(noise_variance)

    for component_index in range(int(weights.shape[0])):
        mean = means[component_index]
        covariance = covariances[component_index]
        observed_mean = mean.index_select(0, observed_indices)
        observed_covariance = covariance.index_select(
            0,
            observed_indices,
        ).index_select(1, observed_indices)
        full_observed_covariance = covariance.index_select(1, observed_indices)
        observed_full_covariance = covariance.index_select(0, observed_indices)
        noisy_observed_covariance = observed_covariance + noise_covariance

        distribution = torch.distributions.MultivariateNormal(
            observed_mean,
            covariance_matrix=noisy_observed_covariance,
        )
        component_log_likelihoods.append(distribution.log_prob(observations))
        gain = torch.linalg.solve(
            noisy_observed_covariance,
            full_observed_covariance.transpose(0, 1),
        ).transpose(0, 1)
        residual = observations - observed_mean
        component_posterior_means.append(mean + residual @ gain.transpose(0, 1))
        posterior_covariance = covariance - gain @ observed_full_covariance
        component_posterior_diagonals.append(
            posterior_covariance.diagonal().expand(observations.shape[0], -1)
        )

    log_likelihoods = torch.stack(component_log_likelihoods, dim=-1)
    log_joint = log_likelihoods + torch.log(weights)
    responsibilities = torch.softmax(log_joint, dim=-1)
    posterior_means = torch.stack(component_posterior_means, dim=1)
    posterior_diagonals = torch.stack(component_posterior_diagonals, dim=1)
    mixture_mean = (responsibilities.unsqueeze(-1) * posterior_means).sum(dim=1)
    mixture_variance = (
        responsibilities.unsqueeze(-1)
        * (
            posterior_diagonals
            + (posterior_means - mixture_mean.unsqueeze(1)).square()
        )
    ).sum(dim=1)
    return log_likelihoods, responsibilities, mixture_mean, mixture_variance


def test_spatial_gmm_matches_conditional_mixture_equations() -> None:
    """Stacking, likelihoods, responsibilities, and posterior moments agree."""
    cfg = system_variant_config("baseline", precision="double")
    resource_grid = build_resource_grid(cfg)
    weights, means, covariances = _gmm_parameters(2 * cfg.num_rx_ant)
    estimator = SpatialGMMChannelEstimator(
        resource_grid,
        weights,
        means,
        covariances,
        precision=cfg.precision,
        device=cfg.device,
    )
    values = torch.tensor(
        [[0.4 + 0.2j, -0.3 + 0.5j], [-0.2 - 0.1j, 0.6 + 0.3j]],
        dtype=torch.complex128,
    )
    error_variance = torch.tensor([0.2, 0.7], dtype=torch.float64)
    observations = torch.cat([values.real, values.imag], dim=-1)
    torch.testing.assert_close(
        observations[0],
        torch.tensor([0.4, -0.3, 0.2, 0.5], dtype=torch.float64),
    )
    observed_indices = torch.arange(2 * cfg.num_rx_ant)
    real_noise_variance = 0.5 * torch.cat(
        [error_variance, error_variance]
    )
    (
        expected_log_likelihoods,
        responsibilities,
        expected_mean,
        expected_variance,
    ) = _conditional_mixture_reference(
        weights,
        means,
        covariances,
        observations,
        observed_indices,
        real_noise_variance,
    )

    means_expanded, cholesky, _, _, log_norms = (
        estimator._compute_gmm_term_quantities(
            error_variance,
            num_observations=1,
            num_complex_values=cfg.num_rx_ant,
        )
    )
    residual = observations.unsqueeze(0).unsqueeze(2) - means_expanded.unsqueeze(1)
    solved = torch.cholesky_solve(
        residual.permute(0, 2, 3, 1),
        cholesky,
    ).permute(0, 3, 1, 2)
    actual_log_joint = (
        log_norms.unsqueeze(1) - 0.5 * (residual * solved).sum(dim=-1)
    ).squeeze(0)
    actual_log_likelihoods = actual_log_joint - torch.log(weights)
    torch.testing.assert_close(actual_log_likelihoods, expected_log_likelihoods)
    actual_responsibilities = torch.softmax(actual_log_joint, dim=-1)
    torch.testing.assert_close(actual_responsibilities, responsibilities)
    torch.testing.assert_close(
        actual_responsibilities.sum(dim=-1),
        torch.ones(values.shape[0], dtype=torch.float64),
    )

    actual_mean, actual_variance = estimator._estimate_groups(
        values.unsqueeze(0),
        error_variance,
    )
    expected_complex_mean = torch.complex(
        expected_mean[:, :cfg.num_rx_ant],
        expected_mean[:, cfg.num_rx_ant:],
    )
    expected_complex_variance = (
        expected_variance[:, :cfg.num_rx_ant]
        + expected_variance[:, cfg.num_rx_ant:]
    )
    torch.testing.assert_close(actual_mean.squeeze(0), expected_complex_mean)
    torch.testing.assert_close(actual_variance.squeeze(0), expected_complex_variance)
    assert bool(torch.all(actual_variance >= 0.0))


def test_spatial_gmm_forwards_custom_interpolator() -> None:
    """A custom Sionna interpolator overrides the spatial GMM default."""
    cfg = system_variant_config("baseline")
    resource_grid = build_resource_grid(cfg)
    weights, means, covariances = _gmm_parameters(2 * cfg.num_rx_ant)
    interpolator = NearestNeighborInterpolator(resource_grid.pilot_pattern)

    estimator = SpatialGMMChannelEstimator(
        resource_grid,
        weights,
        means,
        covariances,
        interpolation_type="lin",
        interpolator=interpolator,
        precision=cfg.precision,
        device=cfg.device,
    )

    assert estimator._interpol is interpolator


@pytest.mark.parametrize(
    ("batch_size", "num_receivers", "num_groups", "num_antennas"),
    [
        (1, 1, 3, 1),
        (2, 1, 5, 2),
        (2, 3, 7, 4),
    ],
)
def test_spatial_group_vectorization_matches_group_loop(
    batch_size: int,
    num_receivers: int,
    num_groups: int,
    num_antennas: int,
) -> None:
    """Joint spatial-group processing equals independent group processing."""
    cfg = system_variant_config(
        "baseline",
        precision="double",
        bs_num_rows=1,
        bs_num_cols=num_antennas,
    )
    resource_grid = build_resource_grid(cfg)
    weights, means, covariances = _gmm_parameters(2 * num_antennas)
    estimator = SpatialGMMChannelEstimator(
        resource_grid,
        weights,
        means,
        covariances,
        precision=cfg.precision,
        device=cfg.device,
    )

    num_values = batch_size * num_receivers * num_groups * num_antennas
    values = torch.complex(
        torch.linspace(-0.7, 0.8, num_values, dtype=torch.float64),
        torch.linspace(0.5, -0.4, num_values, dtype=torch.float64),
    ).reshape(batch_size, num_receivers, num_groups, num_antennas)
    error_variance = torch.linspace(
        0.05,
        0.4,
        batch_size * num_receivers * num_antennas,
        dtype=torch.float64,
    ).reshape(batch_size, num_receivers, num_antennas)

    vectorized_estimates, vectorized_variances = estimator._estimate_groups(
        values,
        error_variance,
    )
    group_estimates = []
    group_variances = []
    for group_index in range(num_groups):
        estimates, variances = estimator._estimate_groups(
            values[..., group_index : group_index + 1, :],
            error_variance,
        )
        group_estimates.append(estimates)
        group_variances.append(variances)

    torch.testing.assert_close(
        vectorized_estimates,
        torch.cat(group_estimates, dim=-2),
    )
    torch.testing.assert_close(
        vectorized_variances,
        torch.cat(group_variances, dim=-2),
    )


@pytest.mark.parametrize("group_index", [0, 2, 4])
def test_spectral_gmm_matches_selected_subcarrier_posterior(
    group_index: int,
) -> None:
    """All noncanonical comb offsets select the expected posterior coordinates."""
    cfg = system_variant_config("alternate", precision="double")
    resource_grid = build_resource_grid(cfg)
    num_subcarriers = int(resource_grid.num_effective_subcarriers)
    weights, means, covariances = _gmm_parameters(2 * num_subcarriers)
    estimator = SpectralGMMChannelEstimator(
        resource_grid,
        weights,
        means,
        covariances,
        precision=cfg.precision,
        device=cfg.device,
    )
    observed_subcarriers = estimator._group_subcarrier_indices[group_index]
    num_observed = int(observed_subcarriers.numel())
    values = torch.complex(
        torch.linspace(-0.5, 0.4, num_observed, dtype=torch.float64),
        torch.linspace(0.3, -0.2, num_observed, dtype=torch.float64),
    )
    error_variance = torch.linspace(0.1, 0.4, num_observed, dtype=torch.float64)
    observations = torch.cat([values.real, values.imag], dim=-1).unsqueeze(0)
    observed_indices = torch.cat(
        [observed_subcarriers, observed_subcarriers + num_subcarriers]
    )
    real_noise_variance = 0.5 * torch.cat(
        [error_variance, error_variance]
    )
    _, responsibilities, expected_mean, expected_variance = (
        _conditional_mixture_reference(
            weights,
            means,
            covariances,
            observations,
            observed_indices,
            real_noise_variance,
        )
    )

    actual_mean, actual_variance = estimator._estimate_full_spectral_group(
        values,
        error_variance,
        observed_subcarriers,
    )
    expected_complex_mean = torch.complex(
        expected_mean[:, :num_subcarriers],
        expected_mean[:, num_subcarriers:],
    ).squeeze(0)
    expected_complex_variance = (
        expected_variance[:, :num_subcarriers]
        + expected_variance[:, num_subcarriers:]
    ).squeeze(0)
    torch.testing.assert_close(
        responsibilities.sum(dim=-1),
        torch.ones(1, dtype=torch.float64),
    )
    torch.testing.assert_close(actual_mean, expected_complex_mean)
    torch.testing.assert_close(actual_variance, expected_complex_variance)
    assert bool(torch.all(actual_variance >= 0.0))


def test_spatial_gmm_rejects_unequal_pilot_power() -> None:
    """Spatial joint processing rejects group-dependent LS noise power."""
    cfg = system_variant_config("baseline")
    resource_grid = build_resource_grid(cfg)
    changed_pilots = resource_grid.pilot_pattern.pilots.clone()
    changed_pilots[..., 0] *= 2.0
    resource_grid.pilot_pattern = PilotPattern(
        resource_grid.pilot_pattern.mask,
        changed_pilots,
        precision=cfg.precision,
        device=cfg.device,
    )
    weights, means, covariances = _gmm_parameters(2 * cfg.num_rx_ant)

    with pytest.raises(AssertionError, match="equal-power pilots"):
        SpatialGMMChannelEstimator(
            resource_grid,
            weights,
            means,
            covariances,
            precision=cfg.precision,
            device=cfg.device,
        )


def test_spectral_gmm_rejects_unequal_pilot_symbol_counts() -> None:
    """Spectral output assembly requires equal pilot-symbol counts per stream."""
    cfg = system_variant_config("alternate")
    resource_grid = build_resource_grid(cfg)
    layout = koller_asset._PilotGroupLayout(
        resource_grid.pilot_pattern,
        METHOD_SPECTRAL,
    )
    tx_indices = layout._pilot_group_constant_coord("tx")
    symbol_indices = layout._pilot_group_constant_coord("symbol")
    target_group = torch.nonzero(
        (tx_indices == 0) & (symbol_indices == cfg.pilot_ofdm_symbol_indices[-1]),
        as_tuple=False,
    )[0, 0]
    target_pilot_indices = layout._pilot_group_vector_indices[target_group]
    changed_pilots = resource_grid.pilot_pattern.pilots.clone()
    changed_pilots[0, 0, target_pilot_indices] = 0.0
    resource_grid.pilot_pattern = PilotPattern(
        resource_grid.pilot_pattern.mask,
        changed_pilots,
        precision=cfg.precision,
        device=cfg.device,
    )
    num_subcarriers = int(resource_grid.num_effective_subcarriers)
    weights, means, covariances = _gmm_parameters(2 * num_subcarriers)

    with pytest.raises(AssertionError, match="same number of pilot OFDM symbols"):
        SpectralGMMChannelEstimator(
            resource_grid,
            weights,
            means,
            covariances,
            precision=cfg.precision,
            device=cfg.device,
        )


@pytest.mark.compile
@pytest.mark.parametrize(
    ("estimator_class", "uses_spectral_vectors"),
    [
        (SpatialGMMChannelEstimator, False),
        (SpectralGMMChannelEstimator, True),
    ],
)
def test_gmm_estimators_compile_as_full_graph(
    estimator_class,
    uses_spectral_vectors: bool,
) -> None:
    """Both complete online GMM paths compile without graph breaks."""
    cfg = system_variant_config("alternate")
    resource_grid = build_resource_grid(cfg)
    vector_length = (
        int(resource_grid.num_effective_subcarriers)
        if uses_spectral_vectors
        else cfg.num_rx_ant
    )
    real_vector_length = 2 * vector_length
    weights, means, covariances = _gmm_parameters(real_vector_length)
    estimator = estimator_class(
        resource_grid,
        weights,
        means,
        covariances,
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
    noise_variance = torch.full(
        (1, 1, cfg.num_rx_ant),
        0.1,
        dtype=torch.float32,
    )

    assert_compiled_estimator_matches_eager(
        estimator,
        received_grid,
        noise_variance,
        fullgraph=True,
    )


@pytest.mark.system_variants
@pytest.mark.training
@pytest.mark.parametrize("system_variant", ["baseline", "alternate", "wide"])
@pytest.mark.parametrize(
    ("method", "estimator_class"),
    [
        (METHOD_SPATIAL, SpatialGMMChannelEstimator),
        (METHOD_SPECTRAL, SpectralGMMChannelEstimator),
    ],
)
def test_gmm_estimators_support_changed_user_and_antenna_counts(
    tmp_path: Path,
    system_variant: str,
    method: str,
    estimator_class,
) -> None:
    """Both methods run for 1/3/4 UEs and 2/4/12 BS antennas."""
    cfg = system_variant_config(system_variant)

    def estimator_factory(resource_grid, current_cfg, asset_root):
        parameters, _ = fit_model(
            resource_grid,
            current_cfg,
            method=method,
            num_components=1,
            train_batches=2,
            train_batch_size=4,
            em_iters=2,
            covariance_reg=1e-3,
            seed=current_cfg.seed,
            verbose=False,
            asset_root=asset_root,
        )
        return estimator_class(
            resource_grid,
            *parameters,
            precision=current_cfg.precision,
            device=current_cfg.device,
        )

    system = build_test_system(cfg, estimator_factory, tmp_path)
    result = run_nmse_probe(system, snr_db=5.0)

    assert 0.0 <= result.nmse < result.zero_estimate_nmse
