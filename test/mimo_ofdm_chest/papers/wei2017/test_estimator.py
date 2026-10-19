# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from dataclasses import replace
from sionna.phy import config
from sionna.phy.ofdm import NearestNeighborInterpolator, PilotPattern

from sionna.bench.mimo_ofdm_chest.configs import umi_3gpp_config
from sionna.bench.mimo_ofdm_chest.benchmarking.communication_system import OFDMChannelEstimationSystem, build_resource_grid
from sionna.bench.mimo_ofdm_chest.papers.wei2017.estimator import (
    SBLChannelEstimator,
)
from test.mimo_ofdm_chest.system_variants import (
    assert_compiled_estimator_matches_eager,
    run_nmse_probe,
    system_variant_config,
)


_COMB_CASES = (
    pytest.param(4, 1, 1, id="comb-48-ant-1"),
    pytest.param(4, 2, 3, id="comb-24-ant-3"),
    pytest.param(5, 3, 5, id="comb-20-ant-5"),
)


@pytest.fixture(autouse=True)
def reset_sionna_config():
    old_device = config.device
    old_precision = config.precision
    config.device = "cpu"
    config.precision = "double"
    yield
    config.device = old_device
    config.precision = old_precision


def _build_estimator(
    num_prb: int,
    num_ue: int,
    num_rx_ant: int,
    *,
    num_taps: int | None = 8,
    num_iter: int = 20,
    alpha_init: float = 0.3,
) -> SBLChannelEstimator:
    """Build a small valid comb estimator for one analytical test case."""
    cfg = replace(
        umi_3gpp_config,
        num_prb=num_prb,
        num_ofdm_symbols=6,
        cyclic_prefix_length=6,
        pilot_ofdm_symbol_indices=(1,),
        num_ue=num_ue,
        bs_num_rows=1,
        bs_num_cols=num_rx_ant,
        precision="double",
        device="cpu",
    )
    resource_grid = build_resource_grid(cfg)
    return SBLChannelEstimator(
        resource_grid,
        num_taps=num_taps,
        num_iter=num_iter,
        alpha_init=alpha_init,
        precision="double",
        device="cpu",
    )


def test_default_num_taps_uses_full_pilot_group():
    """Omitting ``num_taps`` retains every comb-delay coefficient."""
    estimator = _build_estimator(4, 2, 3, num_taps=None)

    assert estimator._num_taps == estimator._pilot_group_len


def test_custom_interpolator_is_forwarded_to_sionna() -> None:
    """A custom Sionna interpolator overrides the SBL interpolation default."""
    cfg = system_variant_config("baseline")
    resource_grid = build_resource_grid(cfg)
    interpolator = NearestNeighborInterpolator(resource_grid.pilot_pattern)

    estimator = SBLChannelEstimator(
        resource_grid,
        interpolation_type="lin",
        interpolator=interpolator,
        num_taps=8,
        num_iter=1,
        precision=cfg.precision,
        device=cfg.device,
    )

    assert estimator._interpol is interpolator


@pytest.mark.system_variants
@pytest.mark.parametrize("system_variant", ["baseline", "alternate", "wide"])
def test_full_estimator_supports_changed_ue_and_ofdm_geometry(
    system_variant: str,
) -> None:
    """Complete inference supports changed UE and OFDM pilot-symbol layouts."""
    cfg = system_variant_config(system_variant)
    resource_grid = build_resource_grid(cfg)
    estimator = SBLChannelEstimator(
        resource_grid,
        num_taps=8,
        num_iter=5,
        precision=cfg.precision,
        device=cfg.device,
    )
    system = OFDMChannelEstimationSystem(cfg, estimator, resource_grid)

    result = run_nmse_probe(system, snr_db=5.0)

    assert result.nmse < result.zero_estimate_nmse


@pytest.mark.compile
@pytest.mark.parametrize("system_variant", ["baseline", "alternate"])
def test_complete_estimator_compiles_as_full_graph(system_variant: str) -> None:
    """Complete SBL estimation and interpolation compile as one graph."""
    cfg = system_variant_config(system_variant)
    resource_grid = build_resource_grid(cfg)
    estimator = SBLChannelEstimator(
        resource_grid,
        num_taps=8,
        num_iter=2,
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


def test_posterior_variance_scale_changes_only_error_variance() -> None:
    """Detector calibration leaves channel estimates unchanged."""
    cfg = system_variant_config("baseline")
    resource_grid = build_resource_grid(cfg)
    common_kwargs = {
        "num_taps": 8,
        "num_iter": 2,
        "precision": cfg.precision,
        "device": cfg.device,
    }
    prior_scale = 0.1
    prior_estimator = SBLChannelEstimator(
        resource_grid,
        posterior_variance_scale=prior_scale,
        **common_kwargs,
    )
    calibrated_scale = 0.25
    calibrated_estimator = SBLChannelEstimator(
        resource_grid,
        posterior_variance_scale=calibrated_scale,
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
    noise_variance = torch.tensor(0.1)

    prior_channel, prior_variance = prior_estimator(
        received_grid,
        noise_variance,
    )
    calibrated_channel, calibrated_variance = calibrated_estimator(
        received_grid,
        noise_variance,
    )

    torch.testing.assert_close(
        calibrated_channel,
        prior_channel,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        calibrated_variance,
        prior_variance * (calibrated_scale / prior_scale),
    )


def test_nonuniform_comb_is_rejected() -> None:
    """Moving one pilot off the comb lattice fails with a specific message."""
    cfg = system_variant_config("baseline", num_ue=2)
    resource_grid = build_resource_grid(cfg)
    pilots = resource_grid.pilot_pattern.pilots.clone()
    active_indices = torch.nonzero(
        pilots[0, 0].abs() > 0,
        as_tuple=False,
    ).flatten()
    moved_index = int(active_indices[2])
    pilots[0, 0, moved_index + 1] = pilots[0, 0, moved_index]
    pilots[0, 0, moved_index] = 0
    resource_grid.pilot_pattern = PilotPattern(
        resource_grid.pilot_pattern.mask,
        pilots,
        precision=cfg.precision,
        device=cfg.device,
    )

    with pytest.raises(AssertionError, match="requires uniform pilot spacing"):
        SBLChannelEstimator(
            resource_grid,
            num_iter=1,
            precision=cfg.precision,
            device=cfg.device,
        )


def test_comb_that_does_not_span_fft_is_rejected() -> None:
    """A uniform but incomplete comb fails the FFT-coverage requirement."""
    cfg = system_variant_config("baseline")
    resource_grid = build_resource_grid(cfg)
    pilots = resource_grid.pilot_pattern.pilots.clone()
    retained_pilots = pilots[0, 0, :46:2].clone()
    pilots[0, 0] = 0
    pilots[0, 0, :46:2] = retained_pilots
    resource_grid.pilot_pattern = PilotPattern(
        resource_grid.pilot_pattern.mask,
        pilots,
        precision=cfg.precision,
        device=cfg.device,
    )

    with pytest.raises(
        AssertionError,
        match=r"requires spacing\*group_len == fft_size",
    ):
        SBLChannelEstimator(
            resource_grid,
            num_iter=1,
            precision=cfg.precision,
            device=cfg.device,
        )


def _complex_randn(*shape: int) -> torch.Tensor:
    real = torch.randn(*shape, dtype=torch.float64)
    imag = torch.randn(*shape, dtype=torch.float64)
    return torch.complex(real, imag)


@pytest.mark.parametrize("num_prb,num_ue,num_rx_ant", _COMB_CASES)
def test_comb_transform_recovers_delay_coefficients(
    num_prb: int,
    num_ue: int,
    num_rx_ant: int,
):
    """The comb IFFT recovers every antenna's LS delay vector."""
    torch.manual_seed(0)
    estimator = _build_estimator(num_prb, num_ue, num_rx_ant)
    num_taps = estimator._num_taps
    group_len = estimator._pilot_group_len
    num_groups = estimator._pilot_group_num_groups
    delay_values = _complex_randn(1, 1, num_groups, num_rx_ant, num_taps)
    padded_delay_values = torch.zeros(
        1,
        1,
        num_groups,
        num_rx_ant,
        group_len,
        dtype=torch.complex128,
    )
    padded_delay_values[..., :num_taps] = delay_values

    h_ls = torch.fft.fft(padded_delay_values, dim=-1).movedim(-3, -2)
    noise_var = torch.zeros(1, 1, num_rx_ant, dtype=torch.float64)

    delay_observations, delay_noise_var = estimator._delay_observations(
        h_ls,
        noise_var,
    )

    torch.testing.assert_close(
        delay_observations,
        delay_values,
        atol=1e-12,
        rtol=1e-12,
    )
    torch.testing.assert_close(
        delay_noise_var,
        torch.zeros_like(delay_noise_var),
        atol=0.0,
        rtol=0.0,
    )


@pytest.mark.parametrize("num_prb,num_ue,num_rx_ant", _COMB_CASES)
def test_delay_estimator_shapes_and_variances(
    num_prb: int,
    num_ue: int,
    num_rx_ant: int,
):
    """Posterior moments preserve shape and remain finite and non-negative."""
    torch.manual_seed(1)
    estimator = _build_estimator(num_prb, num_ue, num_rx_ant)
    observations = _complex_randn(
        2,
        1,
        estimator._pilot_group_num_groups,
        num_rx_ant,
        estimator._num_taps,
    )
    noise_var = torch.full(
        (2, 1, estimator._pilot_group_num_groups, num_rx_ant),
        0.05,
        dtype=torch.float64,
    )

    estimates, variances = estimator._estimate_delay_observations(
        observations,
        noise_var,
    )

    assert estimates.shape == observations.shape
    assert variances.shape == observations.shape
    assert estimates.dtype == observations.dtype
    assert variances.dtype == torch.float64
    assert bool(torch.all(torch.isfinite(estimates.real)))
    assert bool(torch.all(torch.isfinite(estimates.imag)))
    assert bool(torch.all(torch.isfinite(variances)))
    assert float(variances.min()) >= 0.0


def test_delay_estimator_matches_one_em_update_and_final_posterior() -> None:
    """One iteration implements Equations (14), (21), and final Equation (16)."""
    alpha_init = 0.4
    estimator = _build_estimator(
        4,
        2,
        2,
        num_taps=3,
        num_iter=1,
        alpha_init=alpha_init,
    )
    observations = torch.tensor(
        [[[[[0.5 + 0.2j, -0.1 + 0.4j, 0.3 - 0.2j],
             [-0.2 + 0.1j, 0.6 - 0.3j, -0.4 + 0.5j]]]]],
        dtype=torch.complex128,
    )
    noise_variance = torch.tensor(
        [[[[0.2, 0.5]]]],
        dtype=torch.float64,
    )
    noise_grid = noise_variance.unsqueeze(-1)
    initial_variance = 1.0 / (1.0 / alpha_init + 1.0 / noise_grid)
    initial_mean = initial_variance * observations / noise_grid
    updated_alpha = (
        initial_mean.abs().square() + initial_variance
    ).mean(dim=-2)
    expected_variance = 1.0 / (
        1.0 / updated_alpha.unsqueeze(-2) + 1.0 / noise_grid
    )
    expected_mean = expected_variance * observations / noise_grid

    actual_mean, actual_variance = estimator._estimate_delay_observations(
        observations,
        noise_variance,
    )

    torch.testing.assert_close(actual_mean, expected_mean)
    torch.testing.assert_close(actual_variance, expected_variance)


@pytest.mark.parametrize("num_prb,num_ue,num_rx_ant", _COMB_CASES)
def test_high_snr_pilot_reconstruction_limiting_case(
    num_prb: int,
    num_ue: int,
    num_rx_ant: int,
):
    """Vanishing noise reproduces delay-limited responses at pilot positions."""
    torch.manual_seed(2)
    estimator = _build_estimator(num_prb, num_ue, num_rx_ant)
    batch_size = 1
    num_rx = 1
    group_index = estimator._pilot_group_num_groups - 1
    num_taps = estimator._num_taps
    group_len = estimator._pilot_group_len

    delay_values = torch.zeros(
        batch_size,
        num_rx,
        num_rx_ant,
        num_taps,
        dtype=torch.complex128,
    )
    delay_values[..., 0] = 1.0 + 0.25j
    delay_values[..., 3] = -0.2 + 0.4j
    padded_delay_values = torch.zeros(
        batch_size,
        num_rx,
        num_rx_ant,
        group_len,
        dtype=torch.complex128,
    )
    padded_delay_values[..., :num_taps] = delay_values
    pilot_response = torch.fft.fft(padded_delay_values, n=group_len, dim=-1)

    grouped_y = torch.zeros(
        batch_size,
        num_rx,
        num_rx_ant,
        estimator._pilot_group_num_groups,
        group_len,
        dtype=torch.complex128,
    )
    group_pilot_values = estimator._gather_pilot_groups(
        estimator._pilot_pattern.pilots
    )[group_index].cpu()
    grouped_y[..., group_index, :] = (
        pilot_response * group_pilot_values
    )
    y_pilots = estimator._scatter_pilot_groups(
        torch.zeros(
            batch_size,
            num_rx,
            num_rx_ant,
            estimator._pilot_pattern.num_tx,
            estimator._pilot_pattern.num_streams_per_tx,
            estimator._pilot_pattern.num_pilot_symbols,
            dtype=torch.complex128,
        ),
        grouped_y,
    )

    h_hat, err_var = estimator.estimate_at_pilot_locations(
        y_pilots,
        torch.tensor(1e-10, dtype=torch.float64),
    )
    reconstructed_pilot_response = estimator._gather_pilot_groups(h_hat)[
        ..., group_index, :
    ]

    torch.testing.assert_close(
        reconstructed_pilot_response,
        pilot_response,
        atol=2e-5,
        rtol=2e-5,
    )
    assert float(err_var.min()) >= 0.0
