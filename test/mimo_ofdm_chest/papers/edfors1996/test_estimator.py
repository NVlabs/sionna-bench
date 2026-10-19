# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Integration tests for the Edfors channel-estimation methods."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from sionna.phy.ofdm import (
    LMMSEChannelEstimator,
    LSChannelEstimator,
    NearestNeighborInterpolator,
    PilotPattern,
)

from sionna.bench.mimo_ofdm_chest.benchmarking import build_resource_grid
from sionna.bench.mimo_ofdm_chest.benchmarking.communication_system import OFDMChannelEstimationSystem
from sionna.bench.mimo_ofdm_chest.papers.edfors1996.asset import estimate_covariance
from sionna.bench.mimo_ofdm_chest.papers.edfors1996.estimator import (
    METHOD_LMMSE,
    METHOD_LS,
    METHOD_NAMES,
    TruncatedChannelEstimator,
)
from test.mimo_ofdm_chest.system_variants import (
    assert_compiled_estimator_matches_eager,
    run_nmse_probe,
    system_variant_config,
)


_NUM_TAPS = 4


@pytest.mark.parametrize("system_variant", ["baseline", "alternate", "wide"])
@pytest.mark.system_variants
def test_all_methods_support_changed_grid_and_receive_array_dimensions(
    tmp_path: Path,
    system_variant: str,
) -> None:
    """All methods satisfy the output contract and beat a zero estimate."""
    cfg = system_variant_config(system_variant)
    resource_grid = build_resource_grid(cfg)
    covariance, _ = estimate_covariance(
        resource_grid,
        cfg,
        cov_num_iters=1,
        cov_batch_size=1,
        seed=cfg.seed,
        asset_root=tmp_path,
    )

    for method in METHOD_NAMES:
        estimator = TruncatedChannelEstimator(
            resource_grid,
            covariance,
            method=method,
            num_taps=_NUM_TAPS,
            precision=cfg.precision,
            device=cfg.device,
        )
        system = OFDMChannelEstimationSystem(cfg, estimator, resource_grid)
        result = run_nmse_probe(system, snr_db=5.0)

        assert result.nmse < result.zero_estimate_nmse, (
            f"{method} produced an implausible NMSE for {system_variant}."
        )


def test_ragged_pilot_frequency_groups_are_rejected() -> None:
    """Unequal active pilot counts fail before transform setup begins."""
    cfg = system_variant_config("alternate")
    resource_grid = build_resource_grid(cfg)
    pilots = resource_grid.pilot_pattern.pilots.clone()
    active_pilot_index = torch.nonzero(
        pilots[0, 0].abs() > 0,
        as_tuple=False,
    )[0, 0]
    pilots[0, 0, active_pilot_index] = 0
    resource_grid.pilot_pattern = PilotPattern(
        resource_grid.pilot_pattern.mask,
        pilots,
        precision=cfg.precision,
        device=cfg.device,
    )

    with pytest.raises(
        ValueError,
        match="_PilotGroupMixin currently requires equal-length groups",
    ):
        TruncatedChannelEstimator(
            resource_grid,
            torch.eye(cfg.fft_size, dtype=torch.complex64),
            method=METHOD_LS,
            num_taps=_NUM_TAPS,
            precision=cfg.precision,
            device=cfg.device,
        )


@pytest.mark.parametrize("interpolation_type", ["nn", "lin", "lin_time_avg"])
def test_all_tap_ls_is_an_exact_transform_round_trip(
    interpolation_type: str,
) -> None:
    """With every delay coefficient retained, Edfors LS equals Sionna LS."""
    cfg = system_variant_config("baseline", precision="double")
    resource_grid = build_resource_grid(cfg)
    num_subcarriers = cfg.fft_size
    covariance = torch.eye(num_subcarriers, dtype=torch.complex128)
    estimator = TruncatedChannelEstimator(
        resource_grid,
        covariance,
        interpolation_type=interpolation_type,
        method=METHOD_LS,
        num_taps=num_subcarriers,
        precision=cfg.precision,
        device=cfg.device,
    )
    reference = LSChannelEstimator(
        resource_grid,
        interpolation_type=interpolation_type,
        precision=cfg.precision,
        device=cfg.device,
    )
    received_grid = torch.complex(
        torch.randn(
            2, 1, 3, cfg.num_ofdm_symbols, num_subcarriers, dtype=torch.float64
        ),
        torch.randn(
            2, 1, 3, cfg.num_ofdm_symbols, num_subcarriers, dtype=torch.float64
        ),
    )
    noise_variance = torch.full((2, 1, 3), 0.2, dtype=torch.float64)

    estimate, _ = estimator(received_grid, noise_variance)
    reference_estimate, _ = reference(received_grid, noise_variance)

    torch.testing.assert_close(estimate, reference_estimate)


def test_custom_interpolator_is_forwarded_to_sionna() -> None:
    """A custom Sionna interpolator overrides the built-in Edfors default."""
    cfg = system_variant_config("baseline", precision="double")
    resource_grid = build_resource_grid(cfg)
    interpolator = NearestNeighborInterpolator(resource_grid.pilot_pattern)

    estimator = TruncatedChannelEstimator(
        resource_grid,
        torch.eye(cfg.fft_size, dtype=torch.complex128),
        interpolator=interpolator,
        method=METHOD_LS,
        num_taps=cfg.fft_size,
        precision=cfg.precision,
        device=cfg.device,
    )

    assert estimator._interpol is interpolator


def test_all_tap_lmmse_matches_sionna_lmmse_estimator() -> None:
    """Full-rank Edfors LMMSE equals Sionna LMMSE for the same covariance."""
    cfg = system_variant_config("baseline", precision="double")
    resource_grid = build_resource_grid(cfg)
    num_subcarriers = cfg.fft_size
    delay_powers = torch.linspace(
        2.0,
        0.1,
        num_subcarriers,
        dtype=torch.float64,
    )
    delay_covariance = torch.diag(delay_powers).to(torch.complex128)
    dft_matrix = torch.fft.fft(
        torch.eye(num_subcarriers, dtype=torch.complex128),
        dim=0,
        norm="ortho",
    )
    frequency_covariance = (
        dft_matrix @ delay_covariance @ dft_matrix.mH
    )
    channel_variance = frequency_covariance.diagonal().real[0]
    # A constant channel over OFDM symbols makes Sionna's temporal LMMSE pass
    # identical to the one-pilot-symbol linear interpolation used by Edfors.
    ofdm_time_covariance = torch.ones(
        cfg.num_ofdm_symbols,
        cfg.num_ofdm_symbols,
        dtype=torch.complex128,
    ) * channel_variance

    estimator = TruncatedChannelEstimator(
        resource_grid,
        delay_covariance,
        method=METHOD_LMMSE,
        num_taps=num_subcarriers,
        precision=cfg.precision,
        device=cfg.device,
    )
    reference = LMMSEChannelEstimator(
        resource_grid,
        cov_mat_time=ofdm_time_covariance,
        cov_mat_freq=frequency_covariance,
        order="f-t",
        precision=cfg.precision,
        device=cfg.device,
    )
    received_grid = torch.complex(
        torch.randn(
            2, 1, 3, cfg.num_ofdm_symbols, num_subcarriers, dtype=torch.float64
        ),
        torch.randn(
            2, 1, 3, cfg.num_ofdm_symbols, num_subcarriers, dtype=torch.float64
        ),
    )
    noise_variance = torch.full((2, 1, 3), 0.2, dtype=torch.float64)

    estimate, _ = estimator(received_grid, noise_variance)
    reference_estimate, _ = reference(received_grid, noise_variance)

    torch.testing.assert_close(
        estimate,
        reference_estimate,
        rtol=1e-10,
        atol=1e-10,
    )


def test_lmmse_applies_complex_delay_covariance_in_the_correct_order() -> None:
    """A complex covariance produces the direct frequency-domain LMMSE map."""
    cfg = system_variant_config("baseline", precision="double")
    resource_grid = build_resource_grid(cfg)
    num_subcarriers = cfg.fft_size
    covariance_factors = torch.complex(
        torch.randn(num_subcarriers, 5, dtype=torch.float64),
        torch.randn(num_subcarriers, 5, dtype=torch.float64),
    )
    delay_covariance = (
        covariance_factors @ covariance_factors.mH
        + 0.5 * torch.eye(num_subcarriers, dtype=torch.complex128)
    )
    estimator = TruncatedChannelEstimator(
        resource_grid,
        delay_covariance,
        method=METHOD_LMMSE,
        num_taps=num_subcarriers,
        precision=cfg.precision,
        device=cfg.device,
    )
    frequency_values = torch.complex(
        torch.randn(num_subcarriers, dtype=torch.float64),
        torch.randn(num_subcarriers, dtype=torch.float64),
    )
    pilots = resource_grid.pilot_pattern.pilots
    pilot_observations = (
        frequency_values.reshape(1, 1, 1, 1, 1, num_subcarriers) * pilots
    )
    noise_variance = torch.tensor(0.2, dtype=torch.float64)

    estimate, _ = estimator.estimate_at_pilot_locations(
        pilot_observations,
        noise_variance,
    )
    grouped_estimate = estimator._gather_pilot_groups(estimate)

    dft_matrix = torch.fft.fft(
        torch.eye(num_subcarriers, dtype=torch.complex128),
        dim=0,
        norm="ortho",
    )
    frequency_covariance = (
        dft_matrix @ delay_covariance @ dft_matrix.mH
    )
    expected = frequency_covariance @ torch.linalg.solve(
        frequency_covariance
        + noise_variance * torch.eye(num_subcarriers, dtype=torch.complex128),
        frequency_values,
    )

    torch.testing.assert_close(
        grouped_estimate[0, 0, 0, 0],
        expected,
        rtol=1e-10,
        atol=1e-10,
    )


def test_maximal_truncation_keeps_only_the_strongest_delay_coefficient() -> None:
    """Retaining one tap removes every other transformed LS coefficient."""
    cfg = system_variant_config("baseline", precision="double")
    resource_grid = build_resource_grid(cfg)
    num_subcarriers = cfg.fft_size
    kept_tap = 7
    delay_powers = torch.full(
        (num_subcarriers,),
        0.1,
        dtype=torch.float64,
    )
    delay_powers[kept_tap] = 2.0
    estimator = TruncatedChannelEstimator(
        resource_grid,
        torch.diag(delay_powers),
        method=METHOD_LS,
        num_taps=1,
        precision=cfg.precision,
        device=cfg.device,
    )
    delay_values = torch.complex(
        torch.linspace(-1.0, 1.0, num_subcarriers, dtype=torch.float64),
        torch.linspace(1.0, -1.0, num_subcarriers, dtype=torch.float64),
    )
    frequency_values = torch.fft.fft(delay_values, norm="ortho")
    pilots = resource_grid.pilot_pattern.pilots
    pilot_observations = (
        frequency_values.reshape(1, 1, 1, 1, 1, num_subcarriers) * pilots
    )

    estimate, _ = estimator.estimate_at_pilot_locations(
        pilot_observations,
        torch.tensor(0.0, dtype=torch.float64),
    )
    grouped_estimate = estimator._gather_pilot_groups(estimate)
    estimated_delay = torch.fft.ifft(
        grouped_estimate[0, 0, 0, 0],
        norm="ortho",
    )
    expected_delay = torch.zeros_like(delay_values)
    expected_delay[kept_tap] = delay_values[kept_tap]

    torch.testing.assert_close(estimated_delay, expected_delay)


def test_delay_limited_channel_is_exactly_recovered_without_noise() -> None:
    """Truncation is lossless when the channel lies in the retained subspace."""
    cfg = system_variant_config("baseline", precision="double")
    resource_grid = build_resource_grid(cfg)
    num_subcarriers = cfg.fft_size
    support = torch.tensor([2, 9, 17])
    delay_powers = torch.full(
        (num_subcarriers,),
        0.1,
        dtype=torch.float64,
    )
    delay_powers[support] = torch.tensor(
        [3.0, 2.0, 1.0],
        dtype=torch.float64,
    )
    estimator = TruncatedChannelEstimator(
        resource_grid,
        torch.diag(delay_powers),
        method=METHOD_LS,
        num_taps=len(support),
        precision=cfg.precision,
        device=cfg.device,
    )
    delay_values = torch.zeros(num_subcarriers, dtype=torch.complex128)
    delay_values[support] = torch.tensor(
        [1.0 + 0.5j, -0.25 + 0.75j, 0.4 - 0.2j],
        dtype=torch.complex128,
    )
    frequency_values = torch.fft.fft(delay_values, norm="ortho")
    pilots = resource_grid.pilot_pattern.pilots
    pilot_observations = (
        frequency_values.reshape(1, 1, 1, 1, 1, num_subcarriers) * pilots
    )

    estimate, _ = estimator.estimate_at_pilot_locations(
        pilot_observations,
        torch.tensor(0.0, dtype=torch.float64),
    )
    grouped_estimate = estimator._gather_pilot_groups(estimate)

    torch.testing.assert_close(
        grouped_estimate[0, 0, 0, 0],
        frequency_values,
    )


def test_confidence_scale_propagates_through_linear_interpolation() -> None:
    """The fallback remains scale times interpolated LS observation variance."""
    cfg = system_variant_config("baseline", precision="double")
    resource_grid = build_resource_grid(cfg)
    num_subcarriers = cfg.fft_size
    confidence_scale = 0.25
    estimator = TruncatedChannelEstimator(
        resource_grid,
        torch.eye(num_subcarriers, dtype=torch.complex128),
        method=METHOD_LS,
        num_taps=num_subcarriers,
        chest_err_const=confidence_scale,
        precision=cfg.precision,
        device=cfg.device,
    )
    reference = LSChannelEstimator(
        resource_grid,
        interpolation_type="lin",
        precision=cfg.precision,
        device=cfg.device,
    )
    received_grid = torch.zeros(
        2,
        1,
        3,
        cfg.num_ofdm_symbols,
        num_subcarriers,
        dtype=torch.complex128,
    )
    noise_variance = torch.tensor(
        [[[0.1, 0.2, 0.3]], [[0.4, 0.5, 0.6]]],
        dtype=torch.float64,
    )

    _, error_variance = estimator(received_grid, noise_variance)
    _, ls_error_variance = reference(received_grid, noise_variance)

    torch.testing.assert_close(
        error_variance,
        confidence_scale * ls_error_variance,
    )


@pytest.mark.compile
@pytest.mark.filterwarnings(
    "ignore:.*torch.jit.script_method.*:DeprecationWarning"
)
def test_all_methods_match_compiled_full_graph() -> None:
    """Every complete online method agrees with Tier A compiled inference."""
    torch._dynamo.reset()
    cfg = system_variant_config("alternate")
    resource_grid = build_resource_grid(cfg)
    active_pilots = resource_grid.pilot_pattern.pilots.abs() > 0
    num_pilot_subcarriers = (
        int(active_pilots.sum(dim=-1).min().item())
        // len(cfg.pilot_ofdm_symbol_indices)
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

    for method in METHOD_NAMES:
        estimator = TruncatedChannelEstimator(
            resource_grid,
            torch.eye(num_pilot_subcarriers, dtype=torch.complex64),
            method=method,
            num_taps=_NUM_TAPS,
            precision=cfg.precision,
            device=cfg.device,
        )
        channel_estimate, error_variance = assert_compiled_estimator_matches_eager(
            estimator,
            received_grid,
            noise_variance,
            fullgraph=True,
        )

        assert channel_estimate.shape == error_variance.shape
