# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import math

import pytest
import torch
from sionna.phy import config
from sionna.phy.ofdm import PilotPattern

from sionna.bench.mimo_ofdm_chest.benchmarking.communication_system import build_resource_grid
from sionna.bench.mimo_ofdm_chest.papers.coleri2002.estimator import (
    CombInterpolator,
    METHOD_LOWPASS,
    METHOD_NAMES,
    METHOD_SECOND_ORDER,
    METHOD_SPLINE,
    METHOD_TIME_DOMAIN,
    LSCombEstimator,
)
from test.mimo_ofdm_chest.system_variants import (
    assert_compiled_estimator_matches_eager,
    system_variant_config,
    validate_estimator_output,
)


_SYSTEM_VARIANTS = ("baseline", "alternate", "wide")


@pytest.fixture(autouse=True)
def reset_sionna_config():
    old_device = config.device
    old_precision = config.precision
    config.device = "cpu"
    config.precision = "double"
    yield
    config.device = old_device
    config.precision = old_precision


def _resource_grid(variant: str):
    cfg = system_variant_config(variant, precision="double")
    return cfg, build_resource_grid(cfg)


def _interpolator(
    resource_grid,
    method: str,
    **kwargs,
) -> CombInterpolator:
    return CombInterpolator(
        resource_grid.pilot_pattern,
        method=method,
        precision="double",
        device="cpu",
        **kwargs,
    )


def _unique_offset_groups(interpolator: CombInterpolator) -> list[tuple[int, int]]:
    groups_by_offset = {}
    for group_index, offset in enumerate(interpolator._group_comb_offsets):
        groups_by_offset.setdefault(offset, group_index)
    return sorted(groups_by_offset.items())


def _pilot_positions(offset: int, spacing: int, num_pilots: int) -> torch.Tensor:
    return offset + torch.arange(num_pilots, dtype=torch.int64) * spacing


def _complex_randn(*shape: int) -> torch.Tensor:
    real = torch.randn(*shape, dtype=torch.float64)
    imag = torch.randn(*shape, dtype=torch.float64)
    return torch.complex(real, imag)


def _gather_pilots(
    full_values: torch.Tensor,
    offset: int,
    spacing: int,
    num_pilots: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    positions = _pilot_positions(offset, spacing, num_pilots)
    return full_values.index_select(-1, positions), positions


def _row_noise_gain(output_from_identity: torch.Tensor) -> torch.Tensor:
    return output_from_identity.abs().square().sum(dim=0)


# Complete estimator integration


@pytest.mark.parametrize("variant", _SYSTEM_VARIANTS)
@pytest.mark.parametrize("method", METHOD_NAMES)
@pytest.mark.system_variants
def test_estimator_output_contract(variant: str, method: str) -> None:
    """Every method satisfies the Sionna output contract on supported grids."""
    cfg, resource_grid = _resource_grid(variant)
    estimator = LSCombEstimator(
        resource_grid,
        method=method,
        precision=cfg.precision,
        device=cfg.device,
    )
    received_grid = torch.randn(
        1,
        1,
        cfg.num_rx_ant,
        cfg.num_ofdm_symbols,
        cfg.fft_size,
        dtype=torch.complex128,
    )
    channel_estimate, error_variance = estimator(
        received_grid,
        torch.tensor(0.1, dtype=torch.float64),
    )
    expected_channel = torch.empty(
        1,
        1,
        cfg.num_rx_ant,
        cfg.num_ue,
        cfg.num_streams_per_tx,
        cfg.num_ofdm_symbols,
        cfg.fft_size,
        dtype=torch.complex128,
    )

    validate_estimator_output(
        channel_estimate,
        error_variance,
        expected_channel,
    )


# Unsupported pilot geometry


def test_nonuniform_comb_is_rejected() -> None:
    """A pilot group with unequal subcarrier gaps is rejected explicitly."""
    mask = torch.zeros(1, 1, 4, 8, dtype=torch.int32)
    mask[0, 0, 1, [0, 2, 5, 7]] = 1
    pilot_pattern = PilotPattern(
        mask,
        torch.ones(1, 1, 4, dtype=torch.complex128),
        precision="double",
        device="cpu",
    )

    with pytest.raises(AssertionError, match="requires uniform comb pilots"):
        CombInterpolator(pilot_pattern, precision="double", device="cpu")


def test_incomplete_full_grid_comb_is_rejected() -> None:
    """A uniform comb that does not span one complete grid is rejected."""
    mask = torch.zeros(1, 1, 4, 8, dtype=torch.int32)
    mask[0, 0, 1, [0, 2, 4]] = 1
    pilot_pattern = PilotPattern(
        mask,
        torch.ones(1, 1, 3, dtype=torch.complex128),
        precision="double",
        device="cpu",
    )

    with pytest.raises(AssertionError, match="comb pilots to span the full grid"):
        CombInterpolator(pilot_pattern, precision="double", device="cpu")


def test_unequal_active_pilot_symbol_counts_are_rejected() -> None:
    """Streams must contribute the same number of active pilot groups."""
    mask = torch.zeros(2, 1, 4, 8, dtype=torch.int32)
    mask[:, 0, :3, ::2] = 1
    pilots = torch.ones(2, 1, 12, dtype=torch.complex128)
    pilots[0, 0, 4:] = 0.0
    pilot_pattern = PilotPattern(
        mask,
        pilots,
        precision="double",
        device="cpu",
    )

    with pytest.raises(
        AssertionError,
        match="requires equal pilot-symbol counts per stream",
    ):
        CombInterpolator(pilot_pattern, precision="double", device="cpu")


# Shared frequency-domain invariants


@pytest.mark.parametrize("variant", _SYSTEM_VARIANTS)
def test_frequency_interpolators_pass_through_pilots(variant: str) -> None:
    """Every frequency interpolator reproduces its input pilot samples."""
    _, resource_grid = _resource_grid(variant)
    torch.manual_seed(0)

    for method in METHOD_NAMES:
        interpolator = _interpolator(resource_grid, method)
        spacing = interpolator._pilot_spacing
        num_pilots = interpolator._num_pilots_per_symbol

        for offset, group_index in _unique_offset_groups(interpolator):
            pilot_values = _complex_randn(3, num_pilots)
            interpolated = interpolator._interpolate_frequency(
                pilot_values,
                group_index,
            )
            positions = _pilot_positions(offset, spacing, num_pilots)
            torch.testing.assert_close(
                interpolated.index_select(-1, positions),
                pilot_values,
                atol=1e-10,
                rtol=1e-10,
            )


@pytest.mark.parametrize("variant", _SYSTEM_VARIANTS)
def test_frequency_interpolation_noise_gain_is_bounded(variant: str) -> None:
    """All interpolation matrices have finite and bounded row energy."""
    _, resource_grid = _resource_grid(variant)
    for method in METHOD_NAMES:
        interpolator = _interpolator(resource_grid, method)
        num_pilots = interpolator._num_pilots_per_symbol

        for _offset, group_index in _unique_offset_groups(interpolator):
            basis = torch.eye(num_pilots, dtype=torch.complex128)
            interpolated_basis = interpolator._interpolate_frequency(
                basis,
                group_index,
            )
            gain = _row_noise_gain(interpolated_basis)
            assert bool(torch.all(torch.isfinite(gain)))
            assert float(gain.min()) >= 0.0
            assert float(gain.mean()) <= 1.05
            assert float(gain.max()) <= 1.5


@pytest.mark.parametrize("method", METHOD_NAMES)
def test_error_variance_scale_does_not_change_channel_estimate(method: str) -> None:
    """The shared detector scale changes only interpolated error variances."""
    cfg, resource_grid = _resource_grid("alternate")
    received_grid = torch.randn(
        1,
        1,
        cfg.num_rx_ant,
        cfg.num_ofdm_symbols,
        cfg.fft_size,
        dtype=torch.complex128,
    )
    noise_variance = torch.tensor(0.1, dtype=torch.float64)
    unscaled_estimator = LSCombEstimator(
        resource_grid,
        method=method,
        err_var_scale=1.0,
        precision=cfg.precision,
        device=cfg.device,
    )
    scaled_estimator = LSCombEstimator(
        resource_grid,
        method=method,
        err_var_scale=2.5,
        precision=cfg.precision,
        device=cfg.device,
    )

    unscaled_channel, unscaled_variance = unscaled_estimator(
        received_grid,
        noise_variance,
    )
    scaled_channel, scaled_variance = scaled_estimator(
        received_grid,
        noise_variance,
    )

    torch.testing.assert_close(scaled_channel, unscaled_channel)
    torch.testing.assert_close(scaled_variance, 2.5 * unscaled_variance)


# Method-specific frequency-domain invariants


@pytest.mark.parametrize("variant", _SYSTEM_VARIANTS)
def test_second_order_recovers_quadratics_away_from_edges(variant: str) -> None:
    """Second-order interpolation exactly recovers interior quadratics."""
    _, resource_grid = _resource_grid(variant)
    interpolator = _interpolator(resource_grid, METHOD_SECOND_ORDER)
    spacing = interpolator._pilot_spacing
    num_pilots = interpolator._num_pilots_per_symbol
    num_subcarriers = interpolator._num_subcarriers

    for offset, group_index in _unique_offset_groups(interpolator):
        subcarriers = torch.arange(num_subcarriers, dtype=torch.float64)
        pilot_grid_coordinate = (subcarriers - float(offset)) / float(spacing)
        coeffs = torch.tensor(
            [0.013 + 0.007j, -0.11 + 0.03j, 0.9 - 0.2j],
            dtype=torch.complex128,
        )
        full_values = (
            coeffs[0] * pilot_grid_coordinate.to(torch.complex128).square()
            + coeffs[1] * pilot_grid_coordinate.to(torch.complex128)
            + coeffs[2]
        )
        pilot_values, _ = _gather_pilots(
            full_values,
            offset,
            spacing,
            num_pilots,
        )

        interpolated = interpolator._interpolate_frequency(
            pilot_values.unsqueeze(0),
            group_index,
        ).squeeze(0)
        interval_indices = torch.div(
            torch.arange(num_subcarriers) - offset,
            spacing,
            rounding_mode="floor",
        )
        interior = (interval_indices >= 1) & (interval_indices <= num_pilots - 2)
        torch.testing.assert_close(
            interpolated[interior],
            full_values[interior],
            atol=1e-10,
            rtol=1e-10,
        )


@pytest.mark.parametrize("variant", ["alternate", "wide"])
def test_second_order_uses_clamped_edge_neighbors(variant: str) -> None:
    """Second-order boundaries reuse the nearest missing neighbor pilot."""
    _, resource_grid = _resource_grid(variant)
    interpolator = _interpolator(resource_grid, METHOD_SECOND_ORDER)
    spacing = interpolator._pilot_spacing
    num_pilots = interpolator._num_pilots_per_symbol
    pilot_values = torch.complex(
        torch.arange(num_pilots, dtype=torch.float64).square(),
        torch.arange(num_pilots, dtype=torch.float64),
    )

    for offset, group_index in _unique_offset_groups(interpolator):
        interpolated = interpolator._interpolate_frequency(
            pilot_values,
            group_index,
        )
        if offset > 0:
            torch.testing.assert_close(
                interpolated[:offset],
                pilot_values[0].expand(offset),
            )

        last_pilot_position = offset + (num_pilots - 1) * spacing
        tail_positions = torch.arange(
            last_pilot_position + 1,
            interpolator._num_subcarriers,
        )
        interpolation_fraction = (
            tail_positions - last_pilot_position
        ).to(torch.float64) / float(spacing)
        previous_weight = (
            interpolation_fraction * (interpolation_fraction - 1.0) / 2.0
        )
        current_weight = -(
            (interpolation_fraction - 1.0) * (interpolation_fraction + 1.0)
        )
        next_weight = (
            interpolation_fraction * (interpolation_fraction + 1.0) / 2.0
        )
        expected_tail = (
            previous_weight * pilot_values[-2]
            + current_weight * pilot_values[-1]
            + next_weight * pilot_values[-1]
        )
        torch.testing.assert_close(
            interpolated.index_select(-1, tail_positions),
            expected_tail,
        )


@pytest.mark.parametrize("variant", _SYSTEM_VARIANTS)
def test_lowpass_interpolator_preserves_smooth_signals(variant: str) -> None:
    """Low-pass interpolation preserves constants through both grid edges."""
    _, resource_grid = _resource_grid(variant)
    interpolator = _interpolator(
        resource_grid,
        METHOD_LOWPASS,
        lowpass_filter_half_length=4,
    )
    spacing = interpolator._pilot_spacing
    num_pilots = interpolator._num_pilots_per_symbol
    num_subcarriers = interpolator._num_subcarriers

    for offset, group_index in _unique_offset_groups(interpolator):
        constant = torch.full(
            (num_subcarriers,),
            1.0 + 0.5j,
            dtype=torch.complex128,
        )
        pilot_values, _ = _gather_pilots(constant, offset, spacing, num_pilots)
        interpolated = interpolator._interpolate_frequency(
            pilot_values.unsqueeze(0),
            group_index,
        ).squeeze(0)
        assert float((interpolated - constant).abs().max()) < 5e-4

        subcarriers = torch.arange(num_subcarriers, dtype=torch.float64)
        sinusoid = torch.exp(
            1j
            * 2.0
            * math.pi
            * 2.0
            * (subcarriers - float(offset))
            / float(num_subcarriers)
        ).to(torch.complex128)
        pilot_values, _ = _gather_pilots(sinusoid, offset, spacing, num_pilots)
        interpolated = interpolator._interpolate_frequency(
            pilot_values.unsqueeze(0),
            group_index,
        ).squeeze(0)
        filter_padding = int(interpolator._lowpass_coefficients.numel() // 2)
        interior = (
            slice(None)
            if filter_padding == 0
            else slice(filter_padding, -filter_padding)
        )
        relative_rmse = torch.sqrt(
            (interpolated[..., interior] - sinusoid[interior]).abs().square().mean()
        ) / torch.sqrt(sinusoid[interior].abs().square().mean())
        assert float(relative_rmse) < 5e-3


@pytest.mark.parametrize("variant", _SYSTEM_VARIANTS)
def test_cubic_spline_recovers_cubics_inside_pilot_span(variant: str) -> None:
    """Not-a-knot spline interpolation recovers cubics within its data span."""
    _, resource_grid = _resource_grid(variant)
    interpolator = _interpolator(resource_grid, METHOD_SPLINE)
    spacing = interpolator._pilot_spacing
    num_pilots = interpolator._num_pilots_per_symbol
    num_subcarriers = interpolator._num_subcarriers

    for offset, group_index in _unique_offset_groups(interpolator):
        subcarriers = torch.arange(num_subcarriers, dtype=torch.float64)
        x = (subcarriers - float(offset)) / float(num_subcarriers)
        coeffs = torch.tensor(
            [0.4 - 0.1j, -0.2 + 0.05j, 0.7 + 0.2j, -0.3j],
            dtype=torch.complex128,
        )
        full_values = (
            coeffs[0] * x.to(torch.complex128).pow(3)
            + coeffs[1] * x.to(torch.complex128).square()
            + coeffs[2] * x.to(torch.complex128)
            + coeffs[3]
        )
        pilot_values, positions = _gather_pilots(
            full_values,
            offset,
            spacing,
            num_pilots,
        )

        interpolated = interpolator._interpolate_frequency(
            pilot_values.unsqueeze(0),
            group_index,
        ).squeeze(0)
        inside_span = (subcarriers >= positions[0]) & (subcarriers <= positions[-1])
        torch.testing.assert_close(
            interpolated[inside_span],
            full_values[inside_span],
            atol=1e-10,
            rtol=1e-10,
        )


@pytest.mark.parametrize("variant", ["alternate", "wide"])
def test_cubic_spline_clips_queries_outside_pilot_span(variant: str) -> None:
    """Spline queries beyond observed pilots use constant edge values."""
    _, resource_grid = _resource_grid(variant)
    interpolator = _interpolator(resource_grid, METHOD_SPLINE)
    spacing = interpolator._pilot_spacing
    num_pilots = interpolator._num_pilots_per_symbol
    pilot_values = _complex_randn(num_pilots)

    for offset, group_index in _unique_offset_groups(interpolator):
        interpolated = interpolator._interpolate_frequency(
            pilot_values,
            group_index,
        )
        if offset > 0:
            torch.testing.assert_close(
                interpolated[:offset],
                pilot_values[0].expand(offset),
            )

        last_pilot_position = offset + (num_pilots - 1) * spacing
        num_trailing_subcarriers = (
            interpolator._num_subcarriers - last_pilot_position - 1
        )
        if num_trailing_subcarriers > 0:
            torch.testing.assert_close(
                interpolated[last_pilot_position + 1 :],
                pilot_values[-1].expand(num_trailing_subcarriers),
            )


@pytest.mark.parametrize("variant", _SYSTEM_VARIANTS)
def test_time_domain_recovers_delay_limited_responses(variant: str) -> None:
    """Transform interpolation recovers responses in its delay subspace."""
    _, resource_grid = _resource_grid(variant)
    torch.manual_seed(1)
    interpolator = _interpolator(resource_grid, METHOD_TIME_DOMAIN)
    spacing = interpolator._pilot_spacing
    num_pilots = interpolator._num_pilots_per_symbol
    num_subcarriers = interpolator._num_subcarriers
    split = num_pilots // 2 + 1

    for offset, group_index in _unique_offset_groups(interpolator):
        retained_taps = _complex_randn(num_pilots)
        time_domain = torch.zeros(num_subcarriers, dtype=torch.complex128)
        time_domain[:split] = retained_taps[:split]
        time_domain[-(num_pilots - split):] = retained_taps[split:]

        full_values = torch.roll(
            torch.fft.fft(time_domain),
            shifts=offset,
            dims=-1,
        )
        pilot_values, _ = _gather_pilots(
            full_values,
            offset,
            spacing,
            num_pilots,
        )
        interpolated = interpolator._interpolate_frequency(
            pilot_values.unsqueeze(0),
            group_index,
        ).squeeze(0)
        torch.testing.assert_close(
            interpolated,
            full_values,
            atol=1e-10,
            rtol=1e-10,
        )


# OFDM-time adaptation


@pytest.mark.parametrize("variant", _SYSTEM_VARIANTS)
def test_ofdm_time_interpolation_is_linear(variant: str) -> None:
    """OFDM-time interpolation extends one pilot or recovers linear trends."""
    system_config, resource_grid = _resource_grid(variant)
    interpolator = _interpolator(resource_grid, METHOD_SECOND_ORDER)
    base_values = _complex_randn(2, system_config.fft_size)
    slopes = _complex_randn(2, system_config.fft_size)
    pilot_symbol_indices = system_config.pilot_ofdm_symbol_indices
    symbol_values = tuple(
        base_values + float(symbol_index) * slopes
        for symbol_index in pilot_symbol_indices
    )

    interpolated = interpolator._interpolate_time(
        symbol_values,
        pilot_symbol_indices,
    )
    if len(pilot_symbol_indices) == 1:
        expected = symbol_values[0].unsqueeze(-2).expand_as(interpolated)
    else:
        symbol_indices = torch.arange(
            system_config.num_ofdm_symbols,
            dtype=torch.float64,
        ).reshape(1, -1, 1)
        expected = base_values.unsqueeze(-2) + symbol_indices * slopes.unsqueeze(-2)
    torch.testing.assert_close(interpolated, expected, atol=1e-12, rtol=1e-12)


# Compilation


@pytest.mark.compile
@pytest.mark.parametrize("method", METHOD_NAMES)
def test_estimator_matches_compile_eager(method: str) -> None:
    """Each complete Coleri estimator call agrees under Torch compilation."""
    cfg, resource_grid = _resource_grid("alternate")
    estimator = LSCombEstimator(
        resource_grid,
        method=method,
        precision=cfg.precision,
        device=cfg.device,
    )
    received_grid = torch.randn(
        1,
        1,
        cfg.num_rx_ant,
        cfg.num_ofdm_symbols,
        cfg.fft_size,
        dtype=torch.complex128,
    )
    noise_variance = torch.tensor(0.1, dtype=torch.float64)

    channel_estimate, error_variance = assert_compiled_estimator_matches_eager(
        estimator,
        received_grid,
        noise_variance,
        fullgraph=True,
    )

    assert channel_estimate.shape == error_variance.shape
