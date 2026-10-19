# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for full-subcarrier interpolation over OFDM symbols."""

from __future__ import annotations

import pytest
import torch
from sionna.phy.ofdm import PilotPattern

from sionna.bench.mimo_ofdm_chest.benchmarking.communication_system import build_resource_grid
from sionna.phy.ofdm.channel_estimation import _pilot_ind_from_pattern
from sionna.bench.mimo_ofdm_chest.time_domain_interpolator import TimeDomainInterpolator
from test.mimo_ofdm_chest.system_variants import system_variant_config


pytestmark = pytest.mark.system_variants


@pytest.mark.parametrize("variant", ["baseline", "alternate", "wide"])
@pytest.mark.parametrize("time_avg", [False, True])
def test_time_domain_interpolator_across_system_variants(
    variant: str,
    time_avg: bool,
) -> None:
    """Verify interpolation for one, two, and three pilot symbols.

    Without time averaging, one pilot symbol must be extended unchanged and
    two or three pilot symbols must reproduce a channel and variance that are
    linear in OFDM-symbol index, including extrapolation at grid boundaries.
    With time averaging, the pilot-symbol mean must be repeated over the full
    OFDM-time axis. Both modes must preserve output shape and dtype and return
    finite, non-negative error variances.
    """
    cfg = system_variant_config(variant)
    resource_grid = build_resource_grid(cfg)
    interpolator = TimeDomainInterpolator(
        resource_grid.pilot_pattern,
        time_avg=time_avg,
    )

    pilot_times = torch.tensor(
        cfg.pilot_ofdm_symbol_indices,
        dtype=torch.float32,
    )
    pilot_shape = (
        1,
        1,
        cfg.num_rx_ant,
        cfg.num_ue,
        cfg.num_streams_per_tx,
        len(cfg.pilot_ofdm_symbol_indices),
        cfg.fft_size,
    )
    channel_at_pilots = (
        (1.0 - 2.0j) + (0.5 + 0.25j) * pilot_times
    ).reshape(1, 1, 1, 1, 1, -1, 1).expand(pilot_shape)
    variance_at_pilots = (10.0 + pilot_times).reshape(
        1, 1, 1, 1, 1, -1, 1
    ).expand(pilot_shape)

    channel_estimate, error_variance = interpolator(
        channel_at_pilots,
        variance_at_pilots,
    )

    output_shape = (*pilot_shape[:-2], cfg.num_ofdm_symbols, cfg.fft_size)
    assert channel_estimate.shape == output_shape
    assert error_variance.shape == output_shape
    assert channel_estimate.dtype == torch.complex64
    assert error_variance.dtype == torch.float32
    assert bool(torch.isfinite(channel_estimate).all())
    assert bool(torch.isfinite(error_variance).all())
    assert bool(torch.all(error_variance >= 0.0))

    if time_avg or len(cfg.pilot_ofdm_symbol_indices) == 1:
        expected_channel_values = channel_at_pilots.mean(dim=-2, keepdim=True)
        expected_variance_values = variance_at_pilots.mean(dim=-2, keepdim=True)
    else:
        all_times = torch.arange(cfg.num_ofdm_symbols, dtype=torch.float32)
        expected_channel_values = (
            (1.0 - 2.0j) + (0.5 + 0.25j) * all_times
        ).reshape(1, 1, 1, 1, 1, -1, 1)
        expected_variance_values = (10.0 + all_times).reshape(
            1, 1, 1, 1, 1, -1, 1
        )

    torch.testing.assert_close(
        channel_estimate,
        expected_channel_values.expand(output_shape),
    )
    torch.testing.assert_close(
        error_variance,
        expected_variance_values.expand(output_shape),
    )


def test_time_domain_interpolator_uses_active_per_stream_symbol_positions() -> None:
    """Compact slots follow nonzero pilot symbols, not the sparse mask layout."""
    cfg = system_variant_config("wide")
    resource_grid = build_resource_grid(cfg)
    original_pattern = resource_grid.pilot_pattern
    pilot_indices = _pilot_ind_from_pattern(original_pattern, device="cpu")
    pilot_symbols = pilot_indices // int(
        original_pattern.num_effective_subcarriers
    )
    pilots = original_pattern.pilots.clone()

    active_symbols = torch.tensor(
        [
            [2, 5],
            [2, 9],
            [5, 9],
            [2, 5],
        ],
        dtype=torch.int64,
    ).reshape(cfg.num_ue, cfg.num_streams_per_tx, 2)
    deactivated_symbols = torch.tensor([9, 5, 2, 9], dtype=torch.int64)
    for tx_index, symbol_index in enumerate(deactivated_symbols):
        pilots[tx_index][pilot_symbols[tx_index] == symbol_index] = 0.0

    pilot_pattern = PilotPattern(
        original_pattern.mask,
        pilots,
        precision=cfg.precision,
        device=cfg.device,
    )
    interpolator = TimeDomainInterpolator(pilot_pattern)
    torch.testing.assert_close(
        interpolator._pilot_ofdm_symbol_indices,
        active_symbols,
    )

    input_shape = (
        1,
        1,
        cfg.num_rx_ant,
        cfg.num_ue,
        cfg.num_streams_per_tx,
        2,
        cfg.fft_size,
    )
    pilot_times = active_symbols.to(torch.float32).reshape(
        1,
        1,
        1,
        cfg.num_ue,
        cfg.num_streams_per_tx,
        2,
        1,
    )
    channel_at_pilots = (
        (1.0 - 2.0j) + (0.5 + 0.25j) * pilot_times
    ).expand(input_shape)
    variance_at_pilots = (10.0 + pilot_times).expand(input_shape)

    channel_estimate, error_variance = interpolator(
        channel_at_pilots,
        variance_at_pilots,
    )
    output_shape = (*input_shape[:-2], cfg.num_ofdm_symbols, cfg.fft_size)
    all_times = torch.arange(cfg.num_ofdm_symbols, dtype=torch.float32).reshape(
        1, 1, 1, 1, 1, -1, 1
    )
    expected_channel = (
        (1.0 - 2.0j) + (0.5 + 0.25j) * all_times
    ).expand(output_shape)
    expected_variance = (10.0 + all_times).expand(output_shape)

    torch.testing.assert_close(channel_estimate, expected_channel)
    torch.testing.assert_close(error_variance, expected_variance)


@pytest.mark.compile
def test_time_domain_interpolator_matches_compile_eager() -> None:
    """Linear OFDM-time interpolation agrees under eager and compiled execution."""
    cfg = system_variant_config("alternate")
    resource_grid = build_resource_grid(cfg)
    interpolator = TimeDomainInterpolator(resource_grid.pilot_pattern)
    input_shape = (
        1,
        1,
        cfg.num_rx_ant,
        cfg.num_ue,
        cfg.num_streams_per_tx,
        len(cfg.pilot_ofdm_symbol_indices),
        cfg.fft_size,
    )
    channel_at_pilots = torch.randn(input_shape, dtype=torch.complex64)
    variance_at_pilots = torch.rand(input_shape, dtype=torch.float32)

    eager_output = interpolator(channel_at_pilots, variance_at_pilots)
    compiled_interpolator = torch.compile(
        interpolator,
        backend="eager",
        fullgraph=False,
    )
    compiled_output = compiled_interpolator(
        channel_at_pilots,
        variance_at_pilots,
    )

    torch.testing.assert_close(compiled_output, eager_output)
