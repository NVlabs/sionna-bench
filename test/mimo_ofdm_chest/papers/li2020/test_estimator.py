# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Architecture, pilot-layout, and inference tests for Li ReEsNet."""

from __future__ import annotations

import pytest
import torch

from sionna.bench.mimo_ofdm_chest.benchmarking import build_resource_grid
from sionna.bench.mimo_ofdm_chest.papers.li2020.estimator import (
    ReEsNet,
    ReEsNetChannelEstimator,
    ResidualBlock,
)
from test.mimo_ofdm_chest.system_variants import (
    assert_compiled_estimator_matches_eager,
    system_variant_config,
    validate_estimator_output,
)


_SYSTEM_VARIANTS = ("baseline", "alternate", "wide")


def _grid_shapes(resource_grid, cfg) -> tuple[tuple[int, int], tuple[int, int]]:
    """Return ReEsNet pilot-image and full-grid shapes for ``cfg``."""
    num_pilot_symbols = len(cfg.pilot_ofdm_symbol_indices)
    num_active_pilots = int(
        torch.count_nonzero(resource_grid.pilot_pattern.pilots[0, 0]).item()
    )
    input_shape = (
        num_active_pilots // num_pilot_symbols,
        num_pilot_symbols,
    )
    output_shape = (
        int(resource_grid.num_effective_subcarriers),
        int(resource_grid.num_ofdm_symbols),
    )
    return input_shape, output_shape


def _linear_reesnet_state(resource_grid, cfg) -> dict[str, torch.Tensor]:
    """Return a state that repeats each real pilot-image cell to its block."""
    input_shape, output_shape = _grid_shapes(resource_grid, cfg)
    model = ReEsNet(input_shape, output_shape)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.input_conv.weight[0, 0, 1, 1] = 1.0
        model.upsampling.weight[0, 0].fill_(1.0)
        model.output_conv.weight[0, 0, 1, 1] = 1.0
    return model.state_dict()


def _constant_reesnet_state(
    resource_grid,
    cfg,
    value: float,
) -> dict[str, torch.Tensor]:
    """Return a valid ReEsNet state with a constant real output."""
    input_shape, output_shape = _grid_shapes(resource_grid, cfg)
    model = ReEsNet(input_shape, output_shape)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.output_conv.bias[0] = value
    return model.state_dict()


def test_reesnet_matches_paper_architecture() -> None:
    """The default network follows Li's compact EDSR-derived architecture."""
    model = ReEsNet((6, 2), (24, 14))

    assert model.input_conv.in_channels == 2
    assert model.input_conv.out_channels == 16
    assert model.input_conv.kernel_size == (3, 3)
    assert len(model.blocks) == 4
    assert all(isinstance(block, ResidualBlock) for block in model.blocks)
    assert all(block.conv1.in_channels == 16 for block in model.blocks)
    assert all(block.conv2.out_channels == 16 for block in model.blocks)
    assert not any(
        isinstance(module, torch.nn.BatchNorm2d) for module in model.modules()
    )
    assert isinstance(model.upsampling, torch.nn.ConvTranspose2d)
    assert model.upsampling.kernel_size == (4, 7)
    assert model.upsampling.stride == (4, 7)
    assert model.upsampling.padding == (0, 0)
    assert model.upsampling.output_padding == (0, 0)
    assert model.output_conv.in_channels == 16
    assert model.output_conv.out_channels == 2


def test_residual_block_identity_with_zero_residual_branch() -> None:
    """A zero two-convolution branch leaves its residual-block input intact."""
    block = ResidualBlock(3)
    with torch.no_grad():
        for parameter in block.parameters():
            parameter.zero_()
    inputs = torch.randn(2, 3, 5, 4)

    torch.testing.assert_close(block(inputs), inputs)


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
    """Complete mixed- and per-SNR inference agree when compiled."""
    cfg = system_variant_config("baseline")
    resource_grid = build_resource_grid(cfg)
    model_states = [
        _constant_reesnet_state(resource_grid, cfg, float(index + 1))
        for index in range(num_states)
    ]
    model_noise_variances = (
        None if num_states == 1 else torch.tensor([0.2, 0.1])
    )
    estimator = ReEsNetChannelEstimator(
        resource_grid,
        model_states,
        model_noise_variances,
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

    channel_estimate, _ = assert_compiled_estimator_matches_eager(
        estimator,
        received_grid,
        torch.tensor(0.1, dtype=torch.float32),
        fullgraph=fullgraph,
    )
    torch.testing.assert_close(
        channel_estimate,
        torch.full_like(channel_estimate, complex(float(num_states), 0.0)),
    )


def test_reesnet_long_skip_bypasses_zero_body() -> None:
    """The post-body long skip carries input features to the upsampler."""
    model = ReEsNet(
        (3, 2),
        (12, 14),
        num_filters=2,
        num_res_blocks=2,
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.input_conv.weight[0, 0, 1, 1] = 1.0
        model.upsampling.weight[0, 0].fill_(1.0)
        model.output_conv.weight[0, 0, 1, 1] = 1.0

    inputs = torch.zeros(1, 2, 3, 2)
    inputs[0, 0] = torch.arange(6, dtype=torch.float32).reshape(3, 2)
    outputs = model(inputs)
    expected = inputs[:, :1].repeat_interleave(4, -2).repeat_interleave(7, -1)

    torch.testing.assert_close(outputs[:, :1], expected)
    torch.testing.assert_close(outputs[:, 1], torch.zeros_like(outputs[:, 1]))


@pytest.mark.system_variants
@pytest.mark.parametrize("system_variant", _SYSTEM_VARIANTS)
def test_super_resolution_dimensions_follow_system_geometry(
    system_variant: str,
) -> None:
    """ReEsNet derives output dimensions for three different OFDM grids."""
    cfg = system_variant_config(system_variant)
    resource_grid = build_resource_grid(cfg)
    input_shape, output_shape = _grid_shapes(resource_grid, cfg)
    model = ReEsNet(input_shape, output_shape)
    outputs = model(torch.randn(2, 2, *input_shape))

    assert outputs.shape == (2, 2, *output_shape)
    assert model.subcarrier_stride == output_shape[0] // input_shape[0]
    assert model.symbol_stride == output_shape[1] // input_shape[1]


@pytest.mark.system_variants
@pytest.mark.parametrize("system_variant", _SYSTEM_VARIANTS)
def test_active_pilot_grid_matches_configured_comb(system_variant: str) -> None:
    """Pilot images preserve each stream's ordered comb and OFDM symbols."""
    cfg = system_variant_config(system_variant)
    resource_grid = build_resource_grid(cfg)
    state = _linear_reesnet_state(resource_grid, cfg)
    estimator = ReEsNetChannelEstimator(
        resource_grid,
        [state],
        precision=cfg.precision,
        device=cfg.device,
    )

    assert estimator._pilot_group_num_groups == cfg.num_ue
    assert estimator._num_pilot_ofdm_symbols == len(
        cfg.pilot_ofdm_symbol_indices
    )
    assert estimator._num_pilot_subcarriers == cfg.fft_size // cfg.num_ue
    expected_symbols = torch.tensor(cfg.pilot_ofdm_symbol_indices)
    for group_index in range(estimator._pilot_group_num_groups):
        tx_index = int(estimator._group_tx_indices[group_index])
        expected_subcarriers = torch.arange(tx_index, cfg.fft_size, cfg.num_ue)
        torch.testing.assert_close(
            estimator._group_symbol_image[group_index, 0].cpu(),
            expected_symbols,
        )
        for symbol_column in range(estimator._num_pilot_ofdm_symbols):
            torch.testing.assert_close(
                estimator._group_subcarrier_image[
                    group_index, :, symbol_column
                ].cpu(),
                expected_subcarriers,
            )


@pytest.mark.system_variants
@pytest.mark.parametrize("system_variant", _SYSTEM_VARIANTS)
def test_pilot_processing_keeps_streams_and_receive_antennas_independent(
    system_variant: str,
) -> None:
    """Changing one stream/antenna pilot grid changes only that channel image."""
    cfg = system_variant_config(system_variant)
    resource_grid = build_resource_grid(cfg)
    estimator = ReEsNetChannelEstimator(
        resource_grid,
        [_linear_reesnet_state(resource_grid, cfg)],
        precision=cfg.precision,
        device=cfg.device,
    )
    target_group = estimator._pilot_group_num_groups - 1
    target_tx = int(estimator._group_tx_indices[target_group])
    target_stream = int(estimator._group_stream_indices[target_group])
    target_antenna = cfg.num_rx_ant - 1

    pilot_subcarriers = estimator._group_subcarrier_image[target_group]
    pilot_symbols = estimator._group_symbol_image[target_group]
    row_values = torch.arange(
        estimator._num_pilot_subcarriers,
        dtype=torch.float32,
    ).unsqueeze(1)
    column_values = torch.arange(
        estimator._num_pilot_ofdm_symbols,
        dtype=torch.float32,
    ).unsqueeze(0)
    pilot_channel = 1.0 + row_values + 0.01 * column_values
    compact_pilots = estimator._group_pilot_values[target_group].reshape(
        estimator._num_pilot_subcarriers,
        estimator._num_pilot_ofdm_symbols,
    )
    fft_subcarriers = torch.as_tensor(
        resource_grid.effective_subcarrier_ind,
        dtype=torch.int64,
    )[pilot_subcarriers]

    received_grid = torch.zeros(
        1,
        1,
        cfg.num_rx_ant,
        cfg.num_ofdm_symbols,
        cfg.fft_size,
        dtype=torch.complex64,
    )
    received_grid[
        0,
        0,
        target_antenna,
        pilot_symbols,
        fft_subcarriers,
    ] = compact_pilots * pilot_channel

    with torch.no_grad():
        h_hat, err_var = estimator(
            received_grid,
            torch.tensor(0.1, dtype=torch.float32),
        )
    validate_estimator_output(h_hat, err_var, torch.zeros_like(h_hat))

    estimated_pilots = h_hat[
        0,
        0,
        target_antenna,
        target_tx,
        target_stream,
        pilot_symbols,
        pilot_subcarriers,
    ]
    torch.testing.assert_close(
        estimated_pilots,
        pilot_channel.to(torch.complex64),
    )
    for antenna_index in range(cfg.num_rx_ant):
        for tx_index in range(cfg.num_ue):
            if antenna_index == target_antenna and tx_index == target_tx:
                continue
            torch.testing.assert_close(
                h_hat[0, 0, antenna_index, tx_index],
                torch.zeros_like(h_hat[0, 0, antenna_index, tx_index]),
            )
