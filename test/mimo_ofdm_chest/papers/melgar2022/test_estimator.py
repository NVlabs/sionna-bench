# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Focused architecture and model-selection tests for Melgar estimators."""

import pytest
import torch

from sionna.bench.mimo_ofdm_chest.benchmarking import build_resource_grid
from sionna.bench.mimo_ofdm_chest.papers.melgar2022.estimator import (
    DNN,
    HIDDEN_SIZES,
    DNNChannelEstimator,
    InterpolatedDNNChannelEstimator,
    METHOD_MODEL_2,
    METHOD_MODEL_3,
    _DNNBank,
)
from test.mimo_ofdm_chest.system_variants import (
    assert_compiled_estimator_matches_eager,
    system_variant_config,
)


def _constant_state(input_dim: int, value: float) -> dict[str, torch.Tensor]:
    """Return a valid DNN state with a constant output."""
    model = DNN(input_dim)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.net[-1].bias.fill_(value)
    return model.state_dict()


def test_dnn_matches_reviewed_hidden_architecture() -> None:
    """The shared real DNN uses the documented widths and linear endpoint."""
    input_dim = 7
    model = DNN(input_dim)
    linear_layers = [
        layer for layer in model.net if isinstance(layer, torch.nn.Linear)
    ]
    relu_layers = [
        layer for layer in model.net if isinstance(layer, torch.nn.ReLU)
    ]

    dimensions = [input_dim, *HIDDEN_SIZES, input_dim]
    assert [layer.in_features for layer in linear_layers] == dimensions[:-1]
    assert [layer.out_features for layer in linear_layers] == dimensions[1:]
    assert len(relu_layers) == len(HIDDEN_SIZES)
    assert isinstance(model.net[-1], torch.nn.Linear)


def test_dnn_bank_selects_state_for_current_noise_variance() -> None:
    """Per-SNR inference selects the state matching the runtime AWGN variance."""
    input_dim = 3
    bank = _DNNBank(
        input_dim,
        [_constant_state(input_dim, 1.0), _constant_state(input_dim, 2.0)],
        torch.tensor([1.0, 0.1]),
        noise_dtype=torch.float32,
        device=torch.device("cpu"),
    )
    values = torch.zeros(4, input_dim)

    torch.testing.assert_close(
        bank(values, torch.tensor(0.1)),
        torch.full_like(values, 2.0),
    )


@pytest.mark.compile
@pytest.mark.parametrize("method", [METHOD_MODEL_2, METHOD_MODEL_3])
@pytest.mark.parametrize(
    ("num_states", "fullgraph"),
    [(1, True), (2, False)],
    ids=["mixed-snr-fullgraph", "per-snr-selector-boundary"],
)
def test_trained_estimators_match_compiled_inference(
    method: str,
    num_states: int,
    fullgraph: bool,
) -> None:
    """Both complete trained-model paths agree in eager and compiled modes."""
    cfg = system_variant_config("baseline")
    resource_grid = build_resource_grid(cfg)
    if method == METHOD_MODEL_2:
        num_pilot_symbols = len(cfg.pilot_ofdm_symbol_indices)
        input_dim = int(
            torch.count_nonzero(resource_grid.pilot_pattern.pilots[0, 0]).item()
        ) // num_pilot_symbols
        estimator_class = DNNChannelEstimator
        estimator_kwargs = {}
    else:
        input_dim = 12
        estimator_class = InterpolatedDNNChannelEstimator
        estimator_kwargs = {"block_len": input_dim}

    model_states = [
        _constant_state(input_dim, float(state_index + 1))
        for state_index in range(num_states)
    ]
    model_noise_variances = (
        None if num_states == 1 else torch.tensor([0.1, 0.2])
    )
    estimator = estimator_class(
        resource_grid,
        model_states,
        model_noise_variances,
        precision=cfg.precision,
        device=cfg.device,
        **estimator_kwargs,
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


@pytest.mark.parametrize("method", [METHOD_MODEL_2, METHOD_MODEL_3])
def test_error_variance_scale_does_not_change_channel_estimate(method: str) -> None:
    """Detector confidence scaling leaves either DNN reconstruction unchanged."""
    cfg = system_variant_config("baseline")
    resource_grid = build_resource_grid(cfg)
    if method == METHOD_MODEL_2:
        num_pilot_symbols = len(cfg.pilot_ofdm_symbol_indices)
        input_dim = int(
            torch.count_nonzero(resource_grid.pilot_pattern.pilots[0, 0]).item()
        ) // num_pilot_symbols
        estimator_class = DNNChannelEstimator
        estimator_kwargs = {}
    else:
        input_dim = 12
        estimator_class = InterpolatedDNNChannelEstimator
        estimator_kwargs = {"block_len": input_dim}

    model_state = _constant_state(input_dim, 1.0)
    common_kwargs = {
        "model_states": [model_state],
        "precision": cfg.precision,
        "device": cfg.device,
        **estimator_kwargs,
    }
    unscaled_estimator = estimator_class(
        resource_grid,
        err_var_scale=1.0,
        **common_kwargs,
    )
    scaled_estimator = estimator_class(
        resource_grid,
        err_var_scale=2.0,
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
        scaled_channel, scaled_variance = scaled_estimator(
            received_grid,
            noise_variance,
        )

    torch.testing.assert_close(scaled_channel, unscaled_channel)
    torch.testing.assert_close(scaled_variance, 2.0 * unscaled_variance)
