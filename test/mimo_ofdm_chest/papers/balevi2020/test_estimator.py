# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import math

import pytest
import torch
from sionna.phy import config as sionna_config
from sionna.phy.ofdm import (
    LSChannelEstimator,
    NearestNeighborInterpolator,
    PilotPattern,
    ResourceGrid,
)

from sionna.bench.mimo_ofdm_chest.benchmarking.communication_system import OFDMChannelEstimationSystem, build_resource_grid
from sionna.bench.mimo_ofdm_chest.papers.balevi2020.estimator import (
    DeepPriorChannelEstimator,
    _DeepPriorDenoiser,
)
from test.mimo_ofdm_chest.system_variants import (
    assert_compiled_estimator_matches_eager,
    run_nmse_probe,
    system_variant_config,
)


def _resource_grid(
    *,
    fft_size: int = 16,
    num_ofdm_symbols: int = 4,
    pilot_ofdm_symbol_indices: tuple[int, ...] = (1, 3),
) -> ResourceGrid:
    return ResourceGrid(
        num_ofdm_symbols=num_ofdm_symbols,
        fft_size=fft_size,
        subcarrier_spacing=15e3,
        num_tx=1,
        num_streams_per_tx=1,
        cyclic_prefix_length=0,
        pilot_pattern="kronecker",
        pilot_ofdm_symbol_indices=list(pilot_ofdm_symbol_indices),
    )


def test_deep_prior_denoiser_matches_requested_output_size():
    model = _DeepPriorDenoiser(
        hidden_channels=4,
        output_channels=6,
        output_size=(13, 5),
        num_hidden_layers=4,
    )
    fixed_input = torch.randn(2, 4, *model.input_size)

    output = model(fixed_input)

    assert model.upsampling_schedule == ((2, 1), (4, 2), (7, 3), (13, 5))
    assert output.shape == (2, 6, 13, 5)


@pytest.mark.compile
def test_deep_prior_denoiser_compiles_as_full_graph() -> None:
    """The tensor-only denoiser forward agrees in eager and full-graph modes."""
    model = _DeepPriorDenoiser(
        hidden_channels=4,
        output_channels=6,
        output_size=(13, 5),
        num_hidden_layers=4,
    )
    fixed_input = torch.randn(2, 4, *model.input_size)

    with torch.no_grad():
        eager_output = model(fixed_input)
        compiled_output = torch.compile(
            model,
            backend="eager",
            fullgraph=True,
        )(fixed_input)

    torch.testing.assert_close(compiled_output, eager_output)


@pytest.mark.compile
def test_complete_estimator_compiles_around_eager_fitting_boundary() -> None:
    """The complete call compiles while observation-specific fitting stays eager."""
    resource_grid = _resource_grid()
    estimator = DeepPriorChannelEstimator(
        resource_grid,
        hidden_channels=2,
        num_hidden_layers=2,
        num_steps=1,
        seed=17,
        precision="single",
        device="cpu",
    )
    y = torch.randn(1, 1, 2, 4, 16, dtype=torch.complex64)
    no = torch.tensor(0.1, dtype=torch.float32)

    assert_compiled_estimator_matches_eager(estimator, y, no)


@pytest.mark.parametrize(
    ("hidden_channels", "output_channels", "expected_num_parameters"),
    [
        pytest.param(8, 2, 496, id="paper-siso-k8"),
        pytest.param(16, 128, 3776, id="paper-massive-mimo-k16"),
    ],
)
def test_deep_prior_parameter_count_matches_paper(
    hidden_channels: int,
    output_channels: int,
    expected_num_parameters: int,
) -> None:
    """Bias-free convolutions reproduce the paper's parameter counts."""
    model = _DeepPriorDenoiser(
        hidden_channels=hidden_channels,
        output_channels=output_channels,
        output_size=(64, 64),
        num_hidden_layers=6,
    )

    assert sum(parameter.numel() for parameter in model.parameters()) == (
        expected_num_parameters
    )
    assert all(
        convolution.bias is None for convolution in model.hidden_convolutions
    )
    assert model.output_convolution.bias is None


def test_hidden_normalization_uses_batch_one_channel_statistics() -> None:
    """Each hidden channel is normalized over its frequency-time grid."""
    model = _DeepPriorDenoiser(
        hidden_channels=4,
        output_channels=2,
        output_size=(8, 4),
        num_hidden_layers=2,
    )
    with torch.no_grad():
        identity = torch.eye(4).reshape(4, 4, 1, 1)
        for convolution in model.hidden_convolutions:
            convolution.weight.copy_(identity)

    captured_features = []
    handle = model.output_convolution.register_forward_pre_hook(
        lambda module, inputs: captured_features.append(inputs[0].detach())
    )
    fixed_input = torch.linspace(
        0.1,
        1.0,
        4 * math.prod(model.input_size),
    ).reshape(1, 4, *model.input_size)
    model(fixed_input)
    handle.remove()

    hidden_features = captured_features[0]
    channel_means = hidden_features.mean(dim=(0, 2, 3))
    channel_variances = hidden_features.var(dim=(0, 2, 3), correction=0)
    torch.testing.assert_close(
        channel_means,
        torch.zeros_like(channel_means),
        atol=1e-5,
        rtol=0.0,
    )
    torch.testing.assert_close(
        channel_variances,
        torch.ones_like(channel_variances),
        atol=2e-4,
        rtol=2e-4,
    )


def test_fixed_input_is_seeded_registered_and_rng_neutral() -> None:
    """The uniform ``Z0`` buffer is repeatable without changing global RNG."""
    resource_grid = _resource_grid()
    torch.manual_seed(123)
    initial_rng_state = torch.random.get_rng_state()

    first = DeepPriorChannelEstimator(
        resource_grid,
        hidden_channels=3,
        num_hidden_layers=2,
        num_steps=0,
        seed=17,
        precision="single",
        device="cpu",
    )
    assert torch.equal(torch.random.get_rng_state(), initial_rng_state)
    second = DeepPriorChannelEstimator(
        resource_grid,
        hidden_channels=3,
        num_hidden_layers=2,
        num_steps=0,
        seed=17,
        precision="single",
        device="cpu",
    )
    third = DeepPriorChannelEstimator(
        resource_grid,
        hidden_channels=3,
        num_hidden_layers=2,
        num_steps=0,
        seed=18,
        precision="single",
        device="cpu",
    )

    assert "_fixed_input" in dict(first.named_buffers())
    torch.testing.assert_close(first._fixed_input, second._fixed_input)
    assert not torch.allclose(first._fixed_input, third._fixed_input)
    assert float(first._fixed_input.min()) >= 0.0
    assert float(first._fixed_input.max()) < first._input_noise_scale


@pytest.mark.training
def test_runtime_fitting_reduces_squared_error_objective() -> None:
    """Adam fitting descends the normalized form of Equation (16)."""
    resource_grid = _resource_grid()
    target = torch.randn(2, 4, 16, 2)
    estimator_kwargs = {
        "hidden_channels": 4,
        "num_hidden_layers": 2,
        "learning_rate": 0.01,
        "seed": 17,
        "precision": "single",
        "device": "cpu",
    }
    initial_estimator = DeepPriorChannelEstimator(
        resource_grid,
        num_steps=0,
        **estimator_kwargs,
    )
    fitted_estimator = DeepPriorChannelEstimator(
        resource_grid,
        num_steps=5,
        **estimator_kwargs,
    )

    initial_output = initial_estimator._fit_real_channel_images(target)
    fitted_output = fitted_estimator._fit_real_channel_images(target)
    initial_loss = (initial_output - target).square().mean()
    fitted_loss = (fitted_output - target).square().mean()

    assert fitted_loss < initial_loss


@pytest.mark.parametrize(
    (
        "num_examples",
        "num_rx_ant",
        "fft_size",
        "pilot_ofdm_symbol_indices",
    ),
    [
        pytest.param(2, 1, 8, (1,), id="two-images-2x8x1"),
        pytest.param(3, 2, 12, (1, 3), id="three-images-4x12x2"),
        pytest.param(4, 3, 16, (0, 2, 4), id="four-images-6x16x3"),
    ],
)
@pytest.mark.training
def test_batched_fitting_matches_independent_optimization(
    num_examples: int,
    num_rx_ant: int,
    fft_size: int,
    pilot_ofdm_symbol_indices: tuple[int, ...],
) -> None:
    """Batched parameters, normalization, and Adam state remain independent."""
    resource_grid = _resource_grid(
        fft_size=fft_size,
        num_ofdm_symbols=max(pilot_ofdm_symbol_indices) + 1,
        pilot_ofdm_symbol_indices=pilot_ofdm_symbol_indices,
    )
    estimator = DeepPriorChannelEstimator(
        resource_grid,
        hidden_channels=3,
        num_hidden_layers=2,
        num_steps=3,
        seed=17,
        precision="single",
        device="cpu",
    )
    target = torch.randn(
        num_examples,
        2 * num_rx_ant,
        fft_size,
        len(pilot_ofdm_symbol_indices),
    )

    batched_output = estimator._fit_real_channel_images(target)
    initial_seed = estimator._seed
    separate_outputs = []
    for example_index in range(num_examples):
        estimator._seed = initial_seed + example_index
        separate_outputs.append(
            estimator._fit_real_channel_images(
                target[example_index : example_index + 1]
            )
        )
    estimator._seed = initial_seed
    separate_output = torch.cat(separate_outputs, dim=0)

    torch.testing.assert_close(batched_output, separate_output)

    perturbed_target = target.clone()
    perturbed_target[0] = perturbed_target[0] + 1.0
    perturbed_output = estimator._fit_real_channel_images(perturbed_target)

    assert not torch.allclose(perturbed_output[0], batched_output[0])
    torch.testing.assert_close(perturbed_output[1:], batched_output[1:])


@pytest.mark.parametrize(
    "interpolation_type",
    ["nn", "lin", "lin_time_avg"],
)
def test_zero_step_path_matches_sionna_ls_interpolation(
    interpolation_type: str,
) -> None:
    """The diagnostic bypass preserves each built-in Sionna LS policy."""
    sionna_config.device = "cpu"
    sionna_config.precision = "single"
    resource_grid = _resource_grid()
    y = torch.randn(2, 1, 2, 4, 16, dtype=torch.complex64)
    no = torch.tensor(0.1, dtype=torch.float32)

    estimator = DeepPriorChannelEstimator(
        resource_grid,
        hidden_channels=2,
        num_hidden_layers=2,
        num_steps=0,
        interpolation_type=interpolation_type,
        err_var_scale=1.0,
        precision="single",
        device="cpu",
    )
    baseline = LSChannelEstimator(
        resource_grid,
        interpolation_type=interpolation_type,
        precision="single",
        device="cpu",
    )

    h_hat, err_var = estimator(y, no)
    expected_h_hat, expected_err_var = baseline(y, no)

    torch.testing.assert_close(h_hat, expected_h_hat)
    torch.testing.assert_close(err_var, expected_err_var)


def test_custom_interpolator_overrides_interpolation_type() -> None:
    """A supplied interpolator takes precedence over the built-in policy."""
    resource_grid = _resource_grid()
    y = torch.randn(2, 1, 2, 4, 16, dtype=torch.complex64)
    no = torch.tensor(0.1, dtype=torch.float32)
    interpolator = NearestNeighborInterpolator(resource_grid.pilot_pattern)
    estimator = DeepPriorChannelEstimator(
        resource_grid,
        interpolation_type="lin_time_avg",
        interpolator=interpolator,
        hidden_channels=2,
        num_hidden_layers=2,
        num_steps=0,
        err_var_scale=1.0,
        precision="single",
        device="cpu",
    )
    baseline = LSChannelEstimator(
        resource_grid,
        interpolation_type="nn",
        precision="single",
        device="cpu",
    )

    h_hat, err_var = estimator(y, no)
    expected_h_hat, expected_err_var = baseline(y, no)

    torch.testing.assert_close(h_hat, expected_h_hat)
    torch.testing.assert_close(err_var, expected_err_var)


def test_identity_fitted_path_preserves_ls_pilot_estimates() -> None:
    """An identity denoiser leaves the LS-first benchmark adaptation unchanged."""
    resource_grid = _resource_grid()
    y = torch.randn(2, 1, 2, 4, 16, dtype=torch.complex64)
    no = torch.tensor(0.1, dtype=torch.float32)
    estimator = DeepPriorChannelEstimator(
        resource_grid,
        hidden_channels=2,
        num_hidden_layers=2,
        num_steps=1,
        err_var_scale=1.0,
        precision="single",
        device="cpu",
    )
    estimator._fit_real_channel_images = lambda target: target
    baseline = LSChannelEstimator(
        resource_grid,
        interpolation_type="lin_time_avg",
        precision="single",
        device="cpu",
    )

    h_hat, err_var = estimator(y, no)
    expected_h_hat, expected_err_var = baseline(y, no)

    torch.testing.assert_close(h_hat, expected_h_hat)
    torch.testing.assert_close(err_var, expected_err_var)


def test_error_variance_scale_changes_only_detector_confidence() -> None:
    """Confidence scaling changes LS variance without changing the channel."""
    resource_grid = _resource_grid()
    y = torch.randn(2, 1, 2, 4, 16, dtype=torch.complex64)
    no = torch.tensor(0.1, dtype=torch.float32)
    common_kwargs = {
        "hidden_channels": 2,
        "num_hidden_layers": 2,
        "num_steps": 0,
        "precision": "single",
        "device": "cpu",
    }
    calibrated_scale = 0.25
    calibrated = DeepPriorChannelEstimator(
        resource_grid,
        err_var_scale=calibrated_scale,
        **common_kwargs,
    )
    unscaled = DeepPriorChannelEstimator(
        resource_grid,
        err_var_scale=1.0,
        **common_kwargs,
    )

    calibrated_channel, calibrated_variance = calibrated(y, no)
    unscaled_channel, unscaled_variance = unscaled(y, no)

    torch.testing.assert_close(calibrated_channel, unscaled_channel)
    torch.testing.assert_close(
        calibrated_variance,
        calibrated_scale * unscaled_variance,
    )


@pytest.mark.training
def test_fitted_pilot_path_is_finite():
    sionna_config.device = "cpu"
    sionna_config.precision = "single"
    resource_grid = _resource_grid()
    y = torch.randn(1, 1, 2, 4, 16, dtype=torch.complex64)
    no = torch.tensor(0.1, dtype=torch.float32)

    estimator = DeepPriorChannelEstimator(
        resource_grid,
        hidden_channels=2,
        num_hidden_layers=2,
        num_steps=1,
        precision="single",
        device="cpu",
    )

    h_hat, err_var = estimator(y, no)

    assert h_hat.shape == (1, 1, 2, 1, 1, 4, 16)
    assert err_var.shape == h_hat.shape
    assert torch.isfinite(h_hat.real).all()
    assert torch.isfinite(h_hat.imag).all()
    assert torch.isfinite(err_var).all()
    assert h_hat.dtype == torch.complex64
    assert err_var.dtype == torch.float32
    assert h_hat.device.type == "cpu"
    assert err_var.device.type == "cpu"
    assert torch.all(err_var >= 0.0)


def test_nonrectangular_pilot_grid_is_rejected() -> None:
    """An incomplete frequency-time pilot rectangle fails explicitly."""
    mask = torch.zeros(1, 1, 4, 8, dtype=torch.int32)
    mask[0, 0, 1, [0, 2]] = 1
    mask[0, 0, 3, 0] = 1
    pilot_pattern = PilotPattern(
        mask,
        torch.ones(1, 1, 3, dtype=torch.complex64),
        precision="single",
        device="cpu",
    )
    resource_grid = ResourceGrid(
        num_ofdm_symbols=4,
        fft_size=8,
        subcarrier_spacing=15e3,
        num_tx=1,
        num_streams_per_tx=1,
        cyclic_prefix_length=0,
        pilot_pattern=pilot_pattern,
    )

    with pytest.raises(
        AssertionError,
        match="rectangular subcarrier-by-symbol grid",
    ):
        DeepPriorChannelEstimator(
            resource_grid,
            hidden_channels=2,
            num_hidden_layers=2,
            num_steps=1,
            precision="single",
            device="cpu",
        )


@pytest.mark.system_variants
@pytest.mark.training
@pytest.mark.parametrize(
    (
        "system_variant",
        "expected_fft_size",
        "expected_num_ue",
        "expected_num_rx_ant",
        "expected_pilot_grid_shape",
    ),
    [
        pytest.param("baseline", 48, 1, 2, (48, 1), id="grid48-ue1-rx2"),
        pytest.param("alternate", 72, 3, 4, (24, 2), id="grid72-ue3-rx4"),
        pytest.param("wide", 96, 4, 12, (24, 3), id="grid96-ue4-rx12"),
    ],
)
def test_fitted_path_supports_changed_system_dimensions(
    system_variant: str,
    expected_fft_size: int,
    expected_num_ue: int,
    expected_num_rx_ant: int,
    expected_pilot_grid_shape: tuple[int, int],
) -> None:
    """One-step fitting remains finite and nonzero across system variants."""
    cfg = system_variant_config(system_variant)
    resource_grid = build_resource_grid(cfg)
    estimator = DeepPriorChannelEstimator(
        resource_grid,
        hidden_channels=2,
        num_hidden_layers=2,
        num_steps=1,
        seed=cfg.seed,
        precision=cfg.precision,
        device=cfg.device,
    )

    assert cfg.fft_size == expected_fft_size
    assert cfg.num_ue == expected_num_ue
    assert cfg.num_rx_ant == expected_num_rx_ant
    assert (
        estimator._num_pilot_subcarriers,
        estimator._num_pilot_ofdm_symbols,
    ) == expected_pilot_grid_shape

    system = OFDMChannelEstimationSystem(cfg, estimator, resource_grid)
    result = run_nmse_probe(system, snr_db=5.0)

    assert math.isfinite(result.nmse)
    assert result.nmse >= 0.0
    assert not math.isclose(result.nmse, result.zero_estimate_nmse)
