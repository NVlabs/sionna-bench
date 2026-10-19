# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Untrained deep-prior channel estimator from Balevi et al., 2020."""

from __future__ import annotations

import math
from contextlib import contextmanager
from typing import Iterator, Optional

import torch
from sionna.phy import Precision
from sionna.phy.ofdm import (
    BasePilotChannelEstimator,
    BaseChannelInterpolator,
    LSChannelEstimator,
    ResourceGrid,
)
from sionna.phy.ofdm.channel_estimation import _PilotGroupMixin
from torch.func import functional_call, stack_module_state, vmap


class _DeepPriorDenoiser(torch.nn.Module):
    """Bias-free deep decoder used for LS pilot-grid denoising."""

    def __init__(
        self,
        *,
        hidden_channels: int,
        output_channels: int,
        output_size: tuple[int, int],
        num_hidden_layers: int,
    ) -> None:
        super().__init__()
        self.hidden_channels = int(hidden_channels)
        self.output_channels = int(output_channels)
        self.output_size = (int(output_size[0]), int(output_size[1]))
        self.num_hidden_layers = int(num_hidden_layers)
        self.upsampling_schedule = self._build_upsampling_schedule(
            self.output_size,
            self.num_hidden_layers,
        )

        self.hidden_convolutions = torch.nn.ModuleList(
            torch.nn.Conv2d(
                self.hidden_channels,
                self.hidden_channels,
                kernel_size=1,
                bias=False,
            )
            for _ in range(self.num_hidden_layers)
        )
        self.batch_norm_scales = torch.nn.ParameterList(
            torch.nn.Parameter(torch.ones(self.hidden_channels))
            for _ in range(self.num_hidden_layers)
        )
        self.batch_norm_offsets = torch.nn.ParameterList(
            torch.nn.Parameter(torch.zeros(self.hidden_channels))
            for _ in range(self.num_hidden_layers)
        )
        self.output_convolution = torch.nn.Conv2d(
            self.hidden_channels,
            self.output_channels,
            kernel_size=1,
            bias=False,
        )

    @staticmethod
    def _build_upsampling_schedule(
        output_size: tuple[int, int],
        num_hidden_layers: int,
    ) -> tuple[tuple[int, int], ...]:
        """Return deterministic bilinear upsampling sizes."""
        num_upsampling_layers = int(num_hidden_layers) - 1
        if num_upsampling_layers < 0:
            raise ValueError("`num_hidden_layers` must be positive.")
        frequency_size, time_size = int(output_size[0]), int(output_size[1])
        return tuple(
            (
                math.ceil(frequency_size / 2 ** (num_upsampling_layers - index)),
                math.ceil(time_size / 2 ** (num_upsampling_layers - index)),
            )
            for index in range(num_upsampling_layers + 1)
        )

    @property
    def input_size(self) -> tuple[int, int]:
        """Return the shape expected for the fixed random input."""
        return self.upsampling_schedule[0]

    def forward(self, fixed_input: torch.Tensor) -> torch.Tensor:
        """Map fixed random input to a denoised observation tensor."""
        features = fixed_input
        for layer_index, (convolution, scale, offset) in enumerate(
            zip(
                self.hidden_convolutions,
                self.batch_norm_scales,
                self.batch_norm_offsets,
            )
        ):
            features = convolution(features)
            if layer_index < self.num_hidden_layers - 1:
                features = torch.nn.functional.interpolate(
                    features,
                    size=self.upsampling_schedule[layer_index + 1],
                    mode="bilinear",
                    align_corners=False,
                )
            features = torch.relu(features)
            # Keep normalization statistics local to each vmapped network.
            mean = features.mean(dim=(0, 2, 3), keepdim=True)
            variance = features.var(dim=(0, 2, 3), correction=0, keepdim=True)
            features = (features - mean) * torch.rsqrt(variance + 1e-5)
            features = (
                features * scale[None, :, None, None]
                + offset[None, :, None, None]
            )
        return self.output_convolution(features)


class DeepPriorChannelEstimator(BasePilotChannelEstimator, _PilotGroupMixin):
    r"""Channel estimator that leverages an untrained neural network as prior,
    by fitting it to online to the noisy channel observations.

    This implementation follows the method proposed in:

    .. container:: bibliography-box

        .. bibliography::
            :filter: key == "balevi2020massive"
            :list: bullet

    .. rubric:: TL;DR

    The estimator first forms an LS estimate of the channel at the pilot
    locations and then uses a neural network as a prior by fitting it to the
    LS estimate for a fixed, random input. The corresponding fitted output is
    used as the denoised estimate of the channel at the pilot locations.

    .. rubric:: Supported System Geometry

    The estimator derives its network dimensions from the resource grid and
    supports different FFT sizes, OFDM-symbol counts, UE counts, and BS
    receive-antenna counts, provided that the following hold:

    - Active pilots for every transmit stream form a complete rectangular
      grid over their distinct OFDM subcarriers and OFDM symbols.
    - All streams have the same number of active pilot subcarriers.
    - All streams have the same number of active pilot-bearing OFDM symbols.

    .. rubric:: Deviations from the Paper

    - Unlike in the paper, the neural network is not fitted to the complete received
      resource grid because the random data symbols were found to destroy the required
      structure for the denoising task. Instead, the estimator forms an LS estimate only
      at the pilot locations and fits the untrained network to these LS estimates.
      This inversion of the paper's denoising-before-LS order was found to be material.
    - The paper assumes pilot observations on all subcarriers and does not define
      interpolation for Sionna's sparse Kronecker pilot pattern. This implementation
      has to be combined with a dedicated interpolator.
    - The paper's repeated bilinear upsampling by a factor of two is adapted to arbitrary
      pilot-grid dimensions using adapted intermediate sizes that result in the right target shape.
    - Our implementation uses a fixed learning rate and number of optimization steps
      whose default values were optimized using Optuna.
    - The paper does not specify detector-facing error variances, so the implementation
      returns ``err_var_scale * no``, where the default value of ``err_var_scale`` was
      selected using Optuna.

    :param resource_grid: Sionna resource grid defining the pilot pattern.
    :param interpolation_type: Sionna interpolation policy. Supported values
        are ``"nn"``, ``"lin"``, and ``"lin_time_avg"``. This argument is
        ignored when ``interpolator`` is provided.
    :param interpolator: Custom Sionna channel interpolator. If `None`, the
        interpolator selected by ``interpolation_type`` is used.
    :param hidden_channels: Hidden spatial-channel width ``k``.
    :param num_hidden_layers: Number of hidden layers in the deep-prior
        network.
    :param num_steps: Number of Adam steps used for per-grid runtime fitting.
        A value of zero bypasses denoising and is intended for diagnostics.
    :param learning_rate: Adam learning rate.
    :param input_noise_scale: Upper endpoint of the fixed network input sampled
        uniformly from ``[0, input_noise_scale)``.
    :param seed: Seed used for the fixed input and per-example weight
        initialization.
    :param err_var_scale: Scale applied to the detector-facing LS error
        variance fallback.
    :param precision: Precision used for internal calculations and outputs.
        If `None`, :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device for tensor operations. If `None`,
        :attr:`~sionna.phy.config.Config.device` is used.
    """

    def __init__(
        self,
        resource_grid: ResourceGrid,
        interpolation_type: str = "lin_time_avg",
        interpolator: Optional[BaseChannelInterpolator] = None,
        *,
        hidden_channels: int = 32,
        num_hidden_layers: int = 6,
        num_steps: int = 1597,
        learning_rate: float = 0.000698,
        input_noise_scale: float = 1.70,
        seed: int = 42,
        err_var_scale: float = 0.454,
        precision: Optional[Precision] = None,
        device: Optional[str] = None,
    ) -> None:
        """Initialize the fixed geometry and fitting hyperparameters."""
        assert isinstance(hidden_channels, int) and hidden_channels > 0, (
            "`hidden_channels` must be a positive integer."
        )
        assert isinstance(num_hidden_layers, int) and num_hidden_layers > 0, (
            "`num_hidden_layers` must be a positive integer."
        )
        assert isinstance(num_steps, int) and num_steps >= 0, (
            "`num_steps` must be a non-negative integer."
        )
        assert isinstance(learning_rate, (int, float)) and math.isfinite(
            learning_rate
        ), "`learning_rate` must be finite."
        assert learning_rate > 0.0, "`learning_rate` must be positive."
        assert isinstance(input_noise_scale, (int, float)) and math.isfinite(
            input_noise_scale
        ), "`input_noise_scale` must be finite."
        assert input_noise_scale > 0.0, "`input_noise_scale` must be positive."
        assert isinstance(seed, int), "`seed` must be an integer."
        assert isinstance(err_var_scale, (int, float)) and math.isfinite(
            err_var_scale
        ), "`err_var_scale` must be finite."
        assert err_var_scale >= 0.0, "`err_var_scale` must be non-negative."

        super().__init__(
            resource_grid=resource_grid,
            interpolation_type=interpolation_type,
            interpolator=interpolator,
            precision=precision,
            device=device,
        )
        self._hidden_channels = int(hidden_channels)
        self._num_hidden_layers = int(num_hidden_layers)
        self._num_steps = int(num_steps)
        self._learning_rate = float(learning_rate)
        self._input_noise_scale = float(input_noise_scale)
        self._seed = int(seed)
        self._err_var_scale = float(err_var_scale)
        self._num_effective_subcarriers = int(
            resource_grid.num_effective_subcarriers
        )

        self._init_pilot_groups(
            self._pilot_pattern,
            group_over=("subcarrier", "symbol"),
            active_only=True,
        )
        self._init_pilot_image_metadata()
        denoiser_shape = (
            self._num_pilot_subcarriers,
            self._num_pilot_ofdm_symbols,
        )
        input_size = _DeepPriorDenoiser._build_upsampling_schedule(
            denoiser_shape,
            self._num_hidden_layers,
        )[0]
        self.register_buffer(
            "_fixed_input",
            self._make_fixed_input(input_size),
        )

    def estimate_at_pilot_locations(
        self,
        y_pilots: torch.Tensor,
        no: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Estimate denoised channels at Sionna pilot locations.

        :param y_pilots: [batch_size, num_rx, num_rx_ant, num_tx, num_streams, num_pilot_symbols], `torch.complex`.
            Observed signals for the pilot-carrying resource elements.
        :param no: [batch_size, num_rx, num_rx_ant] or only the first n>=0 dims, `torch.float`.
            Variance of the AWGN.

        :output h_hat: [batch_size, num_rx, num_rx_ant, num_tx, num_streams, num_pilot_symbols], `torch.complex`.
            Channel estimates for the pilot-carrying resource elements.
        :output err_var: Same shape as ``h_hat``, `torch.float`.
            Channel estimation error variance for the pilot-carrying
            resource elements.
        """
        h_ls, err_var = LSChannelEstimator.estimate_at_pilot_locations(
            self,
            y_pilots,
            no,
        )
        if self._num_steps == 0:
            return h_ls, self._err_var_scale * err_var

        grouped_h_ls = self._gather_pilot_groups(h_ls)
        denoised_grouped_h = self._fit_ls_pilot_denoiser(grouped_h_ls)
        h_hat = self._scatter_pilot_groups(torch.zeros_like(h_ls), denoised_grouped_h)
        return h_hat, self._err_var_scale * err_var

    def _init_pilot_image_metadata(self) -> None:
        """Validate active pilot groups form rectangular frequency-time images."""
        group_subcarrier_indices = self._pilot_group_varying_coord("subcarrier")
        group_symbol_indices = self._pilot_group_varying_coord("symbol")
        num_pilot_subcarriers = None
        num_pilot_ofdm_symbols = None
        for group_index in range(self._pilot_group_num_groups):
            subcarriers = group_subcarrier_indices[group_index]
            symbols = group_symbol_indices[group_index]
            unique_subcarriers = torch.unique_consecutive(subcarriers)
            unique_symbols = torch.unique(symbols, sorted=True)
            if num_pilot_subcarriers is None:
                num_pilot_subcarriers = int(unique_subcarriers.numel())
                num_pilot_ofdm_symbols = int(unique_symbols.numel())
            assert int(unique_subcarriers.numel()) == num_pilot_subcarriers, (
                "Balevi pilot denoising requires equal active pilot-frequency "
                "counts for all streams."
            )
            assert int(unique_symbols.numel()) == num_pilot_ofdm_symbols, (
                "Balevi pilot denoising requires equal active pilot-symbol "
                "counts for all streams."
            )
            assert int(subcarriers.numel()) == (
                num_pilot_subcarriers * num_pilot_ofdm_symbols
            ), (
                "Balevi pilot denoising requires each active pilot group to "
                "form a rectangular subcarrier-by-symbol grid."
            )
            expected_subcarriers = unique_subcarriers.repeat_interleave(
                num_pilot_ofdm_symbols
            )
            expected_symbols = unique_symbols.repeat(num_pilot_subcarriers)
            assert bool(torch.all(subcarriers == expected_subcarriers)), (
                "Balevi pilot denoising expects pilot groups ordered as "
                "frequency-major rectangular images."
            )
            assert bool(torch.all(symbols == expected_symbols)), (
                "Balevi pilot denoising expects pilot groups ordered as "
                "frequency-major rectangular images."
            )

        assert num_pilot_subcarriers is not None and num_pilot_ofdm_symbols is not None
        self._num_pilot_subcarriers = num_pilot_subcarriers
        self._num_pilot_ofdm_symbols = num_pilot_ofdm_symbols

    def _make_fixed_input(self, input_size: tuple[int, int]) -> torch.Tensor:
        """Create the fixed random input ``Z0`` without perturbing global RNG."""
        with self._fork_rng(self._seed):
            fixed_input = torch.rand(
                1,
                self._hidden_channels,
                int(input_size[0]),
                int(input_size[1]),
                dtype=self.dtype,
                device=self.device,
            )
        return self._input_noise_scale * fixed_input

    @contextmanager
    def _fork_rng(self, seed: int) -> Iterator[None]:
        """Temporarily seed torch RNGs while restoring caller RNG state."""
        devices = []
        device = torch.device(self.device)
        if device.type == "cuda" and torch.cuda.is_available():
            devices.append(
                torch.cuda.current_device()
                if device.index is None
                else int(device.index)
            )
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(int(seed))
            if devices:
                torch.cuda.manual_seed_all(int(seed))
            yield

    def _fit_ls_pilot_denoiser(
        self,
        grouped_h_ls: torch.Tensor,
    ) -> torch.Tensor:
        """Fit untrained denoisers to active LS channel-observation grids."""
        batch_size, num_rx, num_rx_ant = grouped_h_ls.shape[:3]
        num_groups = int(grouped_h_ls.shape[-2])
        pilot_images = (
            grouped_h_ls.unflatten(
                -1,
                (
                    self._num_pilot_subcarriers,
                    self._num_pilot_ofdm_symbols,
                ),
            )
            .transpose(2, 3)
            .flatten(0, 2)
        )
        target = torch.cat([pilot_images.real, pilot_images.imag], dim=1)
        denoised_target = self._fit_real_channel_images(target)
        denoised_images = torch.complex(
            denoised_target[:, :num_rx_ant],
            denoised_target[:, num_rx_ant : 2 * num_rx_ant],
        )
        return (
            denoised_images.unflatten(0, (batch_size, num_rx, num_groups))
            .transpose(2, 3)
            .flatten(-2)
        )

    @torch.compiler.disable
    def _fit_real_channel_images(self, target: torch.Tensor) -> torch.Tensor:
        """Fit independent denoisers inside the explicit eager boundary."""
        num_examples = int(target.shape[0])
        output_size = (int(target.shape[-2]), int(target.shape[-1]))
        output_channels = int(target.shape[1])
        denoisers = []
        for example_index in range(num_examples):
            with self._fork_rng(self._seed + example_index):
                denoiser = _DeepPriorDenoiser(
                    hidden_channels=self._hidden_channels,
                    output_channels=output_channels,
                    output_size=output_size,
                    num_hidden_layers=self._num_hidden_layers,
                ).to(device=target.device, dtype=target.dtype)
            denoiser.train()
            denoisers.append(denoiser)

        parameters, buffers = stack_module_state(denoisers)
        template = denoisers[0]
        optimizer = torch.optim.Adam(
            parameters.values(),
            lr=self._learning_rate,
        )
        fixed_input = self._fixed_input.to(
            device=target.device,
            dtype=target.dtype,
        ).expand(num_examples, -1, -1, -1, -1)

        def forward_one(parameter, buffer, network_input):
            return functional_call(
                template,
                (parameter, buffer),
                (network_input,),
            )

        forward_all = vmap(forward_one, in_dims=(0, 0, 0))
        with torch.enable_grad():
            for _ in range(self._num_steps):
                optimizer.zero_grad(set_to_none=True)
                fitted = forward_all(parameters, buffers, fixed_input).squeeze(1)
                loss = (fitted - target).square().flatten(1).mean(1).sum()
                loss.backward()
                optimizer.step()

        with torch.no_grad():
            return (
                forward_all(parameters, buffers, fixed_input)
                .squeeze(1)
                .detach()
            )

__all__ = ["DeepPriorChannelEstimator"]
