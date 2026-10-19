# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Full-grid ReEsNet estimator adapted from Li et al., 2020."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Optional

import torch
from sionna.phy import Precision
from sionna.phy.ofdm import BaseChannelEstimator, LSChannelEstimator, ResourceGrid
from sionna.phy.utils import flatten_last_dims
from sionna.phy.ofdm.channel_estimation import _PilotGroupMixin


class ResidualBlock(torch.nn.Module):
    """Two-layer residual block used by ReEsNet."""

    def __init__(self, num_filters: int) -> None:
        super().__init__()
        self.conv1 = torch.nn.Conv2d(
            num_filters,
            num_filters,
            kernel_size=3,
            padding=1,
        )
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(
            num_filters,
            num_filters,
            kernel_size=3,
            padding=1,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply the residual block."""
        residual = inputs
        outputs = self.conv1(inputs)
        outputs = self.relu(outputs)
        outputs = self.conv2(outputs)
        return outputs + residual


class ReEsNet(torch.nn.Module):
    r"""Residual super-resolution network from Li et al., 2020.

    :param input_shape: Input pilot-image shape ``(N_pf, N_pn)``.
    :param output_shape: Full-grid image shape
        ``(num_effective_subcarriers, num_ofdm_symbols)``.
    :param num_filters: Number of hidden convolutional filters.
    :param num_res_blocks: Number of residual blocks.

    :input inputs: ``[batch_size, 2, N_pf, N_pn]``, `torch.float`.
        Real and imaginary LS pilot images.

    :output outputs:
        ``[batch_size, 2, num_effective_subcarriers, num_ofdm_symbols]``,
        `torch.float`.
    """

    def __init__(
        self,
        input_shape: tuple[int, int],
        output_shape: tuple[int, int],
        *,
        num_filters: int = 16,
        num_res_blocks: int = 4,
    ) -> None:
        super().__init__()
        assert num_filters > 0, "`num_filters` must be positive."
        assert num_res_blocks > 0, "`num_res_blocks` must be positive."
        input_subcarriers, input_symbols = [int(x) for x in input_shape]
        output_subcarriers, output_symbols = [int(x) for x in output_shape]
        assert input_subcarriers > 0 and input_symbols > 0, (
            "`input_shape` entries must be positive."
        )
        assert output_subcarriers > 0 and output_symbols > 0, (
            "`output_shape` entries must be positive."
        )
        assert output_subcarriers % input_subcarriers == 0, (
            "ReEsNet transposed convolution requires integer subcarrier upsampling."
        )
        assert output_symbols % input_symbols == 0, (
            "ReEsNet transposed convolution requires integer OFDM-symbol upsampling."
        )

        self.input_shape = (input_subcarriers, input_symbols)
        self.output_shape = (output_subcarriers, output_symbols)
        self.subcarrier_stride = output_subcarriers // input_subcarriers
        self.symbol_stride = output_symbols // input_symbols

        self.input_conv = torch.nn.Conv2d(2, num_filters, kernel_size=3, padding=1)
        self.blocks = torch.nn.Sequential(
            *[ResidualBlock(num_filters) for _ in range(num_res_blocks)]
        )
        self.post_blocks_conv = torch.nn.Conv2d(
            num_filters,
            num_filters,
            kernel_size=3,
            padding=1,
        )
        # Li et al. replace EDSR's pixel shuffle with an anisotropic
        # transposed convolution but do not specify its kernel geometry. The
        # minimal kernel-equals-stride form gives the required integer scaling
        # without overlap; the surrounding convolutions still mix neighbors.
        self.upsampling = torch.nn.ConvTranspose2d(
            num_filters,
            num_filters,
            kernel_size=(self.subcarrier_stride, self.symbol_stride),
            stride=(self.subcarrier_stride, self.symbol_stride),
        )
        self.output_conv = torch.nn.Conv2d(num_filters, 2, kernel_size=3, padding=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply ReEsNet to pilot images."""
        outputs = self.input_conv(inputs)
        input_features = outputs
        outputs = self.blocks(outputs)
        outputs = self.post_blocks_conv(outputs)
        # EDSR/ReEsNet global feature skip shown in Figure 2 of both papers.
        outputs = outputs + input_features
        outputs = self.upsampling(outputs)
        return self.output_conv(outputs)


class ReEsNetChannelEstimator(BaseChannelEstimator, _PilotGroupMixin):
    r"""Channel estimator that leverages a residual super-resolution network (ReEsNet) for denoising.

    This estimator follows the method proposed in:

    .. container:: bibliography-box

        .. bibliography::
            :filter: key == "li2019deep"
            :list: bullet

    .. rubric:: TL;DR

    This estimator models OFDM channel estimation as image super-resolution. The
    estimator first computes LS estimates at the active pilot resource elements
    and arranges their real and imaginary parts as a two-channel pilot grid. A
    trained residual convolutional network then reconstructs the complete
    frequency-time channel grid directly, without a separate interpolator.

    Each receive-antenna and transmit-stream channel is processed as an independent
    network input. The same trained network weights are shared across these inputs.

    .. rubric:: Supported System Geometry

    - The active nonzero pilots of every transmit stream form a complete rectangular
      subcarrier-by-symbol grid.
    - All transmit streams have pilot grids with the same dimensions.
    - The number of effective subcarriers is an integer multiple of the number of
      active pilot subcarriers.
    - The number of OFDM symbols is an integer multiple of the number of active
      pilot-bearing OFDM symbols.
    - Every prepared ReEsNet state matches the resulting pilot-grid input shape,
      full-grid output shape, and network architecture.

    UE streams and receive antennas are processed independently, so changing their
    counts does not change the neural-network dimensions. Assets loaded through the
    paper's standard asset loader must nevertheless match the complete recorded
    system and pilot configuration.

    .. rubric:: Deviations from the Paper

    - The paper evaluates single-antenna downlink channel estimation. This
      implementation applies the same network independently to every uplink
      transmit-stream and receive-antenna channel and shares its weights across
      them.
    - This implementation derives the input and output dimensions from the supplied
      Sionna resource grid instead of reusing the paper's grid geometry.
    - The paper does not specify the transposed convolution's kernel size, padding,
      or output padding. This implementation uses ``kernel_size=stride`` with zero
      padding.
    - The paper trains one network across several SNRs. This implementation supports
      both such a mixed-SNR state and a bank of independently trained SNR-specific
      tates selected from the input noise variance.
    - The paper does not define detector-facing channel-estimation error variances.
      This implementation bilinearly upsamples the LS pilot error-variance grid and
      applies the configurable ``err_var_scale``.

    :param resource_grid: Sionna resource grid.
    :param model_states: Prepared ReEsNet state dictionaries. Each state must
        contain the tensors for a ReEsNet whose pilot-image input and full-grid
        output dimensions match ``resource_grid``.
    :param model_noise_variances: Positive AWGN variance associated with each
        state, shape ``[num_models]``. These values select SNR-specific states.
        If `None`, exactly one mixed-SNR state must be supplied.
    :param err_var_scale: Scale applied to the detector-facing variance
        fallback.
    :param precision: Sionna precision. If `None`, the global Sionna precision
        is used.
    :param device: Torch device. If `None`, the global Sionna device is used.
    """

    def __init__(
        self,
        resource_grid: ResourceGrid,
        model_states: Sequence[Mapping[str, torch.Tensor]],
        model_noise_variances: Optional[torch.Tensor] = None,
        *,
        err_var_scale: float = 0.234,
        precision: Optional[Precision] = None,
        device: Optional[str] = None,
    ) -> None:
        """Initialize pilot-image metadata and install prepared weights."""
        assert len(model_states) > 0, "At least one ReEsNet state is required."
        assert err_var_scale >= 0.0, "`err_var_scale` must be non-negative."

        super().__init__(
            resource_grid=resource_grid,
            precision=precision,
            device=device,
        )
        self._err_var_scale = float(err_var_scale)
        self._num_effective_subcarriers = int(
            resource_grid.num_effective_subcarriers
        )
        self._num_ofdm_symbols = int(resource_grid.num_ofdm_symbols)

        self._init_pilot_groups(
            self._pilot_pattern,
            group_over=("subcarrier", "symbol"),
        )
        self._init_pilot_image_metadata(resource_grid)
        self._models = torch.nn.ModuleList(
            self._build_model(state) for state in model_states
        )
        if model_noise_variances is None:
            assert len(model_states) == 1, (
                "Multiple ReEsNet states require `model_noise_variances`."
            )
            noise_variances = torch.empty(
                0,
                dtype=self.dtype,
                device=self.device,
            )
        else:
            noise_variances = torch.as_tensor(
                model_noise_variances,
                dtype=self.dtype,
                device=self.device,
            ).flatten()
            assert int(noise_variances.numel()) == len(model_states), (
                "`model_noise_variances` must have one entry per model state."
            )
            assert bool(torch.all(noise_variances > 0.0)), (
                "`model_noise_variances` must be positive."
            )
        self.register_buffer("_model_noise_variances", noise_variances)

    def _build_model(self, state: Mapping[str, torch.Tensor]) -> ReEsNet:
        """Build a ReEsNet instance and load a cached state dictionary."""
        model = ReEsNet(
            (self._num_pilot_subcarriers, self._num_pilot_ofdm_symbols),
            (self._num_effective_subcarriers, self._num_ofdm_symbols),
        ).to(device=self.device, dtype=self.dtype)
        model.load_state_dict(state)
        model.eval()
        return model

    @torch.compiler.disable
    def _select_model_index(self, no: torch.Tensor) -> int:
        """Select one prepared SNR-specific state outside compiled inference."""
        noise_variance = no.to(
            dtype=self._model_noise_variances.dtype,
            device=self._model_noise_variances.device,
        ).reshape(-1).mean()
        log_distance = torch.abs(
            torch.log(noise_variance) - torch.log(self._model_noise_variances)
        )
        model_index = int(torch.argmin(log_distance).item())
        assert bool(log_distance[model_index] < 1e-3), (
            "No SNR-specific Li 2020 ReEsNet is available for the current "
            "noise variance."
        )
        return model_index

    def _init_pilot_image_metadata(self, resource_grid) -> None:
        """Register pilot-image coordinates for the active Kronecker layout."""
        del resource_grid
        group_subcarriers = self._pilot_group_varying_coord("subcarrier")
        group_symbols = self._pilot_group_varying_coord("symbol")
        group_tx_indices = self._pilot_group_constant_coord("tx")
        group_stream_indices = self._pilot_group_constant_coord("stream")
        num_stream_groups = int(self._pilot_pattern.num_tx) * int(
            self._pilot_pattern.num_streams_per_tx
        )
        assert self._pilot_group_num_groups == num_stream_groups, (
            "ReEsNet requires one active rectangular pilot grid per stream."
        )

        unique_subcarriers = torch.unique(group_subcarriers[0], sorted=True)
        unique_symbols = torch.unique(group_symbols[0], sorted=True)
        num_pilot_subcarriers = int(unique_subcarriers.numel())
        num_pilot_ofdm_symbols = int(unique_symbols.numel())
        assert (
            num_pilot_subcarriers * num_pilot_ofdm_symbols == self._pilot_group_len
        ), "Active pilots must form a rectangular ReEsNet input image."

        subcarrier_image = group_subcarriers.reshape(
            self._pilot_group_num_groups,
            num_pilot_subcarriers,
            num_pilot_ofdm_symbols,
        )
        symbol_image = group_symbols.reshape(
            self._pilot_group_num_groups,
            num_pilot_subcarriers,
            num_pilot_ofdm_symbols,
        )
        assert bool(torch.all(subcarrier_image == subcarrier_image[:, :, :1])), (
            "ReEsNet expects the pilot-image rows to have fixed subcarriers."
        )
        assert bool(torch.all(symbol_image == symbol_image[:, :1, :])), (
            "ReEsNet expects the pilot-image columns to have fixed OFDM symbols."
        )

        pilot_values = self._gather_pilot_groups(
            self._pilot_pattern.pilots.to(self.device)
        ).to(dtype=self.cdtype, device=self.device)
        assert bool(torch.all(pilot_values.abs() > 0.0)), (
            "ReEsNet input groups must contain only nonzero pilot symbols."
        )

        self._num_pilot_subcarriers = num_pilot_subcarriers
        self._num_pilot_ofdm_symbols = num_pilot_ofdm_symbols
        self.register_buffer("_group_subcarrier_image", subcarrier_image)
        self.register_buffer("_group_symbol_image", symbol_image)
        self.register_buffer("_group_tx_indices", group_tx_indices)
        self.register_buffer("_group_stream_indices", group_stream_indices)
        self.register_buffer("_group_pilot_values", pilot_values)
        self.register_buffer(
            "_group_pilot_power",
            pilot_values.abs().square().to(dtype=self.dtype),
        )

    def _gather_y_pilots(self, y: torch.Tensor) -> torch.Tensor:
        """Gather Sionna compact pilot observations from ``y``."""
        y_eff = self._remove_nulled_scs(y)
        y_eff_flat = flatten_last_dims(y_eff)
        pilot_ind = self._pilot_ind.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        pilot_ind = pilot_ind.expand(
            y_eff_flat.shape[0],
            y_eff_flat.shape[1],
            y_eff_flat.shape[2],
            -1,
            -1,
            -1,
        )
        return torch.gather(
            y_eff_flat.unsqueeze(3)
            .unsqueeze(4)
            .expand(-1, -1, -1, pilot_ind.shape[3], pilot_ind.shape[4], -1),
            -1,
            pilot_ind,
        )

    def call(
        self,
        y: torch.Tensor,
        no: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""Estimate the full effective resource-grid channel.

        :input y: [batch, num_rx, num_rx_ant, num_ofdm_symbols, fft_size],
            `torch.complex`. Received OFDM resource grid.
        :input no: `torch.float`. Noise variance with a Sionna-compatible
            broadcast shape.

        :output h_hat: [batch, num_rx, num_rx_ant, num_tx,
            num_streams_per_tx, num_ofdm_symbols,
            num_effective_subcarriers], `torch.complex`. Channel estimates.
        :output err_var: Same shape as ``h_hat``, `torch.float`.
            Detector-facing channel-estimation error variances.
        """
        y_pilots = self._gather_y_pilots(y)
        h_ls, ls_err = LSChannelEstimator.estimate_at_pilot_locations(
            self,
            y_pilots,
            no,
        )
        grouped_h_ls = self._gather_pilot_groups(h_ls)
        grouped_ls_err = self._gather_pilot_groups(ls_err)

        leading_shape = grouped_h_ls.shape[:-1]
        h_images = grouped_h_ls.reshape(
            *leading_shape,
            self._num_pilot_subcarriers,
            self._num_pilot_ofdm_symbols,
        )
        err_images = grouped_ls_err.reshape(
            *leading_shape,
            self._num_pilot_subcarriers,
            self._num_pilot_ofdm_symbols,
        )

        model_inputs = torch.stack([h_images.real, h_images.imag], dim=-3)
        model_inputs = model_inputs.reshape(
            -1,
            2,
            self._num_pilot_subcarriers,
            self._num_pilot_ofdm_symbols,
        ).to(dtype=self.dtype)

        if self._model_noise_variances.numel() == 0:
            model = self._models[0]
        else:
            model_index = self._select_model_index(no)
            model = self._models[model_index]
        with torch.no_grad():
            model_outputs = model(model_inputs)
        full_images = torch.complex(model_outputs[:, 0], model_outputs[:, 1])
        full_images = full_images.reshape(
            *leading_shape,
            self._num_effective_subcarriers,
            self._num_ofdm_symbols,
        ).transpose(-2, -1)

        err_inputs = err_images.reshape(
            -1,
            1,
            self._num_pilot_subcarriers,
            self._num_pilot_ofdm_symbols,
        )
        err_outputs = torch.nn.functional.interpolate(
            err_inputs,
            size=(self._num_effective_subcarriers, self._num_ofdm_symbols),
            mode="bilinear",
            align_corners=False,
        )
        err_outputs = err_outputs.reshape(
            *leading_shape,
            self._num_effective_subcarriers,
            self._num_ofdm_symbols,
        ).transpose(-2, -1)
        err_outputs = self._err_var_scale * err_outputs

        output_shape = (
            *y_pilots.shape[:3],
            int(self._pilot_pattern.num_tx),
            int(self._pilot_pattern.num_streams_per_tx),
            self._num_ofdm_symbols,
            self._num_effective_subcarriers,
        )
        # _PilotGroupMixin orders groups by their fixed (tx, stream)
        # coordinates, so the group axis is the flattened stream axis.
        h_hat = full_images.reshape(output_shape)
        err_var = err_outputs.reshape(output_shape)

        err_var = torch.maximum(err_var, torch.zeros_like(err_var))
        return h_hat, err_var
