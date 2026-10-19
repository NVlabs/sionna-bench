# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DNN channel estimators from Melgar et al. 2022."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Optional

import torch
from sionna.phy import Precision
from sionna.phy.ofdm import (
    BaseChannelEstimator,
    BasePilotChannelEstimator,
    BaseChannelInterpolator,
    LSChannelEstimator,
    ResourceGrid,
)
from sionna.phy.ofdm.channel_estimation import _PilotGroupMixin


HIDDEN_SIZES = (1024, 512, 256, 128, 64, 64, 64)
MODEL_2_ERR_VAR_SCALE = 0.454
METHOD_MODEL_2 = "model2"
METHOD_MODEL_3 = "model3"
METHOD_NAMES = [METHOD_MODEL_2, METHOD_MODEL_3]


class DNN(torch.nn.Module):
    """Real-valued Melgar DNN used for real and imaginary channel vectors."""

    def __init__(self, input_dim: int) -> None:
        """Build the fully connected real-valued denoising network."""
        super().__init__()
        layers: list[torch.nn.Module] = []
        prev = int(input_dim)
        for hidden_size in HIDDEN_SIZES:
            layers.append(torch.nn.Linear(prev, hidden_size))
            layers.append(torch.nn.ReLU())
            prev = hidden_size
        layers.append(torch.nn.Linear(prev, int(input_dim)))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the DNN to one batch of real-valued channel vectors."""
        return self.net(x)


class _DNNBank(torch.nn.Module):
    """Own prepared denoisers and select an SNR-specific state at runtime."""

    def __init__(
        self,
        input_dim: int,
        model_states: Sequence[Mapping[str, torch.Tensor]],
        model_noise_variances: Optional[torch.Tensor],
        *,
        noise_dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        """Build denoisers from prepared states and register their SNR grid."""
        super().__init__()
        assert len(model_states) > 0, "At least one DNN state is required."
        denoisers = []
        for state in model_states:
            denoiser = DNN(input_dim).to(device=device, dtype=torch.float32)
            denoiser.load_state_dict(state)
            denoiser.eval()
            denoisers.append(denoiser)
        self._denoisers = torch.nn.ModuleList(denoisers)

        if model_noise_variances is None:
            assert len(model_states) == 1, (
                "Multiple DNN states require `model_noise_variances`."
            )
            noise_variances = torch.empty(
                0,
                dtype=noise_dtype,
                device=device,
            )
        else:
            noise_variances = torch.as_tensor(
                model_noise_variances,
                dtype=noise_dtype,
                device=device,
            ).flatten()
            assert int(noise_variances.numel()) == len(model_states), (
                "`model_noise_variances` must have one entry per DNN state."
            )
            assert bool(torch.all(noise_variances > 0.0)), (
                "`model_noise_variances` must be positive."
            )
        self.register_buffer("_noise_variances", noise_variances)

    def forward(
        self,
        values: torch.Tensor,
        no: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the mixed-SNR or matching SNR-specific denoiser."""
        if self._noise_variances.numel() == 0:
            return self._denoisers[0](values)

        denoiser_index = self._select_denoiser_index(no)
        return self._denoisers[denoiser_index](values)

    @torch.compiler.disable
    def _select_denoiser_index(self, no: torch.Tensor) -> int:
        """Select one prepared state outside the compiled tensor path."""
        noise_variance = no.to(
            dtype=self._noise_variances.dtype,
            device=self._noise_variances.device,
        ).reshape(-1).mean()
        log_distance = torch.abs(
            torch.log(noise_variance) - torch.log(self._noise_variances)
        )
        denoiser_index = int(torch.argmin(log_distance).item())
        assert bool(log_distance[denoiser_index] < 1e-3), (
            "No SNR-specific Melgar DNN is available for the current noise "
            "variance."
        )
        return denoiser_index


class DNNChannelEstimator(BasePilotChannelEstimator, _PilotGroupMixin):
    r"""Channel estimater that leverages a fully connected deep neural network (DNN).

    This estimator follows Model 2 of the paper

    .. container:: bibliography-box

        .. bibliography::
            :filter: key == "melgar2022deep"
            :list: bullet

    .. rubric:: TL;DR

    The estimator first computes LS channel estimates at the active pilot
    locations. It groups the pilots of each transmit stream and pilot-bearing OFDM
    symbol into a frequency vector and applies the same fully connected DNN
    independently to its real and imaginary parts. The denoised pilot estimates are
    then reconstructed over the complete resource grid by the configured Sionna
    interpolator.

    Receive antennas, transmit streams, and pilot-bearing OFDM symbols are
    processed as independent DNN examples and share the same trained weights.

    .. rubric:: Supported System Geometry

    The estimator supports different FFT sizes, OFDM-symbol counts, pilot-bearing
    symbol positions, UE counts, stream counts, and receive-antenna counts,
    provided that the following hold:

    - Every active pilot group contains the same number of pilot subcarriers.
    - Every prepared DNN state has input and output dimensions equal to that pilot
      group length.
    - The prepared model states match the exact pilot coordinates, pilot values,
      pilot powers, and DNN architecture for which they were trained.
    - When SNR-specific states are used, one state matching the scalar operating
      noise variance of the estimator call is available.

    Changing the number of receive antennas or the number of otherwise compatible
    pilot groups does not change the DNN dimensions because those dimensions are
    processed as independent batch examples.

    .. rubric:: Deviations from the Paper

    - This implementation derives the vector length from the active pilot groups
      of the supplied Sionna resource grid instead of using the paper's fixed vector
      length.
    - The paper specifies hidden dimensions
      ``1000, 500, 256, 128, 32, 16, 8``. This implementation uses
      ``1024, 512, 256, 128, 64, 64, 64``. In particular, its narrowest hidden
      dimension is 64 rather than 8.
    - The paper specifies two-dimensional linear interpolation after pilot-domain
      denoising. This implementation exposes Sionna's interpolation interface.
    - The paper does not specify detector-facing error variances, so the implementation
      returns ``err_var_scale * no``, where the default value of ``err_var_scale`` was
      selected using Optuna.

    :param resource_grid: Sionna resource grid defining the pilot pattern and
        interpolation target.
    :param model_states: Prepared DNN state dictionaries. Every state must contain
        the `torch.float32` weights and biases of a ``DNN`` whose input and output
        dimensions equal the active pilot-group length.
    :param model_noise_variances: Noise variance associated with each model
        state, shape ``[num_models]``. These values select an SNR-specific state.
        If `None`, exactly one mixed-SNR state must be supplied.
    :param interpolation_type: Sionna interpolation policy. This argument is
        ignored when ``interpolator`` is provided.
    :param interpolator: Custom Sionna channel interpolator. If `None`, the
        interpolator selected by ``interpolation_type`` is used.
    :param err_var_scale: Nonnegative multiplicative scale applied to the LS pilot
        error variances before interpolation.
    :param precision: Precision used for channel estimation. If `None`,
        :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device used for channel estimation. If `None`,
        :attr:`~sionna.phy.config.Config.device` is used.
    """

    def __init__(
        self,
        resource_grid: ResourceGrid,
        model_states: Sequence[Mapping[str, torch.Tensor]],
        model_noise_variances: Optional[torch.Tensor] = None,
        interpolation_type: str = "lin_time_avg",
        interpolator: Optional[BaseChannelInterpolator] = None,
        err_var_scale: float = MODEL_2_ERR_VAR_SCALE,
        *,
        precision: Optional[Precision] = None,
        device: Optional[str] = None,
    ) -> None:
        """Initialize pilot groups and install prepared Model 2 states."""
        super().__init__(
            resource_grid=resource_grid,
            interpolation_type=interpolation_type,
            interpolator=interpolator,
            precision=precision,
            device=device,
        )
        self._init_pilot_groups(
            self._pilot_pattern,
            group_over=("subcarrier",),
        )
        self._input_dim = self._pilot_group_len
        assert err_var_scale >= 0.0, "`err_var_scale` must be nonnegative."
        self._err_var_scale = float(err_var_scale)
        self._denoiser_bank = _DNNBank(
            self._input_dim,
            model_states,
            model_noise_variances,
            noise_dtype=self.dtype,
            device=self.device,
        )

    def _denoise_values(
        self,
        values: torch.Tensor,
        no: torch.Tensor,
    ) -> torch.Tensor:
        """Denoise grouped complex LS pilot estimates with the real DNN."""
        leading_shape = values.shape[:-1]
        flat = values.reshape(-1, self._input_dim)
        x = torch.cat([flat.real, flat.imag], dim=0).to(torch.float32)
        y = self._denoiser_bank(x, no)
        real, imag = torch.chunk(y, 2, dim=0)
        out = torch.complex(real, imag).to(values.dtype)
        return out.reshape(*leading_shape, self._input_dim)

    def estimate_at_pilot_locations(self, y_pilots: torch.Tensor, no: torch.Tensor):
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
        h_ls, ls_err = LSChannelEstimator.estimate_at_pilot_locations(
            self, y_pilots, no
        )
        h_out = h_ls.clone()

        with torch.no_grad():
            grouped_h_ls = self._gather_pilot_groups(h_ls)
            grouped_estimates = self._denoise_values(grouped_h_ls, no)
            self._scatter_pilot_groups_(h_out, grouped_estimates)

        # The paper does not define detector-facing uncertainty. Use the
        # calibrated LS observation-model variance required by Sionna.
        return h_out, ls_err * self._err_var_scale


class InterpolatedDNNChannelEstimator(BaseChannelEstimator):
    r"""Channel estimator that leverages a fully connected deep neural network (DNN) after linear interpolation.

    This implementation follows Model 3 proposed in:

    .. container:: bibliography-box

        .. bibliography::
            :filter: key == "melgar2022deep"
            :list: bullet

    .. rubric:: TL;DR

    The estimator first computes LS channel estimates at the pilot locations and
    uses linear interpolation to reconstruct the complete frequency-time channel
    grid. It divides every OFDM symbol into contiguous frequency blocks and applies
    the same fully connected DNN independently to the real and imaginary parts of
    each block.

    Receive antennas, transmit streams, OFDM symbols, and frequency blocks are
    processed as independent DNN examples and share the same trained weights.

    .. rubric:: Supported System Geometry

    The estimator supports different FFT sizes, OFDM-symbol counts, pilot layouts,
    UE counts, stream counts, and receive-antenna counts, provided that the
    following hold:

    - Sionna's ``LSChannelEstimator`` supports the supplied pilot pattern.
    - ``block_len`` is positive and exactly divides the number of effective
      subcarriers.
    - Every prepared DNN state has input and output dimensions equal to
      ``block_len``.
    - The prepared model states match the complete resource grid, pilot layout,
      block length, and DNN architecture for which they were trained.
    - When SNR-specific states are used, one state matching the scalar operating
      noise variance of the estimator call is available.

    Changing the number of receive antennas, transmit streams, or OFDM symbols
    does not change the DNN dimensions because those dimensions are processed as
    independent batch examples.

    .. rubric:: Deviations from the Paper

    - This implementation's default block length is 78 subcarriers rather
      than the paper's 72 subcarriers.
    - The paper specifies hidden dimensions
      ``1000, 500, 256, 128, 32, 16, 8``. This implementation uses
      ``1024, 512, 256, 128, 64, 64, 64``. In particular, its narrowest hidden
      dimension is 64 rather than 8.
    - The paper does not define detector-facing channel-estimation error
      variances. This implementation scales the full-grid LS-linear error
      variances by ``err_var_scale``.

    :param resource_grid: Sionna resource grid defining the pilot pattern and
        complete interpolation grid.
    :param model_states: Prepared DNN state dictionaries. Every state must contain
        the `torch.float32` weights and biases of a ``DNN`` whose input and output
        dimensions equal ``block_len``.
    :param model_noise_variances: Noise variance associated with each model
        state, shape ``[num_models]``. These values select an SNR-specific state.
        If `None`, exactly one mixed-SNR state must be supplied.
    :param block_len: Number of contiguous effective subcarriers in each Model 3
        frequency block. It must divide the number of effective subcarriers.
    :param err_var_scale: Nonnegative multiplicative scale applied to the
        full-grid LS-linear error variances.
    :param precision: Precision used for channel estimation. If `None`,
        :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device used for channel estimation. If `None`,
        :attr:`~sionna.phy.config.Config.device` is used.
    """
    def __init__(
        self,
        resource_grid: ResourceGrid,
        model_states: Sequence[Mapping[str, torch.Tensor]],
        model_noise_variances: Optional[torch.Tensor] = None,
        *,
        block_len: int = 78,
        err_var_scale: float = 0.341,
        precision: Optional[Precision] = None,
        device: Optional[str] = None,
    ) -> None:
        """Initialize Model 3 and install its prepared DNN states."""
        super().__init__(
            resource_grid=resource_grid,
            precision=precision,
            device=device,
        )
        self._linear_estimator = LSChannelEstimator(
            resource_grid,
            interpolation_type="lin",
            precision=precision,
            device=device,
        )
        self._input_dim = int(block_len)
        self._num_effective_subcarriers = int(resource_grid.num_effective_subcarriers)
        assert self._input_dim > 0, "`block_len` must be positive."
        assert self._num_effective_subcarriers % self._input_dim == 0, (
            "`block_len` must divide the number of effective subcarriers."
        )
        assert err_var_scale >= 0.0, "`err_var_scale` must be nonnegative."
        self._num_frequency_blocks = self._num_effective_subcarriers // self._input_dim
        self._err_var_scale = float(err_var_scale)
        self._denoiser_bank = _DNNBank(
            self._input_dim,
            model_states,
            model_noise_variances,
            noise_dtype=self.dtype,
            device=self.device,
        )

    def _denoise_blocks(
        self,
        values: torch.Tensor,
        no: torch.Tensor,
    ) -> torch.Tensor:
        """Denoise full-grid frequency blocks with the real-valued DNN."""
        leading_shape = values.shape[:-1]
        blocks = values.reshape(
            *leading_shape,
            self._num_frequency_blocks,
            self._input_dim,
        )
        flat = blocks.reshape(-1, self._input_dim)
        x = torch.cat([flat.real, flat.imag], dim=0).to(torch.float32)
        y = self._denoiser_bank(x, no)
        real, imag = torch.chunk(y, 2, dim=0)
        out_blocks = torch.complex(real, imag).to(values.dtype)
        out_blocks = out_blocks.reshape(
            *leading_shape,
            self._num_frequency_blocks,
            self._input_dim,
        )
        return out_blocks.reshape(*leading_shape, self._num_effective_subcarriers)

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
        h_lin, err_var = self._linear_estimator(y, no)

        with torch.no_grad():
            h_hat = self._denoise_blocks(h_lin, no)

        err_var = torch.maximum(
            err_var * self._err_var_scale,
            torch.zeros_like(err_var),
        )
        return h_hat, err_var


__all__ = [
    "DNN",
    "DNNChannelEstimator",
    "HIDDEN_SIZES",
    "InterpolatedDNNChannelEstimator",
    "METHOD_MODEL_2",
    "METHOD_MODEL_3",
    "METHOD_NAMES",
    "MODEL_2_ERR_VAR_SCALE",
]
