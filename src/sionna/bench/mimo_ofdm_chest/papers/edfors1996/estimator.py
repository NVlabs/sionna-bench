# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pilot-domain DFT estimators adapted from Edfors et al. 1996."""

from __future__ import annotations

import math
from typing import Optional, Union

import numpy as np
import torch
from sionna.phy import Precision
from sionna.phy.ofdm import (
    BasePilotChannelEstimator,
    BaseChannelInterpolator,
    LSChannelEstimator,
    ResourceGrid,
)
from sionna.phy.ofdm.channel_estimation import _PilotGroupMixin


METHOD_LMMSE = "LMMSE"
METHOD_DIAGONAL_LMMSE = "diagonal LMMSE"
METHOD_LS = "LS"
METHOD_NAMES = [
    METHOD_LMMSE,
    METHOD_DIAGONAL_LMMSE,
    METHOD_LS,
]

class TruncatedChannelEstimator(BasePilotChannelEstimator, _PilotGroupMixin):
    r"""Channel estimator that leverages sparsity in the delay domain through truncation.

    This estimator implements the methods proposed in:

    .. container:: bibliography-box

        .. bibliography::
            :filter: key == "edfors1996analysis"
            :list: bullet

    .. rubric:: TL;DR

    For each transmit stream and pilot-bearing OFDM symbol, the estimator first
    computes LS channel estimates on the active pilot subcarriers. It maps this
    pilot-frequency vector to the delay domain with a unitary IDFT, retains the
    ``num_taps`` coefficients with the largest powers according to
    ``time_covariance``, filters the retained coefficients according to ``method``,
    and sets all remaining coefficients to zero. A unitary DFT maps the result
    back to the pilot subcarriers, after which the configured Sionna interpolator
    reconstructs the complete resource grid.

    The following methods are supported:

    - ``"LMMSE"`` applies the full selected-tap LMMSE matrix
      :math:`\mathbf{R}_L(\mathbf{R}_L+\sigma^2\mathbf{I})^{-1}`,
      corresponding to the paper's Estimator A.
    - ``"diagonal LMMSE"`` applies an independent scalar LMMSE weight to each
      retained tap, corresponding to the paper's Estimator B.
    - ``"LS"`` retains the selected delay-domain LS coefficients without
      shrinkage, corresponding to the paper's Estimator C.

    .. rubric:: Supported System Geometry

    The estimator supports different FFT sizes, OFDM-symbol counts,
    pilot-bearing OFDM-symbol positions, transmit-stream counts, and receive-antenna
    counts, provided that the following hold:

    - At least one active pilot group exists.
    - Every transmit-stream and pilot-symbol group contains the same number of
      active pilot subcarriers.
    - ``num_taps`` is between one and this common pilot-group length.
    - ``time_covariance`` is a square matrix whose dimension equals the common
      pilot-group length.

    Receive antennas are processed independently.

    .. rubric:: Deviations from the Paper

    - The paper assumes known transmitted symbols on every OFDM subcarrier. For
      sparse pilot patterns, this implementation applies the paper's DFT-domain
      estimators to each compact pilot-subcarrier vector.
    - The paper does not specify reconstruction from sparse pilots to a complete
      resource grid. This implementation delegates that step to a configurable
      Sionna interpolator.
    - The supplied delay-domain covariance is estimated empirically from the
      active channel model instead of being derived from the analytical channel
      model used in the paper.
    - When ``num_taps`` is `None`, the cyclic-prefix length, capped at the
      pilot-group length, is used. An explicit value may instead be selected
      through validation or tuning.
    - The paper does not specify detector-facing error variances, so the implementation
      returns ``err_var_scale * no``, where the default value of ``err_var_scale`` was
      selected using Optuna.

    :param resource_grid: Resource grid defining the OFDM dimensions and pilot
        pattern.
    :param time_covariance: Delay-domain channel covariance with shape
        ``[num_pilot_subcarriers, num_pilot_subcarriers]``.
    :param interpolation_type: Sionna interpolation policy. Ignored when
        ``interpolator`` is provided.
    :param interpolator: Custom Sionna channel interpolator. If `None`, the
        interpolator selected by ``interpolation_type`` is used.
    :param method: DFT-domain estimation method. Must be ``"LMMSE"``,
        ``"diagonal LMMSE"``, or ``"LS"``.
    :param num_taps: Number of delay-domain coefficients to retain. If `None`, use
        the cyclic-prefix length capped at the pilot-group length.
    :param chest_err_const: Non-negative scale applied to the LS-derived
        detector-facing error variance.
    :param precision: Precision used for internal calculations and outputs. If
        `None`, :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device used for tensor operations. If `None`,
        :attr:`~sionna.phy.config.Config.device` is used.
    """
    def __init__(
        self,
        resource_grid: ResourceGrid,
        time_covariance: Union[np.ndarray, torch.Tensor],
        interpolation_type: str = "lin_time_avg",
        interpolator: Optional[BaseChannelInterpolator] = None,
        *,
        method: str = METHOD_LS,
        num_taps: Optional[int] = 64,
        chest_err_const: float = 0.308,
        precision: Optional[Precision] = None,
        device: Optional[str] = None,
    ) -> None:
        """Initialize delay truncation, covariance weights, and interpolation."""
        assert method in METHOD_NAMES, "Unsupported Edfors DFT method."
        assert isinstance(chest_err_const, (int, float)) and math.isfinite(
            chest_err_const
        ), "`chest_err_const` must be finite."
        assert chest_err_const >= 0.0, "`chest_err_const` must be non-negative."

        super().__init__(
            resource_grid=resource_grid,
            interpolation_type=interpolation_type,
            interpolator=interpolator,
            precision=precision,
            device=device,
        )
        self._method = method
        self._chest_err_const = float(chest_err_const)
        self._init_pilot_groups(
            self._pilot_pattern,
            group_over=("subcarrier",),
        )
        assert self._pilot_group_num_groups > 0, (
            "Edfors DFT estimator requires nonzero pilot groups."
        )

        self._full_taps = self._pilot_group_len
        default_taps = min(
            int(resource_grid.cyclic_prefix_length),
            self._full_taps,
        )
        self._num_taps = int(num_taps) if num_taps is not None else default_taps
        assert 1 <= self._num_taps <= self._full_taps, (
            f"`num_taps` must be between 1 and {self._full_taps}."
        )

        covariance = torch.as_tensor(
            time_covariance,
            dtype=self.cdtype,
            device=self.device,
        )
        assert (
            covariance.shape == (self._full_taps, self._full_taps)
        ), "Edfors covariance asset has incompatible shape."
        powers = covariance.diagonal().real.clamp_min(0.0)
        tap_order = torch.argsort(powers, dim=-1, descending=True)
        self.register_buffer("_time_covariance", covariance)
        self.register_buffer("_tap_order", tap_order)

    def _selected_tap_indices(self, device: torch.device) -> torch.Tensor:
        """Return the strongest delay-tap indices used by the estimator."""
        return self._tap_order[:self._num_taps].to(device=device)

    def _weight(
        self,
        noise_var: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Compute the weight matrix for the selected taps."""
        selected = self._selected_tap_indices(device)
        time_covariance = self._time_covariance.to(device=device, dtype=dtype)

        if self._method == METHOD_LMMSE:
            selected_covariance = time_covariance.index_select(
                0, selected
            ).index_select(1, selected)
            eye = torch.eye(
                selected.numel(),
                dtype=dtype,
                device=device,
            )
            # Appendix A Estimator A: full LMMSE map on selected DFT taps.
            # The solve gives (R+nI)^-1 R, which equals R(R+nI)^-1 because
            # R commutes with R+nI.
            weight = torch.linalg.solve(
                selected_covariance + noise_var * eye,
                selected_covariance,
            )
        elif self._method == METHOD_DIAGONAL_LMMSE:
            powers = (
                time_covariance.diagonal()
                .real.index_select(0, selected)
                .clamp_min(0.0)
            )
            shrinkage = powers / (powers + noise_var)
            weight = torch.diag(shrinkage.to(dtype=dtype))
        else:
            weight = torch.eye(selected.numel(), dtype=dtype, device=device)

        return weight

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
        h_ls, ls_err = LSChannelEstimator.estimate_at_pilot_locations(
            self, y_pilots, no
        )
        grouped_h_ls = self._gather_pilot_groups(h_ls)
        grouped_ls_err = self._gather_pilot_groups(ls_err)
        grouped_h_out = torch.zeros_like(grouped_h_ls)

        for group_index in range(self._pilot_group_num_groups):
            values = grouped_h_ls[..., group_index, :]
            pilot_err = grouped_ls_err[..., group_index, :]
            noise_var = pilot_err.mean().clamp_min(1e-12)

            flat = values.reshape(-1, values.shape[-1])
            time_values = torch.fft.ifft(flat, dim=-1, norm="ortho")

            selected = self._selected_tap_indices(y_pilots.device)
            selected_values = time_values.index_select(-1, selected)
            weight = self._weight(
                noise_var,
                y_pilots.device,
                time_values.dtype,
            )

            filtered = selected_values @ weight.transpose(0, 1)

            time_out = torch.zeros_like(time_values)
            time_out.index_copy_(-1, selected, filtered)
            freq_out = torch.fft.fft(time_out, dim=-1, norm="ortho").reshape_as(values)
            grouped_h_out[..., group_index, :] = freq_out

        # The paper does not define Sionna detector-facing variances. Use the
        # task-contract fallback chest_err_const*no/|pilot|^2.
        h_out = self._scatter_pilot_groups(torch.zeros_like(h_ls), grouped_h_out)
        return h_out, self._chest_err_const * ls_err
