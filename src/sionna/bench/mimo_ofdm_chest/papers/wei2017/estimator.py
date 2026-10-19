# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Comb-specialized pilot-domain SBL estimator adapted from Wei et al. 2017."""

from __future__ import annotations

import math
from typing import Optional

import torch
from sionna.phy import Precision
from sionna.phy.ofdm import (
    BasePilotChannelEstimator,
    BaseChannelInterpolator,
    LSChannelEstimator,
    ResourceGrid,
)
from sionna.phy.ofdm.channel_estimation import _PilotGroupMixin



class SBLChannelEstimator(
    BasePilotChannelEstimator,
    _PilotGroupMixin,
):
    r"""Channel estimator that leverages sparsity in the delay domain through Sparse Bayesian Learning (SBL).

    This estimator implements the method proposed in:

    .. container:: bibliography-box

        .. bibliography::
            :filter: key == "wei2017expectation"
            :list: bullet

    .. rubric:: TL;DR

    The estimator models the retained delay-domain coefficients at delay
    :math:`l` and receive antenna :math:`m` as conditionally independent complex
    Gaussian variables,

    .. math::

        h_{m,l} \mid \alpha_l
        \sim
        \mathcal{CN}(0,\alpha_l),

    where the variance :math:`\alpha_l` is shared across the co-located receive
    antennas. The estimator first computes LS channel observations at the pilot-bearing
    resource elements, transforms each pilot comb to the delay domain, and applies
    sparse Bayesian denoising there. The denoised coefficients are transformed back
    to the observed pilot subcarriers, and the configured Sionna interpolator
    reconstructs the remaining resource elements.

    .. rubric:: Supported System Geometry

    The estimator supports different FFT sizes, UE counts, pilot-bearing OFDM
    symbols, transmit-stream counts, and receive-antenna counts, provided that the
    following hold:

    - Every active pilot group is a uniformly spaced subcarrier comb.
    - All pilot groups have the same number of active subcarriers.
    - For every group, ``comb_spacing * group_length == fft_size``.
    - Pilot power is constant and non-zero within each group.
    - ``num_taps`` is between one and the common pilot-group length.

    The delay-support variances are shared across receive antennas but are
    estimated independently for different batches, receivers, transmit streams,
    and pilot-bearing OFDM symbols.

    .. rubric:: Deviations from the Paper

    - The paper considers downlink estimation with channels from multiple
      co-located transmit antennas superimposed through a generic pilot
      observation matrix. This uplink adaptation instead shares the delay-support
      variances across the separated BS receive-antenna observations.
    - Pilot normalization and the uniform-comb IFFT diagonalize the adapted
      observation model. Consequently, this implementation evaluates exact scalar
      Gaussian posterior moments and does not implement the paper's generic
      expectation-propagation cavity distributions or factor-to-variable
      messages.
    - The paper assumes a finite causal delay-tap model and reconstructs the full
      frequency response by end-padding the delay vector and applying an FFT.
      This implementation estimates only the observed pilot subcarriers and
      delegates full-grid reconstruction to a configurable Sionna interpolator.
    - If ``num_taps`` is `None`, all pilot-comb delay coefficients are retained.
      This avoids imposing the paper's model-matched finite-delay truncation on
      channels with continuous, off-grid delays.
    - The returned pilot-subcarrier error variances are posterior marginals under
      the diagonalized delay model. Correlations between frequency-domain errors
      are not represented by Sionna's marginal ``err_var`` interface or propagated
      by its interpolators.

    :param resource_grid: Resource grid defining the OFDM dimensions and pilot
        pattern.
    :param interpolation_type: Sionna interpolation policy used after estimating
        the observed pilot subcarriers. Ignored when ``interpolator`` is provided.
    :param interpolator: Custom Sionna channel interpolator. If `None`, the
        interpolator selected by ``interpolation_type`` is used.
    :param num_taps: Number of delay-domain coefficients retained from each pilot
        comb. If `None`, retain the complete pilot-group length.
    :param num_iter: Number of expectation-maximization updates.
    :param alpha_init: Positive initial value of every common-support variance
        parameter :math:`\alpha_l`.
    :param posterior_variance_scale: Non-negative confidence scale applied to the
        summed delay-domain posterior variances before interpolation.
    :param precision: Precision used for internal calculations and outputs. If
        `None`, :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device used for tensor operations. If `None`,
        :attr:`~sionna.phy.config.Config.device` is used.
    """
    def __init__(
        self,
        resource_grid: ResourceGrid,
        interpolation_type: str = "lin_time_avg",
        interpolator: Optional[BaseChannelInterpolator] = None,
        *,
        num_taps: Optional[int] = None,
        num_iter: int = 10,
        alpha_init: float = 0.00259,
        posterior_variance_scale: float = 0.755,
        precision: Optional[Precision] = None,
        device: Optional[str] = None,
    ) -> None:
        """Initialize pilot-comb metadata and SBL hyperparameters."""
        assert isinstance(num_iter, int) and num_iter >= 1, (
            "`num_iter` must be a positive integer."
        )
        assert isinstance(alpha_init, (int, float)) and math.isfinite(alpha_init), (
            "`alpha_init` must be finite."
        )
        assert alpha_init > 0.0, "`alpha_init` must be positive."
        assert isinstance(
            posterior_variance_scale, (int, float)
        ) and math.isfinite(posterior_variance_scale), (
            "`posterior_variance_scale` must be finite."
        )
        assert posterior_variance_scale >= 0.0, (
            "`posterior_variance_scale` must be non-negative."
        )
        super().__init__(
            resource_grid=resource_grid,
            interpolation_type=interpolation_type,
            interpolator=interpolator,
            precision=precision,
            device=device,
        )
        self._num_iter = int(num_iter)
        self._alpha_init = float(alpha_init)
        self._posterior_variance_scale = float(posterior_variance_scale)
        self._fft_size = int(resource_grid.fft_size)

        self._init_pilot_groups(self._pilot_pattern, group_over=("subcarrier",))
        self._init_comb_metadata(resource_grid)

        self._num_taps = (
            int(num_taps) if num_taps is not None else self._pilot_group_len
        )
        assert 1 <= self._num_taps <= self._pilot_group_len, (
            f"`num_taps` must be between 1 and {self._pilot_group_len}."
        )

    def _init_comb_metadata(self, resource_grid) -> None:
        """Register pilot values and validate the approved comb assumptions."""
        effective_subcarrier_ind = torch.as_tensor(
            resource_grid.effective_subcarrier_ind,
            dtype=torch.int64,
            device=self.device,
        )
        group_subcarrier_indices = self._pilot_group_varying_coord("subcarrier")
        group_fft_bin_indices = effective_subcarrier_ind.index_select(
            0,
            group_subcarrier_indices.reshape(-1),
        ).reshape_as(group_subcarrier_indices)

        subcarrier_spacings = torch.diff(group_fft_bin_indices, dim=-1)
        first_spacings = subcarrier_spacings[:, :1]
        assert bool(torch.all(subcarrier_spacings == first_spacings)), (
            "Wei comb-specialized estimator requires uniform pilot spacing."
        )
        comb_spacings = first_spacings[:, 0]
        assert bool(torch.all(comb_spacings * self._pilot_group_len == self._fft_size)), (
            "Wei comb-specialized estimator requires spacing*group_len == fft_size."
        )

        pilot_values = self._gather_pilot_groups(
            self._pilot_pattern.pilots.to(self.device)
        ).to(dtype=self.cdtype, device=self.device)
        pilot_power_by_entry = pilot_values.abs().square().to(dtype=self.dtype)
        first_pilot_power = pilot_power_by_entry[:, :1]
        assert bool(torch.all(first_pilot_power > 0.0)), (
            "Wei comb-specialized estimator requires nonzero pilots."
        )
        assert bool(torch.allclose(pilot_power_by_entry, first_pilot_power)), (
            "Wei comb-specialized estimator requires constant pilot power "
            "within each group."
        )
        self.register_buffer("_group_pilot_power", first_pilot_power[:, 0])

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
        no = torch.as_tensor(no, dtype=self.dtype, device=y_pilots.device)
        h_ls, _ = LSChannelEstimator.estimate_at_pilot_locations(
            self,
            y_pilots,
            no,
        )
        grouped_h_ls = self._gather_pilot_groups(h_ls)
        batch_size, num_rx, num_rx_ant = grouped_h_ls.shape[:3]
        noise_var = self._expand_noise_variance(
            no,
            batch_size,
            num_rx,
            num_rx_ant,
            y_pilots.device,
        )
        delay_observations, delay_noise_var = self._delay_observations(
            grouped_h_ls,
            noise_var,
        )
        delay_mean, delay_var = self._estimate_delay_observations(
            delay_observations,
            delay_noise_var,
        )
        padded_delay_mean = torch.nn.functional.pad(
            delay_mean,
            (0, self._pilot_group_len - self._num_taps),
        )
        # Move the jointly processed group axis behind the receive antennas to
        # recover _PilotGroupMixin's [..., antenna, group, pilot] convention.
        grouped_estimates = torch.fft.fft(
            padded_delay_mean,
            dim=-1,
        ).movedim(-3, -2)
        pilot_variance = self._posterior_variance_scale * delay_var.sum(dim=-1)
        grouped_variances = pilot_variance.unsqueeze(-1).expand(
            *pilot_variance.shape,
            self._pilot_group_len,
        ).movedim(-3, -2)
        return (
            self._scatter_pilot_groups(
                torch.zeros_like(y_pilots),
                grouped_estimates,
            ),
            self._scatter_pilot_groups(
                torch.zeros_like(y_pilots.real),
                grouped_variances,
            ),
        )

    def _delay_observations(
        self,
        h_ls: torch.Tensor,
        noise_var: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""Return delay observations for all pilot groups.

        ``h_ls`` has shape
        ``[batch, receiver, antenna, group, pilot]``, and ``noise_var`` has
        shape ``[batch, receiver, antenna]``.
        The returned observations and noise variances have shapes
        ``[batch, receiver, group, antenna, tap]`` and
        ``[batch, receiver, group, antenna]``, respectively.
        """
        all_delay_observations = torch.fft.ifft(
            h_ls,
            n=self._pilot_group_len,
            dim=-1,
        )
        delay_observations = all_delay_observations[
            ..., : self._num_taps
        ].movedim(-2, -3)
        pilot_power = self._group_pilot_power.to(
            dtype=noise_var.dtype,
            device=noise_var.device,
        )
        delay_noise_var = noise_var.unsqueeze(-2) / (
            float(self._pilot_group_len) * pilot_power.reshape(1, 1, -1, 1)
        )
        return delay_observations, delay_noise_var

    def _estimate_delay_observations(
        self,
        delay_observations: torch.Tensor,
        delay_noise_var: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""Apply scalar Gaussian posterior and common-support EM updates.

        The final two input dimensions are receive antenna and delay tap. All
        preceding dimensions, including the pilot-group dimension, are
        processed independently.
        """
        noise_floor = torch.as_tensor(
            torch.finfo(self.dtype).eps,
            dtype=self.dtype,
            device=delay_noise_var.device,
        )
        delay_noise_var = torch.maximum(delay_noise_var, noise_floor)
        alpha = torch.full(
            (*delay_observations.shape[:-2], self._num_taps),
            self._alpha_init,
            dtype=self.dtype,
            device=delay_observations.device,
        )

        for _ in range(self._num_iter):
            delay_mean, delay_var = self._gaussian_posterior_moments(
                delay_observations,
                delay_noise_var,
                alpha,
            )
            # Eq. (21): common support across the BS receive-antenna axis.
            alpha = (delay_mean.abs().square() + delay_var).mean(dim=-2)

        # Eq. (16): return the posterior evaluated with the final EM update.
        return self._gaussian_posterior_moments(
            delay_observations,
            delay_noise_var,
            alpha,
        )

    @staticmethod
    def _gaussian_posterior_moments(
        delay_observations: torch.Tensor,
        delay_noise_var: torch.Tensor,
        alpha: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate the scalar Gaussian posterior from Equations (14)/(16)."""
        alpha_grid = alpha.unsqueeze(-2)
        noise_grid = delay_noise_var.unsqueeze(-1)
        delay_var = 1.0 / (1.0 / alpha_grid + 1.0 / noise_grid)
        delay_mean = delay_var * delay_observations / noise_grid
        return delay_mean, delay_var

    def _expand_noise_variance(
        self,
        no: torch.Tensor,
        batch_size: int,
        num_rx: int,
        num_rx_ant: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Expand Sionna noise variance to ``[batch, num_rx, num_rx_ant]``."""
        no = no.to(dtype=self.dtype, device=device)
        noise_var = no.reshape(*no.shape, *((1,) * (3 - no.dim())))
        return noise_var.expand(batch_size, num_rx, num_rx_ant)
