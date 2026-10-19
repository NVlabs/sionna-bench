# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GMM conditional-mean estimators from Koller et al. 2022."""

from __future__ import annotations

import math
from typing import Optional

import torch
from sionna.phy import Precision
from sionna.phy.ofdm import (
    BaseChannelEstimator,
    BasePilotChannelEstimator,
    BaseChannelInterpolator,
    LSChannelEstimator,
    PilotPattern,
    ResourceGrid,
)
from sionna.phy.ofdm.channel_estimation import _PilotGroupMixin
from sionna.phy.utils import flatten_last_dims

from sionna.bench.mimo_ofdm_chest.time_domain_interpolator import TimeDomainInterpolator


class SpatialGMMChannelEstimator(BasePilotChannelEstimator, _PilotGroupMixin):
    r"""Channel estimator that leverages a Gaussian mixture model (GMM) across receive antennas.

    This estimator implements the method proposed in:

    .. container:: bibliography-box

        .. bibliography::
            :filter: key == "koller2022asymptotically"
            :list: bullet

    .. rubric:: TL;DR

    For every non-zero pilot resource element, the estimator first estimates the
    LS channel vector across the receive antennas, then concatenates real and
    imaginary parts into a real-valued observation vector. A Gaussian
    mixture model across antennas is then used to compute the conditional
    mean and variance for every pilot-bearing resource element. The resulting
    estimates are then expanded to the complete resource grid using the
    configured Sionna interpolator.

    .. rubric:: Supported System Geometry

    The estimator supports different FFT sizes, OFDM-symbol counts, UE counts,
    stream counts, pilot positions, and receive-array geometries, provided that:

    - The GMM real-vector dimension is twice the number of receive antennas.
    - The covariance tensor contains one square covariance matrix per component.
    - All active pilots have equal power, allowing pilot groups to share the same
      noise-dependent posterior terms.
    - The configured interpolator supports the resource grid's pilot pattern.
    - The injected GMM was fitted for a statistically and geometrically compatible
      receive-array configuration.

    .. rubric:: Deviations from the Paper

    - The estimator applies the paper's GMM conditional mean independently at
      every pilot resource element and models only the receive-antenna dimension.
      It does not jointly model the complete spatial-frequency channel.
    - One GMM is fitted by pooling receive-array vectors from all UEs, pilot
      subcarriers, and pilot-bearing OFDM symbols. This assumes common spatial
      statistics across those observations.
    - The paper defines the conditional channel mean but does not provide
      detector-facing error variances. This implementation returns the posterior
      variances implied by the same GMM.

    :param resource_grid: Sionna resource grid defining the OFDM dimensions and
        pilot pattern.
    :param weights: GMM component probabilities with shape
        ``[num_components]``.
    :param means: Real-valued component means with shape
        ``[num_components, 2*num_rx_ant]``. Real channel entries precede
        imaginary channel entries.
    :param covariances: Full real-valued component covariance matrices with shape
        ``[num_components, 2*num_rx_ant, 2*num_rx_ant]`` using the same stacking
        convention as ``means``.
    :param interpolation_type: Sionna interpolation policy used outside the pilot
        positions. Defaults to ``"lin_time_avg"`` and is ignored when
        ``interpolator`` is supplied.
    :param interpolator: Optional custom Sionna channel interpolator.
    :param precision: Precision used for internal calculations and outputs. If
        `None`, use :attr:`~sionna.phy.config.Config.precision`.
    :param device: Device used for tensor operations. If `None`, use
        :attr:`~sionna.phy.config.Config.device`.
    """
    def __init__(
        self,
        resource_grid: ResourceGrid,
        weights: torch.Tensor,
        means: torch.Tensor,
        covariances: torch.Tensor,
        interpolation_type: str = "lin_time_avg",
        interpolator: Optional[BaseChannelInterpolator] = None,
        *,
        precision: Optional[Precision] = None,
        device: Optional[str] = None,
    ) -> None:
        """Initialize pilot groups and register the prepared GMM prior."""
        super().__init__(
            resource_grid=resource_grid,
            interpolation_type=interpolation_type,
            interpolator=interpolator,
            precision=precision,
            device=device,
        )
        self._init_pilot_groups(self._pilot_pattern, group_over=())
        pilot_values = self._gather_pilot_groups(
            self._pilot_pattern.pilots.to(self.device)
        )
        pilot_power = pilot_values.abs().square()
        assert bool(torch.allclose(pilot_power, pilot_power.reshape(-1)[0])), (
            "Spatial GMM joint processing requires equal-power pilots."
        )

        weights = weights.to(dtype=self.dtype, device=self.device)
        means = means.to(dtype=self.dtype, device=self.device)
        covariances = covariances.to(dtype=self.dtype, device=self.device)
        assert weights.dim() == 1 and int(weights.numel()) > 0, (
            "GMM weights must be a nonempty rank-1 tensor."
        )
        num_components = int(weights.shape[0])
        assert means.dim() == 2 and means.shape[0] == num_components, (
            "GMM means must have one row per component."
        )
        real_dim = int(means.shape[1])
        assert real_dim > 0 and real_dim % 2 == 0, (
            "GMM means must encode nonempty real-stacked complex vectors."
        )
        assert covariances.shape == (num_components, real_dim, real_dim), (
            "GMM covariances must be square with one matrix per component."
        )
        self.register_buffer("_weights", weights)
        self.register_buffer("_means", means)
        self.register_buffer("_covariances", covariances)

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
        grouped_h_ls = self._gather_pilot_groups(h_ls)
        grouped_ls_err = self._gather_pilot_groups(ls_err)

        # Pilot groups have length one. Move the receive-antenna dimension
        # last to form [..., num_groups, num_rx_ant] GMM vectors.
        values = grouped_h_ls.squeeze(-1).transpose(-2, -1)
        error_variance = grouped_ls_err[..., 0, 0]
        estimates, variances = self._estimate_groups(values, error_variance)
        grouped_h_out = estimates.transpose(-2, -1).unsqueeze(-1)
        grouped_err_out = variances.transpose(-2, -1).unsqueeze(-1)

        h_out = self._scatter_pilot_groups(torch.zeros_like(h_ls), grouped_h_out)
        err_out = self._scatter_pilot_groups(torch.zeros_like(ls_err), grouped_err_out)
        return h_out, err_out

    def _estimate_groups(
        self,
        values: torch.Tensor,
        error_variance: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""Jointly estimate pilot groups sharing GMM and noise statistics.

        :param values: Complex LS estimates with shape ``[..., num_groups,
            num_complex_values]``.
        :param error_variance: Shared group error variances with shape ``[]``,
            ``[num_complex_values]``, or ``[..., num_complex_values]``.

        :output estimates: Complex estimates with the same shape as ``values``.
        :output variances: Error variances with the same shape as ``values``.
        """
        leading_shape = values.shape[:-2]
        num_groups = int(values.shape[-2])
        num_complex_values = int(values.shape[-1])
        observations = torch.cat([values.real, values.imag], dim=-1).reshape(
            -1,
            num_groups,
            2 * num_complex_values,
        )
        num_noise_vectors = int(observations.shape[0])
        (
            means,
            cholesky_factors,
            gains,
            posterior_diagonals,
            log_norms,
        ) = self._compute_gmm_term_quantities(
            error_variance,
            num_noise_vectors,
            num_complex_values,
        )

        # Real-stacked shapes:
        # observations: [num_noise_vectors, num_groups, 2*num_complex_values]
        # means/posterior_diagonals:
        #   [num_noise_vectors, num_components, 2*num_complex_values]
        # cholesky_factors/gains:
        #   [num_noise_vectors, num_components, 2*num_complex_values,
        #    2*num_complex_values]
        # Equation (13) with A=I in the real-stacked LS observation model.
        residual = observations.unsqueeze(2) - means.unsqueeze(1)
        posterior_means = means.unsqueeze(1) + torch.matmul(
            residual.unsqueeze(-2),
            gains.transpose(-2, -1).unsqueeze(1),
        ).squeeze(-2)

        solved = torch.cholesky_solve(
            residual.permute(0, 2, 3, 1),
            cholesky_factors,
        ).permute(0, 3, 1, 2)
        mahalanobis_distance = (residual * solved).sum(dim=-1)
        # Equations (15) and (16): normalize the component likelihoods and
        # combine the corresponding component-conditional means.
        log_prob = log_norms.unsqueeze(1) - 0.5 * mahalanobis_distance
        responsibilities = torch.softmax(log_prob, dim=-1)
        estimated_real = (
            responsibilities.unsqueeze(-1) * posterior_means
        ).sum(dim=2)
        estimated_complex = torch.complex(
            estimated_real[..., :num_complex_values],
            estimated_real[..., num_complex_values:],
        ).to(values.dtype)
        # variance_real: [num_noise_vectors, num_groups, 2*num_complex_values]
        # variance_complex: [num_noise_vectors, num_groups, num_complex_values]
        variance_real = (
            responsibilities.unsqueeze(-1)
            * (
                posterior_diagonals.unsqueeze(1)
                + (posterior_means - estimated_real.unsqueeze(2)).square()
            )
        ).sum(dim=2)
        variance_complex = (
            variance_real[..., :num_complex_values]
            + variance_real[..., num_complex_values:]
        )

        return (
            estimated_complex.reshape(
                *leading_shape,
                num_groups,
                num_complex_values,
            ),
            variance_complex.reshape(
                *leading_shape,
                num_groups,
                num_complex_values,
            ).to(values.real.dtype),
        )

    def _compute_gmm_term_quantities(
        self,
        error_variance: torch.Tensor,
        num_observations: int,
        num_complex_values: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""Compute noise-dependent GMM posterior terms.

        :param error_variance: Error variances with shape ``[]``,
            ``[num_complex_values]``, or
            ``[num_observations, num_complex_values]``.
        :param num_observations: Number of flattened observations.
        :param num_complex_values: Complex vector length.

        :output means: Shape
            ``[num_observations, num_components, 2*num_complex_values]``.
        :output cholesky_factors: Shape
            ``[num_observations, num_components, 2*num_complex_values,
            2*num_complex_values]``.
        :output gains: Shape
            ``[num_observations, num_components, 2*num_complex_values,
            2*num_complex_values]``.
        :output posterior_diagonals: Shape
            ``[num_observations, num_components, 2*num_complex_values]``.
        :output log_norms: Shape ``[num_observations, num_components]``.
        """
        weights = self._weights
        means = self._means
        covariances = self._covariances
        real_dim = int(means.shape[-1])
        assert real_dim == 2 * num_complex_values, (
            "`error_variance` must match the complex GMM vector length."
        )
        if error_variance.dim() == 0:
            error_variance = error_variance.reshape(1, 1).expand(
                num_observations,
                num_complex_values,
            )
        else:
            error_variance = error_variance.reshape(-1, num_complex_values)
            assert error_variance.shape[0] in (1, num_observations), (
                "`error_variance` must have one row or one row per observation."
            )
            if error_variance.shape[0] == 1:
                error_variance = error_variance.expand(
                    num_observations,
                    num_complex_values,
                )

        # Proper complex LS noise with variance v has independent real and
        # imaginary parts with variance v/2.
        real_error_variance = 0.5 * error_variance
        covariances = covariances.unsqueeze(0).expand(
            num_observations,
            -1,
            -1,
            -1,
        )
        means = means.unsqueeze(0).expand(num_observations, -1, -1)
        covariances_plus_noise = covariances.clone()
        diagonal = torch.diagonal(
            covariances_plus_noise,
            dim1=-2,
            dim2=-1,
        )
        real_error_variance = real_error_variance.unsqueeze(1)
        diagonal[..., :num_complex_values].add_(real_error_variance)
        diagonal[..., num_complex_values:].add_(real_error_variance)

        cholesky_factors = torch.linalg.cholesky(covariances_plus_noise)
        gains = torch.cholesky_solve(
            covariances,
            cholesky_factors,
        ).transpose(-2, -1)
        posterior_covariances = covariances - gains @ covariances
        posterior_diagonals = torch.diagonal(
            posterior_covariances,
            dim1=-2,
            dim2=-1,
        )
        log_determinants = 2.0 * torch.log(
            torch.diagonal(cholesky_factors, dim1=-2, dim2=-1)
        ).sum(dim=-1)
        log_norm = (
            torch.log(weights.clamp_min(1e-12)).reshape(1, -1)
            - 0.5 * (real_dim * math.log(2.0 * math.pi) + log_determinants)
        )

        return means, cholesky_factors, gains, posterior_diagonals, log_norm


class SpectralGMMChannelEstimator(BaseChannelEstimator, _PilotGroupMixin):
    r"""Channel estimator that leverages a Gaussian mixture model (GMM) across subcarriers.

    This estimator implements the method proposed in:

    .. container:: bibliography-box

        .. bibliography::
            :filter: key == "koller2022asymptotically"
            :list: bullet

    .. rubric:: TL;DR

    For every receive antenna, transmit stream, and pilot-bearing OFDM symbol, the
    estimator first computes LS observations at the active pilot subcarriers and
    concatenates their real and imaginary parts into a real-valued observation
    vector. A Gaussian mixture model of the complete frequency-domain channel then
    provides the posterior mean and variance of all effective subcarriers
    conditioned on these observations. The resulting full-frequency estimates on
    pilot-bearing OFDM symbols are expanded across OFDM time using
    :class:`~sionna.bench.TimeDomainInterpolator`.

    .. rubric:: Supported System Geometry

    The estimator supports different UE counts, stream counts, pilot positions,
    OFDM-symbol counts, and receive-antenna counts, provided that:

    - The GMM real-vector dimension is twice the number of effective subcarriers.
    - Every component covariance is square and uses the same complete-subcarrier
      ordering as its mean.
    - All pilot groups have the same number of active pilot subcarriers.
    - Every transmit stream has the same number of pilot-bearing OFDM symbols.
    - Every pilot value used by the estimator is non-zero.
    - The injected GMM was fitted for a compatible resource-grid and pilot
      geometry.

    Receive antennas are processed independently, so changing their number does
    not change the GMM vector dimension.

    .. rubric:: Deviations from the Paper

    - The estimator models one complete frequency-domain vector for each receive
      antenna, stream, and pilot-bearing OFDM symbol. It does not jointly model the
      complete spatial-frequency MIMO channel.
    - One GMM is fitted by pooling full-subcarrier vectors from all receive
      antennas, UEs, and pilot-bearing OFDM symbols. This assumes common spectral
      statistics across those observations.
    - Reconstruction between pilot-bearing OFDM symbols is performed separately
      by :class:`~sionna.bench.TimeDomainInterpolator`.
    - The paper defines the conditional channel mean but does not provide
      detector-facing error variances. This implementation returns the posterior
      variances implied by the same GMM.

    :param resource_grid: Sionna resource grid defining the OFDM dimensions and
        pilot pattern.
    :param weights: GMM component probabilities with shape
        ``[num_components]``.
    :param means: Real-valued component means with shape
        ``[num_components, 2*num_effective_subcarriers]``. Real channel entries
        precede imaginary channel entries.
    :param covariances: Full real-valued component covariance matrices with shape
        ``[num_components, 2*num_effective_subcarriers,
        2*num_effective_subcarriers]`` using the same stacking convention as
        ``means``.
    :param time_avg: If `True`, average the full-frequency estimates from the
        pilot-bearing OFDM symbols before replicating them across time. If
        `False`, interpolate between the pilot-bearing symbols. Defaults to
        `True`.
    :param precision: Precision used for internal calculations and outputs. If
        `None`, use :attr:`~sionna.phy.config.Config.precision`.
    :param device: Device used for tensor operations. If `None`, use
        :attr:`~sionna.phy.config.Config.device`.
    """
    def __init__(
        self,
        resource_grid: ResourceGrid,
        weights: torch.Tensor,
        means: torch.Tensor,
        covariances: torch.Tensor,
        *,
        time_avg: bool = True,
        precision: Optional[Precision] = None,
        device: Optional[str] = None,
    ) -> None:
        """Initialize full-grid metadata and register the prepared GMM prior."""
        assert isinstance(time_avg, bool), "`time_avg` must be boolean."
        super().__init__(
            resource_grid=resource_grid,
            precision=precision,
            device=device,
        )
        self._num_effective_subcarriers = int(
            resource_grid.num_effective_subcarriers
        )

        self._init_pilot_groups(self._pilot_pattern, group_over=("subcarrier",))
        self._init_full_grid_metadata()
        self._init_pilot_symbol_groups()
        self._time_interpolator = TimeDomainInterpolator(
            self._build_time_interpolation_pilot_pattern(),
            time_avg=time_avg,
        )

        weights = weights.to(dtype=self.dtype, device=self.device)
        means = means.to(dtype=self.dtype, device=self.device)
        covariances = covariances.to(dtype=self.dtype, device=self.device)
        real_dim = 2 * self._num_effective_subcarriers
        assert weights.dim() == 1 and int(weights.numel()) > 0, (
            "GMM weights must be a nonempty rank-1 tensor."
        )
        num_components = int(weights.shape[0])
        assert means.shape == (num_components, real_dim), (
            "Full-grid GMM means have incompatible shape."
        )
        assert covariances.shape == (num_components, real_dim, real_dim), (
            "Full-grid GMM covariances have incompatible shape."
        )
        self.register_buffer("_weights", weights)
        self.register_buffer("_means", means)
        self.register_buffer("_covariances", covariances)

    def _init_full_grid_metadata(self) -> None:
        """Register pilot values and observed full-vector coordinates."""
        pilot_values = self._gather_pilot_groups(
            self._pilot_pattern.pilots.to(self.device)
        ).to(dtype=self.cdtype, device=self.device)
        pilot_power = pilot_values.abs().square().to(dtype=self.dtype)
        assert bool(torch.all(pilot_power > 0.0)), (
            "Full-grid spectral GMM requires nonzero pilot symbols."
        )

        self.register_buffer("_group_pilot_values", pilot_values)
        self.register_buffer(
            "_group_subcarrier_indices",
            self._pilot_group_varying_coord("subcarrier"),
        )

    def _init_pilot_symbol_groups(self) -> None:
        """Record the common number of pilot-bearing symbols per stream."""
        num_streams_per_tx = int(self._pilot_pattern.num_streams_per_tx)
        stream_indices = (
            self._pilot_group_constant_coord("tx") * num_streams_per_tx
            + self._pilot_group_constant_coord("stream")
        )
        num_streams = int(self._pilot_pattern.num_tx) * num_streams_per_tx
        group_counts = torch.bincount(stream_indices, minlength=num_streams)
        assert bool(torch.all(group_counts == group_counts[0])), (
            "Full-grid spectral GMM requires the same number of "
            "pilot OFDM symbols for every stream."
        )
        num_pilot_symbols = int(group_counts[0])
        assert num_pilot_symbols > 0, (
            "Full-grid spectral GMM requires at least one pilot OFDM symbol."
        )
        self._num_pilot_symbol_groups_per_stream = num_pilot_symbols

    def _build_time_interpolation_pilot_pattern(self) -> PilotPattern:
        """Return a pilot pattern that marks full pilot-bearing OFDM symbols."""
        original_mask = self._pilot_pattern.mask.to(torch.bool)
        full_symbol_mask = torch.any(original_mask, dim=-1, keepdim=True).expand(
            -1,
            -1,
            -1,
            self._num_effective_subcarriers,
        )
        num_pilot_symbols = int(full_symbol_mask[0, 0].sum().item())
        pilots = torch.ones(
            int(self._pilot_pattern.num_tx),
            int(self._pilot_pattern.num_streams_per_tx),
            num_pilot_symbols,
            dtype=self.cdtype,
            device=full_symbol_mask.device,
        )
        return PilotPattern(
            full_symbol_mask,
            pilots,
            precision=self.precision,
            device=self.device,
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
        y_pilots = torch.gather(
            y_eff_flat.unsqueeze(3)
            .unsqueeze(4)
            .expand(-1, -1, -1, pilot_ind.shape[3], pilot_ind.shape[4], -1),
            -1,
            pilot_ind,
        )

        h_hat, err_var = self._estimate_pilot_symbol_subcarriers(y_pilots, no)
        return self._time_interpolator(h_hat, err_var)

    def _estimate_pilot_symbol_subcarriers(
        self,
        y_pilots: torch.Tensor,
        no: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Estimate all subcarriers on pilot-bearing OFDM symbols."""
        h_ls, ls_err = LSChannelEstimator.estimate_at_pilot_locations(
            self,
            y_pilots,
            no,
        )
        grouped_h_ls = self._gather_pilot_groups(h_ls)
        grouped_ls_err = self._gather_pilot_groups(ls_err)

        group_estimates = []
        group_variances = []
        for group_index in range(self._pilot_group_num_groups):
            estimate, variance = self._estimate_full_spectral_group(
                grouped_h_ls[..., group_index, :],
                grouped_ls_err[..., group_index, :],
                self._group_subcarrier_indices[group_index],
            )
            group_estimates.append(estimate)
            group_variances.append(variance)

        # _PilotGroupMixin orders these groups by (tx, stream, symbol), which
        # matches the three output dimensions introduced by this reshape.
        output_shape = (
            *y_pilots.shape[:-1],
            self._num_pilot_symbol_groups_per_stream,
            self._num_effective_subcarriers,
        )
        return (
            torch.stack(group_estimates, dim=-2).reshape(output_shape),
            torch.stack(group_variances, dim=-2).reshape(output_shape),
        )

    def _estimate_full_spectral_group(
        self,
        values: torch.Tensor,
        error_variance: torch.Tensor,
        observed_subcarriers: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""Estimate a full spectral vector from observed pilot subcarriers.

        :param values: Complex LS pilot observations with shape
            ``[..., num_observed_subcarriers]``.
        :param error_variance: Pilot LS error variances with the same shape as
            ``values``.
        :param observed_subcarriers: Effective-subcarrier coordinates observed
            by this pilot group, shape ``[num_observed_subcarriers]``.

        :output estimates: Complex full-grid estimates with shape
            ``[..., num_effective_subcarriers]``.
        :output variances: Detector-facing variances with the same shape.
        """
        leading_shape = values.shape[:-1]
        num_observed = int(values.shape[-1])
        num_full = self._num_effective_subcarriers
        observations = torch.cat([values.real, values.imag], dim=-1).reshape(
            -1,
            2 * num_observed,
        )
        num_observations = int(observations.shape[0])
        error_variance = error_variance.reshape(-1, num_observed)
        assert error_variance.shape[0] in (1, num_observations), (
            "`error_variance` must have one row or one row per observation."
        )
        if error_variance.shape[0] == 1:
            error_variance = error_variance.expand(num_observations, num_observed)

        observed_subcarriers = observed_subcarriers.to(
            device=values.device,
            dtype=torch.int64,
        )
        observed_real_indices = torch.cat(
            [observed_subcarriers, observed_subcarriers + num_full],
            dim=0,
        )
        real_error_variance = 0.5 * torch.cat(
            [error_variance, error_variance],
            dim=-1,
        ).to(dtype=self.dtype, device=values.device)

        component_posteriors = []
        posterior_diagonals = []
        log_probs = []
        full_diagonal_indices = torch.arange(
            2 * num_full,
            dtype=torch.int64,
            device=values.device,
        )

        for component_index in range(int(self._weights.shape[0])):
            mean_full = self._means[component_index]
            covariance = self._covariances[component_index]
            mean_observed = mean_full.index_select(0, observed_real_indices)
            covariance_observed = covariance.index_select(
                0,
                observed_real_indices,
            ).index_select(1, observed_real_indices)
            covariance_observed_full = covariance.index_select(
                0,
                observed_real_indices,
            )
            covariance_full_observed = covariance.index_select(
                1,
                observed_real_indices,
            )

            covariance_observed_noise = covariance_observed.unsqueeze(0).expand(
                num_observations,
                -1,
                -1,
            ).clone()
            diagonal = torch.diagonal(
                covariance_observed_noise,
                dim1=-2,
                dim2=-1,
            )
            diagonal.add_(real_error_variance)
            cholesky = torch.linalg.cholesky(covariance_observed_noise)

            residual = observations - mean_observed
            solved_residual = torch.cholesky_solve(
                residual.unsqueeze(-1),
                cholesky,
            ).squeeze(-1)
            # Equation (13) for the real-stacked subcarrier-selection model.
            posterior_mean = mean_full + torch.matmul(
                solved_residual,
                covariance_full_observed.transpose(0, 1),
            )
            component_posteriors.append(posterior_mean)

            solved_covariance = torch.cholesky_solve(
                covariance_observed_full.unsqueeze(0).expand(
                    num_observations,
                    -1,
                    -1,
                ),
                cholesky,
            )
            variance_reduction = (
                covariance_observed_full.unsqueeze(0) * solved_covariance
            ).sum(dim=1)
            covariance_diagonal = covariance[
                full_diagonal_indices,
                full_diagonal_indices,
            ]
            posterior_diagonals.append(
                (covariance_diagonal - variance_reduction).clamp_min(0.0)
            )

            mahalanobis_distance = (residual * solved_residual).sum(dim=-1)
            log_determinant = 2.0 * torch.log(
                torch.diagonal(cholesky, dim1=-2, dim2=-1)
            ).sum(dim=-1)
            real_dim = int(observed_real_indices.numel())
            log_norm = torch.log(self._weights[component_index].clamp_min(1e-12))
            log_norm = log_norm - 0.5 * (
                real_dim * math.log(2.0 * math.pi) + log_determinant
            )
            log_probs.append(log_norm - 0.5 * mahalanobis_distance)

        # responsibilities: [num_observations, num_components]
        # posterior_means/posterior_diagonals: [num_observations, K, 2*num_full]
        # Equations (15) and (16): normalize component likelihoods and mix
        # their full-vector conditional means.
        log_prob = torch.stack(log_probs, dim=-1)
        responsibilities = torch.softmax(log_prob, dim=-1)
        posterior_means = torch.stack(component_posteriors, dim=1)
        posterior_diagonal = torch.stack(posterior_diagonals, dim=1)

        estimated_real = (responsibilities.unsqueeze(-1) * posterior_means).sum(dim=1)
        variance_real = (
            responsibilities.unsqueeze(-1)
            * (
                posterior_diagonal
                + (posterior_means - estimated_real.unsqueeze(1)).square()
            )
        ).sum(dim=1)

        estimated_complex = torch.complex(
            estimated_real[:, :num_full],
            estimated_real[:, num_full:],
        ).to(values.dtype)
        variance_complex = variance_real[:, :num_full] + variance_real[:, num_full:]
        return (
            estimated_complex.reshape(*leading_shape, num_full),
            variance_complex.reshape(*leading_shape, num_full).to(values.real.dtype),
        )
