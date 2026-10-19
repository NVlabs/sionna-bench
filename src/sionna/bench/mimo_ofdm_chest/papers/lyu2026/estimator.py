# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pilot-comb Toeplitz-SDP adaptation of Lyu et al. 2026."""

from __future__ import annotations

import math
from typing import Optional

import torch
from sionna.phy import Precision
from sionna.phy.ofdm import (
    BaseChannelInterpolator,
    BasePilotChannelEstimator,
    LSChannelEstimator,
    ResourceGrid,
)
from sionna.phy.ofdm.channel_estimation import _PilotGroupMixin


class SparseDiffuseChannelEstimator(BasePilotChannelEstimator, _PilotGroupMixin):
    r"""Channel estimator that leverages sparsity through hyyrid diffuse/sparse atomic-norm minimization.

    This estimator implements the method proposed in:

    .. container:: bibliography-box

        .. bibliography::
            :filter: key == "lyu2026atomic"
            :list: bullet

    .. rubric:: TL;DR

    The paper models the channel frequency response as the sum of a sparse
    specular component and a diffuse component,
    :math:`\mathbf{h} = \mathbf{h}_{\mathrm{s}} + \mathbf{D}\boldsymbol{\gamma}`.
    The continuous-delay representation of :math:`\mathbf{h}_{\mathrm{s}}` is
    promoted to be sparse through atomic-norm regularization. The diffuse component
    :math:`\mathbf{D}\boldsymbol{\gamma}` lies in the column space of the known
    Fourier dictionary :math:`\mathbf{D}`, whose columns represent uniformly spaced
    delay taps, and its coefficients satisfy the energy constraint
    :math:`\lVert\boldsymbol{\gamma}\rVert_2^2 \leq E_{\mathrm{d}}`.

    This implementation first divides the received pilots by their known values.
    Repeated pilot-symbol observations belonging to the same transmit stream are
    then averaged independently for every receive antenna. Each averaged pilot
    comb is treated as a reduced OFDM grid, and the corresponding reduced-grid
    form of Program P2 is solved approximately using batched Torch ADMM. The
    estimator returns the resulting regularized reconstruction
    :math:`\mathbf{h}_{\mathrm{s}}+\mathbf{D}\boldsymbol{\gamma}` and uses the
    configured Sionna interpolator to reconstruct the remaining resource elements.

    .. rubric:: Supported System Geometry

    The estimator supports different FFT sizes, UE counts, pilot-bearing OFDM
    symbols, uniform pilot-comb spacings, transmit-stream counts, and receive
    antenna counts, provided that the following hold:

    - Every active pilot group is a uniformly spaced subcarrier comb.
    - All pilot groups have the same number of active subcarriers.
    - All active pilot values are non-zero.
    - Repeated pilot groups belonging to the same transmit stream represent the
      same ordered subcarrier comb because they are averaged elementwise.
    - ``diffuse_num_taps`` is between one and the common pilot-comb length.

    Repeated pilot observations are combined independently for each transmit
    stream and receive antenna. No common sparse support is imposed across receive
    antennas.

    .. rubric:: Deviations from the Paper

    - The paper assumes complete observations over all OFDM subcarriers. This
      implementation instead applies the paper's Program P2 to each sparse
      pilot comb, interpreted as a reduced OFDM grid with a correspondingly
      larger effective subcarrier spacing.
    - The paper does not define interpolation. Off-pilot reconstruction is
      therefore delegated to a configurable Sionna interpolator.
    - Program P2 is solved approximately using a fixed number of batched ADMM
      iterations rather than an exact generic semidefinite-program solver.
    - Continuous support extraction from Equation (16) and projection debiasing
      from Equation (17) are not implemented. The regularized, and therefore
      generally biased, P2 reconstruction is returned directly.
    - Receive antennas are estimated independently; the implementation does not
      introduce a multiple-measurement-vector or shared-support model.
    - The paper does not specify detector-facing marginal error variances. For
      averaging weights :math:`w_g`, this implementation uses

    .. math::

        c_{\mathrm{err}}\,\sigma^2
        \sum_g \frac{w_g^2}{|p_g|^2},

    followed by the configured Sionna interpolation.

    :param resource_grid: Resource grid defining the OFDM dimensions and pilot
        pattern.
    :param diffuse_num_taps: Number of columns in the reduced-grid diffuse
        dictionary. If `None`, use the complete pilot-comb length.
    :param tau: Constant weight of the atomic-norm Toeplitz term. Used when
        ``tau_noise_scale`` is `None`.
    :param tau_noise_scale: Optional multiplier used to derive the regularization
        weight from the mean input noise variance. If provided, the effective
        weight is ``clamp(tau_noise_scale*mean(no), tau_min, tau_max)``.
    :param tau_min: Lower bound applied to the noise-dependent regularization
        weight.
    :param tau_max: Upper bound applied to the noise-dependent regularization
        weight.
    :param diffuse_energy: Diffuse-coefficient energy bound
        :math:`E_{\mathrm{d}}`.
    :param err_var_scale: Non-negative scale applied to the detector-facing error
        variance.
    :param admm_rho: Positive augmented-Lagrangian penalty used by ADMM.
    :param admm_num_iters: Number of ADMM iterations performed for every
        observation.
    :param interpolation_type: Sionna interpolation policy used after estimating
        the pilot comb. Ignored when ``interpolator`` is provided.
    :param interpolator: Custom Sionna channel interpolator. If `None`, the
        interpolator selected by ``interpolation_type`` is used.
    :param precision: Precision used for internal calculations and outputs. If
        `None`, :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device used for tensor operations. If `None`,
        :attr:`~sionna.phy.config.Config.device` is used.
    """
    def __init__(
        self,
        resource_grid: ResourceGrid,
        *,
        diffuse_num_taps: Optional[int] = 3,
        tau: float = 3.0,
        tau_noise_scale: Optional[float] = 6.55,
        tau_min: float = 1e-6,
        tau_max: float = 3.0,
        diffuse_energy: float = 0.807,
        err_var_scale: float = 1.0,
        admm_rho: float = 0.08,
        admm_num_iters: int = 18,
        interpolation_type: str = "lin_time_avg",
        interpolator: Optional[BaseChannelInterpolator] = None,
        precision: Optional[Precision] = None,
        device: Optional[str] = None,
    ) -> None:
        """Initialize reduced-grid SDP parameters and pilot-comb metadata."""
        assert isinstance(tau, (int, float)) and math.isfinite(tau), (
            "`tau` must be finite."
        )
        assert tau >= 0.0, "`tau` must be non-negative."
        if tau_noise_scale is not None:
            assert isinstance(tau_noise_scale, (int, float)) and math.isfinite(
                tau_noise_scale
            ), "`tau_noise_scale` must be finite."
            assert tau_noise_scale >= 0.0, (
                "`tau_noise_scale` must be non-negative."
            )
        assert isinstance(tau_min, (int, float)) and math.isfinite(tau_min), (
            "`tau_min` must be finite."
        )
        assert tau_min >= 0.0, "`tau_min` must be non-negative."
        assert isinstance(tau_max, (int, float)), "`tau_max` must be a float."
        assert tau_max >= tau_min, "`tau_max` must be at least `tau_min`."
        assert isinstance(diffuse_energy, (int, float)) and math.isfinite(
            diffuse_energy
        ), "`diffuse_energy` must be finite."
        assert diffuse_energy >= 0.0, "`diffuse_energy` must be non-negative."
        assert isinstance(err_var_scale, (int, float)) and math.isfinite(
            err_var_scale
        ), "`err_var_scale` must be finite."
        assert err_var_scale >= 0.0, "`err_var_scale` must be non-negative."
        assert isinstance(admm_rho, (int, float)) and math.isfinite(admm_rho), (
            "`admm_rho` must be finite."
        )
        assert admm_rho > 0.0, "`admm_rho` must be positive."
        assert isinstance(admm_num_iters, int) and admm_num_iters >= 1, (
            "`admm_num_iters` must be a positive integer."
        )
        assert interpolation_type in ("nn", "lin", "lin_time_avg"), (
            "Unsupported `interpolation_type`."
        )

        super().__init__(
            resource_grid=resource_grid,
            interpolation_type=interpolation_type,
            interpolator=interpolator,
            precision=precision,
            device=device,
        )
        self._tau = float(tau)
        self._tau_noise_scale = (
            None if tau_noise_scale is None else float(tau_noise_scale)
        )
        self._tau_min = float(tau_min)
        self._tau_max = float(tau_max)
        self._diffuse_energy = float(diffuse_energy)
        self._err_var_scale = float(err_var_scale)
        self._admm_rho = float(admm_rho)
        self._admm_num_iters = int(admm_num_iters)

        self._init_pilot_groups(self._pilot_pattern, group_over=("subcarrier",))
        self._init_comb_metadata(resource_grid)

        self._diffuse_num_taps = (
            int(diffuse_num_taps)
            if diffuse_num_taps is not None
            else self._pilot_group_len
        )
        assert 1 <= self._diffuse_num_taps <= self._pilot_group_len, (
            f"`diffuse_num_taps` must be between 1 and {self._pilot_group_len}."
        )
        self.register_buffer(
            "_diffuse_dictionary",
            self._build_diffuse_dictionary(
                self._pilot_group_len,
                self._diffuse_num_taps,
            ),
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
        no = torch.as_tensor(no, dtype=self.dtype, device=y_pilots.device)
        tau = self._effective_tau(no)
        h_ls, _ = LSChannelEstimator.estimate_at_pilot_locations(
            self,
            y_pilots,
            no,
        )
        grouped_ls = self._gather_pilot_groups(h_ls)

        grouped_h_hat = self._average_and_solve_pilot_groups(grouped_ls, tau)
        grouped_err_var = self._grouped_error_variance(
            no,
            y_pilots.shape[0],
            y_pilots.shape[1],
            y_pilots.shape[2],
            y_pilots.device,
        )

        h_hat = self._scatter_pilot_groups(torch.zeros_like(y_pilots), grouped_h_hat)
        err_var = self._scatter_pilot_groups(
            torch.zeros_like(y_pilots, dtype=self.dtype),
            grouped_err_var,
        )
        return h_hat, err_var

    def _effective_tau(self, no: torch.Tensor) -> torch.Tensor:
        """Return the constant or noise-scaled Toeplitz regularization weight."""
        if self._tau_noise_scale is None:
            return torch.as_tensor(self._tau, dtype=self.dtype, device=no.device)
        tau = self._tau_noise_scale * no.mean()
        return torch.clamp(tau, min=self._tau_min, max=self._tau_max)

    def _init_comb_metadata(self, resource_grid) -> None:
        """Register pilot values and validate the reduced-grid comb model."""
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
            "Lyu pilot-comb P2 adaptation requires uniform pilot spacing."
        )

        pilot_values = self._gather_pilot_groups(
            self._pilot_pattern.pilots.to(self.device)
        ).to(dtype=self.cdtype, device=self.device)
        pilot_power = pilot_values.abs().square().to(dtype=self.dtype)
        assert bool(torch.all(pilot_power > 0.0)), (
            "Lyu pilot-comb P2 adaptation requires nonzero pilots."
        )

        self.register_buffer("_group_pilot_power", pilot_power)

        group_tx_indices = self._pilot_group_constant_coord("tx")
        group_stream_indices = self._pilot_group_constant_coord("stream")
        group_pairs = torch.stack(
            [group_tx_indices, group_stream_indices],
            dim=-1,
        )
        unique_pairs = torch.unique(group_pairs, dim=0, sorted=True)
        pair_membership = torch.all(
            unique_pairs[:, None, :] == group_pairs[None, :, :],
            dim=-1,
        )
        averaging_weights = pair_membership.to(self.dtype)
        averaging_weights = averaging_weights / averaging_weights.sum(
            dim=-1,
            keepdim=True,
        )
        self.register_buffer("_group_averaging_weights", averaging_weights)
        self.register_buffer(
            "_group_pair_indices",
            pair_membership.to(torch.int64).argmax(dim=0),
        )

    def _build_diffuse_dictionary(
        self,
        num_observations: int,
        num_taps: int,
    ) -> torch.Tensor:
        """Return the reduced-grid diffuse steering dictionary."""
        frequency_indices = torch.arange(
            num_observations,
            dtype=self.dtype,
            device=self.device,
        ) - (num_observations - 1.0) / 2.0
        tap_indices = torch.arange(num_taps, dtype=self.dtype, device=self.device)
        phases = 2.0 * math.pi * frequency_indices[:, None] * tap_indices[None, :]
        phases = phases / float(num_taps)
        return torch.exp(1j * phases).to(dtype=self.cdtype)

    def _average_and_solve_pilot_groups(
        self,
        grouped_ls: torch.Tensor,
        tau: torch.Tensor,
    ) -> torch.Tensor:
        """Average repeated pilot symbols and solve one vector per link."""
        assert grouped_ls.dim() == 5, (
            "Grouped pilots must have shape "
            "[batch, num_rx, num_rx_ant, num_groups, group_len]."
        )
        observations = torch.einsum(
            "...gk,pg->...pk",
            grouped_ls,
            self._group_averaging_weights.to(grouped_ls.dtype),
        )
        pair_estimates = self._solve_grouped_admm(observations, tau)
        return pair_estimates.index_select(3, self._group_pair_indices)

    def _solve_grouped_admm(
        self,
        observations: torch.Tensor,
        tau: torch.Tensor,
    ) -> torch.Tensor:
        """Solve all averaged pilot observations with batched Torch ADMM."""
        flat_observations = observations.reshape(-1, observations.shape[-1])
        flat_estimates = self._solve_admm(
            flat_observations,
            self._diffuse_dictionary.to(
                dtype=observations.dtype,
                device=observations.device,
            ),
            tau=tau,
            diffuse_energy=self._diffuse_energy,
            rho=self._admm_rho,
            num_iters=self._admm_num_iters,
        )
        return flat_estimates.reshape_as(observations)

    @classmethod
    def _solve_admm(
        cls,
        observation: torch.Tensor,
        diffuse_dictionary: torch.Tensor,
        *,
        tau: float,
        diffuse_energy: float,
        rho: float = 0.03,
        num_iters: int = 300,
    ) -> torch.Tensor:
        """Run batched vector ADMM on the current Torch device."""
        squeeze_batch = observation.dim() == 1
        if squeeze_batch:
            observation = observation.unsqueeze(0)
        observation = torch.as_tensor(observation)
        diffuse_dictionary = torch.as_tensor(
            diffuse_dictionary,
            dtype=observation.dtype,
            device=observation.device,
        )
        num_problems, num_observations = observation.shape

        row_index = torch.arange(num_observations, device=observation.device)
        offset = row_index[:, None] - row_index[None, :]
        abs_offset = offset.abs()
        abs_offset_flat = abs_offset.reshape(-1)
        upper_triangle_mask = (offset < 0).unsqueeze(0)
        offset_counts = torch.cat(
            [
                torch.as_tensor(
                    [num_observations],
                    dtype=observation.real.dtype,
                    device=observation.device,
                ),
                2.0
                * torch.arange(
                    num_observations - 1,
                    0,
                    -1,
                    dtype=observation.real.dtype,
                    device=observation.device,
                ),
            ]
        )

        gamma = cls._constrained_least_squares(
            diffuse_dictionary,
            observation,
            diffuse_energy,
        )
        sparse_response = observation - torch.matmul(
            diffuse_dictionary.unsqueeze(0),
            gamma.unsqueeze(-1),
        ).squeeze(-1)
        iota_zero = sparse_response.abs().mean(dim=-1).real
        iota_tail = torch.zeros(
            num_problems,
            num_observations - 1,
            dtype=observation.dtype,
            device=observation.device,
        )
        lift_scalar = sparse_response.abs().square().sum(dim=-1).real
        primal_block = cls._build_psd_block(
            iota_zero,
            iota_tail,
            sparse_response,
            lift_scalar,
            abs_offset_flat,
            upper_triangle_mask,
        )
        psd_block = cls._project_hermitian_psd(primal_block)
        scaled_dual = torch.zeros_like(psd_block)

        for _ in range(num_iters):
            target_block = psd_block - scaled_dual
            top_left = target_block[:, :num_observations, :num_observations]
            offdiag_target = 0.5 * (
                target_block[:, :num_observations, num_observations]
                + target_block[:, num_observations, :num_observations].conj()
            )

            diagonal_mean = top_left.diagonal(dim1=-2, dim2=-1).real.mean(dim=-1)
            iota_zero = torch.clamp(
                diagonal_mean - tau / (rho * num_observations),
                min=0.0,
            )
            first_col = cls._toeplitz_first_column(
                top_left,
                abs_offset_flat,
                upper_triangle_mask,
                offset_counts,
            )
            iota_tail = first_col[:, 1:]
            lift_scalar = torch.clamp(
                target_block[:, -1, -1].real - tau / rho,
                min=0.0,
            )

            gamma_target = observation - offdiag_target
            gamma = cls._constrained_least_squares(
                diffuse_dictionary,
                gamma_target,
                diffuse_energy,
            )
            residual_without_sparse = observation - torch.matmul(
                diffuse_dictionary.unsqueeze(0),
                gamma.unsqueeze(-1),
            ).squeeze(-1)
            sparse_response = (
                residual_without_sparse + rho * offdiag_target
            ) / (1.0 + rho)

            primal_block = cls._build_psd_block(
                iota_zero,
                iota_tail,
                sparse_response,
                lift_scalar,
                abs_offset_flat,
                upper_triangle_mask,
            )
            projected_input = primal_block + scaled_dual
            psd_block = cls._project_hermitian_psd(projected_input)
            scaled_dual = scaled_dual + primal_block - psd_block

        estimate = sparse_response + torch.matmul(
            diffuse_dictionary.unsqueeze(0),
            gamma.unsqueeze(-1),
        ).squeeze(-1)
        if squeeze_batch:
            estimate = estimate.squeeze(0)
        return estimate

    @staticmethod
    def _toeplitz_first_column(
        matrix: torch.Tensor,
        abs_offset_flat: torch.Tensor,
        upper_triangle_mask: torch.Tensor,
        offset_counts: torch.Tensor,
    ) -> torch.Tensor:
        """Project batched Hermitian matrices onto Toeplitz first columns."""
        num_problems = int(matrix.shape[0])
        values = torch.where(upper_triangle_mask, matrix.conj(), matrix)
        values = values.reshape(num_problems, -1)
        sums = torch.zeros(
            num_problems,
            int(offset_counts.numel()),
            dtype=matrix.dtype,
            device=matrix.device,
        )
        scatter_index = abs_offset_flat.reshape(1, -1).expand(num_problems, -1)
        sums.scatter_add_(1, scatter_index, values)
        return sums / offset_counts.reshape(1, -1)

    @classmethod
    def _build_psd_block(
        cls,
        iota_zero: torch.Tensor,
        iota_tail: torch.Tensor,
        sparse_response: torch.Tensor,
        lift_scalar: torch.Tensor,
        abs_offset_flat: torch.Tensor,
        upper_triangle_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Build batched vector-ADMM lifted blocks on the current device."""
        num_problems, num_observations = sparse_response.shape
        first_col = torch.cat(
            [iota_zero.to(sparse_response.dtype).reshape(num_problems, 1), iota_tail],
            dim=1,
        )
        toeplitz_values = first_col.index_select(1, abs_offset_flat).reshape(
            num_problems,
            num_observations,
            num_observations,
        )
        toeplitz_matrix = torch.where(
            upper_triangle_mask,
            toeplitz_values.conj(),
            toeplitz_values,
        )
        block = torch.empty(
            num_problems,
            num_observations + 1,
            num_observations + 1,
            dtype=sparse_response.dtype,
            device=sparse_response.device,
        )
        block[:, :num_observations, :num_observations] = toeplitz_matrix
        block[:, :num_observations, num_observations] = sparse_response
        block[:, num_observations, :num_observations] = sparse_response.conj()
        block[:, num_observations, num_observations] = lift_scalar.to(
            sparse_response.dtype
        )
        return block

    @staticmethod
    def _project_hermitian_psd(matrix: torch.Tensor) -> torch.Tensor:
        """Project batched Hermitian matrices onto the PSD cone."""
        hermitian = 0.5 * (matrix + matrix.mH)
        eigenvalues, eigenvectors = torch.linalg.eigh(hermitian)
        eigenvalues = torch.clamp(eigenvalues, min=0.0)
        return (
            eigenvectors * eigenvalues.to(eigenvectors.dtype).unsqueeze(-2)
        ) @ eigenvectors.mH

    @staticmethod
    def _constrained_least_squares(
        dictionary: torch.Tensor,
        target: torch.Tensor,
        energy: float,
    ) -> torch.Tensor:
        """Solve batched vector least squares with an ``l2`` constraint."""
        squeeze_batch = target.dim() == 1
        if squeeze_batch:
            target = target.unsqueeze(0)
        num_problems = int(target.shape[0])
        num_coefficients = int(dictionary.shape[1])
        if num_coefficients == 0 or energy <= 0.0:
            result = torch.zeros(
                num_problems,
                num_coefficients,
                dtype=target.dtype,
                device=target.device,
            )
            return result.squeeze(0) if squeeze_batch else result

        gram = dictionary.mH @ dictionary
        rhs = torch.matmul(
            dictionary.mH.unsqueeze(0),
            target.unsqueeze(-1),
        ).squeeze(-1)
        gram_batch = gram.unsqueeze(0).expand(num_problems, -1, -1)
        unconstrained = torch.linalg.solve(
            gram_batch,
            rhs.unsqueeze(-1),
        ).squeeze(-1)
        radius = math.sqrt(float(energy))
        norms = torch.linalg.vector_norm(unconstrained, dim=-1)
        needs_constraint = norms > radius

        identity = torch.eye(
            num_coefficients,
            dtype=target.dtype,
            device=target.device,
        )

        def solve_with_lambdas(lambdas: torch.Tensor) -> torch.Tensor:
            shifted = gram.unsqueeze(0) + lambdas.reshape(-1, 1, 1) * identity
            return torch.linalg.solve(
                shifted,
                rhs.unsqueeze(-1),
            ).squeeze(-1)

        lower = torch.zeros(
            num_problems,
            dtype=target.real.dtype,
            device=target.device,
        )
        # For a positive-semidefinite Gram matrix, this upper bound guarantees
        # that the constrained solution norm does not exceed the radius.
        upper = torch.where(
            needs_constraint,
            torch.linalg.vector_norm(rhs, dim=-1) / radius,
            lower,
        )

        for _ in range(50):
            midpoint = 0.5 * (lower + upper)
            candidate = solve_with_lambdas(midpoint)
            candidate_norm = torch.linalg.vector_norm(candidate, dim=-1)
            too_large = torch.logical_and(needs_constraint, candidate_norm > radius)
            lower = torch.where(too_large, midpoint, lower)
            upper = torch.where(
                torch.logical_and(needs_constraint, torch.logical_not(too_large)),
                midpoint,
                upper,
            )

        result = solve_with_lambdas(upper)
        return result.squeeze(0) if squeeze_batch else result

    def _grouped_error_variance(
        self,
        no: torch.Tensor,
        batch_size: int,
        num_rx: int,
        num_rx_ant: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Return the scaled LS observation-variance proxy by pilot group."""
        noise_var = self._expand_noise_variance(
            no,
            batch_size,
            num_rx,
            num_rx_ant,
            device,
        )
        pilot_power = self._group_pilot_power.to(dtype=self.dtype, device=device)
        averaging_weights = self._group_averaging_weights.to(
            dtype=self.dtype,
            device=device,
        )
        pair_variance_factors = torch.einsum(
            "pg,gk->pk",
            averaging_weights.square(),
            pilot_power.reciprocal(),
        )
        group_variance_factors = pair_variance_factors.index_select(
            0,
            self._group_pair_indices,
        )
        return (
            self._err_var_scale
            * noise_var[..., None, None]
            * group_variance_factors.reshape(
                1,
                1,
                1,
                *group_variance_factors.shape,
            )
        )

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
        if no.dim() == 0:
            noise_var = no.reshape(1, 1, 1)
        elif no.dim() == 1:
            assert no.shape[0] in (1, batch_size), (
                "`no` has incompatible batch dimension."
            )
            noise_var = no.reshape(no.shape[0], 1, 1)
        elif no.dim() == 2:
            assert no.shape[0] in (1, batch_size) and no.shape[1] in (1, num_rx), (
                "`no` has incompatible batch or receiver dimension."
            )
            noise_var = no.reshape(no.shape[0], no.shape[1], 1)
        elif no.dim() == 3:
            assert (
                no.shape[0] in (1, batch_size)
                and no.shape[1] in (1, num_rx)
                and no.shape[2] in (1, num_rx_ant)
            ), "`no` has incompatible batch, receiver, or antenna dimension."
            noise_var = no
        else:
            assert False, "`no` must have rank 0, 1, 2, or 3."

        return noise_var.expand(batch_size, num_rx, num_rx_ant)
