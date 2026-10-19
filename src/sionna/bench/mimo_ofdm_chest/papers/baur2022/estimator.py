# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""VAE-leveraged MMSE channel estimator from Baur et al. 2022."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
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


CovarianceMode = str
COVARIANCE_DIAGONAL = "diagonal"
COVARIANCE_FULL = "full"
COVARIANCE_MODES = (COVARIANCE_DIAGONAL, COVARIANCE_FULL)

ENCODER_CHANNELS = (1, 8, 32, 128)
ENCODER_KERNEL_SIZE = 7
ENCODER_STRIDE = 2
ENCODER_PADDING = ENCODER_KERNEL_SIZE // 2
DECODER_CHANNELS = (128, 32, 8, 1)
DECODER_KERNEL_SIZE = 4
DECODER_STRIDE = 2
DECODER_PADDING = 1


class VariationalAutoEncoder(torch.nn.Module):
    """VAE with DFT-domain complex mean and configurable covariance output."""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        *,
        covariance_floor: float = 1e-5,
        covariance_mode: CovarianceMode = COVARIANCE_DIAGONAL,
    ) -> None:
        """Build the convolutional VAE used by ``VAEChannelEstimator``."""
        super().__init__()
        self.input_dim = int(input_dim)
        self.num_complex_values = self.input_dim // 2
        self.latent_dim = int(latent_dim)
        self.covariance_floor = float(covariance_floor)
        self.covariance_mode = str(covariance_mode)
        assert self.input_dim % 2 == 0, "`input_dim` must be even."
        assert self.input_dim > 0, "`input_dim` must be positive."
        assert self.latent_dim > 0, "`latent_dim` must be positive."
        assert self.covariance_mode in COVARIANCE_MODES, (
            f"`covariance_mode` must be one of {COVARIANCE_MODES}."
        )

        self.encoder = torch.nn.Sequential(
            self._conv_block(ENCODER_CHANNELS[0], ENCODER_CHANNELS[1]),
            self._conv_block(ENCODER_CHANNELS[1], ENCODER_CHANNELS[2]),
            self._conv_block(ENCODER_CHANNELS[2], ENCODER_CHANNELS[3]),
        )
        self.encoded_len = self._encoded_length(self.input_dim)
        flattened_dim = ENCODER_CHANNELS[-1] * self.encoded_len
        self.encoder_mean = torch.nn.Linear(flattened_dim, self.latent_dim)
        self.encoder_log_variance = torch.nn.Linear(flattened_dim, self.latent_dim)

        self.decoder_linear = torch.nn.Linear(self.latent_dim, flattened_dim)
        self.decoder_trunk = torch.nn.Sequential(
            self._deconv_block(DECODER_CHANNELS[0], DECODER_CHANNELS[1]),
            self._deconv_block(DECODER_CHANNELS[1], DECODER_CHANNELS[2]),
        )
        self.decoder_mean = torch.nn.ConvTranspose1d(
            DECODER_CHANNELS[2],
            DECODER_CHANNELS[3],
            kernel_size=DECODER_KERNEL_SIZE,
            stride=DECODER_STRIDE,
            padding=DECODER_PADDING,
        )
        self.decoder_log_covariance = torch.nn.ConvTranspose1d(
            DECODER_CHANNELS[2],
            DECODER_CHANNELS[3],
            kernel_size=DECODER_KERNEL_SIZE,
            stride=DECODER_STRIDE,
            padding=DECODER_PADDING,
        )
        if self.covariance_mode == COVARIANCE_FULL:
            num_tril_entries = self.input_dim * (self.input_dim + 1) // 2
            self.decoder_cholesky = torch.nn.Linear(
                flattened_dim,
                num_tril_entries,
            )

    @staticmethod
    def _conv_block(in_channels: int, out_channels: int) -> torch.nn.Sequential:
        """Return one Conv-BN-ReLU encoder block from Section III-D."""
        return torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=ENCODER_KERNEL_SIZE,
                stride=ENCODER_STRIDE,
                padding=ENCODER_PADDING,
            ),
            torch.nn.BatchNorm1d(out_channels),
            torch.nn.ReLU(),
        )

    @staticmethod
    def _deconv_block(in_channels: int, out_channels: int) -> torch.nn.Sequential:
        """Return one transposed-Conv-BN-ReLU decoder block."""
        return torch.nn.Sequential(
            torch.nn.ConvTranspose1d(
                in_channels,
                out_channels,
                kernel_size=DECODER_KERNEL_SIZE,
                stride=DECODER_STRIDE,
                padding=DECODER_PADDING,
            ),
            torch.nn.BatchNorm1d(out_channels),
            torch.nn.ReLU(),
        )

    @staticmethod
    def _encoded_length(input_dim: int) -> int:
        """Return the Conv1d output length after the three encoder blocks."""
        length = int(input_dim)
        for _ in range(3):
            length = (length + 1) // 2
        return length

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return latent mean and log variance for real-stacked input."""
        encoded = self.encoder(x.unsqueeze(1)).flatten(start_dim=1)
        return self.encoder_mean(encoded), self.encoder_log_variance(encoded)

    def _decode_head(
        self,
        head: torch.nn.Module,
        features: torch.Tensor,
    ) -> torch.Tensor:
        """Apply one decoder head and crop or pad to ``input_dim``."""
        out = head(features).squeeze(1)
        if out.shape[-1] > self.input_dim:
            out = out[..., : self.input_dim]
        elif out.shape[-1] < self.input_dim:
            out = torch.nn.functional.pad(out, (0, self.input_dim - out.shape[-1]))
        return out

    def _full_covariance_from_params(self, params: torch.Tensor) -> torch.Tensor:
        """Return a positive-definite full real covariance matrix."""
        batch_size = int(params.shape[0])
        dim = self.input_dim
        tril_rows, tril_cols = torch.tril_indices(
            dim,
            dim,
            device=params.device,
        )
        cholesky_factor = params.new_zeros(batch_size, dim, dim)
        cholesky_factor[:, tril_rows, tril_cols] = params
        diagonal_indices = torch.arange(dim, device=params.device)
        raw_diagonal = cholesky_factor[:, diagonal_indices, diagonal_indices]
        positive_diagonal = (
            torch.nn.functional.softplus(raw_diagonal)
            + self.covariance_floor ** 0.5
        )
        cholesky_factor[:, diagonal_indices, diagonal_indices] = positive_diagonal
        covariance = cholesky_factor @ cholesky_factor.transpose(-2, -1)
        eye = torch.eye(dim, dtype=params.dtype, device=params.device)
        return covariance + self.covariance_floor * eye.unsqueeze(0)

    def decode(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return DFT-domain real-stacked mean and covariance."""
        decoder_features = self.decoder_linear(z)
        features = decoder_features.reshape(
            z.shape[0],
            ENCODER_CHANNELS[-1],
            self.encoded_len,
        )
        features = self.decoder_trunk(features)
        mean = self._decode_head(self.decoder_mean, features)
        if self.covariance_mode == COVARIANCE_FULL:
            covariance = self._full_covariance_from_params(
                self.decoder_cholesky(decoder_features)
            )
            return mean, covariance

        log_covariance_full = self._decode_head(
            self.decoder_log_covariance,
            features,
        )
        log_covariance = log_covariance_full[..., : self.num_complex_values]
        covariance = (
            torch.nn.functional.softplus(log_covariance)
            + self.covariance_floor
        )
        return mean, covariance

    def forward(
        self,
        x: torch.Tensor,
        *,
        sample: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the VAE and return encoder and decoder parameters."""
        latent_mean, latent_log_variance = self.encode(x)
        if sample:
            std = torch.exp(0.5 * latent_log_variance)
            z = latent_mean + torch.randn_like(std) * std
        else:
            z = latent_mean
        decoder_mean, decoder_covariance = self.decode(z)
        return latent_mean, latent_log_variance, decoder_mean, decoder_covariance


class VAEChannelEstimator(BasePilotChannelEstimator, _PilotGroupMixin):
    r"""Channel estimator that leverages a variational autoencoder (VAE) for denoising.

    This implementation follows the method proposed in:

    .. container:: bibliography-box

        .. bibliography::
            :filter: key == "baur2022variational"
            :list: bullet

    .. rubric:: TL;DR

    This estimator is based on a VAE that learns a conditionally Gaussian
    channel model. The VAE operates on the DFT-transformed receive-array
    channel vectors, separately for each pilot-bearing resource element.
    Its decoder returns a transformed-domain conditional mean and covariance matrix.
    The estimator then applies a conditional LMMSE update to obtain the channel
    estimate at that resource element.

    .. rubric:: Supported System Geometry

    The estimator processes every active pilot resource element independently,
    while jointly processing the corresponding receive-antenna vector. It therefore
    supports different resource-grid sizes, pilot layouts, UE counts, OFDM-symbol
    counts, and positive receive-antenna counts, provided that the following hold:

    - The resource grid contains at least one active nonzero pilot.
    - The prepared VAE model state matches num_rx_ant, the latent dimension,
      covariance parameterization, and exact pilot geometry and values.

    Changing the receive-antenna count or pilot configuration requires a
    separately prepared compatible model state.

    Receive arrays that are not ULAs are supported by flattening their antenna
    dimensions and applying the receive DFT over the resulting
    ``num_rx_ant`` axis.

    .. rubric:: Deviations from the Paper

    - The paper assumes a uniform linear array (ULA) at the receive antennas,
      which motivates the use of a DFT over the receive antennas. This implementation
      supports rectangular receive antenna arrays and uses a 1D DFT over Sionna's
      flattened ``num_rx_ant`` axis.
    - The paper assumes a diagonal covariance matrix in the transform that is motivated
      by the assumption of a receive-side ULA. This implementation supports a full
      real-valued covariance matrix in the transform domain.
    - The paper omits certain architectural details, such as convolutional padding
      and latent dimension, so the faithfulness of the implementation in this regard
      cannot be guaranteed.
    - The paper dos not address interpolation. This implementation has to be combined
      with a dedicated interpolator.
    - The paper does not specify detector-facing error variances. Due to the conditionally
      Gaussian channel model, the conditional variances have an analytic expression, which
      is used in this implementation.

    :param resource_grid: Sionna resource grid defining the pilot pattern.
    :param model_states: Prepared VAE state dictionaries. Every state must use
        the configured VAE architecture and receive-vector length.
    :param num_rx_ant: Number of receive antennas.
    :param model_noise_variances: Noise variance associated with each
        state, shape ``[num_models]``. These values select SNR-specific states.
        If `None`, exactly one mixed-SNR state must be supplied.
    :param latent_dim: Latent dimension of the VAEs.
    :param err_var_scale: Multiplicative scale applied to the detector-facing
        posterior marginal error variances.
    :param covariance_floor: Positive floor added to decoder covariance output.
    :param covariance_mode: Decoder covariance structure. ``"diagonal"``
        uses the DFT-domain diagonal covariance from the paper. ``"full"``
        uses a full real-valued covariance matrix in the transform domain.
    :param interpolation_type: Sionna interpolation policy. This argument is
        ignored when ``interpolator`` is provided.
    :param interpolator: Custom Sionna channel interpolator. If `None`, the
        interpolator selected by ``interpolation_type`` is used.
    :param precision: Precision used for Sionna channel estimation. If `None`,
        :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device for Sionna channel estimation. If `None`,
        :attr:`~sionna.phy.config.Config.device` is used.
    """

    def __init__(
        self,
        resource_grid: ResourceGrid,
        model_states: Sequence[Mapping[str, torch.Tensor]],
        num_rx_ant: int,
        model_noise_variances: Optional[torch.Tensor] = None,
        *,
        latent_dim: int = 16,
        err_var_scale: float = 0.541,
        covariance_floor: float = 1e-5,
        covariance_mode: CovarianceMode = COVARIANCE_FULL,
        interpolation_type: str = "lin_time_avg",
        interpolator: Optional[BaseChannelInterpolator] = None,
        precision: Optional[Precision] = None,
        device: Optional[str] = None,
    ) -> None:
        """Initialize pilot groups and install prepared VAE states."""
        super().__init__(
            resource_grid=resource_grid,
            interpolation_type=interpolation_type,
            interpolator=interpolator,
            precision=precision,
            device=device,
        )
        assert len(model_states) > 0, "At least one VAE state is required."
        assert num_rx_ant >= 1, "`num_rx_ant` must be positive."
        assert latent_dim >= 1, "`latent_dim` must be positive."
        assert err_var_scale >= 0.0, "`err_var_scale` must be nonnegative."
        assert covariance_floor > 0.0, "`covariance_floor` must be positive."
        assert covariance_mode in COVARIANCE_MODES, (
            f"`covariance_mode` must be one of {COVARIANCE_MODES}."
        )

        self._latent_dim = int(latent_dim)
        self._err_var_scale = float(err_var_scale)
        self._covariance_floor = float(covariance_floor)
        self._covariance_mode = str(covariance_mode)
        self._init_pilot_groups(self._pilot_pattern, group_over=())
        self._num_rx_ant = int(num_rx_ant)
        input_dim = 2 * self._num_rx_ant
        self._input_dim = input_dim
        self.register_buffer(
            "_receive_dft",
            self._build_unitary_dft(self._num_rx_ant),
        )
        self.register_buffer(
            "_real_idft_transform",
            self._build_real_idft_transform(),
        )

        self._vaes = torch.nn.ModuleList(
            self._build_vae(state) for state in model_states
        )
        if model_noise_variances is None:
            assert len(model_states) == 1, (
                "Multiple VAE states require `model_noise_variances`."
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
                "`model_noise_variances` must have one entry per VAE state."
            )
            assert bool(torch.all(noise_variances > 0.0)), (
                "`model_noise_variances` must be positive."
            )
        self.register_buffer("_vae_noise_variances", noise_variances)

    def _build_unitary_dft(self, dim: int) -> torch.Tensor:
        """Return the unitary DFT matrix over Sionna's receive-antenna axis."""
        index = torch.arange(dim, dtype=self.dtype, device=self.device)
        row = index.reshape(dim, 1)
        col = index.reshape(1, dim)
        phase = -2.0 * torch.pi * row * col / float(dim)
        matrix = torch.complex(torch.cos(phase), torch.sin(phase))
        return matrix.to(dtype=self.cdtype) / (float(dim) ** 0.5)

    def _build_real_idft_transform(self) -> torch.Tensor:
        """Return the real-stacked matrix for the inverse receive DFT."""
        inverse_dft = self._receive_dft.conj()
        real = inverse_dft.real.to(self.dtype)
        imag = inverse_dft.imag.to(self.dtype)
        top = torch.cat([real, imag], dim=1)
        bottom = torch.cat([-imag, real], dim=1)
        return torch.cat([top, bottom], dim=0)

    def _build_vae(self, state: Mapping[str, torch.Tensor]) -> VariationalAutoEncoder:
        """Build a VAE instance and load a cached state dictionary."""
        vae = VariationalAutoEncoder(
            self._input_dim,
            self._latent_dim,
            covariance_floor=self._covariance_floor,
            covariance_mode=self._covariance_mode,
        )
        vae.load_state_dict(state)
        vae.to(device=self.device, dtype=torch.float32)
        vae.eval()
        return vae

    @torch.compiler.disable
    def _select_vae_index(self, no: torch.Tensor) -> int:
        """Select one prepared SNR-specific state outside compiled inference."""
        noise_variance = no.to(
            dtype=self._vae_noise_variances.dtype,
            device=self._vae_noise_variances.device,
        ).reshape(-1).mean()
        log_distance = torch.abs(
            torch.log(noise_variance) - torch.log(self._vae_noise_variances)
        )
        model_index = int(torch.argmin(log_distance).item())
        assert bool(log_distance[model_index] < 1e-3), (
            "No SNR-specific Baur VAE is available for the current noise "
            "variance."
        )
        return model_index

    def _dft(self, values: torch.Tensor) -> torch.Tensor:
        """Apply the unitary DFT to row-wise receive-array vectors."""
        return values @ self._receive_dft.transpose(0, 1)

    def _idft(self, values: torch.Tensor) -> torch.Tensor:
        """Apply the inverse unitary DFT to row-wise spectral vectors."""
        return values @ self._receive_dft.conj()

    def _stack_complex(self, values: torch.Tensor) -> torch.Tensor:
        """Stack real and imaginary parts for the real-valued VAE."""
        return torch.cat([values.real, values.imag], dim=-1).to(torch.float32)

    def _unstack_complex(self, values: torch.Tensor) -> torch.Tensor:
        """Convert real-stacked VAE output to complex DFT-domain vectors."""
        real, imag = torch.chunk(values, 2, dim=-1)
        return torch.complex(real, imag).to(self.cdtype)

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
            self,
            y_pilots,
            no,
        )
        if self._vae_noise_variances.numel() == 0:
            vae = self._vaes[0]
        else:
            vae = self._vaes[self._select_vae_index(no)]

        grouped_h_ls = self._gather_pilot_groups(h_ls)
        grouped_ls_err = self._gather_pilot_groups(ls_err)
        group_values = grouped_h_ls[..., 0].movedim(-2, -1)
        group_error_variances = grouped_ls_err[..., 0].movedim(-2, -1)
        estimates, variances = self._estimate_group(
            group_values,
            group_error_variances,
            vae,
        )
        grouped_h_out = estimates.movedim(-1, -2).unsqueeze(-1)
        grouped_err_out = variances.movedim(-1, -2).unsqueeze(-1)

        h_out = self._scatter_pilot_groups(torch.zeros_like(h_ls), grouped_h_out)
        err_out = self._scatter_pilot_groups(torch.zeros_like(ls_err), grouped_err_out)
        return h_out, err_out

    def _estimate_group(
        self,
        values: torch.Tensor,
        error_variance: torch.Tensor,
        vae: VariationalAutoEncoder,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""Estimate one receive-array vector with Equation (10).

        :param values: Complex LS estimates with shape
            ``[..., num_rx_ant]``.
        :param error_variance: LS error variances with shape
            ``[..., num_rx_ant]``.
        :param vae: SNR-selected VAE model.

        :output estimates: Complex estimates with the same shape as
            ``values``.
        :output variances: Antenna-domain marginal variances with the same
            shape as ``values``.
        """
        leading_shape = values.shape[:-1]
        spectral_values = self._dft(values.to(self.cdtype))
        flat_spectral = spectral_values.reshape(-1, self._num_rx_ant)
        encoder_input = self._stack_complex(flat_spectral)

        _, _, decoder_mean_real, decoder_covariance = vae(
            encoder_input,
            sample=False,
        )
        decoder_covariance = decoder_covariance.to(
            dtype=self.dtype,
            device=values.device,
        )

        noise_variance = error_variance.to(
            dtype=self.dtype,
            device=values.device,
        ).reshape(-1, self._num_rx_ant).mean(dim=-1, keepdim=True)

        if self._covariance_mode == COVARIANCE_FULL:
            estimate, antenna_variance = self._estimate_group_full_covariance(
                flat_spectral,
                encoder_input.to(dtype=self.dtype, device=values.device),
                decoder_mean_real.to(dtype=self.dtype, device=values.device),
                decoder_covariance,
                noise_variance,
            )
            return (
                estimate.reshape(*leading_shape, -1).to(values.dtype),
                (self._err_var_scale * antenna_variance.reshape(
                    *leading_shape,
                    -1,
                )).to(error_variance.dtype),
            )

        decoder_mean = self._unstack_complex(decoder_mean_real)
        gain = decoder_covariance / (decoder_covariance + noise_variance)
        spectral_estimate = decoder_mean + gain * (flat_spectral - decoder_mean)
        estimate = self._idft(spectral_estimate).reshape(*leading_shape, -1)

        posterior_variance = (
            decoder_covariance
            * noise_variance
            / (decoder_covariance + noise_variance)
        )
        antenna_variance = posterior_variance.mean(dim=-1, keepdim=True).expand(
            -1,
            self._num_rx_ant,
        )
        antenna_variance = antenna_variance.reshape(*leading_shape, -1)
        return (
            estimate.to(values.dtype),
            (self._err_var_scale * antenna_variance).to(error_variance.dtype),
        )

    def _estimate_group_full_covariance(
        self,
        flat_spectral: torch.Tensor,
        observation_real: torch.Tensor,
        decoder_mean_real: torch.Tensor,
        decoder_covariance: torch.Tensor,
        noise_variance: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply the real-stacked full-covariance LMMSE update."""
        real_dim = self._input_dim
        eye = torch.eye(real_dim, dtype=self.dtype, device=flat_spectral.device)
        real_noise_variance = 0.5 * noise_variance
        covariance_plus_noise = (
            decoder_covariance
            + real_noise_variance.reshape(-1, 1, 1) * eye.unsqueeze(0)
        )
        cholesky = torch.linalg.cholesky(covariance_plus_noise)
        residual = observation_real - decoder_mean_real
        solved_residual = torch.cholesky_solve(
            residual.unsqueeze(-1),
            cholesky,
        ).squeeze(-1)
        posterior_mean_real = decoder_mean_real + torch.matmul(
            decoder_covariance,
            solved_residual.unsqueeze(-1),
        ).squeeze(-1)

        solved_covariance = torch.cholesky_solve(decoder_covariance, cholesky)
        posterior_covariance = decoder_covariance - (
            decoder_covariance @ solved_covariance
        )
        posterior_covariance = 0.5 * (
            posterior_covariance + posterior_covariance.transpose(-2, -1)
        )

        posterior_complex = self._unstack_complex(posterior_mean_real)
        estimate = self._idft(posterior_complex)

        transform = self._real_idft_transform.to(
            dtype=self.dtype,
            device=flat_spectral.device,
        )
        antenna_covariance = (
            transform.transpose(0, 1).unsqueeze(0)
            @ posterior_covariance
            @ transform.unsqueeze(0)
        )
        diagonal = torch.diagonal(antenna_covariance, dim1=-2, dim2=-1)
        antenna_variance = (
            diagonal[:, : self._num_rx_ant] + diagonal[:, self._num_rx_ant :]
        ).clamp_min(0.0)
        return estimate, antenna_variance
