# SPDX-FileCopyrightText: Copyright (c) 2026 Benedikt Fesl
# SPDX-License-Identifier: Apache-2.0
"""Fesl et al. (2024) angular-domain deterministic diffusion estimator."""

from __future__ import annotations

import math
from collections.abc import Mapping
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

__all__ = ["DiffusionChannelEstimator"]


class _NoisePredictionCNN(torch.nn.Module):
    """Released 55,025-parameter CNN, independent of grids and asset storage."""

    def __init__(self) -> None:
        super().__init__()
        self.pre = torch.nn.Sequential(
            torch.nn.Conv2d(2, 32, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, 3, padding=1),
        )
        self.time_affine = torch.nn.Linear(16, 128)
        self.post = torch.nn.Sequential(
            torch.nn.Conv2d(64, 43, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(43, 22, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(22, 2, 3, padding=1),
        )
        self.register_buffer(
            "_time_frequencies",
            torch.exp(-math.log(10000.0) * torch.arange(8) / 7.0),
            persistent=False,
        )

    def forward(self, values: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """Predict variance-1/2 real noise from [N,2,rows,cols] and [N]."""
        phase = timesteps[:, None] * self._time_frequencies[None, :]
        embedding = torch.cat((phase.sin(), phase.cos()), dim=-1)
        scale, shift = self.time_affine(embedding).chunk(2, dim=-1)
        features = self.pre(values)
        # Author conditioning is residual affine modulation, not scale*x alone.
        features = features + scale[:, :, None, None] * features
        return self.post(features + shift[:, :, None, None])


class _DiffusionSchedule(torch.nn.Module):
    """The sole WCL VP schedule; coefficient index t represents paper step t+1."""

    def __init__(
        self, *, dtype: torch.dtype = torch.float32, device: Optional[str] = None
    ) -> None:
        super().__init__()
        # Derive all coefficients before casting, as in the released code.
        betas = torch.linspace(1.0 - 10000.0 / 10001.0, 0.1, 100, dtype=torch.float64)
        alphas = 1.0 - betas
        cumulative = alphas.cumprod(0)
        coefficients = {
            "betas": betas,
            "alphas": alphas,
            "alpha_bar": cumulative,
            "signal_sqrt": cumulative.sqrt(),
            "noise_sqrt": (1.0 - cumulative).sqrt(),
            "snrs": cumulative / (1.0 - cumulative),
            "inverse_alpha_sqrt": alphas.rsqrt(),
            "noise_coefficient": betas / (alphas * (1.0 - cumulative)).sqrt(),
        }
        for name, value in coefficients.items():
            self.register_buffer(name, value.to(dtype=dtype, device=device))

    def match_timesteps(self, noise_variance: torch.Tensor) -> torch.Tensor:
        """Match linear SNR, using -1 only for the exact noise-free endpoint."""
        snr = noise_variance.clamp_min(
            torch.finfo(noise_variance.dtype).tiny
        ).reciprocal()
        matched = (self.snrs[None, :] - snr[:, None]).abs().argmin(-1)
        return torch.where(noise_variance == 0, -torch.ones_like(matched), matched)

    def forward_noise(
        self, clean: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor
    ) -> torch.Tensor:
        """Apply the paper's VP marginal, Eq. (3) of the authoritative preprint."""
        shape = (-1,) + (1,) * (clean.ndim - 1)
        return (
            self.signal_sqrt[timesteps].reshape(shape) * clean
            + self.noise_sqrt[timesteps].reshape(shape) * noise
        )

    def reverse_mean(
        self, values: torch.Tensor, predicted_noise: torch.Tensor, timestep: int
    ) -> torch.Tensor:
        """Evaluate the DDPM posterior mean without stochastic resampling."""
        return (
            self.inverse_alpha_sqrt[timestep] * values
            - self.noise_coefficient[timestep] * predicted_noise
        )


class DiffusionChannelEstimator(BasePilotChannelEstimator, _PilotGroupMixin):
    r"""Channel estimator that leverages a diffusion model (DM) as a learned channel prior.

    This implementation follows the method proposed in:

    .. container:: bibliography-box

        .. bibliography::
            :filter: key == "fesl2024diffusion"
            :list: bullet

    .. rubric:: TL;DR

    This estimator uses a denoising diffusion model as a generative prior for the
    angular-domain receive-array channel. A single noise-conditional CNN is
    trained on clean channels and serves all diffusion timesteps and observation
    SNRs. At inference, the least-squares (LS) pilot estimate is normalized by its
    actual noise variance, matched to the step of the diffusion schedule with the
    closest SNR, and refined by deterministic reverse conditional-mean steps
    (Algorithm 1 of the paper). The result approximates the MMSE estimate under
    the learned prior. A Sionna interpolator reconstructs the remaining resource
    grid.

    .. rubric:: Supported System Geometry

    The estimator processes every active pilot resource element of every user
    independently, while jointly processing the corresponding receive-array
    vector. It therefore supports different resource-grid sizes, pilot layouts,
    UE counts, and OFDM-symbol counts, provided that the following hold:

    - The resource grid contains at least one active nonzero pilot.
    - The receive array is a single-polarized rectangular array whose
      ``receive_array_shape`` matches ``num_rx_ant`` in Sionna's antenna order.
    - The noise variance is identical across the receive antennas of each pilot
      resource element.
    - The prepared model state and ``channel_power`` were trained for the same
      receive-array shape and channel distribution.

    Changing the receive-array shape or the channel model requires a separately
    trained compatible model state.

    .. rubric:: Deviations from the Paper

    - The paper denoises a 64 x 16 receive/transmit MIMO channel matrix whose two
      axes are transformed by a receive and a transmit DFT. This implementation
      denoises the physical receive array at one pilot resource element; the two
      axes are the rows and columns of the rectangular array, transformed by a
      unitary 2D DFT. Users and pilot subcarriers are not stacked into a
      transmit-array axis.
    - The paper assumes unitary pilot decorrelation that yields the observation
      ``y = h + n``. This implementation uses the LS estimate at sparse OFDM
      pilots together with its actual noise variance ``no / |pilot|**2``.
    - The paper's unit-average-power convention is implemented by normalizing
      with the frozen training-population channel power ``channel_power``. No
      per-observation or test-channel normalization is applied.
    - Algorithm 1 of the paper applies the reverse step at the matched timestep
      and at all smaller timesteps. The authors' released code skips the matched
      timestep. This implementation follows the paper.
    - The paper does not address interpolation. This implementation has to be
      combined with a Sionna interpolator and defaults to ``lin_time_avg``.
    - The paper does not specify detector-facing error variances. This
      implementation returns the LS pilot error variance scaled by
      ``err_var_scale``, which is an empirical calibration and not a posterior
      variance of the diffusion model.

    :param resource_grid: Sionna resource grid defining the pilot pattern.
    :param model_state: Prepared state dictionary of the noise-prediction CNN.
        It must match the fixed architecture with 55,025 parameters.
    :param channel_power: Mean squared magnitude of the complex channel entries
        of the training population, a positive real scalar. It is never
        estimated from received pilots.
    :param receive_array_shape: Physical ``(rows, columns)`` of the
        single-polarized receive array. Sionna's antenna index is
        ``row + rows * column``.
    :param interpolation_type: Sionna interpolation policy. This argument is
        ignored when ``interpolator`` is provided.
    :param interpolator: Custom Sionna channel interpolator. If `None`, the
        interpolator selected by ``interpolation_type`` is used.
    :param err_var_scale: Multiplicative scale applied to the LS pilot error
        variance that forms the detector-facing error variance.
    :param precision: Precision used for Sionna channel estimation. If `None`,
        :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device for Sionna channel estimation. If `None`,
        :attr:`~sionna.phy.config.Config.device` is used.
    """

    def __init__(
        self,
        resource_grid: ResourceGrid,
        model_state: Mapping[str, torch.Tensor],
        channel_power: torch.Tensor,
        receive_array_shape: tuple[int, int],
        interpolation_type: str = "lin_time_avg",
        interpolator: Optional[BaseChannelInterpolator] = None,
        *,
        err_var_scale: float = 0.0358,
        precision: Optional[Precision] = None,
        device: Optional[str] = None,
    ) -> None:
        super().__init__(
            resource_grid=resource_grid,
            interpolation_type=interpolation_type,
            interpolator=interpolator,
            precision=precision,
            device=device,
        )
        if len(receive_array_shape) != 2 or any(
            int(n) != n or n < 1 for n in receive_array_shape
        ):
            raise ValueError("receive_array_shape must contain two positive integers.")
        if not math.isfinite(err_var_scale) or err_var_scale < 0:
            raise ValueError("err_var_scale must be finite and nonnegative.")
        power = torch.as_tensor(channel_power, dtype=self.dtype, device=self.device)
        if (
            power.numel() != 1
            or not bool(torch.isfinite(power).all())
            or not bool((power > 0).all())
        ):
            raise ValueError("channel_power must be a finite positive scalar.")
        self._receive_array_shape = tuple(int(n) for n in receive_array_shape)
        self._num_rx_ant = math.prod(self._receive_array_shape)
        self._err_var_scale = float(err_var_scale)
        self.register_buffer("_channel_power", power.reshape(()))
        self._init_pilot_groups(self._pilot_pattern, group_over=())
        self._cnn = _NoisePredictionCNN()
        self._cnn.load_state_dict(model_state, strict=True)
        self._cnn.to(device=self.device, dtype=self.dtype).eval()
        self._cnn.requires_grad_(False)
        self._schedule = _DiffusionSchedule(dtype=self.dtype, device=self.device)

    def _to_angular(self, values: torch.Tensor) -> torch.Tensor:
        """Map [...,antennas] into [...,rows,cols] in physical Sionna order."""
        rows, cols = self._receive_array_shape
        array = values.reshape(*values.shape[:-1], cols, rows).transpose(-1, -2)
        return torch.fft.fft2(array, norm="ortho")

    def _from_angular(self, values: torch.Tensor) -> torch.Tensor:
        array = torch.fft.ifft2(values, norm="ortho")
        return array.transpose(-1, -2).reshape(*values.shape[:-2], self._num_rx_ant)

    def _stack_complex(self, values: torch.Tensor) -> torch.Tensor:
        return torch.stack((values.real, values.imag), dim=-3)

    def _unstack_complex(self, values: torch.Tensor) -> torch.Tensor:
        return torch.complex(values[..., 0, :, :], values[..., 1, :, :])

    def _prepare_observations(
        self, values: torch.Tensor, noise_variance: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        normalized_noise = noise_variance / self._channel_power
        angular = self._to_angular(values / self._channel_power.sqrt())
        # Algorithm 1 line 4: use actual noise, not the quantized schedule SNR.
        initial = (
            self._stack_complex(angular)
            * (1.0 + normalized_noise).rsqrt()[:, None, None, None]
        )
        return initial, self._schedule.match_timesteps(normalized_noise)

    @torch.compiler.disable
    def _noise_buckets(
        self, variance: torch.Tensor, timesteps: torch.Tensor
    ) -> list[tuple[int, torch.Tensor]]:
        """Small routing/validation boundary; numerical inference stays tensor-only."""
        if not bool(torch.isfinite(variance).all()) or bool((variance < 0).any()):
            raise ValueError("Pilot noise variances must be finite and nonnegative.")
        if not torch.allclose(
            variance,
            variance[:, :1].expand_as(variance),
            rtol=8 * torch.finfo(variance.dtype).eps,
            atol=0.0,
        ):
            raise ValueError(
                "Diffusion requires isotropic noise across receive antennas."
            )
        return [
            (int(t), torch.where(timesteps == t)[0])
            for t in timesteps.unique().tolist()
        ]

    def _predict_noise(self, values: torch.Tensor, timestep: int) -> torch.Tensor:
        times = torch.full(
            (values.shape[0],), timestep, dtype=torch.int64, device=values.device
        )
        return self._cnn(values, times)

    def _reverse_diffusion(
        self, values: torch.Tensor, start_index: int
    ) -> torch.Tensor:
        # Index k is paper state ell=k+1. Algorithm 1 includes that first update.
        for timestep in range(start_index, -1, -1):
            values = self._schedule.reverse_mean(
                values, self._predict_noise(values, timestep), timestep
            )
        return values

    def estimate_at_pilot_locations(
        self, y_pilots: torch.Tensor, no: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return pilot estimates/variances in [B,rx,ant,tx,stream,pilot] layout."""
        if y_pilots.shape[2] != self._num_rx_ant:
            raise ValueError(
                "Received antenna count does not match receive_array_shape."
            )
        least_squares, variance = LSChannelEstimator.estimate_at_pilot_locations(
            self, y_pilots, no
        )
        variance = variance.expand_as(least_squares.real)
        grouped = self._gather_pilot_groups(least_squares)[..., 0].movedim(-2, -1)
        grouped_variance = self._gather_pilot_groups(variance)[..., 0].movedim(-2, -1)
        values = grouped.reshape(-1, self._num_rx_ant)
        noise = grouped_variance.reshape_as(values.real)
        initial, starts = self._prepare_observations(values, noise[:, 0])
        output = torch.zeros_like(initial)
        for start, indices in self._noise_buckets(noise, starts):
            result = self._reverse_diffusion(initial.index_select(0, indices), start)
            output = output.index_copy(0, indices, result)
        estimates = (
            self._from_angular(self._unstack_complex(output))
            * self._channel_power.sqrt()
        )
        estimates = estimates.reshape_as(grouped).movedim(-1, -2).unsqueeze(-1)
        return (
            self._scatter_pilot_groups(torch.zeros_like(least_squares), estimates),
            self._scatter_pilot_groups(
                torch.zeros_like(variance),
                self._err_var_scale * grouped_variance.movedim(-1, -2).unsqueeze(-1),
            ),
        )
