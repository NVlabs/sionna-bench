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
    r"""Denoise receive-array pilot observations using the original iterative DM.

    Implements Fesl et al., "Diffusion-Based Generative Prior for Low-Complexity
    MIMO Channel Estimation," IEEE WCL, 2024 (arXiv:2403.03545, Algorithm 1).
    A physical rectangular receive array replaces the paper's receive/transmit
    matrix. Each active pilot RE and user is processed separately; Sionna
    interpolates the denoised pilots. No model loading or training occurs here.

    The matched transition is included, following Algorithm 1. The released
    `wcl-2024` code omits that transition by using `reversed(range(k))`.
    Error variances are an empirical scaled-LS fallback, not DM posteriors.

    :param resource_grid: Sionna grid with an active orthogonal pilot pattern.
    :param model_state: Prepared CNN state; named real float tensors, strictly
        matching the fixed 55,025-parameter architecture.
    :param channel_power: Positive real scalar training-population complex-entry
        power in original channel units; never estimated from received pilots.
    :param receive_array_shape: Physical (rows, columns), single polarization,
        with Sionna antenna index ``row + rows*column``.
    :param interpolation_type: Explicit Sionna policy, default ``lin_time_avg``.
    :param interpolator: Optional custom interpolator, taking precedence.
    :param err_var_scale: Empirical multiplier on original-unit LS variance.
        Default 0.623 was selected by independent detector-BLER validation
        for the frozen epoch-170 UMi model; it is not a posterior covariance.
    :param precision: Sionna computation precision, including the CNN.
    :param device: Sionna computation device.
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
        err_var_scale: float = 0.623,
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
