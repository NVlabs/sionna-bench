# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Comb-pilot LS estimators from Coleri et al. 2002."""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from scipy.interpolate import CubicSpline
from scipy.signal import firls
from sionna.phy import Precision
from sionna.phy.ofdm import (
    BaseChannelInterpolator,
    LSChannelEstimator,
    PilotPattern,
    ResourceGrid,
)
from sionna.phy.ofdm.channel_estimation import _PilotGroupMixin


METHOD_TIME_DOMAIN = "time-domain"
METHOD_SECOND_ORDER = "second-order"
METHOD_LOWPASS = "low-pass"
METHOD_SPLINE = "cubic-spline"
METHOD_NAMES = [
    METHOD_SECOND_ORDER,
    METHOD_LOWPASS,
    METHOD_SPLINE,
    METHOD_TIME_DOMAIN,
]

class CombInterpolator(BaseChannelInterpolator, _PilotGroupMixin):
    r"""Channel interpolator that leverages certain smoothness-assumptions to interpolate channel estimates.

    This interpolator implements the methods proposed in:

    .. container:: bibliography-box

        .. bibliography::
            :filter: key == "coleri2002channel"
            :list: bullet

    .. rubric:: TL;DR

    The input pilot estimates are grouped by transmit stream and pilot-bearing
    OFDM symbol. For each group, the channel is reconstructed across frequency
    using one of four methods:

    - ``"second-order"`` applies a local second-order interpolation rule.
    - ``"low-pass"`` inserts zeros between pilot estimates and applies a symmetric
      low-pass FIR interpolation filter.
    - ``"cubic-spline"`` applies cubic-spline interpolation using.
    - ``"time-domain"`` transforms the pilot sequence to the delay domain, inserts
      zeros in the middle of the FFT-ordered delay vector, and transforms it back
      to the complete effective-subcarrier grid.

    After frequency reconstruction, all methods apply linear interpolation between
    pilot-bearing OFDM symbols.

    .. rubric:: Supported System Geometry

    The interpolator supports different effective FFT sizes, OFDM-symbol counts,
    transmit-stream counts, and receive-antenna counts, provided that:

    - Each pilot group forms a uniform comb over subcarriers.
    - All pilot groups have the same comb spacing and number of pilots.
    - Each comb spans the complete effective-subcarrier grid, such that
      ``num_effective_subcarriers == pilot_spacing * num_pilots_per_symbol``.
    - Every transmit stream has the same number of pilot-bearing OFDM symbols.

    Different comb offsets between transmit streams are supported. Nonuniform,
    incomplete, or inconsistent comb layouts are rejected during construction.

    .. rubric:: Deviations from the Paper

    - The paper formulates its interpolation methods independently for each OFDM
      symbol. This implementation additionally applies linear interpolation over
      OFDM time when pilots occupy only a subset of the OFDM symbols.
    - The paper does not fully specify frequency-boundary behavior. Second-order
      and low-pass interpolation therefore use constant edge extension, while
      cubic-spline queries outside the observed pilot span are clipped to the
      nearest edge.
    - The low-pass implementation follows the publicly documented MATLAB
      ``interp``/``intfilt`` construction but is not claimed to reproduce its
      internal implementation bit for bit.
    - Equation (19) is evaluated using the normalized interval coordinate
      :math:`l/L`, consistently with Equation (18), instead of the printed
      :math:`l/N`.
    - The printed zero-padding expression in Equation (21) is inconsistent. The
      time-domain method follows the surrounding description and Figure 3:
      pilot-rate IDFT, middle zero padding, and a DFT over all effective
      subcarriers.
    - The paper does not define detector-facing channel-estimation error
      variances. This implementation expands the mean pilot-domain LS variance
      over frequency, interpolates it over OFDM time, and applies
      ``err_var_scale``.

    :param pilot_pattern: Pilot pattern defining the active comb-pilot positions.
    :param method: Frequency-interpolation method. Must be ``"second-order"``,
        ``"low-pass"``, ``"cubic-spline"``, or ``"time-domain"``.
    :param lowpass_filter_half_length: Half-length parameter of the
        MATLAB-style low-pass interpolation filter. The resulting FIR length is
        ``2 * lowpass_filter_half_length * pilot_spacing + 1``. This parameter is
        used only by ``method="low-pass"``.
    :param lowpass_cutoff: Normalized cutoff of the low-pass interpolation filter,
        expressed as a fraction of the input Nyquist frequency. It must lie in
        `(0, 1]` and is used only by ``method="low-pass"``.
    :param err_var_scale: Non-negative scale applied to the reconstructed
        detector-facing error variances.
    :param precision: Precision used for internal calculations and outputs. If
        `None`, :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device used for tensor operations. If `None`,
        :attr:`~sionna.phy.config.Config.device` is used.

    :input h_hat: [batch_size, num_rx, num_rx_ant, num_tx,
        num_streams_per_tx, num_pilot_symbols], `torch.complex`.
        Channel estimates at the active pilot positions.
    :input err_var: [batch_size, num_rx, num_rx_ant, num_tx,
        num_streams_per_tx, num_pilot_symbols], `torch.float`.
        Corresponding pilot-domain channel-estimation error variances.

    :output h_hat: [batch_size, num_rx, num_rx_ant, num_tx,
        num_streams_per_tx, num_ofdm_symbols, num_effective_subcarriers],
        `torch.complex`.
        Channel estimates over the complete effective resource grid.
    :output err_var: Same shape as ``h_hat``, `torch.float`.
        Corresponding detector-facing channel-estimation error variances.
    """
    def __init__(
        self,
        pilot_pattern: PilotPattern,
        method: str = METHOD_SECOND_ORDER,
        *,
        lowpass_filter_half_length: int = 16,
        lowpass_cutoff: float = 0.105,
        err_var_scale: float = 0.868,
        precision: Optional[Precision] = None,
        device: Optional[str] = None,
    ) -> None:
        """Initialize the interpolator and cache comb-pilot layout data."""
        super().__init__(precision=precision, device=device)
        assert method in METHOD_NAMES, "Unsupported Coleri interpolation method."
        assert (
            isinstance(lowpass_filter_half_length, int)
            and lowpass_filter_half_length > 0
        ), "`lowpass_filter_half_length` must be a positive integer."
        assert isinstance(lowpass_cutoff, (int, float)) and math.isfinite(
            lowpass_cutoff
        ), "`lowpass_cutoff` must be finite."
        assert 0.0 < float(lowpass_cutoff) <= 1.0, (
            "`lowpass_cutoff` must be in (0, 1]."
        )
        assert isinstance(err_var_scale, (int, float)) and math.isfinite(
            err_var_scale
        ), "`err_var_scale` must be finite."
        assert err_var_scale >= 0.0, "`err_var_scale` must be non-negative."

        self._method = method
        self._lowpass_filter_half_length = lowpass_filter_half_length
        self._lowpass_cutoff = float(lowpass_cutoff)
        self._err_var_scale = float(err_var_scale)

        pilot_mask = pilot_pattern.mask
        self._num_tx = int(pilot_mask.shape[0])
        self._num_streams_per_tx = int(pilot_mask.shape[1])
        self._num_ofdm_symbols = int(pilot_mask.shape[2])
        self._num_subcarriers = int(pilot_mask.shape[3])

        self._init_pilot_groups(
            pilot_pattern=pilot_pattern,
            group_over=("subcarrier",),
        )
        self._init_comb_layout()

        if self._method == METHOD_LOWPASS:
            self.register_buffer(
                "_lowpass_coefficients",
                self._design_lowpass_filter(),
            )
        if self._method == METHOD_SPLINE:
            self._spline_comb_offsets, spline_weight_matrices = (
                self._build_spline_weight_matrices()
            )
            self.register_buffer(
                "_spline_weight_matrices",
                spline_weight_matrices,
            )

    def _init_comb_layout(self) -> None:
        """Validate the comb-pilot layout and cache stream group indices."""
        group_symbol_indices = self._pilot_group_constant_coord("symbol")
        group_tx_indices = self._pilot_group_constant_coord("tx")
        group_stream_indices = self._pilot_group_constant_coord("stream")
        group_subcarrier_indices = self._pilot_group_varying_coord("subcarrier")

        if self._pilot_group_len == 1:
            group_spacings = torch.full(
                (self._pilot_group_num_groups,),
                self._num_subcarriers,
                dtype=torch.int64,
                device=self.device,
            )
        else:
            subcarrier_spacings = torch.diff(group_subcarrier_indices, dim=1)
            group_spacings = subcarrier_spacings[:, 0]
            same_spacing = subcarrier_spacings == group_spacings.unsqueeze(1)
            assert bool(torch.all(same_spacing)), (
                "Coleri interpolation requires uniform comb pilots."
            )

        unique_spacings = torch.unique(group_spacings)
        assert int(unique_spacings.numel()) == 1, (
            "Coleri interpolation requires a common comb spacing."
        )

        self._pilot_spacing = int(unique_spacings[0].item())
        self._num_pilots_per_symbol = self._pilot_group_len
        assert self._num_subcarriers == (
            self._pilot_spacing * self._num_pilots_per_symbol
        ), "Coleri interpolators require comb pilots to span the full grid."

        group_comb_offsets = group_subcarrier_indices[:, 0]

        num_streams = self._num_tx * self._num_streams_per_tx
        flat_stream_indices = (
            group_tx_indices * self._num_streams_per_tx + group_stream_indices
        )
        group_counts_per_stream = torch.bincount(
            flat_stream_indices,
            minlength=num_streams,
        )
        assert bool(torch.all(group_counts_per_stream == group_counts_per_stream[0])), (
            "Coleri interpolation requires equal pilot-symbol counts per stream."
        )
        num_pilot_symbols_per_stream = int(group_counts_per_stream[0].item())
        stream_group_indices = torch.arange(
            self._pilot_group_num_groups,
            dtype=torch.int64,
            device=self.device,
        ).reshape(
            self._num_tx,
            self._num_streams_per_tx,
            num_pilot_symbols_per_stream,
        )

        self._group_symbol_indices = tuple(
            int(symbol_index) for symbol_index in group_symbol_indices.tolist()
        )
        self._group_comb_offsets = tuple(
            int(comb_offset) for comb_offset in group_comb_offsets.tolist()
        )
        self._stream_group_indices = tuple(
            tuple(
                tuple(int(group_index) for group_index in stream_indices.tolist())
                for stream_indices in tx_indices
            )
            for tx_indices in stream_group_indices
        )

    def _design_lowpass_filter(self) -> torch.Tensor:
        """Return MATLAB ``interp``-style zero-insertion low-pass FIR taps."""
        if self._pilot_spacing == 1:
            return torch.ones(1, dtype=self.dtype, device=self.device)

        num_filter_taps = (
            2 * self._lowpass_filter_half_length * self._pilot_spacing + 1
        )

        # MATLAB `interp` treats `cutoff` relative to the input Nyquist rate.
        # After zero insertion by `spacing`, the passband contracts by the same
        # factor in the expanded-rate frequency axis.
        passband_edge = self._lowpass_cutoff / self._pilot_spacing
        stopband_edge = min(1.0, (2.0 - self._lowpass_cutoff) / self._pilot_spacing)
        filter_coefficients = firls(
            num_filter_taps,
            [0.0, passband_edge, stopband_edge, 1.0],
            [self._pilot_spacing, self._pilot_spacing, 0.0, 0.0],
            fs=2.0,
        )

        # Enforce the Nyquist interpolation property explicitly: at original
        # sample instants the FIR must pass the LS pilot through unchanged, and
        # neighboring original-sample phases must not leak into it.
        center_tap = num_filter_taps // 2
        for tap_index in range(num_filter_taps):
            if tap_index == center_tap:
                filter_coefficients[tap_index] = 1.0
            elif (tap_index - center_tap) % self._pilot_spacing == 0:
                filter_coefficients[tap_index] = 0.0

        return torch.as_tensor(
            filter_coefficients,
            dtype=self.dtype,
            device=self.device,
        )

    def _build_spline_weight_matrices(
        self,
    ) -> tuple[tuple[int, ...], torch.Tensor]:
        """Precompute MATLAB-like cubic spline interpolation matrices."""
        spline_weight_matrices = []
        spline_comb_offsets = tuple(sorted(set(self._group_comb_offsets)))
        for comb_offset in spline_comb_offsets:
            pilot_positions = (
                float(comb_offset)
                + np.arange(self._num_pilots_per_symbol, dtype=np.float64)
                * float(self._pilot_spacing)
            )
            basis = np.eye(self._num_pilots_per_symbol, dtype=np.float64)

            # MATLAB `spline` uses a not-a-knot cubic spline by default. The
            # paper does not state a periodic frequency boundary, so do not add
            # a wrapped endpoint; instead, clip queries outside the observed
            # pilot span below.
            spline = CubicSpline(
                pilot_positions,
                basis,
                axis=0,
                bc_type="not-a-knot",
            )
            subcarrier_positions = np.arange(self._num_subcarriers, dtype=np.float64)
            # Interpolation is along axis 0 of `basis`, the pilot-coordinate
            # axis. The resulting matrix has shape [num_subcarriers,
            # num_pilots_per_symbol].
            weights = spline(
                np.clip(subcarrier_positions, pilot_positions[0], pilot_positions[-1])
            )
            spline_weight_matrices.append(
                torch.as_tensor(weights, dtype=self.dtype, device=self.device)
            )
        return spline_comb_offsets, torch.stack(spline_weight_matrices, dim=0)

    @staticmethod
    def _zero_pad_middle(signal: torch.Tensor, target_size: int) -> torch.Tensor:
        """Insert zeros between nonnegative- and negative-delay bins."""
        source_size = int(signal.shape[-1])
        if source_size == target_size:
            return signal
        # The IDFT output uses FFT ordering. For even lengths, Eq. (21) defines
        # M=N_p/2+1; the sample at N_p/2 therefore remains in the leading
        # segment. The equation's printed case ranges are inconsistent, but
        # this split implements its stated middle-zero-padding construction.
        if source_size % 2 == 0:
            split_index = source_size // 2 + 1
        else:
            split_index = (source_size + 1) // 2
        padding_shape = (*signal.shape[:-1], target_size - source_size)
        zero_padding = torch.zeros(
            padding_shape,
            dtype=signal.dtype,
            device=signal.device,
        )
        return torch.cat(
            [
                signal[..., :split_index],
                zero_padding,
                signal[..., split_index:],
            ],
            dim=-1,
        )

    def _interpolate_with_time_domain_transform(
        self,
        pilot_values: torch.Tensor,
        comb_offset: int,
    ) -> torch.Tensor:
        """Interpolate pilots by zero-padding their time-domain transform."""
        # Eq. (20): pilot-rate IDFT; Eq. (21): middle zero padding;
        # Eq. (22): DFT back to all subcarriers. PyTorch's FFT scaling is used
        # so that reconstructed values pass through the pilot samples.
        time_domain_pilot_signal = torch.fft.ifft(pilot_values, dim=-1)
        zero_padded_time_signal = self._zero_pad_middle(
            time_domain_pilot_signal,
            self._num_subcarriers,
        )
        interpolated_values = torch.fft.fft(zero_padded_time_signal, dim=-1)
        if comb_offset:
            interpolated_values = torch.roll(
                interpolated_values,
                shifts=comb_offset,
                dims=-1,
            )
        return interpolated_values

    def _interpolate_with_second_order_polynomial(
        self,
        pilot_values: torch.Tensor,
        comb_offset: int,
    ) -> torch.Tensor:
        """Interpolate pilots across frequency with a quadratic formula."""
        num_pilot_values = int(pilot_values.shape[-1])
        subcarrier_indices = torch.arange(
            self._num_subcarriers,
            device=pilot_values.device,
        )
        relative_subcarrier_indices = subcarrier_indices - int(comb_offset)
        raw_pilot_interval_indices = torch.div(
            relative_subcarrier_indices,
            self._pilot_spacing,
            rounding_mode="floor",
        )
        pilot_interval_indices = raw_pilot_interval_indices.clamp(
            min=0,
            max=num_pilot_values - 1,
        )

        # Eq. (19) is evaluated with the interpolation fraction over one pilot
        # interval. Subcarriers before the first stream-specific pilot are held
        # at the first pilot rather than wrapped to the last pilot.
        interpolation_fraction = (
            relative_subcarrier_indices
            - raw_pilot_interval_indices * self._pilot_spacing
        ).to(pilot_values.real.dtype)
        interpolation_fraction = interpolation_fraction / float(self._pilot_spacing)
        interpolation_fraction = torch.where(
            relative_subcarrier_indices < 0,
            torch.zeros_like(interpolation_fraction),
            interpolation_fraction,
        )

        # Constant edge extension implements max(index-1, 0) and
        # min(index+1, num_pilots-1) for the neighbor pilots required by the
        # quadratic formula.
        previous_pilot_values = pilot_values.index_select(
            -1,
            (pilot_interval_indices - 1).clamp(min=0),
        )
        current_pilot_values = pilot_values.index_select(-1, pilot_interval_indices)
        next_pilot_values = pilot_values.index_select(
            -1,
            (pilot_interval_indices + 1).clamp(max=num_pilot_values - 1),
        )

        previous_weight = interpolation_fraction * (interpolation_fraction - 1.0) / 2.0
        current_weight = -(
            (interpolation_fraction - 1.0) * (interpolation_fraction + 1.0)
        )
        next_weight = interpolation_fraction * (interpolation_fraction + 1.0) / 2.0
        weight_shape = [1] * pilot_values.ndim
        weight_shape[-1] = self._num_subcarriers
        return (
            previous_pilot_values * previous_weight.view(weight_shape)
            + current_pilot_values * current_weight.view(weight_shape)
            + next_pilot_values * next_weight.view(weight_shape)
        )

    def _interpolate_with_lowpass_filter(
        self,
        pilot_values: torch.Tensor,
        comb_offset: int,
    ) -> torch.Tensor:
        """Interpolate pilots by zero insertion and low-pass FIR filtering."""
        lowpass_coefficients = self._lowpass_coefficients.to(
            device=pilot_values.device,
            dtype=pilot_values.real.dtype,
        )
        filter_padding = int(lowpass_coefficients.numel() // 2)

        # Work on an extended grid so convolution is finite at the resource-grid
        # edges. The extra samples are only FIR context and are discarded by the
        # valid convolution below.
        extended_samples = torch.zeros(
            *pilot_values.shape[:-1],
            self._num_subcarriers + 2 * filter_padding,
            dtype=pilot_values.dtype,
            device=pilot_values.device,
        )

        num_pilot_values = int(pilot_values.shape[-1])
        pilot_positions = int(comb_offset) + torch.arange(
            num_pilot_values,
            device=pilot_values.device,
            dtype=torch.int64,
        ) * self._pilot_spacing
        extended_pilot_positions = pilot_positions + filter_padding

        # MATLAB-style low-pass interpolation starts from zero insertion: real
        # pilot locations hold LS estimates, and all intermediate subcarriers
        # start at zero before FIR smoothing.
        extended_samples[..., extended_pilot_positions] = pilot_values

        # Fill virtual comb pilots to the left with the first observed pilot.
        # This supplies FIR context without circularly coupling the last pilot
        # to the first pilot.
        left_context_positions = torch.arange(
            int(comb_offset) - self._pilot_spacing,
            -filter_padding - 1,
            -self._pilot_spacing,
            device=pilot_values.device,
            dtype=torch.int64,
        )
        if left_context_positions.numel() > 0:
            extended_samples[..., left_context_positions + filter_padding] = (
                pilot_values[..., :1]
            )

        # Same constant extension on the right edge, using the last observed
        # pilot as virtual FIR context.
        right_context_start = int(comb_offset) + num_pilot_values * self._pilot_spacing
        right_context_positions = torch.arange(
            right_context_start,
            self._num_subcarriers + filter_padding,
            self._pilot_spacing,
            device=pilot_values.device,
            dtype=torch.int64,
        )
        if right_context_positions.numel() > 0:
            extended_samples[..., right_context_positions + filter_padding] = (
                pilot_values[..., -1:]
            )

        # `conv1d` computes correlation, so flip the FIR to implement
        # convolution. Flattening lets the same FIR run independently over all
        # batch, receive, antenna, tx, and stream dimensions.
        flattened_samples = extended_samples.reshape(-1, 1, extended_samples.shape[-1])
        convolution_weight = lowpass_coefficients.flip(0).to(
            dtype=pilot_values.dtype
        ).view(1, 1, -1)
        filtered_samples = F.conv1d(
            flattened_samples,
            convolution_weight,
        ).reshape(*pilot_values.shape[:-1], self._num_subcarriers)

        # Make pilot pass-through exact even if the least-squares FIR has tiny
        # numerical deviations from the intended Nyquist property.
        filtered_samples[..., pilot_positions] = pilot_values
        return filtered_samples

    def _interpolate_with_cubic_spline(
        self,
        pilot_values: torch.Tensor,
        comb_offset: int,
    ) -> torch.Tensor:
        """Interpolate pilots across frequency with cubic-spline weights."""
        spline_matrix_index = self._spline_comb_offsets.index(comb_offset)
        spline_weights = self._spline_weight_matrices[spline_matrix_index].to(
            device=pilot_values.device,
            dtype=pilot_values.real.dtype,
        )
        # The precomputed matrix is real and linear, so applying it directly to
        # complex pilots is equivalent to interpolating real and imaginary parts
        # with the same cubic spline weights.
        return pilot_values @ spline_weights.t().to(dtype=pilot_values.dtype)

    def _interpolate_frequency(
        self,
        pilot_values: torch.Tensor,
        group_index: int,
    ) -> torch.Tensor:
        """Apply the configured frequency interpolation to one pilot group."""
        comb_offset = self._group_comb_offsets[group_index]
        if self._method == METHOD_TIME_DOMAIN:
            return self._interpolate_with_time_domain_transform(pilot_values, comb_offset)
        if self._method == METHOD_SECOND_ORDER:
            return self._interpolate_with_second_order_polynomial(pilot_values, comb_offset)
        if self._method == METHOD_LOWPASS:
            return self._interpolate_with_lowpass_filter(pilot_values, comb_offset)
        if self._method == METHOD_SPLINE:
            return self._interpolate_with_cubic_spline(pilot_values, comb_offset)
        raise ValueError(f"unknown Coleri interpolation method: {self._method}")

    def _interpolate_time(
        self,
        symbol_values: tuple[torch.Tensor, ...],
        pilot_symbol_indices: tuple[int, ...],
    ) -> torch.Tensor:
        """Linearly interpolate reconstructed pilot symbols over OFDM time."""
        # Time interpolation is fixed to linear interpolation between the
        # nearest pilot-bearing OFDM symbols.
        if len(pilot_symbol_indices) == 1:
            return symbol_values[0].unsqueeze(-2).expand(
                *symbol_values[0].shape[:-1],
                self._num_ofdm_symbols,
                self._num_subcarriers,
            )

        stacked_values = torch.stack(symbol_values, dim=-2)
        pilot_symbol_indices = torch.tensor(
            pilot_symbol_indices,
            dtype=torch.int64,
            device=stacked_values.device,
        )
        output_symbol_indices = torch.arange(
            self._num_ofdm_symbols,
            dtype=torch.int64,
            device=stacked_values.device,
        )

        right_symbol_indices = torch.searchsorted(
            pilot_symbol_indices,
            output_symbol_indices,
            right=True,
        )
        left_symbol_indices = torch.where(
            right_symbol_indices == 0,
            torch.zeros_like(right_symbol_indices),
            right_symbol_indices - 1,
        )
        right_symbol_indices = torch.where(
            right_symbol_indices == 0,
            torch.ones_like(right_symbol_indices),
            right_symbol_indices,
        )
        right_symbol_indices_past_end = right_symbol_indices >= len(
            pilot_symbol_indices
        )
        left_symbol_indices = torch.where(
            right_symbol_indices_past_end,
            torch.full_like(left_symbol_indices, len(pilot_symbol_indices) - 2),
            left_symbol_indices,
        )
        right_symbol_indices = torch.where(
            right_symbol_indices_past_end,
            torch.full_like(right_symbol_indices, len(pilot_symbol_indices) - 1),
            right_symbol_indices,
        )

        left_symbols = pilot_symbol_indices.index_select(0, left_symbol_indices)
        right_symbols = pilot_symbol_indices.index_select(0, right_symbol_indices)
        interpolation_weights = (
            (output_symbol_indices - left_symbols).to(stacked_values.real.dtype)
            / (right_symbols - left_symbols).to(stacked_values.real.dtype)
        )
        weight_shape = (
            (1,) * (stacked_values.dim() - 2)
            + (self._num_ofdm_symbols, 1)
        )
        interpolation_weights = interpolation_weights.reshape(weight_shape)

        left_values = stacked_values.index_select(-2, left_symbol_indices)
        right_values = stacked_values.index_select(-2, right_symbol_indices)
        return left_values + interpolation_weights * (
            right_values - left_values
        )

    def _interpolate_error_variance(
        self,
        pilot_error_variances: torch.Tensor,
    ) -> torch.Tensor:
        """Expand pilot error variances across all subcarriers."""
        # Coleri does not derive detector-facing variances. Propagate the
        # Sionna LS pilot variance as a scalar per reconstructed symbol,
        # matching the LS baseline observation model without tuning a
        # confidence scale.
        return self._err_var_scale * pilot_error_variances.mean(
            dim=-1,
            keepdim=True,
        ).expand(*pilot_error_variances.shape[:-1], self._num_subcarriers)

    def __call__(
        self,
        h_hat: torch.Tensor,
        err_var: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Interpolate pilot-domain estimates over the resource grid."""
        output_shape = (
            *h_hat.shape[:-1],
            self._num_ofdm_symbols,
            self._num_subcarriers,
        )
        channel_estimates = torch.empty(
            output_shape,
            dtype=h_hat.dtype,
            device=h_hat.device,
        )
        error_variances = torch.empty(
            output_shape,
            dtype=err_var.dtype,
            device=err_var.device,
        )
        grouped_channel_estimates = self._gather_pilot_groups(h_hat)
        grouped_error_variances = self._gather_pilot_groups(err_var)

        for tx_index in range(self._num_tx):
            for stream_index in range(self._num_streams_per_tx):
                stream_group_indices = self._stream_group_indices[tx_index][stream_index]
                pilot_symbol_indices = tuple(
                    self._group_symbol_indices[group_index]
                    for group_index in stream_group_indices
                )
                channel_values_by_symbol = []
                error_variances_by_symbol = []

                for group_index in stream_group_indices:
                    group_pilot_estimates = grouped_channel_estimates[
                        :, :, :, group_index, :
                    ]
                    channel_values_by_symbol.append(
                        self._interpolate_frequency(
                            group_pilot_estimates,
                            group_index,
                        )
                    )

                    group_error_variances = grouped_error_variances[
                        :, :, :, group_index, :
                    ]
                    error_variances_by_symbol.append(
                        self._interpolate_error_variance(group_error_variances)
                    )

                channel_values_by_symbol = tuple(channel_values_by_symbol)
                error_variances_by_symbol = tuple(error_variances_by_symbol)
                channel_estimates[:, :, :, tx_index, stream_index, :, :] = (
                    self._interpolate_time(
                        channel_values_by_symbol,
                        pilot_symbol_indices,
                    )
                )
                error_variances[:, :, :, tx_index, stream_index, :, :] = (
                    self._interpolate_time(
                        error_variances_by_symbol,
                        pilot_symbol_indices,
                    )
                )

        return channel_estimates, error_variances


class LSCombEstimator(LSChannelEstimator):
    r"""Least-squares comb-pilot channel estimator from Coleri et al. 2002.

    The estimator first applies LS channel estimation at the nonzero comb-pilot
    positions. The pilot-domain channel estimates and error variances are then
    interpolated across the resource grid with :class:`CombInterpolator`.

    For the pilot positions, the same observation model as
    :class:`~sionna.phy.ofdm.LSChannelEstimator` is used:

    .. math::

        y_i = h_i p_i + n_i

    where :math:`p_i` is a nonzero pilot symbol and :math:`n_i` has variance
    :math:`N_0`. The pilot-domain estimate and error variance are

    .. math::

        \hat{h}_i = \frac{y_i}{p_i}, \qquad
        \sigma_i^2 = \frac{N_0}{\left|p_i\right|^2}.

    The supported interpolation methods are ``"second-order"``,
    ``"low-pass"``, ``"cubic-spline"``, and ``"time-domain"``.

    :param resource_grid: Resource grid.
    :param method: Coleri interpolation method.
    :param lowpass_filter_half_length: Half the number of input pilot samples
        used by the MATLAB ``interp``-style low-pass FIR design. This argument
        is only used for ``method="low-pass"``.
    :param lowpass_cutoff: Normalized low-pass cutoff for the MATLAB
        ``interp``-style FIR design. It is only used for
        ``method="low-pass"`` and must be in `(0, 1]`.
    :param err_var_scale: Non-negative scale applied to the detector-facing LS
        error-variance fallback after interpolation.
    :param precision: Precision used for internal calculations and outputs.
        If set to `None`,
        :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device for tensor operations. If `None`,
        :attr:`~sionna.phy.config.Config.device` is used.

    :input y: [batch_size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size],
        `torch.complex`.
        Observed resource grid.
    :input no: [batch_size, num_rx, num_rx_ant] or only the first n>=0 dims,
        `torch.float`.
        Variance of the AWGN.

    :output h_hat: [batch_size, num_rx, num_rx_ant, num_tx,
        num_streams_per_tx, num_ofdm_symbols, num_effective_subcarriers], `torch.complex`.
        Channel estimates across the entire resource grid.
    :output err_var: Same shape as ``h_hat``, `torch.float`.
        Channel estimation error variance across the entire resource grid.
    """

    def __init__(
        self,
        resource_grid: ResourceGrid,
        method: str = METHOD_SECOND_ORDER,
        *,
        lowpass_filter_half_length: int = 16,
        lowpass_cutoff: float = 0.105,
        err_var_scale: float = 0.868,
        precision: Optional[Precision] = None,
        device: Optional[str] = None,
    ) -> None:
        """Initialize LS pilot estimation with a Coleri interpolator."""
        self._method = method
        self._lowpass_filter_half_length = lowpass_filter_half_length
        self._lowpass_cutoff = float(lowpass_cutoff)
        self._err_var_scale = float(err_var_scale)

        super().__init__(
            resource_grid=resource_grid,
            interpolator=CombInterpolator(
                resource_grid.pilot_pattern,
                method=method,
                lowpass_filter_half_length=lowpass_filter_half_length,
                lowpass_cutoff=lowpass_cutoff,
                err_var_scale=err_var_scale,
                precision=precision,
                device=device,
            ),
            precision=precision,
            device=device,
        )
