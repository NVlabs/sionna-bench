# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Interpolation of full-subcarrier channel estimates over OFDM symbols."""

from __future__ import annotations

import torch
from sionna.phy.object import Object
from sionna.phy.ofdm.channel_estimation import _pilot_ind_from_pattern


class TimeDomainInterpolator(Object):
    r"""Interpolate full-subcarrier estimates over OFDM symbols.

    The input's compact OFDM-time axis contains one full-subcarrier estimate
    for each active pilot-bearing OFDM symbol, ordered by OFDM-symbol index.
    The original pilot pattern is used only to recover those physical symbol
    indices. Its sparse subcarrier mask does not describe the input layout.

    :param pilot_pattern: Sionna pilot pattern whose active nonzero pilots
        identify the OFDM symbols represented by the compact input axis.
    :param time_avg: If `True`, average over the compact pilot-symbol axis and
        repeat the result over all OFDM symbols. If `False`, interpolate and
        extrapolate linearly between the represented OFDM symbols.

    :input h_hat: [..., num_pilot_ofdm_symbols,
        num_effective_subcarriers], `torch.complex`.
    :input err_var: [..., num_pilot_ofdm_symbols,
        num_effective_subcarriers], `torch.float`.

    :output h_hat: [..., num_ofdm_symbols, num_effective_subcarriers],
        `torch.complex`.
    :output err_var: [..., num_ofdm_symbols, num_effective_subcarriers],
        `torch.float`.
    """

    def __init__(self, pilot_pattern, time_avg: bool = False) -> None:
        super().__init__(
            precision=pilot_pattern.precision,
            device=pilot_pattern.device,
        )
        assert pilot_pattern.num_pilot_symbols > 0, (
            "The pilot pattern cannot be empty."
        )
        assert isinstance(time_avg, bool), "`time_avg` must be boolean."
        self._time_avg = time_avg
        self._num_ofdm_symbols = int(pilot_pattern.mask.shape[-2])

        pilot_symbol_indices = self._active_pilot_symbol_indices(pilot_pattern)
        self.register_buffer(
            "_pilot_ofdm_symbol_indices",
            pilot_symbol_indices.to(device=self.device),
        )

        lower_slots, upper_slots = self._neighboring_pilot_symbol_slots(
            pilot_symbol_indices
        )
        self.register_buffer(
            "_lower_pilot_symbol_slots",
            lower_slots.to(device=self.device),
        )
        self.register_buffer(
            "_upper_pilot_symbol_slots",
            upper_slots.to(device=self.device),
        )

        lower_symbols = torch.gather(pilot_symbol_indices, 2, lower_slots)
        upper_symbols = torch.gather(pilot_symbol_indices, 2, upper_slots)
        position_difference = upper_symbols - lower_symbols
        safe_difference = torch.where(
            position_difference != 0,
            position_difference,
            torch.ones_like(position_difference),
        )
        target_symbols = torch.arange(self._num_ofdm_symbols).reshape(1, 1, -1)
        interpolation_weights = (
            (target_symbols - lower_symbols) / safe_difference
        ).unsqueeze(-1)
        self.register_buffer(
            "_interpolation_weights",
            interpolation_weights.to(dtype=torch.float32, device=self.device),
        )

    @staticmethod
    def _active_pilot_symbol_indices(pilot_pattern) -> torch.Tensor:
        """Return ordered active pilot-bearing symbols for every stream."""
        pilot_indices = _pilot_ind_from_pattern(pilot_pattern, device="cpu")
        symbol_indices = pilot_indices // int(
            pilot_pattern.num_effective_subcarriers
        )
        active_pilots = pilot_pattern.pilots.detach().cpu().abs() > 0

        stream_symbol_indices = []
        num_pilot_ofdm_symbols = None
        for tx_index in range(int(pilot_pattern.num_tx)):
            tx_symbol_indices = []
            for stream_index in range(int(pilot_pattern.num_streams_per_tx)):
                active_symbols = torch.unique(
                    symbol_indices[tx_index, stream_index][
                        active_pilots[tx_index, stream_index]
                    ],
                    sorted=True,
                )
                assert int(active_symbols.numel()) > 0, (
                    "Each stream requires at least one active pilot-bearing "
                    "OFDM symbol."
                )
                if num_pilot_ofdm_symbols is None:
                    num_pilot_ofdm_symbols = int(active_symbols.numel())
                assert int(active_symbols.numel()) == num_pilot_ofdm_symbols, (
                    "Time-domain interpolation requires the same number of "
                    "active pilot-bearing OFDM symbols for every stream."
                )
                tx_symbol_indices.append(active_symbols)
            stream_symbol_indices.append(torch.stack(tx_symbol_indices))
        return torch.stack(stream_symbol_indices)

    def _neighboring_pilot_symbol_slots(
        self,
        pilot_symbol_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return compact input slots used for each output OFDM symbol."""
        num_pilot_symbols = int(pilot_symbol_indices.shape[-1])
        output_shape = (*pilot_symbol_indices.shape[:-1], self._num_ofdm_symbols)
        if num_pilot_symbols == 1:
            slots = torch.zeros(output_shape, dtype=torch.int64)
            return slots, slots

        target_symbols = torch.arange(self._num_ofdm_symbols, dtype=torch.int64)
        target_symbols = target_symbols.expand(*output_shape)
        upper_slots = torch.searchsorted(
            pilot_symbol_indices.contiguous(),
            target_symbols.contiguous(),
            right=True,
        ).clamp(min=1, max=num_pilot_symbols - 1)
        return upper_slots - 1, upper_slots

    @staticmethod
    def _gather_pilot_symbol_slots(
        values: torch.Tensor,
        slots: torch.Tensor,
    ) -> torch.Tensor:
        """Gather compact pilot-symbol slots for every output OFDM symbol."""
        transposed = values.permute(3, 4, 5, 6, 0, 1, 2).contiguous()
        trailing_shape = transposed.shape[3:]
        gather_indices = slots.reshape(
            *slots.shape,
            *((1,) * len(trailing_shape)),
        ).expand(*slots.shape, *trailing_shape)
        gathered = torch.gather(transposed, 2, gather_indices)
        return gathered.permute(4, 5, 6, 0, 1, 2, 3).contiguous()

    def _interpolate_time(self, values: torch.Tensor) -> torch.Tensor:
        """Complete the compact full-subcarrier input over OFDM time."""
        if self._time_avg:
            mean = values.mean(dim=-2, keepdim=True)
            return mean.expand(*values.shape[:-2], self._num_ofdm_symbols, -1)

        lower_values = self._gather_pilot_symbol_slots(
            values,
            self._lower_pilot_symbol_slots,
        )
        upper_values = self._gather_pilot_symbol_slots(
            values,
            self._upper_pilot_symbol_slots,
        )
        interpolation_weights = self._interpolation_weights.to(
            dtype=values.real.dtype
        ).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        return (
            lower_values
            + interpolation_weights * (upper_values - lower_values)
        ).contiguous()

    def __call__(
        self,
        h_hat: torch.Tensor,
        err_var: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Interpolate channel estimates and error variances over OFDM time."""
        h_hat = self._interpolate_time(h_hat)
        err_var = self._interpolate_time(err_var)
        err_var = torch.maximum(err_var, torch.zeros_like(err_var))
        return h_hat, err_var
