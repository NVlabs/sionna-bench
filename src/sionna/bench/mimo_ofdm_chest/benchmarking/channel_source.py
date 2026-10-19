# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Channel sources for OFDM channel-estimation evaluation."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from sionna.phy.channel import (
    OFDMChannel,
    gen_single_sector_topology,
)
from sionna.phy.channel.tr38901 import Antenna, AntennaArray, UMi
from sionna.phy.ofdm import ResourceGrid

from sionna.bench.mimo_ofdm_chest.configs import SystemConfig


class ChannelSource(ABC):
    """Source of channel realizations for one OFDM evaluation system."""

    def __init__(self, cfg: SystemConfig, resource_grid: ResourceGrid):
        self.cfg = cfg
        self.resource_grid = resource_grid

    @abstractmethod
    def sample(self, batch_size: int) -> None:
        """Sample or select the channel realization for the next call."""

    @abstractmethod
    def __call__(
        self,
        x: torch.Tensor,
        no: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply the selected channel realization.

        :param x: Transmitted resource grid.
        :param no: Noise variance.

        :returns: Tuple ``(y, h_true)``.
        """


def _build_umi_channel_model(cfg: SystemConfig):
    """Build the current benchmark's TR 38.901 UMi channel model."""
    spec_version = cfg.channel_source_parameters.get("spec_version")
    if spec_version != "19.2":
        raise ValueError(
            "The canonical UMi benchmark requires TR 38.901 "
            "`spec_version='19.2'`."
        )
    ut = Antenna(
        polarization=cfg.ue_polarization,
        polarization_type=cfg.ue_polarization_type,
        antenna_pattern=cfg.ue_antenna_pattern,
        carrier_frequency=cfg.carrier_frequency,
        precision=cfg.precision,
        device=cfg.device,
    )
    bs = AntennaArray(
        num_rows=cfg.bs_num_rows,
        num_cols=cfg.bs_num_cols,
        polarization=cfg.bs_polarization,
        polarization_type=cfg.bs_polarization_type,
        antenna_pattern=cfg.bs_antenna_pattern,
        carrier_frequency=cfg.carrier_frequency,
        precision=cfg.precision,
        device=cfg.device,
    )
    return UMi(
        carrier_frequency=cfg.carrier_frequency,
        o2i_model=cfg.umi_o2i_model,
        ut_array=ut,
        bs_array=bs,
        direction=cfg.direction,
        enable_pathloss=cfg.enable_pathloss,
        enable_shadow_fading=cfg.enable_shadow_fading,
        spec_version=spec_version,
        precision=cfg.precision,
        device=cfg.device,
    )


class UMiChannelSource(ChannelSource):
    """TR 38.901 UMi channel source using Sionna's system-level model."""

    def __init__(
        self,
        cfg: SystemConfig,
        resource_grid: ResourceGrid,
    ):
        super().__init__(cfg, resource_grid)
        self._channel_model = _build_umi_channel_model(cfg)
        self.ofdm_channel = OFDMChannel(
            self.channel_model,
            self.resource_grid,
            return_channel=True,
            precision=cfg.precision,
            device=cfg.device,
        )

    @property
    def channel_model(self):
        """Underlying Sionna UMi model for UMi-specific asset generation."""
        return self._channel_model

    def sample(self, batch_size: int) -> None:
        topology = gen_single_sector_topology(
            batch_size=batch_size,
            num_ut=self.cfg.num_ue,
            scenario="umi",
            min_ut_velocity=self.cfg.ue_speed,
            max_ut_velocity=self.cfg.ue_speed,
            precision=self.cfg.precision,
            device=self.cfg.device,
        )
        self.channel_model.set_topology(*topology)

    def __call__(
        self,
        x: torch.Tensor,
        no: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.ofdm_channel(x, no)


def build_channel_source(
    cfg: SystemConfig,
    resource_grid: ResourceGrid,
) -> ChannelSource:
    """Build the channel source configured by ``cfg``.

    The current config schema only defines the UMi benchmark. Future configs can
    add a channel type and extend this factory without changing the evaluation
    system.
    """
    channel_type = cfg.channel_type
    if channel_type == "umi":
        return UMiChannelSource(cfg, resource_grid)
    raise ValueError(f"Unsupported channel type: {channel_type!r}")
