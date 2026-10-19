# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Executable Sionna system for channel-estimator evaluation."""

from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np
import torch

from sionna.phy.config import config as sionna_config
from sionna.phy.fec.ldpc import LDPC5GDecoder, LDPC5GEncoder
from sionna.phy.mapping import BinarySource, Mapper
from sionna.phy.mimo import StreamManagement
from sionna.phy.ofdm import (
    KroneckerPilotPattern,
    LinearDetector,
    PilotPattern,
    ResourceGrid,
    ResourceGridMapper,
)
from sionna.phy.utils import ebnodb2no

from sionna.bench.mimo_ofdm_chest.configs import SystemConfig

from .channel_source import ChannelSource, build_channel_source


def pick_cb_layout(n_total: int, m: int, R: float, max_k: int) -> Tuple[int, int, int]:
    """Return (num_cbs, k_cb, n_cb) with n_cb a multiple of m and k_cb <= max_k."""
    num_cbs = max(1, int(np.ceil(R * n_total / max_k)))
    while True:
        n_cb = (n_total // num_cbs) // m * m
        if n_cb <= 0:
            raise ValueError("cannot find a valid code-block layout")
        k_cb = int(round(R * n_cb))
        if k_cb <= max_k:
            return num_cbs, k_cb, n_cb
        num_cbs += 1


def build_resource_grid(cfg: SystemConfig) -> ResourceGrid:
    """Build the benchmark OFDM resource grid for ``cfg``."""
    pilot_symbol_indices = list(cfg.pilot_ofdm_symbol_indices)
    generator = sionna_config.torch_rng("cpu")
    generator_state = generator.get_state()
    generator.manual_seed(cfg.seed)
    try:
        pilot_template = ResourceGrid(
            num_ofdm_symbols=cfg.num_ofdm_symbols,
            fft_size=cfg.fft_size,
            subcarrier_spacing=cfg.subcarrier_spacing,
            num_tx=cfg.num_ue,
            num_streams_per_tx=cfg.num_streams_per_tx,
            cyclic_prefix_length=cfg.cyclic_prefix_length,
            pilot_pattern="empty",
            precision=cfg.precision,
            device="cpu",
        )
        cpu_pilot_pattern = KroneckerPilotPattern(
            pilot_template,
            pilot_symbol_indices,
            precision=cfg.precision,
            device="cpu",
        )
    finally:
        generator.set_state(generator_state)

    pilot_pattern = PilotPattern(
        cpu_pilot_pattern.mask,
        cpu_pilot_pattern.pilots,
        precision=cfg.precision,
        device=cfg.device,
    )
    return ResourceGrid(
        num_ofdm_symbols=cfg.num_ofdm_symbols,
        fft_size=cfg.fft_size,
        subcarrier_spacing=cfg.subcarrier_spacing,
        num_tx=cfg.num_ue,
        num_streams_per_tx=cfg.num_streams_per_tx,
        cyclic_prefix_length=cfg.cyclic_prefix_length,
        pilot_pattern=pilot_pattern,
        pilot_ofdm_symbol_indices=pilot_symbol_indices,
        precision=cfg.precision,
        device=cfg.device,
    )


class OFDMChannelEstimationSystem:
    """One-batch OFDM channel-estimation system with an injected estimator."""

    def __init__(
        self,
        cfg: SystemConfig,
        estimator: Optional[Any],
        resource_grid: ResourceGrid,
        channel_source: Optional[ChannelSource] = None,
    ):
        self.cfg = cfg
        self.estimator = estimator
        self.rg = resource_grid
        self.channel_source = (
            build_channel_source(cfg, resource_grid)
            if channel_source is None
            else channel_source
        )
        self.num_bits_per_symbol = cfg.num_bits_per_symbol
        self.coderate = cfg.coderate

        num_data_re = int(self.rg.num_data_symbols)
        self.num_data_re = num_data_re
        n_total = num_data_re * self.num_bits_per_symbol
        self.num_cbs, self.k_cb, self.n_cb = pick_cb_layout(
            n_total,
            self.num_bits_per_symbol,
            self.coderate,
            max_k=cfg.max_k_per_cb,
        )
        self.n_used = self.num_cbs * self.n_cb
        self.k_used = self.num_cbs * self.k_cb
        self.pad_bits = n_total - self.n_used

        self.source = BinarySource(
            precision=cfg.precision,
            device=cfg.device,
        )
        self.encoder = LDPC5GEncoder(
            k=self.k_cb,
            n=self.n_cb,
            num_bits_per_symbol=self.num_bits_per_symbol,
            precision=cfg.precision,
            device=cfg.device,
        )
        self.decoder = LDPC5GDecoder(
            self.encoder,
            hard_out=True,
            return_infobits=True,
            precision=cfg.precision,
            device=cfg.device,
        )
        self.mapper = Mapper(
            "qam",
            self.num_bits_per_symbol,
            precision=cfg.precision,
            device=cfg.device,
        )
        self.rg_mapper = ResourceGridMapper(
            self.rg,
            precision=cfg.precision,
            device=cfg.device,
        )

        rx_tx_assoc = np.ones((1, cfg.num_ue), dtype=bool)
        self.sm = StreamManagement(
            rx_tx_assoc,
            num_streams_per_tx=cfg.num_streams_per_tx,
        )

        self.detector = LinearDetector(
            equalizer="lmmse",
            output="bit",
            demapping_method="maxlog",
            resource_grid=self.rg,
            stream_management=self.sm,
            constellation_type="qam",
            num_bits_per_symbol=self.num_bits_per_symbol,
            precision=cfg.precision,
            device=cfg.device,
        )

    def _tx_pipeline(self, batch_size: int):
        b = self.source(
            [
                batch_size,
                self.cfg.num_ue,
                self.cfg.num_streams_per_tx,
                self.num_cbs,
                self.k_cb,
            ]
        )
        c = self.encoder(b)
        c_flat = c.reshape(
            batch_size,
            self.cfg.num_ue,
            self.cfg.num_streams_per_tx,
            self.n_used,
        )
        if self.pad_bits > 0:
            pad = torch.zeros(
                batch_size,
                self.cfg.num_ue,
                self.cfg.num_streams_per_tx,
                self.pad_bits,
                dtype=c_flat.dtype,
                device=c_flat.device,
            )
            c_full = torch.cat([c_flat, pad], dim=-1)
        else:
            c_full = c_flat
        x_sym = self.mapper(c_full)
        x = self.rg_mapper(x_sym)
        return b, x

    def _detect_and_decode(
        self,
        y: torch.Tensor,
        h_for_detector: torch.Tensor,
        err_var: torch.Tensor,
        no: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        llr_full = self.detector(y, h_for_detector, err_var, no)
        llr = llr_full[..., : self.n_used].reshape(
            batch_size,
            self.cfg.num_ue,
            self.cfg.num_streams_per_tx,
            self.num_cbs,
            self.n_cb,
        )
        return self.decoder(llr)

    def __call__(self, batch_size: int, ebno_db: torch.Tensor):
        """Run one Monte Carlo batch.

        :returns: ``(b, b_hat, h_true, h_hat)`` where ``b`` and ``b_hat`` have
            shape ``[B, num_ue, num_streams_per_tx, num_cbs, k_cb]``.
        """
        self.channel_source.sample(batch_size)
        no = ebnodb2no(
            ebno_db,
            self.num_bits_per_symbol,
            self.coderate,
            self.rg,
            precision=self.cfg.precision,
            device=self.cfg.device,
        )
        b, x = self._tx_pipeline(batch_size)
        y, h_true = self.channel_source(x, no)

        if self.estimator is None:
            h_hat = h_true
            err_var = torch.zeros_like(h_true.real)
        else:
            h_hat, err_var = self.estimator(y, no)
        b_hat = self._detect_and_decode(y, h_hat, err_var, no, batch_size)
        return b, b_hat, h_true, h_hat
