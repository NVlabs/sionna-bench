# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Lightweight smoke-test system configuration."""

from __future__ import annotations

from .system_config import SystemConfig

__all__ = ["smoke_config"]

smoke_config = SystemConfig(
    name="smoke",
    scenario_id="3gpp-umi",
    channel_type="umi",
    channel_source_parameters={"spec_version": "19.2"},
    subcarrier_spacing=15e3,
    num_prb=52,
    num_ofdm_symbols=14,
    cyclic_prefix_length=44,
    pilot_ofdm_symbol_indices=(2, 11),
    num_ue=4,
    num_streams_per_tx=1,
    ue_polarization="single",
    ue_polarization_type="V",
    ue_antenna_pattern="omni",
    bs_num_rows=4,
    bs_num_cols=4,
    bs_polarization="single",
    bs_polarization_type="V",
    bs_antenna_pattern="38.901",
    carrier_frequency=2.6e9,
    umi_o2i_model="low",
    direction="uplink",
    enable_pathloss=False,
    enable_shadow_fading=False,
    ue_speed=3.0 / 3.6,
    num_bits_per_symbol=4,
    coderate=0.60,
    max_k_per_cb=8448,
    snr_db_start=5.0,
    snr_db_stop=5.0,
    snr_db_step=2.5,
    batch_size=2,
    num_mc_batches=1,
    precision="single",
    device="cuda:0",
    seed=42,
    evaluation_seed=2026,
    output_dir="outputs/mimo_ofdm_chest",
)
