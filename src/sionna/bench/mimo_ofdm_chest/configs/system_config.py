# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""System configuration schema for channel-estimation evaluations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from ..persistence import sha256_fingerprint, validate_identifier


@dataclass(frozen=True)
class SystemConfig:
    """Configuration for one OFDM MU-MIMO channel-estimation system."""

    # --- Config identity ---
    name: str
    scenario_id: str
    channel_type: str
    channel_source_parameters: Dict[str, Any]

    # --- OFDM resource grid ---
    subcarrier_spacing: float
    num_prb: int
    num_ofdm_symbols: int
    cyclic_prefix_length: int
    pilot_ofdm_symbol_indices: Tuple[int, ...]

    # --- Multi-user setup ---
    num_ue: int
    num_streams_per_tx: int

    # --- UE antenna ---
    ue_polarization: str
    ue_polarization_type: str
    ue_antenna_pattern: str

    # --- BS antenna array ---
    bs_num_rows: int
    bs_num_cols: int
    bs_polarization: str
    bs_polarization_type: str
    bs_antenna_pattern: str

    carrier_frequency: float

    # --- UMi channel ---
    umi_o2i_model: str
    direction: str
    enable_pathloss: bool
    enable_shadow_fading: bool
    ue_speed: float

    # --- MCS ---
    num_bits_per_symbol: int
    coderate: float
    max_k_per_cb: int

    # --- SNR sweep ---
    snr_db_start: float
    snr_db_stop: float
    snr_db_step: float

    # --- Monte-Carlo controls ---
    batch_size: int
    num_mc_batches: int

    # --- Numeric and random state ---
    precision: str
    device: str
    # Paper assets and stochastic estimator components use `seed`; final
    # Monte Carlo channel, bit, and noise sampling uses `evaluation_seed`.
    seed: int
    evaluation_seed: int

    # --- Output ---
    output_dir: str

    def __post_init__(self) -> None:
        validate_identifier(self.scenario_id, "scenario_id")
        validate_identifier(self.channel_type, "channel_type")
        if self.device == "cuda":
            object.__setattr__(self, "device", "cuda:0")

    @property
    def fft_size(self) -> int:
        return self.num_prb * 12

    @property
    def num_rx_ant(self) -> int:
        pol = 2 if self.bs_polarization == "dual" else 1
        return self.bs_num_rows * self.bs_num_cols * pol

    @property
    def num_tx_ant(self) -> int:
        return 1 if self.ue_polarization == "single" else 2

    def snr_db_range(self) -> List[float]:
        import numpy as np

        n = int(round((self.snr_db_stop - self.snr_db_start) / self.snr_db_step)) + 1
        return [float(x) for x in np.linspace(self.snr_db_start, self.snr_db_stop, n)]

    def config_snapshot(self) -> Dict[str, Any]:
        """Return the normalized persisted representation of this config."""
        system = {
            "ofdm": {
                "subcarrier_spacing": self.subcarrier_spacing,
                "num_prb": self.num_prb,
                "num_ofdm_symbols": self.num_ofdm_symbols,
                "cyclic_prefix_length": self.cyclic_prefix_length,
            },
            "pilot_pattern": {
                "type": "kronecker",
                "ofdm_symbol_indices": self.pilot_ofdm_symbol_indices,
            },
            "multi_user": {
                "num_ue": self.num_ue,
                "num_streams_per_tx": self.num_streams_per_tx,
                "direction": self.direction,
            },
            "arrays": {
                "ue": {
                    "polarization": self.ue_polarization,
                    "polarization_type": self.ue_polarization_type,
                    "antenna_pattern": self.ue_antenna_pattern,
                },
                "bs": {
                    "num_rows": self.bs_num_rows,
                    "num_cols": self.bs_num_cols,
                    "polarization": self.bs_polarization,
                    "polarization_type": self.bs_polarization_type,
                    "antenna_pattern": self.bs_antenna_pattern,
                },
            },
            "mcs": {
                "num_bits_per_symbol": self.num_bits_per_symbol,
                "coderate": self.coderate,
                "max_k_per_cb": self.max_k_per_cb,
            },
        }
        if self.channel_type == "umi":
            channel_parameters = {
                "carrier_frequency": self.carrier_frequency,
                "o2i_model": self.umi_o2i_model,
                "enable_pathloss": self.enable_pathloss,
                "enable_shadow_fading": self.enable_shadow_fading,
                "ue_speed": self.ue_speed,
                **self.channel_source_parameters,
            }
        else:
            channel_parameters = dict(self.channel_source_parameters)

        snapshot = {
            "config_schema_version": 4,
            "scenario_id": self.scenario_id,
            "system": system,
            "channel_source": {
                "type": self.channel_type,
                "parameters": channel_parameters,
            },
            "evaluation": {
                "snr_db_start": self.snr_db_start,
                "snr_db_stop": self.snr_db_stop,
                "snr_db_step": self.snr_db_step,
                "batch_size": self.batch_size,
                "num_mc_batches": self.num_mc_batches,
                "setup_seed": self.seed,
                "evaluation_seed": self.evaluation_seed,
            },
            "numeric": {"precision": self.precision},
            "runtime": {
                "device": self.device,
                "output_dir": self.output_dir,
            },
        }
        fingerprint_input = {
            key: value for key, value in snapshot.items() if key != "runtime"
        }
        snapshot["config_fingerprint"] = sha256_fingerprint(fingerprint_input)
        return snapshot

    def config_fingerprint(self) -> str:
        """Return the canonical fingerprint stored with evaluation results."""
        return str(self.config_snapshot()["config_fingerprint"])
