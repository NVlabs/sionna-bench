# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the shared OFDM channel-estimation system."""

from __future__ import annotations

import torch
from sionna.phy.config import config as sionna_config

from sionna.bench.mimo_ofdm_chest.benchmarking.communication_system import (
    OFDMChannelEstimationSystem,
    build_resource_grid,
)
from test.mimo_ofdm_chest.system_variants import system_variant_config


def test_resource_grid_pilots_preserve_runtime_rng_state() -> None:
    """Pilot construction is deterministic without consuming the runtime RNG."""
    cfg = system_variant_config("alternate")
    generator = sionna_config.torch_rng(cfg.device)
    initial_state = generator.get_state()

    first = build_resource_grid(cfg)
    second = build_resource_grid(cfg)

    torch.testing.assert_close(
        first.pilot_pattern.pilots,
        second.pilot_pattern.pilots,
    )
    assert torch.equal(generator.get_state(), initial_state)


def test_system_objects_use_configured_runtime() -> None:
    """Shared Sionna blocks use the configured precision and device."""
    cfg = system_variant_config("baseline", precision="double", device="cpu")
    resource_grid = build_resource_grid(cfg)
    system = OFDMChannelEstimationSystem(cfg, None, resource_grid)

    for obj in (
        resource_grid,
        system.source,
        system.encoder,
        system.decoder,
        system.mapper,
        system.rg_mapper,
        system.detector,
        system.channel_source.channel_model,
        system.channel_source.ofdm_channel,
    ):
        assert obj.precision == "double"
        assert obj.device == "cpu"
    assert (
        system.channel_source.channel_model._scenario.spec_version
        == cfg.channel_source_parameters["spec_version"]
    )


def test_system_preserves_ldpc_codeword_axis() -> None:
    """Transmitted and decoded bits retain each individual LDPC codeword."""
    cfg = system_variant_config("baseline", max_k_per_cb=100)
    resource_grid = build_resource_grid(cfg)
    system = OFDMChannelEstimationSystem(cfg, None, resource_grid)

    bits, decoded_bits, _, _ = system(1, torch.tensor(10.0))

    assert system.num_cbs > 1
    assert bits.shape == decoded_bits.shape
    assert bits.shape == (1, 1, 1, system.num_cbs, system.k_cb)
