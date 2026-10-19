# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for shared channel-estimator evaluation behavior."""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import pytest
import torch
from sionna.phy.config import config as sionna_config

from sionna.bench.mimo_ofdm_chest.benchmarking.channel_estimation import (
    _add_estimator_runtime_kwargs,
    _bit_and_block_error_terms_per_example,
    _channel_nmse_terms_per_example,
    evaluate_channel_estimation_system,
    set_random_seed,
)


@dataclass(frozen=True)
class _EvaluationConfig:
    """Minimal configuration consumed by the shared evaluation loop."""

    seed: int = 42
    batch_size: int = 1
    num_mc_batches: int = 2
    precision: str = "single"
    device: str = "cpu"
    evaluation_seed: int = 2026

    @staticmethod
    def snr_db_range() -> list[float]:
        return [0.0, 5.0]


class _RecordingSystem:
    """Record batch RNG inputs and optionally perturb all RNG streams."""

    def __init__(self, num_extra_draws: int):
        self.cfg = _EvaluationConfig()
        self.num_extra_draws = num_extra_draws
        self.signatures = []

    def __call__(self, batch_size: int, ebno_db: torch.Tensor):
        del ebno_db
        self.signatures.append(
            (
                random.random(),
                float(np.random.random()),
                float(torch.rand(()).item()),
                float(torch.rand((), generator=sionna_config.torch_rng()).item()),
            )
        )

        for _ in range(self.num_extra_draws):
            random.random()
            np.random.random()
            torch.rand(())
            torch.rand((), generator=sionna_config.torch_rng())

        bits = torch.zeros(batch_size, 1, 1, 1, dtype=torch.int64)
        channel = torch.ones(batch_size, 1, 1, 1, 1, 1, dtype=torch.complex64)
        return bits, bits.clone(), channel, channel.clone()


def test_random_seed_resets_all_shared_generators() -> None:
    """The shared seed controls Python, NumPy, Torch, and Sionna streams."""

    def draw_signature() -> tuple[float, ...]:
        return (
            random.random(),
            float(np.random.random()),
            float(torch.rand(()).item()),
            float(torch.rand((), generator=sionna_config.torch_rng()).item()),
            sionna_config.py_rng.random(),
            float(sionna_config.np_rng.random()),
        )

    set_random_seed(1234)
    first = draw_signature()
    set_random_seed(1234)
    second = draw_signature()

    assert first == second


def test_error_counts_preserve_batch_elements_as_clusters() -> None:
    """Each batch element keeps its own bit and codeword sufficient statistics."""
    bits = torch.zeros(2, 1, 1, 3, 4, dtype=torch.int64)
    decoded_bits = bits.clone()
    decoded_bits[0, 0, 0, 1, 0] = 1
    decoded_bits[1, 0, 0, 2, :2] = 1

    bit_errors, block_errors, num_bits, num_blocks = (
        _bit_and_block_error_terms_per_example(bits, decoded_bits)
    )

    assert bit_errors.tolist() == [1, 2]
    assert block_errors.tolist() == [1, 1]
    assert num_bits.tolist() == [12, 12]
    assert num_blocks.tolist() == [3, 3]


def test_nmse_terms_preserve_batch_elements_as_clusters() -> None:
    """Each batch element keeps its own NMSE numerator and denominator."""
    channel = torch.ones(2, 1, 1, 1, 2, dtype=torch.complex64)
    estimate = channel.clone()
    estimate[1] = 0.0

    numerators, denominators = _channel_nmse_terms_per_example(
        estimate,
        channel,
    )

    torch.testing.assert_close(numerators, torch.tensor([0.0, 2.0]))
    torch.testing.assert_close(denominators, torch.tensor([2.0, 2.0]))


def test_estimator_runtime_kwargs_follow_the_system_config() -> None:
    """Estimator runtime settings are injected and conflicts are rejected."""
    cfg = _EvaluationConfig()

    assert _add_estimator_runtime_kwargs(cfg, {"rank": 4}) == {
        "rank": 4,
        "precision": "single",
        "device": "cpu",
    }
    with pytest.raises(ValueError, match="conflicts with cfg.precision"):
        _add_estimator_runtime_kwargs(cfg, {"precision": "double"})
    with pytest.raises(ValueError, match="conflicts with cfg.device"):
        _add_estimator_runtime_kwargs(cfg, {"device": "cuda:0"})


def test_evaluation_batches_are_deterministic_and_paired_across_snrs(tmp_path) -> None:
    """Batch inputs are RNG-independent and repeat across SNR points."""
    reference_system = _RecordingSystem(num_extra_draws=0)
    consuming_system = _RecordingSystem(num_extra_draws=100)

    evaluate_channel_estimation_system(
        reference_system,
        tmp_path / "reference.npz",
        verbose=False,
    )
    evaluate_channel_estimation_system(
        consuming_system,
        tmp_path / "consuming.npz",
        verbose=False,
    )

    assert consuming_system.signatures == reference_system.signatures
    assert reference_system.signatures[:2] == reference_system.signatures[2:]
    assert len(set(reference_system.signatures)) == 2
