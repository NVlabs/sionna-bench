# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Integration tests for Sionna's built-in channel-estimation references."""

from __future__ import annotations

import pytest
import torch

from sionna.bench.mimo_ofdm_chest.benchmarking.communication_system import OFDMChannelEstimationSystem, build_resource_grid
from sionna.bench.mimo_ofdm_chest.papers.builtin.estimator import LMMSEEstimatorWrapper, LSEstimatorWrapper
from test.mimo_ofdm_chest.system_variants import (
    assert_compiled_estimator_matches_eager,
    run_nmse_probe,
    system_variant_config,
)


pytestmark = pytest.mark.system_variants

_SYSTEM_VARIANTS = ("baseline", "alternate", "wide")
_BUILTIN_ESTIMATORS = ("ls-nn", "ls-lin", "lmmse")
_LMMSE_ORDER = "t-f-s"


def _exponential_covariance(size: int, correlation: float) -> torch.Tensor:
    """Return a Hermitian positive-definite covariance of shape [size, size]."""
    indices = torch.arange(size)
    distances = torch.abs(indices[:, None] - indices[None, :])
    return correlation**distances


def _build_builtin_estimator(name: str, resource_grid, cfg):
    if name == "ls-nn":
        return LSEstimatorWrapper(
            resource_grid,
            interpolation_type="nn",
            precision=cfg.precision,
            device=cfg.device,
        )
    if name == "ls-lin":
        return LSEstimatorWrapper(
            resource_grid,
            interpolation_type="lin",
            precision=cfg.precision,
            device=cfg.device,
        )
    if name == "lmmse":
        return LMMSEEstimatorWrapper(
            resource_grid,
            freq_cov=_exponential_covariance(cfg.fft_size, 0.95),
            time_cov=_exponential_covariance(cfg.num_ofdm_symbols, 0.99),
            space_cov=_exponential_covariance(cfg.num_rx_ant, 0.7),
            order=_LMMSE_ORDER,
            precision=cfg.precision,
            device=cfg.device,
        )
    raise ValueError(f"Unknown built-in estimator {name!r}.")


@pytest.mark.parametrize("variant", _SYSTEM_VARIANTS)
@pytest.mark.parametrize("estimator_name", _BUILTIN_ESTIMATORS)
def test_builtin_estimator_reference_integration(
    variant: str,
    estimator_name: str,
) -> None:
    """Built-in references run on every shared system configuration."""
    cfg = system_variant_config(variant)
    resource_grid = build_resource_grid(cfg)
    estimator = _build_builtin_estimator(estimator_name, resource_grid, cfg)
    system = OFDMChannelEstimationSystem(cfg, estimator, resource_grid)

    result = run_nmse_probe(system, snr_db=10.0)

    assert 0.0 <= result.nmse < result.zero_estimate_nmse
    assert result.num_batches == 1


@pytest.mark.compile
@pytest.mark.parametrize("estimator_name", ["ls-lin", "lmmse"])
def test_builtin_estimator_matches_compile_eager(estimator_name: str) -> None:
    """Built-in LS and LMMSE wrappers preserve outputs when compiled."""
    cfg = system_variant_config("baseline")
    resource_grid = build_resource_grid(cfg)
    estimator = _build_builtin_estimator(estimator_name, resource_grid, cfg)
    received_grid = torch.randn(
        1,
        1,
        cfg.num_rx_ant,
        cfg.num_ofdm_symbols,
        cfg.fft_size,
        dtype=torch.complex64,
    )
    noise_variance = torch.tensor(0.1, dtype=torch.float32)

    channel_estimate, error_variance = assert_compiled_estimator_matches_eager(
        estimator,
        received_grid,
        noise_variance,
    )

    assert channel_estimate.shape == error_variance.shape


def test_lmmse_batch_chunking_preserves_outputs() -> None:
    """Chunking the LMMSE batch changes peak memory but not its estimates."""
    cfg = system_variant_config("baseline")
    resource_grid = build_resource_grid(cfg)
    kwargs = {
        "resource_grid": resource_grid,
        "freq_cov": _exponential_covariance(cfg.fft_size, 0.95),
        "time_cov": _exponential_covariance(cfg.num_ofdm_symbols, 0.99),
        "space_cov": _exponential_covariance(cfg.num_rx_ant, 0.7),
        "order": _LMMSE_ORDER,
        "precision": cfg.precision,
        "device": cfg.device,
    }
    unchunked = LMMSEEstimatorWrapper(**kwargs, batch_chunk_size=None)
    chunked = LMMSEEstimatorWrapper(**kwargs, batch_chunk_size=1)
    received_grid = torch.randn(
        3,
        1,
        cfg.num_rx_ant,
        cfg.num_ofdm_symbols,
        cfg.fft_size,
        dtype=torch.complex64,
    )
    noise_variance = torch.tensor(0.1, dtype=torch.float32)

    expected = unchunked(received_grid, noise_variance)
    actual = chunked(received_grid, noise_variance)

    torch.testing.assert_close(actual[0], expected[0])
    torch.testing.assert_close(actual[1], expected[1])


def test_lmmse_covariances_follow_explicit_precision() -> None:
    """Explicit double precision promotes every prepared LMMSE covariance."""
    cfg = system_variant_config("baseline")
    resource_grid = build_resource_grid(cfg)
    estimator = LMMSEEstimatorWrapper(
        resource_grid,
        freq_cov=_exponential_covariance(cfg.fft_size, 0.95),
        time_cov=_exponential_covariance(cfg.num_ofdm_symbols, 0.99),
        space_cov=_exponential_covariance(cfg.num_rx_ant, 0.7),
        order=_LMMSE_ORDER,
        precision="double",
        device=cfg.device,
    )

    assert estimator.precision == "double"
    assert all(
        channel_filter._cov_mat.dtype == torch.complex128
        for channel_filter in estimator._filters
    )
