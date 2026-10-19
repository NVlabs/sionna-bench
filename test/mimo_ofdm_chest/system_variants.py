# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared system-variant and NMSE-probe helpers for estimator tests."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable

import torch
from sionna.phy.ofdm import BaseChannelEstimator, ResourceGrid
from sionna.phy.utils import ebnodb2no

from sionna.bench.mimo_ofdm_chest.configs import SystemConfig, umi_3gpp_config
from sionna.bench.mimo_ofdm_chest.benchmarking.channel_source import ChannelSource
from sionna.bench.mimo_ofdm_chest.benchmarking.communication_system import (
    OFDMChannelEstimationSystem,
    build_resource_grid,
)


EstimatorFactory = Callable[
    [ResourceGrid, SystemConfig, Path],
    BaseChannelEstimator,
]


@dataclass(frozen=True)
class NMSEProbeResult:
    """Aggregate NMSE values from shared channel realizations."""

    nmse: float
    zero_estimate_nmse: float
    reference_nmse: float | None
    num_batches: int


_SYSTEM_VARIANT_OVERRIDES = {
    "baseline": {
        "num_prb": 4,
        "num_ofdm_symbols": 8,
        "cyclic_prefix_length": 6,
        "pilot_ofdm_symbol_indices": (1,),
        "num_ue": 1,
        "bs_num_rows": 1,
        "bs_num_cols": 2,
    },
    "alternate": {
        "num_prb": 6,
        "num_ofdm_symbols": 10,
        "cyclic_prefix_length": 8,
        "pilot_ofdm_symbol_indices": (2, 7),
        "num_ue": 3,
        "bs_num_rows": 2,
        "bs_num_cols": 2,
    },
    "wide": {
        "num_prb": 8,
        "num_ofdm_symbols": 12,
        "cyclic_prefix_length": 10,
        "pilot_ofdm_symbol_indices": (2, 5, 9),
        "num_ue": 4,
        "bs_num_rows": 2,
        "bs_num_cols": 6,
    },
}


def system_variant_config(
    variant: str,
    **overrides: object,
) -> SystemConfig:
    """Build one deterministic, CPU-sized 3GPP UMi test configuration."""
    if variant not in _SYSTEM_VARIANT_OVERRIDES:
        supported = ", ".join(sorted(_SYSTEM_VARIANT_OVERRIDES))
        raise ValueError(f"Unknown system variant {variant!r}; choose {supported}.")
    values = {
        **_SYSTEM_VARIANT_OVERRIDES[variant],
        "name": f"test-{variant}",
        "batch_size": 1,
        "num_mc_batches": 1,
        "num_bits_per_symbol": 2,
        "coderate": 0.5,
        "device": "cpu",
        "seed": 42,
        "evaluation_seed": 2026,
        "output_dir": "unused-test-output",
        **overrides,
    }
    return replace(umi_3gpp_config, **values)


def build_test_system(
    cfg: SystemConfig,
    estimator_factory: EstimatorFactory,
    asset_root: Path,
    *,
    channel_source: ChannelSource | None = None,
) -> OFDMChannelEstimationSystem:
    """Instantiate a paper-local estimator factory and its evaluation system."""
    resource_grid = build_resource_grid(cfg)
    estimator = estimator_factory(resource_grid, cfg, asset_root)
    return OFDMChannelEstimationSystem(
        cfg,
        estimator,
        resource_grid,
        channel_source=channel_source,
    )


def validate_estimator_output(
    h_hat: torch.Tensor,
    err_var: torch.Tensor,
    h_true: torch.Tensor,
) -> None:
    """Validate the estimator-facing shape, numeric, dtype, and device contract."""
    assert h_hat.shape == h_true.shape
    assert err_var.shape == h_hat.shape
    assert h_hat.dtype == h_true.dtype
    assert err_var.dtype == h_hat.real.dtype
    assert h_hat.device == h_true.device
    assert err_var.device == h_hat.device
    assert bool(torch.isfinite(h_hat).all())
    assert bool(torch.isfinite(err_var).all())
    assert bool(torch.all(err_var >= 0.0))


def assert_compiled_estimator_matches_eager(
    estimator: BaseChannelEstimator,
    *inputs: torch.Tensor,
    backend: str = "eager",
    fullgraph: bool = False,
    rtol: float | None = None,
    atol: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run eager and compiled inference and compare both outputs."""
    torch._dynamo.reset()
    with torch.no_grad():
        eager_output = estimator(*inputs)
        compiled_estimator = torch.compile(
            estimator,
            backend=backend,
            fullgraph=fullgraph,
        )
        compiled_output = compiled_estimator(*inputs)

    torch.testing.assert_close(
        compiled_output,
        eager_output,
        rtol=rtol,
        atol=atol,
    )
    return eager_output

def _estimator_nmse_terms(
    estimator: BaseChannelEstimator,
    received_grid: torch.Tensor,
    noise_variance: torch.Tensor,
    h_true: torch.Tensor,
) -> tuple[float, float]:
    """Run one estimator, validate its output, and return NMSE terms."""
    with torch.no_grad():
        h_hat, err_var = estimator(received_grid, noise_variance)
    validate_estimator_output(h_hat, err_var, h_true)
    numerator = float((h_hat - h_true).abs().square().sum().item())
    denominator = float(h_true.abs().square().sum().item())
    return numerator, denominator


def run_nmse_probe(
    system: OFDMChannelEstimationSystem,
    *,
    snr_db: float,
    num_batches: int = 1,
    batch_size: int | None = None,
    reference_estimator: BaseChannelEstimator | None = None,
) -> NMSEProbeResult:
    """Run estimator-only NMSE batches, optionally against a shared LS reference."""
    if num_batches < 1:
        raise ValueError("`num_batches` must be positive.")
    batch_size = system.cfg.batch_size if batch_size is None else int(batch_size)
    if batch_size < 1:
        raise ValueError("`batch_size` must be positive.")

    numerator = 0.0
    denominator = 0.0
    reference_numerator = 0.0
    reference_denominator = 0.0
    ebno_db = torch.tensor(float(snr_db), dtype=torch.float32)

    for _ in range(num_batches):
        system.channel_source.sample(batch_size)
        noise_variance = ebnodb2no(
            ebno_db,
            system.num_bits_per_symbol,
            system.coderate,
            system.rg,
        )
        # The probe intentionally skips detector/decoder work but uses the
        # production transmitter and channel-source paths.
        _, transmitted_grid = system._tx_pipeline(batch_size)
        received_grid, h_true = system.channel_source(
            transmitted_grid,
            noise_variance,
        )
        batch_numerator, batch_denominator = _estimator_nmse_terms(
            system.estimator,
            received_grid,
            noise_variance,
            h_true,
        )
        numerator += batch_numerator
        denominator += batch_denominator

        if reference_estimator is not None:
            batch_numerator, batch_denominator = _estimator_nmse_terms(
                reference_estimator,
                received_grid,
                noise_variance,
                h_true,
            )
            reference_numerator += batch_numerator
            reference_denominator += batch_denominator

    if denominator <= 0.0:
        raise AssertionError("NMSE probe sampled a zero-energy channel.")
    reference_nmse = None
    if reference_estimator is not None:
        reference_nmse = reference_numerator / reference_denominator
    return NMSEProbeResult(
        nmse=numerator / denominator,
        zero_estimate_nmse=1.0,
        reference_nmse=reference_nmse,
        num_batches=num_batches,
    )


__all__ = [
    "EstimatorFactory",
    "NMSEProbeResult",
    "assert_compiled_estimator_matches_eager",
    "build_test_system",
    "run_nmse_probe",
    "system_variant_config",
    "validate_estimator_output",
]
