# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Evaluate channel estimators on explicit channel-estimation configs."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from sionna.phy.config import config as sionna_config
from sionna.phy.ofdm import BaseChannelEstimator

from sionna.bench.mimo_ofdm_chest.configs import SystemConfig

from .channel_source import ChannelSource
from .communication_system import OFDMChannelEstimationSystem, build_resource_grid


def _add_estimator_runtime_kwargs(
    cfg: SystemConfig,
    estimator_kwargs: Optional[dict[str, Any]],
) -> dict[str, Any]:
    """Add authoritative runtime settings to estimator constructor arguments."""
    kwargs = {} if estimator_kwargs is None else dict(estimator_kwargs)
    for name in ("precision", "device"):
        configured_value = getattr(cfg, name)
        if name in kwargs and kwargs[name] != configured_value:
            raise ValueError(
                f"Estimator {name}={kwargs[name]!r} conflicts with "
                f"cfg.{name}={configured_value!r}."
            )
        kwargs[name] = configured_value
    return kwargs


def set_random_seed(seed: int) -> None:
    """Reset Python, NumPy, PyTorch, CUDA, and Sionna random generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    sionna_config.seed = seed


def _bit_and_block_error_terms_per_example(
    b: torch.Tensor,
    b_hat: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return bit and codeword counts separately for each batch element."""
    bit_error_tensor = b != b_hat
    batch_size = b.shape[0]
    bit_errors = bit_error_tensor.reshape(batch_size, -1).sum(dim=-1)
    num_bits = torch.full_like(bit_errors, bit_error_tensor[0].numel())
    block_error_tensor = torch.any(bit_error_tensor, dim=-1)
    block_errors = block_error_tensor.reshape(batch_size, -1).sum(dim=-1)
    num_blocks = torch.full_like(block_errors, block_error_tensor[0].numel())
    return bit_errors, block_errors, num_bits, num_blocks


def _channel_nmse_terms(
    h_hat: torch.Tensor,
    h_true: torch.Tensor,
) -> tuple[float, float]:
    """Return numerator and denominator terms for channel NMSE."""
    numerators, denominators = _channel_nmse_terms_per_example(h_hat, h_true)
    return float(numerators.sum().item()), float(denominators.sum().item())


def _channel_nmse_terms_per_example(
    h_hat: torch.Tensor,
    h_true: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return channel NMSE energy terms separately for each batch element."""
    reduce_dimensions = tuple(range(1, h_true.dim()))
    numerators = (h_hat - h_true).abs().square().sum(dim=reduce_dimensions)
    denominators = h_true.abs().square().sum(dim=reduce_dimensions)
    return numerators, denominators


def _calculate_confidence_intervals(
    numerators: np.ndarray,
    denominators: np.ndarray,
    confidence_level: float = 0.95,
    num_samples: int = 1000,
) -> tuple[float, float]:
    """Calculate confidence intervals for error statistics."""
    assert numerators.shape == denominators.shape

    indices = torch.randint(
        0,
        numerators.shape[0],
        (num_samples, numerators.shape[0]),
    )
    sampled_numerators = numerators[indices]
    sampled_denominators = denominators[indices]
    sampled_ratios = (
        sampled_numerators.sum(axis=1) / sampled_denominators.sum(axis=1)
    )
    lower_confidence_bound = np.quantile(
        sampled_ratios,
        (1.0 - confidence_level) / 2.0,
    )
    upper_confidence_bound = np.quantile(
        sampled_ratios,
        confidence_level / 2.0 + 0.5,
    )
    return float(lower_confidence_bound), float(upper_confidence_bound)


def evaluate_channel_estimation_system(
    system: OFDMChannelEstimationSystem,
    output_path: str | Path,
    *,
    ebno_dbs: Optional[torch.Tensor] = None,
    verbose: bool = True,
) -> dict[str, np.ndarray]:
    """Evaluate one instantiated OFDM channel-estimation system.

    :param system: System object returning ``b``, ``b_hat``, ``h_true``, and
        ``h_hat`` from its ``__call__`` method.
    :param output_path: Exact path of the result ``.npz`` file to write.
    :param ebno_dbs: Optional SNR grid. Defaults to ``cfg.snr_db_range()``.
    :param verbose: If `True`, print one summary line per SNR.

    :returns: Dictionary with ``ebno_dbs``, ``nmse``, ``ber``, and ``bler``.
    """
    cfg = system.cfg
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if ebno_dbs is None:
        ebno_dbs = torch.tensor(cfg.snr_db_range(), dtype=torch.float32)
    else:
        ebno_dbs = torch.as_tensor(ebno_dbs, dtype=torch.float32)

    ber = np.zeros(len(ebno_dbs), dtype=np.float64)
    bler = np.zeros(len(ebno_dbs), dtype=np.float64)
    nmse = np.zeros(len(ebno_dbs), dtype=np.float64)
    ber_upper_confidence_interval = np.zeros(len(ebno_dbs), dtype=np.float64)
    bler_upper_confidence_interval = np.zeros(len(ebno_dbs), dtype=np.float64)
    nmse_upper_confidence_interval = np.zeros(len(ebno_dbs), dtype=np.float64)
    ber_lower_confidence_interval = np.zeros(len(ebno_dbs), dtype=np.float64)
    bler_lower_confidence_interval = np.zeros(len(ebno_dbs), dtype=np.float64)
    nmse_lower_confidence_interval = np.zeros(len(ebno_dbs), dtype=np.float64)

    # Each UMi batch element is an independent channel realization and hence
    # one bootstrap cluster. Correlated UEs and codewords remain in that cluster.
    cluster_shape = (len(ebno_dbs), cfg.num_mc_batches, cfg.batch_size)
    bit_errors = np.zeros(cluster_shape, dtype=np.int64)
    num_bits = np.zeros(cluster_shape, dtype=np.int64)
    block_errors = np.zeros(cluster_shape, dtype=np.int64)
    num_blocks = np.zeros(cluster_shape, dtype=np.int64)
    nmse_numerators = np.zeros(cluster_shape, dtype=np.float64)
    nmse_denominators = np.zeros(cluster_shape, dtype=np.float64)

    for snr_index, ebno_db in enumerate(ebno_dbs):

        for batch_index in range(cfg.num_mc_batches):
            # Address every batch independently so estimator-side randomness
            # cannot shift the channels, bits, or noise seen by other methods.
            batch_seed = int(cfg.evaluation_seed) + batch_index
            set_random_seed(batch_seed)
            with torch.no_grad():
                b, b_hat, h_true, h_hat = system(cfg.batch_size, ebno_db)

            batch_bit_errors, batch_block_errors, batch_bits, batch_blocks = (
                _bit_and_block_error_terms_per_example(b, b_hat)
            )
            batch_nmse_numerators, batch_nmse_denominators = (
                _channel_nmse_terms_per_example(h_hat, h_true)
            )

            array_index = (snr_index, batch_index)
            bit_errors[array_index] = batch_bit_errors.cpu().numpy()
            num_bits[array_index] = batch_bits.cpu().numpy()
            block_errors[array_index] = batch_block_errors.cpu().numpy()
            num_blocks[array_index] = batch_blocks.cpu().numpy()
            nmse_numerators[array_index] = (
                batch_nmse_numerators.detach().cpu().numpy()
            )
            nmse_denominators[array_index] = (
                batch_nmse_denominators.detach().cpu().numpy()
            )

        snr_bit_errors = bit_errors[snr_index].reshape(-1)
        snr_num_bits = num_bits[snr_index].reshape(-1)
        snr_block_errors = block_errors[snr_index].reshape(-1)
        snr_num_blocks = num_blocks[snr_index].reshape(-1)
        snr_nmse_numerators = nmse_numerators[snr_index].reshape(-1)
        snr_nmse_denominators = nmse_denominators[snr_index].reshape(-1)

        ber[snr_index] = snr_bit_errors.sum() / snr_num_bits.sum()
        bler[snr_index] = snr_block_errors.sum() / snr_num_blocks.sum()
        nmse[snr_index] = snr_nmse_numerators.sum() / snr_nmse_denominators.sum()

        (
            ber_lower_confidence_interval[snr_index],
            ber_upper_confidence_interval[snr_index],
        ) = _calculate_confidence_intervals(snr_bit_errors, snr_num_bits)
        (
            bler_lower_confidence_interval[snr_index],
            bler_upper_confidence_interval[snr_index],
        ) = _calculate_confidence_intervals(snr_block_errors, snr_num_blocks)
        (
            nmse_lower_confidence_interval[snr_index],
            nmse_upper_confidence_interval[snr_index],
        ) = _calculate_confidence_intervals(
            snr_nmse_numerators,
            snr_nmse_denominators,
        )

        if verbose:
            print(
                f"Eb/N0 {float(ebno_db.cpu()):5.1f} dB: "
                f"BER={ber[snr_index]:.6e}, "
                f"BLER={bler[snr_index]:.6e}, "
                f"NMSE={nmse[snr_index]:.6e}"
            )

    results = {
        "ebno_dbs": ebno_dbs.cpu().numpy(),
        "nmse": nmse,
        "ber": ber,
        "bler": bler,
        "ber_lower_confidence_interval": ber_lower_confidence_interval,
        "ber_upper_confidence_interval": ber_upper_confidence_interval,
        "bler_lower_confidence_interval": bler_lower_confidence_interval,
        "bler_upper_confidence_interval": bler_upper_confidence_interval,
        "nmse_lower_confidence_interval": nmse_lower_confidence_interval,
        "nmse_upper_confidence_interval": nmse_upper_confidence_interval,
    }

    with output_path.open("wb") as f:
        np.savez(
            f,
            ebno_dbs=results["ebno_dbs"],
            nmse=results["nmse"],
            ber=results["ber"],
            bler=results["bler"],
            ber_lower_confidence_interval=results["ber_lower_confidence_interval"],
            ber_upper_confidence_interval=results["ber_upper_confidence_interval"],
            bler_lower_confidence_interval=results["bler_lower_confidence_interval"],
            bler_upper_confidence_interval=results["bler_upper_confidence_interval"],
            nmse_lower_confidence_interval=results["nmse_lower_confidence_interval"],
            nmse_upper_confidence_interval=results["nmse_upper_confidence_interval"],
        )

    return results


def evaluate_channel_estimator(
    estimator_class: type[BaseChannelEstimator],
    cfg: SystemConfig,
    output_path: str | Path,
    *,
    estimator_kwargs: Optional[dict[str, Any]] = None,
    channel_source: Optional[ChannelSource] = None,
    ebno_dbs: Optional[torch.Tensor] = None,
    verbose: bool = True,
) -> dict[str, np.ndarray]:
    """Evaluate one channel estimator class and store metrics in ``output_path``.

    :param estimator_class: Channel-estimator class. Must be a subclass of
        Sionna's ``BaseChannelEstimator`` and is instantiated as
        ``estimator_class(resource_grid, **estimator_kwargs)``.
    :param cfg: Channel-estimation configuration.
    :param output_path: Exact path of the result ``.npz`` file to write.
    :param estimator_kwargs: Optional keyword arguments for ``estimator_class``.
    :param channel_source: Optional channel source to use instead of the
        config's default source.
    :param ebno_dbs: Optional SNR grid. Defaults to ``cfg.snr_db_range()``.
    :param verbose: If `True`, print one summary line per SNR.

    :returns: Dictionary with ``ebno_dbs``, ``nmse``, ``ber``, and ``bler``.
    """
    estimator_kwargs = _add_estimator_runtime_kwargs(cfg, estimator_kwargs)
    resource_grid = build_resource_grid(cfg)
    estimator = estimator_class(resource_grid, **estimator_kwargs)
    system = OFDMChannelEstimationSystem(
        cfg,
        estimator,
        resource_grid=resource_grid,
        channel_source=channel_source,
    )
    return evaluate_channel_estimation_system(
        system,
        output_path,
        ebno_dbs=ebno_dbs,
        verbose=verbose,
    )
