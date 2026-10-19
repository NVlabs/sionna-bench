# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Optuna helpers for lightweight channel-estimator hyperparameter probes."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional

import numpy as np
import torch
from sionna.phy.utils import ebnodb2no

from sionna.bench.mimo_ofdm_chest.configs import SystemConfig

from .channel_estimation import (
    _add_estimator_runtime_kwargs,
    _bit_and_block_error_terms_per_example,
    _channel_nmse_terms,
    set_random_seed,
)
from .channel_source import ChannelSource
from .communication_system import OFDMChannelEstimationSystem, build_resource_grid


EstimatorKwargsSuggester = Callable[[Any, SystemConfig, Any], Mapping[str, Any]]
EstimatorObjective = Callable[[Mapping[str, Any]], float]

__all__ = [
    "EstimatorKwargsSuggester",
    "build_cached_detector_scale_objective",
    "build_cached_nmse_objective",
    "evaluate_nmse_probe",
    "evaluate_weighted_detector_probe",
    "evaluate_weighted_nmse_probe",
    "tune_channel_estimator",
]


def build_cached_nmse_objective(
    estimator_class: type,
    cfg: SystemConfig,
    *,
    ebno_dbs: Iterable[float],
    snr_weights: Optional[Iterable[float]] = None,
    num_mc_batches: Optional[int] = None,
    batch_size: Optional[int] = None,
    channel_source: Optional[ChannelSource] = None,
    resource_grid: Optional[Any] = None,
    seed: Optional[int] = None,
    verbose: bool = False,
) -> EstimatorObjective:
    """Cache received signals and channels for repeated NMSE evaluations.

    The returned objective instantiates ``estimator_class`` for each candidate
    but reuses one deterministic set of received signals, noise variances, and
    true channels. SNR points consume consecutive random draws.

    :returns: Objective accepting one complete estimator-kwargs mapping.
    """
    ebno_dbs = _float_list(ebno_dbs)
    snr_weights = _normalized_weights(ebno_dbs, snr_weights)
    seed = cfg.seed if seed is None else int(seed)
    resource_grid = (
        build_resource_grid(cfg) if resource_grid is None else resource_grid
    )
    system = OFDMChannelEstimationSystem(
        cfg,
        None,
        resource_grid=resource_grid,
        channel_source=channel_source,
    )
    num_batches = (
        cfg.num_mc_batches if num_mc_batches is None else int(num_mc_batches)
    )
    batch_size = cfg.batch_size if batch_size is None else int(batch_size)

    cached_batches = []
    set_random_seed(seed)
    with torch.no_grad():
        for ebno_db in ebno_dbs:
            ebno_tensor = torch.tensor(float(ebno_db), dtype=torch.float32)
            snr_batches = []
            for _ in range(num_batches):
                system.channel_source.sample(batch_size)
                noise_variance = ebnodb2no(
                    ebno_tensor,
                    system.num_bits_per_symbol,
                    system.coderate,
                    system.rg,
                    precision=system.cfg.precision,
                    device=system.cfg.device,
                )
                _, transmitted = system._tx_pipeline(batch_size)
                received, true_channel = system.channel_source(
                    transmitted,
                    noise_variance,
                )
                snr_batches.append(
                    tuple(
                        value.detach()
                        for value in (
                            received,
                            noise_variance,
                            true_channel,
                        )
                    )
                )
            cached_batches.append(snr_batches)
            if verbose:
                print(f"cached Eb/N0 {ebno_db:5.1f} dB", flush=True)

    def objective(estimator_kwargs: Mapping[str, Any]) -> float:
        runtime_kwargs = _add_estimator_runtime_kwargs(cfg, estimator_kwargs)
        estimator = estimator_class(resource_grid, **runtime_kwargs)
        weighted_nmse = 0.0
        with torch.no_grad():
            for ebno_db, weight, snr_batches in zip(
                ebno_dbs,
                snr_weights,
                cached_batches,
            ):
                numerator = 0.0
                denominator = 0.0
                for received, noise_variance, true_channel in snr_batches:
                    estimate, _ = estimator(received, noise_variance)
                    batch_num, batch_den = _channel_nmse_terms(
                        estimate,
                        true_channel,
                    )
                    numerator += batch_num
                    denominator += batch_den
                nmse = numerator / max(denominator, 1e-30)
                weighted_nmse += weight * nmse
                if verbose:
                    print(
                        f"Eb/N0 {ebno_db:5.1f} dB: NMSE={nmse:.6e}",
                        flush=True,
                    )
        result = weighted_nmse / max(sum(snr_weights), 1e-30)
        if verbose:
            print(f"weighted NMSE={result:.6e}", flush=True)
        return float(result)

    return objective


def evaluate_nmse_probe(
    estimator_class: type,
    cfg: SystemConfig,
    estimator_kwargs: Optional[Mapping[str, Any]] = None,
    *,
    ebno_db: float = 5.0,
    num_mc_batches: Optional[int] = None,
    batch_size: Optional[int] = None,
    channel_source: Optional[ChannelSource] = None,
    resource_grid: Optional[Any] = None,
    seed: Optional[int] = None,
    verbose: bool = False,
) -> float:
    """Evaluate one estimator with a lightweight NMSE-only Monte Carlo probe.

    The probe instantiates ``estimator_class(resource_grid,
    **estimator_kwargs)`` and runs the transmit/channel/estimation part of the
    evaluation system. It intentionally skips detector/decoder work and does
    not write benchmark result files.

    :param estimator_class: Channel-estimator class to instantiate.
    :param cfg: System configuration.
    :param estimator_kwargs: Keyword arguments passed to ``estimator_class``.
    :param ebno_db: Eb/N0 point in dB at which to evaluate NMSE.
    :param num_mc_batches: Number of Monte Carlo batches. Defaults to
        ``cfg.num_mc_batches``, matching one benchmark SNR point.
    :param batch_size: Batch size. Defaults to ``cfg.batch_size``.
    :param channel_source: Optional channel source override.
    :param resource_grid: Optional prebuilt resource grid.
    :param seed: Optional RNG seed. Defaults to ``cfg.seed``.
    :param verbose: If `True`, print the resulting NMSE.

    :returns: Scalar channel NMSE.
    """
    estimator_kwargs = _add_estimator_runtime_kwargs(cfg, estimator_kwargs)
    seed = cfg.seed if seed is None else seed

    resource_grid = (
        build_resource_grid(cfg) if resource_grid is None else resource_grid
    )
    estimator = estimator_class(resource_grid, **estimator_kwargs)
    system = OFDMChannelEstimationSystem(
        cfg,
        estimator,
        resource_grid=resource_grid,
        channel_source=channel_source,
    )
    # Estimator construction may consume a hyperparameter-dependent number of
    # random draws. Reset afterward so probe samples remain common to trials.
    set_random_seed(seed)

    num_batches = (
        cfg.num_mc_batches if num_mc_batches is None else int(num_mc_batches)
    )
    batch_size = cfg.batch_size if batch_size is None else int(batch_size)
    ebno_tensor = torch.tensor(float(ebno_db), dtype=torch.float32)

    nmse_numerator = 0.0
    nmse_denominator = 0.0
    for _ in range(num_batches):
        h_hat, h_true = _channel_estimation_batch(system, batch_size, ebno_tensor)
        batch_num, batch_den = _channel_nmse_terms(h_hat, h_true)
        nmse_numerator += batch_num
        nmse_denominator += batch_den

    nmse = nmse_numerator / max(nmse_denominator, 1e-30)
    if verbose:
        print(f"Eb/N0 {float(ebno_db):5.1f} dB: NMSE={nmse:.6e}")
    return float(nmse)


def evaluate_weighted_nmse_probe(
    estimator_class: type,
    cfg: SystemConfig,
    estimator_kwargs: Optional[Mapping[str, Any]] = None,
    *,
    ebno_dbs: Iterable[float],
    snr_weights: Optional[Iterable[float]] = None,
    num_mc_batches: Optional[int] = None,
    batch_size: Optional[int] = None,
    channel_source: Optional[ChannelSource] = None,
    resource_grid: Optional[Any] = None,
    seed: Optional[int] = None,
    verbose: bool = False,
) -> float:
    """Evaluate a weighted average of per-SNR NMSE probe values.

    Each SNR point uses ``num_mc_batches`` Monte Carlo batches and the returned
    objective is ``sum_s weight_s*NMSE_s / sum_s weight_s``. This is useful for
    tuning hyperparameters against a full SNR region instead of one selected
    operating point.

    :param estimator_class: Channel-estimator class to instantiate.
    :param cfg: System configuration.
    :param estimator_kwargs: Keyword arguments passed to ``estimator_class``.
    :param ebno_dbs: Eb/N0 points in dB.
    :param snr_weights: Optional non-negative objective weights, one per SNR.
        If omitted, all SNR points are weighted equally.
    :param num_mc_batches: Number of Monte Carlo batches per SNR. Defaults to
        ``cfg.num_mc_batches``.
    :param batch_size: Batch size. Defaults to ``cfg.batch_size``.
    :param channel_source: Optional channel source override.
    :param resource_grid: Optional prebuilt resource grid.
    :param seed: Optional RNG seed. Defaults to ``cfg.seed``.
    :param verbose: If `True`, print each per-SNR NMSE and the weighted value.

    :returns: Weighted scalar NMSE objective.
    """
    ebno_dbs = _float_list(ebno_dbs)
    snr_weights = _normalized_weights(ebno_dbs, snr_weights)
    estimator_kwargs = _add_estimator_runtime_kwargs(cfg, estimator_kwargs)
    seed = cfg.seed if seed is None else seed
    resource_grid = build_resource_grid(cfg) if resource_grid is None else resource_grid
    estimator = estimator_class(resource_grid, **estimator_kwargs)
    system = OFDMChannelEstimationSystem(
        cfg,
        estimator,
        resource_grid=resource_grid,
        channel_source=channel_source,
    )
    # Reset once after construction, not once per SNR. This gives Optuna common
    # random numbers across trials while different SNR points consume different
    # consecutive Monte Carlo channel samples.
    set_random_seed(seed)

    num_batches = cfg.num_mc_batches if num_mc_batches is None else int(num_mc_batches)
    batch_size = cfg.batch_size if batch_size is None else int(batch_size)

    weighted_nmse = 0.0
    for ebno_db, weight in zip(ebno_dbs, snr_weights):
        ebno_tensor = torch.tensor(float(ebno_db), dtype=torch.float32)
        nmse_numerator = 0.0
        nmse_denominator = 0.0
        for _ in range(num_batches):
            h_hat, h_true = _channel_estimation_batch(
                system,
                batch_size,
                ebno_tensor,
            )
            batch_num, batch_den = _channel_nmse_terms(h_hat, h_true)
            nmse_numerator += batch_num
            nmse_denominator += batch_den

        nmse = nmse_numerator / max(nmse_denominator, 1e-30)
        weighted_nmse += weight * nmse
        if verbose:
            print(
                f"Eb/N0 {float(ebno_db):5.1f} dB: "
                f"weight={weight:.6e}, NMSE={nmse:.6e}"
            )

    objective = weighted_nmse / max(sum(snr_weights), 1e-30)
    if verbose:
        print(f"weighted NMSE={objective:.6e}")
    return float(objective)


def evaluate_weighted_detector_probe(
    estimator_class: type,
    cfg: SystemConfig,
    estimator_kwargs: Optional[Mapping[str, Any]] = None,
    *,
    ebno_dbs: Iterable[float],
    snr_weights: Optional[Iterable[float]] = None,
    num_mc_batches: Optional[int] = None,
    batch_size: Optional[int] = None,
    channel_source: Optional[ChannelSource] = None,
    resource_grid: Optional[Any] = None,
    seed: Optional[int] = None,
    detector_metric: str = "mean_ber_bler",
    verbose: bool = False,
) -> float:
    """Evaluate a selected detector metric over multiple SNRs.

    Per-SNR BER, BLER, or their equal mean is averaged using ``snr_weights``.
    The same deterministic channel batches are reused across SNR points and
    estimator settings, while ``seed`` keeps calibration samples independent
    of canonical benchmark evaluation.

    :param detector_metric: ``"ber"``, ``"bler"``, or
        ``"mean_ber_bler"``.

    :returns: Weighted detector-facing objective to minimize.
    """
    _validate_detector_metric(detector_metric)
    ebno_dbs = _float_list(ebno_dbs)
    snr_weights = _normalized_weights(ebno_dbs, snr_weights)
    estimator_kwargs = _add_estimator_runtime_kwargs(cfg, estimator_kwargs)
    seed = cfg.seed if seed is None else int(seed)
    resource_grid = build_resource_grid(cfg) if resource_grid is None else resource_grid
    estimator = estimator_class(resource_grid, **estimator_kwargs)
    system = OFDMChannelEstimationSystem(
        cfg,
        estimator,
        resource_grid=resource_grid,
        channel_source=channel_source,
    )
    num_batches = cfg.num_mc_batches if num_mc_batches is None else int(num_mc_batches)
    batch_size = cfg.batch_size if batch_size is None else int(batch_size)

    weighted_objective = 0.0
    for ebno_db, weight in zip(ebno_dbs, snr_weights):
        bit_errors = 0
        block_errors = 0
        num_bits = 0
        num_blocks = 0
        ebno_tensor = torch.tensor(float(ebno_db), dtype=torch.float32)
        for batch_index in range(num_batches):
            set_random_seed(seed + batch_index)
            bits, decoded_bits, _, _ = system(batch_size, ebno_tensor)
            batch_counts = _bit_and_block_errors(bits, decoded_bits)
            bit_errors += batch_counts[0]
            block_errors += batch_counts[1]
            num_bits += batch_counts[2]
            num_blocks += batch_counts[3]

        ber = bit_errors / max(num_bits, 1)
        bler = block_errors / max(num_blocks, 1)
        metric_value = _detector_metric_value(ber, bler, detector_metric)
        weighted_objective += weight * metric_value
        if verbose:
            print(
                f"Eb/N0 {float(ebno_db):5.1f} dB: "
                f"BER={ber:.6e}, BLER={bler:.6e}, "
                f"detector objective={metric_value:.6e}"
            )

    objective = weighted_objective / max(sum(snr_weights), 1e-30)
    if verbose:
        print(f"weighted detector objective={objective:.6e}")
    return float(objective)


def build_cached_detector_scale_objective(
    estimator_class: type,
    cfg: SystemConfig,
    fixed_estimator_kwargs: Optional[Mapping[str, Any]] = None,
    *,
    scale_argument: str = "err_var_scale",
    ebno_dbs: Iterable[float],
    snr_weights: Optional[Iterable[float]] = None,
    num_mc_batches: Optional[int] = None,
    batch_size: Optional[int] = None,
    channel_source: Optional[ChannelSource] = None,
    resource_grid: Optional[Any] = None,
    seed: Optional[int] = None,
    detector_metric: str = "mean_ber_bler",
    verbose: bool = False,
) -> EstimatorObjective:
    """Cache estimates once for a scalar error-variance calibration.

    The estimator is instantiated with ``scale_argument=1.0``. Each channel
    estimate and its unscaled error variance are then cached once, while the
    returned objective reruns only detection and decoding for each candidate
    scale. This is valid only when the selected constructor argument multiplies
    ``err_var`` without changing ``h_hat`` or any other estimator behavior.

    :param estimator_class: Channel-estimator class to instantiate.
    :param cfg: System configuration.
    :param fixed_estimator_kwargs: Frozen estimator constructor arguments.
    :param scale_argument: Name of the scalar error-variance argument.
    :param ebno_dbs: Eb/N0 points in dB.
    :param snr_weights: Optional non-negative objective weights.
    :param num_mc_batches: Number of cached batches per SNR.
    :param batch_size: Batch size. Defaults to ``cfg.batch_size``.
    :param channel_source: Optional channel source override.
    :param resource_grid: Optional prebuilt resource grid.
    :param seed: Probe seed. Defaults to ``cfg.seed``.
    :param detector_metric: ``"ber"``, ``"bler"``, or
        ``"mean_ber_bler"``.
    :param verbose: If `True`, print cache and objective progress.

    :returns: Objective accepting merged estimator keyword arguments.
    """
    _validate_detector_metric(detector_metric)
    ebno_dbs = _float_list(ebno_dbs)
    snr_weights = _normalized_weights(ebno_dbs, snr_weights)
    estimator_kwargs = dict(fixed_estimator_kwargs or {})
    estimator_kwargs[scale_argument] = 1.0
    estimator_kwargs = _add_estimator_runtime_kwargs(cfg, estimator_kwargs)
    seed = cfg.seed if seed is None else int(seed)
    resource_grid = (
        build_resource_grid(cfg) if resource_grid is None else resource_grid
    )
    estimator = estimator_class(resource_grid, **estimator_kwargs)
    system = OFDMChannelEstimationSystem(
        cfg,
        estimator,
        resource_grid=resource_grid,
        channel_source=channel_source,
    )
    num_batches = (
        cfg.num_mc_batches if num_mc_batches is None else int(num_mc_batches)
    )
    batch_size = cfg.batch_size if batch_size is None else int(batch_size)

    cached_batches = []
    with torch.no_grad():
        for ebno_db in ebno_dbs:
            ebno_tensor = torch.tensor(float(ebno_db), dtype=torch.float32)
            snr_batches = []
            for batch_index in range(num_batches):
                set_random_seed(seed + batch_index)
                system.channel_source.sample(batch_size)
                noise_variance = ebnodb2no(
                    ebno_tensor,
                    system.num_bits_per_symbol,
                    system.coderate,
                    system.rg,
                    precision=system.cfg.precision,
                    device=system.cfg.device,
                )
                bits, transmitted = system._tx_pipeline(batch_size)
                received, _ = system.channel_source(
                    transmitted,
                    noise_variance,
                )
                estimate, error_variance = estimator(
                    received,
                    noise_variance,
                )
                snr_batches.append(
                    tuple(
                        value.detach()
                        for value in (
                            bits,
                            received,
                            estimate,
                            error_variance,
                            noise_variance,
                        )
                    )
                )
            cached_batches.append(snr_batches)
            if verbose:
                print(f"cached Eb/N0 {ebno_db:5.1f} dB", flush=True)

    def objective(candidate_kwargs: Mapping[str, Any]) -> float:
        error_variance_scale = float(candidate_kwargs[scale_argument])
        weighted_objective = 0.0
        with torch.no_grad():
            for ebno_db, weight, snr_batches in zip(
                ebno_dbs,
                snr_weights,
                cached_batches,
            ):
                bit_errors = 0
                block_errors = 0
                num_bits = 0
                num_blocks = 0
                for (
                    bits,
                    received,
                    estimate,
                    error_variance,
                    noise_variance,
                ) in snr_batches:
                    decoded_bits = system._detect_and_decode(
                        received,
                        estimate,
                        error_variance_scale * error_variance,
                        noise_variance,
                        batch_size,
                    )
                    counts = _bit_and_block_errors(bits, decoded_bits)
                    bit_errors += counts[0]
                    block_errors += counts[1]
                    num_bits += counts[2]
                    num_blocks += counts[3]
                ber = bit_errors / max(num_bits, 1)
                bler = block_errors / max(num_blocks, 1)
                snr_objective = _detector_metric_value(
                    ber,
                    bler,
                    detector_metric,
                )
                weighted_objective += weight * snr_objective
                if verbose:
                    print(
                        f"Eb/N0 {ebno_db:5.1f} dB: BER={ber:.6e}, "
                        f"BLER={bler:.6e}, "
                        f"detector objective={snr_objective:.6e}",
                        flush=True,
                    )
        result = weighted_objective / max(sum(snr_weights), 1e-30)
        if verbose:
            print(f"weighted detector objective={result:.6e}", flush=True)
        return float(result)

    return objective


def tune_channel_estimator(
    estimator_class: type,
    cfg: SystemConfig,
    suggest_estimator_kwargs: EstimatorKwargsSuggester,
    output_dir: str | Path,
    *,
    fixed_estimator_kwargs: Optional[Mapping[str, Any]] = None,
    estimator_kwargs_variants: Optional[Iterable[Mapping[str, Any]]] = None,
    ebno_db: float = 5.0,
    ebno_dbs: Optional[Iterable[float]] = None,
    snr_weights: Optional[Iterable[float]] = None,
    objective_metric: str = "nmse",
    detector_metric: str = "mean_ber_bler",
    n_trials: int = 50,
    num_mc_batches: Optional[int] = None,
    batch_size: Optional[int] = None,
    channel_source: Optional[ChannelSource] = None,
    study_name: Optional[str] = None,
    storage: Optional[str] = None,
    load_if_exists: bool = True,
    sampler_seed: Optional[int] = None,
    probe_seed: Optional[int] = None,
    enqueued_trials: Optional[Iterable[Mapping[str, Any]]] = None,
    timeout: Optional[float] = None,
    verbose: bool = False,
    estimator_objective: Optional[EstimatorObjective] = None,
) -> Any:
    """Tune constructor hyperparameters with an NMSE or detector probe.

    ``suggest_estimator_kwargs`` defines the paper-specific search space. It
    receives ``(trial, cfg, resource_grid)`` and must return keyword arguments
    that can be passed to ``estimator_class``. The best result is written to
    ``best_params.json`` together with the sampler/probe seeds and optimization
    runtime, and all trial summaries are written to ``trials.csv``.
    Trials execute serially because deterministic probes reset process-global
    Sionna and Torch random-number generators.

    :param estimator_class: Channel-estimator class to instantiate.
    :param cfg: System configuration.
    :param suggest_estimator_kwargs: Function defining Optuna trial
        suggestions for estimator constructor arguments.
    :param output_dir: Directory for the Optuna SQLite study and summaries.
    :param fixed_estimator_kwargs: Constructor arguments held fixed across all
        trials.
    :param estimator_kwargs_variants: Optional constructor-argument variants
        over which each trial objective is averaged. Shared fixed and suggested
        arguments are applied to every variant.
    :param ebno_db: Eb/N0 point in dB at which to optimize NMSE. Ignored when
        ``ebno_dbs`` is provided.
    :param ebno_dbs: Optional list of Eb/N0 points in dB. If provided, evaluate
        the selected objective over all listed SNRs.
    :param snr_weights: Optional non-negative objective weights for
        ``ebno_dbs``.
    :param objective_metric: ``"nmse"`` or ``"detector"``. The detector
        objective requires ``ebno_dbs``.
    :param detector_metric: Detector objective used when
        ``objective_metric="detector"``: ``"ber"``, ``"bler"``, or
        ``"mean_ber_bler"``.
    :param n_trials: Number of Optuna trials.
    :param num_mc_batches: Number of Monte Carlo batches per trial. Defaults to
        ``cfg.num_mc_batches``, matching one benchmark SNR point.
    :param batch_size: Batch size per trial. Defaults to ``cfg.batch_size``.
    :param channel_source: Optional channel source override.
    :param study_name: Optional Optuna study name.
    :param storage: Optional Optuna storage URL. Defaults to
        ``sqlite:///<output_dir>/study.db``.
    :param load_if_exists: If `True`, continue an existing study with the same
        name and storage.
    :param sampler_seed: Optional seed for Optuna's TPE sampler.
    :param probe_seed: Optional seed for every objective probe. Defaults to
        ``cfg.seed``.
    :param enqueued_trials: Optional parameter dictionaries to evaluate before
        sampler-suggested trials. They are enqueued only when creating an empty
        study, so resuming a study does not duplicate them.
    :param timeout: Optional wall-clock timeout in seconds for optimization.
    :param verbose: If `True`, print per-SNR objective values.
    :param estimator_objective: Optional custom objective that receives one
        complete estimator-kwargs mapping. If provided, it replaces the
        standard NMSE or detector probe while retaining the shared Optuna
        study, metadata, and trial-history lifecycle.

    :returns: The completed Optuna study.
    """
    import optuna

    if objective_metric not in ("nmse", "detector"):
        raise ValueError("`objective_metric` must be 'nmse' or 'detector'.")
    if objective_metric == "detector" and ebno_dbs is None:
        raise ValueError("Detector tuning requires `ebno_dbs`.")
    _validate_detector_metric(detector_metric)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    objective_ebno_dbs = (
        [float(ebno_db)] if ebno_dbs is None else _float_list(ebno_dbs)
    )
    objective_weights = _normalized_weights(objective_ebno_dbs, snr_weights)
    weighted_objective = len(objective_ebno_dbs) > 1 or any(
        weight != 1.0 for weight in objective_weights
    )

    if study_name is None:
        snr_label = (
            f"{objective_ebno_dbs[0]:g}dB"
            if len(objective_ebno_dbs) == 1
            else f"{min(objective_ebno_dbs):g}to{max(objective_ebno_dbs):g}dB"
        )
        metric_label = (
            "weighted-detector"
            if objective_metric == "detector"
            else "weighted-nmse" if weighted_objective else "nmse"
        )
        config_identity = f"{cfg.scenario_id}-{cfg.config_fingerprint()[:12]}"
        study_name = (
            f"{estimator_class.__name__}-{config_identity}-"
            f"{snr_label}-{metric_label}"
        )
    if storage is None:
        storage = f"sqlite:///{output_dir / 'study.db'}"

    effective_num_mc_batches = (
        cfg.num_mc_batches if num_mc_batches is None else int(num_mc_batches)
    )
    effective_batch_size = cfg.batch_size if batch_size is None else int(batch_size)

    sampler = None
    if sampler_seed is not None:
        sampler = optuna.samplers.TPESampler(seed=int(sampler_seed))

    resource_grid = build_resource_grid(cfg)
    fixed_estimator_kwargs = (
        {} if fixed_estimator_kwargs is None else dict(fixed_estimator_kwargs)
    )
    estimator_kwargs_variants = (
        [{}]
        if estimator_kwargs_variants is None
        else [dict(kwargs) for kwargs in estimator_kwargs_variants]
    )
    if not estimator_kwargs_variants:
        raise ValueError("`estimator_kwargs_variants` must not be empty.")

    def objective(trial: Any) -> float:
        suggested_kwargs = dict(
            suggest_estimator_kwargs(trial, cfg, resource_grid)
        )
        trial_estimator_kwargs = [
            {**fixed_estimator_kwargs, **variant_kwargs, **suggested_kwargs}
            for variant_kwargs in estimator_kwargs_variants
        ]
        trial.set_user_attr(
            "estimator_kwargs_variants",
            _jsonable(trial_estimator_kwargs),
        )
        if len(trial_estimator_kwargs) == 1:
            trial.set_user_attr(
                "estimator_kwargs",
                _jsonable(trial_estimator_kwargs[0]),
            )

        objective_values = []
        for estimator_kwargs in trial_estimator_kwargs:
            if estimator_objective is not None:
                value = estimator_objective(estimator_kwargs)
            elif objective_metric == "detector":
                value = evaluate_weighted_detector_probe(
                    estimator_class,
                    cfg,
                    estimator_kwargs,
                    ebno_dbs=objective_ebno_dbs,
                    snr_weights=objective_weights,
                    num_mc_batches=num_mc_batches,
                    batch_size=batch_size,
                    channel_source=channel_source,
                    resource_grid=resource_grid,
                    seed=probe_seed,
                    detector_metric=detector_metric,
                    verbose=verbose,
                )
            elif len(objective_ebno_dbs) == 1:
                value = evaluate_nmse_probe(
                    estimator_class,
                    cfg,
                    estimator_kwargs,
                    ebno_db=objective_ebno_dbs[0],
                    num_mc_batches=num_mc_batches,
                    batch_size=batch_size,
                    channel_source=channel_source,
                    resource_grid=resource_grid,
                    seed=probe_seed,
                    verbose=verbose,
                )
            else:
                value = evaluate_weighted_nmse_probe(
                    estimator_class,
                    cfg,
                    estimator_kwargs,
                    ebno_dbs=objective_ebno_dbs,
                    snr_weights=objective_weights,
                    num_mc_batches=num_mc_batches,
                    batch_size=batch_size,
                    channel_source=channel_source,
                    resource_grid=resource_grid,
                    seed=probe_seed,
                    verbose=verbose,
                )
            objective_values.append(value)

        objective_value = sum(objective_values) / len(objective_values)
        trial.set_user_attr(
            "variant_objective_values",
            [float(value) for value in objective_values],
        )
        trial.set_user_attr(objective_metric, float(objective_value))
        return float(objective_value)

    study = optuna.create_study(
        direction="minimize",
        study_name=study_name,
        storage=storage,
        load_if_exists=load_if_exists,
        sampler=sampler,
    )
    if not study.trials and enqueued_trials is not None:
        for parameters in enqueued_trials:
            study.enqueue_trial(dict(parameters))
    optimization_start = time.perf_counter()
    study.optimize(
        objective,
        n_trials=int(n_trials),
        timeout=timeout,
    )
    optimization_elapsed_seconds = time.perf_counter() - optimization_start

    best_trial = study.best_trial
    best_estimator_kwargs_variants = best_trial.user_attrs.get(
        "estimator_kwargs_variants",
        [],
    )
    num_estimator_variants = len(estimator_kwargs_variants)
    best_estimator_kwargs = (
        best_estimator_kwargs_variants[0]
        if num_estimator_variants == 1
        else {}
    )
    metadata = {
        "estimator_class": estimator_class.__name__,
        "config_name": cfg.name,
        "scenario_id": cfg.scenario_id,
        "config_fingerprint": cfg.config_fingerprint(),
        "metric": (
            "weighted_detector"
            if objective_metric == "detector"
            else "weighted_nmse" if weighted_objective else "nmse"
        ),
        "detector_metric": (
            detector_metric if objective_metric == "detector" else None
        ),
        "ebno_db": objective_ebno_dbs[0] if len(objective_ebno_dbs) == 1 else None,
        "ebno_dbs": objective_ebno_dbs,
        "snr_weights": objective_weights,
        "sampler_seed": sampler_seed,
        "probe_seed": cfg.seed if probe_seed is None else probe_seed,
        "n_trials": len(study.trials),
        "requested_n_trials": int(n_trials),
        "optimization_elapsed_seconds": optimization_elapsed_seconds,
        "num_mc_batches_per_snr": effective_num_mc_batches,
        "num_mc_batches_per_trial": (
            effective_num_mc_batches
            * len(objective_ebno_dbs)
            * num_estimator_variants
        ),
        "batch_size_per_snr": effective_batch_size,
        "batch_elements_per_snr": effective_num_mc_batches * effective_batch_size,
        "snr_points_per_trial": len(objective_ebno_dbs),
        "batch_elements_per_trial": (
            effective_num_mc_batches
            * effective_batch_size
            * len(objective_ebno_dbs)
            * num_estimator_variants
        ),
        "best_value": float(study.best_value),
        "best_params": _jsonable(study.best_params),
        "best_estimator_kwargs": _jsonable(best_estimator_kwargs),
        "best_estimator_kwargs_variants": _jsonable(
            best_estimator_kwargs_variants
        ),
        "best_variant_objective_values": best_trial.user_attrs.get(
            "variant_objective_values",
            [],
        ),
        "num_estimator_variants": num_estimator_variants,
        "study_name": study.study_name,
        "storage": storage,
    }
    with (output_dir / "best_params.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)
        f.write("\n")
    _write_trials_csv(study, output_dir / "trials.csv")

    return study


def _bit_and_block_errors(
    bits: torch.Tensor,
    decoded_bits: torch.Tensor,
) -> tuple[int, int, int, int]:
    """Aggregate bit and codeword counts for a detector-facing probe."""
    counts = _bit_and_block_error_terms_per_example(bits, decoded_bits)
    return tuple(int(count.sum().item()) for count in counts)


def _jsonable(value: Any) -> Any:
    """Convert common Python values to JSON-compatible representations."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, np.ndarray):
        return {
            "type": "numpy.ndarray",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
    if isinstance(value, torch.Tensor):
        return {
            "type": "torch.Tensor",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "device": str(value.device),
        }
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return str(value)


def _float_list(values: Iterable[float]) -> list[float]:
    """Return ``values`` as a non-empty list of floats."""
    out = [float(value) for value in values]
    if not out:
        raise ValueError("At least one SNR point is required.")
    return out


def _normalized_weights(
    ebno_dbs: list[float],
    snr_weights: Optional[Iterable[float]],
) -> list[float]:
    """Return validated SNR weights with one entry per SNR point."""
    if snr_weights is None:
        return [1.0 for _ in ebno_dbs]
    weights = _float_list(snr_weights)
    if len(weights) != len(ebno_dbs):
        raise ValueError("`snr_weights` must match the number of SNR points.")
    if any(weight < 0.0 for weight in weights):
        raise ValueError("`snr_weights` must be non-negative.")
    if not any(weight > 0.0 for weight in weights):
        raise ValueError("At least one SNR weight must be positive.")
    return weights


def _validate_detector_metric(detector_metric: str) -> None:
    """Validate a detector-facing objective name."""
    if detector_metric not in ("ber", "bler", "mean_ber_bler"):
        raise ValueError(
            "`detector_metric` must be 'ber', 'bler', or 'mean_ber_bler'."
        )


def _detector_metric_value(
    ber: float,
    bler: float,
    detector_metric: str,
) -> float:
    """Return the selected detector-facing scalar metric."""
    if detector_metric == "ber":
        return ber
    if detector_metric == "bler":
        return bler
    return 0.5 * (ber + bler)


def _channel_estimation_batch(
    system: OFDMChannelEstimationSystem,
    batch_size: int,
    ebno_db: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run one batch up to channel estimation and skip detection/decoding."""
    system.channel_source.sample(batch_size)
    no = ebnodb2no(
        ebno_db,
        system.num_bits_per_symbol,
        system.coderate,
        system.rg,
        precision=system.cfg.precision,
        device=system.cfg.device,
    )
    _, x = system._tx_pipeline(batch_size)
    y, h_true = system.channel_source(x, no)

    if system.estimator is None:
        h_hat = h_true
    else:
        h_hat, _ = system.estimator(y, no)
    return h_hat, h_true


def _write_trials_csv(study: Any, path: Path) -> None:
    """Write a compact Optuna trial table without requiring pandas."""
    param_names = sorted({name for trial in study.trials for name in trial.params})
    fieldnames = [
        "number",
        "state",
        "value",
        "duration_seconds",
        *[f"param_{name}" for name in param_names],
    ]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for trial in study.trials:
            duration_seconds = ""
            if trial.datetime_start is not None and trial.datetime_complete is not None:
                duration_seconds = (
                    trial.datetime_complete - trial.datetime_start
                ).total_seconds()
            row = {
                "number": trial.number,
                "state": trial.state.name,
                "value": "" if trial.value is None else trial.value,
                "duration_seconds": duration_seconds,
            }
            for name in param_names:
                row[f"param_{name}"] = trial.params.get(name, "")
            writer.writerow(row)
