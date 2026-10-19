# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Benchmark runner for the Baur 2022 VAE channel estimator.

Usage:
    python -m sionna.bench.mimo_ofdm_chest.papers.baur2022.run_benchmark \
        --asset-root assets/generated
    python -m sionna.bench.mimo_ofdm_chest.papers.baur2022.run_benchmark \
        --method real --asset-root assets/generated
    python -m sionna.bench.mimo_ofdm_chest.papers.baur2022.run_benchmark --prepare-assets \
        --asset-root assets/generated
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np

from sionna.bench.mimo_ofdm_chest.benchmarking import (
    build_resource_grid,
    evaluate_channel_estimator,
)
from sionna.bench.mimo_ofdm_chest.benchmarking.result_store import (
    prepare_paper_result_dir,
    record_result,
    resolve_constructor_arguments,
)
from sionna.bench.mimo_ofdm_chest.configs import SystemConfig, smoke_config, umi_3gpp_config

from .asset import (
    METHOD_NAMES,
    METHOD_NOISY,
    METHOD_REAL,
    generate_training_data,
    load_model,
    train_model,
)
from .estimator import (
    COVARIANCE_MODES,
    VAEChannelEstimator,
)


PAPER_SLUG = "baur2022"


def _default_training_settings(smoke: bool) -> dict[str, Any]:
    """Return the smoke or canonical VAE generation recipe."""
    if smoke:
        return {
            "train_batches": 1,
            "train_batch_size": 1,
            "validation_batches": 1,
            "validation_batch_size": 1,
            "epochs": 1,
            "optimizer_batch_size": 512,
            "learning_rate": 2.30e-4,
            "free_bits": 0.875,
            "min_train_seconds": 0.0,
            "patience_epochs": 0,
            "relative_min_delta": 0.0,
        }
    return {
        "train_batches": 400,
        "train_batch_size": 16,
        "validation_batches": 16,
        "validation_batch_size": 16,
        "epochs": 200,
        "optimizer_batch_size": 512,
        "learning_rate": 2.30e-4,
        "free_bits": 0.875,
        "min_train_seconds": 0.0,
        "patience_epochs": 20,
        "relative_min_delta": 1e-4,
    }


def prepare_assets(
    cfg: SystemConfig,
    *,
    method: str,
    asset_root: Union[str, Path],
    latent_dim: int,
    train_batches: int,
    train_batch_size: int,
    validation_batches: int,
    validation_batch_size: int,
    epochs: int,
    optimizer_batch_size: int,
    learning_rate: float,
    free_bits: float,
    min_train_seconds: float,
    patience_epochs: int,
    relative_min_delta: float,
    snr_specific_weights: bool,
    training_snr_dbs: Optional[list[float]],
    covariance_floor: float,
    covariance_mode: str,
    model_seed: Optional[int] = None,
    training_data_manifest: Optional[Union[str, Path]] = None,
    overwrite_assets: bool = False,
) -> list[Path]:
    """Train one ordered VAE bank and return its manifest paths."""
    resource_grid = build_resource_grid(cfg)
    seed = cfg.seed if model_seed is None else model_seed
    if method == METHOD_NOISY and training_data_manifest is None:
        training_data_manifest = generate_training_data(
            resource_grid,
            cfg,
            train_batches=train_batches,
            train_batch_size=train_batch_size,
            validation_batches=validation_batches,
            validation_batch_size=validation_batch_size,
            training_snr_dbs=training_snr_dbs,
            seed=seed,
            asset_root=asset_root,
            overwrite=overwrite_assets,
        )
    _, _, manifest_paths = train_model(
        resource_grid,
        cfg,
        method=method,
        latent_dim=latent_dim,
        train_batches=train_batches,
        train_batch_size=train_batch_size,
        validation_batches=validation_batches,
        validation_batch_size=validation_batch_size,
        epochs=epochs,
        optimizer_batch_size=optimizer_batch_size,
        learning_rate=learning_rate,
        free_bits=free_bits,
        min_train_seconds=min_train_seconds,
        patience_epochs=patience_epochs,
        relative_min_delta=relative_min_delta,
        snr_specific_weights=snr_specific_weights,
        training_snr_dbs=training_snr_dbs,
        covariance_floor=covariance_floor,
        covariance_mode=covariance_mode,
        seed=seed,
        training_data_manifest=training_data_manifest,
        asset_root=asset_root,
        overwrite=overwrite_assets,
    )
    return manifest_paths


def run_benchmark(
    cfg: SystemConfig,
    *,
    result_name: str,
    constructor_arguments: Optional[dict[str, Any]] = None,
    smoke: bool = False,
    asset_root: Optional[Union[str, Path]] = None,
    method: str,
    latent_dim: int,
    train_batches: int,
    train_batch_size: int,
    validation_batches: int,
    validation_batch_size: int,
    epochs: int,
    optimizer_batch_size: int,
    learning_rate: float,
    free_bits: float,
    min_train_seconds: float,
    patience_epochs: int,
    relative_min_delta: float,
    snr_specific_weights: bool,
    training_snr_dbs: Optional[list[float]],
    covariance_floor: float,
    covariance_mode: str,
    model_seed: Optional[int] = None,
) -> dict[str, np.ndarray]:
    """Strictly load one VAE bank, evaluate it, and record its result."""
    constructor_arguments = dict(constructor_arguments or {})
    effective_arguments = resolve_constructor_arguments(
        VAEChannelEstimator,
        (
            "latent_dim",
            "err_var_scale",
            "covariance_floor",
            "covariance_mode",
            "interpolation_type",
        ),
        constructor_arguments,
    )
    resource_grid = build_resource_grid(cfg)
    model_states, model_noise_variances, manifest_paths = load_model(
        resource_grid,
        cfg,
        method=method,
        latent_dim=latent_dim,
        train_batches=train_batches,
        train_batch_size=train_batch_size,
        validation_batches=validation_batches,
        validation_batch_size=validation_batch_size,
        epochs=epochs,
        optimizer_batch_size=optimizer_batch_size,
        learning_rate=learning_rate,
        free_bits=free_bits,
        min_train_seconds=min_train_seconds,
        patience_epochs=patience_epochs,
        relative_min_delta=relative_min_delta,
        snr_specific_weights=snr_specific_weights,
        training_snr_dbs=training_snr_dbs,
        covariance_floor=covariance_floor,
        covariance_mode=covariance_mode,
        seed=cfg.seed if model_seed is None else model_seed,
        asset_root=asset_root,
    )
    output_dir = Path(prepare_paper_result_dir(cfg, PAPER_SLUG, smoke=smoke))
    output_path = output_dir / f"{result_name}.npz"
    result = evaluate_channel_estimator(
        VAEChannelEstimator,
        cfg,
        output_path,
        estimator_kwargs={
            "model_states": model_states,
            "model_noise_variances": model_noise_variances,
            "num_rx_ant": cfg.num_rx_ant,
            **constructor_arguments,
        },
    )
    record_result(
        cfg,
        PAPER_SLUG,
        VAEChannelEstimator,
        result_name=result_name,
        constructor_arguments=effective_arguments,
        asset_manifests=manifest_paths,
        smoke=smoke,
    )
    print(f"[{PAPER_SLUG}] result saved to {output_path}")
    return result


def _parse_args() -> argparse.Namespace:
    """Parse benchmark, asset-generation, and estimator arguments."""
    parser = argparse.ArgumentParser()

    benchmark = parser.add_argument_group("benchmark configuration")
    benchmark.add_argument("--smoke", action="store_true")
    benchmark.add_argument("--output-dir", default=None)
    benchmark.add_argument("--device", default=None)
    benchmark.add_argument(
        "--precision", choices=("single", "double"), default=None
    )
    benchmark.add_argument("--batch-size", type=int, default=None)
    benchmark.add_argument("--num-mc-batches", type=int, default=None)

    assets = parser.add_argument_group("asset handling")
    assets.add_argument("--asset-root", type=Path, default=None)
    assets.add_argument("--prepare-assets", action="store_true")
    assets.add_argument("--overwrite-assets", action="store_true")
    assets.add_argument("--train-batches", type=int, default=None)
    assets.add_argument("--train-batch-size", type=int, default=None)
    assets.add_argument("--validation-batches", type=int, default=None)
    assets.add_argument("--validation-batch-size", type=int, default=None)
    assets.add_argument("--epochs", type=int, default=None)
    assets.add_argument("--optimizer-batch-size", type=int, default=None)
    assets.add_argument("--learning-rate", type=float, default=None)
    assets.add_argument("--free-bits", type=float, default=None)
    assets.add_argument("--min-train-seconds", type=float, default=None)
    assets.add_argument("--patience-epochs", type=int, default=None)
    assets.add_argument("--relative-min-delta", type=float, default=None)
    assets.add_argument("--model-seed", type=int, default=None)
    assets.add_argument("--training-data-manifest", type=Path, default=None)
    assets.add_argument("--mixed-snr-weights", action="store_true")
    assets.add_argument(
        "--training-snr-db", type=float, action="append", default=None
    )

    estimator = parser.add_argument_group("estimator configuration")
    estimator.add_argument("--method", choices=METHOD_NAMES, default=METHOD_NOISY)
    estimator.add_argument("--latent-dim", type=int, default=None)
    estimator.add_argument("--err-var-scale", type=float, default=None)
    estimator.add_argument("--covariance-floor", type=float, default=None)
    estimator.add_argument(
        "--covariance-mode", choices=COVARIANCE_MODES, default=None
    )
    estimator.add_argument("--interpolation-type", default=None)
    estimator.add_argument("--estimator-suffix", default="")
    return parser.parse_args()


def _system_config_from_args(args: argparse.Namespace) -> SystemConfig:
    """Select the base configuration and apply explicit CLI overrides."""
    cfg = smoke_config if args.smoke else umi_3gpp_config
    overrides = {
        name: value
        for name, value in {
            "output_dir": args.output_dir,
            "device": args.device,
            "precision": args.precision,
            "batch_size": args.batch_size,
            "num_mc_batches": args.num_mc_batches,
        }.items()
        if value is not None
    }
    return replace(cfg, **overrides)


def main() -> None:
    """Prepare VAE assets or evaluate one selected Baur method."""
    args = _parse_args()
    cfg = _system_config_from_args(args)
    if args.prepare_assets and args.asset_root is None:
        raise ValueError("`--prepare-assets` requires `--asset-root`.")

    defaults = _default_training_settings(args.smoke)
    settings = {
        name: defaults[name] if value is None else value
        for name, value in {
            "train_batches": args.train_batches,
            "train_batch_size": args.train_batch_size,
            "validation_batches": args.validation_batches,
            "validation_batch_size": args.validation_batch_size,
            "epochs": args.epochs,
            "optimizer_batch_size": args.optimizer_batch_size,
            "learning_rate": args.learning_rate,
            "free_bits": args.free_bits,
            "min_train_seconds": args.min_train_seconds,
            "patience_epochs": args.patience_epochs,
            "relative_min_delta": args.relative_min_delta,
        }.items()
    }
    snr_specific_weights = not args.mixed_snr_weights
    if (
        snr_specific_weights
        and args.training_snr_db is not None
        and not args.prepare_assets
    ):
        raise ValueError(
            "Explicit SNR-specific training subsets are supported only with "
            "`--prepare-assets`; evaluation requires one state per config SNR."
        )

    estimator_defaults = resolve_constructor_arguments(
        VAEChannelEstimator,
        ("latent_dim", "covariance_floor", "covariance_mode"),
    )
    asset_arguments = {
        "method": args.method,
        "latent_dim": (
            estimator_defaults["latent_dim"]
            if args.latent_dim is None
            else args.latent_dim
        ),
        **settings,
        "snr_specific_weights": snr_specific_weights,
        "training_snr_dbs": args.training_snr_db,
        "covariance_floor": (
            estimator_defaults["covariance_floor"]
            if args.covariance_floor is None
            else args.covariance_floor
        ),
        "covariance_mode": (
            estimator_defaults["covariance_mode"]
            if args.covariance_mode is None
            else args.covariance_mode
        ),
        "model_seed": args.model_seed,
    }
    asset_root = args.asset_root
    if args.smoke:
        asset_root = asset_root or Path("assets/smoke")
    if args.prepare_assets:
        prepare_assets(
            cfg,
            asset_root=asset_root,
            training_data_manifest=args.training_data_manifest,
            overwrite_assets=args.overwrite_assets,
            **asset_arguments,
        )
        return

    constructor_arguments = {
        name: value
        for name, value in {
            "latent_dim": args.latent_dim,
            "err_var_scale": args.err_var_scale,
            "covariance_floor": args.covariance_floor,
            "covariance_mode": args.covariance_mode,
            "interpolation_type": args.interpolation_type,
        }.items()
        if value is not None
    }
    result_name = VAEChannelEstimator.__name__
    if args.method == METHOD_REAL:
        result_name += " (real)"
    run_benchmark(
        cfg,
        result_name=f"{result_name}{args.estimator_suffix}",
        constructor_arguments=constructor_arguments,
        smoke=args.smoke,
        asset_root=asset_root,
        **asset_arguments,
    )


if __name__ == "__main__":
    main()
