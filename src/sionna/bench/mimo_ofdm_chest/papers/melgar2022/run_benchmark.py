# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Benchmark runner for the Melgar 2022 DNN estimators.

Usage:
    python -m sionna.bench.mimo_ofdm_chest.papers.melgar2022.run_benchmark \
        --asset-root assets/generated
    python -m sionna.bench.mimo_ofdm_chest.papers.melgar2022.run_benchmark --smoke \
        --asset-root assets/smoke
    python -m sionna.bench.mimo_ofdm_chest.papers.melgar2022.run_benchmark \
        --method model3 --asset-root assets/generated
    python -m sionna.bench.mimo_ofdm_chest.papers.melgar2022.run_benchmark --prepare-assets \
        --method model2 --asset-root assets/generated
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np

from sionna.bench.mimo_ofdm_chest.benchmarking import (
    build_resource_grid,
    evaluate_channel_estimator,
)
from sionna.bench.mimo_ofdm_chest.benchmarking.result_store import (
    prepare_paper_result_dir,
    record_result,
    resolve_callable_arguments,
    resolve_constructor_arguments,
)
from sionna.bench.mimo_ofdm_chest.configs import SystemConfig, smoke_config, umi_3gpp_config

from .asset import load_model2, load_model3, train_model2, train_model3
from .estimator import (
    DNNChannelEstimator,
    InterpolatedDNNChannelEstimator,
    METHOD_MODEL_2,
    METHOD_MODEL_3,
    METHOD_NAMES,
)


PAPER_SLUG = "melgar2022"


def _default_training_settings(smoke: bool, method: str) -> dict[str, Any]:
    """Return the selected method's smoke or canonical generation recipe."""
    assert method in METHOD_NAMES, f"`method` must be one of {METHOD_NAMES}."
    if smoke:
        if method == METHOD_MODEL_3:
            return {
                "train_batches": 2,
                "train_batch_size": 2,
                "validation_batches": 0,
                "validation_batch_size": 2,
                "epochs": 5,
                "optimizer_batch_size": 4096,
                "learning_rate": 1e-4,
                "patience_epochs": 0,
                "min_delta": 0.0,
                "max_examples_per_batch": 4096,
            }
        return {
            "train_batches": 8,
            "train_batch_size": 8,
            "validation_batches": 0,
            "validation_batch_size": 8,
            "epochs": 100,
            "optimizer_batch_size": 500,
            "learning_rate": 5e-4,
            "patience_epochs": 0,
            "min_delta": 0.0,
            "max_examples_per_batch": None,
        }

    trainer = train_model2 if method == METHOD_MODEL_2 else train_model3
    argument_names = [
        "train_batches",
        "train_batch_size",
        "validation_batches",
        "validation_batch_size",
        "epochs",
        "optimizer_batch_size",
        "learning_rate",
        "patience_epochs",
        "min_delta",
    ]
    settings = resolve_callable_arguments(trainer, argument_names)
    settings["max_examples_per_batch"] = (
        None
        if method == METHOD_MODEL_2
        else resolve_callable_arguments(
            train_model3,
            ("max_examples_per_batch",),
        )["max_examples_per_batch"]
    )
    return settings


def prepare_assets(
    cfg: SystemConfig,
    *,
    method: str,
    asset_root: Union[str, Path],
    block_len: int,
    train_batches: int,
    train_batch_size: int,
    validation_batches: int,
    validation_batch_size: int,
    epochs: int,
    optimizer_batch_size: int,
    learning_rate: float,
    min_train_seconds: float,
    patience_epochs: int,
    min_delta: float,
    max_examples_per_batch: Optional[int],
    snr_specific_weights: bool,
    training_snr_dbs: Optional[list[float]],
    training_data_manifest: Optional[Union[str, Path]] = None,
    model_seed: Optional[int] = None,
    overwrite_assets: bool = False,
) -> list[Path]:
    """Train one ordered Melgar model bank and return its manifest paths."""
    resource_grid = build_resource_grid(cfg)
    training_arguments = {
        "train_batches": train_batches,
        "train_batch_size": train_batch_size,
        "validation_batches": validation_batches,
        "validation_batch_size": validation_batch_size,
        "epochs": epochs,
        "optimizer_batch_size": optimizer_batch_size,
        "learning_rate": learning_rate,
        "min_train_seconds": min_train_seconds,
        "patience_epochs": patience_epochs,
        "min_delta": min_delta,
        "snr_specific_weights": snr_specific_weights,
        "training_snr_dbs": training_snr_dbs,
        "training_data_manifest": training_data_manifest,
        "seed": cfg.seed if model_seed is None else model_seed,
        "asset_root": asset_root,
        "overwrite": overwrite_assets,
    }
    if method == METHOD_MODEL_2:
        _, _, manifest_paths = train_model2(
            resource_grid,
            cfg,
            **training_arguments,
        )
    elif method == METHOD_MODEL_3:
        _, _, manifest_paths = train_model3(
            resource_grid,
            cfg,
            block_len=block_len,
            max_examples_per_batch=max_examples_per_batch,
            **training_arguments,
        )
    else:
        raise ValueError(f"Unknown Melgar method: {method!r}.")
    return manifest_paths


def run_benchmark(
    cfg: SystemConfig,
    *,
    method: str,
    result_name: str,
    constructor_arguments: Optional[dict[str, Any]] = None,
    smoke: bool = False,
    asset_root: Optional[Union[str, Path]] = None,
    block_len: int,
    train_batches: int,
    train_batch_size: int,
    validation_batches: int,
    validation_batch_size: int,
    epochs: int,
    optimizer_batch_size: int,
    learning_rate: float,
    min_train_seconds: float,
    patience_epochs: int,
    min_delta: float,
    max_examples_per_batch: Optional[int],
    snr_specific_weights: bool,
    training_snr_dbs: Optional[list[float]],
    model_seed: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """Strictly load one model bank, evaluate it, and record its result."""
    constructor_arguments = dict(constructor_arguments or {})
    resource_grid = build_resource_grid(cfg)
    loading_arguments = {
        "train_batches": train_batches,
        "train_batch_size": train_batch_size,
        "validation_batches": validation_batches,
        "validation_batch_size": validation_batch_size,
        "epochs": epochs,
        "optimizer_batch_size": optimizer_batch_size,
        "learning_rate": learning_rate,
        "min_train_seconds": min_train_seconds,
        "patience_epochs": patience_epochs,
        "min_delta": min_delta,
        "snr_specific_weights": snr_specific_weights,
        "training_snr_dbs": training_snr_dbs,
        "seed": cfg.seed if model_seed is None else model_seed,
        "asset_root": asset_root,
    }
    if method == METHOD_MODEL_2:
        model_states, model_noise_variances, manifest_paths = load_model2(
            resource_grid,
            cfg,
            **loading_arguments,
        )
    elif method == METHOD_MODEL_3:
        model_states, model_noise_variances, manifest_paths = load_model3(
            resource_grid,
            cfg,
            block_len=block_len,
            max_examples_per_batch=max_examples_per_batch,
            **loading_arguments,
        )
    else:
        raise ValueError(f"Unknown Melgar method: {method!r}.")

    estimator_kwargs: dict[str, Any] = {
        "model_states": model_states,
        "model_noise_variances": model_noise_variances,
        **constructor_arguments,
    }
    if method == METHOD_MODEL_2:
        estimator_class = DNNChannelEstimator
        argument_names = ("interpolation_type", "err_var_scale")
    elif method == METHOD_MODEL_3:
        estimator_class = InterpolatedDNNChannelEstimator
        argument_names = ("block_len", "err_var_scale")
    effective_arguments = resolve_constructor_arguments(
        estimator_class,
        argument_names,
        constructor_arguments,
    )

    output_dir = Path(prepare_paper_result_dir(cfg, PAPER_SLUG, smoke=smoke))
    output_path = output_dir / f"{result_name}.npz"
    result = evaluate_channel_estimator(
        estimator_class,
        cfg,
        output_path,
        estimator_kwargs=estimator_kwargs,
    )
    record_result(
        cfg,
        PAPER_SLUG,
        estimator_class,
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
        "--precision",
        choices=("single", "double"),
        default=None,
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
    assets.add_argument("--min-train-seconds", type=float, default=0.0)
    assets.add_argument("--patience-epochs", type=int, default=None)
    assets.add_argument("--min-delta", type=float, default=None)
    assets.add_argument("--max-examples-per-batch", type=int, default=None)
    assets.add_argument("--model-seed", type=int, default=None)
    assets.add_argument("--training-data-manifest", type=Path, default=None)
    assets.add_argument(
        "--mixed-snr-weights",
        action="store_true",
        help="Use one DNN trained across the requested SNR mixture.",
    )
    assets.add_argument(
        "--training-snr-db",
        type=float,
        action="append",
        default=None,
        help="Training SNR in dB; repeat to provide an explicit SNR grid.",
    )

    estimator = parser.add_argument_group("estimator configuration")
    estimator.add_argument(
        "--method",
        choices=METHOD_NAMES,
        default=METHOD_MODEL_3,
    )
    estimator.add_argument("--block-len", type=int, default=None)
    estimator.add_argument("--err-var-scale", type=float, default=None)
    estimator.add_argument(
        "--interpolation-type",
        choices=("nn", "lin", "lin_time_avg"),
        default=None,
    )
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
    """Prepare model assets or evaluate one selected Melgar method."""
    args = _parse_args()
    cfg = _system_config_from_args(args)
    if args.prepare_assets and args.asset_root is None:
        raise ValueError("`--prepare-assets` requires `--asset-root`.")

    defaults = _default_training_settings(args.smoke, args.method)
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
            "patience_epochs": args.patience_epochs,
            "min_delta": args.min_delta,
            "max_examples_per_batch": args.max_examples_per_batch,
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
            "`--prepare-assets`; benchmark evaluation requires one state per "
            "configured SNR."
        )

    block_len = resolve_constructor_arguments(
        InterpolatedDNNChannelEstimator,
        ("block_len",),
        {} if args.block_len is None else {"block_len": args.block_len},
    )["block_len"]
    asset_arguments = {
        "method": args.method,
        "block_len": block_len,
        **settings,
        "min_train_seconds": args.min_train_seconds,
        "snr_specific_weights": snr_specific_weights,
        "training_snr_dbs": args.training_snr_db,
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

    estimator_class = (
        DNNChannelEstimator
        if args.method == METHOD_MODEL_2
        else InterpolatedDNNChannelEstimator
    )
    constructor_arguments = {}
    if args.err_var_scale is not None:
        constructor_arguments["err_var_scale"] = args.err_var_scale
    if args.method == METHOD_MODEL_2 and args.interpolation_type is not None:
        constructor_arguments["interpolation_type"] = args.interpolation_type
    if args.method == METHOD_MODEL_3 and args.block_len is not None:
        constructor_arguments["block_len"] = args.block_len

    run_benchmark(
        cfg,
        result_name=f"{estimator_class.__name__}{args.estimator_suffix}",
        constructor_arguments=constructor_arguments,
        smoke=args.smoke,
        asset_root=asset_root,
        **asset_arguments,
    )


if __name__ == "__main__":
    main()
