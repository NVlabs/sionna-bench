# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Runner for the Koller 2022 GMM channel estimators.

Usage:
    python -m sionna.bench.mimo_ofdm_chest.papers.koller2022.run_benchmark \
        --asset-root assets/generated
    python -m sionna.bench.mimo_ofdm_chest.papers.koller2022.run_benchmark --smoke \
        --asset-root assets/smoke
    python -m sionna.bench.mimo_ofdm_chest.papers.koller2022.run_benchmark \
        --method spatial --asset-root assets/generated
    python -m sionna.bench.mimo_ofdm_chest.papers.koller2022.run_benchmark --prepare-assets \
        --method spectral --asset-root assets/generated
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

from .asset import (
    METHOD_NAMES,
    METHOD_SPATIAL,
    METHOD_SPECTRAL,
    fit_model,
    load_model,
)
from .estimator import (
    SpatialGMMChannelEstimator,
    SpectralGMMChannelEstimator,
)


PAPER_SLUG = "koller2022"
GMM_NUM_COMPONENTS = {
    METHOD_SPECTRAL: 15,
    METHOD_SPATIAL: 20,
}
GMM_TRAIN_BATCHES = 100
GMM_TRAIN_BATCH_SIZE = 64
GMM_EM_ITERS = 15
GMM_COVARIANCE_REG = 1e-3
GMM_MODEL_SEED = resolve_callable_arguments(fit_model, ("seed",))["seed"]
SMOKE_GMM_NUM_COMPONENTS = 2
SMOKE_GMM_TRAIN_BATCHES = {
    METHOD_SPECTRAL: 4,
    METHOD_SPATIAL: 8,
}
SMOKE_GMM_TRAIN_BATCH_SIZE = {
    METHOD_SPECTRAL: 8,
    METHOD_SPATIAL: 16,
}
SMOKE_GMM_EM_ITERS = 6


def prepare_assets(
    cfg: SystemConfig,
    *,
    method: str,
    asset_root: Union[str, Path],
    num_components: int,
    train_batches: int,
    train_batch_size: int,
    em_iters: int,
    covariance_reg: float = GMM_COVARIANCE_REG,
    seed: Optional[int] = None,
    overwrite_assets: bool = False,
) -> Path:
    """Fit one pooled GMM asset and return its manifest path."""
    resource_grid = build_resource_grid(cfg)
    _, manifest_path = fit_model(
        resource_grid,
        cfg,
        method=method,
        num_components=num_components,
        train_batches=train_batches,
        train_batch_size=train_batch_size,
        em_iters=em_iters,
        covariance_reg=covariance_reg,
        seed=GMM_MODEL_SEED if seed is None else seed,
        asset_root=asset_root,
        overwrite=overwrite_assets,
    )
    return manifest_path


def run_benchmark(
    cfg: SystemConfig,
    *,
    method: str,
    result_name: str,
    constructor_arguments: Optional[dict[str, Any]] = None,
    smoke: bool = False,
    asset_root: Optional[Union[str, Path]] = None,
    num_components: int,
    train_batches: int,
    train_batch_size: int,
    em_iters: int,
    covariance_reg: float = GMM_COVARIANCE_REG,
    model_seed: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """Strictly load one GMM, evaluate one method, and record its result."""
    if method == METHOD_SPECTRAL:
        estimator_class = SpectralGMMChannelEstimator
    elif method == METHOD_SPATIAL:
        estimator_class = SpatialGMMChannelEstimator
    else:
        raise ValueError(f"Unknown Koller method: {method!r}.")
    constructor_arguments = dict(constructor_arguments or {})
    argument_names = (
        ("time_avg",)
        if method == METHOD_SPECTRAL
        else ("interpolation_type",)
    )
    effective_arguments = resolve_constructor_arguments(
        estimator_class,
        argument_names,
        constructor_arguments,
    )
    resource_grid = build_resource_grid(cfg)
    parameters, manifest_path = load_model(
        resource_grid,
        cfg,
        method=method,
        num_components=num_components,
        train_batches=train_batches,
        train_batch_size=train_batch_size,
        em_iters=em_iters,
        covariance_reg=covariance_reg,
        seed=GMM_MODEL_SEED if model_seed is None else model_seed,
        asset_root=asset_root,
    )
    weights, means, covariances = parameters

    output_dir = Path(prepare_paper_result_dir(cfg, PAPER_SLUG, smoke=smoke))
    output_path = output_dir / f"{result_name}.npz"
    result = evaluate_channel_estimator(
        estimator_class,
        cfg,
        output_path,
        estimator_kwargs={
            **constructor_arguments,
            "weights": weights,
            "means": means,
            "covariances": covariances,
        },
    )
    record_result(
        cfg,
        PAPER_SLUG,
        estimator_class,
        result_name=result_name,
        constructor_arguments=effective_arguments,
        asset_manifests=[manifest_path],
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
    assets.add_argument("--gmm-components", type=int, default=None)
    assets.add_argument("--train-batches", type=int, default=None)
    assets.add_argument("--train-batch-size", type=int, default=None)
    assets.add_argument("--em-iters", type=int, default=None)
    assets.add_argument(
        "--covariance-reg",
        type=float,
        default=None,
    )
    assets.add_argument("--model-seed", type=int, default=None)

    estimator = parser.add_argument_group("estimator configuration")
    estimator.add_argument(
        "--method",
        choices=("all", *METHOD_NAMES),
        default="all",
    )
    estimator.add_argument(
        "--time-avg",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
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
    """Prepare GMM assets or evaluate the selected Koller methods."""
    args = _parse_args()
    cfg = _system_config_from_args(args)
    methods = METHOD_NAMES if args.method == "all" else [args.method]

    if args.prepare_assets and args.asset_root is None:
        raise ValueError("`--prepare-assets` requires `--asset-root`.")

    for method in methods:
        if args.smoke:
            default_num_components = SMOKE_GMM_NUM_COMPONENTS
            default_train_batches = SMOKE_GMM_TRAIN_BATCHES[method]
            default_train_batch_size = SMOKE_GMM_TRAIN_BATCH_SIZE[method]
            default_em_iters = SMOKE_GMM_EM_ITERS
        else:
            default_num_components = GMM_NUM_COMPONENTS[method]
            default_train_batches = GMM_TRAIN_BATCHES
            default_train_batch_size = GMM_TRAIN_BATCH_SIZE
            default_em_iters = GMM_EM_ITERS
        num_components = (
            default_num_components
            if args.gmm_components is None
            else args.gmm_components
        )
        train_batches = (
            default_train_batches
            if args.train_batches is None
            else args.train_batches
        )
        train_batch_size = (
            default_train_batch_size
            if args.train_batch_size is None
            else args.train_batch_size
        )
        em_iters = (
            default_em_iters if args.em_iters is None else args.em_iters
        )
        covariance_reg = (
            GMM_COVARIANCE_REG
            if args.covariance_reg is None
            else args.covariance_reg
        )

        if args.prepare_assets:
            prepare_assets(
                cfg,
                method=method,
                asset_root=args.asset_root,
                num_components=num_components,
                train_batches=train_batches,
                train_batch_size=train_batch_size,
                em_iters=em_iters,
                covariance_reg=covariance_reg,
                seed=args.model_seed,
                overwrite_assets=args.overwrite_assets,
            )
            continue

        asset_root = args.asset_root
        if args.smoke:
            asset_root = asset_root or Path("assets/smoke")
        if method == METHOD_SPECTRAL:
            estimator_class = SpectralGMMChannelEstimator
        else:
            estimator_class = SpatialGMMChannelEstimator
        result_name = f"{estimator_class.__name__}{args.estimator_suffix}"
        constructor_arguments = {}
        if method == METHOD_SPECTRAL and args.time_avg is not None:
            constructor_arguments["time_avg"] = args.time_avg
        if method == METHOD_SPATIAL and args.interpolation_type is not None:
            constructor_arguments["interpolation_type"] = (
                args.interpolation_type
            )
        run_benchmark(
            cfg,
            method=method,
            result_name=result_name,
            constructor_arguments=constructor_arguments,
            smoke=args.smoke,
            asset_root=asset_root,
            num_components=num_components,
            train_batches=train_batches,
            train_batch_size=train_batch_size,
            em_iters=em_iters,
            covariance_reg=covariance_reg,
            model_seed=args.model_seed,
        )


if __name__ == "__main__":
    main()
