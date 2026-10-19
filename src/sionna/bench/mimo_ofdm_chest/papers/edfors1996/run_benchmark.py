# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Driver for the Edfors 1996 DFT-domain channel estimators.

Usage:
    python -m sionna.bench.mimo_ofdm_chest.papers.edfors1996.run_benchmark \
        --asset-root assets/generated
    python -m sionna.bench.mimo_ofdm_chest.papers.edfors1996.run_benchmark --smoke \
        --asset-root assets/smoke
    python -m sionna.bench.mimo_ofdm_chest.papers.edfors1996.run_benchmark \
        --method "diagonal LMMSE" --asset-root assets/generated
    python -m sionna.bench.mimo_ofdm_chest.papers.edfors1996.run_benchmark --prepare-assets \
        --asset-root assets/generated
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
    resolve_constructor_arguments,
)
from sionna.bench.mimo_ofdm_chest.configs import SystemConfig, smoke_config, umi_3gpp_config

from .asset import estimate_covariance, load_covariance
from .estimator import (
    METHOD_DIAGONAL_LMMSE,
    METHOD_LMMSE,
    METHOD_LS,
    METHOD_NAMES,
    TruncatedChannelEstimator,
)


PAPER_SLUG = "edfors1996"
COV_NUM_ITERS = 1000
COV_BATCH_SIZE = 64
SMOKE_COV_NUM_ITERS = 5
SMOKE_COV_BATCH_SIZE = 8
METHOD_LABELS = {
    METHOD_LMMSE: "LMMSE",
    METHOD_DIAGONAL_LMMSE: "diag. LMMSE",
    METHOD_LS: "LS",
}


def prepare_assets(
    cfg: SystemConfig,
    *,
    asset_root: Union[str, Path],
    cov_num_iters: int = COV_NUM_ITERS,
    cov_batch_size: int = COV_BATCH_SIZE,
    seed: Optional[int] = None,
    overwrite_assets: bool = False,
) -> Path:
    """Estimate the Edfors covariance asset and return its manifest path."""
    resource_grid = build_resource_grid(cfg)
    _, manifest_path = estimate_covariance(
        resource_grid,
        cfg,
        cov_num_iters=cov_num_iters,
        cov_batch_size=cov_batch_size,
        seed=cfg.seed if seed is None else seed,
        asset_root=asset_root,
        overwrite=overwrite_assets,
    )
    return manifest_path


def run_benchmark(
    cfg: SystemConfig,
    *,
    result_name: str,
    constructor_arguments: Optional[dict[str, Any]] = None,
    smoke: bool = False,
    asset_root: Optional[Union[str, Path]] = None,
    cov_num_iters: int = COV_NUM_ITERS,
    cov_batch_size: int = COV_BATCH_SIZE,
    covariance_seed: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """Strictly load covariance, evaluate one method, and record its result."""
    constructor_arguments = dict(constructor_arguments or {})
    effective_arguments = resolve_constructor_arguments(
        TruncatedChannelEstimator,
        ("method", "num_taps", "interpolation_type", "chest_err_const"),
        constructor_arguments,
    )
    method = effective_arguments["method"]
    if method not in METHOD_NAMES:
        raise ValueError(f"Unknown Edfors method: {method!r}.")

    resource_grid = build_resource_grid(cfg)
    time_covariance, manifest_path = load_covariance(
        resource_grid,
        cfg,
        cov_num_iters=cov_num_iters,
        cov_batch_size=cov_batch_size,
        seed=cfg.seed if covariance_seed is None else covariance_seed,
        asset_root=asset_root,
    )
    output_dir = Path(
        prepare_paper_result_dir(cfg, PAPER_SLUG, smoke=smoke)
    )
    output_path = output_dir / f"{result_name}.npz"
    result = evaluate_channel_estimator(
        TruncatedChannelEstimator,
        cfg,
        output_path,
        estimator_kwargs={
            **constructor_arguments,
            "time_covariance": time_covariance,
        },
    )
    record_result(
        cfg,
        PAPER_SLUG,
        TruncatedChannelEstimator,
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
    assets.add_argument("--cov-num-iters", type=int, default=None)
    assets.add_argument("--cov-batch-size", type=int, default=None)
    assets.add_argument("--covariance-seed", type=int, default=None)

    estimator = parser.add_argument_group("estimator configuration")
    estimator.add_argument(
        "--method",
        choices=("all", *METHOD_NAMES),
        default="all",
    )
    estimator.add_argument(
        "--interpolation-type",
        choices=("nn", "lin", "lin_time_avg"),
        default=None,
    )
    estimator.add_argument("--num-taps", type=int, default=None)
    estimator.add_argument(
        "--chest-err-const",
        type=float,
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
    """Prepare covariance or evaluate the selected Edfors methods."""
    args = _parse_args()
    cfg = _system_config_from_args(args)
    default_cov_num_iters = SMOKE_COV_NUM_ITERS if args.smoke else COV_NUM_ITERS
    default_cov_batch_size = (
        SMOKE_COV_BATCH_SIZE if args.smoke else COV_BATCH_SIZE
    )
    cov_num_iters = (
        default_cov_num_iters
        if args.cov_num_iters is None
        else args.cov_num_iters
    )
    cov_batch_size = (
        default_cov_batch_size
        if args.cov_batch_size is None
        else args.cov_batch_size
    )

    if args.prepare_assets:
        if args.asset_root is None:
            raise ValueError("`--prepare-assets` requires `--asset-root`.")
        prepare_assets(
            cfg,
            asset_root=args.asset_root,
            cov_num_iters=cov_num_iters,
            cov_batch_size=cov_batch_size,
            seed=args.covariance_seed,
            overwrite_assets=args.overwrite_assets,
        )
        return

    asset_root = args.asset_root
    if args.smoke:
        asset_root = asset_root or Path("assets/smoke")
    methods = METHOD_NAMES if args.method == "all" else [args.method]
    for method in methods:
        result_name = (
            f"{TruncatedChannelEstimator.__name__} "
            f"({METHOD_LABELS[method]}){args.estimator_suffix}"
        )
        constructor_arguments = {
            "method": method,
        }
        constructor_arguments.update(
            {
                name: value
                for name, value in {
                    "num_taps": args.num_taps,
                    "interpolation_type": args.interpolation_type,
                    "chest_err_const": args.chest_err_const,
                }.items()
                if value is not None
            }
        )
        run_benchmark(
            cfg,
            result_name=result_name,
            constructor_arguments=constructor_arguments,
            smoke=args.smoke,
            asset_root=asset_root,
            cov_num_iters=cov_num_iters,
            cov_batch_size=cov_batch_size,
            covariance_seed=args.covariance_seed,
        )


if __name__ == "__main__":
    main()
