# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Driver for Sionna built-in channel-estimation baselines.

Usage:
    python -m sionna.bench.mimo_ofdm_chest.papers.builtin.run_benchmark --estimator ls-lin
    python -m sionna.bench.mimo_ofdm_chest.papers.builtin.run_benchmark --smoke \
        --estimator ls-lin --device cpu
    python -m sionna.bench.mimo_ofdm_chest.papers.builtin.run_benchmark \
        --estimator ls-nn,lmmse --asset-root assets/generated
    python -m sionna.bench.mimo_ofdm_chest.papers.builtin.run_benchmark --prepare-assets \
        --estimator lmmse --asset-root assets/generated
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Dict, Optional

from sionna.bench.mimo_ofdm_chest.configs import SystemConfig, smoke_config, umi_3gpp_config
from sionna.bench.mimo_ofdm_chest.benchmarking import (
    OFDMChannelEstimationSystem,
    build_resource_grid,
    evaluate_channel_estimation_system,
    evaluate_channel_estimator,
)
from sionna.bench.mimo_ofdm_chest.benchmarking.result_store import (
    prepare_paper_result_dir,
    record_result,
    resolve_constructor_arguments,
)

from .asset import (
    estimate_covariance,
    load_covariance,
)
from .estimator import (
    LMMSEEstimatorWrapper,
    LSEstimatorWrapper,
)


PAPER_SLUG = "builtin"
COV_NUM_ITERS = 1000
COV_BATCH_SIZE = 64
SMOKE_COV_NUM_ITERS = 5
SMOKE_COV_BATCH_SIZE = 8
BUILTIN_ESTIMATOR_NAMES = {
    "ls-nn": "LS (nearest neighbor)",
    "ls-lin": "LS (linear)",
    "lmmse": "LMMSE",
    "perfect": "perfect",
}


def _estimator_config(
    name: str,
    cfg: SystemConfig,
    lmmse_cov,
    lmmse_order: Optional[str],
) -> tuple[type, dict]:
    if name == "ls-nn":
        return LSEstimatorWrapper, {
            "interpolation_type": "nn",
            "precision": cfg.precision,
        }
    if name == "ls-lin":
        return LSEstimatorWrapper, {
            "interpolation_type": "lin",
            "precision": cfg.precision,
        }
    if name == "lmmse":
        if lmmse_cov is None:
            raise ValueError("lmmse estimator requires covariance matrices")
        freq_cov, time_cov, space_cov = lmmse_cov
        kwargs = {
            "freq_cov": freq_cov,
            "time_cov": time_cov,
            "space_cov": space_cov,
            "batch_chunk_size": 1,
            "precision": cfg.precision,
        }
        if lmmse_order is not None:
            kwargs["order"] = lmmse_order
        return LMMSEEstimatorWrapper, kwargs
    raise ValueError(f"unknown built-in estimator: {name}")


def _evaluate_perfect_csi(
    cfg: SystemConfig,
    output_path: Path,
    *,
    verbose: bool = True,
) -> Dict:
    resource_grid = build_resource_grid(cfg)
    system = OFDMChannelEstimationSystem(cfg, None, resource_grid=resource_grid)
    return evaluate_channel_estimation_system(
        system,
        output_path,
        verbose=verbose,
    )


def prepare_assets(
    cfg: SystemConfig,
    *,
    asset_root: str | Path,
    cov_num_iters: int = COV_NUM_ITERS,
    cov_batch_size: int = COV_BATCH_SIZE,
    overwrite_assets: bool = False,
) -> Path:
    """Generate the built-in LMMSE covariance asset and return its manifest."""
    _, manifest_path = estimate_covariance(
        cfg,
        num_it=cov_num_iters,
        batch_size=cov_batch_size,
        seed=cfg.seed,
        asset_root=asset_root,
        overwrite=overwrite_assets,
    )
    return manifest_path


def run_benchmark(
    cfg: SystemConfig,
    estimator: str,
    *,
    smoke: bool = False,
    verbose: bool = True,
    asset_root: str | Path | None = None,
    lmmse_order: Optional[str] = None,
) -> Dict:
    """Evaluate and record one built-in channel-estimation method."""
    if estimator not in BUILTIN_ESTIMATOR_NAMES:
        raise ValueError(f"unknown built-in estimator: {estimator}")

    lmmse_cov = None
    covariance_manifest = None
    if estimator == "lmmse":
        print("=== Covariance matrices for LMMSE ===")
        lmmse_cov, covariance_manifest = load_covariance(
            cfg,
            asset_root=asset_root,
        )

    out_dir = Path(
        prepare_paper_result_dir(cfg, PAPER_SLUG, smoke=smoke)
    )
    result_name = BUILTIN_ESTIMATOR_NAMES[estimator]
    output_path = out_dir / f"{result_name}.npz"
    if estimator == "perfect":
        result = _evaluate_perfect_csi(cfg, output_path, verbose=verbose)
        estimator_class = None
        constructor_arguments = {}
    else:
        estimator_class, estimator_kwargs = _estimator_config(
            estimator,
            cfg,
            lmmse_cov,
            lmmse_order,
        )
        result = evaluate_channel_estimator(
            estimator_class,
            cfg,
            output_path,
            estimator_kwargs=estimator_kwargs,
            verbose=verbose,
        )
        constructor_overrides = {
            key: value
            for key, value in estimator_kwargs.items()
            if key not in {
                "freq_cov",
                "time_cov",
                "space_cov",
                "precision",
            }
        }
        argument_names = (
            ("interpolation_type",)
            if estimator.startswith("ls-")
            else ("order", "batch_chunk_size")
        )
        constructor_arguments = resolve_constructor_arguments(
            estimator_class,
            argument_names,
            constructor_overrides,
        )

    asset_manifests = ()
    if covariance_manifest is not None:
        asset_manifests = (covariance_manifest,)
    record_result(
        cfg,
        PAPER_SLUG,
        estimator_class,
        result_name=result_name,
        constructor_arguments=constructor_arguments,
        asset_manifests=asset_manifests,
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

    estimator = parser.add_argument_group("estimator configuration")
    estimator.add_argument(
        "--estimator",
        default="all",
        help=(
            f"Comma-separated subset of {tuple(BUILTIN_ESTIMATOR_NAMES)}, "
            "or 'all'."
        ),
    )
    estimator.add_argument("--lmmse-order", default=None)

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
    """Prepare covariance or evaluate the selected built-in methods."""
    args = _parse_args()
    cfg = _system_config_from_args(args)

    estimators = (
        list(BUILTIN_ESTIMATOR_NAMES)
        if args.estimator == "all"
        else [name.strip() for name in args.estimator.split(",")]
    )
    unknown_estimators = sorted(
        set(estimators).difference(BUILTIN_ESTIMATOR_NAMES)
    )
    if unknown_estimators:
        raise ValueError(f"unknown built-in estimators: {unknown_estimators}")

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

    selected_asset_root = args.asset_root
    if args.prepare_assets:
        if "lmmse" not in estimators:
            raise ValueError("Only the LMMSE estimator has a persistent asset.")
        if selected_asset_root is None:
            raise ValueError("`--prepare-assets` requires `--asset-root`.")
        prepare_assets(
            cfg,
            asset_root=selected_asset_root,
            cov_num_iters=cov_num_iters,
            cov_batch_size=cov_batch_size,
            overwrite_assets=args.overwrite_assets,
        )
        return

    if args.smoke and "lmmse" in estimators:
        selected_asset_root = selected_asset_root or Path("assets/smoke")

    for estimator in estimators:
        run_benchmark(
            cfg,
            estimator,
            smoke=args.smoke,
            asset_root=selected_asset_root,
            lmmse_order=args.lmmse_order,
        )


if __name__ == "__main__":
    main()
