# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Runner for benchmarking the Wei 2017 comb-specialized SBL estimator.

Usage:
    python -m sionna.bench.mimo_ofdm_chest.papers.wei2017.run_benchmark
    python -m sionna.bench.mimo_ofdm_chest.papers.wei2017.run_benchmark --smoke
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Any, Optional

import numpy as np

from sionna.bench.mimo_ofdm_chest.benchmarking import evaluate_channel_estimator
from sionna.bench.mimo_ofdm_chest.benchmarking.result_store import (
    prepare_paper_result_dir,
    record_result,
    resolve_constructor_arguments,
)
from sionna.bench.mimo_ofdm_chest.configs import SystemConfig, smoke_config, umi_3gpp_config

from .estimator import SBLChannelEstimator


PAPER_SLUG = "wei2017"


def run_benchmark(
    cfg: SystemConfig,
    *,
    result_name: str,
    constructor_arguments: Optional[dict[str, Any]] = None,
    smoke: bool = False,
) -> dict[str, np.ndarray]:
    """Evaluate and record one configured SBL estimator."""
    constructor_arguments = dict(constructor_arguments or {})
    effective_arguments = resolve_constructor_arguments(
        SBLChannelEstimator,
        (
            "interpolation_type",
            "num_taps",
            "num_iter",
            "alpha_init",
            "posterior_variance_scale",
        ),
        constructor_arguments,
    )
    output_dir = Path(
        prepare_paper_result_dir(cfg, PAPER_SLUG, smoke=smoke)
    )
    output_path = output_dir / f"{result_name}.npz"

    result = evaluate_channel_estimator(
        SBLChannelEstimator,
        cfg,
        output_path,
        estimator_kwargs=constructor_arguments,
    )
    record_result(
        cfg,
        PAPER_SLUG,
        SBLChannelEstimator,
        result_name=result_name,
        constructor_arguments=effective_arguments,
        smoke=smoke,
    )

    print(f"[{PAPER_SLUG}] result saved to {output_path}")
    return result


def _parse_args() -> argparse.Namespace:
    """Parse benchmark and estimator arguments."""
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

    estimator = parser.add_argument_group("estimator configuration")
    estimator.add_argument(
        "--interpolation-type",
        choices=("nn", "lin", "lin_time_avg"),
        default=None,
    )
    estimator.add_argument("--num-taps", type=int, default=None)
    estimator.add_argument("--num-iter", type=int, default=None)
    estimator.add_argument("--alpha-init", type=float, default=None)
    estimator.add_argument(
        "--posterior-variance-scale",
        type=float,
        default=None,
    )

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
    """Evaluate the configured SBL estimator from the command line."""
    args = _parse_args()
    cfg = _system_config_from_args(args)
    constructor_arguments = {
        name: value
        for name, value in {
            "interpolation_type": args.interpolation_type,
            "num_taps": args.num_taps,
            "num_iter": args.num_iter,
            "alpha_init": args.alpha_init,
            "posterior_variance_scale": args.posterior_variance_scale,
        }.items()
        if value is not None
    }
    run_benchmark(
        cfg,
        result_name=SBLChannelEstimator.__name__,
        constructor_arguments=constructor_arguments,
        smoke=args.smoke,
    )


if __name__ == "__main__":
    main()
