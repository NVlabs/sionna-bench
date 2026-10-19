# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Runner for the Lyu 2026 pilot-comb Toeplitz-P2 adaptation.

Usage:
    python -m sionna.bench.mimo_ofdm_chest.papers.lyu2026.run_benchmark --smoke
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Any, Optional

import numpy as np

from sionna.bench.mimo_ofdm_chest.configs import SystemConfig, smoke_config, umi_3gpp_config
from sionna.bench.mimo_ofdm_chest.benchmarking import evaluate_channel_estimator
from sionna.bench.mimo_ofdm_chest.benchmarking.result_store import (
    prepare_paper_result_dir,
    record_result,
    resolve_constructor_arguments,
)

from .estimator import SparseDiffuseChannelEstimator


PAPER_SLUG = "lyu2026"


def run_benchmark(
    cfg: SystemConfig,
    *,
    result_name: str,
    constructor_arguments: Optional[dict[str, Any]] = None,
    smoke: bool = False,
) -> dict[str, np.ndarray]:
    """Evaluate and record one configured sparse/diffuse estimator."""
    constructor_arguments = dict(constructor_arguments or {})
    effective_arguments = resolve_constructor_arguments(
        SparseDiffuseChannelEstimator,
        (
            "diffuse_num_taps",
            "tau",
            "tau_noise_scale",
            "tau_min",
            "tau_max",
            "diffuse_energy",
            "err_var_scale",
            "admm_rho",
            "admm_num_iters",
            "interpolation_type",
        ),
        constructor_arguments,
    )
    output_dir = Path(
        prepare_paper_result_dir(cfg, PAPER_SLUG, smoke=smoke)
    )
    output_path = output_dir / f"{result_name}.npz"

    result = evaluate_channel_estimator(
        SparseDiffuseChannelEstimator,
        cfg,
        output_path,
        estimator_kwargs=constructor_arguments,
    )
    record_result(
        cfg,
        PAPER_SLUG,
        SparseDiffuseChannelEstimator,
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
        "--diffuse-num-taps",
        type=int,
        default=None,
    )
    estimator.add_argument("--tau", type=float, default=None)
    estimator.add_argument(
        "--tau-noise-scale",
        type=float,
        default=None,
    )
    estimator.add_argument("--tau-min", type=float, default=None)
    estimator.add_argument("--tau-max", type=float, default=None)
    estimator.add_argument(
        "--diffuse-energy",
        type=float,
        default=None,
    )
    estimator.add_argument(
        "--err-var-scale",
        type=float,
        default=None,
    )
    estimator.add_argument("--admm-rho", type=float, default=None)
    estimator.add_argument(
        "--admm-num-iters",
        type=int,
        default=None,
    )
    estimator.add_argument(
        "--interpolation-type",
        choices=("nn", "lin", "lin_time_avg"),
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
    """Run the configured benchmark."""
    args = _parse_args()

    cfg = _system_config_from_args(args)
    constructor_arguments = {
        name: value
        for name, value in {
            "diffuse_num_taps": args.diffuse_num_taps,
            "tau": args.tau,
            "tau_noise_scale": args.tau_noise_scale,
            "tau_min": args.tau_min,
            "tau_max": args.tau_max,
            "diffuse_energy": args.diffuse_energy,
            "err_var_scale": args.err_var_scale,
            "admm_rho": args.admm_rho,
            "admm_num_iters": args.admm_num_iters,
            "interpolation_type": args.interpolation_type,
        }.items()
        if value is not None
    }

    run_benchmark(
        cfg,
        result_name=SparseDiffuseChannelEstimator.__name__,
        constructor_arguments=constructor_arguments,
        smoke=args.smoke,
    )


if __name__ == "__main__":
    main()
