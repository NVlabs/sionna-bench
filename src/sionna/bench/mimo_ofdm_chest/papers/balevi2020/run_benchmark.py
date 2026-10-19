# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Driver for the Balevi 2020 untrained deep-prior estimator.

Usage:
    python -m sionna.bench.mimo_ofdm_chest.papers.balevi2020.run_benchmark --smoke
    python -m sionna.bench.mimo_ofdm_chest.papers.balevi2020.run_benchmark --num-steps 250
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

from .estimator import DeepPriorChannelEstimator


PAPER_SLUG = "balevi2020"


def run_benchmark(
    cfg: SystemConfig,
    *,
    result_name: str,
    constructor_arguments: Optional[dict[str, Any]] = None,
    smoke: bool = False,
) -> dict[str, np.ndarray]:
    """Evaluate and record one configured deep-prior estimator."""
    constructor_arguments = dict(constructor_arguments or {})
    effective_arguments = resolve_constructor_arguments(
        DeepPriorChannelEstimator,
        (
            "hidden_channels",
            "num_hidden_layers",
            "num_steps",
            "learning_rate",
            "input_noise_scale",
            "seed",
            "interpolation_type",
            "err_var_scale",
        ),
        constructor_arguments,
    )
    output_dir = Path(
        prepare_paper_result_dir(cfg, PAPER_SLUG, smoke=smoke)
    )
    output_path = output_dir / f"{result_name}.npz"

    result = evaluate_channel_estimator(
        DeepPriorChannelEstimator,
        cfg,
        output_path,
        estimator_kwargs=constructor_arguments,
    )
    record_result(
        cfg,
        PAPER_SLUG,
        DeepPriorChannelEstimator,
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
        "--hidden-channels",
        type=int,
        default=None,
    )
    estimator.add_argument(
        "--num-hidden-layers",
        type=int,
        default=None,
    )
    estimator.add_argument("--num-steps", type=int, default=None)
    estimator.add_argument(
        "--learning-rate",
        type=float,
        default=None,
    )
    estimator.add_argument(
        "--input-noise-scale",
        type=float,
        default=None,
    )
    estimator.add_argument(
        "--interpolation-type",
        choices=("nn", "lin", "lin_time_avg"),
        default=None,
    )
    estimator.add_argument(
        "--err-var-scale",
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
    """Evaluate the configured deep-prior estimator from the command line."""
    args = _parse_args()
    cfg = _system_config_from_args(args)
    constructor_arguments = {
        "seed": cfg.seed,
    }
    constructor_arguments.update(
        {
            name: value
            for name, value in {
                "hidden_channels": args.hidden_channels,
                "num_hidden_layers": args.num_hidden_layers,
                "num_steps": args.num_steps,
                "learning_rate": args.learning_rate,
                "input_noise_scale": args.input_noise_scale,
                "interpolation_type": args.interpolation_type,
                "err_var_scale": args.err_var_scale,
            }.items()
            if value is not None
        }
    )

    run_benchmark(
        cfg,
        result_name=DeepPriorChannelEstimator.__name__,
        constructor_arguments=constructor_arguments,
        smoke=args.smoke,
    )


if __name__ == "__main__":
    main()
