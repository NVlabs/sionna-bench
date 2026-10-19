# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Runner for benchmarking the Coleri 2002 comb-pilot LS interpolation estimators.

Usage:
    python -m sionna.bench.mimo_ofdm_chest.papers.coleri2002.run_benchmark
    python -m sionna.bench.mimo_ofdm_chest.papers.coleri2002.run_benchmark --smoke
    python -m sionna.bench.mimo_ofdm_chest.papers.coleri2002.run_benchmark --method low-pass
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

from .estimator import (
    METHOD_LOWPASS,
    METHOD_NAMES,
    METHOD_SECOND_ORDER,
    METHOD_SPLINE,
    METHOD_TIME_DOMAIN,
    LSCombEstimator,
)


PAPER_SLUG = "coleri2002"
METHOD_LABELS = {
    METHOD_SECOND_ORDER: "second-order",
    METHOD_LOWPASS: "low-pass",
    METHOD_SPLINE: "cubic spline",
    METHOD_TIME_DOMAIN: "time-domain",
}


def run_benchmark(
    cfg: SystemConfig,
    *,
    result_name: str,
    constructor_arguments: Optional[dict[str, Any]] = None,
    smoke: bool = False,
) -> dict[str, np.ndarray]:
    """Evaluate and record one configured Coleri interpolation method."""
    constructor_arguments = dict(constructor_arguments or {})
    method_overrides = {}
    if "method" in constructor_arguments:
        method_overrides["method"] = constructor_arguments["method"]
    method = resolve_constructor_arguments(
        LSCombEstimator,
        ("method",),
        method_overrides,
    )["method"]
    argument_names = ["method", "err_var_scale"]
    if method == METHOD_LOWPASS:
        argument_names.extend(
            ("lowpass_filter_half_length", "lowpass_cutoff")
        )
    effective_arguments = resolve_constructor_arguments(
        LSCombEstimator,
        argument_names,
        constructor_arguments,
    )
    output_dir = Path(
        prepare_paper_result_dir(cfg, PAPER_SLUG, smoke=smoke)
    )
    output_path = output_dir / f"{result_name}.npz"

    result = evaluate_channel_estimator(
        LSCombEstimator,
        cfg,
        output_path,
        estimator_kwargs=constructor_arguments,
    )
    record_result(
        cfg,
        PAPER_SLUG,
        LSCombEstimator,
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
        "--method",
        choices=("all", *METHOD_NAMES),
        default="all",
    )
    estimator.add_argument(
        "--lowpass-filter-half-length",
        type=int,
        default=None,
    )
    estimator.add_argument(
        "--lowpass-cutoff",
        type=float,
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
    """Evaluate the methods selected through the command-line interface."""
    args = _parse_args()
    cfg = _system_config_from_args(args)
    methods = METHOD_NAMES if args.method == "all" else [args.method]

    for method in methods:
        constructor_arguments = {
            "method": method,
        }
        if args.err_var_scale is not None:
            constructor_arguments["err_var_scale"] = args.err_var_scale
        if method == METHOD_LOWPASS:
            if args.lowpass_filter_half_length is not None:
                constructor_arguments["lowpass_filter_half_length"] = (
                    args.lowpass_filter_half_length
                )
            if args.lowpass_cutoff is not None:
                constructor_arguments["lowpass_cutoff"] = args.lowpass_cutoff
        run_benchmark(
            cfg,
            result_name=f"{LSCombEstimator.__name__} ({METHOD_LABELS[method]})",
            constructor_arguments=constructor_arguments,
            smoke=args.smoke,
        )


if __name__ == "__main__":
    main()
