# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Runner for benchmarking [paper and estimator name].

Usage:
    [show how to run the benchmark from the terminal]
"""

# Whatever imports are needed for the benchmark
import argparse
import numpy as np
from typing import Any, Optional
from pathlib import Path
from dataclasses import replace

from sionna_bench.mimo_ofdm_chest.configs import SystemConfig, smoke_config, umi_3gpp_config
from sionna_bench.mimo_ofdm_chest.benchmarking import build_resource_grid, evaluate_channel_estimator
from sionna_bench.mimo_ofdm_chest.benchmarking.result_store import (
    record_result,
    resolve_constructor_arguments,
)
#import ...

# if applicable to the paper
def prepare_assets(...):
    # Asset preparation
    ...


def run_benchmark(
    cfg: SystemConfig,
    *,
    result_name: str,
    constructor_arguments: Optional[dict[str, Any]] = None,
    smoke: bool = False,
    asset_root: Optional[Path] = None, # if applicable to the paper
) -> dict[str, np.ndarray]:
    # Strict loading (if applicable) and evaluation only.

    resource_grid = build_resource_grid(cfg)
    constructor_arguments = dict(constructor_arguments or {})

    # prepared_arguments, manifest_paths = load_model(...) if applicable

    # prepare output path and estimator arguments
    # output_path = ...
    # constructor_arguments = ...

    # evaluate the channel estimator
    result = evaluate_channel_estimator(
        # ChannelEstimator,
        cfg,
        #output_path,
        # estimator_kwargs={
        #     **constructor_arguments, # Explicit estimator overrides only.
        #     **prepared_arguments, # if applicable
        # },
        )

    # effective_arguments = resolve_constructor_arguments(
    #     ChannelEstimator,
    #     ("behavior_relevant_argument",),
    #     constructor_arguments,
    # )

    # record the result
    # result_name = ...
    record_result(
        cfg,
        # PAPER_SLUG,
        # ChannelEstimator,
        # result_name=result_name,
        #constructor_arguments=effective_arguments,
        #asset_manifests=manifest_paths, # if applicable
        #smoke=smoke,
    )





def _parse_args() -> argparse.Namespace:
    """Parse benchmark, asset, and estimator arguments."""
    parser = argparse.ArgumentParser()

    benchmark = parser.add_argument_group("benchmark configuration")
    benchmark.add_argument("--smoke", action="store_true")
    benchmark.add_argument("--output-dir", default=None)
    benchmark.add_argument("--device", default=None)
    benchmark.add_argument("--precision", choices=("single", "double"),default=None)
    benchmark.add_argument("--batch-size", type=int, default=None)
    benchmark.add_argument("--num-mc-batches", type=int, default=None)

    # this group only if applicable to the paper
    assets = parser.add_argument_group("asset handling")
    assets.add_argument("--asset-root", type=Path, default=None)
    assets.add_argument("--prepare-assets", action="store_true")
    assets.add_argument("--train-batches", type=int, default=None)
    assets.add_argument("--train-batch-size", type=int, default=None)
    assets.add_argument("--epochs", type=int, default=None)

    # this group only if applicable to the paper
    estimator = parser.add_argument_group("estimator configuration")
    # Optional estimator arguments use default=None. Omit values that remain
    # None so the estimator constructor supplies its own defaults.

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

def main():
    args = _parse_args()
    cfg = _system_config_from_args(args)

    if args.prepare_assets:
        prepare_assets(...)
    else:
        run_benchmark(...)


# entry point
if __name__ == "__main__":
    main()
