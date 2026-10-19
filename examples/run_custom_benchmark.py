# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run the custom-estimator example through the canonical benchmark."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Optional

from sionna.bench.mimo_ofdm_chest.benchmarking import evaluate_channel_estimator
from sionna.bench.mimo_ofdm_chest.benchmarking.result_store import (
    prepare_paper_result_dir,
    record_result,
)
from sionna.bench.mimo_ofdm_chest.configs import umi_3gpp_config

from .custom_estimator import ShrinkageLSChannelEstimator


def evaluate_example(
    *,
    output_dir: Path = Path("outputs/mimo_ofdm_chest"),
    device: Optional[str] = None,
) -> dict[str, object]:
    """Evaluate the example estimator and register its canonical result."""
    overrides = {"output_dir": str(output_dir)}
    if device is not None:
        overrides["device"] = device
    cfg = replace(umi_3gpp_config, **overrides)

    constructor_arguments = {"interpolation_type": "lin_time_avg"}
    result_name = ShrinkageLSChannelEstimator.__name__
    result_dir = Path(prepare_paper_result_dir(cfg, "custom"))
    result = evaluate_channel_estimator(
        ShrinkageLSChannelEstimator,
        cfg,
        result_dir / f"{result_name}.npz",
        estimator_kwargs=constructor_arguments,
    )
    record_result(
        cfg,
        "custom",
        ShrinkageLSChannelEstimator,
        result_name=result_name,
        constructor_arguments=constructor_arguments,
    )
    return result


def _parse_args() -> argparse.Namespace:
    """Parse example output and device overrides."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/mimo_ofdm_chest")
    )
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    """Run the example estimator through the canonical benchmark."""
    args = _parse_args()
    evaluate_example(output_dir=args.output_dir, device=args.device)


if __name__ == "__main__":
    main()
