# SPDX-FileCopyrightText: Copyright (c) 2026 Benedikt Fesl
# SPDX-License-Identifier: Apache-2.0
"""Strict evaluation and separate explicit preparation for the Fesl prior."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Any, Optional

import torch

from sionna.bench.mimo_ofdm_chest.benchmarking import (
    build_resource_grid,
    evaluate_channel_estimator,
)
from sionna.bench.mimo_ofdm_chest.benchmarking.result_store import (
    prepare_paper_result_dir,
    record_result,
    resolve_constructor_arguments,
)
from sionna.bench.mimo_ofdm_chest.configs import (
    SystemConfig,
    smoke_config,
    umi_3gpp_config,
)

from .asset import generate_training_data, load_model, load_training_data, train_model
from .estimator import DiffusionChannelEstimator


def prepare_assets(
    cfg: SystemConfig,
    *,
    asset_root: str | Path,
    smoke: bool = False,
    artifact_id: Optional[str] = None,
    training_data_manifest: Optional[str | Path] = None,
    overwrite: bool = False,
    training_device: Optional[str] = None,
    checkpoint_dir: Optional[str | Path] = None,
    resume: bool = False,
) -> Path:
    """Prepare externally and return the validated model manifest; do not evaluate."""
    resource_grid = build_resource_grid(cfg)
    selected_id = artifact_id or ("smoke" if smoke else "canonical")
    if training_data_manifest is None:
        settings = (
            {
                "num_train_channels": 2,
                "num_validation_channels": 2,
                "generation_batch_size": 1,
                "train_seed": 24100000,
                "validation_seed": 24110000,
                "purpose": "smoke",
            }
            if smoke
            else {}
        )
        training_data_manifest = generate_training_data(
            resource_grid,
            cfg,
            asset_root=asset_root,
            artifact_id=f"{selected_id}-data",
            overwrite=overwrite,
            **settings,
        )
    _, _, dataset = load_training_data(
        resource_grid, cfg, manifest_path=training_data_manifest
    )
    expected_purpose = "smoke" if smoke else "canonical"
    if dataset["generation_recipe"].get("purpose") != expected_purpose:
        raise ValueError(
            f"Preparation requires {expected_purpose} data; refusing a mismatched dataset before training."
        )
    training_settings = (
        {
            "epochs": 1,
            "patience_epochs": 0,
            "initialization_seed": 24120000,
            "noise_seed": 24130000,
            "validation_noise_seed": 24140000,
        }
        if smoke
        else {}
    )
    _, _, manifest = train_model(
        resource_grid,
        cfg,
        training_data_manifest=training_data_manifest,
        asset_root=asset_root,
        artifact_id=selected_id,
        overwrite=overwrite,
        training_device=training_device,
        checkpoint_dir=checkpoint_dir,
        resume=resume,
        **training_settings,
    )
    return manifest


def run_benchmark(
    cfg: SystemConfig,
    *,
    asset_root: Optional[str | Path] = None,
    artifact_id: Optional[str] = None,
    smoke: bool = False,
    interpolation_type: Optional[str] = None,
    err_var_scale: Optional[float] = None,
) -> dict[str, Any]:
    """Strictly load a matching prior and evaluate through the shared pipeline.

    Smoke evaluation loads smoke-purpose assets; otherwise canonical-purpose
    assets are required. The variance multiplier defaults to the estimator
    constructor, the single source of truth.
    """
    resource_grid = build_resource_grid(cfg)
    state, power, manifest = load_model(
        resource_grid,
        cfg,
        asset_root=asset_root,
        artifact_id=artifact_id or ("smoke" if smoke else "canonical"),
        expected_purpose="smoke" if smoke else "canonical",
    )
    arguments: dict[str, Any] = {
        "receive_array_shape": (cfg.bs_num_rows, cfg.bs_num_cols)
    }
    if interpolation_type is not None:
        arguments["interpolation_type"] = interpolation_type
    if err_var_scale is not None:
        arguments["err_var_scale"] = err_var_scale
    name = DiffusionChannelEstimator.__name__
    output = Path(prepare_paper_result_dir(cfg, "fesl2024", smoke=smoke))
    result = evaluate_channel_estimator(
        DiffusionChannelEstimator,
        cfg,
        output / f"{name}.npz",
        estimator_kwargs={"model_state": state, "channel_power": power, **arguments},
    )
    effective = resolve_constructor_arguments(
        DiffusionChannelEstimator,
        ("receive_array_shape", "interpolation_type", "err_var_scale"),
        arguments,
    )
    record_result(
        cfg,
        "fesl2024",
        DiffusionChannelEstimator,
        result_name=name,
        constructor_arguments=effective,
        asset_manifests=[manifest],
        smoke=smoke,
    )
    return {name: result}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-assets", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--asset-root", type=Path)
    parser.add_argument("--artifact-id")
    parser.add_argument("--training-data-manifest", type=Path)
    parser.add_argument("--overwrite-assets", action="store_true")
    parser.add_argument("--device")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--interpolation-type", choices=("lin_time_avg", "lin", "nn"))
    parser.add_argument("--err-var-scale", type=float)
    parser.add_argument("--torch-threads", type=int)
    parser.add_argument(
        "--training-device",
        help="Native PyTorch training device; Sionna --device stays separate.",
    )
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _system_config_from_args(args: argparse.Namespace) -> SystemConfig:
    cfg = smoke_config if args.smoke else umi_3gpp_config
    overrides = {}
    if args.device is not None:
        overrides["device"] = args.device
    if args.output_dir is not None:
        overrides["output_dir"] = str(args.output_dir)
    return replace(cfg, **overrides)


def main() -> None:
    args = _parse_args()
    if args.torch_threads is not None:
        torch.set_num_threads(args.torch_threads)
    cfg = _system_config_from_args(args)
    if args.prepare_assets:
        if args.asset_root is None:
            raise ValueError(
                "--prepare-assets requires an explicit external --asset-root."
            )
        print(
            prepare_assets(
                cfg,
                asset_root=args.asset_root,
                smoke=args.smoke,
                artifact_id=args.artifact_id,
                training_data_manifest=args.training_data_manifest,
                overwrite=args.overwrite_assets,
                training_device=args.training_device,
                checkpoint_dir=args.checkpoint_dir,
                resume=args.resume,
            )
        )
        return
    if (
        args.training_data_manifest is not None
        or args.overwrite_assets
        or args.training_device is not None
        or args.checkpoint_dir is not None
        or args.resume
    ):
        raise ValueError(
            "Training-data and overwrite options require --prepare-assets."
        )
    run_benchmark(
        cfg,
        asset_root=args.asset_root,
        artifact_id=args.artifact_id,
        smoke=args.smoke,
        interpolation_type=args.interpolation_type,
        err_var_scale=args.err_var_scale,
    )


if __name__ == "__main__":
    main()
