# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for Koller benchmark result metadata."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from sionna.bench.mimo_ofdm_chest.persistence import sha256_file
from sionna.bench.mimo_ofdm_chest.papers.koller2022 import run_benchmark as koller_runner
from sionna.bench.mimo_ofdm_chest.papers.koller2022.asset import (
    METHOD_SPATIAL,
    METHOD_SPECTRAL,
)
from sionna.bench.mimo_ofdm_chest.papers.koller2022.estimator import (
    SpatialGMMChannelEstimator,
    SpectralGMMChannelEstimator,
)
from test.mimo_ofdm_chest.system_variants import system_variant_config

from .test_asset import _GENERATION_RECIPE, _write_gmm_asset


@pytest.mark.parametrize(
    ("method", "estimator_class", "constructor_arguments"),
    [
        (
            METHOD_SPECTRAL,
            SpectralGMMChannelEstimator,
            {"time_avg": True},
        ),
        (
            METHOD_SPATIAL,
            SpatialGMMChannelEstimator,
            {"interpolation_type": "lin_time_avg"},
        ),
    ],
)
def test_runner_records_arguments_and_selected_gmm_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    method: str,
    estimator_class: type,
    constructor_arguments: dict[str, object],
) -> None:
    """Results contain scalar settings and the exact selected GMM reference."""
    cfg = replace(
        system_variant_config("baseline"),
        output_dir=str(tmp_path / "outputs"),
    )
    resource_grid = koller_runner.build_resource_grid(cfg)
    asset_root = tmp_path / "assets"
    _write_gmm_asset(resource_grid, cfg, method, asset_root, monkeypatch)
    manifest_path = next(asset_root.rglob("manifest.json"))

    def evaluate_stub(_estimator_class, _cfg, output_path, **_kwargs):
        result = {"nmse": np.array([0.1], dtype=np.float64)}
        np.savez(output_path, **result)
        return result

    monkeypatch.setattr(
        koller_runner,
        "evaluate_channel_estimator",
        evaluate_stub,
    )
    result_name = estimator_class.__name__
    koller_runner.run_benchmark(
        cfg,
        method=method,
        result_name=result_name,
        constructor_arguments=constructor_arguments,
        asset_root=asset_root,
        num_components=_GENERATION_RECIPE["num_components"],
        train_batches=_GENERATION_RECIPE["train_batches"],
        train_batch_size=_GENERATION_RECIPE["train_batch_size"],
        em_iters=_GENERATION_RECIPE["em_iters"],
        covariance_reg=_GENERATION_RECIPE["covariance_reg"],
        model_seed=_GENERATION_RECIPE["seed"],
    )

    results_path = (
        Path(cfg.output_dir) / "koller2022" / cfg.scenario_id / "results.json"
    )
    with results_path.open(encoding="utf-8") as file:
        entry = json.load(file)["results"][result_name]

    assert entry["constructor_arguments"] == constructor_arguments
    assert entry["assets"] == [
        {
            "asset_type": f"{method}-gmm",
            "artifact_id": manifest_path.parent.name,
            "manifest_sha256": sha256_file(manifest_path),
        }
    ]
