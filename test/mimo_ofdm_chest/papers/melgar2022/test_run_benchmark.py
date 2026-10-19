# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for Melgar benchmark result metadata."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from sionna.bench.mimo_ofdm_chest.persistence import sha256_file
from sionna.bench.mimo_ofdm_chest.papers.melgar2022 import run_benchmark as melgar_runner
from sionna.bench.mimo_ofdm_chest.papers.melgar2022.estimator import (
    DNNChannelEstimator,
    METHOD_MODEL_2,
)
from test.mimo_ofdm_chest.system_variants import system_variant_config

from .test_asset import _TRAINING_SNR_DB, _train_tiny_model


@pytest.mark.training
def test_runner_records_arguments_and_ordered_model_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A result links JSON settings to the exact selected DNN manifest."""
    cfg = replace(
        system_variant_config(
            "baseline",
            snr_db_start=_TRAINING_SNR_DB,
            snr_db_stop=_TRAINING_SNR_DB,
        ),
        output_dir=str(tmp_path / "outputs"),
    )
    resource_grid = melgar_runner.build_resource_grid(cfg)
    asset_root = tmp_path / "assets"
    _, _, manifest_paths = _train_tiny_model(
        resource_grid,
        cfg,
        METHOD_MODEL_2,
        asset_root,
    )

    def evaluate_stub(_estimator_class, _cfg, output_path, **_kwargs):
        result = {"nmse": np.array([0.1], dtype=np.float64)}
        np.savez(output_path, **result)
        return result

    monkeypatch.setattr(
        melgar_runner,
        "evaluate_channel_estimator",
        evaluate_stub,
    )
    constructor_arguments = {
        "interpolation_type": "lin_time_avg",
        "err_var_scale": 1.0,
    }
    result_name = DNNChannelEstimator.__name__
    melgar_runner.run_benchmark(
        cfg,
        method=METHOD_MODEL_2,
        result_name=result_name,
        constructor_arguments=constructor_arguments,
        asset_root=asset_root,
        block_len=12,
        train_batches=1,
        train_batch_size=1,
        validation_batches=1,
        validation_batch_size=1,
        epochs=1,
        optimizer_batch_size=32,
        learning_rate=1e-4,
        min_train_seconds=0.0,
        patience_epochs=0,
        min_delta=0.0,
        max_examples_per_batch=16,
        snr_specific_weights=True,
        training_snr_dbs=[_TRAINING_SNR_DB],
        model_seed=cfg.seed,
    )

    results_path = (
        Path(cfg.output_dir) / "melgar2022" / cfg.scenario_id / "results.json"
    )
    with results_path.open(encoding="utf-8") as file:
        entry = json.load(file)["results"][result_name]

    assert entry["constructor_arguments"] == constructor_arguments
    assert entry["assets"] == [
        {
            "asset_type": "melgar-model2",
            "artifact_id": manifest_paths[0].parent.name,
            "manifest_sha256": sha256_file(manifest_paths[0]),
        }
    ]
