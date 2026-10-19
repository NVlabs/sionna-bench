# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for Edfors benchmark result metadata."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from sionna.bench.mimo_ofdm_chest.persistence import sha256_file
from sionna.bench.mimo_ofdm_chest.papers.edfors1996 import run_benchmark as edfors_runner
from sionna.bench.mimo_ofdm_chest.papers.edfors1996.estimator import (
    METHOD_DIAGONAL_LMMSE,
    METHOD_LMMSE,
    METHOD_LS,
)
from test.mimo_ofdm_chest.system_variants import system_variant_config

from .test_asset import (
    _COV_BATCH_SIZE,
    _COV_NUM_ITERS,
    _SEED,
    _valid_arrays,
    _write_covariance_asset,
)


def test_all_methods_record_arguments_and_the_shared_covariance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """All methods record their settings and exact shared covariance asset."""
    cfg = replace(
        system_variant_config("baseline"),
        output_dir=str(tmp_path / "outputs"),
    )
    resource_grid = edfors_runner.build_resource_grid(cfg)
    covariance_dir = _write_covariance_asset(
        resource_grid,
        cfg,
        tmp_path / "assets",
        _valid_arrays(cfg.fft_size),
    )
    manifest_path = covariance_dir / "manifest.json"

    def evaluate_stub(_estimator_class, _cfg, output_path, **_kwargs):
        result = {"nmse": np.array([0.1], dtype=np.float64)}
        np.savez(output_path, **result)
        return result

    monkeypatch.setattr(
        edfors_runner,
        "evaluate_channel_estimator",
        evaluate_stub,
    )

    common_arguments = {
        "num_taps": 4,
        "interpolation_type": "lin_time_avg",
        "chest_err_const": 0.25,
    }
    methods = (METHOD_LMMSE, METHOD_DIAGONAL_LMMSE, METHOD_LS)
    for method in methods:
        edfors_runner.run_benchmark(
            cfg,
            result_name=method,
            constructor_arguments={"method": method, **common_arguments},
            asset_root=tmp_path / "assets",
            cov_num_iters=_COV_NUM_ITERS,
            cov_batch_size=_COV_BATCH_SIZE,
            covariance_seed=_SEED,
        )

    results_path = (
        Path(cfg.output_dir) / "edfors1996" / cfg.scenario_id / "results.json"
    )
    with results_path.open(encoding="utf-8") as file:
        entries = json.load(file)["results"]

    expected_asset = [
        {
            "asset_type": "delay-covariance",
            "artifact_id": covariance_dir.name,
            "manifest_sha256": sha256_file(manifest_path),
        }
    ]
    for method in methods:
        assert entries[method]["constructor_arguments"] == {
            "method": method,
            **common_arguments,
        }
        assert entries[method]["assets"] == expected_asset
