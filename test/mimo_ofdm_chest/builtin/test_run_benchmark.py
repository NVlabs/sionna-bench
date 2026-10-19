# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for built-in benchmark result metadata."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from sionna.bench.mimo_ofdm_chest.persistence import sha256_file
from sionna.bench.mimo_ofdm_chest.papers.builtin import run_benchmark as builtin_runner
from test.mimo_ofdm_chest.system_variants import system_variant_config

from .test_asset import _valid_covariances, _write_covariance_asset


def test_builtin_results_reference_only_the_selected_lmmse_asset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """LMMSE records its exact manifest while LS records no persistent asset."""
    cfg = replace(
        system_variant_config("baseline"),
        output_dir=str(tmp_path / "outputs"),
    )
    asset_root = tmp_path / "assets"
    covariance_dir = _write_covariance_asset(
        cfg,
        asset_root,
        _valid_covariances(cfg),
    )
    manifest_path = covariance_dir / "manifest.json"

    def evaluate_stub(_estimator_class, _cfg, output_path, **_kwargs):
        result = {"nmse": np.array([0.1], dtype=np.float64)}
        np.savez(output_path, **result)
        return result

    monkeypatch.setattr(
        builtin_runner,
        "evaluate_channel_estimator",
        evaluate_stub,
    )

    builtin_runner.run_benchmark(cfg, "lmmse", asset_root=asset_root)
    builtin_runner.run_benchmark(cfg, "ls-lin")

    results_path = (
        Path(cfg.output_dir) / "builtin" / cfg.scenario_id / "results.json"
    )
    with results_path.open(encoding="utf-8") as file:
        entries = json.load(file)["results"]
    assert entries["LMMSE"]["assets"] == [
        {
            "asset_type": "lmmse-covariance",
            "artifact_id": "default",
            "manifest_sha256": sha256_file(manifest_path),
        }
    ]
    assert entries["LS (linear)"]["assets"] == []
