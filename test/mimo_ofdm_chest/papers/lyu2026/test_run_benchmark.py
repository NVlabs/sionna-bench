# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for Lyu benchmark result persistence."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from sionna.bench.mimo_ofdm_chest.configs import smoke_config
from sionna.bench.mimo_ofdm_chest.benchmarking.result_store import resolve_constructor_arguments
from sionna.bench.mimo_ofdm_chest.papers.lyu2026 import run_benchmark as runner


def test_smoke_result_records_complete_sparse_diffuse_configuration(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """The result records its NPZ and every behavioral constructor argument."""
    cfg = replace(smoke_config, output_dir=str(tmp_path), device="cpu")
    argument_names = (
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
    )
    arguments = resolve_constructor_arguments(
        runner.SparseDiffuseChannelEstimator,
        argument_names,
    )

    def evaluate_stub(_estimator_class, _cfg, output_path, **_kwargs):
        result = {"nmse": np.array([1.0])}
        np.savez(output_path, **result)
        return result

    monkeypatch.setattr(runner, "evaluate_channel_estimator", evaluate_stub)
    result_name = runner.SparseDiffuseChannelEstimator.__name__
    runner.run_benchmark(
        cfg,
        result_name=result_name,
        smoke=True,
    )

    output_dir = tmp_path / runner.PAPER_SLUG / "smoke" / cfg.scenario_id
    assert (output_dir / "config.json").is_file()
    inventory = json.loads((output_dir / "results.json").read_text())
    assert set(inventory["results"]) == {result_name}

    entry = inventory["results"][result_name]
    assert entry["constructor_arguments"] == arguments
    assert entry["assets"] == []
    assert (output_dir / entry["file"]).is_file()
