# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for Wei benchmark result persistence."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from sionna.bench.mimo_ofdm_chest.configs import smoke_config
from sionna.bench.mimo_ofdm_chest.benchmarking.result_store import resolve_constructor_arguments
from sionna.bench.mimo_ofdm_chest.papers.wei2017 import run_benchmark as runner


def test_smoke_result_records_complete_sbl_configuration(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """The SBL result records its NPZ and all behavioral arguments."""
    cfg = replace(smoke_config, output_dir=str(tmp_path), device="cpu")
    argument_names = (
        "interpolation_type",
        "num_taps",
        "num_iter",
        "alpha_init",
        "posterior_variance_scale",
    )
    arguments = resolve_constructor_arguments(
        runner.SBLChannelEstimator,
        argument_names,
    )

    def evaluate_stub(_estimator_class, _cfg, output_path, **_kwargs):
        result = {"nmse": np.array([1.0])}
        np.savez(output_path, **result)
        return result

    monkeypatch.setattr(runner, "evaluate_channel_estimator", evaluate_stub)
    result_name = runner.SBLChannelEstimator.__name__
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
