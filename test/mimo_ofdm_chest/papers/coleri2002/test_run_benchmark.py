# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for Coleri benchmark result persistence."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from sionna.bench.mimo_ofdm_chest.configs import smoke_config
from sionna.bench.mimo_ofdm_chest.benchmarking.result_store import resolve_constructor_arguments
from sionna.bench.mimo_ofdm_chest.papers.coleri2002 import run_benchmark as runner
from sionna.bench.mimo_ofdm_chest.papers.coleri2002.estimator import (
    METHOD_LOWPASS,
    METHOD_NAMES,
)


def test_smoke_results_record_each_coleri_method(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Every method records its NPZ and exact persistable arguments."""
    cfg = replace(smoke_config, output_dir=str(tmp_path), device="cpu")

    def evaluate_stub(_estimator_class, _cfg, output_path, **_kwargs):
        result = {"nmse": np.array([1.0])}
        np.savez(output_path, **result)
        return result

    monkeypatch.setattr(runner, "evaluate_channel_estimator", evaluate_stub)

    expected_arguments = {}
    for method in METHOD_NAMES:
        arguments = {
            "method": method,
        }
        argument_names = ["method", "err_var_scale"]
        if method == METHOD_LOWPASS:
            argument_names.extend(
                ("lowpass_filter_half_length", "lowpass_cutoff")
            )
        expected = resolve_constructor_arguments(
            runner.LSCombEstimator,
            argument_names,
            arguments,
        )
        result_name = (
            f"{runner.LSCombEstimator.__name__} "
            f"({runner.METHOD_LABELS[method]})"
        )
        runner.run_benchmark(
            cfg,
            result_name=result_name,
            constructor_arguments=arguments,
            smoke=True,
        )
        expected_arguments[result_name] = expected

    output_dir = tmp_path / runner.PAPER_SLUG / "smoke" / cfg.scenario_id
    assert (output_dir / "config.json").is_file()
    inventory = json.loads((output_dir / "results.json").read_text())
    assert set(inventory["results"]) == set(expected_arguments)

    for result_name, arguments in expected_arguments.items():
        entry = inventory["results"][result_name]
        assert entry["constructor_arguments"] == arguments
        assert entry["assets"] == []
        assert (output_dir / entry["file"]).is_file()
