# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for Li ReEsNet training and persistent model assets."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from sionna.bench.mimo_ofdm_chest.asset_store import load_asset_manifest
from sionna.bench.mimo_ofdm_chest.benchmarking import build_resource_grid
from sionna.bench.mimo_ofdm_chest.persistence import sha256_file, write_json_atomic, write_npz_atomic
from sionna.bench.mimo_ofdm_chest.papers.li2020 import asset as li_asset
from sionna.bench.mimo_ofdm_chest.papers.li2020.estimator import ReEsNetChannelEstimator
from test.mimo_ofdm_chest.system_variants import system_variant_config, validate_estimator_output


_TRAINING_SNR_DB = 5.0
_TINY_RECIPE = {
    "train_batches": 1,
    "train_batch_size": 1,
    "validation_batches": 1,
    "validation_batch_size": 1,
    "epochs": 1,
    "optimizer_batch_size": 16,
    "learning_rate": 1e-3,
    "snr_specific_weights": True,
    "training_snr_dbs": [_TRAINING_SNR_DB],
    "paper_style_training": False,
}


def test_missing_model_reports_preparation_and_evaluation(
    tmp_path: Path,
) -> None:
    """A missing ReEsNet bank reports both required runner commands."""
    cfg = system_variant_config(
        "baseline",
        snr_db_start=_TRAINING_SNR_DB,
        snr_db_stop=_TRAINING_SNR_DB,
    )
    resource_grid = build_resource_grid(cfg)

    with pytest.raises(FileNotFoundError, match="--prepare-assets") as error:
        li_asset.load_model(resource_grid, cfg, asset_root=tmp_path)

    assert "then evaluate it with" in str(error.value)
    assert "--asset-root <path>" in str(error.value)


@pytest.fixture(params=("baseline", "alternate"))
def tiny_reesnet_asset(request, tmp_path: Path):
    """Train one isolated checkpoint for each of two small geometries."""
    cfg = system_variant_config(
        request.param,
        snr_db_start=_TRAINING_SNR_DB,
        snr_db_stop=_TRAINING_SNR_DB,
        output_dir=str(tmp_path / "outputs"),
    )
    resource_grid = build_resource_grid(cfg)
    asset_root = tmp_path / "assets"
    training_data_manifest = li_asset.generate_training_data(
        resource_grid,
        cfg,
        train_batches=_TINY_RECIPE["train_batches"],
        train_batch_size=_TINY_RECIPE["train_batch_size"],
        validation_batches=_TINY_RECIPE["validation_batches"],
        validation_batch_size=_TINY_RECIPE["validation_batch_size"],
        training_snr_dbs=_TINY_RECIPE["training_snr_dbs"],
        paper_style_training=False,
        seed=cfg.seed,
        asset_root=asset_root,
    )
    states, noise_variances, manifest_paths = li_asset.train_model(
        resource_grid,
        cfg,
        seed=cfg.seed,
        training_data_manifest=training_data_manifest,
        asset_root=asset_root,
        **_TINY_RECIPE,
    )
    return cfg, resource_grid, states, noise_variances, manifest_paths, asset_root


@pytest.mark.training
@pytest.mark.system_variants
def test_tiny_models_train_reload_and_run(tiny_reesnet_asset) -> None:
    """Two grids train, strictly reload, and produce valid full-grid outputs."""
    cfg, resource_grid, states, noise_variances, paths, asset_root = (
        tiny_reesnet_asset
    )
    assert len(states) == len(paths) == 1
    assert noise_variances is not None and noise_variances.shape == (1,)
    manifest = load_asset_manifest(paths[0], asset_dir=paths[0].parent)
    assert manifest["asset_schema_version"] == 2
    assert "err_var_scale" not in manifest["generation_recipe"]
    assert list(manifest["metadata"]) == ["state_keys", "history"]
    assert manifest["metadata"]["history"]["training_data_manifest"][
        "sha256"
    ]

    loaded_states, loaded_noise_variances, loaded_paths = li_asset.load_model(
        resource_grid,
        cfg,
        seed=cfg.seed,
        asset_root=asset_root,
        **_TINY_RECIPE,
    )
    assert loaded_paths == paths
    torch.testing.assert_close(loaded_noise_variances, noise_variances)
    for key, value in states[0].items():
        torch.testing.assert_close(loaded_states[0][key], value)

    estimator = ReEsNetChannelEstimator(
        resource_grid,
        loaded_states,
        loaded_noise_variances,
        precision=cfg.precision,
        device=cfg.device,
    )
    received_grid = torch.zeros(
        1,
        1,
        cfg.num_rx_ant,
        cfg.num_ofdm_symbols,
        cfg.fft_size,
        dtype=torch.complex64,
    )
    with torch.no_grad():
        h_hat, err_var = estimator(received_grid, loaded_noise_variances[0])
    validate_estimator_output(h_hat, err_var, torch.zeros_like(h_hat))


@pytest.mark.training
@pytest.mark.parametrize("malformation", ("geometry", "keys", "tensor"))
def test_loader_rejects_incompatible_assets(
    tiny_reesnet_asset,
    malformation: str,
) -> None:
    """Geometry, state-key, and checkpoint-tensor changes are rejected."""
    cfg, resource_grid, _, _, paths, asset_root = tiny_reesnet_asset
    manifest_path = paths[0]
    with manifest_path.open(encoding="utf-8") as file:
        manifest = json.load(file)

    if malformation == "geometry":
        manifest["compatibility"]["output_shape"][0] += 1
    elif malformation == "keys":
        manifest["metadata"]["state_keys"] = manifest["metadata"][
            "state_keys"
        ][:-1]
    else:
        params_path = manifest_path.parent / "reesnet_params.npz"
        with np.load(params_path, allow_pickle=False) as data:
            arrays = {name: data[name] for name in data.files}
        arrays["arr_0"] = arrays["arr_0"].astype(np.float64)
        write_npz_atomic(params_path, **arrays)
        record = manifest["files"][0]
        record["size_bytes"] = params_path.stat().st_size
        record["sha256"] = sha256_file(params_path)
        record["arrays"]["arr_0"]["dtype"] = "float64"
    write_json_atomic(manifest_path, manifest)

    message = {
        "geometry": "compatibility metadata does not match",
        "keys": "incompatible state keys",
        "tensor": "incompatible state arrays",
    }[malformation]
    with pytest.raises(ValueError, match=message):
        li_asset.load_model(
            resource_grid,
            cfg,
            seed=cfg.seed,
            asset_root=asset_root,
            **_TINY_RECIPE,
        )
