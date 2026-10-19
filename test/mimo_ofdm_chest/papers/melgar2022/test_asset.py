# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for Melgar DNN training and temporary model assets."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from sionna.phy.utils import ebnodb2no

from sionna.bench.mimo_ofdm_chest.asset_store import build_asset_manifest, load_asset_manifest
from sionna.bench.mimo_ofdm_chest.benchmarking import build_resource_grid
from sionna.bench.mimo_ofdm_chest.papers.melgar2022 import asset as melgar_asset
from sionna.bench.mimo_ofdm_chest.papers.melgar2022 import estimator as melgar_estimator
from sionna.bench.mimo_ofdm_chest.papers.melgar2022.estimator import (
    DNNChannelEstimator,
    InterpolatedDNNChannelEstimator,
    METHOD_MODEL_2,
    METHOD_MODEL_3,
    METHOD_NAMES,
)
from sionna.bench.mimo_ofdm_chest.persistence import write_json_atomic, write_npz_atomic
from test.mimo_ofdm_chest.system_variants import (
    system_variant_config,
    validate_estimator_output,
)


_TRAINING_SNR_DB = 5.0
_MODEL_3_BLOCK_LEN = 12
_SYSTEM_VARIANTS = ("baseline", "alternate", "wide")
_TRAINING_CASES = [
    (method, system_variant)
    for system_variant in _SYSTEM_VARIANTS
    for method in METHOD_NAMES
]


@pytest.mark.parametrize("method", METHOD_NAMES)
def test_missing_model_reports_preparation_and_evaluation(
    tmp_path: Path,
    method: str,
) -> None:
    """A missing DNN bank reports method-specific runner commands."""
    cfg = system_variant_config(
        "baseline",
        snr_db_start=_TRAINING_SNR_DB,
        snr_db_stop=_TRAINING_SNR_DB,
    )
    resource_grid = build_resource_grid(cfg)

    with pytest.raises(FileNotFoundError, match="--prepare-assets") as error:
        _load_model(
            resource_grid,
            cfg,
            method=method,
            block_len=_MODEL_3_BLOCK_LEN,
            asset_root=tmp_path,
        )

    assert "then evaluate it with" in str(error.value)
    assert f"--method {method} --asset-root <path>" in str(error.value)


def _method_arguments(method: str, arguments: dict) -> dict:
    """Remove Model 3-only arguments before a Model 2 asset call."""
    arguments = dict(arguments)
    if method == METHOD_MODEL_2:
        arguments.pop("block_len", None)
        arguments.pop("max_examples_per_batch", None)
    return arguments


def _train_model(resource_grid, cfg, *, method: str, **arguments):
    trainer = (
        melgar_asset.train_model2
        if method == METHOD_MODEL_2
        else melgar_asset.train_model3
    )
    return trainer(resource_grid, cfg, **_method_arguments(method, arguments))


def _load_model(resource_grid, cfg, *, method: str, **arguments):
    loader = (
        melgar_asset.load_model2
        if method == METHOD_MODEL_2
        else melgar_asset.load_model3
    )
    return loader(resource_grid, cfg, **_method_arguments(method, arguments))


def _generate_training_data(resource_grid, cfg, *, method: str, **arguments):
    generator = (
        melgar_asset.generate_model2_training_data
        if method == METHOD_MODEL_2
        else melgar_asset.generate_model3_training_data
    )
    return generator(resource_grid, cfg, **_method_arguments(method, arguments))


def _train_tiny_model(resource_grid, cfg, method, asset_root):
    """Prepare one minimal single-SNR Melgar model asset."""
    return _train_model(
        resource_grid,
        cfg,
        method=method,
        block_len=_MODEL_3_BLOCK_LEN,
        train_batches=1,
        train_batch_size=1,
        validation_batches=1,
        validation_batch_size=1,
        epochs=1,
        optimizer_batch_size=32,
        learning_rate=1e-4,
        patience_epochs=0,
        min_delta=0.0,
        max_examples_per_batch=16,
        snr_specific_weights=True,
        training_snr_dbs=[_TRAINING_SNR_DB],
        seed=cfg.seed,
        asset_root=asset_root,
    )


def test_training_restores_best_validation_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Returned state and final metrics describe the best validation epoch."""

    class ScalarDNN(torch.nn.Module):
        def __init__(self, input_dim: int) -> None:
            super().__init__()
            del input_dim
            self.weight = torch.nn.Parameter(torch.ones(()))

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            return inputs * self.weight

    monkeypatch.setattr(melgar_estimator, "DNN", ScalarDNN)
    inputs = torch.ones(4, 1)
    state, history = melgar_asset._train_state(
        inputs,
        torch.zeros_like(inputs),
        inputs,
        torch.full_like(inputs, 0.8),
        input_dim=1,
        epochs=10,
        optimizer_batch_size=4,
        learning_rate=0.01,
        min_train_seconds=0.0,
        patience_epochs=0,
        min_delta=0.0,
        seed=1,
    )

    restored = ScalarDNN(1)
    restored.load_state_dict(state)
    restored_val_mse = torch.mean((restored(inputs) - 0.8).square()).item()
    assert history["best_epoch"] < len(history["trace"])
    assert history["val_mse"] == pytest.approx(history["best_val_mse"])
    assert restored_val_mse == pytest.approx(history["best_val_mse"])


@pytest.mark.training
@pytest.mark.parametrize("method", METHOD_NAMES)
def test_training_reuses_one_strict_fixed_data_bank(
    method: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both Melgar methods reuse validated data without sampling again."""
    cfg = system_variant_config(
        "baseline",
        snr_db_start=_TRAINING_SNR_DB,
        snr_db_stop=_TRAINING_SNR_DB,
    )
    resource_grid = build_resource_grid(cfg)
    asset_root = tmp_path / "assets"
    data_manifest = _generate_training_data(
        resource_grid,
        cfg,
        method=method,
        train_batches=1,
        train_batch_size=1,
        validation_batches=1,
        validation_batch_size=1,
        training_snr_dbs=[_TRAINING_SNR_DB],
        block_len=_MODEL_3_BLOCK_LEN,
        max_examples_per_batch=(None if method == METHOD_MODEL_2 else 16),
        seed=cfg.seed,
        asset_root=asset_root,
    )

    def unexpected_collection(*args, **kwargs):
        raise AssertionError("Training must reuse the fixed data bank.")

    monkeypatch.setattr(
        melgar_asset,
        "_collect_model2_training_pairs",
        unexpected_collection,
    )
    monkeypatch.setattr(
        melgar_asset,
        "_collect_model3_training_pairs",
        unexpected_collection,
    )
    _, _, model_manifests = _train_model(
        resource_grid,
        cfg,
        method=method,
        block_len=_MODEL_3_BLOCK_LEN,
        train_batches=1,
        train_batch_size=1,
        validation_batches=1,
        validation_batch_size=1,
        epochs=1,
        optimizer_batch_size=16,
        learning_rate=1e-4,
        patience_epochs=1,
        min_delta=1e-4,
        max_examples_per_batch=(None if method == METHOD_MODEL_2 else 16),
        snr_specific_weights=True,
        training_snr_dbs=[_TRAINING_SNR_DB],
        training_data_manifest=data_manifest,
        seed=cfg.seed,
        asset_root=asset_root,
    )
    manifest = load_asset_manifest(model_manifests[0])
    reference = manifest["metadata"]["history"]["training_data_manifest"]
    assert reference["artifact_id"] == data_manifest.parent.name
    assert reference["sha256"]


@pytest.fixture(
    params=_TRAINING_CASES,
    ids=lambda case: f"{case[0]}-{case[1]}",
)
def tiny_trained_melgar_asset(
    request: pytest.FixtureRequest,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """Train one tiny single-SNR model under an isolated asset root."""
    monkeypatch.chdir(tmp_path)
    method, system_variant = request.param
    asset_root = tmp_path / "assets"
    cfg = system_variant_config(
        system_variant,
        snr_db_start=_TRAINING_SNR_DB,
        snr_db_stop=_TRAINING_SNR_DB,
        output_dir=str(tmp_path / "outputs"),
    )
    resource_grid = build_resource_grid(cfg)
    model_states, model_noise_variances, manifest_paths = _train_tiny_model(
        resource_grid, cfg, method, asset_root
    )
    assert model_noise_variances is not None
    return (
        method,
        cfg,
        resource_grid,
        model_states,
        model_noise_variances,
        manifest_paths,
        asset_root,
    )


@pytest.mark.training
@pytest.mark.system_variants
def test_tiny_single_snr_models_train_persist_and_run(
    tiny_trained_melgar_asset,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both models train, reload safely, reject mismatches, and run."""
    (
        method,
        cfg,
        resource_grid,
        model_states,
        model_noise_variances,
        manifest_paths,
        asset_root,
    ) = tiny_trained_melgar_asset

    assert len(model_states) == 1
    assert model_noise_variances.shape == (1,)
    assert len(manifest_paths) == 1
    assert len(list(asset_root.rglob("manifest.json"))) == 1
    assert len(list(asset_root.rglob("dnn_params.npz"))) == 1
    manifest_path = next(asset_root.rglob("manifest.json"))
    assert manifest_paths == [manifest_path]
    manifest = load_asset_manifest(manifest_path, asset_dir=manifest_path.parent)
    assert manifest["asset_schema_version"] == 2
    assert "key_fields" not in manifest
    assert "snr_db_range" not in manifest["generation_recipe"]
    assert manifest["generation_recipe"]["training_snr_dbs"] == [
        _TRAINING_SNR_DB
    ]

    def unexpected_training(*args, **kwargs):
        raise AssertionError("A completed temporary model must not be retrained.")

    monkeypatch.setattr(
        melgar_asset,
        "_collect_model2_training_pairs",
        unexpected_training,
    )
    monkeypatch.setattr(
        melgar_asset,
        "_collect_model3_training_pairs",
        unexpected_training,
    )
    reloaded_states, reloaded_noise_variances, reloaded_manifest_paths = (
        _load_model(
            resource_grid,
            cfg,
            method=method,
            block_len=_MODEL_3_BLOCK_LEN,
            train_batches=1,
            train_batch_size=1,
            validation_batches=1,
            validation_batch_size=1,
            epochs=1,
            optimizer_batch_size=32,
            learning_rate=1e-4,
            patience_epochs=0,
            min_delta=0.0,
            max_examples_per_batch=16,
            snr_specific_weights=True,
            training_snr_dbs=[_TRAINING_SNR_DB],
            seed=cfg.seed,
            asset_root=asset_root,
        )
    )
    assert reloaded_manifest_paths == manifest_paths
    torch.testing.assert_close(
        reloaded_noise_variances,
        model_noise_variances,
    )
    assert reloaded_states[0].keys() == model_states[0].keys()
    for key, value in model_states[0].items():
        torch.testing.assert_close(reloaded_states[0][key], value)

    evaluation_cfg = replace(
        cfg,
        snr_db_start=-10.0,
        snr_db_stop=15.0,
        snr_db_step=2.5,
    )
    evaluation_resource_grid = build_resource_grid(evaluation_cfg)
    evaluation_states, _, _ = _load_model(
        evaluation_resource_grid,
        evaluation_cfg,
        method=method,
        block_len=_MODEL_3_BLOCK_LEN,
        train_batches=1,
        train_batch_size=1,
        validation_batches=1,
        validation_batch_size=1,
        epochs=1,
        optimizer_batch_size=32,
        learning_rate=1e-4,
        patience_epochs=0,
        min_delta=0.0,
        max_examples_per_batch=16,
        snr_specific_weights=True,
        training_snr_dbs=[_TRAINING_SNR_DB],
        seed=cfg.seed,
        asset_root=asset_root,
    )
    for key, value in model_states[0].items():
        torch.testing.assert_close(evaluation_states[0][key], value)

    shifted_pilot_symbols = tuple(
        symbol_index + 1 for symbol_index in cfg.pilot_ofdm_symbol_indices
    )
    assert shifted_pilot_symbols[-1] < cfg.num_ofdm_symbols
    incompatible_configs = [
        replace(
            cfg,
            pilot_ofdm_symbol_indices=shifted_pilot_symbols,
        )
    ]
    if method == METHOD_MODEL_3:
        incompatible_configs.append(replace(cfg, num_prb=cfg.num_prb + 1))
    for incompatible_cfg in incompatible_configs:
        incompatible_resource_grid = build_resource_grid(incompatible_cfg)
        with pytest.raises(
            ValueError,
            match="compatibility metadata does not match",
        ):
            _load_model(
                incompatible_resource_grid,
                incompatible_cfg,
                method=method,
                block_len=_MODEL_3_BLOCK_LEN,
                train_batches=1,
                train_batch_size=1,
                validation_batches=1,
                validation_batch_size=1,
                epochs=1,
                optimizer_batch_size=32,
                learning_rate=1e-4,
                patience_epochs=0,
                min_delta=0.0,
                max_examples_per_batch=16,
                snr_specific_weights=True,
                training_snr_dbs=[_TRAINING_SNR_DB],
                seed=cfg.seed,
                asset_root=asset_root,
            )

    state = model_states[0]
    weight_keys = sorted(
        (key for key in state if key.endswith(".weight")),
        key=lambda key: int(key.split(".")[1]),
    )
    state_input_dim = int(state[weight_keys[0]].shape[1])
    state_output_dim = int(state[weight_keys[-1]].shape[0])

    if method == METHOD_MODEL_2:
        pilot_layout = melgar_asset._PilotGroupLayout(resource_grid)
        group_data = pilot_layout.training_group_data()
        assert "interpolation" not in manifest["compatibility"]
        assert manifest["compatibility"]["pilot_groups"] == (
            melgar_asset._model2_pilot_layout(group_data)
        )
        active_pilot_counts = torch.count_nonzero(
            resource_grid.pilot_pattern.pilots,
            dim=-1,
        )
        num_pilot_symbols = len(cfg.pilot_ofdm_symbol_indices)
        assert bool(torch.all(active_pilot_counts % num_pilot_symbols == 0))
        group_lengths = active_pilot_counts // num_pilot_symbols
        assert int(torch.unique(group_lengths).numel()) == 1
        expected_input_dim = int(group_lengths.flatten()[0].item())
        expected_examples = (
            2
            * cfg.num_rx_ant
            * int(group_data["pilot_values"].shape[0])
        )
        estimator = DNNChannelEstimator(
            resource_grid,
            model_states,
            model_noise_variances,
            precision=cfg.precision,
            device=cfg.device,
        )
    else:
        assert method == METHOD_MODEL_3
        assert manifest["compatibility"][
            "ls_interpolation_layout"
        ] == melgar_asset._model3_ls_interpolation_layout(resource_grid)
        expected_input_dim = _MODEL_3_BLOCK_LEN
        num_frequency_blocks = (
            resource_grid.num_effective_subcarriers // _MODEL_3_BLOCK_LEN
        )
        uncapped_examples = (
            2
            * cfg.num_rx_ant
            * resource_grid.num_tx
            * resource_grid.num_streams_per_tx
            * resource_grid.num_ofdm_symbols
            * num_frequency_blocks
        )
        expected_examples = min(16, uncapped_examples)
        estimator = InterpolatedDNNChannelEstimator(
            resource_grid,
            model_states,
            model_noise_variances,
            block_len=_MODEL_3_BLOCK_LEN,
            precision=cfg.precision,
            device=cfg.device,
        )
    assert state_input_dim == expected_input_dim
    assert state_output_dim == expected_input_dim
    history = manifest["metadata"]["history"]
    assert history["train_channels"] == 1
    assert history["validation_channels"] == 1
    assert history["train_examples"] == expected_examples
    assert history["validation_examples"] == expected_examples

    received_grid = torch.randn(
        1,
        1,
        cfg.num_rx_ant,
        cfg.num_ofdm_symbols,
        cfg.fft_size,
        dtype=torch.complex64,
    )
    noise_variance = ebnodb2no(
        torch.tensor(_TRAINING_SNR_DB, dtype=torch.float32),
        cfg.num_bits_per_symbol,
        cfg.coderate,
        resource_grid,
    )
    with torch.no_grad():
        channel_estimate, error_variance = estimator(
            received_grid,
            noise_variance,
        )

    expected_channel = torch.zeros(
        1,
        1,
        cfg.num_rx_ant,
        resource_grid.num_tx,
        resource_grid.num_streams_per_tx,
        cfg.num_ofdm_symbols,
        resource_grid.num_effective_subcarriers,
        dtype=received_grid.dtype,
        device=received_grid.device,
    )
    validate_estimator_output(
        channel_estimate,
        error_variance,
        expected_channel,
    )
    assert bool(torch.any(channel_estimate != 0.0))


@pytest.mark.training
@pytest.mark.parametrize("method", METHOD_NAMES)
@pytest.mark.parametrize(
    ("malformation", "error_match"),
    [
        ("missing-state-key", "state keys"),
        ("incorrect-shape", "shape"),
        ("incorrect-dtype", "dtype"),
    ],
)
def test_checkpoint_loaders_reject_malformed_state_tensors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    method: str,
    malformation: str,
    error_match: str,
) -> None:
    """Both checkpoint loaders reject invalid state structure and tensors."""
    monkeypatch.chdir(tmp_path)
    cfg = system_variant_config(
        "baseline",
        snr_db_start=_TRAINING_SNR_DB,
        snr_db_stop=_TRAINING_SNR_DB,
        output_dir=str(tmp_path / "outputs"),
    )
    resource_grid = build_resource_grid(cfg)
    asset_root = tmp_path / "assets"
    _train_tiny_model(resource_grid, cfg, method, asset_root)

    manifest_path = next(asset_root.rglob("manifest.json"))
    payload_path = next(asset_root.rglob("dnn_params.npz"))
    manifest = load_asset_manifest(manifest_path, asset_dir=manifest_path.parent)
    assert "key_fields" not in manifest
    if malformation == "missing-state-key":
        manifest["metadata"]["state_keys"] = manifest["metadata"][
            "state_keys"
        ][:-1]
        write_json_atomic(manifest_path, manifest)
    else:
        with np.load(payload_path, allow_pickle=False) as data:
            arrays = {name: data[name].copy() for name in data.files}
        if malformation == "incorrect-shape":
            arrays["arr_0"] = arrays["arr_0"][:, :-1]
        else:
            arrays["arr_0"] = arrays["arr_0"].astype(np.float64)
        write_npz_atomic(payload_path, **arrays)
        refreshed_manifest = build_asset_manifest(
            cfg=cfg,
            asset_type=manifest["asset_type"],
            artifact_id=manifest["artifact_id"],
            compatibility=manifest["compatibility"],
            generation_recipe=manifest["generation_recipe"],
            payload_paths=[payload_path],
            metadata=manifest["metadata"],
        )
        write_json_atomic(manifest_path, refreshed_manifest)

    def unexpected_training(*args, **kwargs):
        raise AssertionError("A malformed asset must not trigger retraining.")

    monkeypatch.setattr(
        melgar_asset,
        "_collect_model2_training_pairs",
        unexpected_training,
    )
    monkeypatch.setattr(
        melgar_asset,
        "_collect_model3_training_pairs",
        unexpected_training,
    )
    with pytest.raises(ValueError, match=error_match):
        _load_model(
            resource_grid,
            cfg,
            method=method,
            block_len=_MODEL_3_BLOCK_LEN,
            train_batches=1,
            train_batch_size=1,
            validation_batches=1,
            validation_batch_size=1,
            epochs=1,
            optimizer_batch_size=32,
            learning_rate=1e-4,
            patience_epochs=0,
            min_delta=0.0,
            max_examples_per_batch=16,
            snr_specific_weights=True,
            training_snr_dbs=[_TRAINING_SNR_DB],
            seed=cfg.seed,
            asset_root=asset_root,
        )


@pytest.mark.training
def test_per_snr_manifests_follow_model_and_noise_order(tmp_path: Path) -> None:
    """Per-SNR states, noise values, and manifests retain one common order."""
    snr_dbs = [-5.0, 5.0]
    cfg = system_variant_config(
        "baseline",
        snr_db_start=snr_dbs[0],
        snr_db_stop=snr_dbs[-1],
        snr_db_step=10.0,
    )
    resource_grid = build_resource_grid(cfg)
    states, noise_variances, manifest_paths = _train_model(
        resource_grid,
        cfg,
        method=METHOD_MODEL_2,
        train_batches=1,
        train_batch_size=1,
        validation_batches=1,
        validation_batch_size=1,
        epochs=1,
        optimizer_batch_size=32,
        learning_rate=1e-4,
        snr_specific_weights=True,
        training_snr_dbs=snr_dbs,
        seed=cfg.seed,
        asset_root=tmp_path / "assets",
    )

    assert len(states) == len(noise_variances) == len(manifest_paths) == 2
    recorded_snr_dbs = [
        load_asset_manifest(path)["generation_recipe"]["training_snr_dbs"][0]
        for path in manifest_paths
    ]
    assert recorded_snr_dbs == snr_dbs
    assert bool(torch.all(noise_variances[:-1] > noise_variances[1:]))
