# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for tiny Baur VAE training assets and changed system dimensions."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch

from sionna.bench.mimo_ofdm_chest.asset_store import load_asset_manifest
from sionna.bench.mimo_ofdm_chest.persistence import (
    sha256_file,
    sha256_fingerprint,
    write_json_atomic,
    write_npz_atomic,
)
from sionna.bench.mimo_ofdm_chest.benchmarking import build_resource_grid
from sionna.bench.mimo_ofdm_chest.papers.baur2022 import asset as baur_asset
from sionna.bench.mimo_ofdm_chest.papers.baur2022 import tuning
from sionna.bench.mimo_ofdm_chest.papers.baur2022.asset import METHOD_NOISY, METHOD_NAMES
from sionna.bench.mimo_ofdm_chest.papers.baur2022.estimator import (
    COVARIANCE_FULL,
    DECODER_CHANNELS,
    DECODER_KERNEL_SIZE,
    DECODER_PADDING,
    DECODER_STRIDE,
    ENCODER_CHANNELS,
    ENCODER_KERNEL_SIZE,
    ENCODER_PADDING,
    ENCODER_STRIDE,
    VAEChannelEstimator,
    VariationalAutoEncoder,
)
from test.mimo_ofdm_chest.system_variants import (
    system_variant_config,
    validate_estimator_output,
)


_TRAINING_SNR_DB = 5.0
_LATENT_DIM = 2
_SYSTEM_VARIANTS = ("baseline", "alternate", "wide")
_TRAINING_CASES = [
    (method, system_variant)
    for system_variant in _SYSTEM_VARIANTS
    for method in METHOD_NAMES
]


def test_missing_model_reports_preparation_and_evaluation(
    tmp_path: Path,
) -> None:
    """A missing VAE bank reports both required runner commands."""
    cfg = system_variant_config(
        "baseline",
        snr_db_start=_TRAINING_SNR_DB,
        snr_db_stop=_TRAINING_SNR_DB,
    )
    resource_grid = build_resource_grid(cfg)

    with pytest.raises(FileNotFoundError, match="--prepare-assets") as error:
        baur_asset.load_model(resource_grid, cfg, asset_root=tmp_path)

    assert "then evaluate it with" in str(error.value)
    assert "--method noisy --asset-root <path>" in str(error.value)


@pytest.mark.parametrize(
    ("loss", "best_loss", "expected"),
    [
        (1.0, float("inf"), True),
        (9.98, 10.0, True),
        (9.995, 10.0, False),
        (-10.02, -10.0, True),
        (-10.005, -10.0, False),
        (-9.995, -10.0, False),
    ],
)
def test_relative_loss_improvement_handles_both_signs(
    loss: float,
    best_loss: float,
    expected: bool,
) -> None:
    """Relative early stopping requires a real decrease for signed losses."""
    assert (
        baur_asset._is_relative_loss_improvement(
            loss,
            best_loss,
            relative_min_delta=1e-3,
        )
        is expected
    )


def test_training_state_is_reproducible_from_its_seed() -> None:
    """A training call does not depend on the incoming global RNG state."""
    generator = torch.Generator().manual_seed(11)
    training_inputs = torch.randn(8, 4, generator=generator)
    training_targets = torch.randn(8, 4, generator=generator)
    training_noise = torch.full((8, 1), 0.1)
    validation_inputs = torch.randn(4, 4, generator=generator)
    validation_targets = torch.randn(4, 4, generator=generator)
    validation_noise = torch.full((4, 1), 0.1)

    def train_once():
        return baur_asset._train_state(
            training_inputs,
            training_targets,
            training_noise,
            validation_inputs,
            validation_targets,
            validation_noise,
            method=METHOD_NOISY,
            input_dim=4,
            latent_dim=2,
            epochs=1,
            optimizer_batch_size=4,
            learning_rate=1e-3,
            free_bits=0.0,
            min_train_seconds=0.0,
            patience_epochs=0,
            relative_min_delta=0.0,
            covariance_floor=1e-5,
            covariance_mode=COVARIANCE_FULL,
            seed=7,
            resample_input_noise=True,
        )

    first_state, first_history = train_once()
    torch.manual_seed(999)
    torch.rand(17)
    second_state, second_history = train_once()

    assert first_state.keys() == second_state.keys()
    for name in first_state:
        torch.testing.assert_close(
            first_state[name],
            second_state[name],
            rtol=0.0,
            atol=0.0,
        )
    for key in ("best_train_loss", "best_val_loss", "best_epoch"):
        assert first_history[key] == second_history[key]


def _tiny_model_recipe(
    cfg,
    method,
    asset_root: Path,
) -> dict:
    """Return the identity of one tiny full-covariance VAE asset."""
    return {
        "method": method,
        "latent_dim": _LATENT_DIM,
        "train_batches": 1,
        "train_batch_size": 1,
        "validation_batches": 1,
        "validation_batch_size": 1,
        "epochs": 1,
        "optimizer_batch_size": 128,
        "learning_rate": 1e-4,
        "free_bits": 0.0,
        "snr_specific_weights": True,
        "training_snr_dbs": [_TRAINING_SNR_DB],
        "covariance_floor": 1e-5,
        "covariance_mode": COVARIANCE_FULL,
        "seed": cfg.seed,
        "asset_root": asset_root,
    }


def _train_tiny_model(resource_grid, cfg, method, asset_root: Path):
    """Prepare one single-epoch, full-covariance VAE asset."""
    training_data_manifest = None
    if method == METHOD_NOISY:
        training_data_manifest = baur_asset.generate_training_data(
            resource_grid,
            cfg,
            train_batches=1,
            train_batch_size=1,
            validation_batches=1,
            validation_batch_size=1,
            training_snr_dbs=[_TRAINING_SNR_DB],
            seed=cfg.seed,
            asset_root=asset_root,
        )
    return baur_asset.train_model(
        resource_grid,
        cfg,
        training_data_manifest=training_data_manifest,
        **_tiny_model_recipe(cfg, method, asset_root),
    )


def _load_tiny_model(resource_grid, cfg, method, asset_root: Path):
    """Load one prepared tiny VAE asset without permitting training."""
    return baur_asset.load_model(
        resource_grid,
        cfg,
        **_tiny_model_recipe(cfg, method, asset_root),
    )


@pytest.fixture(
    scope="module",
    params=_TRAINING_CASES,
    ids=lambda case: f"{case[0]}-{case[1]}",
)
def tiny_trained_baur_asset(
    request: pytest.FixtureRequest,
    tmp_path_factory: pytest.TempPathFactory,
):
    """Train both paper methods on two isolated system geometries."""
    method, system_variant = request.param
    root = tmp_path_factory.mktemp(f"baur-{method}-{system_variant}")
    cfg = system_variant_config(
        system_variant,
        snr_db_start=_TRAINING_SNR_DB,
        snr_db_stop=_TRAINING_SNR_DB,
        output_dir=str(root / "outputs"),
    )
    resource_grid = build_resource_grid(cfg)
    asset_root = root / "assets"
    model_states, model_noise_variances, manifest_paths = _train_tiny_model(
        resource_grid,
        cfg,
        method,
        asset_root,
    )
    return (
        method,
        cfg,
        resource_grid,
        model_states,
        model_noise_variances,
        manifest_paths,
        asset_root,
    )


def _build_estimator(
    cfg,
    resource_grid,
    model_states,
    model_noise_variances,
) -> VAEChannelEstimator:
    """Build the full-covariance estimator from prepared temporary states."""
    return VAEChannelEstimator(
        resource_grid,
        model_states,
        cfg.num_rx_ant,
        model_noise_variances,
        latent_dim=_LATENT_DIM,
        covariance_floor=1e-5,
        covariance_mode=COVARIANCE_FULL,
        precision=cfg.precision,
        device=cfg.device,
    )


def _copy_tiny_asset(
    tiny_trained_baur_asset,
    tmp_path: Path,
):
    """Copy a shared tiny asset before deliberately corrupting it."""
    (
        method,
        cfg,
        resource_grid,
        _,
        _,
        _,
        asset_root,
    ) = tiny_trained_baur_asset
    isolated_asset_root = tmp_path / "assets"
    shutil.copytree(asset_root, isolated_asset_root)
    params_path = next(isolated_asset_root.rglob("vae_params.npz"))
    manifest_path = params_path.parent / "manifest.json"
    return (
        method,
        cfg,
        resource_grid,
        isolated_asset_root,
        manifest_path,
        params_path,
    )


@pytest.mark.training
@pytest.mark.system_variants
def test_tiny_single_snr_models_train_persist_and_reload(
    tiny_trained_baur_asset,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Noisy and real VAEs train and reload only from temporary assets."""
    (
        method,
        cfg,
        resource_grid,
        model_states,
        model_noise_variances,
        manifest_paths,
        asset_root,
    ) = tiny_trained_baur_asset

    assert len(model_states) == 1
    assert model_noise_variances.shape == (1,)
    assert len(manifest_paths) == 1
    assert len(list(asset_root.rglob("manifest.json"))) == (
        2 if method == METHOD_NOISY else 1
    )
    assert len(list(asset_root.rglob("vae_params.npz"))) == 1

    manifest = load_asset_manifest(
        manifest_paths[0],
        asset_dir=manifest_paths[0].parent,
    )
    assert manifest["asset_schema_version"] == 2
    assert manifest["generation_recipe"]["training_snr_dbs"] == [
        _TRAINING_SNR_DB
    ]
    group_data = baur_asset._PilotGroupLayout(
        resource_grid
    ).training_group_data()
    coordinates = group_data["coordinates"]
    pilot_values = group_data["pilot_values"]
    pilot_groups = manifest["compatibility"]["pilot_groups"]
    assert manifest["compatibility"]["encoder_channels"] == list(
        ENCODER_CHANNELS
    )
    assert manifest["compatibility"]["encoder_kernel_size"] == (
        ENCODER_KERNEL_SIZE
    )
    assert manifest["compatibility"]["encoder_stride"] == ENCODER_STRIDE
    assert manifest["compatibility"]["encoder_padding"] == ENCODER_PADDING
    assert manifest["compatibility"]["decoder_channels"] == list(
        DECODER_CHANNELS
    )
    assert manifest["compatibility"]["decoder_kernel_size"] == (
        DECODER_KERNEL_SIZE
    )
    assert manifest["compatibility"]["decoder_stride"] == DECODER_STRIDE
    assert manifest["compatibility"]["decoder_padding"] == DECODER_PADDING
    assert manifest["compatibility"]["covariance_head"] == (
        "full_real_cholesky_from_decoder_linear_features"
    )
    assert manifest["generation_recipe"]["noise_resampling"] == (
        "deterministic-per-snr-epoch"
        if method == METHOD_NOISY
        else "fixed-per-generated-training-example"
    )
    assert pilot_groups["coordinate_order"] == [
        "tx", "stream", "symbol", "subcarrier"
    ]
    assert pilot_groups["coordinates"] == {
        "shape": list(coordinates.shape),
        "dtype": str(coordinates.dtype).removeprefix("torch."),
        "canonical_sha256": sha256_fingerprint(coordinates.tolist()),
    }
    assert pilot_groups["values"] == {
        "shape": list(pilot_values.shape),
        "dtype": str(pilot_values.dtype).removeprefix("torch."),
        "encoding": "[real, imaginary]",
        "canonical_sha256": sha256_fingerprint(
            torch.view_as_real(pilot_values).tolist()
        ),
    }
    pilot_powers = pilot_values.abs().square()
    assert pilot_groups["powers"] == {
        "shape": list(pilot_powers.shape),
        "dtype": str(pilot_powers.dtype).removeprefix("torch."),
        "canonical_sha256": sha256_fingerprint(pilot_powers.tolist()),
    }
    def unexpected_training(*args, **kwargs):
        raise AssertionError("A completed temporary VAE must not be retrained.")

    monkeypatch.setattr(
        baur_asset,
        "_collect_training_examples",
        unexpected_training,
    )
    reloaded_states, reloaded_noise_variances, reloaded_paths = (
        _load_tiny_model(resource_grid, cfg, method, asset_root)
    )
    assert reloaded_paths == manifest_paths
    torch.testing.assert_close(reloaded_noise_variances, model_noise_variances)
    assert reloaded_states[0].keys() == model_states[0].keys()
    for key, value in model_states[0].items():
        torch.testing.assert_close(reloaded_states[0][key], value)


@pytest.mark.training
@pytest.mark.system_variants
def test_trained_estimator_runs_on_changed_system_dimensions(
    tiny_trained_baur_asset,
) -> None:
    """Prepared VAEs support changed arrays, UEs, grids, and pilot symbols."""
    (
        _,
        cfg,
        resource_grid,
        model_states,
        model_noise_variances,
        _,
        _,
    ) = tiny_trained_baur_asset
    estimator = _build_estimator(
        cfg,
        resource_grid,
        model_states,
        model_noise_variances,
    )
    received_grid = torch.randn(
        1,
        1,
        cfg.num_rx_ant,
        cfg.num_ofdm_symbols,
        cfg.fft_size,
        dtype=torch.complex64,
    )
    noise_variance = model_noise_variances[0]

    with torch.no_grad():
        h_hat, err_var = estimator(received_grid, noise_variance)

    expected_shape = (
        1,
        1,
        cfg.num_rx_ant,
        cfg.num_ue,
        1,
        cfg.num_ofdm_symbols,
        cfg.fft_size,
    )
    h_true = torch.zeros(expected_shape, dtype=torch.complex64)
    validate_estimator_output(h_hat, err_var, h_true)
    assert bool(torch.any(h_hat.abs() > 0.0))


@pytest.mark.training
@pytest.mark.system_variants
def test_trained_full_covariance_and_posterior_are_well_formed(
    tiny_trained_baur_asset,
) -> None:
    """Trained covariances are positive definite and yield finite posteriors."""
    (
        _,
        cfg,
        resource_grid,
        model_states,
        model_noise_variances,
        _,
        _,
    ) = tiny_trained_baur_asset
    estimator = _build_estimator(
        cfg,
        resource_grid,
        model_states,
        model_noise_variances,
    )
    model = VariationalAutoEncoder(
        2 * cfg.num_rx_ant,
        _LATENT_DIM,
        covariance_floor=1e-5,
        covariance_mode=COVARIANCE_FULL,
    )
    model.load_state_dict(model_states[0])
    model.eval()

    real_observations = torch.randn(3, 2 * cfg.num_rx_ant)
    with torch.no_grad():
        _, _, _, covariance = model(real_observations, sample=False)
    torch.testing.assert_close(covariance, covariance.transpose(-2, -1))
    assert bool(torch.isfinite(covariance).all())
    torch.linalg.cholesky(covariance)

    antenna_values = torch.randn(3, cfg.num_rx_ant, dtype=torch.complex64)
    torch.testing.assert_close(
        estimator._idft(estimator._dft(antenna_values)),
        antenna_values,
        rtol=1e-5,
        atol=1e-5,
    )
    error_variance = torch.full_like(antenna_values.real, 0.1)
    with torch.no_grad():
        estimates, posterior_variances = estimator._estimate_group(
            antenna_values,
            error_variance,
            estimator._vaes[0],
        )
    assert bool(torch.isfinite(estimates).all())
    assert bool(torch.isfinite(posterior_variances).all())
    assert bool(torch.all(posterior_variances >= 0.0))


@pytest.mark.training
@pytest.mark.parametrize("geometry", ("pilot", "receive"))
def test_checkpoint_loaders_reject_incompatible_geometry(
    tiny_trained_baur_asset,
    tmp_path: Path,
    geometry: str,
) -> None:
    """Noisy and real VAE loaders reject incompatible pilot/receive geometry."""
    (
        method,
        cfg,
        resource_grid,
        asset_root,
        manifest_path,
        _,
    ) = _copy_tiny_asset(tiny_trained_baur_asset, tmp_path)
    with manifest_path.open(encoding="utf-8") as file:
        manifest = json.load(file)
    if geometry == "pilot":
        manifest["compatibility"]["pilot_groups"]["coordinates"][
            "canonical_sha256"
        ] = "0" * 64
    else:
        manifest["compatibility"]["input_dim"] += 2
        manifest["compatibility"]["receive_vector_length"] += 1
    write_json_atomic(manifest_path, manifest)

    with pytest.raises(ValueError, match="compatibility metadata does not match"):
        _load_tiny_model(resource_grid, cfg, method, asset_root)


@pytest.mark.training
@pytest.mark.parametrize(
    ("malformation", "error_match"),
    [
        ("missing-state-key", "state keys"),
        ("incorrect-shape", "state arrays"),
        ("incorrect-dtype", "state arrays"),
    ],
)
def test_checkpoint_loaders_reject_malformed_state_tensors(
    tiny_trained_baur_asset,
    tmp_path: Path,
    malformation: str,
    error_match: str,
) -> None:
    """Noisy and real VAE checkpoints require exact state keys and tensors."""
    (
        method,
        cfg,
        resource_grid,
        asset_root,
        manifest_path,
        params_path,
    ) = _copy_tiny_asset(tiny_trained_baur_asset, tmp_path)
    with manifest_path.open(encoding="utf-8") as file:
        manifest = json.load(file)
    if malformation == "missing-state-key":
        manifest["metadata"]["state_keys"] = manifest["metadata"][
            "state_keys"
        ][:-1]
        write_json_atomic(manifest_path, manifest)
    else:
        with np.load(params_path, allow_pickle=False) as data:
            arrays = {name: data[name].copy() for name in data.files}
        if malformation == "incorrect-shape":
            arrays["arr_0"] = arrays["arr_0"].reshape(-1)[:-1]
        else:
            arrays["arr_0"] = arrays["arr_0"].astype(np.float64)
        write_npz_atomic(params_path, **arrays)
        file_record = manifest["files"][0]
        file_record["size_bytes"] = params_path.stat().st_size
        file_record["sha256"] = sha256_file(params_path)
        file_record["arrays"]["arr_0"] = {
            "shape": list(arrays["arr_0"].shape),
            "dtype": str(arrays["arr_0"].dtype),
        }
        write_json_atomic(manifest_path, manifest)

    with pytest.raises(ValueError, match=error_match):
        _load_tiny_model(resource_grid, cfg, method, asset_root)


def test_variance_calibration_uses_bler_only(monkeypatch, tmp_path) -> None:
    """Both the cached objective and study metadata select codeword BLER."""
    calls = {}
    objective = object()

    def fake_build_objective(*args, **kwargs):
        calls["objective"] = kwargs
        return objective

    def fake_tune(*args, **kwargs):
        calls["study"] = kwargs
        return object()

    monkeypatch.setattr(
        tuning,
        "build_cached_detector_scale_objective",
        fake_build_objective,
    )
    monkeypatch.setattr(
        tuning,
        "_load_canonical_estimator_kwargs",
        lambda *args: {},
    )
    monkeypatch.setattr(tuning, "tune_channel_estimator", fake_tune)

    tuning._run_variance_calibration(
        n_trials=1,
        num_mc_batches=1,
        batch_size=1,
        output_dir=tmp_path,
        algorithm_output_dir=tmp_path,
        asset_root=tmp_path,
    )

    assert calls["objective"]["detector_metric"] == "bler"
    assert calls["study"]["detector_metric"] == "bler"
    assert calls["study"]["estimator_objective"] is objective
