# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for Koller GMM fitting and temporary asset persistence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pytest
import torch

from sionna.bench.mimo_ofdm_chest.benchmarking import build_resource_grid
from sionna.bench.mimo_ofdm_chest.asset_store import build_asset_manifest
from sionna.bench.mimo_ofdm_chest.persistence import (
    write_json_atomic,
    write_npy_atomic,
    write_npz_atomic,
)
from sionna.bench.mimo_ofdm_chest.papers.koller2022 import asset as koller_asset
from sionna.bench.mimo_ofdm_chest.papers.koller2022.asset import (
    METHOD_SPATIAL,
    METHOD_SPECTRAL,
    fit_model,
    generate_fitting_data,
    load_fitting_data,
    load_model,
)
from test.mimo_ofdm_chest.system_variants import system_variant_config


_NUM_COMPONENTS = 2
_GENERATION_RECIPE = {
    "num_components": _NUM_COMPONENTS,
    "train_batches": 1,
    "train_batch_size": 1,
    "em_iters": 1,
    "covariance_reg": 1e-3,
    "seed": 17,
}


@pytest.mark.parametrize("method", [METHOD_SPATIAL, METHOD_SPECTRAL])
def test_vectorized_fitting_groups_match_simple_group_loop(method: str) -> None:
    """Batched offline extraction preserves the original group-major order."""
    cfg = system_variant_config("baseline")
    resource_grid = build_resource_grid(cfg)
    groups = koller_asset._PilotGroupLayout(
        resource_grid.pilot_pattern,
        method,
    )._group_records()
    channel = torch.randn(
        2,
        1,
        cfg.num_rx_ant,
        cfg.num_ue,
        cfg.num_streams_per_tx,
        cfg.num_ofdm_symbols,
        resource_grid.num_effective_subcarriers,
        dtype=torch.complex64,
    )
    mode = koller_asset._METHOD_MODES[method]
    expected = torch.cat(
        [
            koller_asset._group_channel_samples(channel, group, mode)
            for group in groups
        ],
        dim=0,
    )
    actual = koller_asset._group_all_channel_samples(channel, groups, mode)
    torch.testing.assert_close(actual, expected)


@pytest.mark.training
def test_both_gmms_fit_from_one_common_channel_bank(tmp_path: Path) -> None:
    """One generated channel stream supplies both method-specific GMM fits."""
    cfg = system_variant_config("baseline")
    resource_grid = build_resource_grid(cfg)
    manifest_path = generate_fitting_data(
        resource_grid,
        cfg,
        train_batches=1,
        train_batch_size=1,
        seed=17,
        asset_root=tmp_path,
        verbose=False,
    )
    fitting_data, loaded_manifest_path = load_fitting_data(
        resource_grid,
        cfg,
        train_batches=1,
        train_batch_size=1,
        seed=17,
        asset_root=tmp_path,
    )
    assert loaded_manifest_path == manifest_path
    assert fitting_data["channels"].shape[0] == 1

    for method in (METHOD_SPATIAL, METHOD_SPECTRAL):
        _, model_manifest_path = fit_model(
            resource_grid,
            cfg,
            method=method,
            num_components=1,
            train_batches=1,
            train_batch_size=1,
            em_iters=1,
            covariance_reg=1e-3,
            seed=17,
            asset_root=tmp_path,
            verbose=False,
            fitting_data_manifest=manifest_path,
        )
        with model_manifest_path.open(encoding="utf-8") as file:
            model_manifest = json.load(file)
        reference = model_manifest["metadata"]["fitting_data"]
        assert reference["artifact_id"] == manifest_path.parent.name
        assert reference["asset_type"] == "gmm-fitting-data"


def _valid_gmm_arrays(
    resource_grid,
    cfg,
    method: str,
) -> dict[str, np.ndarray]:
    """Return valid two-component parameters for one GMM vector geometry."""
    complex_vector_length = (
        cfg.num_rx_ant
        if method == METHOD_SPATIAL
        else int(resource_grid.num_effective_subcarriers)
    )
    real_vector_length = 2 * complex_vector_length
    covariance = np.eye(real_vector_length, dtype=np.float32)
    return {
        "weights": np.full(
            _NUM_COMPONENTS,
            1.0 / _NUM_COMPONENTS,
            dtype=np.float32,
        ),
        "means": np.zeros(
            (_NUM_COMPONENTS, real_vector_length),
            dtype=np.float32,
        ),
        "covs": np.repeat(
            covariance[None, :, :],
            _NUM_COMPONENTS,
            axis=0,
        ),
    }


def _loader_kwargs(method: str, asset_root: Path) -> dict[str, object]:
    """Return the fixed tiny fitting recipe used by loader rejection tests."""
    return {
        "method": method,
        **_GENERATION_RECIPE,
        "verbose": False,
        "asset_root": asset_root,
    }


def _write_gmm_asset(
    resource_grid,
    cfg,
    method: str,
    asset_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    arrays: Optional[dict[str, np.ndarray]] = None,
) -> None:
    """Write one checksum-consistent temporary GMM asset without sampling."""
    valid_arrays = _valid_gmm_arrays(resource_grid, cfg, method)
    monkeypatch.setattr(
        koller_asset,
        "_fit_from_channel_samples",
        lambda *args, **kwargs: (
            valid_arrays["weights"],
            valid_arrays["means"],
            valid_arrays["covs"],
        ),
    )
    fit_model(
        resource_grid,
        cfg,
        **_loader_kwargs(method, asset_root),
    )

    if arrays is None:
        return

    params_path = next(asset_root.rglob("gmm_params.npz"))
    manifest_path = params_path.with_name("manifest.json")
    with manifest_path.open(encoding="utf-8") as file:
        manifest = json.load(file)
    for record in manifest["files"]:
        (manifest_path.parent / record["path"]).unlink()

    payload_paths = [params_path]
    if method == METHOD_SPATIAL:
        write_npz_atomic(params_path, **arrays)
    else:
        write_npz_atomic(
            params_path,
            **{
                name: value
                for name, value in arrays.items()
                if name in {"weights", "means"}
            },
        )
        for component_index, covariance in enumerate(arrays.get("covs", [])):
            covariance_path = manifest_path.with_name(
                f"gmm_covariance_{component_index:03d}.npy"
            )
            write_npy_atomic(covariance_path, covariance)
            payload_paths.append(covariance_path)

    refreshed_manifest = build_asset_manifest(
        cfg=cfg,
        asset_type=manifest["asset_type"],
        artifact_id=manifest["artifact_id"],
        compatibility=manifest["compatibility"],
        generation_recipe=manifest["generation_recipe"],
        payload_paths=payload_paths,
        metadata=manifest["metadata"],
    )
    write_json_atomic(manifest_path, refreshed_manifest)


@pytest.mark.parametrize("system_variant", ["baseline", "alternate"])
@pytest.mark.parametrize("method", [METHOD_SPATIAL, METHOD_SPECTRAL])
@pytest.mark.training
def test_gmm_fitting_supports_alternative_geometries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    system_variant: str,
    method: str,
) -> None:
    """Both GMM modes fit and reload tiny assets for two system geometries."""
    cfg = system_variant_config(system_variant)
    resource_grid = build_resource_grid(cfg)
    num_components = 1 if system_variant == "baseline" else 2
    fitting_kwargs = {
        "method": method,
        "num_components": num_components,
        "train_batches": 1,
        "train_batch_size": 1,
        "em_iters": 1,
        "covariance_reg": 1e-3,
        "seed": cfg.seed,
        "verbose": False,
        "asset_root": tmp_path,
    }

    parameters, manifest_path = fit_model(
        resource_grid,
        cfg,
        **fitting_kwargs,
    )
    weights, means, covariances = parameters

    complex_vector_length = (
        cfg.num_rx_ant
        if method == METHOD_SPATIAL
        else int(resource_grid.num_effective_subcarriers)
    )
    real_vector_length = 2 * complex_vector_length
    assert weights.shape == (num_components,)
    assert means.shape == (num_components, real_vector_length)
    assert covariances.shape == (
        num_components,
        real_vector_length,
        real_vector_length,
    )
    for value in (weights, means, covariances):
        assert value.dtype == torch.float32
        assert bool(torch.isfinite(value).all())
    torch.testing.assert_close(weights.sum(), torch.tensor(1.0))

    manifest_paths = list(tmp_path.rglob("manifest.json"))
    payload_paths = [
        path
        for path in tmp_path.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    ]
    assert len(manifest_paths) == 1
    expected_payload_names = ["gmm_params.npz"]
    if method == METHOD_SPECTRAL:
        expected_payload_names.extend(
            f"gmm_covariance_{component_index:03d}.npy"
            for component_index in range(num_components)
        )
    assert sorted(path.name for path in payload_paths) == sorted(
        expected_payload_names
    )
    with manifest_paths[0].open(encoding="utf-8") as file:
        manifest = json.load(file)
    assert manifest_path == manifest_paths[0]
    assert manifest["asset_schema_version"] == 2
    assert manifest["generation_recipe"] == {
        "covariance_reg": 1e-3,
        "em_iters": 1,
        "num_components": num_components,
        "seed": cfg.seed,
        "train_batch_size": 1,
        "train_batches": 1,
    }
    layout = koller_asset._PilotGroupLayout(resource_grid.pilot_pattern, method)
    expected_compatibility = koller_asset._asset_compatibility(
        resource_grid,
        cfg,
        layout,
        koller_asset._METHOD_MODES[method],
    )
    assert manifest["compatibility"] == expected_compatibility
    assert "num_groups" not in manifest["compatibility"]
    pilot_groups = manifest["compatibility"]["pilot_groups"]
    assert (
        pilot_groups["coordinates"]["shape"][0]
        == layout._pilot_group_num_groups
    )
    assert pilot_groups["values"]["shape"] == [
        layout._pilot_group_num_groups,
        layout._pilot_group_len,
    ]
    assert pilot_groups["powers"]["shape"] == pilot_groups["values"]["shape"]
    assert manifest["metadata"] == koller_asset._asset_metadata(
        koller_asset._METHOD_MODES[method]
    )

    def fail_if_refitted(*args, **kwargs):
        raise AssertionError("Compatible GMM asset was fitted again.")

    monkeypatch.setattr(koller_asset, "_fit_from_channel_samples", fail_if_refitted)
    reloaded, reloaded_manifest_path = load_model(
        resource_grid, cfg, **fitting_kwargs
    )
    assert reloaded_manifest_path == manifest_path
    for actual, expected in zip(reloaded, (weights, means, covariances)):
        torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("method", [METHOD_SPATIAL, METHOD_SPECTRAL])
def test_strict_gmm_loader_never_fits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    method: str,
) -> None:
    """Strict loading rejects a missing GMM and accepts a prepared one."""
    cfg = system_variant_config("baseline")
    resource_grid = build_resource_grid(cfg)
    loader_kwargs = _loader_kwargs(method, tmp_path)
    with pytest.raises(FileNotFoundError, match="--prepare-assets") as error:
        load_model(resource_grid, cfg, **loader_kwargs)
    assert "then evaluate it with" in str(error.value)
    assert (
        f"--prepare-assets --method {method} --model-seed 17 "
        "--asset-root <path>"
    ) in str(error.value)
    assert (
        f"--method {method} --model-seed 17 --asset-root <path>"
    ) in str(error.value)
    assert not any(tmp_path.rglob("*"))

    expected = _valid_gmm_arrays(resource_grid, cfg, method)
    _write_gmm_asset(resource_grid, cfg, method, tmp_path, monkeypatch)
    loaded, manifest_path = load_model(resource_grid, cfg, **loader_kwargs)
    assert manifest_path == next(tmp_path.rglob("manifest.json"))
    for actual, name in zip(loaded, ("weights", "means", "covs")):
        torch.testing.assert_close(actual, torch.from_numpy(expected[name]))


@pytest.mark.parametrize("method", [METHOD_SPATIAL, METHOD_SPECTRAL])
def test_gmm_loader_rejects_changed_pilot_geometry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    method: str,
) -> None:
    """Neither GMM mode may reuse an asset for different pilot coordinates."""
    cfg = system_variant_config("baseline")
    resource_grid = build_resource_grid(cfg)
    _write_gmm_asset(resource_grid, cfg, method, tmp_path, monkeypatch)

    changed_cfg = system_variant_config(
        "baseline",
        pilot_ofdm_symbol_indices=(2,),
    )
    changed_resource_grid = build_resource_grid(changed_cfg)
    with pytest.raises(ValueError, match="compatibility metadata does not match"):
        load_model(
            changed_resource_grid,
            changed_cfg,
            **_loader_kwargs(method, tmp_path),
        )


@pytest.mark.parametrize("method", [METHOD_SPATIAL, METHOD_SPECTRAL])
def test_gmm_loader_rejects_changed_vector_geometry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    method: str,
) -> None:
    """Assets cannot cross spatial or spectral real-vector dimensions."""
    cfg = system_variant_config("baseline")
    resource_grid = build_resource_grid(cfg)
    _write_gmm_asset(resource_grid, cfg, method, tmp_path, monkeypatch)

    if method == METHOD_SPATIAL:
        changed_cfg = system_variant_config("baseline", bs_num_cols=3)
    else:
        changed_cfg = system_variant_config("baseline", num_prb=5)
    changed_resource_grid = build_resource_grid(changed_cfg)
    with pytest.raises(ValueError, match="compatibility metadata does not match"):
        load_model(
            changed_resource_grid,
            changed_cfg,
            **_loader_kwargs(method, tmp_path),
        )


@pytest.mark.parametrize("method", [METHOD_SPATIAL, METHOD_SPECTRAL])
def test_gmm_loader_does_not_require_legacy_key_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    method: str,
) -> None:
    """Both current GMM schemas load without redundant legacy metadata."""
    cfg = system_variant_config("baseline")
    resource_grid = build_resource_grid(cfg)
    expected = _valid_gmm_arrays(resource_grid, cfg, method)
    _write_gmm_asset(resource_grid, cfg, method, tmp_path, monkeypatch)
    manifest_path = next(tmp_path.rglob("manifest.json"))
    with manifest_path.open(encoding="utf-8") as file:
        manifest = json.load(file)
    assert "key_fields" not in manifest

    loaded, _ = load_model(
        resource_grid,
        cfg,
        **_loader_kwargs(method, tmp_path),
    )

    for actual, name in zip(loaded, ("weights", "means", "covs")):
        torch.testing.assert_close(actual, torch.from_numpy(expected[name]))


@pytest.mark.parametrize("method", [METHOD_SPATIAL, METHOD_SPECTRAL])
def test_gmm_loader_rejects_changed_generation_recipe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    method: str,
) -> None:
    """Both GMM modes validate their explicit fitting recipe metadata."""
    cfg = system_variant_config("baseline")
    resource_grid = build_resource_grid(cfg)
    _write_gmm_asset(resource_grid, cfg, method, tmp_path, monkeypatch)
    manifest_path = next(tmp_path.rglob("manifest.json"))
    with manifest_path.open(encoding="utf-8") as file:
        manifest = json.load(file)
    manifest["generation_recipe"]["seed"] += 1
    write_json_atomic(manifest_path, manifest)

    with pytest.raises(ValueError, match="generation recipe"):
        load_model(
            resource_grid,
            cfg,
            **_loader_kwargs(method, tmp_path),
        )


@pytest.mark.parametrize("method", [METHOD_SPATIAL, METHOD_SPECTRAL])
@pytest.mark.parametrize("missing_key", ["weights", "means", "covs"])
def test_gmm_loader_rejects_missing_parameter_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    method: str,
    missing_key: str,
) -> None:
    """All three GMM parameter arrays are mandatory for both modes."""
    cfg = system_variant_config("baseline")
    resource_grid = build_resource_grid(cfg)
    arrays = _valid_gmm_arrays(resource_grid, cfg, method)
    arrays.pop(missing_key)
    _write_gmm_asset(
        resource_grid,
        cfg,
        method,
        tmp_path,
        monkeypatch,
        arrays,
    )

    with pytest.raises(ValueError, match="unexpected (arrays|payload files)"):
        load_model(
            resource_grid,
            cfg,
            **_loader_kwargs(method, tmp_path),
        )


@pytest.mark.parametrize("method", [METHOD_SPATIAL, METHOD_SPECTRAL])
@pytest.mark.parametrize("parameter", ["weights", "means", "covs"])
@pytest.mark.parametrize("invalid_kind", ["shape", "dtype"])
def test_gmm_loader_rejects_invalid_parameter_array(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    method: str,
    parameter: str,
    invalid_kind: str,
) -> None:
    """GMM arrays must have exact real-vector shapes and float32 dtypes."""
    cfg = system_variant_config("baseline")
    resource_grid = build_resource_grid(cfg)
    arrays = _valid_gmm_arrays(resource_grid, cfg, method)
    if invalid_kind == "shape":
        arrays[parameter] = arrays[parameter][:-1]
    else:
        arrays[parameter] = arrays[parameter].astype(np.float64)
    _write_gmm_asset(
        resource_grid,
        cfg,
        method,
        tmp_path,
        monkeypatch,
        arrays,
    )

    with pytest.raises(
        ValueError,
        match="incompatible shapes or dtypes|unexpected payload files",
    ):
        load_model(
            resource_grid,
            cfg,
            **_loader_kwargs(method, tmp_path),
        )
