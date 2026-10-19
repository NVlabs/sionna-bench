# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for Edfors covariance asset generation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from sionna.phy.ofdm import PilotPattern

from sionna.bench.mimo_ofdm_chest.benchmarking import build_resource_grid
from sionna.bench.mimo_ofdm_chest.asset_store import build_asset_manifest
from sionna.bench.mimo_ofdm_chest.persistence import write_json_atomic, write_npz_atomic
from sionna.bench.mimo_ofdm_chest.papers.edfors1996 import asset as edfors_asset
from sionna.bench.mimo_ofdm_chest.papers.edfors1996.asset import (
    estimate_covariance,
    load_covariance,
)
from test.mimo_ofdm_chest.system_variants import system_variant_config


_COV_NUM_ITERS = 1
_COV_BATCH_SIZE = 1
_SEED = 17


def _write_covariance_asset(
    resource_grid,
    cfg,
    asset_root: Path,
    arrays: dict[str, np.ndarray],
) -> Path:
    """Write one internally consistent temporary Edfors asset."""
    group_subcarrier_indices = edfors_asset._PilotGroupLayout(
        resource_grid.pilot_pattern
    ).subcarrier_indices
    asset_dir = Path(
        edfors_asset._asset_dir(
            cfg,
            cov_num_iters=_COV_NUM_ITERS,
            cov_batch_size=_COV_BATCH_SIZE,
            seed=_SEED,
            asset_root=asset_root,
        )
    )
    asset_dir.mkdir(parents=True)
    payload_path = asset_dir / "time_covariances.npz"
    write_npz_atomic(payload_path, **arrays)
    compatibility = edfors_asset._compatibility(
        cfg,
        group_subcarrier_indices,
    )
    generation_recipe = edfors_asset._generation_recipe(
        cov_num_iters=_COV_NUM_ITERS,
        cov_batch_size=_COV_BATCH_SIZE,
        seed=_SEED,
    )
    manifest = build_asset_manifest(
        cfg=cfg,
        asset_type=edfors_asset._ASSET_TYPE,
        artifact_id=asset_dir.name,
        compatibility=compatibility,
        generation_recipe=generation_recipe,
        payload_paths=[payload_path],
        metadata=edfors_asset._metadata(),
    )
    write_json_atomic(asset_dir / "manifest.json", manifest)
    return asset_dir


def _valid_arrays(num_pilot_subcarriers: int) -> dict[str, np.ndarray]:
    """Return a valid covariance payload for one pilot-vector length."""
    return {
        "time_covariances": np.eye(
            num_pilot_subcarriers,
            dtype=np.complex64,
        ),
        "tap_powers": np.ones(num_pilot_subcarriers, dtype=np.float32),
    }


@pytest.mark.parametrize("system_variant", ["baseline", "alternate", "wide"])
def test_covariance_generation_supports_alternative_geometries(
    tmp_path: Path,
    system_variant: str,
) -> None:
    """Tiny assets match two noncanonical pilot geometries and reload exactly."""
    cfg = system_variant_config(system_variant)
    resource_grid = build_resource_grid(cfg)
    asset_root = tmp_path / system_variant
    generation_kwargs = {
        "cov_num_iters": _COV_NUM_ITERS,
        "cov_batch_size": _COV_BATCH_SIZE,
        "seed": cfg.seed,
        "asset_root": asset_root,
    }

    covariance, manifest_path = estimate_covariance(
        resource_grid,
        cfg,
        **generation_kwargs,
    )

    active_pilots = resource_grid.pilot_pattern.pilots.abs() > 0
    num_pilot_symbols = len(cfg.pilot_ofdm_symbol_indices)
    num_pilot_subcarriers = (
        int(active_pilots.sum(dim=-1).min().item()) // num_pilot_symbols
    )
    assert covariance.shape == (
        num_pilot_subcarriers,
        num_pilot_subcarriers,
    )
    assert covariance.dtype == np.complex64
    assert np.isfinite(covariance).all()
    np.testing.assert_allclose(covariance, covariance.conj().T, atol=1e-6)

    with manifest_path.open(encoding="utf-8") as file:
        manifest = json.load(file)
    assert manifest["asset_schema_version"] == 2
    assert manifest["compatibility"]["fft_size"] == cfg.fft_size
    assert manifest["compatibility"]["num_ofdm_symbols"] == cfg.num_ofdm_symbols
    assert manifest["compatibility"]["num_ue"] == cfg.num_ue
    assert manifest["compatibility"]["num_rx_ant"] == cfg.num_rx_ant
    assert manifest["compatibility"]["group_lengths"] == [
        num_pilot_subcarriers
    ] * manifest["compatibility"]["num_groups"]
    assert manifest["generation_recipe"] == {
        "cov_batch_size": 1,
        "cov_num_iters": 1,
        "seed": cfg.seed,
    }
    assert manifest["metadata"] == edfors_asset._metadata()

    reloaded, reloaded_manifest = load_covariance(
        resource_grid,
        cfg,
        **generation_kwargs,
    )
    np.testing.assert_array_equal(reloaded, covariance)
    assert reloaded_manifest == manifest_path


def test_covariance_generation_is_reproducible(tmp_path: Path) -> None:
    """The recorded seed reproduces channel draws and the pooled covariance."""
    cfg = system_variant_config("baseline")
    resource_grid = build_resource_grid(cfg)
    generation_kwargs = {
        "cov_num_iters": _COV_NUM_ITERS,
        "cov_batch_size": _COV_BATCH_SIZE,
        "seed": cfg.seed,
    }

    first, _ = estimate_covariance(
        resource_grid,
        cfg,
        asset_root=tmp_path / "first",
        **generation_kwargs,
    )
    second, _ = estimate_covariance(
        resource_grid,
        cfg,
        asset_root=tmp_path / "second",
        **generation_kwargs,
    )

    np.testing.assert_array_equal(first, second)


def test_covariance_loader_rejects_changed_ordered_pilot_geometry(
    tmp_path: Path,
) -> None:
    """A reordered nonzero pilot comb cannot reuse the original covariance."""
    cfg = system_variant_config("alternate")
    resource_grid = build_resource_grid(cfg)
    original_groups = edfors_asset._PilotGroupLayout(
        resource_grid.pilot_pattern
    ).subcarrier_indices
    _write_covariance_asset(
        resource_grid,
        cfg,
        tmp_path,
        _valid_arrays(original_groups.shape[1]),
    )

    changed_resource_grid = build_resource_grid(cfg)
    changed_resource_grid.pilot_pattern = PilotPattern(
        resource_grid.pilot_pattern.mask,
        torch.roll(resource_grid.pilot_pattern.pilots, shifts=1, dims=-1),
        precision=cfg.precision,
        device=cfg.device,
    )
    changed_groups = edfors_asset._PilotGroupLayout(
        changed_resource_grid.pilot_pattern
    ).subcarrier_indices
    assert changed_groups.shape == original_groups.shape
    assert not torch.equal(changed_groups, original_groups)

    with pytest.raises(ValueError, match="compatibility metadata does not match"):
        load_covariance(
            changed_resource_grid,
            cfg,
            cov_num_iters=_COV_NUM_ITERS,
            cov_batch_size=_COV_BATCH_SIZE,
            seed=_SEED,
            asset_root=tmp_path,
        )


def test_covariance_loader_does_not_require_legacy_key_fields(
    tmp_path: Path,
) -> None:
    """A schema-2 asset loads without legacy duplicate identity fields."""
    cfg = system_variant_config("baseline")
    resource_grid = build_resource_grid(cfg)
    expected = _valid_arrays(cfg.fft_size)["time_covariances"]
    asset_dir = _write_covariance_asset(
        resource_grid,
        cfg,
        tmp_path,
        {
            "time_covariances": expected,
            "tap_powers": expected.diagonal().real.astype(np.float32),
        },
    )
    manifest_path = asset_dir / "manifest.json"
    with manifest_path.open(encoding="utf-8") as file:
        manifest = json.load(file)
    assert "key_fields" not in manifest

    loaded, _ = load_covariance(
        resource_grid,
        cfg,
        cov_num_iters=_COV_NUM_ITERS,
        cov_batch_size=_COV_BATCH_SIZE,
        seed=_SEED,
        asset_root=tmp_path,
    )

    np.testing.assert_array_equal(loaded, expected)


def test_strict_covariance_loader_never_estimates(tmp_path: Path) -> None:
    """Strict loading rejects a missing asset and accepts a prepared one."""
    cfg = system_variant_config("baseline")
    resource_grid = build_resource_grid(cfg)
    load_kwargs = {
        "cov_num_iters": _COV_NUM_ITERS,
        "cov_batch_size": _COV_BATCH_SIZE,
        "seed": _SEED,
        "asset_root": tmp_path,
    }
    with pytest.raises(FileNotFoundError, match="--prepare-assets") as error:
        load_covariance(resource_grid, cfg, **load_kwargs)
    assert "then evaluate it with" in str(error.value)
    assert "--asset-root <path>" in str(error.value)
    assert not any(tmp_path.rglob("*"))

    expected = _valid_arrays(cfg.fft_size)["time_covariances"]
    _write_covariance_asset(
        resource_grid,
        cfg,
        tmp_path,
        {
            "time_covariances": expected,
            "tap_powers": expected.diagonal().real.astype(np.float32),
        },
    )
    loaded, _ = load_covariance(resource_grid, cfg, **load_kwargs)
    np.testing.assert_array_equal(loaded, expected)


def test_covariance_loader_rejects_changed_generation_recipe(
    tmp_path: Path,
) -> None:
    """The selected artifact must record its exact Monte Carlo recipe."""
    cfg = system_variant_config("baseline")
    resource_grid = build_resource_grid(cfg)
    asset_dir = _write_covariance_asset(
        resource_grid,
        cfg,
        tmp_path,
        _valid_arrays(cfg.fft_size),
    )
    manifest_path = asset_dir / "manifest.json"
    with manifest_path.open(encoding="utf-8") as file:
        manifest = json.load(file)
    manifest["generation_recipe"]["seed"] += 1
    write_json_atomic(manifest_path, manifest)

    with pytest.raises(ValueError, match="generation recipe does not match"):
        load_covariance(
            resource_grid,
            cfg,
            cov_num_iters=_COV_NUM_ITERS,
            cov_batch_size=_COV_BATCH_SIZE,
            seed=_SEED,
            asset_root=tmp_path,
        )


@pytest.mark.parametrize("missing_key", ["time_covariances", "tap_powers"])
def test_covariance_loader_rejects_missing_npz_key(
    tmp_path: Path,
    missing_key: str,
) -> None:
    """Both covariance NPZ arrays are mandatory."""
    cfg = system_variant_config("baseline")
    resource_grid = build_resource_grid(cfg)
    num_pilot_subcarriers = cfg.fft_size
    arrays = _valid_arrays(num_pilot_subcarriers)
    arrays.pop(missing_key)
    _write_covariance_asset(resource_grid, cfg, tmp_path, arrays)

    with pytest.raises(ValueError, match="unexpected arrays"):
        load_covariance(
            resource_grid,
            cfg,
            cov_num_iters=_COV_NUM_ITERS,
            cov_batch_size=_COV_BATCH_SIZE,
            seed=_SEED,
            asset_root=tmp_path,
        )


@pytest.mark.parametrize("invalid_kind", ["shape", "dtype"])
def test_covariance_loader_rejects_invalid_covariance_array(
    tmp_path: Path,
    invalid_kind: str,
) -> None:
    """The covariance must be a geometry-matched complex64 square matrix."""
    cfg = system_variant_config("baseline")
    resource_grid = build_resource_grid(cfg)
    num_pilot_subcarriers = cfg.fft_size
    arrays = _valid_arrays(num_pilot_subcarriers)
    if invalid_kind == "shape":
        arrays["time_covariances"] = arrays["time_covariances"][:-1]
    else:
        arrays["time_covariances"] = arrays["time_covariances"].astype(
            np.complex128
        )
    _write_covariance_asset(resource_grid, cfg, tmp_path, arrays)

    with pytest.raises(ValueError, match="incompatible covariance shape or dtype"):
        load_covariance(
            resource_grid,
            cfg,
            cov_num_iters=_COV_NUM_ITERS,
            cov_batch_size=_COV_BATCH_SIZE,
            seed=_SEED,
            asset_root=tmp_path,
        )
