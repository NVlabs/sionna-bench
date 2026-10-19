# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for built-in LMMSE covariance asset loading."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from sionna.bench.mimo_ofdm_chest.asset_store import build_asset_manifest
from sionna.bench.mimo_ofdm_chest.persistence import write_json_atomic, write_npy_atomic
from sionna.bench.mimo_ofdm_chest.papers.builtin import asset as builtin_asset
from sionna.bench.mimo_ofdm_chest.papers.builtin.asset import (
    estimate_covariance,
    load_covariance,
)
from sionna.bench.mimo_ofdm_chest.papers.builtin.run_benchmark import (
    COV_BATCH_SIZE,
    COV_NUM_ITERS,
)
from test.mimo_ofdm_chest.system_variants import system_variant_config


_NUM_IT = 1
_BATCH_SIZE = 2
_SEED = 17
_PAYLOAD_NAMES = ("freq_cov.npy", "time_cov.npy", "space_cov.npy")


def test_canonical_covariance_recipe_uses_64000_channels() -> None:
    """The default canonical covariance recipe uses 64,000 channels."""
    assert COV_NUM_ITERS * COV_BATCH_SIZE == 64_000


def _valid_covariances(cfg) -> dict[str, np.ndarray]:
    return {
        "freq_cov.npy": np.eye(cfg.fft_size, dtype=np.complex64),
        "time_cov.npy": np.eye(cfg.num_ofdm_symbols, dtype=np.complex64),
        "space_cov.npy": np.eye(cfg.num_rx_ant, dtype=np.complex64),
    }


def _write_covariance_asset(
    cfg,
    asset_root: Path,
    covariances: dict[str, np.ndarray],
) -> Path:
    covariance_dir = Path(
        builtin_asset._covariance_asset_dir(cfg, asset_root=asset_root)
    )
    covariance_dir.mkdir(parents=True, exist_ok=True)
    payload_paths = []
    for name in _PAYLOAD_NAMES:
        path = covariance_dir / name
        write_npy_atomic(path, covariances[name])
        payload_paths.append(path)

    compatibility = builtin_asset._covariance_compatibility(cfg)
    generation_recipe = builtin_asset._generation_recipe(
        _NUM_IT,
        _BATCH_SIZE,
        _SEED,
    )
    manifest = build_asset_manifest(
        cfg=cfg,
        asset_type="lmmse-covariance",
        artifact_id="default",
        compatibility=compatibility,
        generation_recipe=generation_recipe,
        payload_paths=payload_paths,
        metadata=builtin_asset._covariance_metadata(),
    )
    write_json_atomic(covariance_dir / "manifest.json", manifest)
    return covariance_dir


def _load_covariances(cfg, asset_root: Path):
    covariance, _ = load_covariance(cfg, asset_root=asset_root)
    return covariance


def test_covariance_loader_accepts_valid_temporary_asset(tmp_path: Path) -> None:
    """A complete dimension-matched version-2 asset loads unchanged."""
    cfg = system_variant_config("baseline")
    expected = _valid_covariances(cfg)
    _write_covariance_asset(cfg, tmp_path, expected)

    loaded, manifest_path = load_covariance(cfg, asset_root=tmp_path)

    for matrix, name in zip(loaded, _PAYLOAD_NAMES):
        np.testing.assert_array_equal(matrix, expected[name])
    assert manifest_path == (
        builtin_asset._covariance_asset_dir(cfg, asset_root=tmp_path)
        / "manifest.json"
    )


def test_strict_covariance_loader_never_generates(tmp_path: Path) -> None:
    """Strict loading rejects a missing asset and accepts a prepared one."""
    cfg = system_variant_config("baseline")
    with pytest.raises(FileNotFoundError, match="--prepare-assets") as error:
        load_covariance(cfg, asset_root=tmp_path)
    assert "then evaluate it with" in str(error.value)
    assert "--estimator lmmse --asset-root <path>" in str(error.value)
    assert not any(tmp_path.rglob("*"))

    expected = _valid_covariances(cfg)
    _write_covariance_asset(cfg, tmp_path, expected)
    loaded, _ = load_covariance(cfg, asset_root=tmp_path)
    for matrix, name in zip(loaded, _PAYLOAD_NAMES):
        np.testing.assert_array_equal(matrix, expected[name])


def test_covariance_generation_writes_final_external_asset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Generation writes schema 2 metadata and requires explicit overwrite."""
    cfg = system_variant_config("baseline")
    expected = _valid_covariances(cfg)
    monkeypatch.setattr(
        builtin_asset,
        "_estimate_covariance_matrices",
        lambda *_: tuple(expected[name] for name in _PAYLOAD_NAMES),
    )

    loaded, manifest_path = estimate_covariance(
        cfg,
        num_it=_NUM_IT,
        batch_size=_BATCH_SIZE,
        seed=_SEED,
        asset_root=tmp_path,
    )

    for matrix, name in zip(loaded, _PAYLOAD_NAMES):
        np.testing.assert_array_equal(matrix, expected[name])
    with manifest_path.open(encoding="utf-8") as file:
        manifest = json.load(file)
    assert manifest["asset_schema_version"] == 2
    assert manifest["generation_recipe"] == {
        "num_it": _NUM_IT,
        "batch_size": _BATCH_SIZE,
        "seed": _SEED,
    }
    assert manifest["metadata"] == builtin_asset._covariance_metadata()

    with pytest.raises(FileExistsError, match="already exists"):
        estimate_covariance(
            cfg,
            num_it=_NUM_IT,
            batch_size=_BATCH_SIZE,
            seed=_SEED,
            asset_root=tmp_path,
        )

    _, overwritten_manifest = estimate_covariance(
        cfg,
        num_it=_NUM_IT,
        batch_size=_BATCH_SIZE,
        seed=_SEED,
        asset_root=tmp_path,
        overwrite=True,
    )
    assert overwritten_manifest == manifest_path


@pytest.mark.parametrize(
    "changed_cfg",
    [
        lambda cfg: replace(cfg, num_prb=cfg.num_prb + 1),
        lambda cfg: replace(
            cfg,
            cyclic_prefix_length=cfg.cyclic_prefix_length + 1,
        ),
        lambda cfg: replace(cfg, bs_num_cols=cfg.bs_num_cols + 1),
    ],
    ids=["frequency-grid", "ofdm-symbol-duration", "receive-array"],
)
def test_covariance_loader_rejects_incompatible_geometry(
    tmp_path: Path,
    changed_cfg,
) -> None:
    """Grid and receive-array changes invalidate covariance compatibility."""
    cfg = system_variant_config("baseline")
    _write_covariance_asset(cfg, tmp_path, _valid_covariances(cfg))

    with pytest.raises(
        ValueError,
        match="compatibility metadata does not match",
    ):
        _load_covariances(changed_cfg(cfg), tmp_path)


@pytest.mark.parametrize("missing_kind", ["manifest-record", "payload"])
def test_covariance_loader_rejects_missing_payload(
    tmp_path: Path,
    missing_kind: str,
) -> None:
    """A missing named manifest record or payload file is rejected."""
    cfg = system_variant_config("baseline")
    covariance_dir = _write_covariance_asset(
        cfg,
        tmp_path,
        _valid_covariances(cfg),
    )
    if missing_kind == "manifest-record":
        manifest_path = covariance_dir / "manifest.json"
        with manifest_path.open(encoding="utf-8") as file:
            manifest = json.load(file)
        manifest["files"] = manifest["files"][1:]
        write_json_atomic(manifest_path, manifest)
        expected_message = "must list exactly"
    else:
        (covariance_dir / "freq_cov.npy").unlink()
        expected_message = "Asset payload is missing"

    with pytest.raises(ValueError, match=expected_message):
        _load_covariances(cfg, tmp_path)


@pytest.mark.parametrize("payload_name", _PAYLOAD_NAMES)
def test_covariance_loader_rejects_incorrect_shape(
    tmp_path: Path,
    payload_name: str,
) -> None:
    """Every covariance payload must match its configured square dimension."""
    cfg = system_variant_config("baseline")
    covariances = _valid_covariances(cfg)
    covariances[payload_name] = covariances[payload_name][:-1]
    _write_covariance_asset(cfg, tmp_path, covariances)

    with pytest.raises(ValueError, match="must have shape"):
        _load_covariances(cfg, tmp_path)


@pytest.mark.parametrize("payload_name", _PAYLOAD_NAMES)
def test_covariance_loader_rejects_incorrect_dtype(
    tmp_path: Path,
    payload_name: str,
) -> None:
    """Every covariance payload must use complex64 values."""
    cfg = system_variant_config("baseline")
    covariances = _valid_covariances(cfg)
    covariances[payload_name] = covariances[payload_name].real.astype(np.float32)
    _write_covariance_asset(cfg, tmp_path, covariances)

    with pytest.raises(ValueError, match="must have dtype complex64"):
        _load_covariances(cfg, tmp_path)
