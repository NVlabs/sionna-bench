# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the shared persistent-asset manifest store."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from sionna.bench.mimo_ofdm_chest.configs import umi_3gpp_config
from sionna.bench.mimo_ofdm_chest.asset_store import (
    build_asset_manifest,
    external_asset_dir,
    load_asset_manifest,
    paper_asset_dir,
)
from sionna.bench.mimo_ofdm_chest.persistence import (
    write_json_atomic,
    write_npy_atomic,
    write_npz_atomic,
)


def _create_asset(asset_dir: Path) -> tuple[Path, dict, Path]:
    """Create one valid minimal asset and return its manifest and payload."""
    payload_path = asset_dir / "model.npz"
    write_npz_atomic(
        payload_path,
        weights=np.ones(2, dtype=np.float32),
        means=np.zeros((2, 3), dtype=np.float64),
    )
    manifest = build_asset_manifest(
        cfg=umi_3gpp_config,
        asset_type="spectral-gmm",
        artifact_id="k17",
        compatibility={"num_components": 17},
        generation_recipe={"seed": 7},
        payload_paths=[payload_path],
        metadata={"training": {"iterations": 15}},
    )
    manifest_path = asset_dir / "manifest.json"
    write_json_atomic(manifest_path, manifest)
    return manifest_path, manifest, payload_path


def test_build_and_load_asset_manifest(tmp_path: Path) -> None:
    """The public API records and validates NPY and NPZ payload structure."""
    npy_path = tmp_path / "covariance.npy"
    npz_path = tmp_path / "model.npz"
    write_npy_atomic(npy_path, np.ones((2, 3), dtype=np.complex64))
    write_npz_atomic(
        npz_path,
        weights=np.ones(2, dtype=np.float32),
        means=np.zeros((2, 3), dtype=np.float64),
    )
    manifest = build_asset_manifest(
        cfg=umi_3gpp_config,
        asset_type="spectral-gmm",
        artifact_id="k17",
        compatibility={"num_components": 17},
        generation_recipe={"seed": 7},
        payload_paths=[npy_path, npz_path],
        metadata={"training": {"iterations": 15}},
    )
    manifest_path = tmp_path / "manifest.json"
    write_json_atomic(manifest_path, manifest)

    loaded = load_asset_manifest(
        manifest_path,
        asset_dir=tmp_path,
        expected_asset_type="spectral-gmm",
        expected_scenario_id="3gpp-umi",
        expected_artifact_id="k17",
        expected_compatibility={"num_components": 17},
    )

    assert loaded["asset_schema_version"] == 2
    assert loaded["generation_recipe"] == {"seed": 7}
    assert loaded["metadata"] == {"training": {"iterations": 15}}
    assert loaded["files"][0]["format"] == "npy"
    assert loaded["files"][0]["array"] == {
        "shape": [2, 3],
        "dtype": "complex64",
    }
    assert loaded["files"][1]["format"] == "npz"
    assert loaded["files"][1]["arrays"] == {
        "weights": {"shape": [2], "dtype": "float32"},
        "means": {"shape": [2, 3], "dtype": "float64"},
    }


def test_paper_asset_dir_constructs_bundled_path(tmp_path: Path) -> None:
    """Bundled assets are resolved relative to the paper's asset module."""
    module_file = tmp_path / "package" / "paper" / "asset.py"

    path = paper_asset_dir(
        umi_3gpp_config,
        "spectral-gmm",
        "k17",
        module_file,
    )

    assert path == (
        module_file.parent
        / "assets"
        / "3gpp-umi"
        / "spectral-gmm"
        / "k17"
    )
    assert not path.exists()


def test_external_asset_dir_constructs_external_path(tmp_path: Path) -> None:
    """External assets retain their paper owner beneath the supplied root."""
    path = external_asset_dir(
        umi_3gpp_config,
        "koller2022",
        "spectral-gmm",
        "k17",
        asset_root=tmp_path,
    )

    assert path == (
        tmp_path / "3gpp-umi" / "koller2022" / "spectral-gmm" / "k17"
    )
    assert not path.exists()


@pytest.mark.parametrize(
    "failure, match",
    [
        ("required-field", "missing required fields"),
        ("identifier", "asset_type"),
        ("schema", "Unsupported asset schema version"),
        ("compatibility", "compatibility metadata"),
        ("metadata", "metadata.*object"),
    ],
)
def test_load_asset_manifest_rejects_invalid_envelope(
    tmp_path: Path,
    failure: str,
    match: str,
) -> None:
    """The loader rejects invalid common fields and expected compatibility."""
    manifest_path, manifest, _ = _create_asset(tmp_path)
    expected_compatibility = {"num_components": 17}
    if failure == "required-field":
        del manifest["generation_recipe"]
    elif failure == "identifier":
        manifest["asset_type"] = "Spectral GMM"
    elif failure == "schema":
        manifest["asset_schema_version"] = 3
    elif failure == "compatibility":
        expected_compatibility = {"num_components": 16}
    else:
        manifest["metadata"] = []
    write_json_atomic(manifest_path, manifest)

    with pytest.raises(ValueError, match=match):
        load_asset_manifest(
            manifest_path,
            expected_compatibility=expected_compatibility,
        )


@pytest.mark.parametrize(
    "failure, match",
    [
        ("duplicate", "duplicate file paths"),
        ("traversal", "relative and contained"),
    ],
)
def test_load_asset_manifest_rejects_invalid_file_paths(
    tmp_path: Path,
    failure: str,
    match: str,
) -> None:
    """Manifest payload paths must be unique and remain inside the asset."""
    manifest_path, manifest, _ = _create_asset(tmp_path)
    if failure == "duplicate":
        manifest["files"].append(deepcopy(manifest["files"][0]))
    else:
        manifest["files"][0]["path"] = "../model.npz"
    write_json_atomic(manifest_path, manifest)

    with pytest.raises(ValueError, match=match):
        load_asset_manifest(manifest_path, asset_dir=tmp_path)


@pytest.mark.parametrize(
    "failure, match",
    [
        ("missing", "is missing"),
        ("size", "size mismatch"),
        ("checksum", "checksum mismatch"),
    ],
)
def test_load_asset_manifest_rejects_invalid_payload(
    tmp_path: Path,
    failure: str,
    match: str,
) -> None:
    """Payload existence, size, and SHA-256 must match the manifest."""
    manifest_path, _, payload_path = _create_asset(tmp_path)
    if failure == "missing":
        payload_path.unlink()
    elif failure == "size":
        payload_path.write_bytes(payload_path.read_bytes() + b"x")
    else:
        payload_path.write_bytes(b"x" * payload_path.stat().st_size)

    with pytest.raises(ValueError, match=match):
        load_asset_manifest(manifest_path, asset_dir=tmp_path)
