# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""Store and validate persistent estimator assets.

Each asset directory contains one ``manifest.json`` and one or more NumPy
payloads::

    <asset-directory>/
        manifest.json
        <payload>.npy
        <payload>.npz

The manifest has the common structure::

    {
        "asset_schema_version": 2,
        "asset_type": "...",        # e.g., "spectral-gmm"
        "scenario_id": "...",       # e.g., "3gpp-umi"
        "artifact_id": "...",       # e.g., "n32-z8-..."
        "compatibility": {...},     # e.g., {"fft_size": 624, ...}
        "generation_recipe": {...}, # e.g., {"train_batch_size": 8, ...}
        "files": [
            {
                "path": "...",
                "format": "npy" | "npz",
                "size_bytes": ...,
                "sha256": "...",
                "array": {...} | "arrays": {...}
            }
        ],
        "metadata": {...}          # optional, asset-specific
    }

``compatibility`` contains the facts that must match at inference.
``generation_recipe`` records how the payload was generated. ``files`` records
payload integrity and NumPy structure.

Package-local assets, when present, are read-only inputs. Public distributions
may omit their payloads; users can regenerate them beneath an external asset
root. External assets are writable training, tuning, or temporary artifacts.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Union

import numpy as np

from sionna.bench.mimo_ofdm_chest.configs import SystemConfig

from .persistence import canonical_json, sha256_file, validate_identifier


__all__ = [
    "build_asset_manifest",
    "load_asset_manifest",
    "external_asset_dir",
    "paper_asset_dir",
]

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")

_ASSET_SCHEMA_VERSION = 2


def build_asset_manifest(
    *,
    cfg: SystemConfig,
    asset_type: str,
    artifact_id: str,
    compatibility: Mapping[str, Any],
    generation_recipe: Mapping[str, Any],
    payload_paths: Iterable[str | Path],
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    r"""Build a versioned manifest for NumPy asset payloads.

    Every NPY or NPZ payload is recorded with its relative filename, byte size,
    SHA-256 checksum, array keys, shapes, and dtypes.

    :param cfg: System configuration that supplies ``scenario_id``.
    :param asset_type: Asset type identifier identifier.
    :param artifact_id: Human-readable identity of the selected asset.
    :param compatibility: Facts required to interpret and safely reuse the
        asset during inference.
    :param generation_recipe: Settings used to generate the asset.
    :param payload_paths: NPY or NPZ files to list in the manifest.
    :param metadata: Additional type-specific nested manifest fields.

    :returns: Manifest dictionary ready to write as ``manifest.json``.
    """
    manifest = {
        "asset_schema_version": _ASSET_SCHEMA_VERSION,
        "asset_type": asset_type,
        "scenario_id": cfg.scenario_id,
        "artifact_id": artifact_id,
        "compatibility": dict(compatibility),
        "generation_recipe": dict(generation_recipe),
        "files": [_numpy_file_record(path) for path in payload_paths],
    }
    if metadata is not None:
        manifest["metadata"] = dict(metadata)

    return manifest


def load_asset_manifest(
    path: str | Path,
    *,
    asset_dir: Optional[Union[str, Path]] = None,
    expected_asset_type: Optional[str] = None,
    expected_scenario_id: Optional[str] = None,
    expected_artifact_id: Optional[str] = None,
    expected_compatibility: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    r"""Load an asset manifest and validate its envelope.

    The common field types, identifiers, asset schema version, file records, and
    optional expected identity and compatibility values are checked. When
    ``asset_dir`` is provided, every listed payload must exist beneath it with
    the recorded byte size and SHA-256 checksum. Type-specific array keys,
    shapes, and semantic compatibility remain the loader's responsibility.

    :param path: Path to ``manifest.json``.
    :param asset_dir: Directory containing ``manifest.json`` and its payloads.
        If omitted, payload existence and checksums are not checked.
    :param expected_asset_type: Required ``asset_type``, if known.
    :param expected_scenario_id: Required scenario identifier, if known.
    :param expected_artifact_id: Required artifact identifier, if known.
    :param expected_compatibility: Exact compatibility mapping required by the
        loader.

    :returns: Decoded manifest dictionary.

    :raises ValueError: If the manifest or a checked payload is invalid.
    """
    required_fields = frozenset({
        "asset_schema_version",
        "asset_type",
        "scenario_id",
        "artifact_id",
        "compatibility",
        "generation_recipe",
        "files",
    })

    # Load the manifest.
    with Path(path).open(encoding="utf-8") as file:
        manifest = json.load(file)
    if not isinstance(manifest, dict):
        raise ValueError("An asset manifest must contain a JSON object.")

    # Validate the manifest envelope.
    missing_fields = required_fields.difference(manifest)
    if missing_fields:
        missing = ", ".join(sorted(missing_fields))
        raise ValueError(f"Asset manifest is missing required fields: {missing}.")

    # Validate asset schema version
    asset_schema_version = manifest["asset_schema_version"]
    if  not isinstance(asset_schema_version, int) or asset_schema_version != _ASSET_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported asset schema version {asset_schema_version}."
        )

    # Validate asset type
    asset_type = manifest["asset_type"]
    validate_identifier(asset_type, "asset_type")
    if expected_asset_type is not None and asset_type != expected_asset_type:
        raise ValueError(
            f"Asset type mismatch: expected {expected_asset_type!r}, "
            f"found {asset_type!r}."
        )

    # Validate scenario identifier
    scenario_id = manifest["scenario_id"]
    validate_identifier(scenario_id, "scenario_id")
    if expected_scenario_id is not None and scenario_id != expected_scenario_id:
        raise ValueError(
            f"Scenario mismatch: expected {expected_scenario_id!r}, "
            f"found {scenario_id!r}."
        )

    # Validate artifact identifier
    artifact_id = manifest["artifact_id"]
    validate_identifier(artifact_id, "artifact_id")
    if expected_artifact_id is not None and artifact_id != expected_artifact_id:
        raise ValueError(
            f"Artifact mismatch: expected {expected_artifact_id!r}, "
            f"found {artifact_id!r}."
        )

    # Validate compatibility and generation recipe
    for field_name in ("compatibility", "generation_recipe"):
        if not isinstance(manifest[field_name], dict):
            raise ValueError(f"Asset manifest `{field_name}` must be an object.")
    compatibility = manifest["compatibility"]
    if (
        expected_compatibility is not None
        and canonical_json(compatibility)
        != canonical_json(dict(expected_compatibility))
    ):
        raise ValueError("Asset compatibility metadata does not match.")

    # Validate files
    files = manifest["files"]
    if not isinstance(files, list) or not files:
        raise ValueError("Asset manifest `files` must be a nonempty list.")
    relative_paths = [_validate_file_record(record) for record in files]
    if len(set(relative_paths)) != len(relative_paths):
        raise ValueError("Asset manifest contains duplicate file paths.")
    if asset_dir is not None:
        for record, relative_path in zip(files, relative_paths):
            payload_path = Path(asset_dir) / relative_path
            if not payload_path.is_file():
                raise ValueError(f"Asset payload is missing: {relative_path.as_posix()}.")
            actual = {
                "size_bytes": payload_path.stat().st_size,
                "sha256": sha256_file(payload_path)
            }
            if actual["size_bytes"] != record["size_bytes"]:
                raise ValueError(
                    f"Asset payload size mismatch: {relative_path.as_posix()}."
                )
            if actual["sha256"] != record["sha256"]:
                raise ValueError(
                    f"Asset payload checksum mismatch: {relative_path.as_posix()}."
                )

    # Validate optional metadata
    metadata = manifest.get("metadata")
    if metadata is not None and not isinstance(metadata, dict):
        raise ValueError("Asset manifest `metadata` must be an object.")

    return manifest


def external_asset_dir(
    cfg: SystemConfig,
    paper_slug: str,
    asset_type: str,
    artifact_id: str,
    *,
    asset_root: Union[str, Path] = "assets",
) -> Path:
    r"""Return an external directory for one persistent asset.

    The returned path has the form::

        <asset_root>/<scenario_id>/<paper_slug>/<asset_type>/<artifact_id>/

    Identifiers are validated, but the directory is not created.

    :param cfg: System configuration that provides the scenario identifier.
    :param paper_slug: Component that owns the asset, such as a paper slug or
        ``builtin``.
    :param asset_type: Type-specific schema identifier, such as
        ``spectral-gmm``.
    :param artifact_id: Human-readable identity of the selected asset.
    :param asset_root: Root directory beneath which assets are stored.

    :returns: External asset directory.
    """
    validate_identifier(paper_slug, "paper_slug")
    validate_identifier(asset_type, "asset_type")
    validate_identifier(artifact_id, "artifact_id")
    return (
        Path(asset_root) / cfg.scenario_id / paper_slug / asset_type / artifact_id
    )


def paper_asset_dir(
    cfg: SystemConfig,
    asset_type: str,
    artifact_id: str,
    module_file: str | Path,
) -> Path:
    r"""Return a paper's optional package-local asset directory:

        <paper package>/assets/<scenario_id>/<asset_type>/<artifact_id>/

    The returned directory is not created.

    :param cfg: System configuration that provides the scenario identifier.
    :param asset_type: Type-specific schema identifier.
    :param artifact_id: Human-readable identity of the selected asset.
    :param module_file: ``__file__`` of the paper-local asset module.

    :returns: Selected package-local asset directory.
    """
    validate_identifier(asset_type, "asset_type")
    validate_identifier(artifact_id, "artifact_id")
    return (
        Path(module_file).resolve().parent
        / "assets"
        / cfg.scenario_id
        / asset_type
        / artifact_id
    )

def _numpy_file_record(path: str | Path) -> dict[str, Any]:
    """Describe one NPY or NPZ payload, including array shapes and dtypes."""
    path = Path(path)
    record: dict[str, Any] = {"path": path.name, "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
    if path.suffix == ".npy":
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        record.update({
            "format": "npy",
            "array": {"shape": list(array.shape), "dtype": str(array.dtype)},
        })
    elif path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as arrays:
            record.update({
                "format": "npz",
                "arrays": {
                    name: {
                        "shape": list(arrays[name].shape),
                        "dtype": str(arrays[name].dtype),
                    }
                    for name in arrays.files
                },
            })
    else:
        raise ValueError(f"Unsupported NumPy asset format: {path.suffix!r}.")
    return record


def _validate_file_record(record: Any) -> Path:
    """Validate one common manifest file record and return its relative path."""
    if not isinstance(record, dict):
        raise ValueError("Every asset manifest file entry must be an object.")
    relative_path = Path(record.get("path", ""))
    if (
        not relative_path.parts
        or relative_path.is_absolute()
        or ".." in relative_path.parts
    ):
        raise ValueError("Asset manifest file paths must be relative and contained.")
    size_bytes = record.get("size_bytes")
    if isinstance(size_bytes, bool) or not isinstance(size_bytes, int):
        raise ValueError("Asset manifest file sizes must be integers.")
    if size_bytes < 0:
        raise ValueError("Asset manifest file sizes must be non-negative.")
    sha256 = record.get("sha256")
    if not isinstance(sha256, str) or _SHA256_PATTERN.fullmatch(sha256) is None:
        raise ValueError("Asset manifest file checksums must be lowercase SHA-256.")
    return relative_path
