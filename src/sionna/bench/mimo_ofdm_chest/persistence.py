# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared utilities for canonical serialization and atomic persistence.

This module provides identifier validation, deterministic JSON serialization
and fingerprinting, and atomic writers for JSON and NumPy payloads.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import tempfile
import os
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "canonical_json",
    "sha256_fingerprint",
    "sha256_file",
    "validate_identifier",
    "write_json_atomic",
    "write_npy_atomic",
    "write_npz_atomic",
]


def sha256_fingerprint(value: Any) -> str:
    """Return the full lowercase SHA-256 digest of canonical JSON data."""
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of a file."""
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_identifier(value: str, field_name: str) -> None:
    """Validate one human-readable path identifier."""
    if (
        not isinstance(value, str)
        or re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", value) is None
    ):
        raise ValueError(
            f"`{field_name}` must contain lowercase ASCII words separated by "
            f"hyphens, but received {value!r}."
        )


def canonical_json(value: Any) -> str:
    """Serialize a configuration value using the repository's canonical JSON."""
    return json.dumps(
        _canonical_value(value),
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=False,
        separators=(",", ":"),
    )


def _canonical_value(value: Any) -> Any:
    """Convert supported configuration values to canonical JSON values."""
    if isinstance(value, dict):
        return {
            str(key): _canonical_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_canonical_value(item) for item in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Configuration fingerprints require finite floats.")
        return value
    if isinstance(value, Path):
        return value.as_posix()
    raise TypeError(
        "Unsupported value in canonical configuration JSON: "
        f"{type(value).__name__}."
    )


def write_json_atomic(path: str | Path, data: Any) -> None:
    r"""Atomically write JSON through a temporary sibling file.

    The parent directory is created when needed. The temporary file is flushed
    and synchronized before replacing the destination, and is removed if the
    write or replacement fails.

    :param path: Destination JSON path.
    :param data: JSON-serializable value. Non-finite numbers are rejected.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=2, sort_keys=False, allow_nan=False)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def write_npy_atomic(path: str | Path, array: np.ndarray) -> None:
    r"""Atomically write one NPY payload through a temporary sibling.

    :param path: Destination NPY path.
    :param array: NumPy array to persist.
    """
    _write_numpy_atomic(path, lambda file: np.save(file, array))


def write_npz_atomic(path: str | Path, **arrays: Any) -> None:
    r"""Atomically write named arrays as an uncompressed NPZ payload.

    :param path: Destination NPZ path.
    :param arrays: Named arrays to persist in the archive.
    """
    _write_numpy_atomic(path, lambda file: np.savez(file, **arrays))


def _write_numpy_atomic(path: str | Path, writer: Any) -> None:
    """Write a NumPy payload through a temporary sibling."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "wb") as file:
            writer(file)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)
