# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for shared serialization, hashing, and atomic file persistence."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from sionna.bench.mimo_ofdm_chest import persistence
from sionna.bench.mimo_ofdm_chest.persistence import (
    canonical_json,
    sha256_file,
    sha256_fingerprint,
    write_json_atomic,
    write_npy_atomic,
    write_npz_atomic,
)


def test_canonical_json_and_fingerprint_are_stable() -> None:
    """Mapping order does not affect canonical path encoding or fingerprints."""
    first = {
        "nested": {"second": 2, "first": 1},
        "asset_path": Path("assets/model.npy"),
    }
    second = {
        "asset_path": Path("assets/model.npy"),
        "nested": {"first": 1, "second": 2},
    }
    expected_json = (
        '{"asset_path":"assets/model.npy",'
        '"nested":{"first":1,"second":2}}'
    )
    expected_hash = hashlib.sha256(expected_json.encode("utf-8")).hexdigest()

    assert canonical_json(first) == expected_json
    assert canonical_json(second) == expected_json
    assert sha256_fingerprint(first) == expected_hash
    assert sha256_fingerprint(second) == expected_hash


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_canonical_json_rejects_nonfinite_numbers(value: float) -> None:
    """Canonical fingerprints cannot contain non-finite floating-point values."""
    with pytest.raises(ValueError, match="finite floats"):
        canonical_json({"value": value})


def test_sha256_file_matches_known_digest(tmp_path: Path) -> None:
    """File hashing returns the standard digest for a known byte sequence."""
    path = tmp_path / "payload.bin"
    path.write_bytes(b"abc")

    assert sha256_file(path) == (
        "ba7816bf8f01cfea414140de5dae2223"
        "b00361a396177a9cb410ff61f20015ad"
    )


@pytest.mark.parametrize("file_format", ["json", "npy", "npz"])
def test_atomic_replacement_failure_preserves_existing_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    file_format: str,
) -> None:
    """A failed replacement preserves the destination and removes its temporary."""
    path = tmp_path / f"payload.{file_format}"
    if file_format == "json":
        write_json_atomic(path, {"state": "old"})
    elif file_format == "npy":
        write_npy_atomic(path, np.array([1], dtype=np.int64))
    else:
        write_npz_atomic(path, state=np.array([1], dtype=np.int64))
    original_payload = path.read_bytes()

    def fail_replace(source: str | Path, destination: str | Path) -> None:
        del source, destination
        raise OSError("injected replacement failure")

    monkeypatch.setattr(persistence.os, "replace", fail_replace)
    with pytest.raises(OSError, match="injected replacement failure"):
        if file_format == "json":
            write_json_atomic(path, {"state": "new"})
        elif file_format == "npy":
            write_npy_atomic(path, np.array([2], dtype=np.int64))
        else:
            write_npz_atomic(path, state=np.array([2], dtype=np.int64))

    assert path.read_bytes() == original_payload
    assert list(tmp_path.glob(f".{path.name}.*.tmp")) == []
