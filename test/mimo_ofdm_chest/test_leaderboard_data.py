# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for scenario-local leaderboard result discovery."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import leaderboard.mimo_ofdm_chest.generate_data as leaderboard_data
from leaderboard.mimo_ofdm_chest.generate_data import (
    _discover_output_root,
    generate_data,
)


def _write_result(path: Path) -> None:
    """Write one minimal finite leaderboard result."""
    np.savez(
        path,
        ebno_dbs=np.array([0.0]),
        nmse=np.array([0.1]),
        ber=np.array([0.01]),
        bler=np.array([0.02]),
    )


def _write_result_tree(root: Path, *, smoke: bool = False) -> Path:
    """Write one scenario-local result inventory and its listed payload."""
    result_dir = root / "smoke" / "3gpp-umi" if smoke else root / "3gpp-umi"
    result_dir.mkdir(parents=True)
    config = {
        "scenario_id": "3gpp-umi",
        "config_fingerprint": "fingerprint",
    }
    (result_dir / "config.json").write_text(json.dumps(config))
    inventory = {
        "results_schema_version": 1,
        **config,
        "results": {
            "Listed estimator": {
                "file": "listed.npz",
            }
        },
    }
    (result_dir / "results.json").write_text(json.dumps(inventory))
    _write_result(result_dir / "listed.npz")
    return result_dir


def test_leaderboard_uses_inventory_and_ignores_unlisted_results(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Only entries listed by results.json appear in the leaderboard."""
    monkeypatch.setattr(leaderboard_data, "ROOT", tmp_path)
    result_dir = _write_result_tree(tmp_path)
    _write_result(result_dir / "stale.npz")

    scenarios = _discover_output_root(
        tmp_path,
        source="paper",
        paper_slug="paper",
    )

    assert [run["estimator"] for run in scenarios["3gpp-umi"]["runs"]] == [
        "Listed estimator"
    ]
    assert "stale.npz" in capsys.readouterr().out


def test_leaderboard_rejects_missing_listed_result(tmp_path: Path) -> None:
    """An inventory cannot advertise a missing result payload."""
    result_dir = _write_result_tree(tmp_path)
    (result_dir / "listed.npz").unlink()

    with pytest.raises(ValueError, match="Missing inventory-listed result"):
        _discover_output_root(tmp_path, source="paper", paper_slug="paper")


@pytest.mark.parametrize(
    "field, value, match",
    [
        ("results_schema_version", 2, "schema version"),
        ("scenario_id", "another-scenario", "scenario_id"),
        ("config_fingerprint", "different", "config_fingerprint"),
    ],
)
def test_leaderboard_rejects_invalid_inventory_identity(
    tmp_path: Path,
    field: str,
    value: object,
    match: str,
) -> None:
    """The result schema and neighboring configuration identity must agree."""
    result_dir = _write_result_tree(tmp_path)
    inventory_path = result_dir / "results.json"
    inventory = json.loads(inventory_path.read_text())
    inventory[field] = value
    inventory_path.write_text(json.dumps(inventory))

    with pytest.raises(ValueError, match=match):
        _discover_output_root(tmp_path, source="paper", paper_slug="paper")


def test_leaderboard_excludes_smoke_results(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Smoke inventories do not contribute to canonical leaderboard scenarios."""
    monkeypatch.setattr(leaderboard_data, "ROOT", tmp_path)
    _write_result_tree(tmp_path)
    _write_result_tree(tmp_path, smoke=True)

    scenarios = _discover_output_root(
        tmp_path,
        source="paper",
        paper_slug="paper",
    )

    assert set(scenarios) == {"3gpp-umi"}
    assert len(scenarios["3gpp-umi"]["runs"]) == 1


def test_generate_data_accepts_an_alternate_output_root(tmp_path: Path) -> None:
    """Dashboard data can be generated from results outside the repository."""
    output_root = tmp_path / "results"
    _write_result_tree(output_root / "paper")
    output_path = tmp_path / "dashboard" / "data.json"

    data = generate_data(output_root, output_path)

    assert output_path.is_file()
    assert data["scenarios"][0]["runs"][0]["estimator"] == "Listed estimator"
