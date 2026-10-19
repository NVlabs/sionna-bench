# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for system-configuration identity and benchmark-result storage."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from sionna.bench.mimo_ofdm_chest.configs import SystemConfig, umi_3gpp_config
from sionna.bench.mimo_ofdm_chest.asset_store import build_asset_manifest
from sionna.bench.mimo_ofdm_chest.benchmarking.result_store import (
    prepare_paper_result_dir,
    record_result,
    resolve_callable_arguments,
    resolve_constructor_arguments,
)
from sionna.bench.mimo_ofdm_chest.persistence import sha256_file, write_json_atomic, write_npy_atomic


class ExampleEstimator:
    """Minimal estimator type used to verify persisted class names."""

    def __init__(self, rank: int, scale: float = 0.5) -> None:
        self.rank = rank
        self.scale = scale


def test_resolve_constructor_arguments_uses_defaults_and_overrides() -> None:
    """Selected constructor values come from defaults or explicit overrides."""
    assert resolve_constructor_arguments(
        ExampleEstimator,
        ("rank", "scale"),
        {"rank": 4},
    ) == {"rank": 4, "scale": 0.5}

    with pytest.raises(ValueError, match="no default or override"):
        resolve_constructor_arguments(ExampleEstimator, ("rank",))
    with pytest.raises(ValueError, match="not selected for recording"):
        resolve_constructor_arguments(
            ExampleEstimator,
            ("scale",),
            {"rank": 4},
        )


def test_resolve_callable_arguments_uses_function_defaults() -> None:
    """Training settings are read directly from the public function signature."""

    def train_model(learning_rate: float = 1e-3, batch_size: int = 256) -> None:
        pass

    assert resolve_callable_arguments(
        train_model,
        ("learning_rate", "batch_size"),
    ) == {"learning_rate": 1e-3, "batch_size": 256}


def _config(tmp_path: Path, **changes: object) -> SystemConfig:
    """Return the canonical configuration with isolated result output."""
    return replace(
        umi_3gpp_config,
        output_dir=str(tmp_path / "outputs"),
        **changes,
    )


def _write_result_file(
    cfg: SystemConfig,
    paper_slug: str,
    result_name: str,
    *,
    smoke: bool = False,
) -> Path:
    """Write a minimal result payload where the result store expects it."""
    directory = Path(prepare_paper_result_dir(cfg, paper_slug, smoke=smoke))
    path = directory / f"{result_name}.npz"
    np.savez(path, nmse=np.array([0.1], dtype=np.float64))
    return path


def test_config_fingerprint_tracks_reproducible_inputs_only(tmp_path: Path) -> None:
    """Runtime changes are excluded while system and evaluation changes count."""
    baseline = _config(tmp_path)
    runtime = replace(
        baseline,
        name="another-name",
        device="cpu",
        output_dir="another-output-root",
    )
    system = replace(baseline, num_prb=baseline.num_prb + 1)
    channel = replace(
        baseline,
        channel_source_parameters={"spec_version": "16.1"},
    )
    evaluation = replace(baseline, num_mc_batches=baseline.num_mc_batches + 1)

    assert runtime.config_fingerprint() == baseline.config_fingerprint()
    assert system.config_fingerprint() != baseline.config_fingerprint()
    assert channel.config_fingerprint() != baseline.config_fingerprint()
    assert evaluation.config_fingerprint() != baseline.config_fingerprint()


def test_prepare_result_dir_uses_canonical_and_smoke_layouts(tmp_path: Path) -> None:
    """Canonical and smoke runs receive separate normalized snapshots."""
    cfg = _config(tmp_path)

    canonical = Path(prepare_paper_result_dir(cfg, "delfeld2025"))
    smoke = Path(prepare_paper_result_dir(cfg, "delfeld2025", smoke=True))

    assert canonical == tmp_path / "outputs" / "delfeld2025" / "3gpp-umi"
    assert smoke == tmp_path / "outputs" / "delfeld2025" / "smoke" / "3gpp-umi"
    with (canonical / "config.json").open(encoding="utf-8") as file:
        snapshot = json.load(file)
    assert snapshot["config_schema_version"] == 4
    assert snapshot["config_fingerprint"] == cfg.config_fingerprint()


def test_prepare_result_dir_rejects_invalid_public_arguments(tmp_path: Path) -> None:
    """The result path accepts only boolean smoke flags and safe paper slugs."""
    cfg = _config(tmp_path)
    with pytest.raises(TypeError, match="smoke.*boolean"):
        prepare_paper_result_dir(  # type: ignore[arg-type]
            cfg,
            "delfeld2025",
            smoke="smoke",
        )
    with pytest.raises(ValueError, match="paper_slug"):
        prepare_paper_result_dir(cfg, "Delfeld 2025")


@pytest.mark.parametrize("failure", ["unclaimed", "incompatible"])
def test_prepare_result_dir_rejects_occupied_incompatible_path(
    tmp_path: Path,
    failure: str,
) -> None:
    """Existing output cannot be claimed without a matching configuration."""
    cfg = _config(tmp_path)
    directory = tmp_path / "outputs" / "delfeld2025" / "3gpp-umi"
    if failure == "unclaimed":
        directory.mkdir(parents=True)
        (directory / "legacy.npz").write_bytes(b"legacy")
        match = "files but no config.json"
    else:
        prepare_paper_result_dir(cfg, "delfeld2025")
        cfg = replace(cfg, num_mc_batches=cfg.num_mc_batches + 1)
        match = "incompatible config.json"

    with pytest.raises(ValueError, match=match):
        prepare_paper_result_dir(cfg, "delfeld2025")


def test_record_result_supports_default_explicit_and_reference_names(
    tmp_path: Path,
) -> None:
    """Result entries are incremental and canonicalize constructor arguments."""
    cfg = _config(tmp_path)
    paper_slug = "delfeld2025"
    for result_name in ("ExampleEstimator", "RKHS tuned", "Perfect CSI"):
        _write_result_file(cfg, paper_slug, result_name)

    record_result(
        cfg,
        paper_slug,
        ExampleEstimator,
        constructor_arguments={"rank": 5, "path": Path("assets/model.npy")},
    )
    record_result(
        cfg,
        paper_slug,
        ExampleEstimator,
        result_name="RKHS tuned",
    )
    record_result(
        cfg,
        paper_slug,
        None,
        result_name="Perfect CSI"
    )

    directory = Path(prepare_paper_result_dir(cfg, paper_slug))
    with (directory / "results.json").open(encoding="utf-8") as file:
        results = json.load(file)
    entries = results["results"]
    assert results["results_schema_version"] == 1
    assert set(entries) == {"ExampleEstimator", "RKHS tuned", "Perfect CSI"}
    assert entries["ExampleEstimator"]["estimator_class"].endswith(
        ".ExampleEstimator"
    )
    assert entries["ExampleEstimator"]["constructor_arguments"] == {
        "path": "assets/model.npy",
        "rank": 5,
    }
    assert entries["Perfect CSI"]["estimator_class"] is None
    assert not (directory.parent / "manifest.json").exists()


def test_record_result_rejects_missing_payload(tmp_path: Path) -> None:
    """Metadata cannot advertise a result payload that does not exist."""
    cfg = _config(tmp_path)
    with pytest.raises(FileNotFoundError, match="Result file does not exist"):
        record_result(cfg, "delfeld2025", ExampleEstimator)


@pytest.mark.parametrize(
    "field, value, match",
    [
        ("results_schema_version", 2, "schema version"),
        ("config_fingerprint", "0" * 64, "Configuration mismatch"),
    ],
)
def test_record_result_rejects_incompatible_results_file(
    tmp_path: Path,
    field: str,
    value: object,
    match: str,
) -> None:
    """Existing result inventories must retain their schema and configuration."""
    cfg = _config(tmp_path)
    payload = _write_result_file(cfg, "delfeld2025", "ExampleEstimator")
    results = {
        "results_schema_version": 1,
        "scenario_id": cfg.scenario_id,
        "config_fingerprint": cfg.config_fingerprint(),
        "results": {},
    }
    results[field] = value
    write_json_atomic(payload.parent / "results.json", results)

    with pytest.raises(ValueError, match=match):
        record_result(cfg, "delfeld2025", ExampleEstimator)


def test_record_result_reduces_validated_asset_manifest(tmp_path: Path) -> None:
    """Results retain only asset identity and the exact manifest digest."""
    cfg = _config(tmp_path)
    asset_dir = tmp_path / "asset"
    payload_path = asset_dir / "model.npy"
    write_npy_atomic(payload_path, np.ones(2, dtype=np.float32))
    manifest = build_asset_manifest(
        cfg=cfg,
        asset_type="rkhs-model",
        artifact_id="canonical",
        compatibility={"rank": 5},
        generation_recipe={"seed": 7},
        payload_paths=[payload_path],
    )
    manifest_path = asset_dir / "manifest.json"
    write_json_atomic(manifest_path, manifest)
    result_path = _write_result_file(cfg, "delfeld2025", "ExampleEstimator")

    record_result(
        cfg,
        "delfeld2025",
        ExampleEstimator,
        asset_manifests=[manifest_path],
    )

    with (result_path.parent / "results.json").open(encoding="utf-8") as file:
        entry = json.load(file)["results"]["ExampleEstimator"]
    assert entry["assets"] == [
        {
            "asset_type": "rkhs-model",
            "artifact_id": "canonical",
            "manifest_sha256": sha256_file(manifest_path),
        }
    ]

    invalid_manifest = asset_dir / "missing.json"
    with pytest.raises(FileNotFoundError):
        record_result(
            cfg,
            "delfeld2025",
            ExampleEstimator,
            asset_manifests=[invalid_manifest],
        )
