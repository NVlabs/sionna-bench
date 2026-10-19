# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""Store benchmark results and their configuration metadata.

Each canonical scenario directory has the structure::

    <output_dir>/<paper_slug>/<scenario_id>/
        config.json
        results.json
        <result_name>.npz

Smoke results use::

    <output_dir>/<paper_slug>/smoke/<scenario_id>/
        config.json
        results.json
        <result_name>.npz

``config.json`` records the estimator-independent system and evaluation
configuration shared by every result in the directory::

    {
        "config_schema_version": 4,
        "scenario_id": "...",
        "system": {
            "ofdm": {...},
            "pilot_pattern": {...},
            "multi_user": {...},
            "arrays": {...},
            "mcs": {...}
        },
        "channel_source": {
            "type": "...",
            "parameters": {...}
        },
        "evaluation": {
            "snr_db_start": ...,
            "snr_db_stop": ...,
            "snr_db_step": ...,
            "batch_size": ...,
            "num_mc_batches": ...,
            "setup_seed": ...,
            "evaluation_seed": ...
        },
        "numeric": {
            "precision": "..."
        },
        "runtime": {
            "device": "...",
            "output_dir": "..."
        },
        "config_fingerprint": "..."
    }

``config_fingerprint`` is computed from all fields except ``runtime`` and the
fingerprint itself. It prevents results generated with incompatible system or
evaluation configurations from sharing one scenario directory.

``results.json`` records one entry for every completed result file::

    {
        "results_schema_version": 1,
        "scenario_id": "...",
        "config_fingerprint": "...",
        "results": {
            "<result_name>": {
                "file": "<result_name>.npz",
                "estimator_class": "package.module.ClassName",
                "constructor_arguments": {...},
                "assets": [
                    {
                        "asset_type": "...",
                        "artifact_id": "...",
                        "manifest_sha256": "..."
                    }
                ],
                "updated_at": "..."
            }
        }
    }

``constructor_arguments`` records JSON-serializable estimator settings that are
not already represented by ``config.json``. Prepared tensors and other
persistent payloads are represented by references to their asset manifests
rather than embedded in ``results.json``.

The ``.npz`` payloads contain the metric arrays produced by the shared
evaluation pipeline. Result metadata is written only after the corresponding
payload exists.
"""

from __future__ import annotations

import inspect
import json
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, Optional

from ..asset_store import load_asset_manifest
from ..configs import SystemConfig
from ..persistence import (
    canonical_json,
    sha256_file,
    validate_identifier,
    write_json_atomic,
)

__all__ = [
    "prepare_paper_result_dir",
    "record_result",
    "resolve_callable_arguments",
    "resolve_constructor_arguments",
]


def resolve_callable_arguments(
    callable_object: Callable[..., Any],
    argument_names: Iterable[str],
    overrides: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    r"""Resolve selected callable arguments from overrides and defaults.

    This helper keeps public function defaults as the single source of truth
    when tuning or validation code needs their effective values.

    :param callable_object: Callable whose signature defines the defaults.
    :param argument_names: Arguments to resolve.
    :param overrides: Explicit argument overrides. An explicit `None` is
        retained; callers should omit CLI values that were not provided.

    :returns: Resolved argument mapping in the requested order.

    :raises ValueError: If an argument is unknown, has no default and no
        override, or an override was not selected for recording.
    """
    parameters = inspect.signature(callable_object).parameters
    selected_names = tuple(argument_names)
    override_values = dict(overrides or {})

    unselected = set(override_values).difference(selected_names)
    if unselected:
        raise ValueError(
            "Callable overrides were not selected for recording or resolving: "
            f"{sorted(unselected)}."
        )

    resolved: dict[str, Any] = {}
    for name in selected_names:
        parameter = parameters.get(name)
        if parameter is None or name == "self":
            raise ValueError(
                f"{name!r} is not an argument of "
                f"{callable_object.__qualname__}."
            )
        if name in override_values:
            resolved[name] = override_values[name]
        elif parameter.default is not inspect.Parameter.empty:
            resolved[name] = parameter.default
        else:
            raise ValueError(
                f"Argument {name!r} of {callable_object.__qualname__} has no "
                "default or override."
            )
    return resolved


def resolve_constructor_arguments(
    estimator_class: type,
    argument_names: Iterable[str],
    overrides: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    r"""Resolve selected estimator arguments from overrides and defaults.

    This helper keeps constructor defaults as the single source of truth while
    allowing benchmark runners to record the effective algorithmic settings in
    ``results.json``. Callers explicitly select the behavior-relevant argument
    names; prepared tensors, runtime placement, and other non-result metadata
    must not be included.

    :param estimator_class: Estimator class whose constructor defines defaults.
    :param argument_names: Constructor arguments to resolve and record.
    :param overrides: Explicit constructor overrides. An explicit `None` is
        retained; callers should omit CLI values that were not provided.

    :returns: Resolved argument mapping in the requested order.

    :raises ValueError: If an argument is unknown, has no default and no
        override, or an override was not selected for recording.
    """
    return resolve_callable_arguments(
        estimator_class.__init__,
        argument_names,
        overrides,
    )


def prepare_paper_result_dir(
    cfg: SystemConfig,
    paper_slug: str,
    smoke: bool = False,
) -> str:
    r"""Prepare and return one paper's scenario result directory.

    Canonical results are stored at::

        <output_dir>/<paper_slug>/<scenario_id>/

    With ``smoke=True``, results are instead stored at::

        <output_dir>/<paper_slug>/smoke/<scenario_id>/

    Preparing a directory means the following:
    The directory is created if necessary and receives a normalized
    ``config.json`` snapshot. An existing directory is accepted only when its
    snapshot has the same scenario and configuration fingerprint. A nonempty
    directory without a snapshot is never claimed implicitly.

    :param cfg: System configuration that defines the scenario and snapshot.
    :param paper_slug: Stable paper identifier used as the directory name.
    :param smoke: If `True`, select the smoke-result tree.

    :returns: Prepared scenario result directory as a string path.
    """
    if not isinstance(smoke, bool):
        raise TypeError("`smoke` must be a boolean.")
    validate_identifier(paper_slug, "paper_slug")

    directory = os.path.join(
        _paper_root(cfg, paper_slug, smoke=smoke),
        cfg.scenario_id,
    )

    # Validate and atomically write the normalized result configuration.
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, "config.json")
    cfg_snapshot = cfg.config_snapshot()
    if os.path.exists(path):
        existing = _load_json(path)
        existing_scenario = existing.get("scenario_id")
        existing_fingerprint = existing.get("config_fingerprint")
        if (
            existing_scenario != cfg.scenario_id
            or existing_fingerprint != cfg_snapshot["config_fingerprint"]
        ):
            raise ValueError(
                f"Result directory {directory!r} already contains an "
                "incompatible config.json. Refusing to overwrite scenario "
                f"{cfg.scenario_id!r}."
            )
    else:
        existing_payloads = [
            name for name in os.listdir(directory) if name != "config.json.tmp"
        ]
        if existing_payloads:
            raise ValueError(
                f"Result directory {directory!r} contains files but no "
                "config.json; refusing to claim it for a configuration."
            )
    write_json_atomic(path, cfg_snapshot)

    return directory


def record_result(
    cfg: SystemConfig,
    paper_slug: str,
    estimator_class: Optional[type],
    *,
    result_name: Optional[str] = None,
    constructor_arguments: Optional[Mapping[str, Any]] = None,
    asset_manifests: Iterable[str | Path] = (),
    smoke: bool = False,
) -> None:
    r"""Record one evaluated estimator in the scenario-local results file.

    Results are recorded under::

        <output_dir>/<paper_slug>/<scenario_id>/results.json

    or, for smoke runs::

        <output_dir>/<paper_slug>/smoke/<scenario_id>/results.json

    ``constructor_arguments`` contains JSON-serializable estimator parameters.
    Prepared tensors such as model weights and covariance matrices are
    represented through ``asset_manifests`` instead.

    :param cfg: System configuration used for the evaluation.
    :param paper_slug: Stable paper identifier.
    :param result_name: Result display name and ``.npz`` filename stem.
    :param estimator_class: Concrete estimator class, or `None` for references
        such as perfect CSI.
    :param constructor_arguments: Persistable estimator constructor arguments.
        Runtime arguments already recorded by ``cfg``, such as device and
        precision, may be omitted.
    :param asset_manifests: Asset-manifest paths used by the estimator.
    :param smoke: If `True`, record a smoke result.
    """

    directory = Path(prepare_paper_result_dir(cfg, paper_slug, smoke=smoke))
    if result_name is None:
        if estimator_class is None:
            raise ValueError(
                "`result_name` is required when no estimator class is provided."
            )
        result_name = estimator_class.__name__
    result_filename = f"{result_name}.npz"
    result_path = directory / result_filename

    if not result_path.is_file():
        raise FileNotFoundError(f"Result file does not exist: {result_path}")

    path = directory / "results.json"
    data = _load_json(path)

    cfg_fingerprint = cfg.config_fingerprint()
    if data:
        if data.get("results_schema_version") != 1:
            raise ValueError(f"Invalid results schema version in {path}.")
        if data.get("scenario_id") != cfg.scenario_id:
            raise ValueError(f"Scenario mismatch in {path}.")
        if data.get("config_fingerprint") != cfg_fingerprint:
            raise ValueError(f"Configuration mismatch in {path}.")
        if not isinstance(data.get("results"), dict):
            raise ValueError(f"Invalid result entries in {path}.")
    else:
        data = {
            "results_schema_version": 1,
            "scenario_id": cfg.scenario_id,
            "config_fingerprint": cfg_fingerprint,
            "results": {},
        }

    class_name = None
    if estimator_class is not None:
        class_name = (
            f"{estimator_class.__module__}."
            f"{estimator_class.__qualname__}"
        )

    # Canonicalization also verifies that the arguments are persistable.
    arguments = json.loads(canonical_json(dict(constructor_arguments or {})))
    assets = [
        _asset_reference_from_manifest(manifest_path, cfg)
        for manifest_path in asset_manifests
    ]

    data["results"][result_name] = {
        "file": result_filename,
        "estimator_class": class_name,
        "constructor_arguments": arguments,
        "assets": assets,
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    write_json_atomic(path, data)


def _asset_reference_from_manifest(
    manifest_path: str | Path,
    cfg: SystemConfig,
) -> dict[str, str]:
    """Load and validate an asset manifest and return the required fields."""
    manifest_path = Path(manifest_path)

    manifest = load_asset_manifest(
        manifest_path,
        asset_dir=manifest_path.parent,
        expected_scenario_id=cfg.scenario_id,
    )

    return {
        "asset_type": manifest["asset_type"],
        "artifact_id": manifest["artifact_id"],
        "manifest_sha256": sha256_file(manifest_path),
    }

def _paper_root(
    cfg: SystemConfig,
    paper_slug: str,
    smoke: bool = False,
) -> str:
    """Return the canonical or smoke output root for a paper."""
    root = os.path.join(cfg.output_dir, paper_slug)
    return os.path.join(root, "smoke") if smoke else root


def _load_json(path: str) -> Dict:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}
