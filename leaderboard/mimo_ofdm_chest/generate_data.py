# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate MIMO-OFDM leaderboard data from scenario-local result inventories."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT_PATH = Path(__file__).resolve().parent / "data.json"
METRICS = ("ebno_dbs", "nmse", "ber", "bler")
RESULT_METRICS = ("nmse", "ber", "bler")
EXCLUDED_NAMESPACES = {"smoke", "reduced"}
SCENARIO_LABELS = {
    "3gpp-umi": "3GPP UMi",
}
SCENARIO_ORDER = {
    "3gpp-umi": 0,
}

ESTIMATOR_ORDER = {
    ("edfors1996", "TruncatedChannelEstimator (LMMSE)"): 0,
    ("edfors1996", "TruncatedChannelEstimator (diag. LMMSE)"): 1,
    ("edfors1996", "TruncatedChannelEstimator (LS)"): 2,
}


def _estimator_order_key(run: dict[str, Any]) -> tuple[int, str]:
    paper = run["paper"] or "builtin"
    estimator = run["estimator"]
    return (
        ESTIMATOR_ORDER.get((paper, estimator), 1000),
        estimator,
    )


def _as_snr_list(value: np.ndarray) -> list[float] | None:
    values = np.asarray(value, dtype=np.float64).reshape(-1)
    if values.size == 0 or not np.isfinite(values).all():
        return None
    return [float(x) for x in values]


def _as_metric_list(value: np.ndarray) -> tuple[list[float | None], bool]:
    values = np.asarray(value, dtype=np.float64).reshape(-1)
    sanitized = False
    out: list[float | None] = []
    for item in values:
        if np.isfinite(item):
            out.append(float(item))
        else:
            out.append(None)
            sanitized = True
    return out, sanitized


def _load_npz(path: Path) -> dict[str, list[float | None]] | None:
    try:
        with np.load(path) as data:
            if not all(metric in data.files for metric in METRICS):
                return None
            snr = _as_snr_list(data["ebno_dbs"])
            if snr is None:
                print(f"skip result with invalid ebno_dbs {path}")
                return None

            arrays: dict[str, list[float | None]] = {"ebno_dbs": snr}
            sanitized_metrics: list[str] = []
            for metric in RESULT_METRICS:
                values, sanitized = _as_metric_list(data[metric])
                if len(values) != len(snr):
                    print(
                        f"skip result with mismatched {metric} length {path}: "
                        f"{len(values)} != {len(snr)}"
                    )
                    return None
                arrays[metric] = values
                if sanitized:
                    sanitized_metrics.append(metric)

            if sanitized_metrics:
                print(
                    "sanitize non-finite values in "
                    f"{path}: {', '.join(sanitized_metrics)}"
                )
            return arrays
    except Exception as exc:
        print(f"skip unreadable result {path}: {exc}")
        return None


def _load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open() as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def _namespace_and_config(result_dir: Path, output_root: Path) -> tuple[str, str]:
    rel = result_dir.relative_to(output_root)
    if len(rel.parts) == 1:
        return "default", rel.parts[0]
    return "/".join(rel.parts[:-1]), rel.parts[-1]


def _scenario_label(namespace: str, scenario_id: str) -> str:
    if scenario_id in SCENARIO_LABELS:
        return SCENARIO_LABELS[scenario_id]
    ns = namespace.replace("/", " / ").title()
    return scenario_id if namespace == "default" else f"{ns} - {scenario_id}"


def _discover_output_root(
    output_root: Path,
    *,
    source: str,
    paper_slug: str | None,
) -> dict[str, dict[str, Any]]:
    scenarios: dict[str, dict[str, Any]] = {}
    if not output_root.exists():
        return scenarios

    for results_path in sorted(output_root.rglob("results.json")):
        result_dir = results_path.parent
        namespace, directory_name = _namespace_and_config(result_dir, output_root)
        if namespace in EXCLUDED_NAMESPACES:
            continue

        config_path = result_dir / "config.json"
        config = _load_json(config_path)
        inventory = _load_json(results_path)
        if inventory.get("results_schema_version") != 1:
            raise ValueError(f"Invalid results schema version in {results_path}.")
        for identity_field in ("scenario_id", "config_fingerprint"):
            if inventory.get(identity_field) != config.get(identity_field):
                raise ValueError(
                    f"Result inventory and config disagree on {identity_field} "
                    f"for {result_dir}."
                )

        result_entries = inventory.get("results")
        if not isinstance(result_entries, dict):
            raise ValueError(f"Invalid result inventory in {results_path}.")
        listed_results: list[tuple[str, Path]] = []
        for result_name, entry in result_entries.items():
            filename = entry.get("file") if isinstance(entry, dict) else None
            if not isinstance(filename, str):
                raise ValueError(
                    f"Invalid result filename for {result_name!r} in {results_path}."
                )
            npz_path = result_dir / filename
            if not npz_path.is_file():
                raise ValueError(
                    f"Missing inventory-listed result in {result_dir}: {filename}."
                )
            listed_results.append((result_name, npz_path))

        listed_filenames = {path.name for _, path in listed_results}
        unlisted = {path.name for path in result_dir.glob("*.npz")} - listed_filenames
        if unlisted:
            print(
                f"ignore results not listed by {results_path}: "
                f"{', '.join(sorted(unlisted))}"
            )

        stored_scenario_id = config["scenario_id"]
        scenario_id = stored_scenario_id
        if namespace != "default":
            scenario_id = f"{namespace}::{stored_scenario_id}"
        scenario = scenarios.setdefault(
            scenario_id,
            {
                "id": scenario_id,
                "namespace": namespace,
                "configSlug": directory_name,
                "label": _scenario_label(namespace, stored_scenario_id),
                "config": config,
                "snrDb": [],
                "runs": [],
            },
        )

        for estimator, npz_path in listed_results:
            arrays = _load_npz(npz_path)
            if arrays is None:
                continue
            snr = arrays["ebno_dbs"]
            scenario["snrDb"] = sorted(set(scenario["snrDb"]) | set(snr))
            source_key = paper_slug if paper_slug else "builtin"
            run_id = f"{scenario_id}::{source_key}::{estimator}"
            scenario["runs"].append(
                {
                    "id": run_id,
                    "estimator": estimator,
                    "source": source,
                    "paper": paper_slug,
                    "file": _display_path(npz_path),
                    "ebnoDbs": snr,
                    "nmse": arrays["nmse"],
                    "ber": arrays["ber"],
                    "bler": arrays["bler"],
                }
            )

    return scenarios


def _display_path(path: Path) -> str:
    """Return a repository-relative path when possible."""
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def _merge_scenarios(parts: list[dict[str, dict[str, Any]]]) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for part in parts:
        for scenario_id, scenario in part.items():
            target = merged.setdefault(
                scenario_id,
                {
                    **scenario,
                    "snrDb": [],
                    "runs": [],
                },
            )
            target["snrDb"] = sorted(set(target["snrDb"]) | set(scenario["snrDb"]))
            run_by_id = {run["id"]: run for run in target["runs"]}
            for run in scenario["runs"]:
                run_by_id[run["id"]] = run
            target["runs"] = list(run_by_id.values())

    for scenario in merged.values():
        scenario["runs"].sort(
            key=lambda run: (
                0 if run["source"] == "builtin" else 1,
                run["paper"] or "",
                _estimator_order_key(run),
            )
        )

    return sorted(
        merged.values(),
        key=lambda scenario: (
            SCENARIO_ORDER.get(scenario["id"], len(SCENARIO_ORDER)),
            scenario["namespace"],
            scenario["configSlug"],
        ),
    )


def generate_data(output_root: Path, output_path: Path) -> dict[str, Any]:
    """Generate leaderboard data from one canonical result root."""
    parts: list[dict[str, dict[str, Any]]] = []
    parts.append(
        _discover_output_root(
            output_root / "builtin",
            source="builtin",
            paper_slug=None,
        )
    )

    if output_root.exists():
        for paper_outputs in sorted(
            path for path in output_root.iterdir() if path.is_dir()
        ):
            if paper_outputs.name == "builtin":
                continue
            parts.append(
                _discover_output_root(
                    paper_outputs,
                    source="paper",
                    paper_slug=paper_outputs.name,
                )
            )

    data = {
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "scenarios": _merge_scenarios(parts),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(data, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
    print(f"wrote {_display_path(output_path)}")
    return data


def _parse_args() -> argparse.Namespace:
    """Parse result and dashboard-data paths."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "outputs" / "mimo_ofdm_chest",
        help="Canonical result root to scan (default: outputs/mimo_ofdm_chest).",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=OUT_PATH,
        help="Dashboard JSON to write (default: leaderboard/mimo_ofdm_chest/data.json).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    generate_data(args.output_root, args.output_path)


if __name__ == "__main__":
    main()
