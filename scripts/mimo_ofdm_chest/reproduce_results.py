# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run canonical paper benchmarks through their existing command-line APIs."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class _Run:
    """One canonical paper-runner invocation."""

    paper: str
    arguments: tuple[str, ...] = ()
    has_assets: bool = False


_RUNS = (
    _Run("builtin", ("--estimator", "ls-nn,ls-lin,lmmse"), has_assets=True),
    _Run("coleri2002"),
    _Run("edfors1996", has_assets=True),
    _Run("koller2022", has_assets=True),
    _Run("delfeld2025"),
    _Run("lyu2026"),
    _Run("wei2017"),
    _Run("melgar2022", ("--method", "model2"), has_assets=True),
    _Run("melgar2022", ("--method", "model3"), has_assets=True),
    _Run("li2020", has_assets=True),
    _Run("baur2022", has_assets=True),
    _Run("balevi2020"),
)
_PAPERS = tuple(dict.fromkeys(run.paper for run in _RUNS))


def _parse_args() -> argparse.Namespace:
    """Parse orchestration arguments without duplicating runner settings."""
    parser = argparse.ArgumentParser(
        description=(
            "Prepare assets and reproduce canonical benchmark results by "
            "invoking the existing paper runners."
        )
    )
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument(
        "--all",
        action="store_true",
        help="Run every canonical channel-estimator result.",
    )
    selection.add_argument(
        "--paper",
        action="append",
        choices=_PAPERS,
        help="Paper to run; repeat this option to select several papers.",
    )
    parser.add_argument(
        "--prepare-assets",
        action="store_true",
        help="Prepare every selected persistent asset before evaluation.",
    )
    parser.add_argument(
        "--overwrite-assets",
        action="store_true",
        help="Replace selected asset directories during preparation.",
    )
    parser.add_argument(
        "--asset-root",
        type=Path,
        default=None,
        help="External asset root passed to asset-backed paper runners.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Result root passed to every paper runner.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Execution-device override passed to every paper runner.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them.",
    )
    args = parser.parse_args()
    if args.prepare_assets and args.asset_root is None:
        parser.error("--prepare-assets requires --asset-root")
    if args.overwrite_assets and not args.prepare_assets:
        parser.error("--overwrite-assets requires --prepare-assets")
    return args


def _selected_runs(args: argparse.Namespace) -> tuple[_Run, ...]:
    """Return canonical invocations in stable repository order."""
    if args.all:
        return _RUNS
    selected = set(args.paper)
    return tuple(run for run in _RUNS if run.paper in selected)


def _command(
    run: _Run,
    args: argparse.Namespace,
    *,
    prepare_assets: bool,
) -> list[str]:
    """Build one paper-runner command."""
    command = [
        sys.executable,
        "-m",
        f"sionna.bench.mimo_ofdm_chest.papers.{run.paper}.run_benchmark",
        *run.arguments,
    ]
    if args.output_dir is not None:
        command.extend(("--output-dir", str(args.output_dir)))
    if args.device is not None:
        command.extend(("--device", args.device))
    if run.has_assets and args.asset_root is not None:
        command.extend(("--asset-root", str(args.asset_root)))
    if prepare_assets:
        command.append("--prepare-assets")
        if args.overwrite_assets:
            command.append("--overwrite-assets")
    return command


def _execute(command: list[str], *, dry_run: bool) -> None:
    """Print and optionally execute one command."""
    print(f"+ {shlex.join(command)}", flush=True)
    if not dry_run:
        subprocess.run(command, check=True)


def main() -> None:
    """Prepare selected assets, then evaluate all selected methods."""
    args = _parse_args()
    runs = _selected_runs(args)

    if args.prepare_assets:
        for run in runs:
            if run.has_assets:
                _execute(
                    _command(run, args, prepare_assets=True),
                    dry_run=args.dry_run,
                )

    for run in runs:
        _execute(
            _command(run, args, prepare_assets=False),
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
