# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Union

from sionna.bench.mimo_ofdm_chest.benchmarking import (
    build_cached_detector_scale_objective,
    build_cached_nmse_objective,
    build_resource_grid,
    tune_channel_estimator,
)
from sionna.bench.mimo_ofdm_chest.benchmarking.result_store import resolve_constructor_arguments
from sionna.bench.mimo_ofdm_chest.configs import umi_3gpp_config
from sionna.bench.mimo_ofdm_chest.persistence import write_json_atomic

from .asset import load_covariance
from .estimator import (
    METHOD_DIAGONAL_LMMSE,
    METHOD_LMMSE,
    METHOD_LS,
    METHOD_NAMES,
    TruncatedChannelEstimator,
)

__all__ = ["run_tuning"]

TUNING_EBNO_DBS = (-10.0, -5.0, 0.0, 5.0, 10.0)
TUNING_SNR_WEIGHTS = tuple(
    10.0 ** (ebno_db / 10.0) for ebno_db in TUNING_EBNO_DBS
)
DETECTOR_CALIBRATION_EBNO_DBS = (10.0,)

ALGORITHM_SAMPLER_SEED = 123
ALGORITHM_PROBE_SEED = 4242
ALGORITHM_VALIDATION_SEED = 314159
VARIANCE_SAMPLER_SEED = 456
VARIANCE_PROBE_SEED = 271828
VARIANCE_VALIDATION_SEED = 161803

TAP_TUNING_OUTPUT_DIR = Path("tuning/mimo_ofdm_chest/edfors1996/taps_v2")
VARIANCE_TUNING_OUTPUT_DIR = Path("tuning/mimo_ofdm_chest/edfors1996/variance_scale_v2")


def _suggest_num_taps(trial, cfg, resource_grid):
    """Suggest a retained-tap count supported by the active pilot vector."""
    active_pilots_per_stream = (
        resource_grid.pilot_pattern.pilots.abs() > 0
    ).sum(dim=-1)
    num_pilot_symbols = len(cfg.pilot_ofdm_symbol_indices)
    max_num_taps = int(active_pilots_per_stream.min().item()) // num_pilot_symbols
    return {"num_taps": trial.suggest_int("num_taps", 1, max_num_taps)}


def _suggest_chest_err_const(trial, cfg, resource_grid):
    """Suggest the shared detector-facing variance scale."""
    del cfg, resource_grid
    return {
        "chest_err_const": trial.suggest_float(
            "chest_err_const",
            1e-3,
            1.0,
            log=True,
        )
    }


def _load_time_covariance(
    asset_root: Union[str, Path],
):
    """Load the frozen canonical covariance used by all Edfors stages."""
    resource_grid = build_resource_grid(umi_3gpp_config)
    covariance, _ = load_covariance(
        resource_grid,
        umi_3gpp_config,
        asset_root=asset_root,
    )
    return resource_grid, covariance


def _load_best_params(output_dir: Union[str, Path]) -> dict[str, object]:
    """Load the raw Optuna winner from one study summary."""
    with (Path(output_dir) / "best_params.json").open(encoding="utf-8") as file:
        return json.load(file)["best_params"]


def _write_validation(
    output_dir: Union[str, Path],
    result: dict[str, object],
) -> dict[str, object]:
    """Write and return one paired-validation result."""
    write_json_atomic(Path(output_dir) / "paired_validation.json", result)
    return result


def _selected_num_taps(tap_output_dir: Union[str, Path]) -> int:
    """Return the tap count selected by paired validation."""
    path = Path(tap_output_dir) / "paired_validation.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"Tap paired validation is required before calibration: {path}."
        )
    with path.open(encoding="utf-8") as file:
        return int(json.load(file)["selected_num_taps"])


def _run_tap_tuning(
    *,
    n_trials: int,
    num_mc_batches: int,
    batch_size: int,
    output_dir: Union[str, Path],
    asset_root: Union[str, Path],
):
    """Tune the LS retained-tap count on one shared cached probe set."""
    resource_grid, covariance = _load_time_covariance(asset_root)
    objective = build_cached_nmse_objective(
        TruncatedChannelEstimator,
        umi_3gpp_config,
        ebno_dbs=TUNING_EBNO_DBS,
        snr_weights=TUNING_SNR_WEIGHTS,
        num_mc_batches=num_mc_batches,
        batch_size=batch_size,
        resource_grid=resource_grid,
        seed=ALGORITHM_PROBE_SEED,
    )
    return tune_channel_estimator(
        TruncatedChannelEstimator,
        umi_3gpp_config,
        _suggest_num_taps,
        output_dir,
        fixed_estimator_kwargs={
            "time_covariance": covariance,
            "method": METHOD_LS,
        },
        ebno_dbs=TUNING_EBNO_DBS,
        snr_weights=TUNING_SNR_WEIGHTS,
        n_trials=n_trials,
        num_mc_batches=num_mc_batches,
        batch_size=batch_size,
        sampler_seed=ALGORITHM_SAMPLER_SEED,
        probe_seed=ALGORITHM_PROBE_SEED,
        estimator_objective=objective,
    )


def _run_tap_validation(
    *,
    num_mc_batches: int,
    batch_size: int,
    output_dir: Union[str, Path],
    asset_root: Union[str, Path],
) -> dict[str, object]:
    """Compare the integer tap-count winner with the current tap count."""
    raw = _load_best_params(output_dir)
    candidates = {
        "current": resolve_constructor_arguments(
            TruncatedChannelEstimator,
            ("num_taps",),
        )["num_taps"],
        "winner": int(raw["num_taps"]),
    }
    resource_grid, covariance = _load_time_covariance(asset_root)
    objective = build_cached_nmse_objective(
        TruncatedChannelEstimator,
        umi_3gpp_config,
        ebno_dbs=TUNING_EBNO_DBS,
        snr_weights=TUNING_SNR_WEIGHTS,
        num_mc_batches=num_mc_batches,
        batch_size=batch_size,
        resource_grid=resource_grid,
        seed=ALGORITHM_VALIDATION_SEED,
    )
    objectives = {
        name: objective(
            {
                "time_covariance": covariance,
                "method": METHOD_LS,
                "num_taps": num_taps,
            }
        )
        for name, num_taps in candidates.items()
    }
    selected = "winner" if objectives["winner"] < objectives["current"] else "current"
    return _write_validation(
        output_dir,
        {
            "protocol_version": 2,
            "metric": "weighted_nmse",
            "method": METHOD_LS,
            "ebno_dbs": list(TUNING_EBNO_DBS),
            "snr_weights": list(TUNING_SNR_WEIGHTS),
            "seed": ALGORITHM_VALIDATION_SEED,
            "num_mc_batches_per_snr": num_mc_batches,
            "batch_size": batch_size,
            "raw_winner": raw,
            "candidates": candidates,
            "objectives": objectives,
            "selected": selected,
            "selected_num_taps": candidates[selected],
        },
    )


def _variance_objective(
    *,
    num_taps: int,
    num_mc_batches: int,
    batch_size: int,
    seed: int,
    asset_root: Union[str, Path],
):
    """Build the equal-method mean-BLER objective for all three methods."""
    resource_grid, covariance = _load_time_covariance(asset_root)
    method_objectives = [
        build_cached_detector_scale_objective(
            TruncatedChannelEstimator,
            umi_3gpp_config,
            {
                "time_covariance": covariance,
                "interpolation_type": "lin_time_avg",
                "method": method,
                "num_taps": num_taps,
            },
            scale_argument="chest_err_const",
            ebno_dbs=DETECTOR_CALIBRATION_EBNO_DBS,
            num_mc_batches=num_mc_batches,
            batch_size=batch_size,
            resource_grid=resource_grid,
            seed=seed,
            detector_metric="bler",
        )
        for method in METHOD_NAMES
    ]

    def objective(estimator_kwargs):
        values = [
            method_objective(estimator_kwargs)
            for method_objective in method_objectives
        ]
        return sum(values) / len(values)

    return objective


def _run_variance_calibration(
    *,
    n_trials: int,
    num_mc_batches: int,
    batch_size: int,
    output_dir: Union[str, Path],
    tap_output_dir: Union[str, Path],
    asset_root: Union[str, Path],
):
    """Calibrate one shared scale against mean 10 dB BLER."""
    num_taps = _selected_num_taps(tap_output_dir)
    return tune_channel_estimator(
        TruncatedChannelEstimator,
        umi_3gpp_config,
        _suggest_chest_err_const,
        output_dir,
        ebno_dbs=DETECTOR_CALIBRATION_EBNO_DBS,
        objective_metric="detector",
        detector_metric="bler",
        n_trials=n_trials,
        num_mc_batches=num_mc_batches,
        batch_size=batch_size,
        sampler_seed=VARIANCE_SAMPLER_SEED,
        probe_seed=VARIANCE_PROBE_SEED,
        estimator_objective=_variance_objective(
            num_taps=num_taps,
            num_mc_batches=num_mc_batches,
            batch_size=batch_size,
            seed=VARIANCE_PROBE_SEED,
            asset_root=asset_root,
        ),
    )


def _run_variance_validation(
    *,
    num_mc_batches: int,
    batch_size: int,
    output_dir: Union[str, Path],
    tap_output_dir: Union[str, Path],
    asset_root: Union[str, Path],
) -> dict[str, object]:
    """Compare the rounded variance winner with the current shared scale."""
    raw = _load_best_params(output_dir)
    candidates = {
        "current": resolve_constructor_arguments(
            TruncatedChannelEstimator,
            ("chest_err_const",),
        )["chest_err_const"],
        "rounded_winner": float(f"{float(raw['chest_err_const']):.3g}"),
    }
    objective = _variance_objective(
        num_taps=_selected_num_taps(tap_output_dir),
        num_mc_batches=num_mc_batches,
        batch_size=batch_size,
        seed=VARIANCE_VALIDATION_SEED,
        asset_root=asset_root,
    )
    objectives = {
        name: objective({"chest_err_const": scale})
        for name, scale in candidates.items()
    }
    selected = (
        "rounded_winner"
        if objectives["rounded_winner"] < objectives["current"]
        else "current"
    )
    return _write_validation(
        output_dir,
        {
            "protocol_version": 2,
            "metric": "mean_codeword_bler",
            "method_weighting": "equal over all three Edfors methods",
            "ebno_dbs": list(DETECTOR_CALIBRATION_EBNO_DBS),
            "seed": VARIANCE_VALIDATION_SEED,
            "num_mc_batches_per_snr": num_mc_batches,
            "batch_size": batch_size,
            "raw_winner": raw,
            "candidates": candidates,
            "objectives": objectives,
            "selected": selected,
            "selected_scale": candidates[selected],
        },
    )


def run_tuning(
    *,
    study: str = "taps",
    n_trials: Optional[int] = None,
    num_mc_batches: Optional[int] = None,
    batch_size: Optional[int] = None,
    output_dir: Optional[Union[str, Path]] = None,
    tap_output_dir: Optional[Union[str, Path]] = None,
    asset_root: Union[str, Path] = "assets",
):
    r"""

    .. rubric:: Algorithmic Tuning

    The ``"taps"`` stage minimizes weighted NMSE for the LS method at
    -10, -5, 0, 5, and 10 dB. It runs 50 seeded TPE trials. Every trial
    evaluates three batches of four channels at each SNR using the frozen
    delay-covariance asset.

    .. list-table::
        :header-rows: 1

        * - Parameter
          - Search space
        * - ``num_taps``
          - integer from 1 through the number of frequency pilots in one
            pilot-bearing OFDM symbol; the canonical maximum is 156

    Only the LS method determines ``num_taps``. The selected value is then
    used unchanged by the LS, diagonal-LMMSE, and full-LMMSE methods.

    The ``"taps-validation"`` stage compares the integer winner with the
    existing tap-count default using 63 batches of four channels per SNR.

    .. rubric:: Error-Variance Calibration

    The ``"variance"`` stage tunes one shared ``chest_err_const``
    logarithmically from 1e-3 through 1 using 10 TPE trials. Its objective is
    the equally weighted mean codeword BLER of the LS, diagonal-LMMSE, and
    full-LMMSE methods at 10 dB. Each method is evaluated over 63 batches of
    four channels using the selected common tap count.

    The ``"variance-validation"`` stage compares the rounded winner with the
    existing shared scale using the same equal-method objective.

    :param study: Stage to execute: ``"taps"``, ``"taps-validation"``,
        ``"variance"``, or ``"variance-validation"``.
    :param n_trials: Trial-count override. If `None`, use 50 for tap tuning
        and 10 for variance calibration. Validation stages do not run Optuna
        trials.
    :param num_mc_batches: Monte Carlo budget override. If `None`, use three
        batches per SNR for tap tuning and 63 for validation and variance
        calibration.
    :param batch_size: Channels per batch. If `None`, use the canonical
        configuration value of four.
    :param output_dir: Output directory override for the selected stage.
    :param tap_output_dir: Directory containing the tap-tuning and
        paired-validation evidence required by variance calibration.
    :param asset_root: Root containing the frozen delay-covariance asset used
        by all tuning and validation stages.
    :returns: An Optuna study for tuning stages or a dictionary containing
        paired-validation results for validation stages.
    """
    effective_batch_size = (
        umi_3gpp_config.batch_size if batch_size is None else int(batch_size)
    )
    tap_output_dir = (
        TAP_TUNING_OUTPUT_DIR if tap_output_dir is None else tap_output_dir
    )
    if study == "taps":
        return _run_tap_tuning(
            n_trials=50 if n_trials is None else n_trials,
            num_mc_batches=3 if num_mc_batches is None else num_mc_batches,
            batch_size=effective_batch_size,
            output_dir=tap_output_dir if output_dir is None else output_dir,
            asset_root=asset_root,
        )
    if study == "taps-validation":
        return _run_tap_validation(
            num_mc_batches=(
                umi_3gpp_config.num_mc_batches
                if num_mc_batches is None
                else num_mc_batches
            ),
            batch_size=effective_batch_size,
            output_dir=tap_output_dir if output_dir is None else output_dir,
            asset_root=asset_root,
        )
    variance_output_dir = (
        VARIANCE_TUNING_OUTPUT_DIR if output_dir is None else output_dir
    )
    if study == "variance":
        return _run_variance_calibration(
            n_trials=10 if n_trials is None else n_trials,
            num_mc_batches=(
                umi_3gpp_config.num_mc_batches
                if num_mc_batches is None
                else num_mc_batches
            ),
            batch_size=effective_batch_size,
            output_dir=variance_output_dir,
            tap_output_dir=tap_output_dir,
            asset_root=asset_root,
        )
    if study == "variance-validation":
        return _run_variance_validation(
            num_mc_batches=(
                umi_3gpp_config.num_mc_batches
                if num_mc_batches is None
                else num_mc_batches
            ),
            batch_size=effective_batch_size,
            output_dir=variance_output_dir,
            tap_output_dir=tap_output_dir,
            asset_root=asset_root,
        )
    raise ValueError(
        "`study` must be 'taps', 'taps-validation', 'variance', or "
        "'variance-validation'."
    )


def main() -> None:
    """Parse command-line arguments and run one Edfors tuning stage."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--study",
        choices=(
            "taps",
            "taps-validation",
            "variance",
            "variance-validation",
        ),
        default="taps",
    )
    parser.add_argument("--n-trials", type=int, default=None)
    parser.add_argument("--num-mc-batches", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--tap-output-dir", type=str, default=None)
    parser.add_argument("--asset-root", type=str, required=True)
    args = parser.parse_args()
    run_tuning(
        study=args.study,
        n_trials=args.n_trials,
        num_mc_batches=args.num_mc_batches,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        tap_output_dir=args.tap_output_dir,
        asset_root=args.asset_root,
    )


if __name__ == "__main__":
    main()
