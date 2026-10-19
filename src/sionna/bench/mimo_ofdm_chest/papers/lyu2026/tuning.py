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
    tune_channel_estimator,
)
from sionna.bench.mimo_ofdm_chest.benchmarking.result_store import resolve_constructor_arguments
from sionna.bench.mimo_ofdm_chest.configs import umi_3gpp_config
from sionna.bench.mimo_ofdm_chest.persistence import write_json_atomic

from .estimator import SparseDiffuseChannelEstimator

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

ADMM_TUNING_OUTPUT_DIR = Path("tuning/mimo_ofdm_chest/lyu2026/admm_v2")
VARIANCE_TUNING_OUTPUT_DIR = Path("tuning/mimo_ofdm_chest/lyu2026/variance_scale_v2")


def _suggest_parameters(trial, cfg, resource_grid):
    """Suggest sparse/diffuse model and ADMM solver parameters."""
    del cfg, resource_grid
    return {
        "diffuse_num_taps": trial.suggest_int("diffuse_num_taps", 1, 24),
        "tau_noise_scale": trial.suggest_float(
            "tau_noise_scale",
            1e-2,
            1e2,
            log=True,
        ),
        "diffuse_energy": trial.suggest_float(
            "diffuse_energy",
            1e-3,
            1e2,
            log=True,
        ),
        "admm_rho": trial.suggest_float(
            "admm_rho",
            1e-2,
            3.0,
            log=True,
        ),
        "admm_num_iters": trial.suggest_int("admm_num_iters", 5, 20),
    }


def _suggest_err_var_scale(trial, cfg, resource_grid):
    """Suggest only the detector-facing variance scale."""
    del cfg, resource_grid
    return {
        "err_var_scale": trial.suggest_float(
            "err_var_scale",
            1e-3,
            1.0,
            log=True,
        )
    }


def _round_algorithm_parameters(
    parameters: dict[str, object],
) -> dict[str, object]:
    """Round continuous parameters to three significant digits."""
    return {
        "diffuse_num_taps": int(parameters["diffuse_num_taps"]),
        "tau_noise_scale": float(
            f"{float(parameters['tau_noise_scale']):.3g}"
        ),
        "diffuse_energy": float(
            f"{float(parameters['diffuse_energy']):.3g}"
        ),
        "admm_rho": float(f"{float(parameters['admm_rho']):.3g}"),
        "admm_num_iters": int(parameters["admm_num_iters"]),
    }


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


def _selected_algorithm_parameters(
    algorithm_output_dir: Union[str, Path],
) -> dict[str, object]:
    """Return ADMM parameters selected by paired validation."""
    path = Path(algorithm_output_dir) / "paired_validation.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"Lyu paired validation is required before calibration: {path}."
        )
    with path.open(encoding="utf-8") as file:
        return dict(json.load(file)["selected_parameters"])


def _write_trial_states(
    output_dir: Union[str, Path],
    study,
    required_complete_trials: int,
) -> None:
    """Record Optuna trial states and reject an incomplete study."""
    counts: dict[str, int] = {}
    for trial in study.trials:
        name = trial.state.name.lower()
        counts[name] = counts.get(name, 0) + 1
    write_json_atomic(
        Path(output_dir) / "trial_states.json",
        {
            "required_complete_trials": required_complete_trials,
            "states": counts,
        },
    )
    if counts.get("complete", 0) < required_complete_trials:
        raise RuntimeError(
            "Lyu tuning requires "
            f"{required_complete_trials} completed trials, but observed "
            f"{counts.get('complete', 0)}. Resume the study before validation."
        )


def _run_admm_tuning(
    *,
    n_trials: int,
    num_mc_batches: int,
    batch_size: int,
    output_dir: Union[str, Path],
):
    """Tune complete Torch ADMM inference on one cached probe set."""
    objective = build_cached_nmse_objective(
        SparseDiffuseChannelEstimator,
        umi_3gpp_config,
        ebno_dbs=TUNING_EBNO_DBS,
        snr_weights=TUNING_SNR_WEIGHTS,
        num_mc_batches=num_mc_batches,
        batch_size=batch_size,
        seed=ALGORITHM_PROBE_SEED,
    )
    study = tune_channel_estimator(
        SparseDiffuseChannelEstimator,
        umi_3gpp_config,
        _suggest_parameters,
        output_dir,
        ebno_dbs=TUNING_EBNO_DBS,
        snr_weights=TUNING_SNR_WEIGHTS,
        n_trials=n_trials,
        num_mc_batches=num_mc_batches,
        batch_size=batch_size,
        sampler_seed=ALGORITHM_SAMPLER_SEED,
        probe_seed=ALGORITHM_PROBE_SEED,
        estimator_objective=objective,
    )
    _write_trial_states(output_dir, study, n_trials)
    return study


def _run_admm_validation(
    *,
    num_mc_batches: int,
    batch_size: int,
    output_dir: Union[str, Path],
) -> dict[str, object]:
    """Compare the rounded ADMM winner with the current configuration."""
    raw = _load_best_params(output_dir)
    candidates = {
        "current": resolve_constructor_arguments(
            SparseDiffuseChannelEstimator,
            (
                "diffuse_num_taps",
                "tau_noise_scale",
                "diffuse_energy",
                "admm_rho",
                "admm_num_iters",
            ),
        ),
        "rounded_winner": _round_algorithm_parameters(raw),
    }
    objective = build_cached_nmse_objective(
        SparseDiffuseChannelEstimator,
        umi_3gpp_config,
        ebno_dbs=TUNING_EBNO_DBS,
        snr_weights=TUNING_SNR_WEIGHTS,
        num_mc_batches=num_mc_batches,
        batch_size=batch_size,
        seed=ALGORITHM_VALIDATION_SEED,
    )
    objectives = {
        name: objective(parameters)
        for name, parameters in candidates.items()
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
            "metric": "weighted_nmse",
            "ebno_dbs": list(TUNING_EBNO_DBS),
            "snr_weights": list(TUNING_SNR_WEIGHTS),
            "seed": ALGORITHM_VALIDATION_SEED,
            "num_mc_batches_per_snr": num_mc_batches,
            "batch_size": batch_size,
            "raw_winner": raw,
            "candidates": candidates,
            "objectives": objectives,
            "selected": selected,
            "selected_parameters": candidates[selected],
        },
    )


def _variance_objective(
    *,
    algorithm_parameters: dict[str, object],
    num_mc_batches: int,
    batch_size: int,
    seed: int,
):
    """Build the cached 10 dB codeword-BLER scale objective."""
    return build_cached_detector_scale_objective(
        SparseDiffuseChannelEstimator,
        umi_3gpp_config,
        {
            **algorithm_parameters,
        },
        scale_argument="err_var_scale",
        ebno_dbs=DETECTOR_CALIBRATION_EBNO_DBS,
        num_mc_batches=num_mc_batches,
        batch_size=batch_size,
        seed=seed,
        detector_metric="bler",
    )


def _run_variance_calibration(
    *,
    n_trials: int,
    num_mc_batches: int,
    batch_size: int,
    output_dir: Union[str, Path],
    algorithm_output_dir: Union[str, Path],
):
    """Calibrate the variance scale against 10 dB codeword BLER."""
    parameters = _selected_algorithm_parameters(algorithm_output_dir)
    return tune_channel_estimator(
        SparseDiffuseChannelEstimator,
        umi_3gpp_config,
        _suggest_err_var_scale,
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
            algorithm_parameters=parameters,
            num_mc_batches=num_mc_batches,
            batch_size=batch_size,
            seed=VARIANCE_PROBE_SEED,
        ),
    )


def _run_variance_validation(
    *,
    num_mc_batches: int,
    batch_size: int,
    output_dir: Union[str, Path],
    algorithm_output_dir: Union[str, Path],
) -> dict[str, object]:
    """Compare the rounded scale winner with the current scale."""
    raw = _load_best_params(output_dir)
    candidates = {
        "current": resolve_constructor_arguments(
            SparseDiffuseChannelEstimator,
            ("err_var_scale",),
        )["err_var_scale"],
        "rounded_winner": float(f"{float(raw['err_var_scale']):.3g}"),
    }
    objective = _variance_objective(
        algorithm_parameters=_selected_algorithm_parameters(
            algorithm_output_dir
        ),
        num_mc_batches=num_mc_batches,
        batch_size=batch_size,
        seed=VARIANCE_VALIDATION_SEED,
    )
    objectives = {
        name: objective({"err_var_scale": scale})
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
            "metric": "codeword_bler",
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
    study: str = "admm",
    n_trials: Optional[int] = None,
    num_mc_batches: Optional[int] = None,
    batch_size: Optional[int] = None,
    output_dir: Optional[Union[str, Path]] = None,
    algorithm_output_dir: Optional[Union[str, Path]] = None,
):
    r"""

    .. rubric:: Algorithmic Tuning

    The ``"admm"`` stage minimizes weighted NMSE at -10, -5, 0, 5,
    and 10 dB. It runs 80 seeded TPE trials. Every trial evaluates three
    batches of four channels at each SNR and executes the complete batched
    Torch ADMM estimator.

    .. list-table::
        :header-rows: 1

        * - Parameter
          - Search space
        * - ``diffuse_num_taps``
          - integer from 1 through 24
        * - ``tau_noise_scale``
          - logarithmic from 1e-2 through 1e2
        * - ``diffuse_energy``
          - logarithmic from 1e-3 through 1e2
        * - ``admm_rho``
          - logarithmic from 1e-2 through 3
        * - ``admm_num_iters``
          - integer from 5 through 20

    The study requires 80 completed trials. Failed or incomplete trials do not
    satisfy this requirement.

    The ``"admm-validation"`` stage compares the rounded winner with the
    existing estimator defaults using 63 batches of four channels per SNR.

    .. rubric:: Error-Variance Calibration

    The ``"variance"`` stage tunes ``err_var_scale`` logarithmically from
    1e-3 through 1 using 10 TPE trials. Codeword BLER is evaluated at 10 dB
    over 63 batches of four channels using the algorithm parameters selected
    by paired validation.

    The ``"variance-validation"`` stage compares the rounded winner with the
    existing scale.

    :param study: Stage to execute: ``"admm"``, ``"admm-validation"``,
        ``"variance"``, or ``"variance-validation"``.
    :param n_trials: Trial-count override. If `None`, use 80 for ADMM tuning
        and 10 for variance calibration. Validation stages do not run Optuna
        trials.
    :param num_mc_batches: Monte Carlo budget override. If `None`, use three
        batches per SNR for ADMM tuning and 63 for validation and variance
        calibration.
    :param batch_size: Channels per batch. If `None`, use the canonical
        configuration value of four.
    :param output_dir: Output directory override for the selected stage.
    :param algorithm_output_dir: Directory containing the ADMM tuning and
        paired-validation evidence required by variance calibration.
    :returns: An Optuna study for tuning stages or a dictionary containing
        paired-validation results for validation stages.
    """
    effective_batch_size = (
        umi_3gpp_config.batch_size if batch_size is None else int(batch_size)
    )
    algorithm_output_dir = (
        ADMM_TUNING_OUTPUT_DIR
        if algorithm_output_dir is None
        else algorithm_output_dir
    )
    if study == "admm":
        return _run_admm_tuning(
            n_trials=80 if n_trials is None else n_trials,
            num_mc_batches=3 if num_mc_batches is None else num_mc_batches,
            batch_size=effective_batch_size,
            output_dir=(
                algorithm_output_dir if output_dir is None else output_dir
            ),
        )
    if study == "admm-validation":
        return _run_admm_validation(
            num_mc_batches=(
                umi_3gpp_config.num_mc_batches
                if num_mc_batches is None
                else num_mc_batches
            ),
            batch_size=effective_batch_size,
            output_dir=(
                algorithm_output_dir if output_dir is None else output_dir
            ),
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
            algorithm_output_dir=algorithm_output_dir,
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
            algorithm_output_dir=algorithm_output_dir,
        )
    raise ValueError(
        "`study` must be 'admm', 'admm-validation', 'variance', or "
        "'variance-validation'."
    )


def main() -> None:
    """Parse command-line arguments and run one Lyu tuning stage."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--study",
        choices=(
            "admm",
            "admm-validation",
            "variance",
            "variance-validation",
        ),
        default="admm",
    )
    parser.add_argument("--n-trials", type=int, default=None)
    parser.add_argument("--num-mc-batches", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--algorithm-output-dir", type=str, default=None)
    args = parser.parse_args()
    run_tuning(
        study=args.study,
        n_trials=args.n_trials,
        num_mc_batches=args.num_mc_batches,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        algorithm_output_dir=args.algorithm_output_dir,
    )


if __name__ == "__main__":
    main()
