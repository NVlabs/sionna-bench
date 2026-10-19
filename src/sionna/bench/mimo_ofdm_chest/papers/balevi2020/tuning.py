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

from .estimator import DeepPriorChannelEstimator


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

OPTIMIZATION_TUNING_OUTPUT_DIR = Path("tuning/mimo_ofdm_chest/balevi2020/optimization_v2")
VARIANCE_TUNING_OUTPUT_DIR = Path("tuning/mimo_ofdm_chest/balevi2020/variance_scale_v2")

__all__ = ["run_tuning"]

def _suggest_optimization_parameters(trial, cfg, resource_grid):
    """Suggest parameters controlling per-instance network fitting."""
    del cfg, resource_grid
    return {
        "learning_rate": trial.suggest_float(
            "learning_rate",
            1e-4,
            3e-2,
            log=True,
        ),
        "input_noise_scale": trial.suggest_float(
            "input_noise_scale",
            0.1,
            3.0,
            log=True,
        ),
        "num_steps": trial.suggest_int("num_steps", 100, 2000, log=True),
    }


def _suggest_err_var_scale(trial, cfg, resource_grid):
    """Suggest only the detector-facing LS variance scale."""
    del cfg, resource_grid
    return {
        "err_var_scale": trial.suggest_float(
            "err_var_scale",
            1e-3,
            1.0,
            log=True,
        )
    }


def _round_optimization_parameters(
    parameters: dict[str, object],
) -> dict[str, object]:
    """Round continuous parameters to three significant digits."""
    return {
        "learning_rate": float(f"{float(parameters['learning_rate']):.3g}"),
        "input_noise_scale": float(
            f"{float(parameters['input_noise_scale']):.3g}"
        ),
        "num_steps": int(parameters["num_steps"]),
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


def _selected_optimization_parameters(
    algorithm_output_dir: Union[str, Path],
) -> dict[str, object]:
    """Return fitting parameters selected by paired validation."""
    path = Path(algorithm_output_dir) / "paired_validation.json"
    if not path.is_file():
        raise FileNotFoundError(
            "Balevi paired validation is required before calibration: "
            f"{path}."
        )
    with path.open(encoding="utf-8") as file:
        return dict(json.load(file)["selected_parameters"])


def _run_optimization_tuning(
    *,
    n_trials: int,
    num_mc_batches: int,
    batch_size: int,
    output_dir: Union[str, Path],
):
    """Tune full-effort online fitting on one cached observation set."""
    objective = build_cached_nmse_objective(
        DeepPriorChannelEstimator,
        umi_3gpp_config,
        ebno_dbs=TUNING_EBNO_DBS,
        snr_weights=TUNING_SNR_WEIGHTS,
        num_mc_batches=num_mc_batches,
        batch_size=batch_size,
        seed=ALGORITHM_PROBE_SEED,
    )
    return tune_channel_estimator(
        DeepPriorChannelEstimator,
        umi_3gpp_config,
        _suggest_optimization_parameters,
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


def _run_optimization_validation(
    *,
    num_mc_batches: int,
    batch_size: int,
    output_dir: Union[str, Path],
) -> dict[str, object]:
    """Compare the rounded fitting winner with the current configuration."""
    raw = _load_best_params(output_dir)
    candidates = {
        "current": resolve_constructor_arguments(
            DeepPriorChannelEstimator,
            ("learning_rate", "input_noise_scale", "num_steps"),
        ),
        "rounded_winner": _round_optimization_parameters(raw),
    }
    objective = build_cached_nmse_objective(
        DeepPriorChannelEstimator,
        umi_3gpp_config,
        ebno_dbs=TUNING_EBNO_DBS,
        snr_weights=TUNING_SNR_WEIGHTS,
        num_mc_batches=num_mc_batches,
        batch_size=batch_size,
        seed=ALGORITHM_VALIDATION_SEED,
    )
    objectives = {
        name: objective(
            parameters
        )
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
    optimization_parameters: dict[str, object],
    num_mc_batches: int,
    batch_size: int,
    seed: int,
):
    """Fit once and build the cached 10 dB codeword-BLER objective."""
    return build_cached_detector_scale_objective(
        DeepPriorChannelEstimator,
        umi_3gpp_config,
        {
            **optimization_parameters,
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
    """Calibrate the variance scale against cached 10 dB estimates."""
    parameters = _selected_optimization_parameters(algorithm_output_dir)
    return tune_channel_estimator(
        DeepPriorChannelEstimator,
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
            optimization_parameters=parameters,
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
            DeepPriorChannelEstimator,
            ("err_var_scale",),
        )["err_var_scale"],
        "rounded_winner": float(f"{float(raw['err_var_scale']):.3g}"),
    }
    objective = _variance_objective(
        optimization_parameters=_selected_optimization_parameters(
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
    study: str = "optimization",
    n_trials: Optional[int] = None,
    num_mc_batches: Optional[int] = None,
    batch_size: Optional[int] = None,
    output_dir: Optional[Union[str, Path]] = None,
    algorithm_output_dir: Optional[Union[str, Path]] = None,
):
    r"""

    .. rubric:: Algorithmic Tuning

    The ``"optimization"`` stage minimizes weighted NMSE at -10, -5, 0, 5,
    and 10 dB. It runs 30 seeded TPE trials. Every trial evaluates three
    batches of four channels at each SNR.

    .. list-table::
        :header-rows: 1

        * - Parameter
          - Search space
        * - ``learning_rate``
          - logarithmic from 1e-4 through 3e-2
        * - ``input_noise_scale``
          - logarithmic from 0.1 through 3
        * - ``num_steps``
          - logarithmically sampled integer from 100 through 2000

    The ``"optimization-validation"`` stage compares the rounded winner with
    the existing estimator defaults using 63 batches of four channels per SNR.

    .. rubric:: Error-Variance Calibration

    The ``"variance"`` stage tunes ``err_var_scale`` logarithmically from
    1e-3 through 1 using 10 TPE trials. Codeword BLER is evaluated at 10 dB
    over 63 batches of four channels.

    The ``"variance-validation"`` stage compares the rounded winner with the
    existing scale.

    :param study: Stage to execute: ``"optimization"``,
        ``"optimization-validation"``, ``"variance"``, or
        ``"variance-validation"``.
    :param n_trials: Trial-count override. If `None`, use 30 for optimization
        tuning and 10 for variance calibration. Validation stages do not run
        Optuna trials.
    :param num_mc_batches: Monte Carlo budget override. If `None`, use three
        batches per SNR for optimization tuning and 63 for validation and
        variance calibration.
    :param batch_size: Channels per batch. If `None`, use the canonical
        configuration value of four.
    :param output_dir: Output directory override for the selected stage.
    :param algorithm_output_dir: Directory containing the optimization tuning
        and validation evidence required by variance calibration.
    :returns: An Optuna study for tuning stages or a dictionary containing
        paired-validation results for validation stages.
    """
    effective_batch_size = (
        umi_3gpp_config.batch_size if batch_size is None else int(batch_size)
    )
    algorithm_output_dir = (
        OPTIMIZATION_TUNING_OUTPUT_DIR
        if algorithm_output_dir is None
        else algorithm_output_dir
    )
    if study == "optimization":
        return _run_optimization_tuning(
            n_trials=30 if n_trials is None else n_trials,
            num_mc_batches=3 if num_mc_batches is None else num_mc_batches,
            batch_size=effective_batch_size,
            output_dir=(
                algorithm_output_dir if output_dir is None else output_dir
            ),
        )
    if study == "optimization-validation":
        return _run_optimization_validation(
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
        "`study` must be 'optimization', 'optimization-validation', "
        "'variance', or 'variance-validation'."
    )


def main() -> None:
    """Parse command-line arguments and run one Balevi tuning stage."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--study",
        choices=(
            "optimization",
            "optimization-validation",
            "variance",
            "variance-validation",
        ),
        default="optimization",
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
