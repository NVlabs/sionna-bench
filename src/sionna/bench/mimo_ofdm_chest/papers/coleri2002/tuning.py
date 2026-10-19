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

from .estimator import (
    LSCombEstimator,
    METHOD_LOWPASS,
    METHOD_NAMES,
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

LOWPASS_TUNING_OUTPUT_DIR = Path("tuning/mimo_ofdm_chest/coleri2002/lowpass_v2")
VARIANCE_TUNING_OUTPUT_DIR = Path("tuning/mimo_ofdm_chest/coleri2002/variance_scale_v2")


def _suggest_lowpass_kwargs(trial, cfg, resource_grid):
    """Suggest only the MATLAB ``interp``-style low-pass FIR parameters."""
    del cfg, resource_grid
    return {
        "lowpass_filter_half_length": trial.suggest_int(
            "lowpass_filter_half_length",
            1,
            16,
        ),
        "lowpass_cutoff": trial.suggest_float(
            "lowpass_cutoff",
            0.1,
            1.0,
        ),
    }


def _suggest_err_var_scale(trial, cfg, resource_grid):
    """Suggest the shared detector-facing variance scale."""
    del cfg, resource_grid
    return {
        "err_var_scale": trial.suggest_float(
            "err_var_scale",
            1e-3,
            1.0,
            log=True,
        )
    }


def _method_kwargs(method: str) -> dict[str, object]:
    """Return the frozen constructor arguments for one Coleri method."""
    return {"method": method}


def _round_three_significant_digits(value: float) -> float:
    """Round a continuous selected value to three significant digits."""
    return float(f"{value:.3g}")


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


def _run_lowpass_tuning(
    *,
    n_trials: int,
    num_mc_batches: int,
    batch_size: int,
    output_dir: Union[str, Path],
):
    """Tune low-pass interpolation on one shared cached probe set."""
    objective = build_cached_nmse_objective(
        LSCombEstimator,
        umi_3gpp_config,
        ebno_dbs=TUNING_EBNO_DBS,
        snr_weights=TUNING_SNR_WEIGHTS,
        num_mc_batches=num_mc_batches,
        batch_size=batch_size,
        seed=ALGORITHM_PROBE_SEED,
    )
    return tune_channel_estimator(
        LSCombEstimator,
        umi_3gpp_config,
        _suggest_lowpass_kwargs,
        output_dir,
        fixed_estimator_kwargs={
            "method": METHOD_LOWPASS,
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


def _run_lowpass_validation(
    *,
    num_mc_batches: int,
    batch_size: int,
    output_dir: Union[str, Path],
) -> dict[str, object]:
    """Compare the rounded low-pass winner with the current configuration."""
    raw = _load_best_params(output_dir)
    rounded = {
        "lowpass_filter_half_length": int(raw["lowpass_filter_half_length"]),
        "lowpass_cutoff": _round_three_significant_digits(
            float(raw["lowpass_cutoff"])
        ),
    }
    candidates = {
        "current": resolve_constructor_arguments(
            LSCombEstimator,
            ("lowpass_filter_half_length", "lowpass_cutoff"),
        ),
        "rounded_winner": rounded,
    }
    objective = build_cached_nmse_objective(
        LSCombEstimator,
        umi_3gpp_config,
        ebno_dbs=TUNING_EBNO_DBS,
        snr_weights=TUNING_SNR_WEIGHTS,
        num_mc_batches=num_mc_batches,
        batch_size=batch_size,
        seed=ALGORITHM_VALIDATION_SEED,
    )
    objectives = {
        name: objective(
            {
                "method": METHOD_LOWPASS,
                **parameters,
            }
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
    num_mc_batches: int,
    batch_size: int,
    seed: int,
):
    """Build the equal-method mean-BLER objective for all four methods."""
    method_objectives = [
        build_cached_detector_scale_objective(
            LSCombEstimator,
            umi_3gpp_config,
            _method_kwargs(method),
            ebno_dbs=DETECTOR_CALIBRATION_EBNO_DBS,
            num_mc_batches=num_mc_batches,
            batch_size=batch_size,
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
):
    """Calibrate one shared scale against mean 10 dB BLER."""
    return tune_channel_estimator(
        LSCombEstimator,
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
) -> dict[str, object]:
    """Compare the rounded variance winner with the current shared scale."""
    raw = _load_best_params(output_dir)
    candidates = {
        "current": resolve_constructor_arguments(
            LSCombEstimator,
            ("err_var_scale",),
        )["err_var_scale"],
        "rounded_winner": _round_three_significant_digits(
            float(raw["err_var_scale"])
        ),
    }
    objective = _variance_objective(
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
            "metric": "mean_codeword_bler",
            "method_weighting": "equal over all four Coleri methods",
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
    study: str = "lowpass",
    n_trials: Optional[int] = None,
    num_mc_batches: Optional[int] = None,
    batch_size: Optional[int] = None,
    output_dir: Optional[Union[str, Path]] = None,
):
    r"""

    .. rubric:: Training Hyperparameter Tuning

    The ``"training"`` stage minimizes weighted NMSE at -10, -5, 0, 5,
    and 10 dB. It runs 10 seeded TPE trials. Every trial trains a separate
    VAE model for each of these five SNRs and evaluates three batches of four
    channels at each SNR.

    All trials reuse the same training-data bank. Every SNR-specific model uses
    6,400 training channels and 256 validation channels, at most 200 epochs,
    patience of 20 epochs, and a relative minimum validation improvement of
    1e-4. The best checkpoint is restored.

    .. list-table::
        :header-rows: 1

        * - Parameter
          - Search space
        * - ``latent_dim``
          - categorical set {4, 8, 16}
        * - ``learning_rate``
          - logarithmic from 1e-5 through 1e-3
        * - ``free_bits``
          - uniform from 0 through 1
        * - ``optimizer_batch_size``
          - categorical set {256, 512, 1024, 2048}

    The ``"training-validation"`` stage retrains and compares the rounded
    winner with the existing training defaults using 63 batches of four
    channels at each SNR.

    .. rubric:: Error-Variance Calibration

    The ``"variance"`` stage tunes ``err_var_scale`` logarithmically from
    1e-3 through 1 using 10 TPE trials. Codeword BLER is evaluated at 10 dB
    over 63 batches of four channels using the frozen canonical model bank.

    The ``"variance-validation"`` stage compares the rounded winner with the
    existing scale.

    :param study: Stage to execute: ``"training"``,
        ``"training-validation"``, ``"variance"``, or
        ``"variance-validation"``.
    :param n_trials: Trial-count override. If `None`, use 10 for both training
        tuning and variance calibration. Validation stages do not run Optuna
        trials.
    :param num_mc_batches: Monte Carlo budget override. If `None`, use three
        batches per SNR for training tuning and 63 for validation and variance
        calibration.
    :param batch_size: Channels per evaluation batch. If `None`, use the
        canonical configuration value of four.
    :param output_dir: Output directory override for the selected stage.
    :param algorithm_output_dir: Directory containing the training study and
        paired-validation evidence used to select the canonical training recipe.
    :param asset_root: External root containing or receiving trained VAE model
        assets.
    :param training_data_manifest: Manifest of the fixed training and validation
        data bank. This is required by the training and training-validation
        stages.
    :returns: An Optuna study for tuning stages or a dictionary containing
        paired-validation results for validation stages.
    """
    effective_batch_size = (
        umi_3gpp_config.batch_size if batch_size is None else int(batch_size)
    )
    if study == "lowpass":
        return _run_lowpass_tuning(
            n_trials=50 if n_trials is None else n_trials,
            num_mc_batches=3 if num_mc_batches is None else num_mc_batches,
            batch_size=effective_batch_size,
            output_dir=(
                LOWPASS_TUNING_OUTPUT_DIR if output_dir is None else output_dir
            ),
        )
    if study == "lowpass-validation":
        return _run_lowpass_validation(
            num_mc_batches=(
                umi_3gpp_config.num_mc_batches
                if num_mc_batches is None
                else num_mc_batches
            ),
            batch_size=effective_batch_size,
            output_dir=(
                LOWPASS_TUNING_OUTPUT_DIR if output_dir is None else output_dir
            ),
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
            output_dir=(
                VARIANCE_TUNING_OUTPUT_DIR if output_dir is None else output_dir
            ),
        )
    if study == "variance-validation":
        return _run_variance_validation(
            num_mc_batches=(
                umi_3gpp_config.num_mc_batches
                if num_mc_batches is None
                else num_mc_batches
            ),
            batch_size=effective_batch_size,
            output_dir=(
                VARIANCE_TUNING_OUTPUT_DIR if output_dir is None else output_dir
            ),
        )
    raise ValueError(
        "`study` must be 'lowpass', 'lowpass-validation', 'variance', or "
        "'variance-validation'."
    )


def main() -> None:
    """Parse command-line arguments and run one Coleri tuning stage."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--study",
        choices=(
            "lowpass",
            "lowpass-validation",
            "variance",
            "variance-validation",
        ),
        default="lowpass",
    )
    parser.add_argument("--n-trials", type=int, default=None)
    parser.add_argument("--num-mc-batches", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()
    run_tuning(
        study=args.study,
        n_trials=args.n_trials,
        num_mc_batches=args.num_mc_batches,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
