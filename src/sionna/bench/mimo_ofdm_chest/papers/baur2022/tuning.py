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
from sionna.bench.mimo_ofdm_chest.benchmarking.result_store import (
    resolve_callable_arguments,
    resolve_constructor_arguments,
)
from sionna.bench.mimo_ofdm_chest.configs import umi_3gpp_config
from sionna.bench.mimo_ofdm_chest.persistence import write_json_atomic

from .asset import load_model, load_training_data, train_model, METHOD_NOISY
from .estimator import (
    COVARIANCE_FULL,
    VAEChannelEstimator,
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

TRAIN_BATCHES = 400
TRAIN_BATCH_SIZE = 16
VALIDATION_BATCHES = 16
VALIDATION_BATCH_SIZE = 16
MAX_EPOCHS = 200
PATIENCE_EPOCHS = 20
RELATIVE_MIN_DELTA = 1e-4
MODEL_SEED = umi_3gpp_config.seed
COVARIANCE_FLOOR = 1e-5
COVARIANCE_MODE = COVARIANCE_FULL

TRAINING_TUNING_OUTPUT_DIR = Path("tuning/mimo_ofdm_chest/baur2022/training_v2")
VARIANCE_TUNING_OUTPUT_DIR = Path("tuning/mimo_ofdm_chest/baur2022/variance_scale_v2")


def _training_recipe(
    parameters: dict[str, object],
    training_snr_dbs: list[float],
) -> dict[str, object]:
    """Return the fixed full-effort VAE-noisy training recipe."""
    return {
        "method": METHOD_NOISY,
        "latent_dim": int(parameters["latent_dim"]),
        "train_batches": TRAIN_BATCHES,
        "train_batch_size": TRAIN_BATCH_SIZE,
        "validation_batches": VALIDATION_BATCHES,
        "validation_batch_size": VALIDATION_BATCH_SIZE,
        "epochs": MAX_EPOCHS,
        "optimizer_batch_size": int(parameters["optimizer_batch_size"]),
        "learning_rate": float(parameters["learning_rate"]),
        "free_bits": float(parameters["free_bits"]),
        "min_train_seconds": 0.0,
        "patience_epochs": PATIENCE_EPOCHS,
        "relative_min_delta": RELATIVE_MIN_DELTA,
        "snr_specific_weights": True,
        "training_snr_dbs": training_snr_dbs,
        "covariance_floor": COVARIANCE_FLOOR,
        "covariance_mode": COVARIANCE_MODE,
        "seed": MODEL_SEED,
    }


def _load_training_bank(manifest_path: Union[str, Path]):
    """Load and validate the common all-SNR data bank once."""
    resource_grid = build_resource_grid(umi_3gpp_config)
    _, validated_path = load_training_data(
        resource_grid,
        umi_3gpp_config,
        train_batches=TRAIN_BATCHES,
        train_batch_size=TRAIN_BATCH_SIZE,
        validation_batches=VALIDATION_BATCHES,
        validation_batch_size=VALIDATION_BATCH_SIZE,
        training_snr_dbs=list(umi_3gpp_config.snr_db_range()),
        seed=MODEL_SEED,
        manifest_path=manifest_path,
    )
    return resource_grid, validated_path


def _train_estimator_kwargs(
    *,
    resource_grid,
    training_data_manifest: Path,
    asset_root: Union[str, Path],
    parameters: dict[str, object],
    training_snr_dbs: list[float],
) -> tuple[dict[str, object], list[Path]]:
    """Train one VAE-noisy model bank and return inference arguments."""
    recipe = _training_recipe(parameters, training_snr_dbs)
    try:
        states, noise_variances, manifest_paths = train_model(
            resource_grid,
            umi_3gpp_config,
            **recipe,
            training_data_manifest=training_data_manifest,
            asset_root=asset_root,
        )
    except FileExistsError:
        states, noise_variances, manifest_paths = load_model(
            resource_grid,
            umi_3gpp_config,
            **recipe,
            asset_root=asset_root,
        )
    return {
        "model_states": states,
        "model_noise_variances": noise_variances,
        "num_rx_ant": umi_3gpp_config.num_rx_ant,
        "latent_dim": int(parameters["latent_dim"]),
        "covariance_floor": COVARIANCE_FLOOR,
        "covariance_mode": COVARIANCE_MODE,
        "interpolation_type": "lin_time_avg",
    }, manifest_paths


def _suggest_training_parameters(trial, cfg, resource_grid):
    """Suggest the v2 VAE architecture and optimizer settings."""
    del cfg, resource_grid
    return {
        "latent_dim": trial.suggest_categorical("latent_dim", [4, 8, 16]),
        "learning_rate": trial.suggest_float(
            "learning_rate", 1e-5, 1e-3, log=True
        ),
        "free_bits": trial.suggest_float("free_bits", 0.0, 1.0),
        "optimizer_batch_size": trial.suggest_categorical(
            "optimizer_batch_size", [256, 512, 1024, 2048]
        ),
    }


def _suggest_err_var_scale(trial, cfg, resource_grid):
    """Suggest only the detector-facing posterior-variance scale."""
    del cfg, resource_grid
    return {
        "err_var_scale": trial.suggest_float(
            "err_var_scale", 1e-3, 1.0, log=True
        )
    }


def _round_training_parameters(
    parameters: dict[str, object],
) -> dict[str, object]:
    """Round continuous winner parameters to three significant digits."""
    return {
        "latent_dim": int(parameters["latent_dim"]),
        "learning_rate": float(f"{float(parameters['learning_rate']):.3g}"),
        "free_bits": float(f"{float(parameters['free_bits']):.3g}"),
        "optimizer_batch_size": int(parameters["optimizer_batch_size"]),
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


def _selected_training_parameters(
    algorithm_output_dir: Union[str, Path],
) -> dict[str, object]:
    """Return the training recipe selected by paired validation."""
    path = Path(algorithm_output_dir) / "paired_validation.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"Baur paired validation is required before calibration: {path}."
        )
    with path.open(encoding="utf-8") as file:
        return dict(json.load(file)["selected_parameters"])


def _run_training_tuning(
    *,
    n_trials: int,
    num_mc_batches: int,
    batch_size: int,
    output_dir: Union[str, Path],
    asset_root: Union[str, Path],
    training_data_manifest: Union[str, Path],
):
    """Tune five full-effort SNR-specific VAE-noisy models per trial."""
    resource_grid, validated_data_path = _load_training_bank(
        training_data_manifest
    )
    objective = build_cached_nmse_objective(
        VAEChannelEstimator,
        umi_3gpp_config,
        ebno_dbs=TUNING_EBNO_DBS,
        snr_weights=TUNING_SNR_WEIGHTS,
        num_mc_batches=num_mc_batches,
        batch_size=batch_size,
        resource_grid=resource_grid,
        seed=ALGORITHM_PROBE_SEED,
    )

    def train_candidate(trial, cfg, trial_resource_grid):
        parameters = _suggest_training_parameters(
            trial, cfg, trial_resource_grid
        )
        estimator_kwargs, _ = _train_estimator_kwargs(
            resource_grid=resource_grid,
            training_data_manifest=validated_data_path,
            asset_root=asset_root,
            parameters=parameters,
            training_snr_dbs=list(TUNING_EBNO_DBS),
        )
        return estimator_kwargs

    return tune_channel_estimator(
        VAEChannelEstimator,
        umi_3gpp_config,
        train_candidate,
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


def _run_training_validation(
    *,
    num_mc_batches: int,
    batch_size: int,
    output_dir: Union[str, Path],
    asset_root: Union[str, Path],
    training_data_manifest: Union[str, Path],
) -> dict[str, object]:
    """Train and compare the rounded winner with the current full recipe."""
    resource_grid, validated_data_path = _load_training_bank(
        training_data_manifest
    )
    raw = _load_best_params(output_dir)
    candidates = {
        "current": resolve_callable_arguments(
            train_model,
            (
                "latent_dim",
                "learning_rate",
                "free_bits",
                "optimizer_batch_size",
            ),
        ),
        "rounded_winner": _round_training_parameters(raw),
    }
    objective = build_cached_nmse_objective(
        VAEChannelEstimator,
        umi_3gpp_config,
        ebno_dbs=TUNING_EBNO_DBS,
        snr_weights=TUNING_SNR_WEIGHTS,
        num_mc_batches=num_mc_batches,
        batch_size=batch_size,
        resource_grid=resource_grid,
        seed=ALGORITHM_VALIDATION_SEED,
    )
    objectives = {}
    manifests = {}
    for name, parameters in candidates.items():
        estimator_kwargs, manifest_paths = _train_estimator_kwargs(
            resource_grid=resource_grid,
            training_data_manifest=validated_data_path,
            asset_root=asset_root,
            parameters=parameters,
            training_snr_dbs=list(TUNING_EBNO_DBS),
        )
        objectives[name] = objective(estimator_kwargs)
        manifests[name] = [str(path) for path in manifest_paths]
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
            "model_manifests": manifests,
            "selected": selected,
            "selected_parameters": candidates[selected],
        },
    )


def _load_canonical_estimator_kwargs(
    asset_root: Union[str, Path],
    algorithm_output_dir: Union[str, Path],
) -> dict[str, object]:
    """Load the final 11-SNR model bank selected by validation."""
    parameters = _selected_training_parameters(algorithm_output_dir)
    resource_grid = build_resource_grid(umi_3gpp_config)
    states, noise_variances, _ = load_model(
        resource_grid,
        umi_3gpp_config,
        **_training_recipe(
            parameters,
            list(umi_3gpp_config.snr_db_range()),
        ),
        asset_root=asset_root,
    )
    return {
        "model_states": states,
        "model_noise_variances": noise_variances,
        "num_rx_ant": umi_3gpp_config.num_rx_ant,
        "latent_dim": int(parameters["latent_dim"]),
        "covariance_floor": COVARIANCE_FLOOR,
        "covariance_mode": COVARIANCE_MODE,
        "interpolation_type": "lin_time_avg",
    }


def _variance_objective(
    *,
    estimator_kwargs: dict[str, object],
    num_mc_batches: int,
    batch_size: int,
    seed: int,
):
    """Build the cached 10 dB codeword-BLER scale objective."""
    return build_cached_detector_scale_objective(
        VAEChannelEstimator,
        umi_3gpp_config,
        estimator_kwargs,
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
    asset_root: Union[str, Path],
):
    """Tune detector-facing variance scale for the frozen model bank."""
    estimator_kwargs = _load_canonical_estimator_kwargs(
        asset_root, algorithm_output_dir
    )
    objective = _variance_objective(
        estimator_kwargs=estimator_kwargs,
        num_mc_batches=num_mc_batches,
        batch_size=batch_size,
        seed=VARIANCE_PROBE_SEED,
    )
    return tune_channel_estimator(
        VAEChannelEstimator,
        umi_3gpp_config,
        _suggest_err_var_scale,
        output_dir,
        fixed_estimator_kwargs=estimator_kwargs,
        ebno_dbs=DETECTOR_CALIBRATION_EBNO_DBS,
        objective_metric="detector",
        detector_metric="bler",
        n_trials=n_trials,
        num_mc_batches=num_mc_batches,
        batch_size=batch_size,
        sampler_seed=VARIANCE_SAMPLER_SEED,
        probe_seed=VARIANCE_PROBE_SEED,
        estimator_objective=objective,
    )


def _run_variance_validation(
    *,
    num_mc_batches: int,
    batch_size: int,
    output_dir: Union[str, Path],
    algorithm_output_dir: Union[str, Path],
    asset_root: Union[str, Path],
) -> dict[str, object]:
    """Paired-validate the rounded variance winner against the current scale."""
    estimator_kwargs = _load_canonical_estimator_kwargs(
        asset_root, algorithm_output_dir
    )
    raw = _load_best_params(output_dir)
    candidates = {
        "current": resolve_constructor_arguments(
            VAEChannelEstimator,
            ("err_var_scale",),
        )["err_var_scale"],
        "rounded_winner": float(f"{float(raw['err_var_scale']):.3g}"),
    }
    objective = _variance_objective(
        estimator_kwargs=estimator_kwargs,
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
    study: str = "training",
    n_trials: Optional[int] = None,
    num_mc_batches: Optional[int] = None,
    batch_size: Optional[int] = None,
    output_dir: Optional[Union[str, Path]] = None,
    algorithm_output_dir: Optional[Union[str, Path]] = None,
    asset_root: Union[str, Path],
    training_data_manifest: Optional[Union[str, Path]] = None,
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
    algorithm_output_dir = (
        TRAINING_TUNING_OUTPUT_DIR
        if algorithm_output_dir is None
        else algorithm_output_dir
    )
    if study in ("training", "training-validation"):
        if training_data_manifest is None:
            raise ValueError(
                "Baur training tuning requires `training_data_manifest`."
            )
        if study == "training":
            return _run_training_tuning(
                n_trials=10 if n_trials is None else n_trials,
                num_mc_batches=3 if num_mc_batches is None else num_mc_batches,
                batch_size=effective_batch_size,
                output_dir=(
                    algorithm_output_dir if output_dir is None else output_dir
                ),
                asset_root=asset_root,
                training_data_manifest=training_data_manifest,
            )
        return _run_training_validation(
            num_mc_batches=(
                umi_3gpp_config.num_mc_batches
                if num_mc_batches is None
                else num_mc_batches
            ),
            batch_size=effective_batch_size,
            output_dir=(
                algorithm_output_dir if output_dir is None else output_dir
            ),
            asset_root=asset_root,
            training_data_manifest=training_data_manifest,
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
            algorithm_output_dir=algorithm_output_dir,
            asset_root=asset_root,
        )
    raise ValueError(
        "`study` must be 'training', 'training-validation', 'variance', "
        "or 'variance-validation'."
    )


def main() -> None:
    """Parse command-line arguments and run one Baur v2 tuning stage."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--study",
        choices=(
            "training",
            "training-validation",
            "variance",
            "variance-validation",
        ),
        default="training",
    )
    parser.add_argument("--n-trials", type=int, default=None)
    parser.add_argument("--num-mc-batches", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--algorithm-output-dir", type=Path, default=None)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--training-data-manifest", type=Path, default=None)
    args = parser.parse_args()
    run_tuning(
        study=args.study,
        n_trials=args.n_trials,
        num_mc_batches=args.num_mc_batches,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        algorithm_output_dir=args.algorithm_output_dir,
        asset_root=args.asset_root,
        training_data_manifest=args.training_data_manifest,
    )


if __name__ == "__main__":
    main()
