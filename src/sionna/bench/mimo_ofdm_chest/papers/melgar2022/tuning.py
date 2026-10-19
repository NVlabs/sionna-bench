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

from .asset import (
    load_model2,
    load_model2_training_data,
    load_model3,
    load_model3_training_data,
    train_model2,
    train_model3,
)
from .estimator import (
    DNNChannelEstimator,
    InterpolatedDNNChannelEstimator,
    METHOD_MODEL_2,
    METHOD_MODEL_3,
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

TRAIN_BATCHES = 800
TRAIN_BATCH_SIZE = 8
VALIDATION_BATCHES = 32
VALIDATION_BATCH_SIZE = 8
MAX_EPOCHS = 200
PATIENCE_EPOCHS = 20
RELATIVE_MIN_DELTA = 1e-4
BLOCK_LEN = 78
MAX_EXAMPLES_PER_BATCH = 4096
MODEL_SEED = umi_3gpp_config.seed

TRAINING_TUNING_OUTPUT_DIRS = {
    METHOD_MODEL_2: Path("tuning/mimo_ofdm_chest/melgar2022/model2_training_v2"),
    METHOD_MODEL_3: Path("tuning/mimo_ofdm_chest/melgar2022/model3_training_v2"),
}
VARIANCE_TUNING_OUTPUT_DIRS = {
    METHOD_MODEL_2: Path("tuning/mimo_ofdm_chest/melgar2022/model2_variance_v2"),
    METHOD_MODEL_3: Path("tuning/mimo_ofdm_chest/melgar2022/model3_variance_v2"),
}
ESTIMATOR_CLASSES = {
    METHOD_MODEL_2: DNNChannelEstimator,
    METHOD_MODEL_3: InterpolatedDNNChannelEstimator,
}

# Frozen pre-study Model 2 recipe used by the recorded paired validation.
MODEL2_REFERENCE_PARAMETERS = {
    "learning_rate": 5e-4,
    "optimizer_batch_size": 500,
}


def _training_recipe(
    *,
    method: str,
    learning_rate: float,
    optimizer_batch_size: int,
    training_snr_dbs: list[float],
) -> dict[str, object]:
    """Return one method's fixed full-effort training recipe."""
    recipe = {
        "train_batches": TRAIN_BATCHES,
        "train_batch_size": TRAIN_BATCH_SIZE,
        "validation_batches": VALIDATION_BATCHES,
        "validation_batch_size": VALIDATION_BATCH_SIZE,
        "epochs": MAX_EPOCHS,
        "optimizer_batch_size": int(optimizer_batch_size),
        "learning_rate": float(learning_rate),
        "patience_epochs": PATIENCE_EPOCHS,
        "min_delta": RELATIVE_MIN_DELTA,
        "snr_specific_weights": True,
        "training_snr_dbs": training_snr_dbs,
        "seed": MODEL_SEED,
    }
    if method == METHOD_MODEL_3:
        recipe.update(
            {
                "block_len": BLOCK_LEN,
                "max_examples_per_batch": MAX_EXAMPLES_PER_BATCH,
            }
        )
    return recipe


def _load_training_bank(method: str, manifest_path: Union[str, Path]):
    """Load and validate one common all-SNR data bank once."""
    resource_grid = build_resource_grid(umi_3gpp_config)
    arguments = {
        "train_batches": TRAIN_BATCHES,
        "train_batch_size": TRAIN_BATCH_SIZE,
        "validation_batches": VALIDATION_BATCHES,
        "validation_batch_size": VALIDATION_BATCH_SIZE,
        "training_snr_dbs": list(umi_3gpp_config.snr_db_range()),
        "seed": MODEL_SEED,
        "manifest_path": manifest_path,
    }
    if method == METHOD_MODEL_2:
        _, validated_path = load_model2_training_data(
            resource_grid,
            umi_3gpp_config,
            **arguments,
        )
    else:
        _, validated_path = load_model3_training_data(
            resource_grid,
            umi_3gpp_config,
            block_len=BLOCK_LEN,
            max_examples_per_batch=MAX_EXAMPLES_PER_BATCH,
            **arguments,
        )
    return resource_grid, validated_path


def _train_estimator_kwargs(
    *,
    method: str,
    resource_grid,
    training_data_manifest: Path,
    asset_root: Union[str, Path],
    parameters: dict[str, object],
    training_snr_dbs: list[float],
) -> tuple[dict[str, object], list[Path]]:
    """Train one Melgar bank and return inference-ready arguments."""
    recipe = _training_recipe(
        method=method,
        learning_rate=float(parameters["learning_rate"]),
        optimizer_batch_size=int(parameters["optimizer_batch_size"]),
        training_snr_dbs=training_snr_dbs,
    )
    try:
        trainer = train_model2 if method == METHOD_MODEL_2 else train_model3
        states, noise_variances, manifest_paths = trainer(
            resource_grid,
            umi_3gpp_config,
            **recipe,
            training_data_manifest=training_data_manifest,
            asset_root=asset_root,
        )
    except FileExistsError:
        loader = load_model2 if method == METHOD_MODEL_2 else load_model3
        states, noise_variances, manifest_paths = loader(
            resource_grid,
            umi_3gpp_config,
            **recipe,
            asset_root=asset_root,
        )
    estimator_kwargs = {
        "model_states": states,
        "model_noise_variances": noise_variances,
    }
    if method == METHOD_MODEL_3:
        estimator_kwargs["block_len"] = BLOCK_LEN
    return estimator_kwargs, manifest_paths


def _suggest_training_parameters(trial, cfg, resource_grid):
    """Suggest the two optimizer settings selected by the v2 protocol."""
    del cfg, resource_grid
    return {
        "learning_rate": trial.suggest_float(
            "learning_rate", 1e-5, 1e-3, log=True
        ),
        "optimizer_batch_size": trial.suggest_categorical(
            "optimizer_batch_size", [256, 512, 1024, 2048]
        ),
    }


def _suggest_err_var_scale(trial, cfg, resource_grid):
    """Suggest only the detector-facing variance scale."""
    del cfg, resource_grid
    return {
        "err_var_scale": trial.suggest_float(
            "err_var_scale", 1e-3, 1.0, log=True
        )
    }


def _round_training_parameters(
    parameters: dict[str, object],
) -> dict[str, object]:
    """Round the learning rate to three significant digits."""
    return {
        "learning_rate": float(f"{float(parameters['learning_rate']):.3g}"),
        "optimizer_batch_size": int(parameters["optimizer_batch_size"]),
    }


def _load_best_params(output_dir: Union[str, Path]) -> dict[str, object]:
    """Load the raw Optuna winner from one study summary."""
    with (Path(output_dir) / "best_params.json").open(encoding="utf-8") as file:
        return json.load(file)["best_params"]


def _write_validation(
    output_dir: Union[str, Path], result: dict[str, object]
) -> dict[str, object]:
    """Write and return one paired-validation result."""
    write_json_atomic(Path(output_dir) / "paired_validation.json", result)
    return result


def _selected_training_parameters(
    algorithm_output_dir: Union[str, Path],
) -> dict[str, object]:
    """Return the Model 3 recipe selected by paired validation."""
    path = Path(algorithm_output_dir) / "paired_validation.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"Melgar paired validation is required before calibration: {path}."
        )
    with path.open(encoding="utf-8") as file:
        return dict(json.load(file)["selected_parameters"])


def _run_training_tuning(
    *,
    method: str,
    n_trials: int,
    num_mc_batches: int,
    batch_size: int,
    output_dir: Union[str, Path],
    asset_root: Union[str, Path],
    training_data_manifest: Union[str, Path],
):
    """Tune five full-effort SNR-specific networks per trial."""
    resource_grid, validated_data_path = _load_training_bank(
        method, training_data_manifest
    )
    estimator_class = ESTIMATOR_CLASSES[method]
    objective = build_cached_nmse_objective(
        estimator_class,
        umi_3gpp_config,
        ebno_dbs=TUNING_EBNO_DBS,
        snr_weights=TUNING_SNR_WEIGHTS,
        num_mc_batches=num_mc_batches,
        batch_size=batch_size,
        resource_grid=resource_grid,
        seed=ALGORITHM_PROBE_SEED,
    )

    def train_candidate(trial, cfg, trial_resource_grid):
        parameters = _suggest_training_parameters(trial, cfg, trial_resource_grid)
        estimator_kwargs, _ = _train_estimator_kwargs(
            method=method,
            resource_grid=resource_grid,
            training_data_manifest=validated_data_path,
            asset_root=asset_root,
            parameters=parameters,
            training_snr_dbs=list(TUNING_EBNO_DBS),
        )
        return estimator_kwargs

    return tune_channel_estimator(
        estimator_class,
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
    method: str,
    num_mc_batches: int,
    batch_size: int,
    output_dir: Union[str, Path],
    asset_root: Union[str, Path],
    training_data_manifest: Union[str, Path],
) -> dict[str, object]:
    """Train and compare one rounded winner with its current recipe."""
    resource_grid, validated_data_path = _load_training_bank(
        method, training_data_manifest
    )
    raw = _load_best_params(output_dir)
    candidates = {
        "current": (
            dict(MODEL2_REFERENCE_PARAMETERS)
            if method == METHOD_MODEL_2
            else resolve_callable_arguments(
                train_model3,
                ("learning_rate", "optimizer_batch_size"),
            )
        ),
        "rounded_winner": _round_training_parameters(raw),
    }
    estimator_class = ESTIMATOR_CLASSES[method]
    objective = build_cached_nmse_objective(
        estimator_class,
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
            method=method,
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
    method: str,
    asset_root: Union[str, Path],
    algorithm_output_dir: Union[str, Path],
) -> dict[str, object]:
    """Load one final 11-SNR bank selected by validation."""
    parameters = _selected_training_parameters(algorithm_output_dir)
    resource_grid = build_resource_grid(umi_3gpp_config)
    loader = load_model2 if method == METHOD_MODEL_2 else load_model3
    states, noise_variances, _ = loader(
        resource_grid,
        umi_3gpp_config,
        **_training_recipe(
            method=method,
            learning_rate=float(parameters["learning_rate"]),
            optimizer_batch_size=int(parameters["optimizer_batch_size"]),
            training_snr_dbs=list(umi_3gpp_config.snr_db_range()),
        ),
        asset_root=asset_root,
    )
    estimator_kwargs = {
        "model_states": states,
        "model_noise_variances": noise_variances,
    }
    if method == METHOD_MODEL_3:
        estimator_kwargs["block_len"] = BLOCK_LEN
    return estimator_kwargs


def _variance_objective(
    *, method: str, estimator_kwargs: dict[str, object],
    num_mc_batches: int, batch_size: int, seed: int,
):
    """Build the cached 10 dB codeword-BLER scale objective."""
    return build_cached_detector_scale_objective(
        ESTIMATOR_CLASSES[method],
        umi_3gpp_config,
        estimator_kwargs,
        scale_argument="err_var_scale",
        ebno_dbs=DETECTOR_CALIBRATION_EBNO_DBS,
        num_mc_batches=num_mc_batches,
        batch_size=batch_size,
        seed=seed,
        detector_metric="bler",
    )


def _run_variance_calibration(
    *, method: str, n_trials: int, num_mc_batches: int, batch_size: int,
    output_dir: Union[str, Path], algorithm_output_dir: Union[str, Path],
    asset_root: Union[str, Path],
):
    """Calibrate one final bank's variance scale at 10 dB."""
    estimator_kwargs = _load_canonical_estimator_kwargs(
        method, asset_root, algorithm_output_dir
    )
    return tune_channel_estimator(
        ESTIMATOR_CLASSES[method],
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
            method=method,
            estimator_kwargs=estimator_kwargs,
            num_mc_batches=num_mc_batches,
            batch_size=batch_size,
            seed=VARIANCE_PROBE_SEED,
        ),
    )


def _run_variance_validation(
    *, method: str, num_mc_batches: int, batch_size: int,
    output_dir: Union[str, Path],
    algorithm_output_dir: Union[str, Path], asset_root: Union[str, Path],
) -> dict[str, object]:
    """Compare the rounded scale winner with the current method scale."""
    raw = _load_best_params(output_dir)
    estimator_class = ESTIMATOR_CLASSES[method]
    current_scale = resolve_constructor_arguments(
        estimator_class,
        ("err_var_scale",),
    )["err_var_scale"]
    candidates = {
        "current": float(f"{float(current_scale):.3g}"),
        "rounded_winner": float(f"{float(raw['err_var_scale']):.3g}"),
    }
    objective = _variance_objective(
        method=method,
        estimator_kwargs=_load_canonical_estimator_kwargs(
            method, asset_root, algorithm_output_dir
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
    method: str = METHOD_MODEL_3,
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
    and 10 dB. Separate studies are conducted for Model 2 and Model 3. Each
    study runs 10 seeded TPE trials. Every trial trains one method-specific
    network for each of the five tuning SNRs and evaluates three batches of
    four channels at each SNR.

    All trials for one method reuse the same training-data bank. Every
    SNR-specific model uses 6,400 training channels and 256 validation
    channels, at most 200 epochs, patience of 20 epochs, and a relative
    minimum validation improvement of 1e-4. The best checkpoint is restored.

    Model 2 derives pilot-vector training examples from every channel. Model 3
    derives frequency-block examples of length 78 and limits the examples
    retained from each generated channel batch to 4,096.

    .. list-table::
        :header-rows: 1

        * - Parameter
          - Search space
        * - ``learning_rate``
          - logarithmic from 1e-5 through 1e-3
        * - ``optimizer_batch_size``
          - categorical set {256, 512, 1024, 2048}

    The ``"training-validation"`` stage retrains and compares the rounded
    winner with the corresponding method's existing training recipe using
    63 batches of four channels at each SNR.

    .. rubric:: Error-Variance Calibration

    The ``"variance"`` stage independently tunes ``err_var_scale`` for the
    selected Model 2 or Model 3 bank. It searches logarithmically from 1e-3
    through 1 using 10 TPE trials. Codeword BLER is evaluated at 10 dB over
    63 batches of four channels.

    The ``"variance-validation"`` stage compares the rounded winner with the
    existing scale for the selected method.

    :param method: Method to tune, either ``"model2"`` or ``"model3"``.
        The default is ``"model3"``.
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
    :param output_dir: Output directory override for the selected method and
        stage.
    :param algorithm_output_dir: Directory containing the method-specific
        training study and paired-validation evidence used to select the
        canonical training recipe.
    :param asset_root: External root containing or receiving trained model
        assets.
    :param training_data_manifest: Manifest of the fixed method-specific
        training and validation data bank. This is required by the training
        and training-validation stages.
    :returns: An Optuna study for tuning stages or a dictionary containing
        paired-validation results for validation stages.
    """
    if method not in METHOD_NAMES:
        raise ValueError(f"`method` must be one of {METHOD_NAMES}.")
    effective_batch_size = (
        umi_3gpp_config.batch_size if batch_size is None else int(batch_size)
    )
    algorithm_output_dir = (
        TRAINING_TUNING_OUTPUT_DIRS[method]
        if algorithm_output_dir is None
        else algorithm_output_dir
    )
    if study in ("training", "training-validation"):
        if training_data_manifest is None:
            raise ValueError(
                "Melgar Model 3 tuning requires `training_data_manifest`."
                if method == METHOD_MODEL_3
                else "Melgar Model 2 tuning requires `training_data_manifest`."
            )
        if study == "training":
            return _run_training_tuning(
                method=method,
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
            method=method,
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
        VARIANCE_TUNING_OUTPUT_DIRS[method]
        if output_dir is None
        else output_dir
    )
    if study == "variance":
        return _run_variance_calibration(
            method=method,
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
            method=method,
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
    """Parse command-line arguments and run one Melgar tuning stage."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        choices=METHOD_NAMES,
        default=METHOD_MODEL_3,
    )
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
        method=args.method,
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
