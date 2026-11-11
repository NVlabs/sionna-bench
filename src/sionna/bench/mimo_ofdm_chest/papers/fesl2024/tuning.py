# SPDX-FileCopyrightText: Copyright (c) 2026 Benedikt Fesl
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import math
import resource
import statistics
import sys
import time
from dataclasses import replace
from importlib.metadata import version
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from sionna.phy.ofdm import LSChannelEstimator
from sionna.phy.utils import ebnodb2no

from sionna.bench.mimo_ofdm_chest.benchmarking import (
    OFDMChannelEstimationSystem,
    build_resource_grid,
    build_cached_nmse_objective,
    build_cached_detector_scale_objective,
    evaluate_weighted_nmse_probe,
    set_random_seed,
    tune_channel_estimator,
)
from sionna.bench.mimo_ofdm_chest.benchmarking.result_store import (
    resolve_callable_arguments,
    resolve_constructor_arguments,
)
from sionna.bench.mimo_ofdm_chest.configs import SystemConfig, umi_3gpp_config
from sionna.bench.mimo_ofdm_chest.persistence import sha256_file, write_json_atomic

from .asset import (
    _angular_batch,
    _runtime,
    generate_training_data,
    load_model,
    load_training_data,
    train_model,
)
from .estimator import (
    DiffusionChannelEstimator,
    _DiffusionSchedule,
    _NoisePredictionCNN,
)

__all__ = ["run_tuning"]


def _inference_arguments(
    cfg: SystemConfig, state: dict, power: torch.Tensor
) -> dict[str, Any]:
    return {
        "model_state": state,
        "channel_power": power,
        "receive_array_shape": (cfg.bs_num_rows, cfg.bs_num_cols),
    }


def _study_context(
    cfg: SystemConfig, kind: str, inputs: dict[str, Any]
) -> dict[str, Any]:
    runtime = _runtime()
    return {
        "context_version": 1,
        "study": kind,
        "completed_trial_target": 10,
        "config_fingerprint": cfg.config_fingerprint(),
        "source_files_sha256": runtime["source_files_sha256"],
        "dependencies": {
            key: runtime[key] for key in ("python", "torch", "numpy", "sionna")
        }
        | {"optuna": version("optuna"), "scipy": version("scipy")},
        "inputs": inputs,
    }


def _check_context(
    output: Path, expected: dict[str, Any], *, create: bool = False
) -> None:
    """Bind resumed studies and paired validation to identical data/model/code."""
    path = output / "study_context.json"
    if path.exists():
        if json.loads(path.read_text()) != expected:
            raise ValueError(
                "Frozen study context changed (data, model, configuration or source)."
            )
    elif create:
        if (output / "study.db").exists():
            raise ValueError(
                "Existing study database has no frozen context; refusing to adopt old trials."
            )
        write_json_atomic(path, expected)
    else:
        raise ValueError(
            "Missing frozen study context; paired validation cannot infer it."
        )


def _bounded_study(
    output: Path,
    *,
    kind: str,
    timeout: Optional[float],
    max_failed_trials: int = 0,
    **kwargs: Any,
) -> Any:
    """Use the shared lifecycle for ten total completed trials, including resumes.

    Default failure budget is zero: a failed fit is reported and stops work.
    Explicitly increasing it requires the separate full-study cost approval.
    Timeout is a soft per-invocation Optuna limit, checked between fits; an
    in-flight fit is never cancelled. It can overrun by that fit's duration.
    A timeout leaves a partial study; it never triggers parameter selection.
    Count/context restoration does not restore the shared TPE sampler RNG.
    Exact reproducible continuation of a partial stochastic study is unsupported;
    the reviewed detached experiments refuse partial-study restarts.
    """
    import optuna

    if max_failed_trials < 0:
        raise ValueError("max_failed_trials must be nonnegative.")
    storage = f"sqlite:///{(output / 'study.db').resolve()}"
    name = f"fesl2024-{kind}"
    deadline = None if timeout is None else time.perf_counter() + timeout
    study = None
    while True:
        if (output / "study.db").exists():
            try:
                study = optuna.load_study(study_name=name, storage=storage)
            except KeyError:
                study = None
        trials = [] if study is None else study.trials
        completed = sum(t.state == optuna.trial.TrialState.COMPLETE for t in trials)
        failed = sum(t.state == optuna.trial.TrialState.FAIL for t in trials)
        if completed > 10 or failed > max_failed_trials:
            raise RuntimeError(
                "Study exceeds its approved completed/failure budget; no automatic extra fits."
            )
        if completed == 10:
            return study
        remaining_time = None if deadline is None else deadline - time.perf_counter()
        if remaining_time is not None and remaining_time <= 0:
            return study
        try:
            study = tune_channel_estimator(
                output_dir=output,
                n_trials=10 - completed,
                study_name=name,
                storage=storage,
                timeout=remaining_time,
                **kwargs,
            )
        except Exception:
            current = optuna.load_study(study_name=name, storage=storage)
            new_failed = sum(
                t.state == optuna.trial.TrialState.FAIL for t in current.trials
            )
            if new_failed <= failed or new_failed > max_failed_trials:
                raise
            continue
        count = sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
        if count == 10 or (deadline is not None and time.perf_counter() >= deadline):
            return study
        if count == completed:
            raise RuntimeError(
                "Study made no completed-trial progress; stopped without another fit."
            )


def _synchronize(device: str) -> None:
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


def _throughput(cfg: SystemConfig, arrays: dict, power: float) -> list[dict[str, Any]]:
    records = []
    shape = (cfg.bs_num_rows, cfg.bs_num_cols)
    examples = arrays["train"].reshape(-1, cfg.num_rx_ant)
    for batch_size in (128, 256, 512, 1024, 2048):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(24243000)
            model = _NoisePredictionCNN().to(cfg.device)
        schedule = _DiffusionSchedule(device=cfg.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        rng = torch.Generator().manual_seed(24244000)
        epsilon = (torch.randn(batch_size, 2, *shape, generator=rng) / math.sqrt(2)).to(
            cfg.device
        )
        steps = torch.randint(100, (batch_size,), generator=rng).to(cfg.device)
        durations = []
        for step in range(12):
            offset = (step * batch_size) % (len(examples) - batch_size + 1)
            _synchronize(cfg.device)
            started = time.perf_counter()
            clean = _angular_batch(
                examples[offset : offset + batch_size], shape, power, cfg.device
            )
            loss = torch.nn.functional.mse_loss(
                model(schedule.forward_noise(clean, epsilon, steps), steps), epsilon
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            _synchronize(cfg.device)
            if step >= 2:
                durations.append(time.perf_counter() - started)
        validation_times = []
        with torch.no_grad():
            for _ in range(10):
                _synchronize(cfg.device)
                started = time.perf_counter()
                model(schedule.forward_noise(clean, epsilon, steps), steps)
                _synchronize(cfg.device)
                validation_times.append(time.perf_counter() - started)
        train_seconds = statistics.mean(durations)
        validation_seconds = statistics.mean(validation_times)
        epoch = (
            math.ceil(7987200 / batch_size) * train_seconds
            + math.ceil(319488 / batch_size) * validation_seconds
        )
        records.append(
            {
                "batch_size": batch_size,
                "measured_batches": 10,
                "train_step_seconds": train_seconds,
                "validation_step_seconds": validation_seconds,
                "examples_per_second": batch_size / train_seconds,
                "extrapolated_full_epoch_seconds": epoch,
                "eleven_fits_200_epochs_seconds": 11 * 200 * epoch,
                "safety_factor_two_training_seconds": 2 * 11 * 200 * epoch,
            }
        )
    return records


def _development_probe(
    cfg: SystemConfig, arguments: list[dict[str, Any]]
) -> dict[str, Any]:
    rg = build_resource_grid(cfg)
    models = [
        DiffusionChannelEstimator(rg, **kw, device=cfg.device, precision=cfg.precision)
        for kw in arguments
    ]
    references = [
        LSChannelEstimator(
            rg, interpolation_type=policy, device=cfg.device, precision=cfg.precision
        )
        for policy in ("nn", "lin_time_avg")
    ]
    system = OFDMChannelEstimationSystem(cfg, None, rg)
    no = ebnodb2no(
        10.0,
        cfg.num_bits_per_symbol,
        cfg.coderate,
        rg,
        device=cfg.device,
        precision=cfg.precision,
    )
    numerator = np.zeros(len(models) + 2)
    inference_times = [[] for _ in models]
    denominator = 0.0
    generation_seconds = 0.0
    for batch in range(4):
        set_random_seed(24242000 + batch)
        started = time.perf_counter()
        system.channel_source.sample(4)
        _, transmitted = system._tx_pipeline(4)
        received, truth = system.channel_source(transmitted, no)
        _synchronize(cfg.device)
        generation_seconds += time.perf_counter() - started
        denominator += float(truth.abs().square().sum())
        with torch.no_grad():
            for index, estimator in enumerate(models + references):
                _synchronize(cfg.device)
                started = time.perf_counter()
                estimate, variance = estimator(received, no)
                _synchronize(cfg.device)
                if (
                    not bool(torch.isfinite(estimate).all())
                    or not bool(torch.isfinite(variance).all())
                    or bool((variance < 0).any())
                ):
                    raise FloatingPointError(
                        "Invalid development-probe estimator output."
                    )
                if index < len(models):
                    inference_times[index].append(time.perf_counter() - started)
                numerator[index] += float((estimate - truth).abs().square().sum())
    detector_seconds = None
    with torch.no_grad():
        estimate, variance = models[1](received, no)
        _synchronize(cfg.device)
        started = time.perf_counter()
        decoded = system._detect_and_decode(received, estimate, variance, no, 4)
        _synchronize(cfg.device)
        detector_seconds = time.perf_counter() - started
    return {
        "ebno_db": 10.0,
        "independent_channels": 16,
        "seed_base": 24242000,
        "nmse": (numerator / denominator).tolist(),
        "nmse_order": [
            "lr1e-5-b128",
            "lr1e-4-b512",
            "lr1e-3-b2048",
            "ls-nn",
            "ls-lin-time-avg",
        ],
        "inference_batch_seconds": [
            statistics.mean(times) for times in inference_times
        ],
        "channel_tx_generation_seconds": generation_seconds,
        "detector_decode_four_channel_seconds": detector_seconds,
        "decoded_finite": bool(torch.isfinite(decoded).all()),
    }


def run_feasibility(
    cfg: SystemConfig, *, asset_root: str | Path, output_dir: str | Path
) -> dict[str, Any]:
    """Run only the approved 64/16 development study and representative throughput."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rg = build_resource_grid(cfg)
    started = time.perf_counter()
    dataset_path = generate_training_data(
        rg,
        cfg,
        asset_root=asset_root,
        artifact_id="feasibility-data",
        num_train_channels=64,
        num_validation_channels=16,
        generation_batch_size=4,
        train_seed=24240000,
        validation_seed=24241000,
        purpose="feasibility",
    )
    generation_seconds = time.perf_counter() - started
    started = time.perf_counter()
    arrays, power, data_manifest = load_training_data(
        rg, cfg, manifest_path=dataset_path
    )
    loading_seconds = time.perf_counter() - started
    records, inference = [], []
    for index, (lr, batch) in enumerate(((1e-5, 128), (1e-4, 512), (1e-3, 2048))):
        state, model_power, path = train_model(
            rg,
            cfg,
            training_data_manifest=dataset_path,
            asset_root=asset_root,
            artifact_id=f"feasibility-{index}",
            epochs=3,
            optimizer_batch_size=batch,
            learning_rate=lr,
            patience_epochs=0,
            initialization_seed=24245000,
            noise_seed=24246000,
            validation_noise_seed=24247000,
        )
        manifest = json.loads(path.read_text())
        records.append(
            {
                "learning_rate": lr,
                "optimizer_batch_size": batch,
                "manifest_path": str(path),
                "manifest_sha256": sha256_file(path),
                "training": manifest["metadata"],
            }
        )
        inference.append(_inference_arguments(cfg, state, model_power))
        write_json_atomic(output / "development_training.json", records)
    probe = _development_probe(cfg, inference)
    throughput = _throughput(cfg, arrays, power)
    # Unix reports bytes on macOS and KiB on Linux. This is cumulative process peak.
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_bytes = int(peak if sys.platform == "darwin" else peak * 1024)
    report = {
        "purpose": "development-only-not-canonical-selection",
        "runtime": _runtime(),
        "dataset_manifest": str(dataset_path),
        "dataset_manifest_sha256": sha256_file(dataset_path),
        "dataset": data_manifest,
        "generation_wall_seconds": generation_seconds,
        "strict_loading_seconds": loading_seconds,
        "models": records,
        "probe": probe,
        "throughput": throughput,
        "process_peak_rss_bytes": peak_bytes,
        "full_dataset_payload_bytes": 1063256064,
        "full_epoch_noise_bytes": 7987200 * 2 * 4 * 4 * 4,
        "scope": "No canonical bank, Optuna study, variance calibration or final benchmark executed.",
    }
    write_json_atomic(output / "feasibility.json", report)
    return report


def run_tuning(
    cfg: SystemConfig,
    *,
    asset_root: str | Path,
    output_dir: str | Path,
    study: str = "training",
    training_data_manifest: Optional[str | Path] = None,
    artifact_id: str = "canonical",
    timeout: Optional[float] = None,
    max_failed_trials: int = 0,
) -> Any:
    r"""

    .. rubric:: Training Hyperparameter Tuning

    The ``"training"`` study minimizes weighted NMSE at -10, -5, 0, 5, and
    10 dB. It runs 10 seeded TPE trials, the first of which uses the training
    defaults of ``train_model()``. Every trial trains one diffusion model that
    serves all SNRs and evaluates three batches of four channels at each SNR.

    All trials reuse the same training-data bank of 6,400 training and 256
    validation channels, at most 200 epochs, patience of 20 epochs, and a
    relative minimum validation improvement of 1e-4. The best checkpoint is
    restored.

    .. list-table::
        :header-rows: 1

        * - Parameter
          - Search space
        * - ``learning_rate``
          - logarithmic from 1e-5 through 1e-3
        * - ``optimizer_batch_size``
          - categorical set {128, 256, 512, 1024, 2048}

    ``validate_training_recipe()`` compares the rounded winner with the training
    defaults using 63 batches of four channels at each SNR.

    This study is the contributing authors' tooling. The canonical recipe was
    selected by the authors in their own external study and was not reproduced
    by the maintainers; see the paper README.

    .. rubric:: Error-Variance Calibration

    The ``"variance"`` study tunes ``err_var_scale`` logarithmically from
    1e-3 through 1 using 10 TPE trials. Codeword BLER is evaluated at 10 dB
    over 63 batches of four channels using the frozen canonical model.
    ``validate_variance_scale()`` compares the rounded winner with the existing
    scale.

    :param cfg: System configuration of the canonical benchmark.
    :param asset_root: External root containing the training data and receiving
        the trial model assets.
    :param output_dir: Directory for the Optuna study database, the frozen study
        context, and the study summary.
    :param study: Study to execute: ``"training"`` or ``"variance"``.
    :param training_data_manifest: Manifest of the fixed training-data bank.
        Required by the training study.
    :param artifact_id: Identifier of the frozen model asset calibrated by the
        variance study.
    :param timeout: Soft time limit in seconds for this invocation, checked
        between trials. A partial study is continued by calling again.
    :param max_failed_trials: Number of failed trials tolerated before the study
        stops. The default of zero stops at the first failure.
    :returns: The completed Optuna study.
    """
    rg = build_resource_grid(cfg)
    output = Path(output_dir)
    if study == "training":
        if training_data_manifest is None:
            raise ValueError(
                "Training study requires an explicit immutable training-data manifest."
            )
        _, _, dataset = load_training_data(
            rg, cfg, manifest_path=training_data_manifest
        )
        if dataset["generation_recipe"]["purpose"] != "canonical":
            raise ValueError("Full tuning refuses smoke/development data.")
        if (
            dataset["generation_recipe"]["num_train_channels"],
            dataset["generation_recipe"]["num_validation_channels"],
        ) != (6400, 256):
            raise ValueError(
                "The provisional full study requires exactly 6,400/256 system realizations."
            )
        _check_context(
            output,
            _study_context(
                cfg,
                "training",
                {
                    "dataset_manifest_sha256": sha256_file(training_data_manifest),
                    "dataset_files": dataset["files"],
                },
            ),
            create=True,
        )
        snrs = [-10.0, -5.0, 0.0, 5.0, 10.0]
        objective = build_cached_nmse_objective(
            DiffusionChannelEstimator,
            cfg,
            ebno_dbs=snrs,
            resource_grid=rg,
            snr_weights=[10 ** (s / 10) for s in snrs],
            num_mc_batches=3,
            batch_size=4,
            seed=24310000,
        )

        def suggest(trial: Any, config: Any, grid: Any) -> dict[str, Any]:
            lr = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
            batch = trial.suggest_categorical(
                "optimizer_batch_size", [128, 256, 512, 1024, 2048]
            )
            state, power, manifest = train_model(
                grid,
                config,
                asset_root=asset_root,
                training_data_manifest=training_data_manifest,
                artifact_id=f"trial-{trial.number}",
                learning_rate=lr,
                optimizer_batch_size=batch,
            )
            trial.set_user_attr("model_manifest", str(manifest))
            trial.set_user_attr("model_manifest_sha256", sha256_file(manifest))
            return _inference_arguments(cfg, state, power)

        return _bounded_study(
            output,
            kind="training",
            timeout=timeout,
            max_failed_trials=max_failed_trials,
            estimator_class=DiffusionChannelEstimator,
            cfg=cfg,
            suggest_estimator_kwargs=suggest,
            ebno_dbs=snrs,
            snr_weights=[10 ** (s / 10) for s in snrs],
            num_mc_batches=3,
            batch_size=4,
            sampler_seed=24300000,
            probe_seed=24310000,
            enqueued_trials=[
                resolve_callable_arguments(
                    train_model, ("learning_rate", "optimizer_batch_size")
                )
            ],
            estimator_objective=objective,
        )
    if study == "variance":
        state, power, manifest = load_model(
            rg,
            cfg,
            asset_root=asset_root,
            artifact_id=artifact_id,
            expected_purpose="canonical",
        )
        _check_context(
            output,
            _study_context(
                cfg,
                "variance",
                {
                    "model_manifest": str(manifest.resolve()),
                    "model_manifest_sha256": sha256_file(manifest),
                },
            ),
            create=True,
        )
        arguments = _inference_arguments(cfg, state, power)
        objective = build_cached_detector_scale_objective(
            DiffusionChannelEstimator,
            cfg,
            fixed_estimator_kwargs=arguments,
            scale_argument="err_var_scale",
            ebno_dbs=[10.0],
            num_mc_batches=63,
            batch_size=4,
            seed=24330000,
            detector_metric="bler",
        )

        def suggest_scale(trial: Any, config: Any, grid: Any) -> dict[str, Any]:
            return {
                "err_var_scale": trial.suggest_float(
                    "err_var_scale", 1e-3, 1.0, log=True
                )
            }

        return _bounded_study(
            output,
            kind="variance",
            timeout=timeout,
            max_failed_trials=max_failed_trials,
            estimator_class=DiffusionChannelEstimator,
            cfg=cfg,
            suggest_estimator_kwargs=suggest_scale,
            fixed_estimator_kwargs=arguments,
            ebno_dbs=[10.0],
            objective_metric="detector",
            detector_metric="bler",
            num_mc_batches=63,
            batch_size=4,
            sampler_seed=24350000,
            probe_seed=24330000,
            estimator_objective=objective,
        )
    raise ValueError(
        "study must be training or variance; feasibility has a separate entry point."
    )


def validate_training_recipe(
    cfg: SystemConfig,
    *,
    asset_root: str | Path,
    output_dir: str | Path,
    training_data_manifest: str | Path,
) -> dict[str, Any]:
    """Fit the rounded winner and select its exact checkpoint by paired NMSE.

    This is a full-quality training operation, separate from the inexpensive
    feasibility command. It is not authorized merely by approving code/smoke.
    """
    import optuna

    output = Path(output_dir)
    summary = json.loads((output / "best_params.json").read_text())
    rg = build_resource_grid(cfg)
    _, _, dataset = load_training_data(rg, cfg, manifest_path=training_data_manifest)
    dataset_hash = sha256_file(training_data_manifest)
    _check_context(
        output,
        _study_context(
            cfg,
            "training",
            {
                "dataset_manifest_sha256": dataset_hash,
                "dataset_files": dataset["files"],
            },
        ),
    )
    study = optuna.load_study(
        study_name=summary["study_name"], storage=summary["storage"]
    )
    completed = [
        trial
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE
    ]
    if len(completed) != 10:
        raise ValueError(
            "Paired selection requires exactly ten successful training trials."
        )
    for trial in completed:
        path = Path(trial.user_attrs["model_manifest"])
        if sha256_file(path) != trial.user_attrs["model_manifest_sha256"]:
            raise ValueError("Trial checkpoint manifest changed after its evaluation.")
        recipe = json.loads(path.read_text())["generation_recipe"]
        if (
            recipe.get("dataset_manifest_sha256") != dataset_hash
            or recipe.get("dataset_files") != dataset["files"]
        ):
            raise ValueError(
                "Trial checkpoints do not share the supplied immutable training dataset."
            )
    reference_recipe = resolve_callable_arguments(
        train_model, ("learning_rate", "optimizer_batch_size")
    )
    reference = next(
        trial for trial in completed if trial.params == reference_recipe
    )
    rounded = {
        "learning_rate": float(f"{study.best_params['learning_rate']:.3g}"),
        "optimizer_batch_size": int(study.best_params["optimizer_batch_size"]),
    }
    ref_path = Path(reference.user_attrs["model_manifest"])
    if sha256_file(ref_path) != reference.user_attrs["model_manifest_sha256"]:
        raise ValueError("Reference checkpoint manifest changed after its trial.")
    ref_state, ref_power, loaded_ref = load_model(
        rg, cfg, asset_root=asset_root, artifact_id=ref_path.parent.name
    )
    if (
        loaded_ref.resolve() != ref_path.resolve()
        or sha256_file(loaded_ref) != reference.user_attrs["model_manifest_sha256"]
    ):
        raise ValueError(
            "Loaded reference asset differs from the exact trial manifest."
        )
    if rounded == study.best_params:
        candidate_path = Path(study.best_trial.user_attrs["model_manifest"])
        if (
            sha256_file(candidate_path)
            != study.best_trial.user_attrs["model_manifest_sha256"]
        ):
            raise ValueError("Winning checkpoint manifest changed after its trial.")
        state, power, loaded_candidate = load_model(
            rg, cfg, asset_root=asset_root, artifact_id=candidate_path.parent.name
        )
        if (
            loaded_candidate.resolve() != candidate_path.resolve()
            or sha256_file(loaded_candidate)
            != study.best_trial.user_attrs["model_manifest_sha256"]
        ):
            raise ValueError(
                "Loaded candidate asset differs from the exact trial manifest."
            )
    else:
        state, power, candidate_path = train_model(
            rg,
            cfg,
            asset_root=asset_root,
            training_data_manifest=training_data_manifest,
            artifact_id="rounded-candidate",
            **rounded,
        )
    snrs = [-10.0, -5.0, 0.0, 5.0, 10.0]
    scores = []
    for weights, normalization in ((ref_state, ref_power), (state, power)):
        scores.append(
            evaluate_weighted_nmse_probe(
                DiffusionChannelEstimator,
                cfg,
                _inference_arguments(cfg, weights, normalization),
                ebno_dbs=snrs,
                snr_weights=[10 ** (s / 10) for s in snrs],
                num_mc_batches=63,
                batch_size=4,
                resource_grid=rg,
                seed=24320000,
            )
        )
    selected = candidate_path if scores[1] < scores[0] else ref_path
    record = {
        "raw_winner": study.best_params,
        "rounded_candidate": rounded,
        "reference_nmse": scores[0],
        "rounded_nmse": scores[1],
        "paired_seed": 24320000,
        "channels_per_snr": 252,
        "selected_manifest": str(selected),
        "selected_manifest_sha256": sha256_file(selected),
        "reference_manifest": str(ref_path),
        "candidate_manifest": str(candidate_path),
        "dataset_manifest_sha256": sha256_file(training_data_manifest),
        "runtime": _runtime(),
        "policy": "strict-improvement-otherwise-reference-no-retraining",
    }
    write_json_atomic(output / "paired_selection.json", record)
    return record


def validate_variance_scale(
    cfg: SystemConfig,
    *,
    asset_root: str | Path,
    output_dir: str | Path,
    artifact_id: str,
) -> dict[str, Any]:
    """Validate a rounded scale against the constructor default on paired data.

    The frozen canonical model is loaded strictly. A completed study is bound
    to its original source context, so old records must be inspected with their
    frozen source snapshot after the selected constructor default is adopted.
    """
    import optuna

    output = Path(output_dir)
    summary = json.loads((output / "best_params.json").read_text())
    study = optuna.load_study(
        study_name=summary["study_name"], storage=summary["storage"]
    )
    if sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials) != 10:
        raise ValueError(
            "Variance validation requires exactly ten completed calibration trials."
        )
    # SQLite study state is authoritative if a summary write was interrupted.
    raw = float(study.best_params["err_var_scale"])
    rounded = float(f"{raw:.3g}")
    incumbent = resolve_constructor_arguments(
        DiffusionChannelEstimator, ("err_var_scale",)
    )["err_var_scale"]
    rg = build_resource_grid(cfg)
    state, power, manifest = load_model(
        rg,
        cfg,
        asset_root=asset_root,
        artifact_id=artifact_id,
        expected_purpose="canonical",
    )
    _check_context(
        output,
        _study_context(
            cfg,
            "variance",
            {
                "model_manifest": str(manifest.resolve()),
                "model_manifest_sha256": sha256_file(manifest),
            },
        ),
    )
    arguments = _inference_arguments(cfg, state, power)
    objective = build_cached_detector_scale_objective(
        DiffusionChannelEstimator,
        cfg,
        fixed_estimator_kwargs=arguments,
        ebno_dbs=[10.0],
        num_mc_batches=63,
        batch_size=4,
        seed=24340000,
        detector_metric="bler",
        resource_grid=rg,
    )
    reference_bler = objective({**arguments, "err_var_scale": incumbent})
    rounded_bler = objective({**arguments, "err_var_scale": rounded})
    selected = rounded if rounded_bler < reference_bler else incumbent
    record = {
        "raw_winner": raw,
        "rounded_candidate": rounded,
        "incumbent": incumbent,
        "incumbent_bler": reference_bler,
        "rounded_bler": rounded_bler,
        "selected_scale": selected,
        "paired_seed": 24340000,
        "independent_channels": 252,
        "model_manifest": str(manifest),
        "model_manifest_sha256": sha256_file(manifest),
        "runtime": _runtime(),
    }
    write_json_atomic(output / "paired_variance_validation.json", record)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--study",
        choices=(
            "feasibility",
            "training",
            "training-validation",
            "variance",
            "variance-validation",
        ),
    )
    parser.add_argument("--prepare-training-data", action="store_true")
    parser.add_argument("--training-data-manifest", type=Path)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--artifact-id", default="canonical")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--torch-threads", type=int)
    parser.add_argument("--timeout", type=float)
    parser.add_argument("--max-failed-trials", type=int, default=0)
    args = parser.parse_args()
    if args.torch_threads is not None:
        torch.set_num_threads(args.torch_threads)
    cfg = replace(umi_3gpp_config, device=args.device)
    if args.prepare_training_data:
        if args.study:
            parser.error("Data preparation and study execution are separate commands.")
        print(
            generate_training_data(
                build_resource_grid(cfg), cfg, asset_root=args.asset_root
            )
        )
    elif args.study == "feasibility":
        print(
            run_feasibility(
                cfg, asset_root=args.asset_root, output_dir=args.output_dir
            )["scope"]
        )
    elif args.study == "training-validation":
        if args.training_data_manifest is None:
            parser.error("training-validation requires --training-data-manifest")
        print(
            validate_training_recipe(
                cfg,
                asset_root=args.asset_root,
                output_dir=args.output_dir,
                training_data_manifest=args.training_data_manifest,
            )
        )
    elif args.study == "variance-validation":
        print(
            validate_variance_scale(
                cfg,
                asset_root=args.asset_root,
                output_dir=args.output_dir,
                artifact_id=args.artifact_id,
            )
        )
    elif args.study:
        run_tuning(
            cfg,
            asset_root=args.asset_root,
            output_dir=args.output_dir,
            study=args.study,
            training_data_manifest=args.training_data_manifest,
            artifact_id=args.artifact_id,
            timeout=args.timeout,
            max_failed_trials=args.max_failed_trials,
        )
    else:
        parser.error("Choose --prepare-training-data or --study.")


if __name__ == "__main__":
    main()
