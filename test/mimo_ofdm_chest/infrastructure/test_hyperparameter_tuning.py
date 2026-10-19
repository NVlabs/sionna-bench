# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the shared hyperparameter-tuning lifecycle."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

import sionna.bench.mimo_ofdm_chest.benchmarking.hyperparameter_tuning as hyperparameter_tuning
from sionna.bench.mimo_ofdm_chest.configs import smoke_config
from sionna.bench.mimo_ofdm_chest.benchmarking.hyperparameter_tuning import (
    build_cached_detector_scale_objective,
    build_cached_nmse_objective,
    evaluate_weighted_detector_probe,
    evaluate_nmse_probe,
    evaluate_weighted_nmse_probe,
    tune_channel_estimator,
)
from sionna.bench.mimo_ofdm_chest.papers.builtin.estimator import LSEstimatorWrapper


class _RandomConsumingLSEstimator(LSEstimatorWrapper):
    """LS estimator whose constructor deliberately advances the Torch RNG."""

    def __init__(self, resource_grid, *, random_draws: int, **kwargs) -> None:
        torch.rand(random_draws)
        super().__init__(resource_grid, **kwargs)


class _ScaledLSEstimator(LSEstimatorWrapper):
    """LS estimator that counts calls and scales only its error variance."""

    num_calls = 0

    def __init__(
        self,
        resource_grid,
        *,
        err_var_scale: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(resource_grid, **kwargs)
        self._err_var_scale = err_var_scale

    def call(self, y, no):
        """Return unchanged LS estimates and scaled error variances."""
        type(self).num_calls += 1
        estimate, error_variance = super().call(y, no)
        return estimate, self._err_var_scale * error_variance


def test_weighted_nmse_probe_reuses_seeded_sionna_channels() -> None:
    """Repeated probes reproduce Sionna channels, signals, and NMSE exactly."""
    cfg = replace(smoke_config, device="cpu")
    kwargs = {
        "ebno_dbs": (-5.0, 5.0),
        "num_mc_batches": 1,
        "batch_size": 1,
        "seed": 1234,
    }

    first = evaluate_weighted_nmse_probe(
        LSEstimatorWrapper,
        cfg,
        {"interpolation_type": "lin"},
        **kwargs,
    )
    second = evaluate_weighted_nmse_probe(
        LSEstimatorWrapper,
        cfg,
        {"interpolation_type": "lin"},
        **kwargs,
    )

    assert first == second


def test_weighted_nmse_probe_preserves_cross_snr_channel_diversity(
    monkeypatch,
) -> None:
    """SNR points use distinct consecutive draws that repeat across probes."""
    recorded_draws = []

    def record_seeded_draw(system, batch_size, ebno_db):
        del system, batch_size
        draw = torch.rand(())
        recorded_draws.append((float(ebno_db), float(draw)))
        channel = torch.ones(1, dtype=torch.complex64)
        return (1.0 + draw) * channel, channel

    monkeypatch.setattr(
        hyperparameter_tuning,
        "_channel_estimation_batch",
        record_seeded_draw,
    )
    cfg = replace(smoke_config, device="cpu")
    kwargs = {
        "ebno_dbs": (-5.0, 5.0),
        "num_mc_batches": 2,
        "batch_size": 1,
        "seed": 1234,
    }

    evaluate_weighted_nmse_probe(
        LSEstimatorWrapper,
        cfg,
        {"interpolation_type": "lin"},
        **kwargs,
    )
    first_probe = recorded_draws.copy()
    recorded_draws.clear()
    evaluate_weighted_nmse_probe(
        LSEstimatorWrapper,
        cfg,
        {"interpolation_type": "lin"},
        **kwargs,
    )

    assert recorded_draws == first_probe
    assert [ebno_db for ebno_db, _ in first_probe] == [-5.0, -5.0, 5.0, 5.0]
    assert len({draw for _, draw in first_probe}) == len(first_probe)


@pytest.mark.parametrize(
    ("probe", "snr_kwargs"),
    (
        (evaluate_nmse_probe, {"ebno_db": 5.0}),
        (evaluate_weighted_nmse_probe, {"ebno_dbs": (-5.0, 5.0)}),
    ),
    ids=("single-snr", "weighted-snr"),
)
def test_probe_seed_is_applied_after_estimator_construction(
    probe,
    snr_kwargs,
) -> None:
    """Constructor RNG consumption cannot change the probe channel samples."""
    cfg = replace(smoke_config, device="cpu")
    probe_kwargs = {
        "num_mc_batches": 1,
        "batch_size": 1,
        "seed": 1234,
        **snr_kwargs,
    }

    one_draw = probe(
        _RandomConsumingLSEstimator,
        cfg,
        {"interpolation_type": "lin", "random_draws": 1},
        **probe_kwargs,
    )
    many_draws = probe(
        _RandomConsumingLSEstimator,
        cfg,
        {"interpolation_type": "lin", "random_draws": 1000},
        **probe_kwargs,
    )

    assert one_draw == many_draws


def test_custom_estimator_objective_uses_shared_study_lifecycle(
    tmp_path: Path,
) -> None:
    """A custom objective receives merged kwargs and resumes normal summaries."""
    pytest.importorskip("optuna")
    observed_kwargs = []

    class DummyEstimator:
        pass

    def suggest_parameters(trial, cfg, resource_grid):
        del cfg, resource_grid
        return {"x": trial.suggest_float("x", 0.25, 0.25)}

    def estimator_objective(estimator_kwargs):
        observed_kwargs.append(dict(estimator_kwargs))
        return (float(estimator_kwargs["x"]) - 0.5) ** 2

    cfg = replace(smoke_config, device="cpu")
    study = tune_channel_estimator(
        DummyEstimator,
        cfg,
        suggest_parameters,
        tmp_path,
        fixed_estimator_kwargs={"fixed": 7},
        n_trials=1,
        sampler_seed=1,
        estimator_objective=estimator_objective,
    )
    resumed_study = tune_channel_estimator(
        DummyEstimator,
        cfg,
        suggest_parameters,
        tmp_path,
        fixed_estimator_kwargs={"fixed": 7},
        n_trials=1,
        sampler_seed=1,
        estimator_objective=estimator_objective,
    )

    with (tmp_path / "best_params.json").open(encoding="utf-8") as file:
        summary = json.load(file)
    assert observed_kwargs == [
        {"fixed": 7, "x": 0.25},
        {"fixed": 7, "x": 0.25},
    ]
    assert study.best_value == 0.0625
    assert resumed_study.best_value == 0.0625
    assert summary["best_value"] == 0.0625
    assert summary["n_trials"] == 2
    assert summary["requested_n_trials"] == 1
    assert summary["sampler_seed"] == 1
    assert summary["probe_seed"] == cfg.seed
    assert summary["optimization_elapsed_seconds"] >= 0.0
    assert (tmp_path / "trials.csv").is_file()


def test_cached_detector_scale_objective_reuses_channel_estimates() -> None:
    """Cached scale probes match direct probes without rerunning estimation."""
    cfg = replace(smoke_config, device="cpu")
    probe_kwargs = {
        "ebno_dbs": (-5.0, 5.0),
        "num_mc_batches": 1,
        "batch_size": 1,
        "seed": 1234,
    }
    _ScaledLSEstimator.num_calls = 0
    objective = build_cached_detector_scale_objective(
        _ScaledLSEstimator,
        cfg,
        {"interpolation_type": "lin"},
        **probe_kwargs,
    )
    calls_after_caching = _ScaledLSEstimator.num_calls

    cached_value = objective({"err_var_scale": 0.5})
    repeated_value = objective({"err_var_scale": 0.5})
    calls_after_objectives = _ScaledLSEstimator.num_calls
    direct_value = evaluate_weighted_detector_probe(
        _ScaledLSEstimator,
        cfg,
        {"interpolation_type": "lin", "err_var_scale": 0.5},
        **probe_kwargs,
    )

    assert cached_value == direct_value
    assert repeated_value == cached_value
    assert calls_after_caching == 2
    assert calls_after_objectives == calls_after_caching


def test_cached_nmse_objective_matches_direct_weighted_probe() -> None:
    """Cached and direct weighted NMSE use the same deterministic samples."""
    cfg = replace(smoke_config, device="cpu")
    probe_kwargs = {
        "ebno_dbs": (-5.0, 5.0),
        "num_mc_batches": 1,
        "batch_size": 1,
        "seed": 1234,
    }
    estimator_kwargs = {"interpolation_type": "lin"}
    objective = build_cached_nmse_objective(
        LSEstimatorWrapper,
        cfg,
        **probe_kwargs,
    )

    cached_value = objective(estimator_kwargs)
    direct_value = evaluate_weighted_nmse_probe(
        LSEstimatorWrapper,
        cfg,
        estimator_kwargs,
        **probe_kwargs,
    )

    assert cached_value == direct_value


def test_cached_detector_objective_supports_bler_only() -> None:
    """The cached detector objective matches direct BLER-only evaluation."""
    cfg = replace(smoke_config, device="cpu")
    probe_kwargs = {
        "ebno_dbs": (5.0,),
        "num_mc_batches": 1,
        "batch_size": 1,
        "seed": 1234,
        "detector_metric": "bler",
    }
    fixed_kwargs = {"interpolation_type": "lin"}
    objective = build_cached_detector_scale_objective(
        _ScaledLSEstimator,
        cfg,
        fixed_kwargs,
        **probe_kwargs,
    )

    cached_value = objective({"err_var_scale": 0.5})
    direct_value = evaluate_weighted_detector_probe(
        _ScaledLSEstimator,
        cfg,
        {**fixed_kwargs, "err_var_scale": 0.5},
        **probe_kwargs,
    )

    assert cached_value == direct_value
