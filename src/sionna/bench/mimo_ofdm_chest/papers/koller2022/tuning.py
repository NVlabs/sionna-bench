# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Union

from sionna.bench.mimo_ofdm_chest.benchmarking import (
    build_cached_nmse_objective,
    build_resource_grid,
    tune_channel_estimator,
)
from sionna.bench.mimo_ofdm_chest.configs import umi_3gpp_config

from .asset import (
    METHOD_NAMES,
    METHOD_SPATIAL,
    METHOD_SPECTRAL,
    fit_model,
    generate_fitting_data,
    load_model,
)
from .estimator import (
    SpatialGMMChannelEstimator,
    SpectralGMMChannelEstimator,
)

__all__ = ["run_tuning"]

TUNING_EBNO_DBS = (-10.0, -5.0, 0.0, 5.0, 10.0)
TUNING_SNR_WEIGHTS = tuple(
    10.0 ** (ebno_db / 10.0) for ebno_db in TUNING_EBNO_DBS
)
COMPONENT_COUNTS = (5, 10, 15, 20)
TRAIN_BATCHES = 100
TRAIN_BATCH_SIZE = 64
EM_ITERS = 15
COVARIANCE_REG = 1e-3
FITTING_SEED = 22022
SAMPLER_SEED = 123
PROBE_SEED = 4242


def _model_parameter_suggester(
    method: str,
    asset_root: Union[str, Path],
    fitting_data_manifest: Union[str, Path],
):
    """Return a suggester that loads or fits one exhaustive GMM candidate."""

    def suggest_model_parameters(trial, cfg, resource_grid):
        num_components = trial.suggest_categorical(
            "num_components",
            COMPONENT_COUNTS,
        )
        asset_arguments = {
            "method": method,
            "num_components": num_components,
            "train_batches": TRAIN_BATCHES,
            "train_batch_size": TRAIN_BATCH_SIZE,
            "em_iters": EM_ITERS,
            "covariance_reg": COVARIANCE_REG,
            "seed": FITTING_SEED,
            "asset_root": asset_root,
            "verbose": True,
        }
        try:
            parameters, _ = load_model(
                resource_grid,
                cfg,
                **asset_arguments,
            )
        except FileNotFoundError:
            parameters, _ = fit_model(
                resource_grid,
                cfg,
                **asset_arguments,
                fitting_data_manifest=fitting_data_manifest,
            )
        weights, means, covariances = parameters
        return {
            "weights": weights,
            "means": means,
            "covariances": covariances,
        }

    return suggest_model_parameters


def prepare_fitting_data(
    *,
    asset_root: Union[str, Path],
    overwrite: bool = False,
) -> Path:
    """Generate the one 6,400-channel data bank shared by both methods."""
    return generate_fitting_data(
        build_resource_grid(umi_3gpp_config),
        umi_3gpp_config,
        train_batches=TRAIN_BATCHES,
        train_batch_size=TRAIN_BATCH_SIZE,
        seed=FITTING_SEED,
        asset_root=asset_root,
        overwrite=overwrite,
    )


def run_tuning(
    *,
    method: str = METHOD_SPECTRAL,
    output_dir: Optional[Union[str, Path]] = None,
    asset_root: Union[str, Path],
    fitting_data_manifest: Union[str, Path],
):
    r"""

    .. rubric:: Component-Count Tuning

    The study minimizes weighted NMSE at -10, -5, 0, 5, and 10 dB. It
    exhaustively evaluates four component counts. Every candidate is evaluated
    on three batches of four probe channels at each SNR.

    .. list-table::
        :header-rows: 1

        * - Parameter
          - Search space
        * - ``num_components``
          - exhaustive categorical set {5, 10, 15, 20}

    Each candidate GMM is fitted to the same data bank containing 6,400
    independent system-channel realizations. Spatial and spectral studies reuse
    these channels but derive method-specific fitting vectors from them. Every
    fit uses 15 EM iterations and covariance regularization of 1e-3.

    The four component counts are explicitly enqueued, so the study is an
    exhaustive finite comparison rather than an adaptive TPE search. The count
    with the smallest weighted-NMSE objective is selected directly; no incumbent
    or separate paired-validation stage is used.

    :param method: GMM method to tune, either ``"spectral"`` or ``"spatial"``.
        The default is ``"spectral"``.
    :param output_dir: Output-directory override. If `None`, use the canonical
        method-specific tuning directory.
    :param asset_root: External root containing or receiving fitted GMM assets.
    :param fitting_data_manifest: Manifest of the fixed 6,400-channel fitting
        data bank shared by both methods.
    :returns: The completed four-candidate Optuna study.
    """
    if method not in METHOD_NAMES:
        raise ValueError(f"`method` must be one of {METHOD_NAMES}.")

    if method == METHOD_SPECTRAL:
        estimator_class = SpectralGMMChannelEstimator
        fixed_kwargs = {"time_avg": True}
    else:
        estimator_class = SpatialGMMChannelEstimator
        fixed_kwargs = {"interpolation_type": "lin_time_avg"}

    objective = build_cached_nmse_objective(
        estimator_class,
        umi_3gpp_config,
        ebno_dbs=TUNING_EBNO_DBS,
        snr_weights=TUNING_SNR_WEIGHTS,
        num_mc_batches=3,
        batch_size=4,
        seed=PROBE_SEED,
    )
    return tune_channel_estimator(
        estimator_class,
        umi_3gpp_config,
        _model_parameter_suggester(
            method,
            asset_root,
            fitting_data_manifest,
        ),
        output_dir or f"tuning/mimo_ofdm_chest/koller2022/{method}_components_v2",
        fixed_estimator_kwargs=fixed_kwargs,
        ebno_dbs=TUNING_EBNO_DBS,
        snr_weights=TUNING_SNR_WEIGHTS,
        n_trials=len(COMPONENT_COUNTS),
        num_mc_batches=3,
        batch_size=4,
        sampler_seed=SAMPLER_SEED,
        probe_seed=PROBE_SEED,
        enqueued_trials=(
            {"num_components": count} for count in COMPONENT_COUNTS
        ),
        estimator_objective=objective,
        verbose=True,
    )


def main() -> None:
    """Generate common data or run one exhaustive component-count study."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=METHOD_NAMES, default=METHOD_SPECTRAL)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--fitting-data-manifest", type=Path, default=None)
    parser.add_argument("--prepare-fitting-data", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.prepare_fitting_data:
        print(
            prepare_fitting_data(
                asset_root=args.asset_root,
                overwrite=args.overwrite,
            )
        )
        return
    if args.fitting_data_manifest is None:
        raise ValueError(
            "`--fitting-data-manifest` is required for component tuning."
        )
    run_tuning(
        method=args.method,
        output_dir=args.output_dir,
        asset_root=args.asset_root,
        fitting_data_manifest=args.fitting_data_manifest,
    )


if __name__ == "__main__":
    main()
