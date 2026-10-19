# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os
import shutil
import tempfile
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import torch
from sionna.phy.channel import GenerateOFDMChannel, gen_single_sector_topology
from sionna.phy.object import Object
from sionna.phy.ofdm import LinearInterpolator, ResourceGrid
from sionna.phy.ofdm.channel_estimation import (
    _PilotGroupMixin,
    _pilot_ind_from_pattern,
)
from sionna.phy.utils import ebnodb2no, expand_to_rank

from sionna.bench.mimo_ofdm_chest.asset_store import (
    build_asset_manifest,
    external_asset_dir,
    load_asset_manifest,
    paper_asset_dir,
)
from sionna.bench.mimo_ofdm_chest.benchmarking import UMiChannelSource, set_random_seed
from sionna.bench.mimo_ofdm_chest.configs import SystemConfig
from sionna.bench.mimo_ofdm_chest.persistence import (
    sha256_fingerprint,
    sha256_file,
    write_json_atomic,
    write_npz_atomic,
)
from .estimator import (
    DNN,
    HIDDEN_SIZES,
    METHOD_MODEL_2,
    METHOD_MODEL_3,
    METHOD_NAMES,
)


_PILOT_COORDINATE_ORDER = ["tx", "stream", "symbol", "subcarrier"]
_MODEL2_TRAINING_DATA_ASSET_TYPE = "melgar-model2-training-data"
_MODEL3_TRAINING_DATA_ASSET_TYPE = "melgar-model3-training-data"
_TRAINING_DATA_ASSET_TYPES = {
    METHOD_MODEL_2: _MODEL2_TRAINING_DATA_ASSET_TYPE,
    METHOD_MODEL_3: _MODEL3_TRAINING_DATA_ASSET_TYPE,
}

__all__ = [
    "generate_model2_training_data",
    "generate_model3_training_data",
    "load_model2",
    "load_model2_training_data",
    "load_model3",
    "load_model3_training_data",
    "train_model2",
    "train_model3",
]


# Public asset lifecycle


def load_model2(
    resource_grid: ResourceGrid,
    cfg: SystemConfig,
    *,
    train_batches: int = 800,
    train_batch_size: int = 8,
    validation_batches: int = 32,
    validation_batch_size: Optional[int] = 8,
    epochs: int = 200,
    optimizer_batch_size: int = 256,
    learning_rate: float = 7.02e-5,
    min_train_seconds: float = 0.0,
    patience_epochs: int = 20,
    min_delta: float = 1e-4,
    snr_specific_weights: bool = True,
    training_snr_dbs: Optional[list[float]] = None,
    seed: int = 42,
    asset_root: Optional[Union[str, Path]] = None,
) -> tuple[
    list[dict[str, torch.Tensor]],
    Optional[torch.Tensor],
    list[Path],
]:
    r"""Strictly load the ordered Model 2 DNN states.

    No channels are sampled and no assets are trained or written. If
    ``asset_root`` is `None`, an optional package-local model bank is used.

    :returns: The model states, their SNR noise variances when applicable, and
        manifest paths in the same order.
    """
    return _load_model(
        resource_grid,
        cfg,
        method=METHOD_MODEL_2,
        train_batches=train_batches,
        train_batch_size=train_batch_size,
        validation_batches=validation_batches,
        validation_batch_size=validation_batch_size,
        epochs=epochs,
        optimizer_batch_size=optimizer_batch_size,
        learning_rate=learning_rate,
        min_train_seconds=min_train_seconds,
        patience_epochs=patience_epochs,
        min_delta=min_delta,
        max_examples_per_batch=None,
        snr_specific_weights=snr_specific_weights,
        training_snr_dbs=training_snr_dbs,
        seed=seed,
        asset_root=asset_root,
    )


def load_model3(
    resource_grid: ResourceGrid,
    cfg: SystemConfig,
    *,
    block_len: int = 78,
    train_batches: int = 800,
    train_batch_size: int = 8,
    validation_batches: int = 32,
    validation_batch_size: Optional[int] = 8,
    epochs: int = 200,
    optimizer_batch_size: int = 512,
    learning_rate: float = 3.17e-5,
    min_train_seconds: float = 0.0,
    patience_epochs: int = 20,
    min_delta: float = 1e-4,
    max_examples_per_batch: Optional[int] = 4096,
    snr_specific_weights: bool = True,
    training_snr_dbs: Optional[list[float]] = None,
    seed: int = 42,
    asset_root: Optional[Union[str, Path]] = None,
) -> tuple[
    list[dict[str, torch.Tensor]],
    Optional[torch.Tensor],
    list[Path],
]:
    r"""Strictly load the ordered Model 3 DNN states.

    No channels are sampled and no assets are trained or written. If
    ``asset_root`` is `None`, an optional package-local model bank is used.

    :returns: The model states, their SNR noise variances when applicable, and
        manifest paths in the same order.
    """
    return _load_model(
        resource_grid,
        cfg,
        method=METHOD_MODEL_3,
        block_len=block_len,
        train_batches=train_batches,
        train_batch_size=train_batch_size,
        validation_batches=validation_batches,
        validation_batch_size=validation_batch_size,
        epochs=epochs,
        optimizer_batch_size=optimizer_batch_size,
        learning_rate=learning_rate,
        min_train_seconds=min_train_seconds,
        patience_epochs=patience_epochs,
        min_delta=min_delta,
        max_examples_per_batch=max_examples_per_batch,
        snr_specific_weights=snr_specific_weights,
        training_snr_dbs=training_snr_dbs,
        seed=seed,
        asset_root=asset_root,
    )


def train_model2(
    resource_grid: ResourceGrid,
    cfg: SystemConfig,
    *,
    train_batches: int = 800,
    train_batch_size: int = 8,
    validation_batches: int = 32,
    validation_batch_size: Optional[int] = 8,
    epochs: int = 200,
    optimizer_batch_size: int = 256,
    learning_rate: float = 7.02e-5,
    min_train_seconds: float = 0.0,
    patience_epochs: int = 20,
    min_delta: float = 1e-4,
    snr_specific_weights: bool = True,
    training_snr_dbs: Optional[list[float]] = None,
    training_data_manifest: Optional[Union[str, Path]] = None,
    seed: int = 42,
    asset_root: Union[str, Path],
    overwrite: bool = False,
) -> tuple[
    list[dict[str, torch.Tensor]],
    Optional[torch.Tensor],
    list[Path],
]:
    r"""Train and persist a bank of Model 2 pilot-domain DNNs.

    The defaults reproduce the canonical Model 2 training recipe. RMSprop
    minimizes mean-squared error between noisy LS pilot vectors and clean
    pilot-domain channels. SNR-specific training produces one independently
    trained state per configured SNR.

    .. rubric:: Deviations from the Paper

    - Channels come from the configured Sionna model rather than the paper's
      MATLAB 5G Toolbox CDL-A data.
    - Models are specialized by SNR, but not by delay spread and Doppler.
    - Validation, best-checkpoint restoration, and early stopping are
      repository additions.

    :param resource_grid: Resource grid defining the pilot geometry.
    :param cfg: System and channel-source configuration.
    :param train_batches: Number of independently generated training batches.
    :param train_batch_size: Complete system-channel realizations per batch.
    :param validation_batches: Number of independent validation batches.
    :param validation_batch_size: Channel realizations per validation batch. If
        `None`, use ``train_batch_size``.
    :param epochs: Maximum number of optimizer epochs.
    :param optimizer_batch_size: Derived pilot-vector examples per update.
    :param learning_rate: RMSprop learning rate.
    :param min_train_seconds: Minimum time before early stopping is allowed.
    :param patience_epochs: Epochs without sufficient validation improvement.
        Zero disables early stopping.
    :param min_delta: Minimum relative validation-loss improvement.
    :param snr_specific_weights: Whether to train one state per SNR.
    :param training_snr_dbs: Training SNRs. If `None`, use the configured grid.
    :param training_data_manifest: Optional fixed Model 2 data-bank manifest.
    :param seed: Seed for data generation, initialization, and optimization.
    :param asset_root: Writable external root for the generated assets.
    :param overwrite: Whether matching existing assets may be replaced.
    :returns: The ordered states, corresponding noise variances, and manifests.
    """
    return _train_model(
        resource_grid,
        cfg,
        method=METHOD_MODEL_2,
        train_batches=train_batches,
        train_batch_size=train_batch_size,
        validation_batches=validation_batches,
        validation_batch_size=validation_batch_size,
        epochs=epochs,
        optimizer_batch_size=optimizer_batch_size,
        learning_rate=learning_rate,
        min_train_seconds=min_train_seconds,
        patience_epochs=patience_epochs,
        min_delta=min_delta,
        max_examples_per_batch=None,
        snr_specific_weights=snr_specific_weights,
        training_snr_dbs=training_snr_dbs,
        training_data_manifest=training_data_manifest,
        seed=seed,
        asset_root=asset_root,
        overwrite=overwrite,
    )


def train_model3(
    resource_grid: ResourceGrid,
    cfg: SystemConfig,
    *,
    block_len: int = 78,
    train_batches: int = 800,
    train_batch_size: int = 8,
    validation_batches: int = 32,
    validation_batch_size: Optional[int] = 8,
    epochs: int = 200,
    optimizer_batch_size: int = 512,
    learning_rate: float = 3.17e-5,
    min_train_seconds: float = 0.0,
    patience_epochs: int = 20,
    min_delta: float = 1e-4,
    max_examples_per_batch: Optional[int] = 4096,
    snr_specific_weights: bool = True,
    training_snr_dbs: Optional[list[float]] = None,
    training_data_manifest: Optional[Union[str, Path]] = None,
    seed: int = 42,
    asset_root: Union[str, Path],
    overwrite: bool = False,
) -> tuple[
    list[dict[str, torch.Tensor]],
    Optional[torch.Tensor],
    list[Path],
]:
    r"""Train and persist a bank of Model 3 full-grid block DNNs.

    The defaults reproduce the canonical Model 3 training recipe. RMSprop
    minimizes mean-squared error between linearly interpolated LS frequency
    blocks and clean full-grid channel blocks. SNR-specific training produces
    one independently trained state per configured SNR.

    .. rubric:: Deviations from the Paper

    - Channels come from the configured Sionna model rather than the paper's
      MATLAB 5G Toolbox CDL-A data.
    - Models are specialized by SNR, but not by delay spread and Doppler.
    - Validation, best-checkpoint restoration, and early stopping are
      repository additions.
    - Only a deterministic subset of derived blocks may be retained per
      generated channel batch.

    :param resource_grid: Resource grid defining pilot and full-grid geometry.
    :param cfg: System and channel-source configuration.
    :param block_len: Effective subcarriers in each frequency block.
    :param train_batches: Number of independently generated training batches.
    :param train_batch_size: Complete system-channel realizations per batch.
    :param validation_batches: Number of independent validation batches.
    :param validation_batch_size: Channel realizations per validation batch. If
        `None`, use ``train_batch_size``.
    :param epochs: Maximum number of optimizer epochs.
    :param optimizer_batch_size: Derived frequency-block examples per update.
    :param learning_rate: RMSprop learning rate.
    :param min_train_seconds: Minimum time before early stopping is allowed.
    :param patience_epochs: Epochs without sufficient validation improvement.
        Zero disables early stopping.
    :param min_delta: Minimum relative validation-loss improvement.
    :param max_examples_per_batch: Maximum derived examples retained from each
        generated channel batch. `None` retains all examples.
    :param snr_specific_weights: Whether to train one state per SNR.
    :param training_snr_dbs: Training SNRs. If `None`, use the configured grid.
    :param training_data_manifest: Optional fixed Model 3 data-bank manifest.
    :param seed: Seed for data generation, initialization, and optimization.
    :param asset_root: Writable external root for the generated assets.
    :param overwrite: Whether matching existing assets may be replaced.
    :returns: The ordered states, corresponding noise variances, and manifests.
    """
    return _train_model(
        resource_grid,
        cfg,
        method=METHOD_MODEL_3,
        block_len=block_len,
        train_batches=train_batches,
        train_batch_size=train_batch_size,
        validation_batches=validation_batches,
        validation_batch_size=validation_batch_size,
        epochs=epochs,
        optimizer_batch_size=optimizer_batch_size,
        learning_rate=learning_rate,
        min_train_seconds=min_train_seconds,
        patience_epochs=patience_epochs,
        min_delta=min_delta,
        max_examples_per_batch=max_examples_per_batch,
        snr_specific_weights=snr_specific_weights,
        training_snr_dbs=training_snr_dbs,
        training_data_manifest=training_data_manifest,
        seed=seed,
        asset_root=asset_root,
        overwrite=overwrite,
    )


def generate_model2_training_data(
    resource_grid: ResourceGrid,
    cfg: SystemConfig,
    *,
    train_batches: int = 800,
    train_batch_size: int = 8,
    validation_batches: int = 32,
    validation_batch_size: int = 8,
    training_snr_dbs: Optional[list[float]] = None,
    seed: int = 42,
    asset_root: Union[str, Path],
    overwrite: bool = False,
) -> Path:
    r"""Generate a deterministic Model 2 training-data bank.

    Each channel contributes noisy LS pilot vectors and matching noiseless
    pilot-domain targets. The same channels and clean targets are reused across
    SNRs, with an independently seeded fixed noisy view for each SNR.

    :param resource_grid: Resource grid defining the pilot geometry.
    :param cfg: System and channel configuration used to sample channels.
    :param train_batches: Number of independent training-channel batches.
    :param train_batch_size: Complete channel realizations per training batch.
    :param validation_batches: Number of independent validation batches.
    :param validation_batch_size: Complete channels per validation batch.
    :param training_snr_dbs: Generated SNR views. If `None`, use ``cfg``.
    :param seed: Base seed for channels and SNR-specific observation noise.
    :param asset_root: Writable external root for the generated data.
    :param overwrite: Whether a matching data asset may be replaced.
    :returns: Path to the strictly validated training-data manifest.
    """
    return _generate_training_data(
        resource_grid,
        cfg,
        method=METHOD_MODEL_2,
        train_batches=train_batches,
        train_batch_size=train_batch_size,
        validation_batches=validation_batches,
        validation_batch_size=validation_batch_size,
        training_snr_dbs=training_snr_dbs,
        max_examples_per_batch=None,
        seed=seed,
        asset_root=asset_root,
        overwrite=overwrite,
    )


def generate_model3_training_data(
    resource_grid: ResourceGrid,
    cfg: SystemConfig,
    *,
    train_batches: int = 800,
    train_batch_size: int = 8,
    validation_batches: int = 32,
    validation_batch_size: int = 8,
    training_snr_dbs: Optional[list[float]] = None,
    block_len: int = 78,
    max_examples_per_batch: Optional[int] = 4096,
    seed: int = 42,
    asset_root: Union[str, Path],
    overwrite: bool = False,
) -> Path:
    r"""Generate a deterministic Model 3 training-data bank.

    Noisy LS estimates are linearly interpolated over the resource grid and
    stored as real-valued frequency blocks with matching clean-channel targets.
    The same channels, targets, and selected examples are reused across SNRs.

    :param resource_grid: Resource grid defining interpolation and block geometry.
    :param cfg: System and channel configuration used to sample channels.
    :param train_batches: Number of independent training-channel batches.
    :param train_batch_size: Complete channel realizations per training batch.
    :param validation_batches: Number of independent validation batches.
    :param validation_batch_size: Complete channels per validation batch.
    :param training_snr_dbs: Generated SNR views. If `None`, use ``cfg``.
    :param block_len: Effective subcarriers in each frequency block.
    :param max_examples_per_batch: Maximum blocks retained from each batch.
    :param seed: Base seed for channels, selection, and observation noise.
    :param asset_root: Writable external root for the generated data.
    :param overwrite: Whether a matching data asset may be replaced.
    :returns: Path to the strictly validated training-data manifest.
    """
    return _generate_training_data(
        resource_grid,
        cfg,
        method=METHOD_MODEL_3,
        train_batches=train_batches,
        train_batch_size=train_batch_size,
        validation_batches=validation_batches,
        validation_batch_size=validation_batch_size,
        training_snr_dbs=training_snr_dbs,
        block_len=block_len,
        max_examples_per_batch=max_examples_per_batch,
        seed=seed,
        asset_root=asset_root,
        overwrite=overwrite,
    )


def load_model2_training_data(
    resource_grid: ResourceGrid,
    cfg: SystemConfig,
    *,
    train_batches: int,
    train_batch_size: int,
    validation_batches: int,
    validation_batch_size: int,
    training_snr_dbs: Optional[list[float]] = None,
    seed: int = 42,
    manifest_path: Union[str, Path],
) -> tuple["_TrainingDataBank", Path]:
    """Strictly validate and memory-map one fixed Model 2 data bank."""
    return _load_training_data(
        resource_grid,
        cfg,
        method=METHOD_MODEL_2,
        train_batches=train_batches,
        train_batch_size=train_batch_size,
        validation_batches=validation_batches,
        validation_batch_size=validation_batch_size,
        training_snr_dbs=training_snr_dbs,
        max_examples_per_batch=None,
        seed=seed,
        manifest_path=manifest_path,
    )


def load_model3_training_data(
    resource_grid: ResourceGrid,
    cfg: SystemConfig,
    *,
    train_batches: int,
    train_batch_size: int,
    validation_batches: int,
    validation_batch_size: int,
    training_snr_dbs: Optional[list[float]] = None,
    block_len: int = 78,
    max_examples_per_batch: Optional[int] = 4096,
    seed: int = 42,
    manifest_path: Union[str, Path],
) -> tuple["_TrainingDataBank", Path]:
    """Strictly validate and memory-map one fixed Model 3 data bank."""
    return _load_training_data(
        resource_grid,
        cfg,
        method=METHOD_MODEL_3,
        train_batches=train_batches,
        train_batch_size=train_batch_size,
        validation_batches=validation_batches,
        validation_batch_size=validation_batch_size,
        training_snr_dbs=training_snr_dbs,
        block_len=block_len,
        max_examples_per_batch=max_examples_per_batch,
        seed=seed,
        manifest_path=manifest_path,
    )


# Shared implementation


def _load_model(
    resource_grid: ResourceGrid,
    cfg: SystemConfig,
    *,
    method: str = METHOD_MODEL_3,
    block_len: int = 78,
    train_batches: int = 800,
    train_batch_size: int = 8,
    validation_batches: int = 32,
    validation_batch_size: Optional[int] = 8,
    epochs: int = 200,
    optimizer_batch_size: int = 512,
    learning_rate: float = 3.17e-5,
    min_train_seconds: float = 0.0,
    patience_epochs: int = 20,
    min_delta: float = 1e-4,
    max_examples_per_batch: Optional[int] = 4096,
    snr_specific_weights: bool = True,
    training_snr_dbs: Optional[list[float]] = None,
    seed: int = 42,
    asset_root: Optional[Union[str, Path]] = None,
) -> tuple[
    list[dict[str, torch.Tensor]],
    Optional[torch.Tensor],
    list[Path],
]:
    r"""Strictly load the ordered DNN states required by one estimator.

    This function never samples channels, trains a network, or writes an
    asset. With no ``asset_root``, it selects optional package-local assets.

    :returns: ``(model_states, model_noise_variances, manifest_paths)``.
        Manifest paths follow exactly the same order as per-SNR model states.
    """
    contexts, noise_variances, _ = _model_contexts(
        resource_grid,
        cfg,
        method=method,
        block_len=block_len,
        train_batches=train_batches,
        train_batch_size=train_batch_size,
        validation_batches=validation_batches,
        validation_batch_size=validation_batch_size,
        epochs=epochs,
        optimizer_batch_size=optimizer_batch_size,
        learning_rate=learning_rate,
        min_train_seconds=min_train_seconds,
        patience_epochs=patience_epochs,
        min_delta=min_delta,
        max_examples_per_batch=max_examples_per_batch,
        snr_specific_weights=snr_specific_weights,
        training_snr_dbs=training_snr_dbs,
        seed=seed,
        asset_root=asset_root,
    )
    states = []
    manifest_paths = []
    for context in contexts:
        state, manifest_path = _load_model_asset(context)
        states.append(state)
        manifest_paths.append(manifest_path)
    return states, noise_variances, manifest_paths


def _train_model(
    resource_grid: ResourceGrid,
    cfg: SystemConfig,
    *,
    method: str = METHOD_MODEL_3,
    block_len: int = 78,
    train_batches: int = 800,
    train_batch_size: int = 8,
    validation_batches: int = 32,
    validation_batch_size: Optional[int] = 8,
    epochs: int = 200,
    optimizer_batch_size: int = 512,
    learning_rate: float = 3.17e-5,
    min_train_seconds: float = 0.0,
    patience_epochs: int = 20,
    min_delta: float = 1e-4,
    max_examples_per_batch: Optional[int] = 4096,
    snr_specific_weights: bool = True,
    training_snr_dbs: Optional[list[float]] = None,
    training_data_manifest: Optional[Union[str, Path]] = None,
    seed: int = 42,
    asset_root: Union[str, Path],
    overwrite: bool = False,
) -> tuple[
    list[dict[str, torch.Tensor]],
    Optional[torch.Tensor],
    list[Path],
]:
    r"""Train and persist a bank of DNN channel estimators.

    The function trains either the pilot-domain Model 2 network or the
    interpolated full-grid Model 3 network. By default, one independently trained
    state is produced for every configured SNR. Both methods can reuse a fixed
        training-data bank generated by the corresponding public data-generation
        function.

    Training uses RMSprop and mean-squared error on separate real and imaginary
    channel vectors. When validation data are provided, the state with the best
    validation loss is restored. Assets are written below the explicitly supplied
    external root and strictly reloaded before being returned.

    .. rubric:: Deviations from the Paper

    - Training uses channels generated by the configured Sionna channel model
      instead of the paper's MATLAB 5G Toolbox CDL-A dataset.
    - Canonical training specializes models by SNR, but not separately by delay
      spread and Doppler frequency as in the paper's specialized training mode.
    - Independent validation, best-checkpoint restoration, and configurable early
      stopping are repository additions.
    - Model 3 may retain only a deterministic subset of the frequency-block
      examples derived from each channel batch.

    :param resource_grid: Resource grid defining the channel and pilot geometry.
    :param cfg: System and channel-source configuration.
    :param method: Model to train, either ``"model2"`` or ``"model3"``.
    :param block_len: Number of subcarriers in each Model 3 frequency block.
    :param train_batches: Number of independently generated training-channel
        batches.
    :param train_batch_size: Number of complete system-channel realizations in
        each training batch.
    :param validation_batches: Number of independently generated validation
        batches.
    :param validation_batch_size: Number of channel realizations per validation
        batch. If `None`, use ``train_batch_size``.
    :param epochs: Maximum number of optimizer epochs.
    :param optimizer_batch_size: Number of derived DNN examples per optimizer
        update.
    :param learning_rate: RMSprop learning rate.
    :param min_train_seconds: Minimum elapsed time before early stopping is
        permitted.
    :param patience_epochs: Number of epochs without sufficient validation
        improvement before stopping. Zero disables early stopping.
    :param min_delta: Minimum relative validation-loss improvement.
    :param max_examples_per_batch: Maximum number of Model 3 examples retained
        from each generated channel batch. `None` retains all examples. Ignored
        by Model 2.
    :param snr_specific_weights: If `True`, train one state per SNR; otherwise,
        train one state using mixed-SNR examples.
    :param training_snr_dbs: SNR values used for training. If `None`, use the
        configured SNR grid.
    :param training_data_manifest: Optional fixed data-bank manifest for the
        selected method.
    :param seed: Seed for channel generation, noise, initialization, and training.
    :param asset_root: External directory under which generated assets are stored.
    :param overwrite: Whether to replace matching existing assets.

    :returns: A tuple containing the ordered model states, the corresponding
        SNR noise variances when SNR-specific states are used, and the ordered
        paths to their asset manifests.
    """
    effective_validation_batch_size = (
        train_batch_size
        if validation_batch_size is None
        else int(validation_batch_size)
    )
    contexts, noise_variances, group_data = _model_contexts(
        resource_grid,
        cfg,
        method=method,
        block_len=block_len,
        train_batches=train_batches,
        train_batch_size=train_batch_size,
        validation_batches=validation_batches,
        validation_batch_size=effective_validation_batch_size,
        epochs=epochs,
        optimizer_batch_size=optimizer_batch_size,
        learning_rate=learning_rate,
        min_train_seconds=min_train_seconds,
        patience_epochs=patience_epochs,
        min_delta=min_delta,
        max_examples_per_batch=max_examples_per_batch,
        snr_specific_weights=snr_specific_weights,
        training_snr_dbs=training_snr_dbs,
        seed=seed,
        asset_root=asset_root,
    )
    training_data = None
    validated_training_data_manifest = None
    if training_data_manifest is not None:
        training_data, validated_training_data_manifest = _load_training_data(
            resource_grid,
            cfg,
            method=method,
            train_batches=train_batches,
            train_batch_size=train_batch_size,
            validation_batches=validation_batches,
            validation_batch_size=effective_validation_batch_size,
            training_snr_dbs=None,
            block_len=block_len,
            max_examples_per_batch=max_examples_per_batch,
            seed=seed,
            manifest_path=training_data_manifest,
        )

    states = []
    manifest_paths = []
    for context in contexts:
        selected_snr_dbs = context.generation_recipe["training_snr_dbs"]
        if method == METHOD_MODEL_2:
            assert group_data is not None
            state, manifest_path = _train_model2(
                resource_grid,
                cfg,
                group_data,
                context,
                input_dim=context.input_dim,
                train_batches=train_batches,
                train_batch_size=train_batch_size,
                validation_batches=validation_batches,
                validation_batch_size=effective_validation_batch_size,
                epochs=epochs,
                optimizer_batch_size=optimizer_batch_size,
                learning_rate=learning_rate,
                min_train_seconds=min_train_seconds,
                patience_epochs=patience_epochs,
                min_delta=min_delta,
                training_snr_dbs=selected_snr_dbs,
                training_data=training_data,
                training_data_manifest=validated_training_data_manifest,
                seed=seed,
                overwrite=overwrite,
            )
        else:
            state, manifest_path = _train_model3(
                resource_grid,
                cfg,
                context,
                input_dim=context.input_dim,
                train_batches=train_batches,
                train_batch_size=train_batch_size,
                validation_batches=validation_batches,
                validation_batch_size=effective_validation_batch_size,
                epochs=epochs,
                optimizer_batch_size=optimizer_batch_size,
                learning_rate=learning_rate,
                min_train_seconds=min_train_seconds,
                patience_epochs=patience_epochs,
                min_delta=min_delta,
                training_snr_dbs=selected_snr_dbs,
                max_examples_per_batch=max_examples_per_batch,
                training_data=training_data,
                training_data_manifest=validated_training_data_manifest,
                seed=seed,
                overwrite=overwrite,
            )
        states.append(state)
        manifest_paths.append(manifest_path)
    return states, noise_variances, manifest_paths


def _generate_training_data(
    resource_grid: ResourceGrid,
    cfg: SystemConfig,
    *,
    method: str = METHOD_MODEL_3,
    train_batches: int = 800,
    train_batch_size: int = 8,
    validation_batches: int = 32,
    validation_batch_size: int = 8,
    training_snr_dbs: Optional[list[float]] = None,
    block_len: int = 78,
    max_examples_per_batch: Optional[int] = 4096,
    seed: int = 42,
    asset_root: Union[str, Path],
    overwrite: bool = False,
) -> Path:
    r"""Generate and persist one deterministic Melgar training-data bank.

    Model 2 stores noisy LS pilot vectors and their noiseless pilot-domain
    targets. Model 3 applies linear interpolation before storing real-valued
    frequency blocks and their noiseless full-grid targets.

    The same channel realizations and clean targets are reused across all
    requested SNR points. Each SNR receives an independently seeded but fixed
    noisy input view. The completed asset is strictly validated before its
    manifest path is returned.

    .. rubric:: Deviations from the Paper

    - The paper generates its data with MATLAB 5G Toolbox CDL-A channels. This
      implementation uses the channel distribution defined by ``cfg``.
    - The paper reports 800,000 training and 50,000 validation DNN data points
      generated from 20 channel realizations. This implementation controls the
      number of independently sampled system channels. Model 3 may additionally
      cap the derived real-valued block examples retained from each batch; Model
      2 retains all pilot-vector examples.

    :param resource_grid: Sionna resource grid defining the pilot and full-grid
        geometry.
    :param cfg: System and channel configuration used to generate channel
        realizations and SNR-dependent observation noise.
    :param method: Melgar method whose training examples are generated.
    :param train_batches: Number of independent training-channel batches.
    :param train_batch_size: Number of complete system-channel realizations in
        each training batch. Their product is the number of independently sampled
        training channels.
    :param validation_batches: Number of independent validation-channel batches.
    :param validation_batch_size: Number of complete system-channel realizations
        in each validation batch. Their product is the number of independently
        sampled validation channels.
    :param training_snr_dbs: SNR points for which noisy input views are generated.
        If `None`, use the SNR grid defined by ``cfg``.
    :param block_len: Number of contiguous effective subcarriers in every
        ``InterpolatedDNNChannelEstimator`` frequency block. It must divide the
        number of effective subcarriers.
    :param max_examples_per_batch: Maximum number of derived real-valued Model 3
        block examples retained from each generated channel batch. If `None`,
        retain all candidate examples. Model 2 always retains all pilot-vector
        examples.
    :param seed: Base random seed for deterministic channel generation, example
        selection, and SNR-specific observation noise. Training and validation use
        separate deterministic random streams.
    :param asset_root: Writable external root beneath which the generated payloads
        and manifest are stored.
    :param overwrite: If `True`, replace an existing asset with the same identity.
        If `False`, raise an error when that asset already exists.

    :returns: Path to the strictly validated ``manifest.json`` of the generated
        training-data asset.
    """
    if method not in METHOD_NAMES:
        raise ValueError(f"`method` must be one of {METHOD_NAMES}.")
    group_data = (
        _PilotGroupLayout(resource_grid).training_group_data()
        if method == METHOD_MODEL_2
        else None
    )
    context = _training_data_context(
        resource_grid,
        cfg,
        method=method,
        group_data=group_data,
        train_batches=train_batches,
        train_batch_size=train_batch_size,
        validation_batches=validation_batches,
        validation_batch_size=validation_batch_size,
        training_snr_dbs=training_snr_dbs,
        block_len=block_len,
        max_examples_per_batch=max_examples_per_batch,
        seed=seed,
        asset_root=asset_root,
    )
    if context.asset_dir.exists() and not overwrite:
        raise FileExistsError(
            f"Melgar {method} training data already exist at {context.asset_dir}."
        )
    context.asset_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{context.asset_dir.name}.",
            dir=context.asset_dir.parent,
        )
    )
    try:
        payload_paths = []
        payload_paths.extend(
            _write_training_split(
                temporary_dir,
                "train",
                resource_grid,
                cfg,
                context,
                group_data=group_data,
                channel_batches=train_batches,
                channel_batch_size=train_batch_size,
                seed=seed,
            )
        )
        payload_paths.extend(
            _write_training_split(
                temporary_dir,
                "validation",
                resource_grid,
                cfg,
                context,
                group_data=group_data,
                channel_batches=validation_batches,
                channel_batch_size=validation_batch_size,
                seed=seed + 100003,
            )
        )
        manifest = build_asset_manifest(
            cfg=cfg,
            asset_type=_TRAINING_DATA_ASSET_TYPES[method],
            artifact_id=context.asset_dir.name,
            compatibility=context.compatibility,
            generation_recipe=context.generation_recipe,
            payload_paths=payload_paths,
            metadata=context.metadata,
        )
        write_json_atomic(temporary_dir / "manifest.json", manifest)
        if context.asset_dir.exists():
            shutil.rmtree(context.asset_dir)
        os.replace(temporary_dir, context.asset_dir)
    finally:
        shutil.rmtree(temporary_dir, ignore_errors=True)

    _, manifest_path = _load_training_data(
        resource_grid,
        cfg,
        method=method,
        train_batches=train_batches,
        train_batch_size=train_batch_size,
        validation_batches=validation_batches,
        validation_batch_size=validation_batch_size,
        training_snr_dbs=training_snr_dbs,
        block_len=block_len,
        max_examples_per_batch=max_examples_per_batch,
        seed=seed,
        manifest_path=context.manifest_path,
    )
    return manifest_path


def _load_training_data(
    resource_grid: ResourceGrid,
    cfg: SystemConfig,
    *,
    method: str = METHOD_MODEL_3,
    train_batches: int,
    train_batch_size: int,
    validation_batches: int,
    validation_batch_size: int,
    training_snr_dbs: Optional[list[float]] = None,
    block_len: int = 78,
    max_examples_per_batch: Optional[int] = 4096,
    seed: int = 42,
    manifest_path: Union[str, Path],
) -> tuple["_TrainingDataBank", Path]:
    """Strictly validate and memory-map one fixed Melgar data bank."""
    if method not in METHOD_NAMES:
        raise ValueError(f"`method` must be one of {METHOD_NAMES}.")
    manifest_path = Path(manifest_path)
    envelope = load_asset_manifest(
        manifest_path,
        expected_asset_type=_TRAINING_DATA_ASSET_TYPES[method],
        expected_scenario_id=cfg.scenario_id,
    )
    bank_snr_dbs = envelope["generation_recipe"].get("training_snr_dbs")
    if not isinstance(bank_snr_dbs, list):
        raise ValueError(f"Melgar {method} training data have invalid SNR metadata.")
    requested_snr_dbs = _training_snr_list(cfg, training_snr_dbs)
    if not set(requested_snr_dbs).issubset(set(bank_snr_dbs)):
        raise ValueError(
            f"Melgar {method} training data do not contain all requested SNRs."
        )
    group_data = (
        _PilotGroupLayout(resource_grid).training_group_data()
        if method == METHOD_MODEL_2
        else None
    )
    context = _training_data_context(
        resource_grid,
        cfg,
        method=method,
        group_data=group_data,
        train_batches=train_batches,
        train_batch_size=train_batch_size,
        validation_batches=validation_batches,
        validation_batch_size=validation_batch_size,
        training_snr_dbs=bank_snr_dbs,
        block_len=block_len,
        max_examples_per_batch=max_examples_per_batch,
        seed=seed,
        asset_root=manifest_path.parents[4],
    )
    manifest, tensors = _load_training_data_cached(
        str(manifest_path),
        manifest_path.stat().st_mtime_ns,
    )
    if manifest["artifact_id"] != context.asset_dir.name:
        raise ValueError(f"Melgar {method} training data have incompatible identity.")
    if sha256_fingerprint(manifest["compatibility"]) != sha256_fingerprint(
        context.compatibility
    ):
        raise ValueError(
            f"Melgar {method} training-data compatibility does not match."
        )
    if manifest["generation_recipe"] != context.generation_recipe:
        raise ValueError(f"Melgar {method} training-data recipe does not match.")
    if manifest.get("metadata") != context.metadata:
        raise ValueError(f"Melgar {method} training-data metadata do not match.")
    _validate_training_data_files(manifest, context)
    return (
        _TrainingDataBank(
            train_inputs={
                float(snr_db): tensors[_training_input_filename("train", snr_db)]
                for snr_db in context.training_snr_dbs
            },
            train_targets=tensors["train-targets.npy"],
            validation_inputs={
                float(snr_db): tensors[
                    _training_input_filename("validation", snr_db)
                ]
                for snr_db in context.training_snr_dbs
            },
            validation_targets=tensors["validation-targets.npy"],
        ),
        manifest_path,
    )


# Asset identity and compatibility


@dataclass(frozen=True)
class _ModelAssetContext:
    """Identity, compatibility, and paths for one trained DNN state."""

    asset_type: str
    scenario_id: str
    input_dim: int
    asset_dir: Path
    compatibility: dict[str, Any]
    generation_recipe: dict[str, Any]

    @property
    def params_path(self) -> Path:
        """Return the model-state payload path."""
        return self.asset_dir / "dnn_params.npz"

    @property
    def manifest_path(self) -> Path:
        """Return the asset manifest path."""
        return self.asset_dir / "manifest.json"


@dataclass(frozen=True)
class _TrainingDataContext:
    """Identity and expected payload shapes for one fixed Melgar data bank."""

    method: str
    asset_dir: Path
    compatibility: dict[str, Any]
    generation_recipe: dict[str, Any]
    metadata: dict[str, Any]
    training_snr_dbs: tuple[float, ...]
    input_dim: int
    train_examples: int
    validation_examples: int

    @property
    def manifest_path(self) -> Path:
        """Return the data-bank manifest path."""
        return self.asset_dir / "manifest.json"


@dataclass(frozen=True)
class _TrainingDataBank:
    """Memory-mapped SNR-specific inputs and common clean targets."""

    train_inputs: dict[float, torch.Tensor]
    train_targets: torch.Tensor
    validation_inputs: dict[float, torch.Tensor]
    validation_targets: torch.Tensor


def _model_asset_dir(
    cfg: SystemConfig,
    asset_type: str,
    artifact_id: str,
    asset_root: Optional[Union[str, Path]],
) -> Path:
    """Select the package-local or external model artifact directory."""
    if asset_root is None:
        return paper_asset_dir(cfg, asset_type, artifact_id, __file__)
    return external_asset_dir(
        cfg,
        "melgar2022",
        asset_type,
        artifact_id,
        asset_root=asset_root,
    )


# Pilot layouts and compatibility metadata


class _PilotGroupLayout(_PilotGroupMixin, Object):
    """Expose Model 2 pilot groups for offline dataset generation."""

    def __init__(self, resource_grid: ResourceGrid) -> None:
        super().__init__(
            precision=resource_grid.precision,
            device=resource_grid.device,
        )
        self._init_pilot_groups(
            pilot_pattern=resource_grid.pilot_pattern,
            group_over=("subcarrier",),
            persistent=False,
        )
        self.register_buffer(
            "_training_pilot_values",
            self._gather_pilot_groups(
                resource_grid.pilot_pattern.pilots.to(self.device)
            ),
            persistent=False,
        )

    def training_group_data(self) -> dict[str, torch.Tensor]:
        """Return ordered pilot coordinates and values used by Model 2."""
        return {
            "coordinates": self._pilot_group_coords.detach().cpu(),
            "tx": self._pilot_group_constant_coord("tx").detach().cpu(),
            "stream": self._pilot_group_constant_coord("stream").detach().cpu(),
            "symbol": self._pilot_group_constant_coord("symbol").detach().cpu(),
            "subcarriers": self._pilot_group_varying_coord(
                "subcarrier"
            ).detach().cpu(),
            "pilot_values": self._training_pilot_values.detach().cpu(),
        }


def _tensor_identity(values: torch.Tensor) -> dict[str, Any]:
    """Return shape, dtype, encoding, and canonical hash for a tensor."""
    values = values.detach().cpu()
    identity = {
        "shape": list(values.shape),
        "dtype": str(values.dtype).removeprefix("torch."),
    }
    if values.is_complex():
        identity["encoding"] = "[real, imaginary]"
        canonical_values = torch.view_as_real(values).tolist()
    else:
        canonical_values = values.tolist()
    identity["canonical_sha256"] = sha256_fingerprint(canonical_values)
    return identity


def _model2_pilot_layout(
    group_data: dict[str, torch.Tensor],
) -> dict[str, Any]:
    """Return the exact ordered active-pilot layout used by Model 2."""
    pilot_values = group_data["pilot_values"]
    return {
        "coordinate_order": _PILOT_COORDINATE_ORDER,
        "coordinates": _tensor_identity(group_data["coordinates"]),
        "values": _tensor_identity(pilot_values),
        "powers": _tensor_identity(pilot_values.abs().square()),
    }


def _model3_ls_interpolation_layout(
    resource_grid: ResourceGrid,
) -> dict[str, Any]:
    """Return the complete resource and pilot layout used by LS-linear."""
    pilot_pattern = resource_grid.pilot_pattern
    pilot_values = pilot_pattern.pilots.detach().cpu()
    effective_subcarrier_indices = torch.as_tensor(
        resource_grid.effective_subcarrier_ind,
        dtype=torch.int64,
    )
    return {
        "resource_grid": {
            "num_tx": int(resource_grid.num_tx),
            "num_streams_per_tx": int(resource_grid.num_streams_per_tx),
            "num_ofdm_symbols": int(resource_grid.num_ofdm_symbols),
            "fft_size": int(resource_grid.fft_size),
            "num_effective_subcarriers": int(
                resource_grid.num_effective_subcarriers
            ),
            "effective_subcarrier_indices": _tensor_identity(
                effective_subcarrier_indices
            ),
        },
        "pilot_pattern": {
            "mask": _tensor_identity(pilot_pattern.mask),
            "values": _tensor_identity(pilot_values),
            "powers": _tensor_identity(pilot_values.abs().square()),
        },
    }


def _training_snr_list(
    cfg: SystemConfig,
    training_snr_dbs: list[float] | None,
) -> list[float]:
    """Return the explicit or configured training SNR grid."""
    if training_snr_dbs is None:
        return [float(value) for value in cfg.snr_db_range()]
    if not training_snr_dbs:
        raise ValueError("`training_snr_dbs` must not be empty.")
    return [float(value) for value in training_snr_dbs]


def _training_input_filename(split: str, snr_db: float) -> str:
    """Return the stable payload name for one SNR-specific input view."""
    return f"{split}-inputs-snr-{_float_token(snr_db)}.npy"


def _training_data_context(
    resource_grid: ResourceGrid,
    cfg: SystemConfig,
    *,
    method: str,
    group_data: Optional[dict[str, torch.Tensor]],
    train_batches: int,
    train_batch_size: int,
    validation_batches: int,
    validation_batch_size: int,
    training_snr_dbs: Optional[list[float]],
    block_len: int,
    max_examples_per_batch: Optional[int],
    seed: int,
    asset_root: Union[str, Path],
) -> _TrainingDataContext:
    """Resolve one external fixed training-data contract."""
    counts = (
        int(train_batches),
        int(train_batch_size),
        int(validation_batches),
        int(validation_batch_size),
    )
    if any(value < 1 for value in counts):
        raise ValueError("Melgar training-data batch counts must be positive.")
    snr_dbs = tuple(_training_snr_list(cfg, training_snr_dbs))
    if len(set(snr_dbs)) != len(snr_dbs):
        raise ValueError("Melgar training-data SNRs must be unique.")
    if method == METHOD_MODEL_2:
        if group_data is None:
            raise ValueError("Model 2 training data require pilot-group metadata.")
        input_dim = int(group_data["pilot_values"].shape[-1])
        examples_per_channel = (
            2
            * int(cfg.num_rx_ant)
            * int(group_data["pilot_values"].shape[0])
        )
        effective_example_cap = None
    else:
        num_subcarriers = int(resource_grid.num_effective_subcarriers)
        if num_subcarriers % int(block_len) != 0:
            raise ValueError("`block_len` must divide the effective subcarriers.")
        input_dim = int(block_len)
        num_blocks = num_subcarriers // input_dim
        examples_per_channel = (
            2
            * int(cfg.num_rx_ant)
            * int(resource_grid.num_tx)
            * int(resource_grid.num_streams_per_tx)
            * int(resource_grid.num_ofdm_symbols)
            * num_blocks
        )
        effective_example_cap = max_examples_per_batch

    def split_examples(num_batches: int, batch_size: int) -> int:
        examples = examples_per_channel * int(batch_size)
        if effective_example_cap is not None:
            examples = min(examples, int(effective_example_cap))
        return int(num_batches) * examples

    train_examples = split_examples(counts[0], counts[1])
    validation_examples = split_examples(counts[2], counts[3])
    train_channels = counts[0] * counts[1]
    validation_channels = counts[2] * counts[3]
    snr_suffix = "-".join(_float_token(value) for value in snr_dbs)
    cap_token = _example_cap_token(effective_example_cap)
    artifact_id = (
        f"train{train_channels}-val{validation_channels}-snr{snr_suffix}"
        f"-n{input_dim}-cap{cap_token}-seed{seed}"
    ).replace("_", "-").replace(".", "p")
    snapshot = cfg.config_snapshot()
    system = snapshot["system"]
    compatibility = {
        "channel_source": snapshot["channel_source"],
        "ofdm": system["ofdm"],
        "multi_user": system["multi_user"],
        "arrays": system["arrays"],
        method: (
            {
                "pilot_groups": _model2_pilot_layout(group_data),
            }
            if method == METHOD_MODEL_2
            else {
                "block_len": input_dim,
                "max_examples_per_batch": (
                    None
                    if effective_example_cap is None
                    else int(effective_example_cap)
                ),
                "ls_interpolation_layout": _model3_ls_interpolation_layout(
                    resource_grid
                ),
            }
        ),
    }
    generation_recipe = {
        "train_batches": counts[0],
        "train_batch_size": counts[1],
        "validation_batches": counts[2],
        "validation_batch_size": counts[3],
        "training_snr_dbs": list(snr_dbs),
        "seed": int(seed),
        "validation_seed": int(seed) + 100003,
        "noise_seed_rule": "split_seed+200003+snr_index",
        "selection_seed_rule": (
            None if method == METHOD_MODEL_2 else "split_seed+300007"
        ),
        "storage_dtype": "float16",
    }
    metadata = {
        "train_independent_channel_count": train_channels,
        "validation_independent_channel_count": validation_channels,
        "candidate_examples_per_channel": examples_per_channel,
        "train_derived_example_count_per_snr": train_examples,
        "validation_derived_example_count_per_snr": validation_examples,
        "clean_channels_and_selected_examples_shared_across_snrs": True,
        "snr_specific_noise": "fixed-independent-seeded-view",
    }
    return _TrainingDataContext(
        method=method,
        asset_dir=external_asset_dir(
            cfg,
            "melgar2022",
            _TRAINING_DATA_ASSET_TYPES[method],
            artifact_id,
            asset_root=asset_root,
        ),
        compatibility=compatibility,
        generation_recipe=generation_recipe,
        metadata=metadata,
        training_snr_dbs=snr_dbs,
        input_dim=input_dim,
        train_examples=train_examples,
        validation_examples=validation_examples,
    )


def _validate_training_data_files(
    manifest: dict[str, Any],
    context: _TrainingDataContext,
) -> None:
    """Validate exact file order, shapes, and dtypes for one data bank."""
    expected: list[tuple[str, list[int]]] = []
    for split, num_examples in (
        ("train", context.train_examples),
        ("validation", context.validation_examples),
    ):
        expected.append(
            (f"{split}-targets.npy", [num_examples, context.input_dim])
        )
        expected.extend(
            (
                _training_input_filename(split, snr_db),
                [num_examples, context.input_dim],
            )
            for snr_db in context.training_snr_dbs
        )
    records = manifest["files"]
    if [record["path"] for record in records] != [name for name, _ in expected]:
        raise ValueError("Melgar training data have unexpected files.")
    for record, (_, shape) in zip(records, expected):
        if record.get("format") != "npy" or record.get("array") != {
            "shape": shape,
            "dtype": "float16",
        }:
            raise ValueError("Melgar training data have invalid arrays.")


@lru_cache(maxsize=2)
def _load_training_data_cached(
    manifest_path: str,
    manifest_mtime_ns: int,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    """Validate once and cache memory maps for one fixed data bank."""
    del manifest_mtime_ns
    path = Path(manifest_path)
    manifest = load_asset_manifest(path, asset_dir=path.parent)
    tensors = {
        record["path"]: torch.from_numpy(
            np.load(path.parent / record["path"], mmap_mode="c")
        )
        for record in manifest["files"]
    }
    return manifest, tensors


# Dataset generation


def _write_training_split(
    directory: Path,
    split: str,
    resource_grid: ResourceGrid,
    cfg: SystemConfig,
    context: _TrainingDataContext,
    *,
    group_data: Optional[dict[str, torch.Tensor]],
    channel_batches: int,
    channel_batch_size: int,
    seed: int,
) -> list[Path]:
    """Stream one common-target, all-SNR Melgar split to NPY files."""
    num_examples = (
        context.train_examples if split == "train" else context.validation_examples
    )
    target_path = directory / f"{split}-targets.npy"
    targets = np.lib.format.open_memmap(
        target_path,
        mode="w+",
        dtype=np.float16,
        shape=(num_examples, context.input_dim),
    )
    input_paths = {
        float(snr_db): directory / _training_input_filename(split, snr_db)
        for snr_db in context.training_snr_dbs
    }
    inputs = {
        snr_db: np.lib.format.open_memmap(
            path,
            mode="w+",
            dtype=np.float16,
            shape=(num_examples, context.input_dim),
        )
        for snr_db, path in input_paths.items()
    }

    set_random_seed(seed)
    channel_source = UMiChannelSource(cfg, resource_grid)
    channel_model = channel_source.channel_model
    sampler = GenerateOFDMChannel(
        channel_model,
        resource_grid,
        precision=cfg.precision,
        device=cfg.device,
    )
    if context.method == METHOD_MODEL_2:
        if group_data is None:
            raise ValueError("Model 2 training data require pilot-group metadata.")
        pilot_pattern = None
        pilot_indices = None
        pilots = None
        interpolator = None
        num_blocks = None
        selection_generator = None
    else:
        pilot_pattern = resource_grid.pilot_pattern
        pilot_indices = _pilot_ind_from_pattern(pilot_pattern, device=cfg.device)
        pilots = pilot_pattern.pilots
        interpolator = LinearInterpolator(
            pilot_pattern,
            precision=cfg.precision,
            device=cfg.device,
        )
        num_subcarriers = int(resource_grid.num_effective_subcarriers)
        num_blocks = num_subcarriers // context.input_dim
        selection_generator = torch.Generator(device=cfg.device)
        selection_generator.manual_seed(int(seed) + 300007)
    noise_generators = {}
    for index, snr_db in enumerate(context.training_snr_dbs):
        generator = torch.Generator(device=cfg.device)
        generator.manual_seed(int(seed) + 200003 + index)
        noise_generators[float(snr_db)] = generator

    offset = 0
    try:
        for batch_index in range(int(channel_batches)):
            topology = gen_single_sector_topology(
                batch_size=channel_batch_size,
                num_ut=cfg.num_ue,
                scenario="umi",
                min_ut_velocity=cfg.ue_speed,
                max_ut_velocity=cfg.ue_speed,
                precision=cfg.precision,
                device=cfg.device,
            )
            channel_model.set_topology(*topology)
            channel = sampler(channel_batch_size)
            if context.method == METHOD_MODEL_2:
                group_values = []
                target_parts = []
                for group_index in range(
                    int(group_data["subcarriers"].shape[0])
                ):
                    values = channel[
                        :,
                        :,
                        :,
                        int(group_data["tx"][group_index].item()),
                        int(group_data["stream"][group_index].item()),
                        int(group_data["symbol"][group_index].item()),
                        :,
                    ].index_select(
                        -1,
                        group_data["subcarriers"][group_index].to(channel.device),
                    )
                    values = values.reshape(-1, context.input_dim).to(
                        torch.complex64
                    )
                    group_values.append(values)
                    target_parts.extend([values.real, values.imag])
                batch_targets = torch.cat(target_parts, dim=0).to(torch.float32)
                selected_indices = None
            else:
                assert num_blocks is not None
                target_blocks = channel.reshape(
                    *channel.shape[:-1], num_blocks, context.input_dim
                )
                flat_targets = target_blocks.reshape(-1, context.input_dim)
                batch_targets = torch.cat(
                    [flat_targets.real, flat_targets.imag], dim=0
                ).to(torch.float32)
                max_examples = context.compatibility[METHOD_MODEL_3][
                    "max_examples_per_batch"
                ]
                if (
                    max_examples is not None
                    and batch_targets.shape[0] > max_examples
                ):
                    assert selection_generator is not None
                    selected_indices = torch.randperm(
                        batch_targets.shape[0],
                        generator=selection_generator,
                        device=batch_targets.device,
                    )[:max_examples]
                    batch_targets = batch_targets.index_select(
                        0, selected_indices
                    )
                else:
                    selected_indices = None

            next_offset = offset + int(batch_targets.shape[0])
            targets[offset:next_offset] = batch_targets.cpu().numpy().astype(
                np.float16,
                copy=False,
            )
            for snr_db in context.training_snr_dbs:
                noise_variance = ebnodb2no(
                    torch.tensor(
                        snr_db,
                        dtype=torch.float32,
                        device=channel.device,
                    ),
                    cfg.num_bits_per_symbol,
                    cfg.coderate,
                    resource_grid,
                    precision=cfg.precision,
                    device=cfg.device,
                )
                generator = noise_generators[float(snr_db)]
                if context.method == METHOD_MODEL_2:
                    input_parts = []
                    for group_index, values in enumerate(group_values):
                        pilot_power = (
                            group_data["pilot_values"][group_index]
                            .to(channel.device)
                            .abs()
                            .square()
                        )
                        noise_scale = torch.sqrt(
                            noise_variance.to(channel.real.dtype)
                            / pilot_power.to(channel.real.dtype)
                            / 2.0
                        ).view(1, -1)
                        noise = torch.complex(
                            torch.randn(
                                values.shape,
                                dtype=torch.float32,
                                device=channel.device,
                                generator=generator,
                            ),
                            torch.randn(
                                values.shape,
                                dtype=torch.float32,
                                device=channel.device,
                                generator=generator,
                            ),
                        )
                        noisy_values = values + noise_scale * noise
                        input_parts.extend(
                            [noisy_values.real, noisy_values.imag]
                        )
                    batch_inputs = torch.cat(input_parts, dim=0).to(
                        torch.float32
                    )
                else:
                    assert pilot_indices is not None
                    assert pilots is not None
                    assert interpolator is not None
                    assert num_blocks is not None
                    flat_channel = channel.flatten(start_dim=-2)
                    gather_indices = (
                        pilot_indices.to(channel.device)
                        .unsqueeze(0)
                        .unsqueeze(0)
                        .unsqueeze(0)
                    ).expand(
                        channel.shape[0],
                        channel.shape[1],
                        channel.shape[2],
                        -1,
                        -1,
                        -1,
                    )
                    true_pilots = torch.gather(
                        flat_channel, -1, gather_indices
                    )
                    expanded_pilots = expand_to_rank(
                        pilots.to(channel.device), true_pilots.dim(), 0
                    )
                    pilot_power = expanded_pilots.abs().square()
                    expanded_noise_variance = expand_to_rank(
                        noise_variance, true_pilots.dim(), -1
                    )
                    ls_error_variance = torch.where(
                        pilot_power > 0.0,
                        expanded_noise_variance / pilot_power,
                        torch.zeros_like(expanded_noise_variance),
                    )
                    noise_scale = torch.sqrt(
                        ls_error_variance.to(channel.real.dtype) / 2.0
                    )
                    noise = torch.complex(
                        torch.randn(
                            true_pilots.shape,
                            dtype=torch.float32,
                            device=channel.device,
                            generator=generator,
                        ),
                        torch.randn(
                            true_pilots.shape,
                            dtype=torch.float32,
                            device=channel.device,
                            generator=generator,
                        ),
                    )
                    noisy_pilots = torch.where(
                        pilot_power > 0.0,
                        true_pilots + noise_scale * noise,
                        torch.zeros_like(true_pilots),
                    )
                    interpolated_channel, _ = interpolator(
                        noisy_pilots,
                        ls_error_variance,
                    )
                    input_blocks = interpolated_channel.reshape(
                        *interpolated_channel.shape[:-1],
                        num_blocks,
                        context.input_dim,
                    )
                    flat_inputs = input_blocks.reshape(
                        -1, context.input_dim
                    )
                    batch_inputs = torch.cat(
                        [flat_inputs.real, flat_inputs.imag], dim=0
                    ).to(torch.float32)
                if selected_indices is not None:
                    batch_inputs = batch_inputs.index_select(0, selected_indices)
                inputs[float(snr_db)][offset:next_offset] = (
                    batch_inputs.cpu().numpy().astype(np.float16, copy=False)
                )
            offset = next_offset
            if (batch_index + 1) % max(1, int(channel_batches) // 10) == 0:
                print(
                    f"[melgar2022] {split} {context.method} channel batch "
                    f"{batch_index + 1}/{channel_batches}"
                )
    finally:
        targets.flush()
        for values in inputs.values():
            values.flush()

    if offset != num_examples:
        raise RuntimeError(
            f"Expected {num_examples} Model 3 examples, generated {offset}."
            if context.method == METHOD_MODEL_3
            else f"Expected {num_examples} Model 2 examples, generated {offset}."
        )
    del targets, inputs
    return [target_path, *input_paths.values()]


def _collect_model2_training_pairs(
    resource_grid: ResourceGrid,
    cfg: SystemConfig,
    group_data: dict[str, torch.Tensor],
    *,
    input_dim: int,
    channel_batches: int,
    channel_batch_size: int,
    training_snr_dbs: list[float] | None,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate noisy LS and true pilot-vector pairs for Model 2."""
    set_random_seed(seed)

    channel_source = UMiChannelSource(cfg, resource_grid)
    channel_model = channel_source.channel_model
    sampler = GenerateOFDMChannel(
        channel_model,
        resource_grid,
        precision=cfg.precision,
        device=cfg.device,
    )
    snr_dbs = _training_snr_list(cfg, training_snr_dbs)
    print(
        f"[melgar2022] collecting DNN pairs: channel_batches={channel_batches} "
        f"batch_size={channel_batch_size} training_snr_dbs={snr_dbs}"
    )
    inputs: torch.Tensor | None = None
    targets: torch.Tensor | None = None
    offset = 0
    group_tx_indices = group_data["tx"]
    group_stream_indices = group_data["stream"]
    group_symbol_indices = group_data["symbol"]
    group_subcarrier_indices = group_data["subcarriers"]
    group_pilot_values = group_data["pilot_values"]

    for batch_index in range(channel_batches):
        topology = gen_single_sector_topology(
            batch_size=channel_batch_size,
            num_ut=cfg.num_ue,
            scenario="umi",
            min_ut_velocity=cfg.ue_speed,
            max_ut_velocity=cfg.ue_speed,
            precision=cfg.precision,
            device=cfg.device,
        )
        channel_model.set_topology(*topology)
        channel = sampler(channel_batch_size)

        snr_db = float(snr_dbs[batch_index % len(snr_dbs)])
        noise_variance = ebnodb2no(
            torch.tensor(snr_db, dtype=torch.float32, device=channel.device),
            cfg.num_bits_per_symbol,
            cfg.coderate,
            resource_grid,
            precision=cfg.precision,
            device=cfg.device,
        )
        input_parts: list[torch.Tensor] = []
        target_parts: list[torch.Tensor] = []
        for group_index in range(int(group_subcarrier_indices.shape[0])):
            tx_index = int(group_tx_indices[group_index].item())
            stream_index = int(group_stream_indices[group_index].item())
            symbol_index = int(group_symbol_indices[group_index].item())
            subcarrier_indices = group_subcarrier_indices[group_index].to(
                channel.device
            )
            values = channel[
                :, :, :, tx_index, stream_index, symbol_index, :
            ].index_select(-1, subcarrier_indices)
            values = values.reshape(-1, values.shape[-1]).to(torch.complex64)
            pilot_power = (
                group_pilot_values[group_index]
                .to(channel.device)
                .abs()
                .square()
            )
            noise_scale = torch.sqrt(
                noise_variance.to(channel.real.dtype)
                / pilot_power.to(channel.real.dtype)
                / 2.0
            ).view(1, -1)
            noise = torch.complex(
                torch.randn(
                    values.shape,
                    dtype=torch.float32,
                    device=channel.device,
                ),
                torch.randn(
                    values.shape,
                    dtype=torch.float32,
                    device=channel.device,
                ),
            )
            noisy_values = values + noise_scale * noise
            input_parts.extend([noisy_values.real, noisy_values.imag])
            target_parts.extend([values.real, values.imag])

        batch_inputs = torch.cat(input_parts, dim=0)
        batch_targets = torch.cat(target_parts, dim=0)
        if batch_inputs.shape[-1] != int(input_dim):
            raise ValueError(
                f"Expected input_dim={input_dim}, got {batch_inputs.shape[-1]}."
            )

        if inputs is None or targets is None:
            total_examples = int(channel_batches) * int(batch_inputs.shape[0])
            inputs = torch.empty(
                (total_examples, int(input_dim)),
                dtype=torch.float32,
                device=batch_inputs.device,
            )
            targets = torch.empty_like(inputs)

        next_offset = offset + batch_inputs.shape[0]
        inputs[offset:next_offset].copy_(batch_inputs)
        targets[offset:next_offset].copy_(batch_targets)
        offset = next_offset

        if (batch_index + 1) % max(1, channel_batches // 4) == 0:
            print(
                f"[melgar2022] DNN training channel batch {batch_index + 1}/"
                f"{channel_batches}"
            )

    if inputs is None or targets is None:
        raise ValueError("The training set must contain at least one batch.")
    return inputs[:offset], targets[:offset]


def _collect_model3_training_pairs(
    resource_grid: ResourceGrid,
    cfg: SystemConfig,
    *,
    input_dim: int,
    channel_batches: int,
    channel_batch_size: int,
    training_snr_dbs: list[float] | None,
    seed: int,
    max_examples_per_batch: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate LS-linear frequency blocks and true targets for Model 3."""
    set_random_seed(seed)

    channel_source = UMiChannelSource(cfg, resource_grid)
    channel_model = channel_source.channel_model
    sampler = GenerateOFDMChannel(
        channel_model,
        resource_grid,
        precision=cfg.precision,
        device=cfg.device,
    )
    pilot_pattern = resource_grid.pilot_pattern
    pilot_indices = _pilot_ind_from_pattern(
        pilot_pattern,
        device=cfg.device,
    )
    pilots = pilot_pattern.pilots
    interpolator = LinearInterpolator(
        pilot_pattern,
        precision=cfg.precision,
        device=cfg.device,
    )
    snr_dbs = _training_snr_list(cfg, training_snr_dbs)
    num_effective_subcarriers = int(resource_grid.num_effective_subcarriers)
    assert num_effective_subcarriers % int(input_dim) == 0, (
        "`input_dim` must divide the number of effective subcarriers."
    )

    print(
        f"[melgar2022] collecting Model 3 DNN pairs: "
        f"channel_batches={channel_batches} batch_size={channel_batch_size} "
        f"training_snr_dbs={snr_dbs} block_len={input_dim}"
    )

    inputs: torch.Tensor | None = None
    targets: torch.Tensor | None = None
    offset = 0
    sample_generator: torch.Generator | None = None

    for batch_index in range(channel_batches):
        topology = gen_single_sector_topology(
            batch_size=channel_batch_size,
            num_ut=cfg.num_ue,
            scenario="umi",
            min_ut_velocity=cfg.ue_speed,
            max_ut_velocity=cfg.ue_speed,
            precision=cfg.precision,
            device=cfg.device,
        )
        channel_model.set_topology(*topology)
        channel = sampler(channel_batch_size)
        if sample_generator is None:
            sample_generator = torch.Generator(device=channel.device)
            sample_generator.manual_seed(seed + 200003)

        snr_db = float(snr_dbs[batch_index % len(snr_dbs)])
        noise_variance = ebnodb2no(
            torch.tensor(snr_db, dtype=torch.float32, device=channel.device),
            cfg.num_bits_per_symbol,
            cfg.coderate,
            resource_grid,
            precision=cfg.precision,
            device=cfg.device,
        )

        flat_channel = channel.flatten(start_dim=-2)
        gather_indices = (
            pilot_indices.to(channel.device)
            .unsqueeze(0)
            .unsqueeze(0)
            .unsqueeze(0)
        )
        gather_indices = gather_indices.expand(
            channel.shape[0],
            channel.shape[1],
            channel.shape[2],
            -1,
            -1,
            -1,
        )
        true_pilots = torch.gather(flat_channel, -1, gather_indices)
        expanded_pilots = expand_to_rank(
            pilots.to(channel.device), true_pilots.dim(), 0
        )
        expanded_noise_variance = expand_to_rank(
            noise_variance, true_pilots.dim(), -1
        )
        pilot_power = expanded_pilots.abs().square()
        ls_error_variance = torch.where(
            pilot_power > 0.0,
            expanded_noise_variance / pilot_power,
            torch.zeros_like(expanded_noise_variance),
        )
        noise_scale = torch.sqrt(
            ls_error_variance.to(channel.real.dtype) / 2.0
        )
        noise = torch.complex(
            torch.randn(
                true_pilots.shape,
                dtype=torch.float32,
                device=channel.device,
            ),
            torch.randn(
                true_pilots.shape,
                dtype=torch.float32,
                device=channel.device,
            ),
        )
        noisy_pilots = torch.where(
            pilot_power > 0.0,
            true_pilots + noise_scale * noise,
            torch.zeros_like(true_pilots),
        )
        interpolated_channel, _ = interpolator(
            noisy_pilots, ls_error_variance
        )

        num_blocks = num_effective_subcarriers // int(input_dim)
        input_blocks = interpolated_channel.reshape(
            *interpolated_channel.shape[:-1], num_blocks, int(input_dim)
        )
        target_blocks = channel.reshape(
            *channel.shape[:-1], num_blocks, int(input_dim)
        )
        flat_inputs = input_blocks.reshape(-1, int(input_dim))
        flat_targets = target_blocks.reshape(-1, int(input_dim))
        batch_inputs = torch.cat(
            [flat_inputs.real, flat_inputs.imag], dim=0
        ).to(torch.float32)
        batch_targets = torch.cat(
            [flat_targets.real, flat_targets.imag], dim=0
        ).to(torch.float32)

        if (
            max_examples_per_batch is not None
            and batch_inputs.shape[0] > int(max_examples_per_batch)
        ):
            selected_indices = torch.randperm(
                batch_inputs.shape[0],
                generator=sample_generator,
                device=batch_inputs.device,
            )[: int(max_examples_per_batch)]
            batch_inputs = batch_inputs.index_select(0, selected_indices)
            batch_targets = batch_targets.index_select(0, selected_indices)

        if inputs is None or targets is None:
            total_examples = int(channel_batches) * int(batch_inputs.shape[0])
            inputs = torch.empty(
                (total_examples, int(input_dim)),
                dtype=torch.float32,
                device=batch_inputs.device,
            )
            targets = torch.empty_like(inputs)

        next_offset = offset + batch_inputs.shape[0]
        inputs[offset:next_offset].copy_(batch_inputs)
        targets[offset:next_offset].copy_(batch_targets)
        offset = next_offset

        if (batch_index + 1) % max(1, channel_batches // 4) == 0:
            print(
                f"[melgar2022] Model 3 DNN training channel batch "
                f"{batch_index + 1}/{channel_batches}"
            )

    if inputs is None or targets is None:
        raise ValueError("The training set must contain at least one batch.")
    return inputs[:offset], targets[:offset]


def _float_token(value: float) -> str:
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def _training_snr_model_cache_suffix(training_snr_dbs: list[float] | None) -> str:
    if training_snr_dbs is None:
        return ""
    return "_trsnr" + "-".join(_float_token(x) for x in training_snr_dbs)


def _example_cap_token(max_examples_per_batch: int | None) -> str:
    if max_examples_per_batch is None:
        return "all"
    return str(int(max_examples_per_batch))


# Model asset contexts


def _model2_asset_context(
    cfg,
    group_data: dict[str, torch.Tensor],
    *,
    input_dim: int,
    train_batches: int,
    train_batch_size: int,
    validation_batches: int,
    validation_batch_size: int,
    epochs: int,
    optimizer_batch_size: int,
    learning_rate: float,
    min_train_seconds: float,
    patience_epochs: int,
    min_delta: float,
    training_snr_dbs: list[float] | None,
    seed: int,
    asset_root: str | Path | None = None,
) -> _ModelAssetContext:
    """Resolve one Model 2 checkpoint identity and compatibility contract."""
    name = (
        f"n{input_dim}_tb{train_batches}_bs{train_batch_size}"
        f"_vb{validation_batches}_vbs{validation_batch_size}"
        f"_ep{epochs}_mb{optimizer_batch_size}_lr{learning_rate:g}"
        f"_mins{min_train_seconds:g}_pat{patience_epochs}_md{min_delta:g}"
        f"{_training_snr_model_cache_suffix(training_snr_dbs)}"
        f"_h{HIDDEN_SIZES[-1]}_seed{seed}"
    )
    artifact_id = name.replace("_", "-").replace(".", "p")
    compatibility, generation_recipe = _model2_manifest_fields(
        cfg,
        group_data,
        input_dim=input_dim,
        train_batches=train_batches,
        train_batch_size=train_batch_size,
        validation_batches=validation_batches,
        validation_batch_size=validation_batch_size,
        epochs=epochs,
        optimizer_batch_size=optimizer_batch_size,
        learning_rate=learning_rate,
        min_train_seconds=min_train_seconds,
        patience_epochs=patience_epochs,
        min_delta=min_delta,
        training_snr_dbs=training_snr_dbs,
        seed=seed,
    )
    return _ModelAssetContext(
        asset_type="melgar-model2",
        scenario_id=cfg.scenario_id,
        input_dim=input_dim,
        asset_dir=_model_asset_dir(
            cfg,
            "melgar-model2",
            artifact_id,
            asset_root,
        ),
        compatibility=compatibility,
        generation_recipe=generation_recipe,
    )


def _model2_manifest_fields(
    cfg,
    group_data: dict[str, torch.Tensor],
    *,
    input_dim: int,
    train_batches: int,
    train_batch_size: int,
    validation_batches: int,
    validation_batch_size: int,
    epochs: int,
    optimizer_batch_size: int,
    learning_rate: float,
    min_train_seconds: float,
    patience_epochs: int,
    min_delta: float,
    training_snr_dbs: list[float] | None,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    compatibility = {
        "model_cache_type": "melgar2022_model2_pilot_dnn",
        "input_dim": int(input_dim),
        "dimension_adaptation": "input_output_match_active_pilot_group_length",
        "hidden_sizes": list(HIDDEN_SIZES),
        "activation": "relu",
        "output_activation": "linear",
        "ls_noise_scaling": "no_over_abs_pilot_squared",
        "channel_sampler": "sionna_umi_3gpp_config",
        "pilot_groups": _model2_pilot_layout(group_data),
    }
    generation_recipe = {
        "optimizer": "rmsprop",
        "loss": "mse",
        "learning_rate": float(learning_rate),
        "optimizer_batch_size": int(optimizer_batch_size),
        "train_batches": int(train_batches),
        "train_batch_size": int(train_batch_size),
        "validation_batches": int(validation_batches),
        "validation_batch_size": int(validation_batch_size),
        "epochs": int(epochs),
        "min_train_seconds": float(min_train_seconds),
        "patience_epochs": int(patience_epochs),
        "min_delta": float(min_delta),
        "seed": int(seed),
        "validation_seed": int(seed) + 100003,
        "training_snr_dbs": _training_snr_list(cfg, training_snr_dbs),
    }
    return compatibility, generation_recipe


# Payload persistence and validation


def _save_state_npz(path: Path, state: dict[str, torch.Tensor]) -> list[str]:
    """Write one ordered DNN state and return its parameter names."""
    keys = list(state.keys())
    arrays = {
        f"arr_{i}": value.detach().cpu().numpy()
        for i, value in enumerate(state.values())
    }
    write_npz_atomic(path, **arrays)
    return keys


def _load_state_npz(
    path: Path,
    keys: list[str],
    *,
    input_dim: int,
) -> dict[str, torch.Tensor]:
    with torch.random.fork_rng(devices=[]):
        expected_state = DNN(input_dim).state_dict()
    expected_keys = list(expected_state)
    if keys != expected_keys:
        raise ValueError("Melgar DNN asset has incompatible state keys.")

    with np.load(path, allow_pickle=False) as data:
        expected_arrays = {f"arr_{i}" for i in range(len(keys))}
        if set(data.files) != expected_arrays:
            raise ValueError("Melgar DNN asset has incompatible state arrays.")
        state = {}
        for index, key in enumerate(keys):
            array = data[f"arr_{index}"]
            if array.shape != tuple(expected_state[key].shape):
                raise ValueError(
                    f"Melgar DNN tensor {key!r} has incompatible shape."
                )
            if array.dtype != np.float32:
                raise ValueError(
                    f"Melgar DNN tensor {key!r} has incompatible dtype."
                )
            state[key] = torch.from_numpy(array.copy())
        return state


def _load_model_asset(
    context: _ModelAssetContext,
) -> tuple[dict[str, torch.Tensor], Path]:
    """Strictly load and validate one prepared DNN asset."""
    if not context.manifest_path.is_file():
        if context.asset_dir.exists() and any(context.asset_dir.iterdir()):
            raise ValueError(f"Incomplete DNN asset at {context.asset_dir}.")
        method = context.asset_type.removeprefix("melgar-")
        raise FileNotFoundError(
            f"No compatible Melgar DNN asset at {context.asset_dir}. Generate "
            "it with `python -m "
            "sionna.bench.mimo_ofdm_chest.papers.melgar2022.run_benchmark "
            f"--prepare-assets --method {method} "
            "--asset-root <path>`, then evaluate it with `python -m "
            "sionna.bench.mimo_ofdm_chest.papers.melgar2022.run_benchmark "
            f"--method {method} "
            "--asset-root <path>`."
        )
    manifest = load_asset_manifest(
        context.manifest_path,
        asset_dir=context.asset_dir,
        expected_asset_type=context.asset_type,
        expected_scenario_id=context.scenario_id,
        expected_artifact_id=context.asset_dir.name,
        expected_compatibility=context.compatibility,
    )
    if manifest["generation_recipe"] != context.generation_recipe:
        raise ValueError(
            f"Incompatible DNN generation recipe at {context.asset_dir}."
        )
    payload_names = [record["path"] for record in manifest["files"]]
    if payload_names != [context.params_path.name]:
        raise ValueError("Melgar DNN asset has unexpected payload files.")
    metadata = manifest.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("Melgar DNN asset has invalid metadata.")
    state_keys = metadata.get("state_keys")
    if not isinstance(state_keys, list) or not all(
        isinstance(key, str) for key in state_keys
    ):
        raise ValueError("Melgar DNN asset has invalid state keys.")
    history = metadata.get("history")
    if not isinstance(history, dict):
        raise ValueError("Melgar DNN asset has invalid training history.")
    state = _load_state_npz(
        context.params_path,
        state_keys,
        input_dim=context.input_dim,
    )
    return state, context.manifest_path


def _write_model_asset(
    context: _ModelAssetContext,
    state: dict[str, torch.Tensor],
    history: dict[str, Any],
    *,
    cfg,
    overwrite: bool,
) -> tuple[dict[str, torch.Tensor], Path]:
    """Atomically persist one DNN asset and strictly reload it."""
    if context.asset_dir.exists() and not overwrite:
        raise FileExistsError(f"DNN asset already exists at {context.asset_dir}.")
    context.asset_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{context.asset_dir.name}.",
            dir=context.asset_dir.parent,
        )
    )
    try:
        temporary_params_path = temporary_dir / context.params_path.name
        state_keys = _save_state_npz(temporary_params_path, state)
        manifest = build_asset_manifest(
            cfg=cfg,
            asset_type=context.asset_type,
            artifact_id=context.asset_dir.name,
            compatibility=context.compatibility,
            generation_recipe=context.generation_recipe,
            payload_paths=[temporary_params_path],
            metadata={"state_keys": state_keys, "history": history},
        )
        write_json_atomic(temporary_dir / "manifest.json", manifest)
        if context.asset_dir.exists():
            shutil.rmtree(context.asset_dir)
        os.replace(temporary_dir, context.asset_dir)
    finally:
        shutil.rmtree(temporary_dir, ignore_errors=True)
    return _load_model_asset(context)


def _mse_in_batches(
    model: torch.nn.Module | None,
    x: torch.Tensor,
    y: torch.Tensor,
    batch_size: int,
) -> float:
    num = 0.0
    den = 0
    for start in range(0, x.shape[0], int(batch_size)):
        xb = x[start : start + int(batch_size)]
        yb = y[start : start + int(batch_size)]
        pred = xb if model is None else model(xb)
        err = pred - yb
        num += err.square().sum().item()
        den += err.numel()
    return num / max(den, 1)


# Model training


def _state_to_cpu(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in state.items()}


def _train_state(
    x: torch.Tensor,
    y: torch.Tensor,
    x_val: torch.Tensor | None,
    y_val: torch.Tensor | None,
    *,
    input_dim: int,
    epochs: int,
    optimizer_batch_size: int,
    learning_rate: float,
    min_train_seconds: float,
    patience_epochs: int,
    min_delta: float,
    seed: int,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    from .estimator import DNN

    device = x.device
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    if x_val is None or y_val is None:
        perm = torch.randperm(x.shape[0], generator=gen, device=device)
        val_count = max(1, int(0.1 * x.shape[0]))
        val_idx = perm[:val_count]
        train_idx = perm[val_count:]
        x_train = x[train_idx]
        y_train = y[train_idx]
        x_val = x[val_idx]
        y_val = y[val_idx]
        split_mode = "random_example_split"
    else:
        x_train = x
        y_train = y
        split_mode = "separate_channel_holdout"

    fork_devices = [device.index or 0] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=fork_devices):
        torch.manual_seed(seed)
        model = DNN(input_dim).to(device=device, dtype=torch.float32)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=float(learning_rate))
    loss_fn = torch.nn.MSELoss()

    eval_batch_size = max(int(optimizer_batch_size), 16384)
    train_ls_mse = _mse_in_batches(None, x_train, y_train, eval_batch_size)
    val_ls_mse = _mse_in_batches(None, x_val, y_val, eval_batch_size)
    print(
        f"[melgar2022] DNN dataset split={split_mode} "
        f"train_examples={x_train.shape[0]} val_examples={x_val.shape[0]} "
        f"train_ls_mse={train_ls_mse:.6e} val_ls_mse={val_ls_mse:.6e}"
    )

    trace = []
    log_every = 1
    best_state = _state_to_cpu(model.state_dict())
    best_epoch = 0
    best_val_mse = float("inf")
    best_train_mse = float("inf")
    t0 = time.time()
    for epoch in range(int(epochs)):
        order = torch.randperm(x_train.shape[0], generator=gen, device=device)
        for start in range(0, order.numel(), int(optimizer_batch_size)):
            idx = order[start : start + int(optimizer_batch_size)]
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x_train[idx]), y_train[idx])
            loss.backward()
            optimizer.step()
        if (epoch + 1) % log_every == 0 or epoch + 1 == int(epochs):
            with torch.no_grad():
                train_loss = _mse_in_batches(
                    model, x_train, y_train, eval_batch_size
                )
                val_loss = _mse_in_batches(model, x_val, y_val, eval_batch_size)
            improved = (
                not np.isfinite(best_val_mse)
                or val_loss < best_val_mse * (1.0 - float(min_delta))
            )
            if improved:
                best_val_mse = float(val_loss)
                best_train_mse = float(train_loss)
                best_epoch = int(epoch + 1)
                best_state = _state_to_cpu(model.state_dict())
            elapsed = time.time() - t0
            trace.append(
                {
                    "epoch": int(epoch + 1),
                    "elapsed_seconds": float(elapsed),
                    "train_mse": float(train_loss),
                    "val_mse": float(val_loss),
                    "train_ls_mse": float(train_ls_mse),
                    "val_ls_mse": float(val_ls_mse),
                    "best_epoch": int(best_epoch),
                    "best_val_mse": float(best_val_mse),
                }
            )
            print(
                f"[melgar2022] DNN epoch {epoch + 1}/{epochs} "
                f"elapsed={elapsed:.1f}s train_mse={train_loss:.6e} "
                f"val_mse={val_loss:.6e} val_ls_mse={val_ls_mse:.6e} "
                f"best_val={best_val_mse:.6e}@{best_epoch}"
            )
            saturated = (
                int(patience_epochs) > 0
                and best_epoch > 0
                and epoch + 1 - best_epoch >= int(patience_epochs)
                and elapsed >= float(min_train_seconds)
            )
            if saturated:
                print(
                    f"[melgar2022] early stop: no validation improvement "
                    f"by at least {min_delta:g} relatively for "
                    f"{patience_epochs} epochs after "
                    f"{elapsed:.1f}s"
                )
                break

    model.load_state_dict(best_state)
    with torch.no_grad():
        train_mse = _mse_in_batches(model, x_train, y_train, eval_batch_size)
        val_mse = _mse_in_batches(model, x_val, y_val, eval_batch_size)

    history = {
        "split_mode": split_mode,
        "train_mse": float(train_mse),
        "val_mse": float(val_mse),
        "best_train_mse": float(best_train_mse),
        "best_val_mse": float(best_val_mse),
        "best_epoch": int(best_epoch),
        "train_ls_mse": float(train_ls_mse),
        "val_ls_mse": float(val_ls_mse),
        "train_examples": int(x_train.shape[0]),
        "validation_examples": int(x_val.shape[0]),
        "elapsed_seconds": float(time.time() - t0),
        "trace": trace,
    }
    return best_state, history


def _train_model2(
    resource_grid,
    cfg,
    group_data: dict[str, torch.Tensor],
    context: _ModelAssetContext,
    *,
    input_dim: int,
    train_batches: int,
    train_batch_size: int,
    epochs: int,
    optimizer_batch_size: int,
    learning_rate: float,
    seed: int,
    validation_batches: int = 0,
    validation_batch_size: int,
    min_train_seconds: float = 0.0,
    patience_epochs: int = 0,
    min_delta: float = 0.0,
    training_snr_dbs: list[float] | None = None,
    training_data: _TrainingDataBank | None = None,
    training_data_manifest: Path | None = None,
    overwrite: bool = False,
) -> tuple[dict[str, torch.Tensor], Path]:
    """Train and persist one Model 2 state."""
    print(f"[melgar2022] training Model 2 DNN asset -> {context.asset_dir}")
    if training_data is None:
        x, y = _collect_model2_training_pairs(
            resource_grid,
            cfg,
            group_data,
            input_dim=input_dim,
            channel_batches=train_batches,
            channel_batch_size=train_batch_size,
            training_snr_dbs=training_snr_dbs,
            seed=seed,
        )
        if int(validation_batches) > 0:
            x_val, y_val = _collect_model2_training_pairs(
                resource_grid,
                cfg,
                group_data,
                input_dim=input_dim,
                channel_batches=int(validation_batches),
                channel_batch_size=int(validation_batch_size),
                training_snr_dbs=training_snr_dbs,
                seed=int(seed) + 100003,
            )
        else:
            x_val = None
            y_val = None
    else:
        selected_snr_dbs = _training_snr_list(cfg, training_snr_dbs)
        if len(selected_snr_dbs) != 1:
            raise ValueError(
                "Fixed Model 2 data require one SNR-specific model at a time."
            )
        selected_snr_db = float(selected_snr_dbs[0])
        if selected_snr_db not in training_data.train_inputs:
            raise ValueError(
                f"Training data do not contain SNR {selected_snr_db:g} dB."
            )
        x = training_data.train_inputs[selected_snr_db].to(
            device=cfg.device,
            dtype=torch.float32,
        )
        y = training_data.train_targets.to(
            device=cfg.device,
            dtype=torch.float32,
        )
        x_val = training_data.validation_inputs[selected_snr_db].to(
            device=cfg.device,
            dtype=torch.float32,
        )
        y_val = training_data.validation_targets.to(
            device=cfg.device,
            dtype=torch.float32,
        )

    state, history = _train_state(
        x,
        y,
        x_val,
        y_val,
        input_dim=input_dim,
        epochs=epochs,
        optimizer_batch_size=optimizer_batch_size,
        learning_rate=learning_rate,
        min_train_seconds=min_train_seconds,
        patience_epochs=patience_epochs,
        min_delta=min_delta,
        seed=seed,
    )
    history.update(
        {
            "train_channels": int(train_batches) * int(train_batch_size),
            "validation_channels": int(validation_batches)
            * int(validation_batch_size),
            "training_snr_dbs": _training_snr_list(cfg, training_snr_dbs),
            "training_snr_mode": (
                "config_cycle" if training_snr_dbs is None else "explicit"
            ),
            "training_data_manifest": (
                None
                if training_data_manifest is None
                else {
                    "asset_type": _MODEL2_TRAINING_DATA_ASSET_TYPE,
                    "artifact_id": training_data_manifest.parent.name,
                    "sha256": sha256_file(training_data_manifest),
                }
            ),
        }
    )
    return _write_model_asset(
        context,
        state,
        history,
        cfg=cfg,
        overwrite=overwrite,
    )


def _model3_asset_context(
    cfg,
    resource_grid: ResourceGrid,
    *,
    input_dim: int,
    train_batches: int,
    train_batch_size: int,
    validation_batches: int,
    validation_batch_size: int,
    epochs: int,
    optimizer_batch_size: int,
    learning_rate: float,
    min_train_seconds: float,
    patience_epochs: int,
    min_delta: float,
    training_snr_dbs: list[float] | None,
    max_examples_per_batch: int | None,
    seed: int,
    asset_root: Optional[Union[str, Path]] = None,
) -> _ModelAssetContext:
    """Resolve one Model 3 checkpoint identity and compatibility contract."""
    name = (
        f"n{input_dim}_tb{train_batches}_bs{train_batch_size}"
        f"_vb{validation_batches}_vbs{validation_batch_size}"
        f"_ep{epochs}_mb{optimizer_batch_size}_lr{learning_rate:g}"
        f"_mins{min_train_seconds:g}_pat{patience_epochs}_md{min_delta:g}"
        f"_mx{_example_cap_token(max_examples_per_batch)}"
        f"{_training_snr_model_cache_suffix(training_snr_dbs)}"
        f"_h{HIDDEN_SIZES[-1]}_seed{seed}"
    )
    artifact_id = name.replace("_", "-").replace(".", "p")
    compatibility, generation_recipe = _model3_manifest_fields(
        cfg,
        resource_grid,
        input_dim=input_dim,
        train_batches=train_batches,
        train_batch_size=train_batch_size,
        validation_batches=validation_batches,
        validation_batch_size=validation_batch_size,
        epochs=epochs,
        optimizer_batch_size=optimizer_batch_size,
        learning_rate=learning_rate,
        min_train_seconds=min_train_seconds,
        patience_epochs=patience_epochs,
        min_delta=min_delta,
        training_snr_dbs=training_snr_dbs,
        max_examples_per_batch=max_examples_per_batch,
        seed=seed,
    )
    return _ModelAssetContext(
        asset_type="melgar-model3",
        scenario_id=cfg.scenario_id,
        input_dim=input_dim,
        asset_dir=_model_asset_dir(
            cfg,
            "melgar-model3",
            artifact_id,
            asset_root,
        ),
        compatibility=compatibility,
        generation_recipe=generation_recipe,
    )


def _model3_manifest_fields(
    cfg,
    resource_grid,
    *,
    input_dim: int,
    train_batches: int,
    train_batch_size: int,
    validation_batches: int,
    validation_batch_size: int,
    epochs: int,
    optimizer_batch_size: int,
    learning_rate: float,
    min_train_seconds: float,
    patience_epochs: int,
    min_delta: float,
    training_snr_dbs: list[float] | None,
    max_examples_per_batch: int | None,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    num_effective_subcarriers = int(resource_grid.num_effective_subcarriers)
    compatibility = {
        "model_cache_type": "melgar2022_model3_interpolated_dnn",
        "input_source": "ls_linear_full_grid",
        "interpolation": "lin",
        "input_dim": int(input_dim),
        "dimension_adaptation": "frequency_block_length_divides_effective_subcarriers",
        "num_effective_subcarriers": num_effective_subcarriers,
        "num_frequency_blocks": num_effective_subcarriers // int(input_dim),
        "hidden_sizes": list(HIDDEN_SIZES),
        "activation": "relu",
        "output_activation": "linear",
        "ls_noise_scaling": "no_over_abs_pilot_squared",
        "channel_sampler": "sionna_umi_3gpp_config",
        "ls_interpolation_layout": _model3_ls_interpolation_layout(
            resource_grid
        ),
    }
    generation_recipe = {
        "optimizer": "rmsprop",
        "loss": "mse",
        "learning_rate": float(learning_rate),
        "optimizer_batch_size": int(optimizer_batch_size),
        "train_batches": int(train_batches),
        "train_batch_size": int(train_batch_size),
        "validation_batches": int(validation_batches),
        "validation_batch_size": int(validation_batch_size),
        "epochs": int(epochs),
        "min_train_seconds": float(min_train_seconds),
        "patience_epochs": int(patience_epochs),
        "min_delta": float(min_delta),
        "max_examples_per_batch": (
            None if max_examples_per_batch is None else int(max_examples_per_batch)
        ),
        "seed": int(seed),
        "validation_seed": int(seed) + 100003,
        "training_snr_dbs": _training_snr_list(cfg, training_snr_dbs),
    }
    return compatibility, generation_recipe


def _train_model3(
    resource_grid,
    cfg,
    context: _ModelAssetContext,
    *,
    input_dim: int,
    train_batches: int,
    train_batch_size: int,
    epochs: int,
    optimizer_batch_size: int,
    learning_rate: float,
    seed: int,
    validation_batches: int = 0,
    validation_batch_size: int,
    min_train_seconds: float = 0.0,
    patience_epochs: int = 0,
    min_delta: float = 0.0,
    training_snr_dbs: list[float] | None = None,
    max_examples_per_batch: int | None = None,
    training_data: _TrainingDataBank | None = None,
    training_data_manifest: Path | None = None,
    overwrite: bool = False,
) -> tuple[dict[str, torch.Tensor], Path]:
    """Train and persist one Model 3 state."""
    print(f"[melgar2022] training Model 3 DNN asset -> {context.asset_dir}")
    if training_data is None:
        x, y = _collect_model3_training_pairs(
            resource_grid,
            cfg,
            input_dim=input_dim,
            channel_batches=train_batches,
            channel_batch_size=train_batch_size,
            training_snr_dbs=training_snr_dbs,
            seed=seed,
            max_examples_per_batch=max_examples_per_batch,
        )
        if int(validation_batches) > 0:
            x_val, y_val = _collect_model3_training_pairs(
                resource_grid,
                cfg,
                input_dim=input_dim,
                channel_batches=int(validation_batches),
                channel_batch_size=int(validation_batch_size),
                training_snr_dbs=training_snr_dbs,
                seed=int(seed) + 100003,
                max_examples_per_batch=max_examples_per_batch,
            )
        else:
            x_val = None
            y_val = None
    else:
        selected_snr_dbs = _training_snr_list(cfg, training_snr_dbs)
        if len(selected_snr_dbs) != 1:
            raise ValueError(
                "Fixed Model 3 data require one SNR-specific model at a time."
            )
        selected_snr_db = float(selected_snr_dbs[0])
        if selected_snr_db not in training_data.train_inputs:
            raise ValueError(
                f"Training data do not contain SNR {selected_snr_db:g} dB."
            )
        x = training_data.train_inputs[selected_snr_db].to(
            device=cfg.device,
            dtype=torch.float32,
        )
        y = training_data.train_targets.to(
            device=cfg.device,
            dtype=torch.float32,
        )
        x_val = training_data.validation_inputs[selected_snr_db].to(
            device=cfg.device,
            dtype=torch.float32,
        )
        y_val = training_data.validation_targets.to(
            device=cfg.device,
            dtype=torch.float32,
        )

    state, history = _train_state(
        x,
        y,
        x_val,
        y_val,
        input_dim=input_dim,
        epochs=epochs,
        optimizer_batch_size=optimizer_batch_size,
        learning_rate=learning_rate,
        min_train_seconds=min_train_seconds,
        patience_epochs=patience_epochs,
        min_delta=min_delta,
        seed=seed,
    )
    history.update(
        {
            "train_channels": int(train_batches) * int(train_batch_size),
            "validation_channels": int(validation_batches)
            * int(validation_batch_size),
            "training_snr_dbs": _training_snr_list(cfg, training_snr_dbs),
            "training_snr_mode": (
                "config_cycle" if training_snr_dbs is None else "explicit"
            ),
            "max_examples_per_batch": max_examples_per_batch,
            "training_data_manifest": (
                None
                if training_data_manifest is None
                else {
                    "asset_type": _MODEL3_TRAINING_DATA_ASSET_TYPE,
                    "artifact_id": training_data_manifest.parent.name,
                    "sha256": sha256_file(training_data_manifest),
                }
            ),
        }
    )
    return _write_model_asset(
        context,
        state,
        history,
        cfg=cfg,
        overwrite=overwrite,
    )


# Ordered model-bank orchestration


def _model_contexts(
    resource_grid: ResourceGrid,
    cfg: SystemConfig,
    *,
    method: str = METHOD_MODEL_2,
    block_len: int = 78,
    train_batches: int = 8,
    train_batch_size: int = 8,
    validation_batches: int = 0,
    validation_batch_size: int | None = None,
    epochs: int = 100,
    optimizer_batch_size: int = 500,
    learning_rate: float = 5e-4,
    min_train_seconds: float = 0.0,
    patience_epochs: int = 0,
    min_delta: float = 0.0,
    max_examples_per_batch: int | None = None,
    snr_specific_weights: bool = True,
    training_snr_dbs: list[float] | None = None,
    seed: int = 42,
    asset_root: Optional[Union[str, Path]] = None,
) -> tuple[
    list[_ModelAssetContext],
    Optional[torch.Tensor],
    Optional[dict[str, torch.Tensor]],
]:
    """Resolve ordered model assets and their inference noise grid."""
    if method not in METHOD_NAMES:
        raise ValueError(f"`method` must be one of {METHOD_NAMES}.")

    effective_validation_batch_size = (
        train_batch_size
        if validation_batch_size is None
        else int(validation_batch_size)
    )
    snr_dbs = _training_snr_list(cfg, training_snr_dbs)

    if method == METHOD_MODEL_2:
        layout = _PilotGroupLayout(resource_grid)
        group_data = layout.training_group_data()
        input_dim = int(group_data["pilot_values"].shape[-1])

        def context_for(selected_snr_dbs: Optional[list[float]]):
            return _model2_asset_context(
                cfg,
                group_data,
                input_dim=input_dim,
                train_batches=int(train_batches),
                train_batch_size=int(train_batch_size),
                validation_batches=int(validation_batches),
                validation_batch_size=effective_validation_batch_size,
                epochs=int(epochs),
                optimizer_batch_size=int(optimizer_batch_size),
                learning_rate=float(learning_rate),
                min_train_seconds=float(min_train_seconds),
                patience_epochs=int(patience_epochs),
                min_delta=float(min_delta),
                training_snr_dbs=selected_snr_dbs,
                seed=int(seed),
                asset_root=asset_root,
            )

    else:
        input_dim = int(block_len)

        group_data = None

        def context_for(selected_snr_dbs: Optional[list[float]]):
            return _model3_asset_context(
                cfg,
                resource_grid,
                input_dim=input_dim,
                train_batches=int(train_batches),
                train_batch_size=int(train_batch_size),
                validation_batches=int(validation_batches),
                validation_batch_size=effective_validation_batch_size,
                epochs=int(epochs),
                optimizer_batch_size=int(optimizer_batch_size),
                learning_rate=float(learning_rate),
                min_train_seconds=float(min_train_seconds),
                patience_epochs=int(patience_epochs),
                min_delta=float(min_delta),
                training_snr_dbs=selected_snr_dbs,
                max_examples_per_batch=max_examples_per_batch,
                seed=int(seed),
                asset_root=asset_root,
            )

    if snr_specific_weights:
        contexts = [context_for([float(snr_db)]) for snr_db in snr_dbs]
        snr_tensor = torch.as_tensor(
            snr_dbs,
            dtype=torch.float32,
            device=resource_grid.device,
        )
        noise_variances = ebnodb2no(
            snr_tensor,
            cfg.num_bits_per_symbol,
            cfg.coderate,
            resource_grid,
            precision=cfg.precision,
            device=cfg.device,
        ).detach().cpu()
        return contexts, noise_variances, group_data

    return [context_for(training_snr_dbs)], None, group_data
