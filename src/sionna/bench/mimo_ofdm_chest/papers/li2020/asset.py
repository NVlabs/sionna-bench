# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dataset generation, training, and model assets for Li et al. 2020."""

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
from sionna.phy.object import Object
from sionna.phy.channel import GenerateOFDMChannel, gen_single_sector_topology
from sionna.phy.ofdm import ResourceGrid
from sionna.phy.ofdm.channel_estimation import _PilotGroupMixin
from sionna.phy.utils import ebnodb2no

from sionna.bench.mimo_ofdm_chest.configs import SystemConfig
from sionna.bench.mimo_ofdm_chest.benchmarking import UMiChannelSource, set_random_seed
from sionna.bench.mimo_ofdm_chest.asset_store import (
    build_asset_manifest,
    external_asset_dir,
    load_asset_manifest,
    paper_asset_dir,
)
from sionna.bench.mimo_ofdm_chest.persistence import (
    sha256_fingerprint,
    sha256_file,
    write_json_atomic,
    write_npz_atomic,
)
from .estimator import ReEsNet


_ASSET_TYPE = "reesnet"
_TRAINING_DATA_ASSET_TYPE = "reesnet-training-data"

__all__ = [
    "generate_training_data",
    "load_model",
    "load_training_data",
    "train_model",
]


# Public asset lifecycle


def load_model(
    resource_grid: ResourceGrid,
    cfg: SystemConfig,
    *,
    train_batches: int = 400,
    train_batch_size: int = 16,
    validation_batches: int = 16,
    validation_batch_size: Optional[int] = 16,
    epochs: int = 200,
    optimizer_batch_size: int = 256,
    learning_rate: float = 0.000186,
    patience_epochs: int = 20,
    relative_min_delta: float = 1e-4,
    snr_specific_weights: bool = True,
    training_snr_dbs: Optional[list[float]] = None,
    paper_style_training: bool = True,
    seed: int = 42,
    asset_root: Optional[Union[str, Path]] = None,
) -> tuple[
    list[dict[str, torch.Tensor]],
    Optional[torch.Tensor],
    list[Path],
]:
    r"""Strictly load the ordered ReEsNet states required for inference.

    This function never samples channels, trains a network, or writes an
    asset. With no ``asset_root``, it selects optional package-local assets.

    :returns: ``(model_states, model_noise_variances, manifest_paths)``.
        Manifest paths follow exactly the same order as per-SNR model states.
    """
    contexts, noise_variances = _model_contexts(
        resource_grid,
        cfg,
        train_batches=train_batches,
        train_batch_size=train_batch_size,
        validation_batches=validation_batches,
        validation_batch_size=validation_batch_size,
        epochs=epochs,
        optimizer_batch_size=optimizer_batch_size,
        learning_rate=learning_rate,
        patience_epochs=patience_epochs,
        relative_min_delta=relative_min_delta,
        snr_specific_weights=snr_specific_weights,
        training_snr_dbs=training_snr_dbs,
        paper_style_training=paper_style_training,
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


def train_model(
    resource_grid: ResourceGrid,
    cfg: SystemConfig,
    *,
    train_batches: int = 400,
    train_batch_size: int = 16,
    validation_batches: int = 16,
    validation_batch_size: Optional[int] = 16,
    epochs: int = 200,
    optimizer_batch_size: int = 256,
    learning_rate: float = 0.000186,
    patience_epochs: int = 20,
    relative_min_delta: float = 1e-4,
    snr_specific_weights: bool = True,
    training_snr_dbs: Optional[list[float]] = None,
    paper_style_training: bool = True,
    seed: int = 42,
    training_data_manifest: Union[str, Path],
    asset_root: Union[str, Path],
    overwrite: bool = False,
) -> tuple[
    list[dict[str, torch.Tensor]],
    Optional[torch.Tensor],
    list[Path],
]:
    r"""Train and persist an ordered bank of ReEsNet models.

    The supplied training-data asset contains fixed clean pilot-channel grids,
    pilot powers, and full-grid channel targets. For every trained SNR-specific
    model, deterministic observation noise with the corresponding variance is
    generated afresh in each training epoch. Training minimizes the paper's L1
    objective using Adam and retains the state with the lowest validation L1 loss.

    Every completed model asset is persisted atomically and strictly reloaded
    before being returned.

    .. rubric:: Deviations from the Paper

    - The paper trains one ReEsNet across SNRs. This implementation trains one
      independent model per requested SNR.

    - The paper uses 25,000 channel realizations with a 60/20/20
      training/validation/test split, 100 epochs, minibatches of 128, and Adam with
      learning rate ``1e-3``. This implementation uses separate deterministic
      training and validation channel banks, validation-based early stopping, and
      configurable training budgets and optimizer parameters.

    - The paper samples UE velocity uniformly between 0 and 50 km/h. With
      ``paper_style_training=False``, this implementation instead uses the velocity
      specified by ``cfg`` so that training and evaluation can use the same channel
      distribution.

    - The fixed clean channel bank is reused across SNR-specific models and epochs.
      Fresh deterministic observation noise is generated for every SNR and training
      epoch. The paper describes generating multiple noisy SNR versions of each
      channel sample but does not specify this epoch-wise resampling procedure.

    - The paper considers a single-antenna downlink. This implementation derives one
      training example for every transmit-stream and receive-antenna channel and
      shares the trained ReEsNet weights across all such examples.

    :param resource_grid: Sionna resource grid defining the ReEsNet pilot-grid
        input dimensions and full-grid output dimensions.
    :param cfg: System and channel configuration defining the channel distribution,
        SNR grid, numerical precision, and training device.
    :param train_batches: Number of independent training-channel batches represented
        by the supplied training-data asset.
    :param train_batch_size: Number of complete system-channel realizations in each
        training batch. Their product is the number of independently sampled
        training channels.
    :param validation_batches: Number of independent validation-channel batches
        represented by the supplied training-data asset.
    :param validation_batch_size: Number of complete system-channel realizations in
        each validation batch. If `None`, use ``train_batch_size``.
    :param epochs: Maximum number of complete training epochs.
    :param optimizer_batch_size: Number of derived ReEsNet examples in each Adam
        update. Each system-channel realization contributes one example per
        transmit-stream and receive-antenna pair.
    :param learning_rate: Adam learning rate.
    :param patience_epochs: Stop after this many epochs without the required
        validation-loss improvement. Zero disables early stopping.
    :param relative_min_delta: Minimum relative reduction in validation L1 loss
        counted as an improvement.
    :param snr_specific_weights: If `True`, train one independent model for every
        requested SNR. The current fixed-data training path requires this to be
        `True`.
    :param training_snr_dbs: SNR values for which models are trained. If `None`,
        use the SNR grid defined by ``cfg``.
    :param paper_style_training: Identifies the velocity distribution used to
        generate the training-data asset. If `True`, expect UE velocities sampled
        uniformly between 0 and 50 km/h. If `False`, expect the fixed UE velocity
        defined by ``cfg``.
    :param seed: Base seed controlling model initialization, training-example
        ordering, and deterministic observation-noise generation.
    :param training_data_manifest: Manifest of the compatible fixed ReEsNet
        training-data asset.
    :param asset_root: Writable external root beneath which trained model assets
        are stored.
    :param overwrite: If `True`, replace model assets with the same identity.
        If `False`, raise an error when such an asset already exists.

    :returns: A tuple ``(model_states, model_noise_variances, manifest_paths)``.
        ``model_states`` contains the strictly reloaded ReEsNet state dictionaries.
        ``model_noise_variances`` contains the ordered inference noise variances for
        the SNR-specific model bank. ``manifest_paths`` contains the corresponding
        validated model-manifest paths.
    """
    contexts, noise_variances = _model_contexts(
        resource_grid,
        cfg,
        train_batches=train_batches,
        train_batch_size=train_batch_size,
        validation_batches=validation_batches,
        validation_batch_size=validation_batch_size,
        epochs=epochs,
        optimizer_batch_size=optimizer_batch_size,
        learning_rate=learning_rate,
        patience_epochs=patience_epochs,
        relative_min_delta=relative_min_delta,
        snr_specific_weights=snr_specific_weights,
        training_snr_dbs=training_snr_dbs,
        paper_style_training=paper_style_training,
        seed=seed,
        asset_root=asset_root,
    )
    training_data, validated_training_data_manifest = load_training_data(
        resource_grid,
        cfg,
        train_batches=train_batches,
        train_batch_size=train_batch_size,
        validation_batches=validation_batches,
        validation_batch_size=(
            train_batch_size
            if validation_batch_size is None
            else validation_batch_size
        ),
        training_snr_dbs=training_snr_dbs,
        paper_style_training=paper_style_training,
        seed=seed,
        manifest_path=training_data_manifest,
    )
    training_data = _materialize_training_data_cached(
        str(validated_training_data_manifest),
        validated_training_data_manifest.stat().st_mtime_ns,
        str(torch.device(cfg.device)),
    )
    states = []
    manifest_paths = []
    for context in contexts:
        state, manifest_path = _train_model_asset(
            resource_grid,
            cfg,
            context,
            train_batches=train_batches,
            train_batch_size=train_batch_size,
            validation_batches=validation_batches,
            validation_batch_size=context.validation_batch_size,
            epochs=epochs,
            optimizer_batch_size=optimizer_batch_size,
            learning_rate=learning_rate,
            patience_epochs=patience_epochs,
            relative_min_delta=relative_min_delta,
            paper_style_training=paper_style_training,
            seed=seed,
            training_data=training_data,
            training_data_manifest=validated_training_data_manifest,
            overwrite=overwrite,
        )
        states.append(state)
        manifest_paths.append(manifest_path)
    return states, noise_variances, manifest_paths


def generate_training_data(
    resource_grid: ResourceGrid,
    cfg: SystemConfig,
    *,
    train_batches: int = 400,
    train_batch_size: int = 16,
    validation_batches: int = 16,
    validation_batch_size: int = 16,
    training_snr_dbs: Optional[list[float]] = None,
    paper_style_training: bool = True,
    seed: int = 42,
    asset_root: Union[str, Path],
    overwrite: bool = False,
) -> Path:
    r"""Generate and persist a deterministic ReEsNet training-data bank.

    The asset contains clean channel values at the active pilot positions, their
    pilot powers, and the corresponding full frequency-time channel targets. The
    same propagated channel realizations are reused across the requested SNR
    points. During model training, fresh deterministic SNR-specific noise is added
    to the clean pilot observations in every epoch.

    Each complete system-channel realization produces one ReEsNet example for
    every transmit stream and receive antenna. The numbers of derived neural-network
    examples are therefore generally larger than the numbers of independently
    sampled channels.
    The completed asset is strictly validated before its manifest path is returned.

    :param resource_grid: Sionna resource grid whose pilot geometry, pilot values,
        and full-grid dimensions define the ReEsNet inputs and targets.
    :param cfg: System and channel configuration used to generate the channel
        realizations.
    :param train_batches: Number of independent training-channel batches.
    :param train_batch_size: Number of complete system-channel realizations in
        each training batch. The total number of independently sampled training
        channels is ``train_batches * train_batch_size``.
    :param validation_batches: Number of independent validation-channel batches.
    :param validation_batch_size: Number of complete system-channel realizations
        in each validation batch. The total number of independently sampled
        validation channels is
        ``validation_batches * validation_batch_size``.
    :param training_snr_dbs: SNR points for which the data bank is prepared. The
        clean pilot grids and full-grid targets are shared across these SNR
        points. If `None`, use the SNR grid defined by ``cfg``.
    :param paper_style_training: If `True`, sample UE velocities uniformly between
        0 and 50 km/h as described by Li et al. If `False`, use the fixed UE
        velocity defined by ``cfg``.
    :param seed: Base random seed for deterministic channel generation. Training
        and validation use separate deterministic random streams.
    :param asset_root: Writable external root beneath which the generated payloads
        and manifest are stored.
    :param overwrite: If `True`, replace an existing asset with the same identity.
        If `False`, raise an error when that asset already exists.

    :returns: Path to the strictly validated training-data manifest.
    """
    context, group_data = _training_data_context(
        resource_grid,
        cfg,
        train_batches=train_batches,
        train_batch_size=train_batch_size,
        validation_batches=validation_batches,
        validation_batch_size=validation_batch_size,
        training_snr_dbs=training_snr_dbs,
        paper_style_training=paper_style_training,
        seed=seed,
        asset_root=asset_root,
    )
    if context.asset_dir.exists() and not overwrite:
        raise FileExistsError(
            f"ReEsNet training data already exist at {context.asset_dir}."
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
                group_data,
                channel_batches=train_batches,
                channel_batch_size=train_batch_size,
                training_snr_dbs=context.training_snr_dbs,
                paper_style_training=paper_style_training,
                seed=seed,
            )
        )
        payload_paths.extend(
            _write_training_split(
                temporary_dir,
                "validation",
                resource_grid,
                cfg,
                group_data,
                channel_batches=validation_batches,
                channel_batch_size=validation_batch_size,
                training_snr_dbs=context.training_snr_dbs,
                paper_style_training=paper_style_training,
                seed=seed + 100003,
            )
        )
        manifest = build_asset_manifest(
            cfg=cfg,
            asset_type=_TRAINING_DATA_ASSET_TYPE,
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

    _, manifest_path = load_training_data(
        resource_grid,
        cfg,
        train_batches=train_batches,
        train_batch_size=train_batch_size,
        validation_batches=validation_batches,
        validation_batch_size=validation_batch_size,
        training_snr_dbs=training_snr_dbs,
        paper_style_training=paper_style_training,
        seed=seed,
        manifest_path=context.manifest_path,
    )
    return manifest_path


def load_training_data(
    resource_grid: ResourceGrid,
    cfg: SystemConfig,
    *,
    train_batches: int,
    train_batch_size: int,
    validation_batches: int,
    validation_batch_size: int,
    training_snr_dbs: Optional[list[float]] = None,
    paper_style_training: bool = True,
    seed: int = 42,
    manifest_path: Union[str, Path],
) -> tuple["_TrainingDataBank", Path]:
    """Strictly validate and memory-map one fixed ReEsNet data bank."""
    manifest_path = Path(manifest_path)
    envelope = load_asset_manifest(
        manifest_path,
        expected_asset_type=_TRAINING_DATA_ASSET_TYPE,
        expected_scenario_id=cfg.scenario_id,
    )
    bank_snr_dbs = envelope["generation_recipe"].get("training_snr_dbs")
    if not isinstance(bank_snr_dbs, list):
        raise ValueError("ReEsNet training data have invalid SNR metadata.")
    requested_snr_dbs = _training_snr_list(cfg, training_snr_dbs)
    if not set(requested_snr_dbs).issubset(set(bank_snr_dbs)):
        raise ValueError("ReEsNet training data do not contain all requested SNRs.")
    context, _ = _training_data_context(
        resource_grid,
        cfg,
        train_batches=train_batches,
        train_batch_size=train_batch_size,
        validation_batches=validation_batches,
        validation_batch_size=validation_batch_size,
        training_snr_dbs=bank_snr_dbs,
        paper_style_training=paper_style_training,
        seed=seed,
        asset_root=Path(manifest_path).parents[4],
    )
    manifest, tensors = _load_training_data_cached(
        str(manifest_path),
        manifest_path.stat().st_mtime_ns,
    )
    if manifest["asset_type"] != _TRAINING_DATA_ASSET_TYPE:
        raise ValueError("ReEsNet training data have an incompatible asset type.")
    if manifest["scenario_id"] != cfg.scenario_id:
        raise ValueError("ReEsNet training data have an incompatible scenario.")
    if manifest["artifact_id"] != context.asset_dir.name:
        raise ValueError("ReEsNet training data have an incompatible identity.")
    if sha256_fingerprint(manifest["compatibility"]) != sha256_fingerprint(
        context.compatibility
    ):
        raise ValueError("ReEsNet training-data compatibility does not match.")
    if manifest["generation_recipe"] != context.generation_recipe:
        raise ValueError("ReEsNet training-data generation recipe does not match.")
    if manifest.get("metadata") != context.metadata:
        raise ValueError("ReEsNet training-data metadata do not match.")
    _validate_training_data_files(manifest, context)

    return (
        _TrainingDataBank(
            train_pilots=tensors["train-pilots.npy"],
            train_targets=tensors["train-targets.npy"],
            train_pilot_powers=tensors["train-pilot-powers.npy"],
            validation_pilots=tensors["validation-pilots.npy"],
            validation_targets=tensors["validation-targets.npy"],
            validation_pilot_powers=tensors[
                "validation-pilot-powers.npy"
            ],
        ),
        manifest_path,
    )


@lru_cache(maxsize=2)
def _load_training_data_cached(
    manifest_path: str,
    manifest_mtime_ns: int,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    """Validate once and cache memory maps for one large fixed data bank."""
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


@lru_cache(maxsize=1)
def _materialize_training_data_cached(
    manifest_path: str,
    manifest_mtime_ns: int,
    device: str,
) -> "_TrainingDataBank":
    """Keep one validated fixed bank resident on its training device."""
    _, tensors = _load_training_data_cached(
        manifest_path,
        manifest_mtime_ns,
    )
    return _TrainingDataBank(
        train_pilots=tensors["train-pilots.npy"].to(device),
        train_targets=tensors["train-targets.npy"].to(device),
        train_pilot_powers=tensors["train-pilot-powers.npy"].to(device),
        validation_pilots=tensors["validation-pilots.npy"].to(device),
        validation_targets=tensors["validation-targets.npy"].to(device),
        validation_pilot_powers=tensors["validation-pilot-powers.npy"].to(
            device
        ),
    )


# Asset identity and compatibility


@dataclass(frozen=True)
class _ModelAssetContext:
    """Identity, compatibility, and paths for one ReEsNet state."""

    asset_dir: Path
    scenario_id: str
    compatibility: dict[str, Any]
    generation_recipe: dict[str, Any]
    input_shape: tuple[int, int]
    output_shape: tuple[int, int]
    validation_batch_size: int

    @property
    def manifest_path(self) -> Path:
        return self.asset_dir / "manifest.json"

    @property
    def params_path(self) -> Path:
        return self.asset_dir / "reesnet_params.npz"


@dataclass(frozen=True)
class _TrainingDataContext:
    """Identity, schema, and paths for one fixed ReEsNet data bank."""

    asset_dir: Path
    compatibility: dict[str, Any]
    generation_recipe: dict[str, Any]
    metadata: dict[str, Any]
    training_snr_dbs: tuple[float, ...]
    train_examples: int
    validation_examples: int
    input_shape: tuple[int, int]
    output_shape: tuple[int, int]

    @property
    def manifest_path(self) -> Path:
        return self.asset_dir / "manifest.json"


@dataclass(frozen=True)
class _TrainingDataBank:
    """Memory-mapped clean ReEsNet pilots, powers, and channel targets."""

    train_pilots: torch.Tensor
    train_targets: torch.Tensor
    train_pilot_powers: torch.Tensor
    validation_pilots: torch.Tensor
    validation_targets: torch.Tensor
    validation_pilot_powers: torch.Tensor


class _PilotGroupLayout(_PilotGroupMixin, Object):
    """Expose ordered pilot images required by offline ReEsNet training."""

    def __init__(self, resource_grid: ResourceGrid) -> None:
        super().__init__(
            precision=resource_grid.precision,
            device=resource_grid.device,
        )
        self._init_pilot_groups(
            pilot_pattern=resource_grid.pilot_pattern,
            group_over=("subcarrier", "symbol"),
            persistent=False,
        )
        group_subcarriers = self._pilot_group_varying_coord("subcarrier")
        group_symbols = self._pilot_group_varying_coord("symbol")
        self._group_tx_indices = self._pilot_group_constant_coord("tx")
        self._group_stream_indices = self._pilot_group_constant_coord("stream")

        unique_subcarriers = torch.unique(group_subcarriers[0], sorted=True)
        unique_symbols = torch.unique(group_symbols[0], sorted=True)
        self._num_pilot_subcarriers = int(unique_subcarriers.numel())
        self._num_pilot_ofdm_symbols = int(unique_symbols.numel())
        assert (
            self._num_pilot_subcarriers * self._num_pilot_ofdm_symbols
            == self._pilot_group_len
        ), "Active pilots must form a rectangular ReEsNet input image."

        self._group_subcarrier_image = group_subcarriers.reshape(
            self._pilot_group_num_groups,
            self._num_pilot_subcarriers,
            self._num_pilot_ofdm_symbols,
        )
        self._group_symbol_image = group_symbols.reshape(
            self._pilot_group_num_groups,
            self._num_pilot_subcarriers,
            self._num_pilot_ofdm_symbols,
        )
        assert bool(
            torch.all(
                self._group_subcarrier_image
                == self._group_subcarrier_image[:, :, :1]
            )
        ), "ReEsNet expects pilot-image rows to have fixed subcarriers."
        assert bool(
            torch.all(
                self._group_symbol_image == self._group_symbol_image[:, :1, :]
            )
        ), "ReEsNet expects pilot-image columns to have fixed OFDM symbols."
        self._pilot_values = self._gather_pilot_groups(
            resource_grid.pilot_pattern.pilots.to(self.device)
        ).detach().cpu()

    @property
    def input_shape(self) -> tuple[int, int]:
        """Return the low-resolution pilot-image shape."""
        return self._num_pilot_subcarriers, self._num_pilot_ofdm_symbols

    def training_group_data(self, resource_grid: ResourceGrid) -> dict[str, Any]:
        """Return the exact ordered pilot data consumed during training."""
        output_shape = (
            int(resource_grid.num_effective_subcarriers),
            int(resource_grid.num_ofdm_symbols),
        )
        layout = {
            "num_groups": int(self._pilot_group_num_groups),
            "group_len": int(self._pilot_group_len),
            "input_shape": list(self.input_shape),
            "output_shape": list(output_shape),
            "tx": [
                int(value)
                for value in self._group_tx_indices.detach().cpu().tolist()
            ],
            "stream": [
                int(value)
                for value in self._group_stream_indices.detach().cpu().tolist()
            ],
            "pilot_symbols": [
                int(value)
                for value in torch.unique(
                    self._group_symbol_image[0],
                    sorted=True,
                ).detach().cpu().tolist()
            ],
            "first_subcarriers": [
                int(self._group_subcarrier_image[index, 0, 0].item())
                for index in range(self._pilot_group_num_groups)
            ],
        }
        return {
            "layout": layout,
            "tx": self._group_tx_indices.detach().cpu(),
            "stream": self._group_stream_indices.detach().cpu(),
            "subcarrier_image": self._group_subcarrier_image.detach().cpu(),
            "symbol_image": self._group_symbol_image.detach().cpu(),
            "pilot_values": self._pilot_values,
        }


def _training_snr_list(
    cfg: SystemConfig,
    training_snr_dbs: list[float] | None,
) -> list[float]:
    """Return explicit training SNRs."""
    if training_snr_dbs is None:
        return [float(x) for x in cfg.snr_db_range()]
    if not training_snr_dbs:
        raise ValueError("`training_snr_dbs` must not be empty.")
    return [float(x) for x in training_snr_dbs]


def _float_token(value: float) -> str:
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def _tensor_identity(tensor: torch.Tensor) -> dict[str, Any]:
    """Describe a compatibility tensor without expanding it into JSON."""
    tensor = tensor.detach().cpu()
    values = torch.view_as_real(tensor) if tensor.is_complex() else tensor
    identity = {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "canonical_sha256": sha256_fingerprint(values.tolist()),
    }
    if tensor.is_complex():
        identity["encoding"] = "[real, imaginary]"
    return identity


def _snr_suffix(training_snr_dbs: list[float] | None, cfg) -> str:
    snrs = _training_snr_list(cfg, training_snr_dbs)
    return "snr" + "-".join(_float_token(x) for x in snrs)


def _artifact_id(
    cfg: SystemConfig,
    *,
    input_shape: tuple[int, int],
    output_shape: tuple[int, int],
    train_batches: int,
    train_batch_size: int,
    validation_batches: int,
    validation_batch_size: int,
    epochs: int,
    optimizer_batch_size: int,
    learning_rate: float,
    patience_epochs: int,
    relative_min_delta: float,
    training_snr_dbs: list[float] | None,
    paper_style_training: bool,
    seed: int,
) -> str:
    """Return the human-readable identity of one trained state."""
    speed_mode = "paper-speed" if paper_style_training else "cfg-speed"
    selected_snr_dbs = _training_snr_list(cfg, training_snr_dbs)
    weight_mode = "per-snr" if len(selected_snr_dbs) == 1 else "mixed-snr"
    name = (
        f"in{input_shape[0]}x{input_shape[1]}"
        f"_out{output_shape[0]}x{output_shape[1]}"
        f"_tb{train_batches}_bs{train_batch_size}"
        f"_vb{validation_batches}_vbs{validation_batch_size}"
        f"_ep{epochs}_mb{optimizer_batch_size}_lr{learning_rate:g}"
        f"_pat{patience_epochs}_rmd{relative_min_delta:g}"
        f"_{_snr_suffix(training_snr_dbs, cfg)}"
        f"_{weight_mode}"
        f"_bodyskip"
        f"_{speed_mode}_seed{seed}"
    )
    return name.replace("_", "-").replace(".", "p")


def _asset_compatibility(
    group_data: dict[str, Any],
    *,
    input_shape: tuple[int, int],
    output_shape: tuple[int, int],
) -> dict[str, Any]:
    """Return the exact geometry and architecture required by ReEsNet."""
    pilot_values = group_data["pilot_values"]
    # Pilot normalization has differed by about 1e-6 across Sionna resource-
    # grid construction paths. Hash rounded powers so equivalent pilot energy
    # remains compatible while meaningful power changes are still rejected.
    pilot_power = pilot_values.abs().square().to(torch.float32).round(decimals=5)
    subcarrier_image = group_data["subcarrier_image"]
    symbol_image = group_data["symbol_image"]
    return {
        "architecture": {
            "input_channels": 2,
            "output_channels": 2,
            "num_filters": 16,
            "num_res_blocks": 4,
            "kernel_size": 3,
            "body_skip": True,
            "upsampling": "conv_transpose_kernel_equals_stride",
        },
        "input_shape": [int(x) for x in input_shape],
        "output_shape": [int(x) for x in output_shape],
        "layout": group_data["layout"],
        "pilot_values": _tensor_identity(pilot_values),
        "pilot_powers": _tensor_identity(pilot_power),
        "subcarrier_image": _tensor_identity(subcarrier_image),
        "symbol_image": _tensor_identity(symbol_image),
    }


def _training_data_context(
    resource_grid: ResourceGrid,
    cfg: SystemConfig,
    *,
    train_batches: int,
    train_batch_size: int,
    validation_batches: int,
    validation_batch_size: int,
    training_snr_dbs: Optional[list[float]],
    paper_style_training: bool,
    seed: int,
    asset_root: Union[str, Path],
) -> tuple[_TrainingDataContext, dict[str, Any]]:
    """Resolve one external ReEsNet data-bank identity and contract."""
    counts = (
        int(train_batches),
        int(train_batch_size),
        int(validation_batches),
        int(validation_batch_size),
    )
    if any(value < 1 for value in counts):
        raise ValueError("ReEsNet training-data batch counts must be positive.")
    snrs = tuple(_training_snr_list(cfg, training_snr_dbs))
    if len(set(snrs)) != len(snrs):
        raise ValueError("ReEsNet training-data SNRs must be unique.")

    layout = _PilotGroupLayout(resource_grid)
    group_data = layout.training_group_data(resource_grid)
    input_shape = layout.input_shape
    output_shape = (
        int(resource_grid.num_effective_subcarriers),
        int(resource_grid.num_ofdm_symbols),
    )
    examples_per_channel = int(group_data["layout"]["num_groups"]) * int(
        cfg.num_rx_ant
    )
    train_channels = counts[0] * counts[1]
    validation_channels = counts[2] * counts[3]
    train_examples = train_channels * examples_per_channel
    validation_examples = validation_channels * examples_per_channel
    speed_mode = "paper-speed" if paper_style_training else "cfg-speed"
    artifact_id = (
        f"train{train_channels}-val{validation_channels}"
        f"-{_snr_suffix(list(snrs), cfg)}-{speed_mode}-seed{seed}"
    ).replace("_", "-").replace(".", "p")
    snapshot = cfg.config_snapshot()
    system = snapshot["system"]
    compatibility = {
        "channel_source": snapshot["channel_source"],
        "ofdm": system["ofdm"],
        "multi_user": system["multi_user"],
        "arrays": system["arrays"],
        "reesnet": _asset_compatibility(
            group_data,
            input_shape=input_shape,
            output_shape=output_shape,
        ),
    }
    generation_recipe = {
        "train_batches": counts[0],
        "train_batch_size": counts[1],
        "validation_batches": counts[2],
        "validation_batch_size": counts[3],
        "training_snr_dbs": list(snrs),
        "training_speed_mode": (
            "paper_uniform_0_50_kmh" if paper_style_training else "config_fixed"
        ),
        "config_ue_speed_mps": float(cfg.ue_speed),
        "seed": int(seed),
        "validation_seed": int(seed) + 100003,
        "storage_dtype": "float16",
    }
    metadata = {
        "train_independent_channel_count": train_channels,
        "validation_independent_channel_count": validation_channels,
        "derived_examples_per_channel": examples_per_channel,
        "train_derived_example_count": train_examples,
        "validation_derived_example_count": validation_examples,
        "snr_reuse": "clean-pilots-and-targets-shared-across-snrs",
        "noise_resampling": "deterministic-per-snr-epoch",
    }
    return (
        _TrainingDataContext(
            asset_dir=external_asset_dir(
                cfg,
                "li2020",
                _TRAINING_DATA_ASSET_TYPE,
                artifact_id,
                asset_root=asset_root,
            ),
            compatibility=compatibility,
            generation_recipe=generation_recipe,
            metadata=metadata,
            training_snr_dbs=snrs,
            train_examples=train_examples,
            validation_examples=validation_examples,
            input_shape=input_shape,
            output_shape=output_shape,
        ),
        group_data,
    )


def _validate_training_data_files(
    manifest: dict[str, Any],
    context: _TrainingDataContext,
) -> None:
    """Validate the exact file order, shapes, and dtypes of one data bank."""
    expected: list[tuple[str, list[int]]] = []
    for split, num_examples in (
        ("train", context.train_examples),
        ("validation", context.validation_examples),
    ):
        expected.extend((
            (
                f"{split}-pilots.npy",
                [num_examples, 2, *context.input_shape],
            ),
            (
                f"{split}-targets.npy",
                [num_examples, 2, *context.output_shape],
            ),
            (
                f"{split}-pilot-powers.npy",
                [num_examples, 1, *context.input_shape],
            ),
        ))
    records = manifest["files"]
    if [record["path"] for record in records] != [name for name, _ in expected]:
        raise ValueError("ReEsNet training data have unexpected payload files.")
    for record, (_, shape) in zip(records, expected):
        if record.get("format") != "npy" or record.get("array") != {
            "shape": shape,
            "dtype": "float16",
        }:
            raise ValueError("ReEsNet training data have incompatible arrays.")


def _generation_recipe(
    cfg,
    *,
    train_batches: int,
    train_batch_size: int,
    validation_batches: int,
    validation_batch_size: int,
    epochs: int,
    optimizer_batch_size: int,
    learning_rate: float,
    patience_epochs: int,
    relative_min_delta: float,
    training_snr_dbs: list[float] | None,
    paper_style_training: bool,
    seed: int,
) -> dict[str, Any]:
    """Return the complete recipe that produced one ReEsNet checkpoint."""
    return {
        "optimizer": "adam",
        "loss": "l1",
        "learning_rate": float(learning_rate),
        "optimizer_batch_size": int(optimizer_batch_size),
        "train_batches": int(train_batches),
        "train_batch_size": int(train_batch_size),
        "validation_batches": int(validation_batches),
        "validation_batch_size": int(validation_batch_size),
        "epochs": int(epochs),
        "patience_epochs": int(patience_epochs),
        "relative_min_delta": float(relative_min_delta),
        "training_snr_dbs": _training_snr_list(cfg, training_snr_dbs),
        "training_data": "fixed-clean-channel-bank",
        "noise_resampling": "deterministic-per-snr-epoch",
        "training_speed_mode": (
            "paper_uniform_0_50_kmh" if paper_style_training else "config_fixed"
        ),
        "config_ue_speed_mps": float(cfg.ue_speed),
        "seed": int(seed),
        "validation_seed": int(seed) + 100003,
        "dataset_storage_dtype": "float16",
        "validation_storage_device": "cpu",
    }


# Payload persistence and validation


def _save_state_npz(path: str, state: dict[str, torch.Tensor]) -> list[str]:
    keys = list(state.keys())
    arrays = {
        f"arr_{i}": value.detach().cpu().numpy()
        for i, value in enumerate(state.values())
    }
    write_npz_atomic(path, **arrays)
    return keys


def _load_state_npz(
    path: str,
    expected_state: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Load a ReEsNet state after exact array-shape and dtype validation."""
    keys = list(expected_state)
    array_names = [f"arr_{index}" for index in range(len(keys))]
    with np.load(path, allow_pickle=False) as data:
        if data.files != array_names:
            raise ValueError("ReEsNet asset has incompatible state arrays.")
        state = {}
        for index, key in enumerate(keys):
            array = data[array_names[index]]
            expected_tensor = expected_state[key]
            if (
                array.shape != tuple(expected_tensor.shape)
                or array.dtype != np.float32
            ):
                raise ValueError(
                    "ReEsNet asset has incompatible state shapes or dtypes."
                )
            state[key] = torch.from_numpy(array)
    return state


def _load_model_asset(
    context: _ModelAssetContext,
) -> tuple[dict[str, torch.Tensor], Path]:
    """Strictly load and validate one prepared ReEsNet asset."""
    if not context.manifest_path.is_file():
        if context.asset_dir.exists() and any(context.asset_dir.iterdir()):
            raise ValueError(f"Incomplete ReEsNet asset at {context.asset_dir}.")
        raise FileNotFoundError(
            f"No compatible Li ReEsNet asset at {context.asset_dir}. Generate "
            "it with `python -m sionna.bench.mimo_ofdm_chest.papers.li2020.run_benchmark "
            "--prepare-assets --asset-root <path>`, then evaluate it with "
            "`python -m sionna.bench.mimo_ofdm_chest.papers.li2020.run_benchmark "
            "--asset-root <path>`."
        )
    manifest = load_asset_manifest(
        context.manifest_path,
        asset_dir=context.asset_dir,
        expected_asset_type=_ASSET_TYPE,
        expected_scenario_id=context.scenario_id,
        expected_artifact_id=context.asset_dir.name,
        expected_compatibility=context.compatibility,
    )
    if manifest["generation_recipe"] != context.generation_recipe:
        raise ValueError("ReEsNet asset generation recipe does not match.")

    with torch.random.fork_rng(devices=[]):
        expected_state = ReEsNet(
            context.input_shape,
            context.output_shape,
        ).state_dict()
    expected_keys = list(expected_state)
    metadata = manifest.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("ReEsNet asset has invalid metadata.")
    if metadata.get("state_keys") != expected_keys:
        raise ValueError("ReEsNet asset has incompatible state keys.")
    if not isinstance(metadata.get("history"), dict):
        raise ValueError("ReEsNet asset has invalid training history.")

    expected_arrays = {
        f"arr_{index}": {
            "shape": list(expected_state[key].shape),
            "dtype": "float32",
        }
        for index, key in enumerate(expected_keys)
    }
    files = manifest["files"]
    if (
        len(files) != 1
        or files[0].get("path") != context.params_path.name
        or files[0].get("format") != "npz"
        or files[0].get("arrays") != expected_arrays
    ):
        raise ValueError("ReEsNet manifest has incompatible state arrays.")

    return _load_state_npz(context.params_path, expected_state), context.manifest_path


def _write_model_asset(
    context: _ModelAssetContext,
    state: dict[str, torch.Tensor],
    history: dict[str, Any],
    *,
    cfg,
    overwrite: bool,
) -> tuple[dict[str, torch.Tensor], Path]:
    """Atomically persist one ReEsNet asset and strictly reload it."""
    if context.asset_dir.exists() and not overwrite:
        raise FileExistsError(
            f"ReEsNet asset already exists at {context.asset_dir}."
        )
    context.asset_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{context.asset_dir.name}.",
            dir=context.asset_dir.parent,
        )
    )
    try:
        temporary_params = temporary_dir / context.params_path.name
        state_keys = _save_state_npz(temporary_params, state)
        manifest = build_asset_manifest(
            cfg=cfg,
            asset_type=_ASSET_TYPE,
            artifact_id=context.asset_dir.name,
            compatibility=context.compatibility,
            generation_recipe=context.generation_recipe,
            payload_paths=[temporary_params],
            metadata={"state_keys": state_keys, "history": history},
        )
        write_json_atomic(temporary_dir / "manifest.json", manifest)
        if context.asset_dir.exists():
            shutil.rmtree(context.asset_dir)
        os.replace(temporary_dir, context.asset_dir)
    finally:
        shutil.rmtree(temporary_dir, ignore_errors=True)
    return _load_model_asset(context)


# Training data and optimization


def _complex_to_channels(values: torch.Tensor) -> torch.Tensor:
    """Convert complex image tensor to real/imag channels."""
    return torch.stack([values.real, values.imag], dim=1).to(torch.float32)


def _write_training_split(
    directory: Path,
    split: str,
    resource_grid: ResourceGrid,
    cfg: SystemConfig,
    group_data: dict[str, Any],
    *,
    channel_batches: int,
    channel_batch_size: int,
    training_snr_dbs: tuple[float, ...],
    paper_style_training: bool,
    seed: int,
) -> list[Path]:
    """Generate clean pilot grids, targets, and pilot powers for one split."""
    set_random_seed(seed)
    channel_source = UMiChannelSource(cfg, resource_grid)
    channel_model = channel_source.channel_model
    sampler = GenerateOFDMChannel(
        channel_model,
        resource_grid,
        precision=cfg.precision,
        device=cfg.device,
    )
    input_shape = tuple(group_data["layout"]["input_shape"])
    output_shape = tuple(group_data["layout"]["output_shape"])
    num_groups = int(group_data["layout"]["num_groups"])
    num_examples = (
        int(channel_batches)
        * int(channel_batch_size)
        * num_groups
        * int(cfg.num_rx_ant)
    )
    pilot_path = directory / f"{split}-pilots.npy"
    target_path = directory / f"{split}-targets.npy"
    power_path = directory / f"{split}-pilot-powers.npy"
    pilots = np.lib.format.open_memmap(
        pilot_path,
        mode="w+",
        dtype=np.float16,
        shape=(num_examples, 2, *input_shape),
    )
    targets = np.lib.format.open_memmap(
        target_path,
        mode="w+",
        dtype=np.float16,
        shape=(num_examples, 2, *output_shape),
    )
    pilot_powers = np.lib.format.open_memmap(
        power_path,
        mode="w+",
        dtype=np.float16,
        shape=(num_examples, 1, *input_shape),
    )

    group_tx_indices = group_data["tx"]
    group_stream_indices = group_data["stream"]
    subcarrier_image = group_data["subcarrier_image"]
    symbol_image = group_data["symbol_image"]
    pilot_values = group_data["pilot_values"]
    num_pilot_subcarriers, num_pilot_symbols = input_shape
    num_effective_subcarriers, num_ofdm_symbols = output_shape
    offset = 0
    print(
        f"[li2020] generating {split} data: channels="
        f"{channel_batches * channel_batch_size} snrs={list(training_snr_dbs)}"
    )
    for batch_index in range(int(channel_batches)):
        if paper_style_training:
            min_velocity = 0.0
            max_velocity = 50.0 / 3.6
        else:
            min_velocity = cfg.ue_speed
            max_velocity = cfg.ue_speed
        topology = gen_single_sector_topology(
            batch_size=channel_batch_size,
            num_ut=cfg.num_ue,
            scenario="umi",
            min_ut_velocity=min_velocity,
            max_ut_velocity=max_velocity,
            precision=cfg.precision,
            device=cfg.device,
        )
        channel_model.set_topology(*topology)
        h = sampler(channel_batch_size)

        pilot_parts: list[torch.Tensor] = []
        power_parts: list[torch.Tensor] = []
        target_parts: list[torch.Tensor] = []
        for group_index in range(num_groups):
            tx_index = int(group_tx_indices[group_index].item())
            stream_index = int(group_stream_indices[group_index].item())
            symbols = symbol_image[group_index].reshape(-1).to(h.device)
            subcarriers = subcarrier_image[group_index].reshape(-1).to(h.device)
            values = h[:, :, :, tx_index, stream_index]
            pilot_channel = values[..., symbols, subcarriers].reshape(
                    -1,
                    num_pilot_subcarriers,
                    num_pilot_symbols,
                )
            pilot_power = (
                pilot_values[group_index]
                .abs()
                .square()
                .reshape(num_pilot_subcarriers, num_pilot_symbols)
                .to(device=h.device, dtype=h.real.dtype)
            )
            pilot_parts.append(_complex_to_channels(pilot_channel))
            power_parts.append(
                pilot_power.expand(pilot_channel.shape[0], 1, *input_shape)
            )
            target_parts.append(
                _complex_to_channels(
                    values.reshape(
                        -1,
                        num_ofdm_symbols,
                        num_effective_subcarriers,
                    ).transpose(-2, -1)
                )
            )
        pilot_batch = torch.cat(pilot_parts, dim=0)
        target_batch = torch.cat(target_parts, dim=0)
        power_batch = torch.cat(power_parts, dim=0)
        next_offset = offset + int(target_batch.shape[0])
        pilots[offset:next_offset] = pilot_batch.cpu().numpy().astype(
            np.float16,
            copy=False,
        )
        targets[offset:next_offset] = target_batch.cpu().numpy().astype(
            np.float16,
            copy=False,
        )
        pilot_powers[offset:next_offset] = power_batch.cpu().numpy().astype(
            np.float16,
            copy=False,
        )
        offset = next_offset
        if (batch_index + 1) % max(1, int(channel_batches) // 10) == 0:
            print(
                f"[li2020] {split} channel batch "
                f"{batch_index + 1}/{channel_batches}"
            )

    if offset != num_examples:
        raise RuntimeError(
            f"Expected {num_examples} ReEsNet examples, generated {offset}."
        )
    pilots.flush()
    targets.flush()
    pilot_powers.flush()
    del pilots, targets, pilot_powers
    return [pilot_path, target_path, power_path]


def _l1_in_batches(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    batch_size: int,
    *,
    pilot_powers: torch.Tensor | None = None,
    noise_variance: float | None = None,
    noise_seed: int | None = None,
) -> float:
    """Evaluate mean L1 with an optional deterministic noisy-pilot view."""
    device = next(model.parameters()).device
    if (pilot_powers is None) != (noise_variance is None):
        raise ValueError(
            "`pilot_powers` and `noise_variance` must be provided together."
        )
    noise_generator = None
    if noise_variance is not None:
        noise_generator = torch.Generator(device=device)
        noise_generator.manual_seed(int(noise_seed or 0))
    num = 0.0
    den = 0
    for start in range(0, int(x.shape[0]), int(batch_size)):
        xb = x[start : start + int(batch_size)].to(
            device=device,
            dtype=torch.float32,
        )
        if noise_variance is not None:
            powers = pilot_powers[start : start + int(batch_size)].to(
                device=device,
                dtype=torch.float32,
            )
            noise_std = torch.sqrt(float(noise_variance) / (2.0 * powers))
            xb = xb + noise_std * torch.randn(
                xb.shape,
                generator=noise_generator,
                device=device,
                dtype=xb.dtype,
            )
        yb = y[start : start + int(batch_size)].to(
            device=device,
            dtype=torch.float32,
        )
        pred = model(xb)
        num += float((pred - yb).abs().sum().item())
        den += int(yb.numel())
    return num / max(den, 1)


def _state_to_cpu(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in state.items()}


def _train_state(
    x: torch.Tensor,
    y: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    *,
    pilot_powers: torch.Tensor,
    validation_pilot_powers: torch.Tensor,
    noise_variance: float,
    input_shape: tuple[int, int],
    output_shape: tuple[int, int],
    epochs: int,
    optimizer_batch_size: int,
    learning_rate: float,
    patience_epochs: int,
    relative_min_delta: float,
    seed: int,
    device: Union[str, torch.device],
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Train ReEsNet and return the validation-selected state."""
    device = torch.device(device)
    index_device = x.device
    gen = torch.Generator(device=index_device)
    gen.manual_seed(seed)
    x_train = x
    y_train = y
    split_mode = "separate_channel_holdout"

    fork_devices = [device.index or 0] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=fork_devices):
        torch.manual_seed(seed)
        model = ReEsNet(input_shape, output_shape).to(
            device=device,
            dtype=torch.float32,
        )
    optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate))
    eval_batch_size = 1024

    best_state = _state_to_cpu(model.state_dict())
    best_epoch = 0
    best_val_loss = float("inf")
    best_train_loss = float("inf")
    trace = []
    t0 = time.time()
    print(
        f"[li2020] ReEsNet dataset split={split_mode} "
        f"train_examples={x_train.shape[0]} val_examples={x_val.shape[0]}"
    )
    for epoch in range(int(epochs)):
        model.train()
        noise_generator = torch.Generator(device=device)
        noise_generator.manual_seed(int(seed) + 1000003 * (epoch + 1))
        order = torch.randperm(
            x_train.shape[0],
            generator=gen,
            device=index_device,
        )
        train_loss_sum = 0.0
        train_value_count = 0
        for start in range(0, int(order.numel()), int(optimizer_batch_size)):
            idx = order[start : start + int(optimizer_batch_size)]
            optimizer.zero_grad(set_to_none=True)
            x_batch = x_train[idx].to(device=device, dtype=torch.float32)
            powers = pilot_powers[idx].to(
                device=device,
                dtype=torch.float32,
            )
            noise_std = torch.sqrt(float(noise_variance) / (2.0 * powers))
            x_batch = x_batch + noise_std * torch.randn(
                x_batch.shape,
                generator=noise_generator,
                device=device,
                dtype=x_batch.dtype,
            )
            y_batch = y_train[idx].to(device=device, dtype=torch.float32)
            loss = torch.nn.functional.l1_loss(model(x_batch), y_batch)
            loss.backward()
            optimizer.step()
            train_loss_sum += float(loss.item()) * int(y_batch.numel())
            train_value_count += int(y_batch.numel())

        model.eval()
        with torch.no_grad():
            train_loss = train_loss_sum / max(train_value_count, 1)
            val_loss = _l1_in_batches(
                model,
                x_val,
                y_val,
                eval_batch_size,
                pilot_powers=validation_pilot_powers,
                noise_variance=noise_variance,
                noise_seed=int(seed) + 2000003,
            )
        improved = val_loss < best_val_loss * (1.0 - float(relative_min_delta))
        if improved:
            best_epoch = int(epoch + 1)
            best_val_loss = float(val_loss)
            best_train_loss = float(train_loss)
            best_state = _state_to_cpu(model.state_dict())
        elapsed = time.time() - t0
        trace.append(
            {
                "epoch": int(epoch + 1),
                "elapsed_seconds": float(elapsed),
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "best_epoch": int(best_epoch),
                "best_val_loss": float(best_val_loss),
            }
        )
        print(
            f"[li2020] ReEsNet epoch {epoch + 1}/{epochs} "
            f"elapsed={elapsed:.1f}s train_l1={train_loss:.6e} "
            f"val_l1={val_loss:.6e} "
            f"best_val={best_val_loss:.6e}@{best_epoch}"
        )
        if (
            int(patience_epochs) > 0
            and best_epoch > 0
            and epoch + 1 - best_epoch >= int(patience_epochs)
        ):
            print(
                f"[li2020] early stop: no validation improvement > "
                f"{relative_min_delta:g} relative for {patience_epochs} epochs"
            )
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        train_loss = _l1_in_batches(
            model,
            x_train,
            y_train,
            eval_batch_size,
            pilot_powers=pilot_powers,
            noise_variance=noise_variance,
            noise_seed=int(seed) + 3000003,
        )
        val_loss = _l1_in_batches(
            model,
            x_val,
            y_val,
            eval_batch_size,
            pilot_powers=validation_pilot_powers,
            noise_variance=noise_variance,
            noise_seed=int(seed) + 2000003,
        )
    history = {
        "split_mode": split_mode,
        "training_loss": "l1",
        "train_examples": int(x_train.shape[0]),
        "validation_examples": int(x_val.shape[0]),
        "input_noise": "deterministic-fresh-per-training-epoch",
        "best_epoch": int(best_epoch),
        "best_train_loss": float(best_train_loss),
        "best_val_loss": float(best_val_loss),
        "train_loss": float(train_loss),
        "val_loss": float(val_loss),
        "elapsed_seconds": float(time.time() - t0),
        "trace": trace,
    }
    history.update(
        {
            "best_train_l1": float(best_train_loss),
            "best_val_l1": float(best_val_loss),
            "train_l1": float(train_loss),
            "val_l1": float(val_loss),
        }
    )
    return best_state, history


def _train_model_asset(
    resource_grid: ResourceGrid,
    cfg: SystemConfig,
    context: _ModelAssetContext,
    *,
    train_batches: int,
    train_batch_size: int,
    validation_batches: int,
    validation_batch_size: int,
    epochs: int,
    optimizer_batch_size: int,
    learning_rate: float,
    patience_epochs: int,
    relative_min_delta: float,
    paper_style_training: bool,
    seed: int,
    training_data: _TrainingDataBank,
    training_data_manifest: Path,
    overwrite: bool,
) -> tuple[dict[str, torch.Tensor], Path]:
    """Train, persist, and strictly reload one ReEsNet state."""
    selected_snr_dbs = context.generation_recipe["training_snr_dbs"]
    print(f"[li2020] training ReEsNet asset -> {context.asset_dir}")
    if len(selected_snr_dbs) != 1:
        raise ValueError(
            "Fixed ReEsNet data banks require SNR-specific model weights."
        )
    snr_db = float(selected_snr_dbs[0])
    noise_variance = float(
        ebnodb2no(
            torch.tensor(
                snr_db,
                dtype=torch.float32,
                device=cfg.device,
            ),
            cfg.num_bits_per_symbol,
            cfg.coderate,
            resource_grid,
            precision=cfg.precision,
            device=cfg.device,
        ).item()
    )

    state, history = _train_state(
        training_data.train_pilots,
        training_data.train_targets,
        training_data.validation_pilots,
        training_data.validation_targets,
        pilot_powers=training_data.train_pilot_powers,
        validation_pilot_powers=training_data.validation_pilot_powers,
        noise_variance=noise_variance,
        input_shape=context.input_shape,
        output_shape=context.output_shape,
        epochs=epochs,
        optimizer_batch_size=optimizer_batch_size,
        learning_rate=learning_rate,
        patience_epochs=patience_epochs,
        relative_min_delta=relative_min_delta,
        seed=seed,
        device=cfg.device,
    )
    history.update(
        {
            "train_channels": int(train_batches) * int(train_batch_size),
            "validation_channels": int(validation_batches)
            * int(validation_batch_size),
            "training_snr_dbs": selected_snr_dbs,
            "training_loss": "l1",
            "training_speed_mode": (
                "paper_uniform_0_50_kmh"
                if paper_style_training
                else "config_fixed"
            ),
            "training_data_manifest": (
                {
                    "asset_type": _TRAINING_DATA_ASSET_TYPE,
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
    train_batches: int = 16,
    train_batch_size: int = 16,
    validation_batches: int = 0,
    validation_batch_size: Optional[int] = None,
    epochs: int = 100,
    optimizer_batch_size: int = 128,
    learning_rate: float = 1e-3,
    patience_epochs: int = 0,
    relative_min_delta: float = 0.0,
    snr_specific_weights: bool = True,
    training_snr_dbs: Optional[list[float]] = None,
    paper_style_training: bool = False,
    seed: int = 42,
    asset_root: Optional[Union[str, Path]] = None,
) -> tuple[
    list[_ModelAssetContext],
    Optional[torch.Tensor],
]:
    """Resolve ordered model identities and the inference noise grid."""
    if not 0.0 <= float(relative_min_delta) < 1.0:
        raise ValueError("`relative_min_delta` must lie in [0, 1).")
    effective_validation_batch_size = (
        train_batch_size
        if validation_batch_size is None
        else int(validation_batch_size)
    )
    layout = _PilotGroupLayout(resource_grid)
    group_data = layout.training_group_data(resource_grid)
    input_shape = layout.input_shape
    output_shape = (
        int(resource_grid.num_effective_subcarriers),
        int(resource_grid.num_ofdm_symbols),
    )
    compatibility = _asset_compatibility(
        group_data,
        input_shape=input_shape,
        output_shape=output_shape,
    )

    def context_for(selected_snr_dbs: list[float]) -> _ModelAssetContext:
        generation_recipe = _generation_recipe(
            cfg,
            train_batches=train_batches,
            train_batch_size=train_batch_size,
            validation_batches=validation_batches,
            validation_batch_size=effective_validation_batch_size,
            epochs=epochs,
            optimizer_batch_size=optimizer_batch_size,
            learning_rate=learning_rate,
            patience_epochs=patience_epochs,
            relative_min_delta=relative_min_delta,
            training_snr_dbs=selected_snr_dbs,
            paper_style_training=paper_style_training,
            seed=seed,
        )
        artifact_id = _artifact_id(
            cfg,
            input_shape=input_shape,
            output_shape=output_shape,
            train_batches=train_batches,
            train_batch_size=train_batch_size,
            validation_batches=validation_batches,
            validation_batch_size=effective_validation_batch_size,
            epochs=epochs,
            optimizer_batch_size=optimizer_batch_size,
            learning_rate=learning_rate,
            patience_epochs=patience_epochs,
            relative_min_delta=relative_min_delta,
            training_snr_dbs=selected_snr_dbs,
            paper_style_training=paper_style_training,
            seed=seed,
        )
        asset_dir = (
            paper_asset_dir(cfg, _ASSET_TYPE, artifact_id, __file__)
            if asset_root is None
            else external_asset_dir(
                cfg,
                "li2020",
                _ASSET_TYPE,
                artifact_id,
                asset_root=asset_root,
            )
        )
        return _ModelAssetContext(
            asset_dir=asset_dir,
            scenario_id=cfg.scenario_id,
            compatibility=compatibility,
            generation_recipe=generation_recipe,
            input_shape=input_shape,
            output_shape=output_shape,
            validation_batch_size=effective_validation_batch_size,
        )

    snr_dbs = _training_snr_list(cfg, training_snr_dbs)
    if snr_specific_weights:
        contexts = [context_for([snr_db]) for snr_db in snr_dbs]
        snr_tensor = torch.as_tensor(
            snr_dbs,
            dtype=torch.float64 if cfg.precision == "double" else torch.float32,
            device=resource_grid.device,
        )
        model_noise_variances = ebnodb2no(
            snr_tensor,
            cfg.num_bits_per_symbol,
            cfg.coderate,
            resource_grid,
            precision=cfg.precision,
            device=cfg.device,
        ).detach().cpu()
        return contexts, model_noise_variances

    if paper_style_training and training_snr_dbs is None:
        snr_dbs = [0.0, 5.0, 10.0, 15.0, 20.0]
    return [context_for(snr_dbs)], None
