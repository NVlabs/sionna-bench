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
from sionna.phy.ofdm import ResourceGrid
from sionna.phy.ofdm.channel_estimation import _PilotGroupMixin
from sionna.phy.utils import ebnodb2no

from sionna.bench.mimo_ofdm_chest.configs import SystemConfig
from sionna.bench.mimo_ofdm_chest.persistence import (
    sha256_fingerprint,
    sha256_file,
    write_json_atomic,
    write_npz_atomic,
)
from sionna.bench.mimo_ofdm_chest.benchmarking import UMiChannelSource, set_random_seed
from sionna.bench.mimo_ofdm_chest.asset_store import (
    build_asset_manifest,
    external_asset_dir,
    load_asset_manifest,
    paper_asset_dir,
)

from .estimator import (
    VariationalAutoEncoder,
    DECODER_CHANNELS,
    DECODER_KERNEL_SIZE,
    DECODER_PADDING,
    DECODER_STRIDE,
    ENCODER_CHANNELS,
    ENCODER_KERNEL_SIZE,
    ENCODER_PADDING,
    ENCODER_STRIDE,
    COVARIANCE_DIAGONAL,
    COVARIANCE_FULL,
)

METHOD_NOISY = "noisy"
METHOD_REAL = "real"
METHOD_NAMES = [METHOD_NOISY, METHOD_REAL]
_PILOT_COORDINATE_ORDER = ["tx", "stream", "symbol", "subcarrier"]
_TRAINING_DATA_ASSET_TYPE = "vae-noisy-training-data"

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
    method: str = METHOD_NOISY,
    latent_dim: int = 16,
    train_batches: int = 400,
    train_batch_size: int = 16,
    validation_batches: int = 16,
    validation_batch_size: Optional[int] = 16,
    epochs: int = 200,
    optimizer_batch_size: int = 512,
    learning_rate: float = 2.30e-4,
    free_bits: float = 0.875,
    min_train_seconds: float = 0.0,
    patience_epochs: int = 20,
    relative_min_delta: float = 1e-4,
    snr_specific_weights: bool = True,
    training_snr_dbs: Optional[list[float]] = None,
    covariance_floor: float = 1e-5,
    covariance_mode: str = COVARIANCE_FULL,
    seed: int = 42,
    asset_root: Optional[Union[str, Path]] = None,
) -> tuple[
    list[dict[str, torch.Tensor]],
    Optional[torch.Tensor],
    list[Path],
]:
    r"""Strictly load the ordered VAE states required for inference.

    This function never samples channels, trains a network, or writes an
    asset. With no ``asset_root``, it selects optional package-local assets.

    :returns: ``(model_states, model_noise_variances, manifest_paths)``.
        Manifest paths follow exactly the same order as per-SNR model states.
    """
    contexts, noise_variances, _ = _model_contexts(
        resource_grid,
        cfg,
        method=method,
        latent_dim=latent_dim,
        train_batches=train_batches,
        train_batch_size=train_batch_size,
        validation_batches=validation_batches,
        validation_batch_size=validation_batch_size,
        epochs=epochs,
        optimizer_batch_size=optimizer_batch_size,
        learning_rate=learning_rate,
        free_bits=free_bits,
        min_train_seconds=min_train_seconds,
        patience_epochs=patience_epochs,
        relative_min_delta=relative_min_delta,
        snr_specific_weights=snr_specific_weights,
        training_snr_dbs=training_snr_dbs,
        covariance_floor=covariance_floor,
        covariance_mode=covariance_mode,
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
    method: str = METHOD_NOISY,
    latent_dim: int = 16,
    train_batches: int = 400,
    train_batch_size: int = 16,
    validation_batches: int = 16,
    validation_batch_size: Optional[int] = 16,
    epochs: int = 200,
    optimizer_batch_size: int = 512,
    learning_rate: float = 2.30e-4,
    free_bits: float = 0.875,
    min_train_seconds: float = 0.0,
    patience_epochs: int = 20,
    relative_min_delta: float = 1e-4,
    snr_specific_weights: bool = True,
    training_snr_dbs: Optional[list[float]] = None,
    covariance_floor: float = 1e-5,
    covariance_mode: str = COVARIANCE_FULL,
    seed: int = 42,
    training_data_manifest: Optional[Union[str, Path]] = None,
    asset_root: Union[str, Path],
    overwrite: bool = False,
) -> tuple[
    list[dict[str, torch.Tensor]],
    Optional[torch.Tensor],
    list[Path],
]:
    r"""Train and persist an ordered bank of VAE channel-estimation models.

    For SNR-specific training, one independent model is trained for every requested
    SNR. The supplied training-data asset provides fixed clean channel realizations;
    fresh deterministic observation noise is generated for every training epoch.
    Each completed model asset is persisted atomically and strictly reloaded before
    being returned.

    .. rubric:: Deviations from the Paper

    - The paper trains a separate VAE at each SNR until the ELBO saturates and retains
      the model with the highest ELBO, but does not specify a dataset size, latent
      dimension, free-bits threshold, optimizer batch size, stopping patience, or
      minimum improvement. This implementation uses finite deterministic training
      and validation channel banks, a maximum epoch count, and validation-loss early
      stopping with restoration of the best validation state.
    - For ``VAE-noisy``, the clean channel bank is reused across epochs while fresh
      deterministic observation noise is generated every epoch. This retains the
      paper's observation-noise resampling while making the channel sample and
      training procedure reproducible. Shared mixed-SNR training is an additional
      configurable mode; per-SNR models follow the paper.
    - The paper uses a diagonal complex Gaussian decoder covariance. The ``"full"``
      mode is an adaptation that learns a full covariance over the real-stacked
      transform-domain channel vector and uses the corresponding multivariate
      Gaussian likelihood.

    :param resource_grid: Resource grid defining the pilot geometry and pilot
        values used to construct the VAE observations.
    :param cfg: System and channel configuration defining the channel distribution,
        SNR grid, numerical precision, and device.
    :param method: VAE training method. ``"noisy"`` uses noisy encoder inputs and
        clean channel targets. ``"real"`` uses noisy inputs and noisy targets.
    :param latent_dim: Dimension of the Gaussian latent representation.
    :param train_batches: Number of independent training-channel batches.
    :param train_batch_size: Number of complete channel realizations per training
        batch. Their product is the number of training channels.
    :param validation_batches: Number of independent validation-channel batches.
    :param validation_batch_size: Number of channel realizations per validation
        batch. If `None`, use ``train_batch_size``.
    :param epochs: Maximum number of complete training epochs.
    :param optimizer_batch_size: Number of derived VAE examples in each optimizer
        update.
    :param learning_rate: Adam learning rate.
    :param free_bits: Minimum KL contribution assigned to each latent dimension.
    :param min_train_seconds: Minimum wall-clock duration before early stopping is
        permitted.
    :param patience_epochs: Stop after this many epochs without the required
        validation-loss improvement. Zero disables early stopping.
    :param relative_min_delta: Minimum relative validation-loss reduction counted
        as an improvement.
    :param snr_specific_weights: If `True`, train one model per requested SNR.
        Otherwise, train one model shared across all requested SNRs.
    :param training_snr_dbs: SNR values used for training. If `None`, use the SNR
        grid defined by ``cfg``.
    :param covariance_floor: Positive floor applied to decoder covariance outputs.
    :param covariance_mode: Decoder covariance parameterization, either
        ``"diagonal"`` or ``"full"``.
    :param seed: Base seed controlling model initialization, data ordering, and
        deterministic observation-noise generation.
    :param training_data_manifest: Manifest of the fixed training-data asset.
        This is required for ``method="noisy"``.
    :param asset_root: Writable external root beneath which model assets are stored.
    :param overwrite: If `True`, replace assets with the same identity.

    :returns: A tuple ``(model_states, model_noise_variances, manifest_paths)``.
        ``model_states`` contains the strictly reloaded state dictionaries.
        ``model_noise_variances`` contains the ordered inference noise variances for
        an SNR-specific model bank, or `None` for a shared model.
        ``manifest_paths`` contains the corresponding validated manifest paths.
    """
    contexts, noise_variances, group_data = _model_contexts(
        resource_grid,
        cfg,
        method=method,
        latent_dim=latent_dim,
        train_batches=train_batches,
        train_batch_size=train_batch_size,
        validation_batches=validation_batches,
        validation_batch_size=validation_batch_size,
        epochs=epochs,
        optimizer_batch_size=optimizer_batch_size,
        learning_rate=learning_rate,
        free_bits=free_bits,
        min_train_seconds=min_train_seconds,
        patience_epochs=patience_epochs,
        relative_min_delta=relative_min_delta,
        snr_specific_weights=snr_specific_weights,
        training_snr_dbs=training_snr_dbs,
        covariance_floor=covariance_floor,
        covariance_mode=covariance_mode,
        seed=seed,
        asset_root=asset_root,
    )
    training_data = None
    validated_training_data_manifest = None
    if method == METHOD_NOISY and training_data_manifest is None:
        raise ValueError(
            "VAE-noisy training requires a fixed `training_data_manifest`."
        )
    if training_data_manifest is not None:
        if method != METHOD_NOISY or not snr_specific_weights:
            raise ValueError(
                "Fixed Baur training banks support only SNR-specific "
                "VAE-noisy models."
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
            group_data,
            context,
            train_batches=train_batches,
            train_batch_size=train_batch_size,
            validation_batches=validation_batches,
            validation_batch_size=context.validation_batch_size,
            epochs=epochs,
            optimizer_batch_size=optimizer_batch_size,
            learning_rate=learning_rate,
            free_bits=free_bits,
            min_train_seconds=min_train_seconds,
            patience_epochs=patience_epochs,
            relative_min_delta=relative_min_delta,
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
    seed: int = 42,
    asset_root: Union[str, Path],
    overwrite: bool = False,
) -> Path:
    r"""Generate and persist a deterministic VAE-noisy training-data bank.

    The asset contains clean propagated channels. The same channel realizations
    are reused across the requested SNR points, while deterministic SNR-specific
    observation noise is generated during model training. The completed asset is
    strictly validated before its manifest path is returned.

    :param resource_grid: Sionna resource grid whose pilot geometry and values
        define the derived VAE training observations.
    :param cfg: System and channel configuration used to generate the channel
        realizations.
    :param train_batches: Number of independent training-channel batches.
    :param train_batch_size: Number of complete system-channel realizations in
        each training batch. The total number of training channels is
        ``train_batches * train_batch_size``.
    :param validation_batches: Number of independent validation-channel batches.
    :param validation_batch_size: Number of complete system-channel realizations
        in each validation batch. The total number of validation channels is
        ``validation_batches * validation_batch_size``.
    :param training_snr_dbs: SNR points for which the data bank is prepared. The
        clean channel realizations are shared across these SNR points. If `None`,
        use the SNR grid defined by ``cfg``.
    :param seed: Base random seed for deterministic channel generation. Training
        and validation use separate deterministic random streams.
    :param asset_root: Writable external root beneath which the generated payloads
        and manifest are stored.
    :param overwrite: If `True`, replace an existing asset with the same identity.
        If `False`, raise an error when that asset already exists.

    :returns: Path to the strictly validated ``manifest.json`` of the generated
        training-data asset.
    """
    context, group_data = _training_data_context(
        resource_grid,
        cfg,
        train_batches=train_batches,
        train_batch_size=train_batch_size,
        validation_batches=validation_batches,
        validation_batch_size=validation_batch_size,
        training_snr_dbs=training_snr_dbs,
        seed=seed,
        asset_root=asset_root,
    )
    if context.asset_dir.exists() and not overwrite:
        raise FileExistsError(
            f"VAE-noisy training data already exist at {context.asset_dir}."
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
                input_dim=context.input_dim,
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
                input_dim=context.input_dim,
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
    seed: int = 42,
    manifest_path: Union[str, Path],
) -> tuple["_TrainingDataBank", Path]:
    """Strictly validate and memory-map one fixed VAE-noisy data bank."""
    manifest_path = Path(manifest_path)
    envelope = load_asset_manifest(
        manifest_path,
        expected_asset_type=_TRAINING_DATA_ASSET_TYPE,
        expected_scenario_id=cfg.scenario_id,
    )
    bank_snr_dbs = envelope["generation_recipe"].get("training_snr_dbs")
    if not isinstance(bank_snr_dbs, list):
        raise ValueError("VAE-noisy training data have invalid SNR metadata.")
    requested_snr_dbs = _training_snr_list(cfg, training_snr_dbs)
    if not set(requested_snr_dbs).issubset(set(bank_snr_dbs)):
        raise ValueError(
            "VAE-noisy training data do not contain all requested SNRs."
        )
    context, _ = _training_data_context(
        resource_grid,
        cfg,
        train_batches=train_batches,
        train_batch_size=train_batch_size,
        validation_batches=validation_batches,
        validation_batch_size=validation_batch_size,
        training_snr_dbs=bank_snr_dbs,
        seed=seed,
        asset_root=manifest_path.parents[4],
    )
    manifest, tensors = _load_training_data_cached(
        str(manifest_path),
        manifest_path.stat().st_mtime_ns,
    )
    if manifest["artifact_id"] != context.asset_dir.name:
        raise ValueError("VAE-noisy training data have an incompatible identity.")
    if sha256_fingerprint(manifest["compatibility"]) != sha256_fingerprint(
        context.compatibility
    ):
        raise ValueError("VAE-noisy training-data compatibility does not match.")
    if manifest["generation_recipe"] != context.generation_recipe:
        raise ValueError(
            "VAE-noisy training-data generation recipe does not match."
        )
    if manifest.get("metadata") != context.metadata:
        raise ValueError("VAE-noisy training-data metadata do not match.")
    _validate_training_data_files(manifest, context)
    return (
        _TrainingDataBank(
            train_channels=tensors["train-channels.npy"],
            train_pilot_powers=tensors["train-pilot-powers.npy"],
            validation_channels=tensors["validation-channels.npy"],
            validation_pilot_powers=tensors[
                "validation-pilot-powers.npy"
            ],
        ),
        manifest_path,
    )


# Asset identity and compatibility


@dataclass(frozen=True)
class _ModelAssetContext:
    """Identity, compatibility, and paths for one VAE state."""

    method: str
    asset_type: str
    asset_dir: Path
    scenario_id: str
    compatibility: dict[str, Any]
    generation_recipe: dict[str, Any]
    input_dim: int
    latent_dim: int
    covariance_floor: float
    covariance_mode: str
    validation_batch_size: int

    @property
    def manifest_path(self) -> Path:
        return self.asset_dir / "manifest.json"

    @property
    def params_path(self) -> Path:
        return self.asset_dir / "vae_params.npz"


@dataclass(frozen=True)
class _TrainingDataContext:
    """Identity, schema, and paths for one fixed VAE-noisy data bank."""

    asset_dir: Path
    compatibility: dict[str, Any]
    generation_recipe: dict[str, Any]
    metadata: dict[str, Any]
    train_examples: int
    validation_examples: int
    input_dim: int

    @property
    def manifest_path(self) -> Path:
        return self.asset_dir / "manifest.json"


@dataclass(frozen=True)
class _TrainingDataBank:
    """Memory-mapped clean transform-domain vectors and pilot powers."""

    train_channels: torch.Tensor
    train_pilot_powers: torch.Tensor
    validation_channels: torch.Tensor
    validation_pilot_powers: torch.Tensor


class _PilotGroupLayout(_PilotGroupMixin, Object):
    """Expose the ordered pilot observations used for VAE training."""

    def __init__(self, resource_grid: ResourceGrid) -> None:
        super().__init__(
            precision=resource_grid.precision,
            device=resource_grid.device,
        )
        self._init_pilot_groups(
            pilot_pattern=resource_grid.pilot_pattern,
            group_over=(),
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
        """Return ordered pilot coordinates and values on the CPU."""
        return {
            "coordinates": self._pilot_group_coords.detach().cpu(),
            "tx": self._pilot_group_constant_coord("tx").detach().cpu(),
            "stream": self._pilot_group_constant_coord("stream").detach().cpu(),
            "symbol": self._pilot_group_constant_coord("symbol").detach().cpu(),
            "subcarrier": self._pilot_group_constant_coord(
                "subcarrier"
            ).detach().cpu(),
            "pilot_values": self._training_pilot_values.detach().cpu(),
        }


def _tensor_identity(values: torch.Tensor) -> dict[str, Any]:
    """Describe a compatibility tensor without expanding it into JSON."""
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


def _pilot_group_identity(
    group_data: dict[str, torch.Tensor],
) -> dict[str, Any]:
    """Return the exact ordered pilot layout required by a VAE asset."""
    pilot_values = group_data["pilot_values"]
    return {
        "coordinate_order": _PILOT_COORDINATE_ORDER,
        "coordinates": _tensor_identity(group_data["coordinates"]),
        "values": _tensor_identity(pilot_values),
        "powers": _tensor_identity(pilot_values.abs().square()),
    }


def _training_snr_list(cfg, training_snr_dbs: list[float] | None) -> list[float]:
    """Return explicit training SNR values."""
    if training_snr_dbs is None:
        return [float(x) for x in cfg.snr_db_range()]
    if not training_snr_dbs:
        raise ValueError("training_snr_dbs must not be empty")
    return [float(x) for x in training_snr_dbs]


def _float_token(value: float) -> str:
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def _training_snr_suffix(training_snr_dbs: list[float] | None) -> str:
    if training_snr_dbs is None:
        return ""
    return "_trsnr" + "-".join(_float_token(x) for x in training_snr_dbs)


def _unitary_dft_matrix(dim: int, device: torch.device | str) -> torch.Tensor:
    index = torch.arange(dim, dtype=torch.float32, device=device)
    row = index.reshape(dim, 1)
    col = index.reshape(1, dim)
    phase = -2.0 * torch.pi * row * col / float(dim)
    matrix = torch.complex(torch.cos(phase), torch.sin(phase))
    return matrix / (float(dim) ** 0.5)


def _dft(values: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    return values @ matrix.transpose(0, 1)


def _stack_complex(values: torch.Tensor) -> torch.Tensor:
    return torch.cat([values.real, values.imag], dim=-1).to(torch.float32)


def _complex_from_stack(values: torch.Tensor) -> torch.Tensor:
    real, imag = torch.chunk(values, 2, dim=-1)
    return torch.complex(real, imag)


def _artifact_id(
    *,
    method: str,
    input_dim: int,
    latent_dim: int,
    train_batches: int,
    train_batch_size: int,
    validation_batches: int,
    validation_batch_size: int,
    epochs: int,
    optimizer_batch_size: int,
    learning_rate: float,
    free_bits: float,
    min_train_seconds: float,
    patience_epochs: int,
    relative_min_delta: float,
    covariance_floor: float,
    covariance_mode: str,
    training_snr_dbs: list[float] | None,
    seed: int,
) -> str:
    """Return the human-readable identity of one trained VAE state."""
    covariance_token = (
        "" if covariance_mode == COVARIANCE_DIAGONAL else f"_cov{covariance_mode}"
    )
    name = (
        f"n{input_dim}_z{latent_dim}"
        f"_tb{train_batches}_bs{train_batch_size}"
        f"_vb{validation_batches}_vbs{validation_batch_size}"
        f"_ep{epochs}_mb{optimizer_batch_size}_lr{learning_rate:g}"
        f"_fb{free_bits:g}_mins{min_train_seconds:g}"
        f"_pat{patience_epochs}_rmd{relative_min_delta:g}"
        f"_cf{covariance_floor:g}"
        f"{covariance_token}"
        f"{_training_snr_suffix(training_snr_dbs)}"
        f"_seed{seed}"
    )
    return name.replace("_", "-").replace(".", "p")


def _asset_metadata(
    cfg: SystemConfig,
    resource_grid: ResourceGrid,
    group_data: dict[str, torch.Tensor],
    *,
    method: str,
    input_dim: int,
    latent_dim: int,
    train_batches: int,
    train_batch_size: int,
    validation_batches: int,
    validation_batch_size: int,
    epochs: int,
    optimizer_batch_size: int,
    learning_rate: float,
    free_bits: float,
    min_train_seconds: float,
    patience_epochs: int,
    relative_min_delta: float,
    covariance_floor: float,
    covariance_mode: str,
    training_snr_dbs: list[float] | None,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    group_pilot_values = group_data["pilot_values"]
    resolved_training_snr_dbs = _training_snr_list(cfg, training_snr_dbs)
    compatibility = {
        "method": method,
        "input_dim": int(input_dim),
        "receive_vector_length": int(input_dim) // 2,
        "antenna_order": "sionna_num_rx_ant_axis",
        "dft_convention": "unitary_1d_dft_over_sionna_num_rx_ant_axis",
        "encoder_channels": list(ENCODER_CHANNELS),
        "encoder_kernel_size": ENCODER_KERNEL_SIZE,
        "encoder_stride": ENCODER_STRIDE,
        "encoder_padding": ENCODER_PADDING,
        "decoder_channels": list(DECODER_CHANNELS),
        "decoder_kernel_size": DECODER_KERNEL_SIZE,
        "decoder_stride": DECODER_STRIDE,
        "decoder_padding": DECODER_PADDING,
        "decoder_output_adjustment": "crop_or_zero_pad_to_input_dim",
        "covariance_head": (
            "full_real_cholesky_from_decoder_linear_features"
            if covariance_mode == COVARIANCE_FULL
            else "diagonal_complex_softplus_transposed_convolution"
        ),
        "covariance_parameterization": (
            "cholesky_plus_diagonal_floor"
            if covariance_mode == COVARIANCE_FULL
            else "softplus_plus_floor"
        ),
        "covariance_floor": float(covariance_floor),
        "latent_dim": int(latent_dim),
        "pilot_power_convention": "ls_error_variance_is_no_over_abs_pilot_squared",
        "pilot_groups": _pilot_group_identity(group_data),
        "num_groups": int(group_pilot_values.shape[0]),
        "group_len": int(group_pilot_values.shape[1]),
        "channel_sampler": "sionna_umi_3gpp_config",
        "decoder_target": "clean" if method == METHOD_NOISY else "noisy",
        "likelihood_variance": (
            "decoder_channel_variance"
            if method == METHOD_NOISY
            else "decoder_channel_variance_plus_observation_noise"
        ),
    }
    generation_recipe = {
        "optimizer": "adam",
        "learning_rate": float(learning_rate),
        "optimizer_batch_size": int(optimizer_batch_size),
        "free_bits": float(free_bits),
        "train_batches": int(train_batches),
        "train_batch_size": int(train_batch_size),
        "validation_batches": int(validation_batches),
        "validation_batch_size": int(validation_batch_size),
        "epochs": int(epochs),
        "min_train_seconds": float(min_train_seconds),
        "patience_epochs": int(patience_epochs),
        "relative_min_delta": float(relative_min_delta),
        "seed": int(seed),
        "validation_seed": int(seed) + 100003,
        "training_snr_dbs": resolved_training_snr_dbs,
        "noise_variances": [
            float(
                ebnodb2no(
                    torch.tensor(float(x), dtype=torch.float32),
                    cfg.num_bits_per_symbol,
                    cfg.coderate,
                    resource_grid,
                    precision=cfg.precision,
                    device=cfg.device,
                ).detach().cpu().item()
            )
            for x in resolved_training_snr_dbs
        ],
        "training_data": (
            "fixed-clean-channel-bank"
            if method == METHOD_NOISY
            else "generated-noisy-observations"
        ),
        "noise_resampling": (
            "deterministic-per-snr-epoch"
            if method == METHOD_NOISY
            else "fixed-per-generated-training-example"
        ),
    }
    if covariance_mode != COVARIANCE_DIAGONAL:
        compatibility["covariance_mode"] = covariance_mode
    return compatibility, generation_recipe


def _training_data_context(
    resource_grid: ResourceGrid,
    cfg: SystemConfig,
    *,
    train_batches: int,
    train_batch_size: int,
    validation_batches: int,
    validation_batch_size: int,
    training_snr_dbs: Optional[list[float]],
    seed: int,
    asset_root: Union[str, Path],
) -> tuple[_TrainingDataContext, dict[str, torch.Tensor]]:
    """Resolve one external clean VAE-noisy data-bank contract."""
    counts = (
        int(train_batches),
        int(train_batch_size),
        int(validation_batches),
        int(validation_batch_size),
    )
    if any(value < 1 for value in counts):
        raise ValueError("VAE-noisy training-data counts must be positive.")
    snrs = tuple(_training_snr_list(cfg, training_snr_dbs))
    if len(set(snrs)) != len(snrs):
        raise ValueError("VAE-noisy training-data SNRs must be unique.")

    group_data = _PilotGroupLayout(resource_grid).training_group_data()
    input_dim = 2 * int(cfg.num_rx_ant)
    examples_per_channel = int(group_data["tx"].numel())
    train_channels = counts[0] * counts[1]
    validation_channels = counts[2] * counts[3]
    train_examples = train_channels * examples_per_channel
    validation_examples = validation_channels * examples_per_channel
    artifact_id = (
        f"train{train_channels}-val{validation_channels}"
        f"{_training_snr_suffix(list(snrs))}-seed{seed}"
    ).replace("_", "-").replace(".", "p")
    snapshot = cfg.config_snapshot()
    system = snapshot["system"]
    compatibility = {
        "channel_source": snapshot["channel_source"],
        "ofdm": system["ofdm"],
        "multi_user": system["multi_user"],
        "arrays": system["arrays"],
        "vae_noisy": {
            "input_dim": input_dim,
            "receive_vector_length": int(cfg.num_rx_ant),
            "antenna_order": "sionna_num_rx_ant_axis",
            "dft_convention": "unitary_1d_dft_over_sionna_num_rx_ant_axis",
            "pilot_groups": _pilot_group_identity(group_data),
        },
    }
    generation_recipe = {
        "train_batches": counts[0],
        "train_batch_size": counts[1],
        "validation_batches": counts[2],
        "validation_batch_size": counts[3],
        "training_snr_dbs": list(snrs),
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
        "snr_reuse": "clean-transform-vectors-shared-across-snrs",
        "noise_resampling": "deterministic-per-snr-epoch",
    }
    return (
        _TrainingDataContext(
            asset_dir=external_asset_dir(
                cfg,
                "baur2022",
                _TRAINING_DATA_ASSET_TYPE,
                artifact_id,
                asset_root=asset_root,
            ),
            compatibility=compatibility,
            generation_recipe=generation_recipe,
            metadata=metadata,
            train_examples=train_examples,
            validation_examples=validation_examples,
            input_dim=input_dim,
        ),
        group_data,
    )


def _validate_training_data_files(
    manifest: dict[str, Any],
    context: _TrainingDataContext,
) -> None:
    """Validate the exact clean-vector payload schema."""
    expected = []
    for split, num_examples in (
        ("train", context.train_examples),
        ("validation", context.validation_examples),
    ):
        expected.extend((
            (
                f"{split}-channels.npy",
                [num_examples, context.input_dim],
            ),
            (f"{split}-pilot-powers.npy", [num_examples, 1]),
        ))
    records = manifest["files"]
    if [record["path"] for record in records] != [name for name, _ in expected]:
        raise ValueError("VAE-noisy training data have unexpected payload files.")
    for record, (_, shape) in zip(records, expected):
        if record.get("format") != "npy" or record.get("array") != {
            "shape": shape,
            "dtype": "float16",
        }:
            raise ValueError("VAE-noisy training data have incompatible arrays.")


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


@lru_cache(maxsize=1)
def _materialize_training_data_cached(
    manifest_path: str,
    manifest_mtime_ns: int,
    device: str,
) -> _TrainingDataBank:
    """Keep one validated VAE-noisy bank resident on its training device."""
    _, tensors = _load_training_data_cached(
        manifest_path,
        manifest_mtime_ns,
    )
    return _TrainingDataBank(
        train_channels=tensors["train-channels.npy"].to(
            device=device,
            dtype=torch.float32,
        ),
        train_pilot_powers=tensors["train-pilot-powers.npy"].to(
            device=device,
            dtype=torch.float32,
        ),
        validation_channels=tensors["validation-channels.npy"].to(
            device=device,
            dtype=torch.float32,
        ),
        validation_pilot_powers=tensors[
            "validation-pilot-powers.npy"
        ].to(device=device, dtype=torch.float32),
    )


def _write_training_split(
    directory: Path,
    split: str,
    resource_grid: ResourceGrid,
    cfg: SystemConfig,
    group_data: dict[str, torch.Tensor],
    *,
    channel_batches: int,
    channel_batch_size: int,
    input_dim: int,
    seed: int,
) -> list[Path]:
    """Generate clean transform-domain vectors and pilot powers."""
    set_random_seed(seed)
    channel_model = UMiChannelSource(cfg, resource_grid).channel_model
    sampler = GenerateOFDMChannel(
        channel_model,
        resource_grid,
        precision=cfg.precision,
        device=cfg.device,
    )
    num_rx_ant = int(input_dim) // 2
    num_groups = int(group_data["tx"].numel())
    num_examples = int(channel_batches) * int(channel_batch_size) * num_groups
    channel_path = directory / f"{split}-channels.npy"
    power_path = directory / f"{split}-pilot-powers.npy"
    channels = np.lib.format.open_memmap(
        channel_path,
        mode="w+",
        dtype=np.float16,
        shape=(num_examples, input_dim),
    )
    pilot_powers = np.lib.format.open_memmap(
        power_path,
        mode="w+",
        dtype=np.float16,
        shape=(num_examples, 1),
    )
    dft_matrix = _unitary_dft_matrix(num_rx_ant, cfg.device)
    group_pilot_values = group_data["pilot_values"].reshape(-1)
    offset = 0
    print(
        f"[baur2022] generating {split} data: "
        f"channels={channel_batches * channel_batch_size}"
    )
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
        h = sampler(channel_batch_size)
        vector_parts = []
        power_parts = []
        for group_index in range(num_groups):
            values = h[
                :,
                :,
                :,
                int(group_data["tx"][group_index].item()),
                int(group_data["stream"][group_index].item()),
                int(group_data["symbol"][group_index].item()),
                int(group_data["subcarrier"][group_index].item()),
            ].reshape(-1, num_rx_ant)
            vector_parts.append(
                _stack_complex(_dft(values.to(torch.complex64), dft_matrix))
            )
            power_parts.append(
                torch.full(
                    (values.shape[0], 1),
                    float(group_pilot_values[group_index].abs().square()),
                    dtype=torch.float32,
                    device=h.device,
                )
            )
        channel_batch = torch.cat(vector_parts, dim=0)
        power_batch = torch.cat(power_parts, dim=0)
        next_offset = offset + int(channel_batch.shape[0])
        channels[offset:next_offset] = channel_batch.cpu().numpy().astype(
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
                f"[baur2022] {split} channel batch "
                f"{batch_index + 1}/{channel_batches}"
            )
    if offset != num_examples:
        raise RuntimeError(
            f"Expected {num_examples} VAE examples, generated {offset}."
        )
    channels.flush()
    pilot_powers.flush()
    del channels, pilot_powers
    return [channel_path, power_path]


# Payload persistence and validation


def _save_state_npz(path: str, state: dict[str, torch.Tensor]) -> list[str]:
    keys = list(state.keys())
    arrays = {
        f"arr_{i}": value.detach().cpu().numpy()
        for i, value in enumerate(state.values())
    }
    write_npz_atomic(path, **arrays)
    return keys


def _expected_state(
    input_dim: int,
    latent_dim: int,
    covariance_floor: float,
    covariance_mode: str,
) -> dict[str, torch.Tensor]:
    """Return the state-dictionary structure required by one VAE."""
    with torch.random.fork_rng(devices=[]):
        return VariationalAutoEncoder(
            input_dim,
            latent_dim,
            covariance_floor=covariance_floor,
            covariance_mode=covariance_mode,
        ).state_dict()


def _load_state_npz(
    path: str,
    expected_state: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Load a VAE state after exact key, shape, and dtype validation."""
    keys = list(expected_state)
    array_names = [f"arr_{index}" for index in range(len(keys))]
    with np.load(path, allow_pickle=False) as data:
        if data.files != array_names:
            raise ValueError("VAE asset has incompatible state arrays.")
        state = {}
        for index, key in enumerate(keys):
            array = data[array_names[index]]
            if array.shape != tuple(expected_state[key].shape):
                raise ValueError(
                    f"VAE tensor {key!r} has incompatible shape."
                )
            expected_dtype = expected_state[key].detach().cpu().numpy().dtype
            if array.dtype != expected_dtype:
                raise ValueError(
                    f"VAE tensor {key!r} has incompatible dtype."
                )
            state[key] = torch.from_numpy(array.copy())
        return state


def _load_model_asset(
    context: _ModelAssetContext,
) -> tuple[dict[str, torch.Tensor], Path]:
    """Strictly load and validate one prepared VAE asset."""
    if not context.manifest_path.is_file():
        if context.asset_dir.exists() and any(context.asset_dir.iterdir()):
            raise ValueError(f"Incomplete VAE asset at {context.asset_dir}.")
        raise FileNotFoundError(
            f"No compatible Baur VAE asset at {context.asset_dir}. Generate "
            "it with `python -m sionna.bench.mimo_ofdm_chest.papers.baur2022.run_benchmark "
            f"--prepare-assets --method {context.method} "
            "--asset-root <path>`, then evaluate it with `python -m "
            "sionna.bench.mimo_ofdm_chest.papers.baur2022.run_benchmark "
            f"--method {context.method} --asset-root <path>`."
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
        raise ValueError("VAE asset generation recipe does not match.")
    expected_state = _expected_state(
        context.input_dim,
        context.latent_dim,
        context.covariance_floor,
        context.covariance_mode,
    )
    metadata = manifest.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("VAE asset has invalid metadata.")
    if metadata.get("state_keys") != list(expected_state):
        raise ValueError("VAE asset has incompatible state keys.")
    if not isinstance(metadata.get("history"), dict):
        raise ValueError("VAE asset has invalid training history.")

    expected_arrays = {
        f"arr_{index}": {
            "shape": list(expected_state[key].shape),
            "dtype": str(expected_state[key].detach().cpu().numpy().dtype),
        }
        for index, key in enumerate(expected_state)
    }
    files = manifest["files"]
    if (
        len(files) != 1
        or files[0].get("path") != context.params_path.name
        or files[0].get("format") != "npz"
        or files[0].get("arrays") != expected_arrays
    ):
        raise ValueError("VAE manifest has incompatible state arrays.")
    state = _load_state_npz(context.params_path, expected_state)
    return state, context.manifest_path


def _write_model_asset(
    context: _ModelAssetContext,
    state: dict[str, torch.Tensor],
    history: dict[str, Any],
    *,
    cfg,
    overwrite: bool,
) -> tuple[dict[str, torch.Tensor], Path]:
    """Atomically persist one VAE asset and strictly reload it."""
    if context.asset_dir.exists() and not overwrite:
        raise FileExistsError(f"VAE asset already exists at {context.asset_dir}.")
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
            asset_type=context.asset_type,
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


def _state_to_cpu(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in state.items()}


def _collect_training_examples(
    resource_grid,
    cfg,
    group_data: dict[str, torch.Tensor],
    *,
    method: str,
    input_dim: int,
    channel_batches: int,
    channel_batch_size: int,
    training_snr_dbs: list[float],
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collect real-stacked DFT-domain VAE inputs and targets."""
    assert method in METHOD_NAMES, f"`method` must be one of {METHOD_NAMES}."
    set_random_seed(seed)

    channel_source = UMiChannelSource(cfg, resource_grid)
    channel_model = channel_source.channel_model
    sampler = GenerateOFDMChannel(
        channel_model,
        resource_grid,
        precision=cfg.precision,
        device=cfg.device,
    )
    snrs = _training_snr_list(cfg, training_snr_dbs)
    num_rx_ant = int(input_dim) // 2
    dft_matrix: torch.Tensor | None = None
    print(
        f"[baur2022] collecting VAE examples: method={method} "
        f"channel_batches={channel_batches} batch_size={channel_batch_size} "
        f"training_snr_dbs={snrs}"
    )

    x_out: torch.Tensor | None = None
    target_out: torch.Tensor | None = None
    noise_out: torch.Tensor | None = None
    offset = 0
    group_tx_indices = group_data["tx"]
    group_stream_indices = group_data["stream"]
    group_symbol_indices = group_data["symbol"]
    group_subcarrier_indices = group_data["subcarrier"]
    group_pilot_values = group_data["pilot_values"].reshape(-1)

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
        h = sampler(channel_batch_size)
        if dft_matrix is None:
            dft_matrix = _unitary_dft_matrix(num_rx_ant, h.device)

        snr_db = float(snrs[batch_index % len(snrs)])
        no = ebnodb2no(
            torch.tensor(snr_db, dtype=torch.float32, device=h.device),
            cfg.num_bits_per_symbol,
            cfg.coderate,
            resource_grid,
            precision=cfg.precision,
            device=cfg.device,
        )

        encoder_parts = []
        target_parts = []
        noise_parts = []
        for group_index in range(int(group_tx_indices.numel())):
            tx_index = int(group_tx_indices[group_index].item())
            stream_index = int(group_stream_indices[group_index].item())
            symbol_index = int(group_symbol_indices[group_index].item())
            subcarrier_index = int(group_subcarrier_indices[group_index].item())
            values = h[
                :,
                :,
                :,
                tx_index,
                stream_index,
                symbol_index,
                subcarrier_index,
            ].reshape(-1, num_rx_ant)
            values = values.to(torch.complex64)
            pilot_power = group_pilot_values[group_index].to(h.device).abs().square()
            noise_variance = no.to(torch.float32) / pilot_power.to(torch.float32)
            noise_scale = torch.sqrt(noise_variance / 2.0)
            noise = torch.complex(
                torch.randn(values.shape, dtype=torch.float32, device=h.device),
                torch.randn(values.shape, dtype=torch.float32, device=h.device),
            )
            noisy = values + noise_scale * noise
            clean_spectral = _dft(values, dft_matrix)
            noisy_spectral = _dft(noisy, dft_matrix)
            target_spectral = (
                clean_spectral if method == METHOD_NOISY else noisy_spectral
            )
            encoder_parts.append(_stack_complex(noisy_spectral))
            target_parts.append(_stack_complex(target_spectral))
            noise_parts.append(
                torch.full(
                    (values.shape[0], 1),
                    float(noise_variance.detach().cpu().item()),
                    dtype=torch.float32,
                    device=h.device,
                )
            )

        x_batch = torch.cat(encoder_parts, dim=0)
        target_batch = torch.cat(target_parts, dim=0)
        noise_batch = torch.cat(noise_parts, dim=0)
        if x_batch.shape[-1] != int(input_dim):
            raise ValueError(
                f"expected input_dim={input_dim}, got {x_batch.shape[-1]}"
            )

        if x_out is None or target_out is None or noise_out is None:
            total_examples = int(channel_batches) * int(x_batch.shape[0])
            x_out = torch.empty(
                (total_examples, int(input_dim)),
                dtype=torch.float32,
                device=x_batch.device,
            )
            target_out = torch.empty_like(x_out)
            noise_out = torch.empty(
                (total_examples, 1),
                dtype=torch.float32,
                device=x_batch.device,
            )

        next_offset = offset + x_batch.shape[0]
        x_out[offset:next_offset].copy_(x_batch)
        target_out[offset:next_offset].copy_(target_batch)
        noise_out[offset:next_offset].copy_(noise_batch)
        offset = next_offset

        if (batch_index + 1) % max(1, channel_batches // 4) == 0:
            print(
                f"[baur2022] VAE training channel batch "
                f"{batch_index + 1}/{channel_batches}"
            )

    if x_out is None or target_out is None or noise_out is None:
        raise ValueError("training set must contain at least one batch")
    return x_out[:offset], target_out[:offset], noise_out[:offset]


def _vae_loss(
    model: VariationalAutoEncoder,
    x: torch.Tensor,
    target: torch.Tensor,
    noise_variance: torch.Tensor,
    *,
    method: str,
    free_bits: float,
    sample: bool,
) -> torch.Tensor:
    latent_mean, latent_log_variance, decoder_mean, decoder_covariance = model(
        x,
        sample=sample,
    )
    if model.covariance_mode == COVARIANCE_FULL:
        residual = target - decoder_mean
        real_dim = int(target.shape[-1])
        likelihood_covariance = decoder_covariance
        if method == METHOD_REAL:
            eye = torch.eye(real_dim, dtype=x.dtype, device=x.device)
            likelihood_covariance = likelihood_covariance + (
                0.5 * noise_variance.reshape(-1, 1, 1) * eye.unsqueeze(0)
            )
        cholesky = torch.linalg.cholesky(likelihood_covariance)
        solved_residual = torch.cholesky_solve(
            residual.unsqueeze(-1),
            cholesky,
        ).squeeze(-1)
        mahalanobis = (residual * solved_residual).sum(dim=-1)
        log_det = 2.0 * torch.log(
            torch.diagonal(cholesky, dim1=-2, dim2=-1)
        ).sum(dim=-1)
        log_two_pi = torch.log(
            torch.tensor(2.0 * torch.pi, dtype=x.dtype, device=x.device)
        )
        reconstruction = 0.5 * (real_dim * log_two_pi + log_det + mahalanobis)
    else:
        target_complex = _complex_from_stack(target)
        mean_complex = _complex_from_stack(decoder_mean)
        likelihood_variance = decoder_covariance
        if method == METHOD_REAL:
            likelihood_variance = likelihood_variance + noise_variance
        squared_error = (target_complex - mean_complex).abs().square()
        reconstruction = (
            torch.log(torch.tensor(torch.pi, dtype=x.dtype, device=x.device))
            + torch.log(likelihood_variance)
            + squared_error / likelihood_variance
        ).sum(dim=-1)
    kl_per_dim = 0.5 * (
        -latent_log_variance
        + latent_mean.square()
        + torch.exp(latent_log_variance)
        - 1.0
    )
    if free_bits > 0.0:
        kl_per_dim = torch.clamp(kl_per_dim, min=float(free_bits))
    kl = kl_per_dim.sum(dim=-1)
    return (reconstruction + kl).mean()


def _loss_in_batches(
    model: VariationalAutoEncoder,
    x: torch.Tensor,
    target: torch.Tensor,
    noise_variance: torch.Tensor,
    batch_size: int,
    *,
    method: str,
    free_bits: float,
    resample_input_noise: bool = False,
    noise_seed: int = 0,
) -> float:
    noise_generator = None
    if resample_input_noise:
        noise_generator = torch.Generator(device=x.device)
        noise_generator.manual_seed(int(noise_seed))
    num = 0.0
    den = 0
    for start in range(0, x.shape[0], int(batch_size)):
        xb = x[start : start + int(batch_size)]
        yb = target[start : start + int(batch_size)]
        nb = noise_variance[start : start + int(batch_size)]
        if resample_input_noise:
            xb = xb + torch.sqrt(nb / 2.0) * torch.randn(
                xb.shape,
                generator=noise_generator,
                device=xb.device,
                dtype=xb.dtype,
            )
        loss = _vae_loss(
            model,
            xb,
            yb,
            nb,
            method=method,
            free_bits=free_bits,
            sample=False,
        )
        num += float(loss.item()) * int(xb.shape[0])
        den += int(xb.shape[0])
    return num / max(den, 1)


def _is_relative_loss_improvement(
    loss: float,
    best_loss: float,
    relative_min_delta: float,
) -> bool:
    """Return whether a loss decreased by the required relative amount."""
    if not np.isfinite(best_loss):
        return True
    return loss < best_loss - abs(best_loss) * relative_min_delta


def _train_state(
    x: torch.Tensor,
    target: torch.Tensor,
    noise_variance: torch.Tensor,
    x_val: torch.Tensor | None,
    target_val: torch.Tensor | None,
    noise_variance_val: torch.Tensor | None,
    *,
    method: str,
    input_dim: int,
    latent_dim: int,
    epochs: int,
    optimizer_batch_size: int,
    learning_rate: float,
    free_bits: float,
    min_train_seconds: float,
    patience_epochs: int,
    relative_min_delta: float,
    covariance_floor: float,
    covariance_mode: str,
    seed: int,
    resample_input_noise: bool = False,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    device = x.device
    set_random_seed(seed)
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    if x_val is None or target_val is None or noise_variance_val is None:
        permutation = torch.randperm(x.shape[0], generator=gen, device=device)
        validation_count = max(1, int(0.1 * x.shape[0]))
        validation_indices = permutation[:validation_count]
        training_indices = permutation[validation_count:]
        x_train = x[training_indices]
        target_train = target[training_indices]
        noise_train = noise_variance[training_indices]
        x_val = x[validation_indices]
        target_val = target[validation_indices]
        noise_variance_val = noise_variance[validation_indices]
        split_mode = "random_example_split"
    else:
        x_train = x
        target_train = target
        noise_train = noise_variance
        split_mode = "separate_channel_holdout"

    model = VariationalAutoEncoder(
        input_dim,
        latent_dim,
        covariance_floor=covariance_floor,
        covariance_mode=covariance_mode,
    ).to(device=device, dtype=torch.float32)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate))
    eval_batch_size = max(int(optimizer_batch_size), 16384)

    trace = []
    best_state = _state_to_cpu(model.state_dict())
    best_epoch = 0
    best_train_loss = float("inf")
    best_val_loss = float("inf")
    t0 = time.time()
    for epoch in range(int(epochs)):
        model.train()
        noise_generator = None
        if resample_input_noise:
            noise_generator = torch.Generator(device=device)
            noise_generator.manual_seed(int(seed) + 1000003 * (epoch + 1))
        order = torch.randperm(x_train.shape[0], generator=gen, device=device)
        train_loss_sum = 0.0
        train_example_count = 0
        for start in range(0, order.numel(), int(optimizer_batch_size)):
            indices = order[start : start + int(optimizer_batch_size)]
            optimizer.zero_grad(set_to_none=True)
            x_batch = x_train[indices]
            noise_batch = noise_train[indices]
            if resample_input_noise:
                x_batch = x_batch + torch.sqrt(noise_batch / 2.0) * torch.randn(
                    x_batch.shape,
                    generator=noise_generator,
                    device=device,
                    dtype=x_batch.dtype,
                )
            loss = _vae_loss(
                model,
                x_batch,
                target_train[indices],
                noise_batch,
                method=method,
                free_bits=free_bits,
                sample=True,
            )
            loss.backward()
            optimizer.step()
            train_loss_sum += float(loss.item()) * int(indices.numel())
            train_example_count += int(indices.numel())

        model.eval()
        with torch.no_grad():
            train_loss = train_loss_sum / max(train_example_count, 1)
            val_loss = _loss_in_batches(
                model,
                x_val,
                target_val,
                noise_variance_val,
                eval_batch_size,
                method=method,
                free_bits=free_bits,
                resample_input_noise=resample_input_noise,
                noise_seed=int(seed) + 2000003,
            )
        improved = _is_relative_loss_improvement(
            val_loss,
            best_val_loss,
            float(relative_min_delta),
        )
        if improved:
            best_val_loss = float(val_loss)
            best_train_loss = float(train_loss)
            best_epoch = int(epoch + 1)
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
            f"[baur2022] VAE {method} epoch {epoch + 1}/{epochs} "
            f"elapsed={elapsed:.1f}s train_loss={train_loss:.6e} "
            f"val_loss={val_loss:.6e} best_val={best_val_loss:.6e}@{best_epoch}"
        )
        saturated = (
            int(patience_epochs) > 0
            and best_epoch > 0
            and epoch + 1 - best_epoch >= int(patience_epochs)
            and elapsed >= float(min_train_seconds)
        )
        if saturated:
            print(
                f"[baur2022] early stop: no validation improvement > "
                f"{relative_min_delta:g} relative for {patience_epochs} "
                f"epochs after {elapsed:.1f}s"
            )
            break

    history = {
        "split_mode": split_mode,
        "best_train_loss": float(best_train_loss),
        "best_val_loss": float(best_val_loss),
        "best_epoch": int(best_epoch),
        "train_examples": int(x_train.shape[0]),
        "validation_examples": int(x_val.shape[0]),
        "input_noise": (
            "deterministic-fresh-per-training-epoch"
            if resample_input_noise
            else "fixed-per-generated-example"
        ),
        "elapsed_seconds": float(time.time() - t0),
        "trace": trace,
    }
    return best_state, history


def _train_model_asset(
    resource_grid: ResourceGrid,
    cfg: SystemConfig,
    group_data: dict[str, torch.Tensor],
    context: _ModelAssetContext,
    *,
    train_batches: int,
    train_batch_size: int,
    epochs: int,
    optimizer_batch_size: int,
    learning_rate: float,
    free_bits: float,
    seed: int,
    validation_batches: int = 0,
    validation_batch_size: int,
    min_train_seconds: float = 0.0,
    patience_epochs: int = 0,
    relative_min_delta: float = 0.0,
    training_data: Optional[_TrainingDataBank],
    training_data_manifest: Optional[Path],
    overwrite: bool,
) -> tuple[dict[str, torch.Tensor], Path]:
    """Train, persist, and strictly reload one VAE state."""
    snrs = context.generation_recipe["training_snr_dbs"]
    print(f"[baur2022] training VAE asset -> {context.asset_dir}")
    if training_data is None:
        x, target, noise_variance = _collect_training_examples(
            resource_grid,
            cfg,
            group_data,
            method=context.method,
            input_dim=context.input_dim,
            channel_batches=train_batches,
            channel_batch_size=train_batch_size,
            training_snr_dbs=snrs,
            seed=seed,
        )
        if int(validation_batches) > 0:
            x_val, target_val, noise_val = _collect_training_examples(
                resource_grid,
                cfg,
                group_data,
                method=context.method,
                input_dim=context.input_dim,
                channel_batches=int(validation_batches),
                channel_batch_size=int(validation_batch_size),
                training_snr_dbs=snrs,
                seed=int(seed) + 100003,
            )
        else:
            x_val = None
            target_val = None
            noise_val = None
        resample_input_noise = False
    else:
        if context.method != METHOD_NOISY or len(snrs) != 1:
            raise ValueError(
                "Fixed VAE training banks require one SNR-specific "
                "VAE-noisy model."
            )
        no = ebnodb2no(
            torch.tensor(float(snrs[0]), dtype=torch.float32, device=cfg.device),
            cfg.num_bits_per_symbol,
            cfg.coderate,
            resource_grid,
            precision=cfg.precision,
            device=cfg.device,
        ).to(torch.float32)
        x = training_data.train_channels
        target = training_data.train_channels
        noise_variance = no / training_data.train_pilot_powers
        x_val = training_data.validation_channels
        target_val = training_data.validation_channels
        noise_val = no / training_data.validation_pilot_powers
        resample_input_noise = True

    state, history = _train_state(
        x,
        target,
        noise_variance,
        x_val,
        target_val,
        noise_val,
        method=context.method,
        input_dim=context.input_dim,
        latent_dim=context.latent_dim,
        epochs=epochs,
        optimizer_batch_size=optimizer_batch_size,
        learning_rate=learning_rate,
        free_bits=free_bits,
        min_train_seconds=min_train_seconds,
        patience_epochs=patience_epochs,
        relative_min_delta=relative_min_delta,
        covariance_floor=context.covariance_floor,
        covariance_mode=context.covariance_mode,
        seed=seed,
        resample_input_noise=resample_input_noise,
    )
    history.update(
        {
            "train_channels": int(train_batches) * int(train_batch_size),
            "validation_channels": int(validation_batches)
            * int(validation_batch_size),
            "training_snr_dbs": snrs,
            "training_data_manifest": (
                None
                if training_data_manifest is None
                else {
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


def _model_contexts(
    resource_grid: ResourceGrid,
    cfg: SystemConfig,
    *,
    method: str = METHOD_NOISY,
    latent_dim: int = 8,
    train_batches: int = 4,
    train_batch_size: int = 8,
    validation_batches: int = 0,
    validation_batch_size: Optional[int] = None,
    epochs: int = 20,
    optimizer_batch_size: int = 4096,
    learning_rate: float = 1e-4,
    free_bits: float = 0.0,
    min_train_seconds: float = 0.0,
    patience_epochs: int = 0,
    relative_min_delta: float = 0.0,
    snr_specific_weights: bool = True,
    training_snr_dbs: Optional[list[float]] = None,
    covariance_floor: float = 1e-5,
    covariance_mode: str = COVARIANCE_DIAGONAL,
    seed: int = 42,
    asset_root: Optional[Union[str, Path]] = None,
) -> tuple[
    list[_ModelAssetContext],
    Optional[torch.Tensor],
    dict[str, Any],
]:
    """Resolve ordered model identities and the inference noise grid."""
    if not 0.0 <= float(relative_min_delta) < 1.0:
        raise ValueError("`relative_min_delta` must lie in [0, 1).")
    if method not in METHOD_NAMES:
        raise ValueError(f"`method` must be one of {METHOD_NAMES}.")
    if validation_batch_size is None:
        validation_batch_size = train_batch_size

    layout = _PilotGroupLayout(resource_grid)
    group_data = layout.training_group_data()
    input_dim = 2 * int(cfg.num_rx_ant)

    def context_for(selected_snr_dbs: list[float]) -> _ModelAssetContext:
        compatibility, generation_recipe = _asset_metadata(
            cfg,
            resource_grid,
            group_data,
            method=method,
            input_dim=input_dim,
            latent_dim=int(latent_dim),
            train_batches=int(train_batches),
            train_batch_size=int(train_batch_size),
            validation_batches=int(validation_batches),
            validation_batch_size=int(validation_batch_size),
            epochs=int(epochs),
            optimizer_batch_size=int(optimizer_batch_size),
            learning_rate=float(learning_rate),
            free_bits=float(free_bits),
            min_train_seconds=float(min_train_seconds),
            patience_epochs=int(patience_epochs),
            relative_min_delta=float(relative_min_delta),
            covariance_floor=float(covariance_floor),
            covariance_mode=covariance_mode,
            training_snr_dbs=selected_snr_dbs,
            seed=int(seed),
        )
        artifact_id = _artifact_id(
            method=method,
            input_dim=input_dim,
            latent_dim=int(latent_dim),
            train_batches=int(train_batches),
            train_batch_size=int(train_batch_size),
            validation_batches=int(validation_batches),
            validation_batch_size=int(validation_batch_size),
            epochs=int(epochs),
            optimizer_batch_size=int(optimizer_batch_size),
            learning_rate=float(learning_rate),
            free_bits=float(free_bits),
            min_train_seconds=float(min_train_seconds),
            patience_epochs=int(patience_epochs),
            relative_min_delta=float(relative_min_delta),
            covariance_floor=float(covariance_floor),
            covariance_mode=covariance_mode,
            training_snr_dbs=selected_snr_dbs,
            seed=int(seed),
        )
        asset_type = f"vae-{method}"
        asset_dir = (
            paper_asset_dir(cfg, asset_type, artifact_id, __file__)
            if asset_root is None
            else external_asset_dir(
                cfg,
                "baur2022",
                asset_type,
                artifact_id,
                asset_root=asset_root,
            )
        )
        return _ModelAssetContext(
            method=method,
            asset_type=asset_type,
            asset_dir=asset_dir,
            scenario_id=cfg.scenario_id,
            compatibility=compatibility,
            generation_recipe=generation_recipe,
            input_dim=input_dim,
            latent_dim=int(latent_dim),
            covariance_floor=float(covariance_floor),
            covariance_mode=covariance_mode,
            validation_batch_size=int(validation_batch_size),
        )

    if snr_specific_weights:
        snr_dbs = _training_snr_list(cfg, training_snr_dbs)
        contexts = [context_for([float(snr_db)]) for snr_db in snr_dbs]
        snr_tensor = torch.as_tensor(
            snr_dbs,
            dtype=(
                torch.float64 if cfg.precision == "double" else torch.float32
            ),
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

    selected_snr_dbs = _training_snr_list(cfg, training_snr_dbs)
    return [context_for(selected_snr_dbs)], None, group_data
