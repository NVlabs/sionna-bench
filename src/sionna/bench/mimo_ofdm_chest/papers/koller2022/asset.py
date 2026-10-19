# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import math
import os
import shutil
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, Optional, TypeAlias, Union

import numpy as np
from sionna.phy.ofdm import ResourceGrid
import torch
from sionna.phy.channel import GenerateOFDMChannel, gen_single_sector_topology
from sionna.phy.object import Object
from sionna.phy.ofdm.channel_estimation import _PilotGroupMixin

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
    write_npy_atomic,
    write_npz_atomic,
)


__all__ = [
    "METHOD_NAMES",
    "METHOD_SPATIAL",
    "METHOD_SPECTRAL",
    "generate_fitting_data",
    "fit_model",
    "load_fitting_data",
    "load_model",
]


_GMMMode: TypeAlias = Literal["spatial", "spectral"]
_NumpyGMMParameters: TypeAlias = tuple[np.ndarray, np.ndarray, np.ndarray]
_GMMParameters: TypeAlias = tuple[torch.Tensor, torch.Tensor, torch.Tensor]
_LoadedGMMAsset: TypeAlias = tuple[_GMMParameters, Path]
_PilotGroupRecord: TypeAlias = dict[str, Any]
_FittingData: TypeAlias = dict[str, np.ndarray]
_LoadedFittingData: TypeAlias = tuple[_FittingData, Path]

METHOD_SPECTRAL = "spectral"
METHOD_SPATIAL = "spatial"
METHOD_NAMES = [METHOD_SPECTRAL, METHOD_SPATIAL]

_MODE_SPATIAL = "spatial"
_MODE_SPECTRAL = "spectral"

_ASSET_TYPES = {
    _MODE_SPATIAL: "spatial-gmm",
    _MODE_SPECTRAL: "spectral-gmm",
}
_FITTING_DATA_ASSET_TYPE = "gmm-fitting-data"
_FITTING_DATA_FILES = {
    "channels": "pilot_symbol_channels.npy",
    METHOD_SPATIAL: "spatial_vectors.npy",
    METHOD_SPECTRAL: "spectral_vectors.npy",
}
_METHOD_MODES = {
    METHOD_SPATIAL: _MODE_SPATIAL,
    METHOD_SPECTRAL: _MODE_SPECTRAL,
}
_REAL_VECTOR_CONVENTION = "real-parts-then-imaginary-parts"
_PILOT_COORDINATE_ORDER = ["tx", "stream", "symbol", "subcarrier"]


@dataclass(frozen=True)
class _GMMAssetContext:
    """Resolved identity, geometry, and paths for one selected GMM asset."""

    mode: _GMMMode
    scenario_id: str
    groups: list[_PilotGroupRecord]
    expected_dim: int
    compatibility: dict[str, Any]
    generation_recipe: dict[str, Any]
    asset_dir: Path
    payload_paths: list[Path]
    manifest_path: Path


@dataclass(frozen=True)
class _FittingDataContext:
    """Resolved identity and paths for the common GMM fitting data."""

    asset_type: str
    scenario_id: str
    artifact_id: str
    compatibility: dict[str, Any]
    generation_recipe: dict[str, Any]
    asset_dir: Path
    manifest_path: Path

    @property
    def compatibility_fingerprint(self) -> str:
        """Return a hashable identity for expected compatibility."""
        return sha256_fingerprint(self.compatibility)

    @property
    def generation_recipe_fingerprint(self) -> str:
        """Return a hashable identity for the generation recipe."""
        return sha256_fingerprint(self.generation_recipe)


# Public asset lifecycle


def load_fitting_data(
    resource_grid,
    cfg,
    *,
    train_batches: int,
    train_batch_size: int,
    seed: int,
    asset_root: Union[str, Path],
) -> _LoadedFittingData:
    r"""Strictly load the common spatial/spectral GMM fitting data bank.

    The returned arrays are copy-on-write memory maps. The bank contains the
    pilot-bearing channel slices from the configured independent channel
    draws and the real-stacked vectors derived for both Koller methods.

    :returns: A mapping with ``channels``, ``spatial``, and ``spectral``
        arrays, together with the selected ``manifest.json`` path.
    """
    context = _fitting_data_context(
        resource_grid,
        cfg,
        train_batches=train_batches,
        train_batch_size=train_batch_size,
        seed=seed,
        asset_root=asset_root,
    )
    loaded = _load_fitting_data_cached(
        str(context.manifest_path),
        str(context.asset_dir),
        context.asset_type,
        context.scenario_id,
        context.artifact_id,
        context.compatibility_fingerprint,
        context.generation_recipe_fingerprint,
    )
    _validate_fitting_data_shapes(
        loaded[0],
        resource_grid,
        cfg,
        train_batches * train_batch_size,
    )
    return loaded


def generate_fitting_data(
    resource_grid: ResourceGrid,
    cfg: SystemConfig,
    *,
    train_batches: int = 100,
    train_batch_size: int = 64,
    seed: int = 22022,
    asset_root: Union[str, Path],
    verbose: bool = True,
    overwrite: bool = False,
) -> Path:
    r"""Generate the common spatial and spectral GMM fitting-data bank.

    The function samples complete noiseless system-channel realizations from the
    configured Sionna channel model. From every realization, it retains the
    pilot-bearing OFDM-symbol slices and derives both representations used by the
    Koller estimators:

    - Spatial vectors contain the complex channel across all receive antennas at
      one pilot resource element.
    - Spectral vectors contain the complete effective-subcarrier channel for one
      receive antenna, transmit stream, and pilot-bearing OFDM symbol.

    The same independently generated channels are used for both representations,
    so spatial and spectral GMMs can be fitted and tuned without regenerating the
    underlying propagation channels. The data are written under the supplied
    external asset root together with a manifest recording the system geometry,
    generation recipe, payload identities, and independent-channel count. The
    completed asset is strictly reloaded and validated before its manifest path is
    returned.

    .. rubric:: Deviations from the Paper

    - The spatial data pool vectors from all UEs, pilot subcarriers, and
      pilot-bearing OFDM symbols, assuming common spatial statistics.
    - The spectral data pool vectors from all UEs, receive antennas, and
      pilot-bearing OFDM symbols, assuming common spectral statistics.
    - Only pilot-bearing OFDM-symbol slices are retained because the implemented
      estimators do not consume the remaining symbols during GMM fitting.
    - The fitting channels are generated by the configured Sionna channel model
      rather than loaded from a paper-provided dataset.

    :param resource_grid: Sionna resource grid defining the OFDM dimensions,
        effective subcarriers, streams, and pilot geometry.
    :param cfg: System and channel-source configuration used to generate the
        independent channel realizations.
    :param train_batches: Number of independently generated channel batches.
    :param train_batch_size: Number of complete system-channel realizations in
        each batch. The total independent-channel count is
        ``train_batches * train_batch_size``.
    :param seed: Seed used for topology and channel generation.
    :param asset_root: External directory under which the fitting-data asset is
        written.
    :param verbose: If `True`, report generation progress.
    :param overwrite: If `True`, replace an existing asset with the same identity;
        otherwise raise :class:`FileExistsError`.

    :returns: Path to the strictly validated fitting-data ``manifest.json``.
    """
    context = _fitting_data_context(
        resource_grid,
        cfg,
        train_batches=train_batches,
        train_batch_size=train_batch_size,
        seed=seed,
        asset_root=asset_root,
    )
    if context.asset_dir.exists() and not overwrite:
        raise FileExistsError(
            f"GMM fitting data already exist at {context.asset_dir}."
        )

    context.asset_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{context.asset_dir.name}.",
            dir=context.asset_dir.parent,
        )
    )
    try:
        payload_paths = [
            temporary_dir / filename
            for filename in _FITTING_DATA_FILES.values()
        ]
        _write_fitting_data(
            resource_grid,
            cfg,
            payload_paths,
            train_batches=train_batches,
            train_batch_size=train_batch_size,
            seed=seed,
            verbose=verbose,
        )
        manifest = build_asset_manifest(
            cfg=cfg,
            asset_type=_FITTING_DATA_ASSET_TYPE,
            artifact_id=context.artifact_id,
            compatibility=context.compatibility,
            generation_recipe=context.generation_recipe,
            payload_paths=payload_paths,
            metadata={
                "independent_channel_count": (
                    train_batches * train_batch_size
                ),
                "retained_channel_content": (
                    "pilot-bearing-ofdm-symbol-slices"
                ),
                "derived_vector_convention": _REAL_VECTOR_CONVENTION,
            },
        )
        write_json_atomic(temporary_dir / "manifest.json", manifest)
        if context.asset_dir.exists():
            shutil.rmtree(context.asset_dir)
        os.replace(temporary_dir, context.asset_dir)
    finally:
        shutil.rmtree(temporary_dir, ignore_errors=True)

    _load_fitting_data_cached.cache_clear()
    _, manifest_path = load_fitting_data(
        resource_grid,
        cfg,
        train_batches=train_batches,
        train_batch_size=train_batch_size,
        seed=seed,
        asset_root=asset_root,
    )
    return manifest_path


def load_model(
    resource_grid,
    cfg,
    *,
    method: str,
    num_components: int,
    train_batches: int,
    train_batch_size: int,
    em_iters: int,
    covariance_reg: float,
    seed: int,
    verbose: bool = True,
    asset_root: Optional[Union[str, Path]] = None,
) -> _LoadedGMMAsset:
    r"""Strictly load one prepared pooled GMM and its manifest path.

    This function never samples channels, fits a GMM, or writes an asset.
    With no ``asset_root``, it selects an optional package-local asset.

    :returns: A tuple containing ``(weights, means, covariances)`` as CPU
        `torch.float32` tensors and the selected ``manifest.json`` path.
    """
    context = _asset_context(
        resource_grid,
        cfg,
        method=method,
        num_components=num_components,
        train_batches=train_batches,
        train_batch_size=train_batch_size,
        em_iters=em_iters,
        covariance_reg=covariance_reg,
        seed=seed,
        asset_root=asset_root,
    )
    if not context.manifest_path.is_file():
        if context.asset_dir.exists() and any(context.asset_dir.iterdir()):
            raise ValueError(f"Incomplete GMM asset at {context.asset_dir}.")
        seed = context.generation_recipe["seed"]
        raise FileNotFoundError(
            f"No compatible Koller GMM asset at {context.asset_dir}. Generate "
            "it with `python -m "
            "sionna.bench.mimo_ofdm_chest.papers.koller2022.run_benchmark "
            f"--prepare-assets --method {context.mode} --model-seed {seed} "
            "--asset-root <path>`, "
            "then evaluate it with `python -m "
            "sionna.bench.mimo_ofdm_chest.papers.koller2022.run_benchmark "
            f"--method {context.mode} --model-seed {seed} "
            "--asset-root <path>`."
        )

    _load_model_manifest(context)
    if verbose:
        print(
            f"[koller2022] loading {context.mode} GMM asset from "
            f"{context.asset_dir}"
        )
    parameters = _load_gmm_parameters(
        context.payload_paths,
        context.mode,
        num_components,
        context.expected_dim,
    )
    tensors = tuple(torch.from_numpy(value) for value in parameters)
    return tensors, context.manifest_path


def fit_model(
    resource_grid: ResourceGrid,
    cfg: SystemConfig,
    *,
    method: str,
    num_components: int,
    train_batches: int = 100,
    train_batch_size: int = 64,
    em_iters: int = 15,
    covariance_reg: float = 1e-3,
    seed: int = 22022,
    asset_root: Union[str, Path],
    verbose: bool = True,
    overwrite: bool = False,
    fitting_data_manifest: Optional[Union[str, Path]] = None,
) -> _LoadedGMMAsset:
    r"""Fit and persist one spatial or spectral Gaussian mixture model.

    The selected method's real-valued channel vectors are fitted with a
    full-covariance Gaussian mixture model using expectation maximization. Complex
    vectors are represented by concatenating all real entries followed by all
    imaginary entries.

    When ``fitting_data_manifest`` is supplied, the function loads the selected
    method's vectors from the common deterministic fitting-data bank. Otherwise,
    it generates the configured number of channel realizations directly. The
    canonical fitting procedure supplies the common data-bank manifest so all
    component-count candidates and both estimator methods use the same underlying
    channels.

    The fitted weights, means, and covariance matrices are written beneath the
    explicit external asset root. The completed asset is strictly reloaded and
    validated before its tensors and manifest path are returned.

    .. rubric:: Deviations from the Paper

    - The spatial GMM pools receive-array vectors from all UEs, pilot subcarriers,
      and pilot-bearing OFDM symbols.
    - The spectral GMM pools full-frequency vectors from all UEs, receive antennas,
      and pilot-bearing OFDM symbols.
    - Complex channels are fitted as real-valued vectors with real entries followed
      by imaginary entries.
    - The paper prescribes EM but does not fully specify initialization,
      convergence, or covariance stabilization. This implementation uses seeded
      sample initialization, a mean-log-likelihood stopping tolerance of
      ``1e-4``, and variance-scaled diagonal covariance loading.
    - Fitting uses channels from the configured Sionna channel model rather than a
      paper-provided dataset.

    :param resource_grid: Sionna resource grid defining the channel-vector and
        pilot geometry.
    :param cfg: System and channel-source configuration.
    :param method: GMM representation to fit, either ``"spatial"`` or
        ``"spectral"``.
    :param num_components: Number of Gaussian mixture components.
    :param train_batches: Number of independently generated channel batches.
    :param train_batch_size: Number of complete system-channel realizations in
        each batch.
    :param em_iters: Maximum number of expectation-maximization iterations.
    :param covariance_reg: Diagonal covariance-loading factor. The applied loading
        is this value times the mean sample variance, with a numerical floor.
    :param seed: Seed used for channel generation and component initialization.
    :param asset_root: External directory under which the fitted GMM asset is
        stored.
    :param verbose: If `True`, report fitting progress and EM likelihoods.
    :param overwrite: If `True`, replace an existing asset with the same identity;
        otherwise raise :class:`FileExistsError`.
    :param fitting_data_manifest: Optional manifest for the common deterministic
        fitting-data bank. If omitted, channels are generated during this call.

    :returns: A tuple ``((weights, means, covariances), manifest_path)``. The GMM
        parameters are CPU `torch.float32` tensors, and ``manifest_path`` identifies
        the strictly validated fitted asset.
    """
    context = _asset_context(
        resource_grid,
        cfg,
        method=method,
        num_components=num_components,
        train_batches=train_batches,
        train_batch_size=train_batch_size,
        em_iters=em_iters,
        covariance_reg=covariance_reg,
        seed=seed,
        asset_root=asset_root,
    )
    if context.asset_dir.exists() and not overwrite:
        raise FileExistsError(f"GMM asset already exists at {context.asset_dir}.")

    if verbose:
        print(f"[koller2022] fitting {context.mode} GMM asset -> {context.asset_dir}")
    fitting_data_reference = None
    if fitting_data_manifest is None:
        weights, means, covariances = _fit_from_channel_samples(
            resource_grid,
            cfg,
            context.groups,
            context.mode,
            num_components=num_components,
            train_batches=train_batches,
            train_batch_size=train_batch_size,
            em_iters=em_iters,
            covariance_reg=covariance_reg,
            seed=seed,
            verbose=verbose,
        )
    else:
        fitting_data, validated_manifest_path = _load_fitting_data_from_manifest(
            resource_grid,
            cfg,
            train_batches=train_batches,
            train_batch_size=train_batch_size,
            seed=seed,
            manifest_path=fitting_data_manifest,
        )
        weights, means, covariances = _fit_gmm_real(
            fitting_data[context.mode],
            num_components,
            max_iter=em_iters,
            covariance_reg=covariance_reg,
            seed=seed,
            device=cfg.device,
            verbose=verbose,
        )
        fitting_data_reference = {
            "asset_type": _FITTING_DATA_ASSET_TYPE,
            "artifact_id": validated_manifest_path.parent.name,
            "manifest_sha256": sha256_file(validated_manifest_path),
        }

    context.asset_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{context.asset_dir.name}.",
            dir=context.asset_dir.parent,
        )
    )
    try:
        temporary_payloads = _gmm_payload_paths(
            temporary_dir,
            context.mode,
            num_components,
        )
        _write_gmm_parameters(
            temporary_payloads,
            context.mode,
            weights,
            means,
            covariances,
        )
        manifest = build_asset_manifest(
            cfg=cfg,
            asset_type=_ASSET_TYPES[context.mode],
            artifact_id=context.asset_dir.name,
            compatibility=context.compatibility,
            generation_recipe=context.generation_recipe,
            payload_paths=temporary_payloads,
            metadata=_asset_metadata(
                context.mode,
                fitting_data_reference=fitting_data_reference,
            ),
        )
        write_json_atomic(temporary_dir / "manifest.json", manifest)
        if context.asset_dir.exists():
            shutil.rmtree(context.asset_dir)
        os.replace(temporary_dir, context.asset_dir)
    finally:
        shutil.rmtree(temporary_dir, ignore_errors=True)

    return load_model(
        resource_grid,
        cfg,
        method=method,
        num_components=num_components,
        train_batches=train_batches,
        train_batch_size=train_batch_size,
        em_iters=em_iters,
        covariance_reg=covariance_reg,
        seed=seed,
        verbose=verbose,
        asset_root=asset_root,
    )


# Asset identity and compatibility


def _fitting_data_context(
    resource_grid,
    cfg,
    *,
    train_batches: int,
    train_batch_size: int,
    seed: int,
    asset_root: Union[str, Path],
) -> _FittingDataContext:
    """Resolve the common external fitting-data identity and contract."""
    if train_batches < 1 or train_batch_size < 1:
        raise ValueError("Fitting-data batch counts must be positive.")
    artifact_id = f"tb{train_batches}-bs{train_batch_size}-seed{seed}"
    asset_dir = external_asset_dir(
        cfg,
        "koller2022",
        _FITTING_DATA_ASSET_TYPE,
        artifact_id,
        asset_root=asset_root,
    )
    compatibility = _fitting_data_compatibility(resource_grid, cfg)
    generation_recipe = {
        "train_batches": int(train_batches),
        "train_batch_size": int(train_batch_size),
        "seed": int(seed),
    }
    return _FittingDataContext(
        asset_type=_FITTING_DATA_ASSET_TYPE,
        scenario_id=cfg.scenario_id,
        artifact_id=artifact_id,
        compatibility=compatibility,
        generation_recipe=generation_recipe,
        asset_dir=asset_dir,
        manifest_path=asset_dir / "manifest.json",
    )


def _fitting_data_compatibility(resource_grid, cfg) -> dict[str, Any]:
    """Return the channel and vector contract shared by both GMM modes."""
    snapshot = cfg.config_snapshot()
    system = snapshot["system"]
    return {
        "channel_source": snapshot["channel_source"],
        "ofdm": system["ofdm"],
        "multi_user": system["multi_user"],
        "arrays": system["arrays"],
        "methods": {
            method: _asset_compatibility(
                resource_grid,
                cfg,
                _PilotGroupLayout(resource_grid.pilot_pattern, method),
                _METHOD_MODES[method],
            )
            for method in METHOD_NAMES
        },
    }


def _load_fitting_data_from_manifest(
    resource_grid,
    cfg,
    *,
    train_batches: int,
    train_batch_size: int,
    seed: int,
    manifest_path: Union[str, Path],
) -> _LoadedFittingData:
    """Validate and memory-map one explicitly selected fitting data bank."""
    manifest_path = Path(manifest_path)
    expected_artifact_id = (
        f"tb{train_batches}-bs{train_batch_size}-seed{seed}"
    )
    compatibility = _fitting_data_compatibility(resource_grid, cfg)
    generation_recipe = {
        "train_batches": int(train_batches),
        "train_batch_size": int(train_batch_size),
        "seed": int(seed),
    }
    loaded = _load_fitting_data_cached(
        str(manifest_path),
        str(manifest_path.parent),
        _FITTING_DATA_ASSET_TYPE,
        cfg.scenario_id,
        expected_artifact_id,
        sha256_fingerprint(compatibility),
        sha256_fingerprint(generation_recipe),
    )
    _validate_fitting_data_shapes(
        loaded[0],
        resource_grid,
        cfg,
        train_batches * train_batch_size,
    )
    return loaded


@lru_cache(maxsize=4)
def _load_fitting_data_cached(
    manifest_path: str,
    asset_dir: str,
    expected_asset_type: str,
    expected_scenario_id: str,
    expected_artifact_id: str,
    compatibility_fingerprint: str,
    generation_recipe_fingerprint: str,
) -> _LoadedFittingData:
    """Validate and cache read-only memory maps for one large data bank."""
    manifest = load_asset_manifest(
        manifest_path,
        asset_dir=asset_dir,
        expected_asset_type=expected_asset_type,
        expected_scenario_id=expected_scenario_id,
        expected_artifact_id=expected_artifact_id,
    )
    if (
        sha256_fingerprint(manifest["compatibility"])
        != compatibility_fingerprint
    ):
        raise ValueError("GMM fitting-data compatibility does not match.")
    if (
        sha256_fingerprint(manifest["generation_recipe"])
        != generation_recipe_fingerprint
    ):
        raise ValueError("GMM fitting-data generation recipe does not match.")
    recipe = manifest["generation_recipe"]
    expected_metadata = {
        "independent_channel_count": (
            recipe["train_batches"] * recipe["train_batch_size"]
        ),
        "retained_channel_content": "pilot-bearing-ofdm-symbol-slices",
        "derived_vector_convention": _REAL_VECTOR_CONVENTION,
    }
    if manifest.get("metadata") != expected_metadata:
        raise ValueError("GMM fitting-data metadata do not match.")
    payload_names = [record["path"] for record in manifest["files"]]
    if payload_names != list(_FITTING_DATA_FILES.values()):
        raise ValueError("GMM fitting data have unexpected payload files.")
    arrays = {
        name: np.load(
            Path(asset_dir) / filename,
            mmap_mode="c",
            allow_pickle=False,
        )
        for name, filename in _FITTING_DATA_FILES.items()
    }
    if arrays["channels"].dtype != np.complex64:
        raise ValueError("GMM fitting channel slices must use complex64.")
    for method in METHOD_NAMES:
        if arrays[method].dtype != np.float32:
            raise ValueError("GMM fitting vectors must use float32.")
        if arrays[method].ndim != 2:
            raise ValueError("GMM fitting vectors must be matrices.")
    return arrays, Path(manifest_path)


def _validate_fitting_data_shapes(
    arrays: _FittingData,
    resource_grid,
    cfg,
    num_channels: int,
) -> None:
    """Validate exact channel-slice and derived-vector dimensions."""
    layouts = {
        method: _PilotGroupLayout(resource_grid.pilot_pattern, method)
        for method in METHOD_NAMES
    }
    records = {
        method: layout._group_records()
        for method, layout in layouts.items()
    }
    pilot_symbols = sorted(
        {record["symbol"] for method_records in records.values()
         for record in method_records}
    )
    expected_shapes = {
        "channels": (
            num_channels,
            1,
            cfg.num_rx_ant,
            cfg.num_ue,
            cfg.num_streams_per_tx,
            len(pilot_symbols),
            resource_grid.num_effective_subcarriers,
        ),
        METHOD_SPATIAL: (
            num_channels * len(records[METHOD_SPATIAL]),
            2 * cfg.num_rx_ant,
        ),
        METHOD_SPECTRAL: (
            num_channels
            * cfg.num_rx_ant
            * len(records[METHOD_SPECTRAL]),
            2 * resource_grid.num_effective_subcarriers,
        ),
    }
    for name, expected_shape in expected_shapes.items():
        if arrays[name].shape != expected_shape:
            raise ValueError(
                f"GMM fitting array {name!r} has shape "
                f"{arrays[name].shape}, expected {expected_shape}."
            )


def _asset_context(
    resource_grid,
    cfg,
    *,
    method: str,
    num_components: int,
    train_batches: int,
    train_batch_size: int,
    em_iters: int,
    covariance_reg: float,
    seed: int,
    asset_root: Optional[Union[str, Path]],
) -> _GMMAssetContext:
    """Resolve and validate one GMM asset's geometry and identity."""
    if method not in METHOD_NAMES:
        raise ValueError(f"`method` must be one of {METHOD_NAMES}.")
    mode: _GMMMode = _METHOD_MODES[method]
    layout = _PilotGroupLayout(resource_grid.pilot_pattern, method)
    if method == METHOD_SPATIAL:
        pilot_values = layout._gather_pilot_groups(
            resource_grid.pilot_pattern.pilots
        )
        pilot_power = pilot_values.abs().square()
        assert bool(torch.allclose(pilot_power, pilot_power.reshape(-1)[0])), (
            "Spatial GMM joint processing requires equal-power pilots."
        )
    groups = layout._group_records()
    assert len(groups) > 0, "`groups` must not be empty."
    assert num_components >= 1, "`num_components` must be positive."
    assert train_batches >= 1, "`train_batches` must be positive."
    assert train_batch_size >= 1, "`train_batch_size` must be positive."
    assert em_iters >= 1, "`em_iters` must be positive."
    assert covariance_reg >= 0.0, "`covariance_reg` must be nonnegative."

    group_lengths = []
    required_group_keys = {
        "tx",
        "stream",
        "symbol",
        "vector_indices",
        "subcarriers",
    }
    for group in groups:
        missing_keys = required_group_keys.difference(group.keys())
        assert not missing_keys, (
            f"Pilot group is missing keys {sorted(missing_keys)}."
        )
        if mode == _MODE_SPATIAL:
            assert int(group["vector_indices"].numel()) == 1, (
                "Spatial GMM groups must contain one pilot resource element."
            )
            assert int(group["subcarriers"].numel()) == 1, (
                "Spatial GMM groups must contain one pilot subcarrier."
            )
            group_lengths.append(int(cfg.num_rx_ant))
        else:
            group_lengths.append(int(resource_grid.num_effective_subcarriers))
    assert min(group_lengths) > 0, "GMM pilot vectors must not be empty."
    assert len(set(group_lengths)) == 1, (
        "Pooled GMM training requires equal-length pilot vectors."
    )

    artifact_id = (
        f"k{num_components}_tb{train_batches}_bs{train_batch_size}"
        f"_em{em_iters}_reg{covariance_reg:g}_seed{seed}"
    ).replace("_", "-").replace(".", "p")
    asset_dir = _asset_dir(cfg, mode, artifact_id, asset_root)
    generation_recipe = {
        "num_components": int(num_components),
        "train_batches": int(train_batches),
        "train_batch_size": int(train_batch_size),
        "em_iters": int(em_iters),
        "covariance_reg": float(covariance_reg),
        "seed": int(seed),
    }
    return _GMMAssetContext(
        mode=mode,
        scenario_id=cfg.scenario_id,
        groups=groups,
        expected_dim=2 * group_lengths[0],
        compatibility=_asset_compatibility(resource_grid, cfg, layout, mode),
        generation_recipe=generation_recipe,
        asset_dir=asset_dir,
        payload_paths=_gmm_payload_paths(asset_dir, mode, num_components),
        manifest_path=asset_dir / "manifest.json",
    )


def _asset_dir(
    cfg,
    mode: _GMMMode,
    artifact_id: str,
    asset_root: Optional[Union[str, Path]],
) -> Path:
    """Select the package-local or external GMM artifact directory."""
    if asset_root is None:
        return paper_asset_dir(cfg, _ASSET_TYPES[mode], artifact_id, __file__)
    return external_asset_dir(
        cfg,
        "koller2022",
        _ASSET_TYPES[mode],
        artifact_id,
        asset_root=asset_root,
    )


def _asset_compatibility(
    resource_grid,
    cfg,
    layout: _PilotGroupLayout,
    mode: _GMMMode,
) -> dict[str, Any]:
    """Return the exact vector and pilot contract for one pooled GMM."""
    pilot_values = layout._gather_pilot_groups(
        resource_grid.pilot_pattern.pilots
    ).detach().cpu()
    pilot_coordinates = layout._pilot_group_coords.detach().cpu()
    pilot_value_pairs = torch.view_as_real(pilot_values)
    pilot_powers = pilot_values.abs().square()
    complex_vector_length = (
        int(cfg.num_rx_ant)
        if mode == _MODE_SPATIAL
        else int(resource_grid.num_effective_subcarriers)
    )
    vector_axis = (
        "receive-antenna"
        if mode == _MODE_SPATIAL
        else "effective-subcarrier"
    )
    return {
        "estimator_mode": mode,
        "real_vector_convention": _REAL_VECTOR_CONVENTION,
        "vector_axis": vector_axis,
        "vector_dimensions": {
            "complex": complex_vector_length,
            "real_stacked": 2 * complex_vector_length,
        },
        "pilot_groups": {
            "coordinate_order": _PILOT_COORDINATE_ORDER,
            "coordinates": {
                "shape": list(pilot_coordinates.shape),
                "dtype": "int64",
                "canonical_sha256": sha256_fingerprint(
                    pilot_coordinates.tolist()
                ),
            },
            "values": {
                "shape": list(pilot_values.shape),
                "dtype": str(pilot_values.dtype).removeprefix("torch."),
                "encoding": "[real, imaginary]",
                "canonical_sha256": sha256_fingerprint(
                    pilot_value_pairs.tolist()
                ),
            },
            "powers": {
                "shape": list(pilot_powers.shape),
                "dtype": str(pilot_powers.dtype).removeprefix("torch."),
                "canonical_sha256": sha256_fingerprint(
                    pilot_powers.tolist()
                ),
            },
        },
    }


def _asset_metadata(
    mode: _GMMMode,
    *,
    fitting_data_reference: Optional[dict[str, str]] = None,
) -> dict[str, Any]:
    """Describe the spatial or split-spectral payload layout."""
    metadata = {
        "payload_layout": (
            "single-npz" if mode == _MODE_SPATIAL else "split-covariances"
        ),
        "parameter_arrays": ["weights", "means", "covariances"],
    }
    if fitting_data_reference is not None:
        metadata["fitting_data"] = dict(fitting_data_reference)
    return metadata


# Payload persistence and validation


def _gmm_payload_paths(
    out_dir: Path,
    mode: _GMMMode,
    num_components: int,
) -> list[Path]:
    """Return the complete ordered payload path list for one GMM asset."""
    paths = [out_dir / "gmm_params.npz"]
    if mode == _MODE_SPECTRAL:
        paths.extend(
            out_dir / f"gmm_covariance_{component_index:03d}.npy"
            for component_index in range(num_components)
        )
    return paths


def _load_model_manifest(context: _GMMAssetContext) -> dict[str, Any]:
    """Load and validate one global GMM manifest and its payload ordering."""
    manifest = load_asset_manifest(
        context.manifest_path,
        asset_dir=context.asset_dir,
        expected_asset_type=_ASSET_TYPES[context.mode],
        expected_scenario_id=context.scenario_id,
        expected_artifact_id=context.asset_dir.name,
        expected_compatibility=context.compatibility,
    )
    if manifest["generation_recipe"] != context.generation_recipe:
        raise ValueError(
            f"Incompatible GMM generation recipe at {context.asset_dir}."
        )
    metadata = manifest.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("GMM asset metadata does not match.")
    expected_metadata = _asset_metadata(context.mode)
    if any(metadata.get(key) != value for key, value in expected_metadata.items()):
        raise ValueError("GMM asset metadata does not match.")
    fitting_data_reference = metadata.get("fitting_data")
    if fitting_data_reference is not None and (
        not isinstance(fitting_data_reference, dict)
        or set(fitting_data_reference)
        != {"asset_type", "artifact_id", "manifest_sha256"}
        or fitting_data_reference["asset_type"] != _FITTING_DATA_ASSET_TYPE
    ):
        raise ValueError("GMM fitting-data reference is invalid.")
    payload_names = [record["path"] for record in manifest["files"]]
    if payload_names != [path.name for path in context.payload_paths]:
        raise ValueError("GMM asset has unexpected payload files.")
    return manifest


def _load_gmm_parameters(
    payload_paths: list[Path],
    mode: _GMMMode,
    num_components: int,
    expected_dim: int,
) -> _NumpyGMMParameters:
    """Load and validate one spatial or chunked spectral GMM payload."""
    with np.load(payload_paths[0], allow_pickle=False) as data:
        expected_keys = (
            {"weights", "means", "covs"}
            if mode == _MODE_SPATIAL
            else {"weights", "means"}
        )
        if set(data.files) != expected_keys:
            raise ValueError("GMM asset has unexpected arrays.")
        weights = data["weights"]
        means = data["means"]
        if mode == _MODE_SPATIAL:
            covariances = data["covs"]
        else:
            covariances = np.stack(
                [
                    np.load(path, allow_pickle=False)
                    for path in payload_paths[1:]
                ],
                axis=0,
            )

    if (
        weights.shape != (num_components,)
        or means.shape != (num_components, expected_dim)
        or covariances.shape
        != (num_components, expected_dim, expected_dim)
        or any(
            value.dtype != np.float32
            for value in (weights, means, covariances)
        )
    ):
        raise ValueError("GMM asset has incompatible shapes or dtypes.")
    return weights, means, covariances

def _write_gmm_parameters(
    payload_paths: list[Path],
    mode: _GMMMode,
    weights: np.ndarray,
    means: np.ndarray,
    covariances: np.ndarray,
) -> None:
    """Persist one GMM while bounding every spectral covariance file."""
    if mode == _MODE_SPATIAL:
        write_npz_atomic(
            payload_paths[0],
            weights=weights,
            means=means,
            covs=covariances,
        )
        return

    write_npz_atomic(payload_paths[0], weights=weights, means=means)
    assert len(payload_paths) == covariances.shape[0] + 1
    for path, covariance in zip(payload_paths[1:], covariances):
        write_npy_atomic(path, covariance)


# Pilot grouping and GMM fitting


class _PilotGroupLayout(_PilotGroupMixin, Object):
    """Expose pilot-group coordinates needed for offline GMM fitting."""

    def __init__(self, pilot_pattern, method: str) -> None:
        super().__init__(
            precision=pilot_pattern.precision,
            device=pilot_pattern.device,
        )
        group_over = () if method == METHOD_SPATIAL else ("subcarrier",)
        self._init_pilot_groups(
            pilot_pattern=pilot_pattern,
            group_over=group_over,
            persistent=False,
        )

    def _group_records(self) -> list[_PilotGroupRecord]:
        """Return ordered coordinates for all pooled training groups."""
        tx_indices = self._pilot_group_constant_coord("tx").detach().cpu()
        stream_indices = self._pilot_group_constant_coord("stream").detach().cpu()
        symbol_indices = self._pilot_group_constant_coord("symbol").detach().cpu()
        vector_indices = self._pilot_group_vector_indices.detach().cpu()
        if self._pilot_group_over:
            subcarrier_indices = self._pilot_group_varying_coord(
                "subcarrier"
            ).detach().cpu()
        else:
            subcarrier_indices = self._pilot_group_constant_coord(
                "subcarrier"
            ).detach().cpu()

        return [
            {
                "tx": int(tx_indices[group_index].item()),
                "stream": int(stream_indices[group_index].item()),
                "symbol": int(symbol_indices[group_index].item()),
                "vector_indices": vector_indices[group_index],
                "subcarriers": subcarrier_indices[group_index],
            }
            for group_index in range(self._pilot_group_num_groups)
        ]


def _group_channel_samples(
    h: torch.Tensor,
    group: _PilotGroupRecord,
    mode: _GMMMode,
) -> torch.Tensor:
    """Extract one pilot group's spatial or spectral training vectors."""
    if mode == _MODE_SPATIAL:
        subcarrier_index = int(group["subcarriers"].item())
        values = h[
            ...,
            group["tx"],
            group["stream"],
            group["symbol"],
            subcarrier_index,
        ]
    else:
        values = h[..., group["tx"], group["stream"], group["symbol"], :]

    # Spatial vectors span receive antennas; spectral vectors span subcarriers.
    return values.reshape(-1, values.shape[-1])


def _group_all_channel_samples(
    channel: torch.Tensor,
    groups: list[_PilotGroupRecord],
    mode: _GMMMode,
) -> torch.Tensor:
    """Extract all group-major fitting vectors in one device operation."""
    device = channel.device
    tx_indices = torch.tensor(
        [group["tx"] for group in groups],
        device=device,
        dtype=torch.int64,
    )
    stream_indices = torch.tensor(
        [group["stream"] for group in groups],
        device=device,
        dtype=torch.int64,
    )
    symbol_indices = torch.tensor(
        [group["symbol"] for group in groups],
        device=device,
        dtype=torch.int64,
    )
    if mode == _MODE_SPATIAL:
        subcarrier_indices = torch.stack(
            [group["subcarriers"] for group in groups]
        ).reshape(-1).to(device)
        values = channel[
            ...,
            tx_indices,
            stream_indices,
            symbol_indices,
            subcarrier_indices,
        ]
        values = values.movedim(-1, 0)
    else:
        values = channel[
            ...,
            tx_indices,
            stream_indices,
            symbol_indices,
            :,
        ]
        values = values.movedim(-2, 0)
    return values.reshape(-1, values.shape[-1])


def _write_fitting_data(
    resource_grid,
    cfg,
    payload_paths: list[Path],
    *,
    train_batches: int,
    train_batch_size: int,
    seed: int,
    verbose: bool,
) -> None:
    """Generate one channel stream and persist both methods' fitting data."""
    path_by_name = dict(zip(_FITTING_DATA_FILES, payload_paths))
    groups = {
        method: _PilotGroupLayout(
            resource_grid.pilot_pattern,
            method,
        )._group_records()
        for method in METHOD_NAMES
    }
    pilot_symbols = sorted(
        {record["symbol"] for method_groups in groups.values()
         for record in method_groups}
    )
    num_channels = train_batches * train_batch_size
    shapes = {
        "channels": (
            num_channels,
            1,
            cfg.num_rx_ant,
            cfg.num_ue,
            cfg.num_streams_per_tx,
            len(pilot_symbols),
            resource_grid.num_effective_subcarriers,
        ),
        METHOD_SPATIAL: (
            num_channels * len(groups[METHOD_SPATIAL]),
            2 * cfg.num_rx_ant,
        ),
        METHOD_SPECTRAL: (
            num_channels * cfg.num_rx_ant * len(groups[METHOD_SPECTRAL]),
            2 * resource_grid.num_effective_subcarriers,
        ),
    }
    arrays = {
        "channels": np.lib.format.open_memmap(
            path_by_name["channels"],
            mode="w+",
            dtype=np.complex64,
            shape=shapes["channels"],
        ),
        METHOD_SPATIAL: np.lib.format.open_memmap(
            path_by_name[METHOD_SPATIAL],
            mode="w+",
            dtype=np.float32,
            shape=shapes[METHOD_SPATIAL],
        ),
        METHOD_SPECTRAL: np.lib.format.open_memmap(
            path_by_name[METHOD_SPECTRAL],
            mode="w+",
            dtype=np.float32,
            shape=shapes[METHOD_SPECTRAL],
        ),
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
    for batch_index in range(train_batches):
        topology = gen_single_sector_topology(
            batch_size=train_batch_size,
            num_ut=cfg.num_ue,
            scenario="umi",
            min_ut_velocity=cfg.ue_speed,
            max_ut_velocity=cfg.ue_speed,
            precision=cfg.precision,
            device=cfg.device,
        )
        channel_model.set_topology(*topology)
        channel = sampler(train_batch_size)
        channel_start = batch_index * train_batch_size
        channel_stop = channel_start + train_batch_size
        arrays["channels"][channel_start:channel_stop] = (
            channel[..., pilot_symbols, :]
            .detach()
            .cpu()
            .numpy()
            .astype(np.complex64, copy=False)
        )

        for method in METHOD_NAMES:
            complex_vectors = (
                _group_all_channel_samples(
                    channel,
                    groups[method],
                    _METHOD_MODES[method],
                )
                .detach()
                .cpu()
                .numpy()
            )
            real_vectors = np.concatenate(
                [complex_vectors.real, complex_vectors.imag],
                axis=-1,
            ).astype(np.float32, copy=False)
            rows_per_batch = real_vectors.shape[0]
            row_start = batch_index * rows_per_batch
            arrays[method][row_start:row_start + rows_per_batch] = real_vectors

        if verbose and (batch_index + 1) % max(1, train_batches // 10) == 0:
            print(
                "  shared GMM fitting-data channel batch "
                f"{batch_index + 1}/{train_batches}",
            )

    for array in arrays.values():
        array.flush()


def _fit_from_channel_samples(
    resource_grid,
    cfg,
    groups: list[_PilotGroupRecord],
    mode: _GMMMode,
    *,
    num_components: int,
    train_batches: int,
    train_batch_size: int,
    em_iters: int,
    covariance_reg: float,
    seed: int,
    verbose: bool,
) -> _NumpyGMMParameters:
    """Pool configured channel vectors and fit their shared real GMM."""
    # The EM initialization additionally uses a local NumPy generator.
    set_random_seed(seed)

    channel_source = UMiChannelSource(cfg, resource_grid)
    channel_model = channel_source.channel_model
    sampler = GenerateOFDMChannel(
        channel_model,
        resource_grid,
        precision=cfg.precision,
        device=cfg.device,
    )
    # Every selected vector from every group and realization shares one prior.
    pooled_samples = []

    for batch_index in range(train_batches):
        topology = gen_single_sector_topology(
            batch_size=train_batch_size,
            num_ut=cfg.num_ue,
            scenario="umi",
            min_ut_velocity=cfg.ue_speed,
            max_ut_velocity=cfg.ue_speed,
            precision=cfg.precision,
            device=cfg.device,
        )
        channel_model.set_topology(*topology)
        channel = sampler(train_batch_size)
        for group in groups:
            group_samples = _group_channel_samples(channel, group, mode)
            pooled_samples.append(group_samples.detach().cpu().numpy())

        if verbose and (batch_index + 1) % max(1, train_batches // 4) == 0:
            print(
                f"  {mode} GMM training channel batch "
                f"{batch_index + 1}/{train_batches}",
            )

    # Stack real parts before imaginary parts, as recorded in compatibility.
    complex_samples = np.concatenate(pooled_samples, axis=0)
    real_samples = np.concatenate(
        [complex_samples.real, complex_samples.imag],
        axis=-1,
    ).astype(np.float64, copy=False)
    weights, means, covariances = _fit_gmm_real(
        real_samples,
        num_components,
        max_iter=em_iters,
        covariance_reg=covariance_reg,
        seed=seed,
        device=cfg.device,
        verbose=verbose,
    )

    return (
        weights.astype(np.float32),
        means.astype(np.float32),
        covariances.astype(np.float32),
    )


# Full-covariance EM implementation


def _fit_gmm_real(
    samples: np.ndarray,
    num_components: int,
    *,
    max_iter: int,
    covariance_reg: float,
    seed: int,
    device: Union[str, torch.device],
    verbose: bool = False,
) -> _NumpyGMMParameters:
    r"""Fit a full-covariance real GMM by Torch EM for Algorithm 1 line 1.

    :param samples: Real samples with shape ``[num_samples, dim]``.
    :param num_components: Number of Gaussian mixture components.
    :param max_iter: Maximum number of EM iterations.
    :param covariance_reg: Diagonal covariance regularization factor.
    :param seed: Random seed used for component initialization.
    :param device: Torch device used for the offline EM fit.
    :param verbose: If `True`, print one progress line per EM iteration.

    :output weights: Mixture weights with shape ``[num_components]``.
    :output means: Component means with shape ``[num_components, dim]``.
    :output covariances: Component covariance matrices with shape
        ``[num_components, dim, dim]``.
    """
    if samples.ndim != 2 or samples.shape[0] < 1:
        raise ValueError("GMM fitting samples must be a nonempty matrix.")
    num_samples, dim = samples.shape
    num_components = min(int(num_components), int(num_samples))
    rng = np.random.default_rng(seed)

    # Float32 matches the persisted inference parameters and makes the large
    # full-covariance spectral fits practical on the configured GPU.
    samples_tensor = torch.as_tensor(np.asarray(samples)).to(
        device=device,
        dtype=torch.float32,
    )
    with torch.no_grad():
        shared_mean = samples_tensor.mean(dim=0)
        centered_samples = samples_tensor - shared_mean
        sample_variance = centered_samples.square().mean(dim=0).mean()
        loading = max(float(covariance_reg * sample_variance.item()), 1e-7)
        shared_covariance = (
            centered_samples.mT @ centered_samples / float(num_samples)
        )
        shared_covariance = 0.5 * (
            shared_covariance + shared_covariance.mT
        )
        shared_covariance.diagonal().add_(loading)
        del centered_samples

        if num_components == 1:
            return (
                np.ones(1, dtype=np.float32),
                shared_mean[None].cpu().numpy(),
                shared_covariance[None].cpu().numpy(),
            )

        init_indices = torch.as_tensor(
            rng.choice(num_samples, size=num_components, replace=False),
            device=device,
            dtype=torch.int64,
        )
        means = samples_tensor.index_select(0, init_indices).clone()
        covariances = shared_covariance.expand(
            num_components,
            dim,
            dim,
        ).clone()
        weights = torch.full(
            (num_components,),
            1.0 / num_components,
            device=device,
            dtype=torch.float32,
        )
        previous_log_likelihood = -math.inf
        normalizer = dim * math.log(2.0 * math.pi)

        for iteration in range(max_iter):
            log_probabilities = torch.empty(
                (num_samples, num_components),
                device=device,
                dtype=torch.float32,
            )
            for component_index in range(num_components):
                covariance = covariances[component_index]
                cholesky_factor, cholesky_info = torch.linalg.cholesky_ex(
                    covariance,
                )
                if cholesky_info.item() != 0:
                    covariance = covariance.clone()
                    extra_loading = loading
                    added_loading = 0.0
                    for _ in range(6):
                        covariance.diagonal().add_(extra_loading)
                        added_loading += extra_loading
                        cholesky_factor, cholesky_info = (
                            torch.linalg.cholesky_ex(covariance)
                        )
                        if cholesky_info.item() == 0:
                            covariances[component_index] = covariance
                            if verbose:
                                print(
                                    "  Added adaptive covariance loading "
                                    f"{added_loading:.6e} to component "
                                    f"{component_index + 1}",
                                    flush=True,
                                )
                            break
                        extra_loading *= 10.0
                    else:
                        raise RuntimeError(
                            "GMM covariance remained non-positive-definite "
                            "after adaptive diagonal loading."
                        )
                centered_samples = samples_tensor - means[component_index]
                whitened_samples = torch.linalg.solve_triangular(
                    cholesky_factor,
                    centered_samples.mT,
                    upper=False,
                )
                squared_mahalanobis = whitened_samples.square().sum(dim=0)
                log_determinant = 2.0 * torch.log(
                    cholesky_factor.diagonal()
                ).sum()
                log_probabilities[:, component_index] = (
                    torch.log(weights[component_index].clamp_min(1e-12))
                    - 0.5
                    * (normalizer + log_determinant + squared_mahalanobis)
                )
                del centered_samples, whitened_samples

            log_normalizers = torch.logsumexp(log_probabilities, dim=1)
            log_likelihood = float(log_normalizers.mean().item())
            responsibilities = torch.exp(
                log_probabilities - log_normalizers[:, None]
            )
            component_weights = responsibilities.sum(dim=0).clamp_min(1e-12)
            weights = component_weights / float(num_samples)
            means = (
                responsibilities.mT @ samples_tensor
            ) / component_weights[:, None]

            for component_index in range(num_components):
                centered_samples = samples_tensor - means[component_index]
                weighted_samples = centered_samples * torch.sqrt(
                    responsibilities[:, component_index:component_index + 1]
                )
                covariance = (
                    weighted_samples.mT @ weighted_samples
                    / component_weights[component_index]
                )
                covariance = 0.5 * (covariance + covariance.mT)
                covariance.diagonal().add_(loading)
                covariances[component_index] = covariance
                del centered_samples, weighted_samples

            if verbose:
                print(
                    f"  GMM EM iteration {iteration + 1}/{max_iter}: "
                    f"mean log likelihood {log_likelihood:.6e}",
                    flush=True,
                )
            if abs(log_likelihood - previous_log_likelihood) < 1e-4:
                break
            previous_log_likelihood = log_likelihood

        return (
            weights.cpu().numpy(),
            means.cpu().numpy(),
            covariances.cpu().numpy(),
        )
