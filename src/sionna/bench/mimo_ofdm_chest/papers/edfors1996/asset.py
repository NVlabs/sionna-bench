# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional, Tuple, Union

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
from sionna.bench.mimo_ofdm_chest.persistence import write_json_atomic, write_npz_atomic


__all__ = ["estimate_covariance", "load_covariance"]


_ASSET_TYPE = "delay-covariance"
_PAPER_SLUG = "edfors1996"
_PAYLOAD_NAME = "time_covariances.npz"


def load_covariance(
    resource_grid,
    cfg,
    *,
    cov_num_iters: int = 1000,
    cov_batch_size: int = 64,
    seed: int = 42,
    asset_root: Optional[Union[str, Path]] = None,
) -> Tuple[np.ndarray, Path]:
    r"""Load one prepared Edfors IDFT-domain covariance matrix.

    This function never estimates or writes an asset. With no ``asset_root``,
    it selects an optional package-local covariance. Otherwise, it selects the
    corresponding artifact beneath the external root.

    :param resource_grid: Resource grid defining the pilot geometry.
    :param cfg: Channel-estimation system configuration.
    :param cov_num_iters: Number of Monte Carlo batches in the selected asset.
    :param cov_batch_size: Channel batch size used by the selected asset.
    :param seed: Generation seed recorded by the selected asset.
    :param asset_root: Optional external asset root.

    :returns: IDFT-domain covariance matrix of shape
        ``[num_pilot_subcarriers, num_pilot_subcarriers]`` and the selected
        ``manifest.json`` path.
    """
    group_subcarrier_indices = _PilotGroupLayout(
        resource_grid.pilot_pattern
    ).subcarrier_indices
    asset_dir = _asset_dir(
        cfg,
        cov_num_iters=cov_num_iters,
        cov_batch_size=cov_batch_size,
        seed=seed,
        asset_root=asset_root,
    )
    manifest_path = asset_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"No Edfors covariance asset at {asset_dir}. Generate it with "
            "`python -m sionna.bench.mimo_ofdm_chest.papers.edfors1996.run_benchmark "
            "--prepare-assets --asset-root <path>`, then evaluate it with "
            "`python -m sionna.bench.mimo_ofdm_chest.papers.edfors1996.run_benchmark "
            "--asset-root <path>`."
        )

    manifest = load_asset_manifest(
        manifest_path,
        asset_dir=asset_dir,
        expected_asset_type=_ASSET_TYPE,
        expected_scenario_id=cfg.scenario_id,
        expected_artifact_id=asset_dir.name,
        expected_compatibility=_compatibility(
            cfg,
            group_subcarrier_indices,
        ),
    )
    generation_recipe = _generation_recipe(
        cov_num_iters=cov_num_iters,
        cov_batch_size=cov_batch_size,
        seed=seed,
    )
    if manifest["generation_recipe"] != generation_recipe:
        raise ValueError(
            f"Edfors covariance generation recipe does not match at {asset_dir}."
        )
    if manifest.get("metadata") != _metadata():
        raise ValueError("Edfors covariance metadata does not match.")
    if [record["path"] for record in manifest["files"]] != [_PAYLOAD_NAME]:
        raise ValueError(
            f"Edfors covariance manifest must list exactly {_PAYLOAD_NAME}."
        )

    print(f"[edfors1996] loading DFT covariance asset from {asset_dir}")
    covariance = _load_payload(
        asset_dir / _PAYLOAD_NAME,
        num_pilot_subcarriers=int(group_subcarrier_indices.shape[1]),
    )
    return covariance, manifest_path


def estimate_covariance(
    resource_grid: ResourceGrid,
    cfg: SystemConfig,
    *,
    asset_root: Union[str, Path],
    cov_num_iters: int = 1000,
    cov_batch_size: int = 64,
    seed: int = 42,
    overwrite: bool = False,
) -> Tuple[np.ndarray, Path]:
    r"""Estimate and persist an IDFT-domain channel covariance.

    The function samples noiseless channels from the configured UMi channel
    model and gathers the active pilot subcarriers for every transmit stream
    and pilot-bearing OFDM symbol. Each resulting frequency-domain vector is
    transformed with a unitary IDFT. Their empirical second moments are then
    pooled into one covariance matrix,

    .. math::

        \mathbf{R}_{\tau}
        =
        \frac{1}{N}
        \sum_{n=1}^{N}
        \mathbf{h}_{\tau,n}\mathbf{h}_{\tau,n}^{\mathsf{H}}.

    Every realization contributes all compatible transmit-stream, receive-antenna,
    and pilot-symbol observations to the covariance estimate.

    The covariance, its diagonal tap powers, and a manifest describing the
    generation recipe and compatible pilot geometry are written atomically
    beneath the explicitly supplied external asset root. The completed asset
    is strictly reloaded and validated before being returned.

    .. rubric:: Deviations from the Paper

    - The paper derives the covariance from its analytical channel and
      power-delay-profile model. This function instead estimates it empirically
      from the configured Sionna UMi channel model.
    - The paper assumes observations on every OFDM subcarrier. This function
      estimates covariance from the active pilot-subcarrier vectors used by
      the implemented sparse-pilot adaptation.
    - One covariance is pooled over all compatible transmit streams,
      receive antennas, and pilot-bearing OFDM symbols rather than estimating
      a separate covariance for each group.

    :param resource_grid: Resource grid defining the OFDM dimensions, pilot
        pattern, and ordered pilot-subcarrier groups.
    :param cfg: System configuration defining the UMi channel, topology,
        antenna geometry, precision, and device.
    :param asset_root: External writable root beneath which the generated asset
        is stored.
    :param cov_num_iters: Number of independently generated Monte Carlo channel
        batches.
    :param cov_batch_size: Number of complete channel realizations generated
        in each Monte Carlo batch.
    :param seed: Random seed controlling topology and channel generation.
    :param overwrite: If `True`, atomically replace an existing artifact with
        the same identity. If `False`, raise :class:`FileExistsError`.

    :returns: The validated IDFT-domain covariance matrix with shape
        ``[num_pilot_subcarriers, num_pilot_subcarriers]`` and the path to its
        ``manifest.json`` file.
    """
    group_subcarrier_indices = _PilotGroupLayout(
        resource_grid.pilot_pattern
    ).subcarrier_indices
    asset_dir = _asset_dir(
        cfg,
        cov_num_iters=cov_num_iters,
        cov_batch_size=cov_batch_size,
        seed=seed,
        asset_root=asset_root,
    )
    if asset_dir.exists() and not overwrite:
        raise FileExistsError(
            f"Edfors covariance asset already exists at {asset_dir}."
        )

    print(f"[edfors1996] estimating DFT covariance asset -> {asset_dir}")
    covariance = _estimate_time_covariance(
        resource_grid,
        cfg,
        group_subcarrier_indices,
        cov_num_iters=cov_num_iters,
        cov_batch_size=cov_batch_size,
        seed=seed,
    )
    tap_powers = np.real(np.diagonal(covariance)).astype(np.float32)

    asset_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(
        tempfile.mkdtemp(prefix=f".{asset_dir.name}.", dir=asset_dir.parent)
    )
    try:
        payload_path = temporary_dir / _PAYLOAD_NAME
        write_npz_atomic(
            payload_path,
            time_covariances=covariance,
            tap_powers=tap_powers,
        )
        manifest = build_asset_manifest(
            cfg=cfg,
            asset_type=_ASSET_TYPE,
            artifact_id=asset_dir.name,
            compatibility=_compatibility(cfg, group_subcarrier_indices),
            generation_recipe=_generation_recipe(
                cov_num_iters=cov_num_iters,
                cov_batch_size=cov_batch_size,
                seed=seed,
            ),
            payload_paths=[payload_path],
            metadata=_metadata(),
        )
        write_json_atomic(temporary_dir / "manifest.json", manifest)
        if asset_dir.exists():
            shutil.rmtree(asset_dir)
        os.replace(temporary_dir, asset_dir)
    finally:
        shutil.rmtree(temporary_dir, ignore_errors=True)

    return load_covariance(
        resource_grid,
        cfg,
        cov_num_iters=cov_num_iters,
        cov_batch_size=cov_batch_size,
        seed=seed,
        asset_root=asset_root,
    )


class _PilotGroupLayout(_PilotGroupMixin, Object):
    """Expose shared pilot-group coordinates for covariance preparation."""

    def __init__(self, pilot_pattern) -> None:
        super().__init__(
            precision=pilot_pattern.precision,
            device=pilot_pattern.device,
        )
        self._init_pilot_groups(
            pilot_pattern=pilot_pattern,
            group_over=("subcarrier",),
            persistent=False,
        )

    @property
    def subcarrier_indices(self) -> torch.Tensor:
        """Return ordered pilot subcarriers for every covariance group."""
        return self._pilot_group_varying_coord("subcarrier").detach().cpu()


def _asset_dir(
    cfg,
    *,
    cov_num_iters: int,
    cov_batch_size: int,
    seed: int,
    asset_root: Optional[Union[str, Path]] = None,
) -> Path:
    """Select the package-local or external covariance artifact directory."""
    artifact_id = _artifact_id(cov_num_iters, cov_batch_size, seed)
    if asset_root is None:
        return paper_asset_dir(cfg, _ASSET_TYPE, artifact_id, __file__)
    return external_asset_dir(
        cfg,
        _PAPER_SLUG,
        _ASSET_TYPE,
        artifact_id,
        asset_root=asset_root,
    )


def _artifact_id(cov_num_iters: int, cov_batch_size: int, seed: int) -> str:
    """Return the human-readable covariance recipe identity."""
    return f"it{cov_num_iters}-bs{cov_batch_size}-seed{seed}"


def _compatibility(
    cfg,
    group_subcarrier_indices: torch.Tensor,
) -> dict[str, object]:
    """Return system and pilot facts required to reuse a covariance asset."""
    num_groups = int(group_subcarrier_indices.shape[0])
    group_len = int(group_subcarrier_indices.shape[1])
    return {
        "covariance_scope": "pooled_all_pilot_groups",
        "array_format": "single_matrix_2d",
        "channel_model": str(cfg.channel_type),
        "channel_source_parameters": dict(cfg.channel_source_parameters),
        "carrier_frequency": float(cfg.carrier_frequency),
        "subcarrier_spacing": float(cfg.subcarrier_spacing),
        "cyclic_prefix_length": int(cfg.cyclic_prefix_length),
        "fft_size": int(cfg.fft_size),
        "num_ofdm_symbols": int(cfg.num_ofdm_symbols),
        "num_ue": int(cfg.num_ue),
        "num_rx_ant": int(cfg.num_rx_ant),
        "bs_num_rows": int(cfg.bs_num_rows),
        "bs_num_cols": int(cfg.bs_num_cols),
        "bs_polarization": str(cfg.bs_polarization),
        "bs_polarization_type": str(cfg.bs_polarization_type),
        "bs_antenna_pattern": str(cfg.bs_antenna_pattern),
        "ue_polarization": str(cfg.ue_polarization),
        "ue_polarization_type": str(cfg.ue_polarization_type),
        "ue_antenna_pattern": str(cfg.ue_antenna_pattern),
        "umi_o2i_model": str(cfg.umi_o2i_model),
        "direction": str(cfg.direction),
        "enable_pathloss": bool(cfg.enable_pathloss),
        "enable_shadow_fading": bool(cfg.enable_shadow_fading),
        "ue_speed": float(cfg.ue_speed),
        "unitary_idft": "torch.fft.ifft(norm='ortho')",
        "num_groups": num_groups,
        "group_lengths": [group_len] * num_groups,
        "subcarrier_groups": [
            [int(value) for value in subcarriers.tolist()]
            for subcarriers in group_subcarrier_indices
        ],
    }


def _generation_recipe(
    *,
    cov_num_iters: int,
    cov_batch_size: int,
    seed: int,
) -> dict[str, int]:
    """Return the Monte Carlo recipe used to estimate a covariance asset."""
    return {
        "cov_num_iters": int(cov_num_iters),
        "cov_batch_size": int(cov_batch_size),
        "seed": int(seed),
    }


def _metadata() -> dict[str, object]:
    """Describe the semantic role of each NPZ array."""
    return {
        "arrays": {
            "time_covariances": "pooled IDFT-domain covariance",
            "tap_powers": "real covariance diagonal",
        }
    }


def _load_payload(path: Path, num_pilot_subcarriers: int) -> np.ndarray:
    """Load and validate the covariance and tap-power arrays."""
    with np.load(path, allow_pickle=False) as data:
        if set(data.files) != {"time_covariances", "tap_powers"}:
            raise ValueError("Edfors covariance asset has unexpected arrays.")
        covariance = data["time_covariances"]
        tap_powers = data["tap_powers"]

    expected_covariance_shape = (
        num_pilot_subcarriers,
        num_pilot_subcarriers,
    )
    if (
        covariance.shape != expected_covariance_shape
        or covariance.dtype != np.complex64
    ):
        raise ValueError(
            "Edfors covariance asset has incompatible covariance shape or dtype."
        )
    if (
        tap_powers.shape != (num_pilot_subcarriers,)
        or tap_powers.dtype != np.float32
    ):
        raise ValueError(
            "Edfors covariance asset has incompatible tap-power shape or dtype."
        )
    return covariance


def _estimate_time_covariance(
    resource_grid,
    cfg,
    group_subcarrier_indices: torch.Tensor,
    *,
    cov_num_iters: int,
    cov_batch_size: int,
    seed: int,
) -> np.ndarray:
    """Estimate one covariance pooled over all comb-pilot groups."""
    if group_subcarrier_indices.ndim != 2:
        raise ValueError("Edfors covariance expects rank-2 subcarrier groups.")

    set_random_seed(seed)
    num_pilot_subcarriers = int(group_subcarrier_indices.shape[1])
    channel_source = UMiChannelSource(cfg, resource_grid)
    channel_model = channel_source.channel_model
    sampler = GenerateOFDMChannel(
        channel_model,
        resource_grid,
        precision=cfg.precision,
        device=cfg.device,
    )
    covariance = torch.zeros(
        (num_pilot_subcarriers, num_pilot_subcarriers),
        dtype=torch.complex64,
        device=resource_grid.pilot_pattern.mask.device,
    )
    count = 0

    for iteration in range(cov_num_iters):
        topology = gen_single_sector_topology(
            batch_size=cov_batch_size,
            num_ut=cfg.num_ue,
            scenario="umi",
            min_ut_velocity=cfg.ue_speed,
            max_ut_velocity=cfg.ue_speed,
            precision=cfg.precision,
            device=cfg.device,
        )
        channel_model.set_topology(*topology)
        channel = sampler(cov_batch_size)

        for subcarriers in group_subcarrier_indices.to(channel.device):
            values = channel.index_select(-1, subcarriers).reshape(
                -1,
                num_pilot_subcarriers,
            )
            time_values = torch.fft.ifft(values, dim=-1, norm="ortho")
            covariance += time_values.transpose(0, 1) @ time_values.conj()
            count += int(time_values.shape[0])

        if (iteration + 1) % max(1, cov_num_iters // 4) == 0:
            print(f"  Edfors covariance batch {iteration + 1}/{cov_num_iters}")

    covariance = covariance / float(max(count, 1))
    covariance = 0.5 * (covariance + covariance.mH)
    return covariance.detach().cpu().numpy().astype(np.complex64)
