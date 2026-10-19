# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Persistent covariance assets for Sionna's built-in LMMSE estimator.

Follows the recipe in OFDM_MIMO_Detection.ipynb:
samples single-UE channel frequency responses from UMi under random topology,
accumulates H·H^H across the relevant dimension, and averages. The resulting
covariance matrices are passed to `LMMSEChannelEstimator(order='t-f-s')`.

The covariances depend on the BS/UE arrays and the UMi config. Optional
package-local assets live below this module's ``assets/`` directory; generated
assets live below an explicit external asset root.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch

from sionna.phy.channel import GenerateOFDMChannel, gen_single_sector_topology
from sionna.phy.ofdm import ResourceGrid

from sionna.bench.mimo_ofdm_chest.configs import SystemConfig
from sionna.bench.mimo_ofdm_chest.benchmarking import UMiChannelSource, set_random_seed
from sionna.bench.mimo_ofdm_chest.asset_store import (
    build_asset_manifest,
    external_asset_dir,
    load_asset_manifest,
    paper_asset_dir,
)
from sionna.bench.mimo_ofdm_chest.persistence import write_json_atomic, write_npy_atomic


__all__ = ["estimate_covariance", "load_covariance"]


_ASSET_TYPE = "lmmse-covariance"
_ARTIFACT_ID = "default"
_PAPER_SLUG = "builtin"
_PAYLOAD_NAMES = ("freq_cov.npy", "time_cov.npy", "space_cov.npy")
_TOPOLOGY_PARAMETERS = {
    "min_bs_ut_dist": 10.0,
    "isd": 200.0,
    "bs_height": 10.0,
    "min_ut_height": 1.5,
    "max_ut_height": 1.5,
    "indoor_probability": 0.8,
}


def load_covariance(
    cfg: SystemConfig,
    *,
    asset_root: Optional[Union[str, Path]] = None,
) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], Path]:
    r"""Load the prepared covariance matrices for built-in LMMSE.

    This function never estimates or writes an asset. Use
    :func:`estimate_covariance` with an explicit external root to generate one.

    :param cfg: System configuration defining channel, grid, and array
        compatibility.
    :param asset_root: Optional external asset root. If omitted, use the
        optional package-local covariance asset.

    :returns: Frequency, OFDM-time, and receive-space covariance matrices,
        followed by the selected ``manifest.json`` path.
    """
    cov_dir = _covariance_asset_dir(cfg, asset_root=asset_root)
    manifest_path = cov_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"No complete built-in LMMSE covariance asset at {cov_dir}. "
            "Generate it with `python -m "
            "sionna.bench.mimo_ofdm_chest.papers.builtin.run_benchmark --prepare-assets "
            "--estimator lmmse --asset-root <path>`, then evaluate it with "
            "`python -m sionna.bench.mimo_ofdm_chest.papers.builtin.run_benchmark "
            "--estimator lmmse --asset-root <path>`."
        )
    _load_covariance_manifest(manifest_path, cfg)
    print(f"  loading covariance matrices from {cov_dir}")
    covariance = (
        _load_covariance_matrix(
            cov_dir / "freq_cov.npy", "Frequency", cfg.fft_size
        ),
        _load_covariance_matrix(
            cov_dir / "time_cov.npy", "Time", cfg.num_ofdm_symbols
        ),
        _load_covariance_matrix(
            cov_dir / "space_cov.npy", "Space", cfg.num_rx_ant
        ),
    )
    return covariance, manifest_path


def estimate_covariance(
    cfg: SystemConfig,
    num_it: int = 1000,
    batch_size: int = 64,
    seed: Optional[int] = 42,
    *,
    asset_root: Union[str, Path],
    overwrite: bool = False,
) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], Path]:
    r"""Estimate and persist covariance matrices under an external asset root.

    The payload and manifest are first written to a temporary sibling
    directory. The completed directory is then published and reloaded through
    :func:`load_covariance`, so generated and package-local assets follow the
    same validation path.

    :param cfg: System configuration defining channel, grid, and array
        compatibility.
    :param num_it: Number of Monte Carlo channel batches used for estimation.
    :param batch_size: Number of channel realizations in each batch.
    :param seed: Explicit generation seed. Defaults to ``cfg.seed``.
    :param asset_root: Explicit external writable asset root.
    :param overwrite: If `True`, replace an existing selected asset.

    :returns: Reloaded covariance matrices and their ``manifest.json`` path.
    """
    seed = int(cfg.seed if seed is None else seed)
    compatibility = _covariance_compatibility(cfg)
    generation_recipe = _generation_recipe(num_it, batch_size, seed)
    cov_dir = _covariance_asset_dir(cfg, asset_root=asset_root)
    if cov_dir.exists() and not overwrite:
        raise FileExistsError(f"Covariance asset already exists at {cov_dir}.")

    print(
        "  estimating cov matrices "
        f"(iters={num_it}, batch={batch_size}, seed={seed}) → {cov_dir}"
    )
    t0 = time.time()
    freq_cov, time_cov, space_cov = _estimate_covariance_matrices(
        cfg,
        num_it,
        batch_size,
        seed,
    )
    print(f"  cov estimation done in {time.time()-t0:.1f}s")
    cov_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(
        tempfile.mkdtemp(prefix=f".{cov_dir.name}.", dir=cov_dir.parent)
    )
    try:
        for name, matrix in zip(
            _PAYLOAD_NAMES,
            (freq_cov, time_cov, space_cov),
        ):
            write_npy_atomic(temporary_dir / name, matrix)
        _write_covariance_manifest(
            temporary_dir,
            cfg,
            compatibility,
            generation_recipe,
        )
        _load_covariance_manifest(temporary_dir / "manifest.json", cfg)
        if cov_dir.exists():
            shutil.rmtree(cov_dir)
        os.replace(temporary_dir, cov_dir)
    finally:
        shutil.rmtree(temporary_dir, ignore_errors=True)

    return load_covariance(cfg, asset_root=asset_root)


def _covariance_compatibility(cfg: SystemConfig) -> Dict:
    """Return system facts required to reuse LMMSE covariances."""
    return {
        "carrier_frequency": cfg.carrier_frequency,
        "subcarrier_spacing": cfg.subcarrier_spacing,
        "fft_size": cfg.fft_size,
        "num_ofdm_symbols": cfg.num_ofdm_symbols,
        "cyclic_prefix_length": cfg.cyclic_prefix_length,
        "channel_model": "umi",
        "spec_version": cfg.channel_source_parameters["spec_version"],
        "topology": dict(_TOPOLOGY_PARAMETERS),
        "bs_num_rows": cfg.bs_num_rows,
        "bs_num_cols": cfg.bs_num_cols,
        "bs_polarization": cfg.bs_polarization,
        "bs_polarization_type": cfg.bs_polarization_type,
        "ue_polarization": cfg.ue_polarization,
        "ue_polarization_type": cfg.ue_polarization_type,
        "ue_antenna_pattern": cfg.ue_antenna_pattern,
        "bs_antenna_pattern": cfg.bs_antenna_pattern,
        "umi_o2i_model": cfg.umi_o2i_model,
        "direction": cfg.direction,
        "enable_pathloss": cfg.enable_pathloss,
        "enable_shadow_fading": cfg.enable_shadow_fading,
        "ue_speed": cfg.ue_speed,
    }


def _generation_recipe(
    num_it: int,
    batch_size: int,
    seed: int,
) -> Dict:
    """Return the Monte Carlo recipe used to estimate the covariances."""
    return {
        "num_it": num_it,
        "batch_size": batch_size,
        "seed": seed,
    }


def _covariance_asset_dir(
    cfg: SystemConfig,
    *,
    asset_root: Optional[Union[str, Path]] = None,
) -> Path:
    """Select the package-local or external LMMSE covariance directory."""
    if asset_root is None:
        return paper_asset_dir(cfg, _ASSET_TYPE, _ARTIFACT_ID, __file__)
    return external_asset_dir(
        cfg,
        _PAPER_SLUG,
        _ASSET_TYPE,
        _ARTIFACT_ID,
        asset_root=asset_root,
    )


def _covariance_metadata() -> Dict:
    """Map each covariance dimension to its payload file."""
    return {
        "matrices": {
            "frequency": "freq_cov.npy",
            "time": "time_cov.npy",
            "space": "space_cov.npy",
        }
    }


def _write_covariance_manifest(
    cov_dir: Path,
    cfg: SystemConfig,
    compatibility: Dict,
    generation_recipe: Dict,
) -> None:
    payloads = [Path(cov_dir) / name for name in (
        "freq_cov.npy", "time_cov.npy", "space_cov.npy"
    )]
    manifest = build_asset_manifest(
        cfg=cfg,
        asset_type=_ASSET_TYPE,
        artifact_id=_ARTIFACT_ID,
        compatibility=compatibility,
        generation_recipe=generation_recipe,
        payload_paths=payloads,
        metadata=_covariance_metadata(),
    )
    write_json_atomic(Path(cov_dir) / "manifest.json", manifest)


def _load_covariance_manifest(
    manifest_path: Path,
    cfg: SystemConfig,
) -> dict:
    """Load the global manifest envelope and validate covariance filenames."""
    manifest = load_asset_manifest(
        manifest_path,
        asset_dir=Path(manifest_path).parent,
        expected_asset_type=_ASSET_TYPE,
        expected_scenario_id=cfg.scenario_id,
        expected_artifact_id=_ARTIFACT_ID,
        expected_compatibility=_covariance_compatibility(cfg),
    )
    payload_names = tuple(record["path"] for record in manifest["files"])
    if set(payload_names) != set(_PAYLOAD_NAMES) or len(payload_names) != len(
        _PAYLOAD_NAMES
    ):
        raise ValueError(
            "LMMSE covariance manifest must list exactly "
            f"{', '.join(_PAYLOAD_NAMES)}."
        )
    if manifest.get("metadata") != _covariance_metadata():
        raise ValueError("LMMSE covariance metadata does not match.")
    return manifest


def _load_covariance_matrix(
    path: Path,
    name: str,
    size: int,
) -> np.ndarray:
    """Load one covariance matrix and validate its array contract."""
    matrix = np.load(path, allow_pickle=False)
    expected_shape = (size, size)
    if matrix.shape != expected_shape:
        raise ValueError(
            f"{name} covariance must have shape {expected_shape}; "
            f"found {matrix.shape}."
        )
    if matrix.dtype != np.complex64:
        raise ValueError(
            f"{name} covariance must have dtype complex64; found {matrix.dtype}."
        )
    return matrix


def _estimate_covariance_matrices(
    cfg: SystemConfig,
    num_it: int,
    batch_size: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate (freq, time, space) covariance matrices from UMi samples."""
    set_random_seed(seed)
    device = cfg.device
    fft_size = cfg.fft_size
    num_ofdm = cfg.num_ofdm_symbols
    num_rx_ant = cfg.num_rx_ant

    rg = ResourceGrid(
        num_ofdm_symbols=num_ofdm,
        fft_size=fft_size,
        subcarrier_spacing=cfg.subcarrier_spacing,
        cyclic_prefix_length=cfg.cyclic_prefix_length,
        precision=cfg.precision,
        device=cfg.device,
    )
    channel_source = UMiChannelSource(cfg, rg)
    channel_model = channel_source.channel_model
    sampler = GenerateOFDMChannel(
        channel_model,
        rg,
        precision=cfg.precision,
        device=cfg.device,
    )

    freq_cov = torch.zeros([fft_size, fft_size], dtype=torch.complex64, device=device)
    time_cov = torch.zeros([num_ofdm, num_ofdm], dtype=torch.complex64, device=device)
    space_cov = torch.zeros([num_rx_ant, num_rx_ant], dtype=torch.complex64, device=device)

    for it in range(num_it):
        topo = gen_single_sector_topology(
            batch_size=batch_size,
            num_ut=1,
            scenario="umi",
            **_TOPOLOGY_PARAMETERS,
            min_ut_velocity=cfg.ue_speed,
            max_ut_velocity=cfg.ue_speed,
            precision=cfg.precision,
            device=cfg.device,
        )
        channel_model.set_topology(*topo)
        h = sampler(batch_size)[:, 0, :, 0, 0]            # [B, Ant, T, F]

        h_ = h.permute(0, 1, 3, 2)                        # [B, Ant, F, T]
        freq_cov += (h_ @ h_.mH).mean(dim=(0, 1))
        time_cov += (h @ h.mH).mean(dim=(0, 1))
        h_ = h.permute(0, 2, 1, 3)                        # [B, T, Ant, F]
        space_cov += (h_ @ h_.mH).mean(dim=(0, 1))
        if (it + 1) % max(1, num_it // 5) == 0:
            print(f"    cov iter {it+1}/{num_it}")

    freq_cov /= float(num_ofdm * num_it)
    time_cov /= float(fft_size * num_it)
    space_cov /= float(fft_size * num_it)
    return freq_cov.cpu().numpy(), time_cov.cpu().numpy(), space_cov.cpu().numpy()
