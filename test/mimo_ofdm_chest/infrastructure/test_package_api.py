# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the installed package's public estimator imports."""

from importlib import import_module

import sionna.bench


_ESTIMATOR_MODULES = {
    "DeepPriorChannelEstimator": "sionna.bench.mimo_ofdm_chest.papers.balevi2020",
    "DNNChannelEstimator": "sionna.bench.mimo_ofdm_chest.papers.melgar2022",
    "InterpolatedDNNChannelEstimator": "sionna.bench.mimo_ofdm_chest.papers.melgar2022",
    "LSCombEstimator": "sionna.bench.mimo_ofdm_chest.papers.coleri2002",
    "ReEsNetChannelEstimator": "sionna.bench.mimo_ofdm_chest.papers.li2020",
    "SBLChannelEstimator": "sionna.bench.mimo_ofdm_chest.papers.wei2017",
    "SparseDiffuseChannelEstimator": "sionna.bench.mimo_ofdm_chest.papers.lyu2026",
    "SpatialGMMChannelEstimator": "sionna.bench.mimo_ofdm_chest.papers.koller2022",
    "SpectralGMMChannelEstimator": "sionna.bench.mimo_ofdm_chest.papers.koller2022",
    "TruncatedChannelEstimator": "sionna.bench.mimo_ofdm_chest.papers.edfors1996",
    "VAEChannelEstimator": "sionna.bench.mimo_ofdm_chest.papers.baur2022",
}


def test_root_exports_match_paper_local_estimators() -> None:
    """Every documented convenience export resolves to its paper-owned class."""
    assert set(sionna.bench.__all__) == set(_ESTIMATOR_MODULES)
    for name, module_name in _ESTIMATOR_MODULES.items():
        root_estimator = getattr(sionna.bench, name)
        paper_estimator = getattr(import_module(module_name), name)
        assert root_estimator is paper_estimator
