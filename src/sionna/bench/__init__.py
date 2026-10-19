# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Paper-derived channel estimators for Sionna."""

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .mimo_ofdm_chest.papers.balevi2020 import DeepPriorChannelEstimator
    from .mimo_ofdm_chest.papers.baur2022 import VAEChannelEstimator
    from .mimo_ofdm_chest.papers.coleri2002 import LSCombEstimator
    from .mimo_ofdm_chest.papers.edfors1996 import TruncatedChannelEstimator
    from .mimo_ofdm_chest.papers.koller2022 import (
        SpatialGMMChannelEstimator,
        SpectralGMMChannelEstimator,
    )
    from .mimo_ofdm_chest.papers.li2020 import ReEsNetChannelEstimator
    from .mimo_ofdm_chest.papers.lyu2026 import SparseDiffuseChannelEstimator
    from .mimo_ofdm_chest.papers.melgar2022 import (
        DNNChannelEstimator,
        InterpolatedDNNChannelEstimator,
    )
    from .mimo_ofdm_chest.papers.wei2017 import SBLChannelEstimator


_ESTIMATOR_MODULES = {
    "DeepPriorChannelEstimator": ".mimo_ofdm_chest.papers.balevi2020",
    "DNNChannelEstimator": ".mimo_ofdm_chest.papers.melgar2022",
    "InterpolatedDNNChannelEstimator": ".mimo_ofdm_chest.papers.melgar2022",
    "LSCombEstimator": ".mimo_ofdm_chest.papers.coleri2002",
    "ReEsNetChannelEstimator": ".mimo_ofdm_chest.papers.li2020",
    "SBLChannelEstimator": ".mimo_ofdm_chest.papers.wei2017",
    "SparseDiffuseChannelEstimator": ".mimo_ofdm_chest.papers.lyu2026",
    "SpatialGMMChannelEstimator": ".mimo_ofdm_chest.papers.koller2022",
    "SpectralGMMChannelEstimator": ".mimo_ofdm_chest.papers.koller2022",
    "TruncatedChannelEstimator": ".mimo_ofdm_chest.papers.edfors1996",
    "VAEChannelEstimator": ".mimo_ofdm_chest.papers.baur2022",
}

__all__ = sorted(_ESTIMATOR_MODULES)


def __getattr__(name: str) -> Any:
    """Load root-level estimator exports only when requested."""
    module_name = _ESTIMATOR_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value
