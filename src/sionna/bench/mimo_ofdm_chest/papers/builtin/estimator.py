# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Wrappers for Sionna's built-in OFDM channel estimators."""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import torch

from sionna.phy import Precision, config, dtypes
from sionna.phy.ofdm import (
    BaseChannelInterpolator,
    LMMSEChannelEstimator,
    LSChannelEstimator,
    ResourceGrid,
)


class LSEstimatorWrapper(LSChannelEstimator):
    """Least-squares estimator with configurable interpolation."""

    def __init__(
        self,
        resource_grid: ResourceGrid,
        interpolation_type: str = "nn",
        interpolator: Optional[BaseChannelInterpolator] = None,
        precision: Optional[Precision] = None,
        device: Optional[str] = None,
    ) -> None:
        if interpolator is None:
            LSChannelEstimator.__init__(
                self,
                resource_grid=resource_grid,
                interpolation_type=interpolation_type,
                precision=precision,
                device=device,
            )
        else:
            LSChannelEstimator.__init__(
                self,
                resource_grid=resource_grid,
                interpolator=interpolator,
                precision=precision,
                device=device,
            )


class LMMSEEstimatorWrapper(LMMSEChannelEstimator):
    """LMMSE channel estimator with configurable covariance matrices."""

    def __init__(
        self,
        resource_grid: ResourceGrid,
        freq_cov: Union[np.ndarray, torch.Tensor],
        time_cov: Union[np.ndarray, torch.Tensor],
        space_cov: Union[np.ndarray, torch.Tensor],
        order: str = "t-f-s",
        batch_chunk_size: Optional[int] = None,
        precision: Optional[Precision] = None,
        device: Optional[str] = None,
    ) -> None:
        if batch_chunk_size is not None and batch_chunk_size < 1:
            raise ValueError("`batch_chunk_size` must be positive or `None`.")

        selected_precision = config.precision if precision is None else precision
        covariance_dtype = dtypes[selected_precision]["torch"]["cdtype"]

        def as_complex_tensor(
            value: Union[np.ndarray, torch.Tensor],
        ) -> torch.Tensor:
            if isinstance(value, torch.Tensor):
                return value.to(dtype=covariance_dtype)
            return torch.as_tensor(value, dtype=covariance_dtype)

        super().__init__(
            resource_grid=resource_grid,
            cov_mat_time=as_complex_tensor(time_cov),
            cov_mat_freq=as_complex_tensor(freq_cov),
            cov_mat_space=as_complex_tensor(space_cov),
            order=order,
            precision=precision,
            device=device,
        )
        self._batch_chunk_size = batch_chunk_size

    def call(
        self,
        y: torch.Tensor,
        no: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply LMMSE estimation in memory-bounded batch chunks."""
        chunk_size = self._batch_chunk_size
        if chunk_size is None or y.shape[0] <= chunk_size:
            return super().call(y, no)

        channel_estimates = []
        error_variances = []
        for start in range(0, y.shape[0], chunk_size):
            stop = min(start + chunk_size, y.shape[0])
            chunk_noise = no
            if no.ndim > 0 and no.shape[0] == y.shape[0]:
                chunk_noise = no[start:stop]
            channel_estimate, error_variance = super().call(
                y[start:stop],
                chunk_noise,
            )
            channel_estimates.append(channel_estimate)
            error_variances.append(error_variance)
        return (
            torch.cat(channel_estimates, dim=0),
            torch.cat(error_variances, dim=0),
        )
