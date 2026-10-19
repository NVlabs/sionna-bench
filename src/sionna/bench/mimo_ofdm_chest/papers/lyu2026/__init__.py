# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Lyu et al. 2026 channel-estimation implementation."""

from .estimator import SparseDiffuseChannelEstimator

__all__ = ["SparseDiffuseChannelEstimator"]
