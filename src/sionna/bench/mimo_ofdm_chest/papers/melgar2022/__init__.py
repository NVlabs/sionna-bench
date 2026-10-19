# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Melgar et al. 2022 channel-estimation implementation."""

from .estimator import DNNChannelEstimator, InterpolatedDNNChannelEstimator

__all__ = ["DNNChannelEstimator", "InterpolatedDNNChannelEstimator"]
