# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Edfors et al. 1996 channel-estimation implementation."""

from .estimator import TruncatedChannelEstimator

__all__ = ["TruncatedChannelEstimator"]
