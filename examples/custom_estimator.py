# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Minimal custom pilot-domain channel estimator."""

from __future__ import annotations

import torch
from sionna.phy.ofdm import BasePilotChannelEstimator, LSChannelEstimator


class ShrinkageLSChannelEstimator(BasePilotChannelEstimator):
    """Apply a unit-variance Gaussian shrinkage prior to pilot LS estimates."""

    def estimate_at_pilot_locations(
        self,
        y_pilots: torch.Tensor,
        no: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return posterior pilot estimates and their error variances."""

        # Reuse the LS implementation, which only requires the pilot pattern
        # provided by BasePilotChannelEstimator.
        h_ls, ls_err_var = (
            LSChannelEstimator.estimate_at_pilot_locations(
                self,
                y_pilots,
                no,
            )
        )
        gain = torch.reciprocal(1.0 + ls_err_var)
        h_hat = gain * h_ls
        err_var = torch.clamp(gain * ls_err_var, min=0.0)
        return h_hat, err_var
