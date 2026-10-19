# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Global deterministic runtime for tests."""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch
from sionna.phy.config import config as sionna_config


_TEST_SEED = 42


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add opt-in test modes that require local accelerator toolchains."""
    parser.addoption(
        "--cuda-inductor",
        action="store_true",
        help="run optional CUDA/Inductor compilation smoke tests",
    )


@pytest.fixture(autouse=True)
def deterministic_runtime():
    """Seed test randomness and restore process-global state after each test."""
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    old_device = sionna_config.device
    old_precision = sionna_config.precision
    old_sionna_seed = sionna_config.seed

    random.seed(_TEST_SEED)
    np.random.seed(_TEST_SEED)
    torch.manual_seed(_TEST_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(_TEST_SEED)
    sionna_config.seed = _TEST_SEED
    sionna_config.device = "cpu"
    sionna_config.precision = "single"

    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
        sionna_config.seed = old_sionna_seed
        sionna_config.device = old_device
        sionna_config.precision = old_precision
