# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Focused tests for the Lyu 2026 diagnostic SDP estimator."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from sionna.bench.mimo_ofdm_chest.benchmarking import build_resource_grid
from sionna.bench.mimo_ofdm_chest.benchmarking.communication_system import OFDMChannelEstimationSystem
from sionna.bench.mimo_ofdm_chest.papers.lyu2026.estimator import SparseDiffuseChannelEstimator
from test.mimo_ofdm_chest.system_variants import (
    assert_compiled_estimator_matches_eager,
    run_nmse_probe,
    system_variant_config,
)


SOLVER_SHAPES = ((6, 2), (8, 4), (10, 7))


class _CVXPYP2Reference:
    """Independent CVXPY reference for one reduced P2 vector problem."""

    ACCEPTED_STATUSES = {"optimal", "optimal_inaccurate"}

    @staticmethod
    def _hermitian_toeplitz(cp, iota_zero, iota_tail, size: int):
        """Build the Hermitian Toeplitz matrix in the P2 lift."""

        def diagonal_value(offset: int):
            return iota_zero if offset == 0 else iota_tail[offset - 1]

        rows = []
        for row_index in range(size):
            row = []
            for column_index in range(size):
                if row_index >= column_index:
                    row.append(diagonal_value(row_index - column_index))
                else:
                    row.append(cp.conj(diagonal_value(column_index - row_index)))
            rows.append(row)
        return cp.bmat(rows)

    @classmethod
    def solve(
        cls,
        observation: np.ndarray,
        diffuse_dictionary: np.ndarray,
        *,
        tau: float,
        diffuse_energy: float,
        solver: str,
        solver_eps: float,
        solver_max_iters: int,
    ) -> tuple[np.ndarray, str]:
        """Solve the reduced P2 SDP independently of the estimator code."""
        cp = pytest.importorskip("cvxpy")
        observation = np.asarray(observation, dtype=np.complex128)
        diffuse_dictionary = np.asarray(diffuse_dictionary, dtype=np.complex128)
        num_observations = int(observation.shape[0])
        num_taps = int(diffuse_dictionary.shape[1])

        iota_zero = cp.Variable(nonneg=True)
        iota_tail = cp.Variable(num_observations - 1, complex=True)
        sparse_response = cp.Variable(num_observations, complex=True)
        diffuse_coefficients = cp.Variable(num_taps, complex=True)
        lift_scalar = cp.Variable(nonneg=True)
        toeplitz_matrix = cls._hermitian_toeplitz(
            cp,
            iota_zero,
            iota_tail,
            num_observations,
        )
        psd_block = cp.bmat(
            [
                [
                    toeplitz_matrix,
                    cp.reshape(
                        sparse_response,
                        (num_observations, 1),
                        order="C",
                    ),
                ],
                [
                    cp.reshape(
                        cp.conj(sparse_response),
                        (1, num_observations),
                        order="C",
                    ),
                    cp.reshape(lift_scalar, (1, 1), order="C"),
                ],
            ]
        )
        residual = observation - sparse_response - (
            diffuse_dictionary @ diffuse_coefficients
        )
        problem = cp.Problem(
            cp.Minimize(
                cp.sum_squares(cp.abs(residual))
                + tau * (lift_scalar + iota_zero)
            ),
            [
                psd_block >> 0,
                cp.sum_squares(cp.abs(diffuse_coefficients)) <= diffuse_energy,
            ],
        )
        solve_kwargs = {"solver": solver, "verbose": False}
        if solver == "SCS":
            solve_kwargs.update({"eps": solver_eps, "max_iters": solver_max_iters})
        elif solver == "CLARABEL":
            solve_kwargs.update(
                {
                    "tol_gap_abs": solver_eps,
                    "tol_gap_rel": solver_eps,
                    "tol_feas": solver_eps,
                    "max_iter": solver_max_iters,
                }
            )
        problem.solve(**solve_kwargs)

        status = str(problem.status)
        if (
            status not in cls.ACCEPTED_STATUSES
            or sparse_response.value is None
            or diffuse_coefficients.value is None
        ):
            return observation, status
        estimate = np.asarray(sparse_response.value, dtype=np.complex128) + (
            diffuse_dictionary
            @ np.asarray(diffuse_coefficients.value, dtype=np.complex128)
        )
        return estimate, status


def _diffuse_dictionary(num_observations: int, num_taps: int) -> np.ndarray:
    frequency_indices = np.arange(num_observations) - (num_observations - 1.0) / 2.0
    tap_indices = np.arange(num_taps)
    return np.exp(
        1j
        * 2.0
        * np.pi
        * frequency_indices[:, None]
        * tap_indices[None, :]
        / float(num_taps)
    )


def _mixed_synthetic_case(
    num_observations: int = 12,
    num_taps: int = 6,
):
    rng = np.random.default_rng(42)
    frequency_indices = np.arange(num_observations) - (num_observations - 1.0) / 2.0
    diffuse_dictionary = _diffuse_dictionary(num_observations, num_taps)
    sparse_frequency = 0.23
    sparse_component = 1.2 * np.exp(
        1j * 2.0 * np.pi * frequency_indices * sparse_frequency
    )
    diffuse_coefficients = (
        rng.normal(size=num_taps) + 1j * rng.normal(size=num_taps)
    )
    diffuse_coefficients *= 0.15 / np.sqrt(num_taps)
    channel = sparse_component + diffuse_dictionary @ diffuse_coefficients
    noise = 0.05 * (
        rng.normal(size=num_observations) + 1j * rng.normal(size=num_observations)
    )
    return channel + noise, channel, diffuse_dictionary


def _random_reduced_p2_case(
    rng: np.random.Generator,
    num_observations: int,
    num_taps: int,
):
    diffuse_dictionary = _diffuse_dictionary(num_observations, num_taps)
    frequency_indices = np.arange(num_observations) - (num_observations - 1.0) / 2.0

    sparse_component = np.zeros(num_observations, dtype=np.complex128)
    for _ in range(int(rng.integers(1, 3))):
        sparse_frequency = rng.uniform(-0.45, 0.45)
        sparse_gain = (rng.normal() + 1j * rng.normal()) / np.sqrt(2.0)
        sparse_component += sparse_gain * np.exp(
            1j * 2.0 * np.pi * frequency_indices * sparse_frequency
        )

    diffuse_coefficients = (
        rng.normal(size=num_taps) + 1j * rng.normal(size=num_taps)
    )
    diffuse_coefficients *= rng.uniform(0.02, 0.12) / np.sqrt(num_taps)
    channel = sparse_component + diffuse_dictionary @ diffuse_coefficients
    noise_sigma = rng.uniform(0.0, 0.06)
    noise = noise_sigma * (
        rng.normal(size=num_observations)
        + 1j * rng.normal(size=num_observations)
    ) / np.sqrt(2.0)
    diffuse_energy = float(
        max(
            np.sum(np.abs(diffuse_coefficients) ** 2) * rng.uniform(1.2, 3.0),
            0.03,
        )
    )
    tau = float(rng.uniform(0.02, 0.12))
    return channel + noise, diffuse_dictionary, tau, diffuse_energy


@pytest.mark.system_variants
@pytest.mark.parametrize(
    (
        "system_variant",
        "expected_comb_spacing",
        "expected_num_ue",
        "expected_num_rx_ant",
    ),
    (
        pytest.param("baseline", 1, 1, 2, id="spacing1-ue1-rx2"),
        pytest.param("alternate", 3, 3, 4, id="spacing3-ue3-rx4"),
        pytest.param("wide", 4, 4, 12, id="spacing4-ue4-rx12"),
    ),
)
def test_estimator_runs_across_supported_system_variants(
    system_variant: str,
    expected_comb_spacing: int,
    expected_num_ue: int,
    expected_num_rx_ant: int,
) -> None:
    """Full inference supports changed uniform combs, grids, UEs, and arrays."""
    cfg = system_variant_config(system_variant)
    resource_grid = build_resource_grid(cfg)
    estimator = SparseDiffuseChannelEstimator(
        resource_grid,
        diffuse_num_taps=2,
        tau=0.1,
        tau_noise_scale=None,
        diffuse_energy=2.0,
        admm_rho=0.1,
        admm_num_iters=2,
        precision=cfg.precision,
        device=cfg.device,
    )

    pilot_subcarrier_indices = estimator._pilot_group_varying_coord(
        "subcarrier"
    )
    subcarrier_spacings = torch.diff(pilot_subcarrier_indices, dim=-1)
    assert bool(torch.all(subcarrier_spacings == expected_comb_spacing))
    assert cfg.num_ue == expected_num_ue
    assert cfg.num_rx_ant == expected_num_rx_ant

    system = OFDMChannelEstimationSystem(cfg, estimator, resource_grid)
    result = run_nmse_probe(system, snr_db=5.0)

    assert 0.0 <= result.nmse < result.zero_estimate_nmse


@pytest.mark.compile
@pytest.mark.parametrize("system_variant", ["baseline", "alternate", "wide"])
def test_complete_estimator_compiles_as_full_graph(system_variant: str) -> None:
    """The complete batched ADMM and interpolation path reaches Tier A."""
    cfg = system_variant_config(system_variant)
    resource_grid = build_resource_grid(cfg)
    estimator = SparseDiffuseChannelEstimator(
        resource_grid,
        diffuse_num_taps=2,
        tau=0.1,
        tau_noise_scale=None,
        diffuse_energy=2.0,
        admm_rho=0.1,
        admm_num_iters=1,
        precision=cfg.precision,
        device=cfg.device,
    )
    received_grid = torch.randn(
        1,
        1,
        cfg.num_rx_ant,
        cfg.num_ofdm_symbols,
        cfg.fft_size,
        dtype=estimator.cdtype,
    )
    noise_variance = torch.full(
        (1, 1, cfg.num_rx_ant),
        0.1,
        dtype=estimator.dtype,
    )

    assert_compiled_estimator_matches_eager(
        estimator,
        received_grid,
        noise_variance,
        fullgraph=True,
    )


def test_repeated_pilot_symbols_are_averaged_per_link() -> None:
    """ADMM receives one Equation (12) average per link."""
    cfg = system_variant_config("alternate")
    estimator = SparseDiffuseChannelEstimator(
        build_resource_grid(cfg),
        diffuse_num_taps=2,
        admm_num_iters=1,
        precision=cfg.precision,
        device=cfg.device,
    )
    generator = torch.Generator(device=cfg.device).manual_seed(123)
    shape = (1, 1, cfg.num_rx_ant, estimator._pilot_group_num_groups, 4)
    grouped_ls = torch.complex(
        torch.randn(shape, generator=generator, device=cfg.device),
        torch.randn(shape, generator=generator, device=cfg.device),
    ).to(estimator.cdtype)
    estimator._solve_grouped_admm = lambda observations, tau: observations

    estimates = estimator._average_and_solve_pilot_groups(
        grouped_ls,
        torch.tensor(0.1, dtype=estimator.dtype, device=cfg.device),
    )
    group_tx_indices = estimator._pilot_group_constant_coord("tx")
    group_stream_indices = estimator._pilot_group_constant_coord("stream")
    for tx_index in range(cfg.num_ue):
        group_indices = torch.nonzero(
            torch.logical_and(
                group_tx_indices == tx_index,
                group_stream_indices == 0,
            ),
            as_tuple=False,
        ).flatten()
        expected = grouped_ls.index_select(3, group_indices).mean(dim=3)
        actual = estimates.index_select(3, group_indices)
        torch.testing.assert_close(
            actual,
            expected.unsqueeze(3).expand_as(actual),
        )


def test_repeated_pilot_averaging_reduces_error_variance() -> None:
    """Error variance includes the noise reduction from Equation (12)."""
    cfg = system_variant_config("alternate")
    estimator = SparseDiffuseChannelEstimator(
        build_resource_grid(cfg),
        diffuse_num_taps=2,
        err_var_scale=2.0,
        admm_num_iters=1,
        precision=cfg.precision,
        device=cfg.device,
    )
    pilot_power = estimator._group_pilot_power[0, 0]
    torch.testing.assert_close(
        estimator._group_pilot_power,
        torch.full_like(estimator._group_pilot_power, pilot_power),
    )
    num_groups_per_link = estimator._group_averaging_weights.count_nonzero(
        dim=-1
    )
    assert bool(torch.all(num_groups_per_link == num_groups_per_link[0]))

    noise_variance = torch.tensor(0.3, dtype=estimator.dtype, device=cfg.device)
    grouped_variance = estimator._grouped_error_variance(
        noise_variance,
        batch_size=1,
        num_rx=1,
        num_rx_ant=cfg.num_rx_ant,
        device=torch.device(cfg.device),
    )
    expected = 2.0 * noise_variance / (
        num_groups_per_link[0] * pilot_power
    )
    torch.testing.assert_close(
        grouped_variance,
        torch.full_like(grouped_variance, expected),
    )


@pytest.mark.parametrize(("num_observations", "num_taps"), SOLVER_SHAPES)
def test_toeplitz_p2_recovers_diffuse_only_vector(
    num_observations: int,
    num_taps: int,
):
    """The reduced SDP can represent a feasible diffuse-only channel."""
    rng = np.random.default_rng(100 + 10 * num_observations + num_taps)
    diffuse_dictionary = _diffuse_dictionary(num_observations, num_taps)
    diffuse_coefficients = 0.1 * (
        rng.normal(size=num_taps) + 1j * rng.normal(size=num_taps)
    ) / np.sqrt(num_taps)
    observation = diffuse_dictionary @ diffuse_coefficients
    diffuse_energy = float(np.sum(np.abs(diffuse_coefficients) ** 2) * 1.2)

    estimate, status = _CVXPYP2Reference.solve(
        observation,
        diffuse_dictionary,
        tau=0.1,
        diffuse_energy=diffuse_energy,
        solver="SCS",
        solver_eps=1e-4,
        solver_max_iters=3000,
    )

    relative_error = np.linalg.norm(estimate - observation) / np.linalg.norm(
        observation
    )
    assert status in _CVXPYP2Reference.ACCEPTED_STATUSES
    assert relative_error < 5e-2


def test_admm_tracks_cvxpy_on_small_mixed_case():
    """ADMM remains close to CVXPY on a small mixed reference case."""
    observation, channel, diffuse_dictionary = _mixed_synthetic_case()
    cvx_estimate, cvx_status = _CVXPYP2Reference.solve(
        observation,
        diffuse_dictionary,
        tau=0.05,
        diffuse_energy=2.0,
        solver="SCS",
        solver_eps=1e-3,
        solver_max_iters=500,
    )
    admm_estimate = SparseDiffuseChannelEstimator._solve_admm(
        torch.as_tensor(observation, dtype=torch.complex128),
        torch.as_tensor(diffuse_dictionary, dtype=torch.complex128),
        tau=0.05,
        diffuse_energy=2.0,
        rho=0.03,
        num_iters=300,
    )
    admm_estimate = admm_estimate.detach().cpu().numpy()

    ls_nmse = np.sum(np.abs(observation - channel) ** 2) / np.sum(
        np.abs(channel) ** 2
    )
    admm_nmse = np.sum(np.abs(admm_estimate - channel) ** 2) / np.sum(
        np.abs(channel) ** 2
    )
    admm_relative_to_cvx = np.linalg.norm(
        admm_estimate - cvx_estimate
    ) / np.linalg.norm(cvx_estimate)

    assert cvx_status in _CVXPYP2Reference.ACCEPTED_STATUSES
    assert admm_nmse < ls_nmse
    assert admm_relative_to_cvx < 0.01


@pytest.mark.parametrize(
    ("num_observations", "num_taps", "num_cases"),
    ((6, 2, 17), (8, 4, 17), (10, 7, 16)),
)
def test_admm_tracks_cvxpy_across_random_reduced_p2_cases(
    num_observations: int,
    num_taps: int,
    num_cases: int,
):
    """ADMM remains close to CVXPY across small randomized reduced SDPs."""
    try:
        import cvxpy as cp
    except ImportError:
        pytest.skip("CVXPY is required for the Lyu 2026 reference comparison.")
    if "CLARABEL" not in cp.installed_solvers():
        pytest.skip("Clarabel is required for the randomized CVXPY comparison.")

    rng = np.random.default_rng(123 + 10 * num_observations + num_taps)
    solution_relative_errors = []
    solution_norm_ratios = []
    for _ in range(num_cases):
        observation, diffuse_dictionary, tau, diffuse_energy = (
            _random_reduced_p2_case(
                rng,
                num_observations,
                num_taps,
            )
        )
        cvx_estimate, cvx_status = _CVXPYP2Reference.solve(
            observation,
            diffuse_dictionary,
            tau=tau,
            diffuse_energy=diffuse_energy,
            solver="CLARABEL",
            solver_eps=1e-6,
            solver_max_iters=10000,
        )
        admm_estimate = SparseDiffuseChannelEstimator._solve_admm(
            torch.as_tensor(observation, dtype=torch.complex128),
            torch.as_tensor(diffuse_dictionary, dtype=torch.complex128),
            tau=tau,
            diffuse_energy=diffuse_energy,
            rho=0.03,
            num_iters=300,
        )
        admm_estimate = admm_estimate.detach().cpu().numpy()

        assert cvx_status in _CVXPYP2Reference.ACCEPTED_STATUSES
        solution_relative_errors.append(
            np.linalg.norm(admm_estimate - cvx_estimate)
            / max(np.linalg.norm(cvx_estimate), 1e-30)
        )
        solution_norm_ratios.append(
            np.linalg.norm(admm_estimate) / max(np.linalg.norm(cvx_estimate), 1e-30)
        )
    assert max(solution_relative_errors) < 1e-3
    assert min(solution_norm_ratios) > 0.99
    assert max(solution_norm_ratios) < 1.01
