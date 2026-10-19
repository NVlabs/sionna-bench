# Implementation Report: Lyu et al., 2026

## Implemented Scope

The implementation follows the human-approved diagnostic scope: apply Program
(P2) to LS-normalized pilot-comb observations treated as a reduced OFDM grid.
It is not the paper's full-grid HALS estimator.

`SparseDiffuseChannelEstimator` uses `_PilotGroupMixin` to gather uniform pilot
combs. It divides received pilots by their known values, averages repeated
pilot-symbol observations for each stream and receive antenna as in Equation
(12), solves one independent reduced P2 vector problem per antenna, and copies
that estimate to the corresponding pilot symbols. `BasePilotChannelEstimator` then
applies the configured Sionna interpolation.

The estimator uses a batched Torch ADMM approximation of the reduced Toeplitz
SDP. An independent CVXPY formulation is retained only in the test module as a
reference oracle; it is not an estimator backend.

The ADMM path preserves the Hermitian Toeplitz lift and diffuse-energy
constraint. It uses PSD projection and alternating closed-form updates. It is
an approximate solver and does not provide a conic optimality certificate.

## Implementation Notes

The fixed diffuse dictionary is registered as a buffer and follows the paper's
matrix `D` on the reduced comb. The Torch ADMM implementation batches all
independent vector problems over batch entries, receivers, antennas, and
streams. It does not couple receive antennas or assume common support across
them.

Repeated pilot symbols are averaged before solving. This is the direct use of
the paper's Equation (12) available under the benchmark's uplink observation
model. The resulting estimate is reused at every averaged pilot-bearing OFDM
symbol.

The estimator returns the ADMM reconstruction directly. It does not silently
replace non-finite or strongly amplified estimates with LS observations;
numerical failures therefore remain visible to callers and tests.

The group memberships used for Equation (12) averaging are fixed by the pilot
pattern and prepared during construction. Online inference applies registered
averaging weights and group-to-stream indices without Python loops.

The paper-local architecture follows `docs/paper_architecture.md`. The online
solver lives in `estimator.py`; `run_benchmark.py` exposes `run_benchmark()`;
and `tuning.py` exposes `run_tuning()`. No asset module is needed.

## Deviations

- Equations (5)-(6) define length-`N` observations over every subcarrier, and
  Equation (12) averages complete length-`N` LS observations. The paper has no
  sparse-pilot sampling or interpolation stage. This implementation therefore
  remains pilot-domain: it observes sparse pilot combs, treats each comb as a
  reduced OFDM grid, and delegates off-pilot values to explicit Sionna
  interpolation.
- The estimator returns the biased `h_s + D gamma` reconstruction. Continuous
  support extraction in Equation (16) and projection debiasing in Equation
  (17) are omitted.
- Off-pilot estimates use explicit Sionna interpolation because the paper does
  not define interpolation from sparse pilot combs.
- The uplink adaptation estimates each receive antenna independently.
- The canonical Torch ADMM solver is approximate. CVXPY is used only as a
  test-local reference oracle.
- Detector-facing error variances are not specified by the paper. For the
  Equation (12) average with weights `w_g`, the implementation returns
  `err_var_scale * no * sum_g w_g^2/|pilot_g|^2` before interpolation. This
  reduces to `err_var_scale * no/(G*|pilot|^2)` for `G` equal-power pilots.

## P2 and ADMM Audit (2026-08-03)

The reduced solver was reread against Programs (P2) and Equations (9), (12),
and (16)-(18). The implementation retains the complex sparse response and
diffuse coefficients, Hermitian Toeplitz first-column convention, lifted PSD
block, squared residual, `tau*(t+iota_1)` penalty, and
`||gamma||_2^2 <= E_d` constraint. The diffuse dictionary uses the centered
frequency indices and `a(-r/L)` sign convention from Equation (9).

The ADMM updates are the closed-form minimizers of the reduced P2 augmented
Lagrangian: Toeplitz diagonals are averaged with their multiplicities, the
sparse response and diffuse coefficients are minimized jointly, the diffuse
constraint is solved by a monotone Lagrange-multiplier search, and the lifted
matrix is projected onto the Hermitian PSD cone. Clamping `iota_1` and `t` to
nonnegative values is redundant at a feasible PSD solution but is retained as
a bounded numerical safeguard during the fixed-iteration solve. No LS fallback
or reconstruction-gain mask is present.
- The final calibrated-source v2 regeneration completed 80 ADMM trials,
  canonical-size paired validation, ten detector-scale trials, paired scale
  validation, and a canonical benchmark. The selected rounded parameters are
  now the estimator and runner defaults.

## Calibrated-Source V2 Regeneration Evidence (2026-08-15)

All runs imported Sionna from source commit
`f16208347246d8d17ec69628dfb856b19287b80c`, including calibration commit
`ad1609ea1`, with explicit UMi TR 38.901 version `19.2`. The solver study used
five SNRs `[-10, -5, 0, 5, 10]` dB, three batches of four channels per SNR,
sampler seed 123, and probe seed 4242. All 80 requested trials completed and
none failed. The 1 h 32 min 12 s study selected raw
`diffuse_num_taps=3`, `tau_noise_scale=6.5484840637`,
`diffuse_energy=0.8068198786`, `admm_rho=0.07996883845`, and
`admm_num_iters=18`, with weighted-NMSE objective `0.0058468221`.

Paired validation used seed 314159 and 63 batches of four channels at each of
the five SNRs. The former configuration scored `0.0085787708`; rounded
`(3, 6.55, 0.807, 0.08, 18)` scored `0.0061135085` and was selected. This
validation took 41 min 8 s.

With the rounded solver frozen, a cached ten-trial 10 dB codeword-BLER study
selected raw `err_var_scale=0.0055752934`. Cache preparation and the search
took 5 min 21 s. Independent seed-161803 paired validation gave identical
BLER `0.0201719577` for incumbent `1.0` and rounded `0.00558`, so the strict
tie rule retained `err_var_scale=1.0`. Scale validation took 5 min 11 s.

## Verification

The reduced solver tests cover `(num_observations, diffuse_num_taps)` values
`(6, 2)`, `(8, 4)`, and `(10, 7)`. Fifty randomized ADMM-versus-CVXPY cases are
distributed across those shapes. Additional checks cover diffuse-only
feasibility and a mixed sparse/diffuse case. The ADMM comparisons exercise the
Torch implementation directly in CPU complex128 precision.

End-to-end tests cover uniform comb spacings 1 and 3, one and two pilot-bearing
OFDM symbols, 1 and 3 UEs, and 2 and 4 BS antennas. They use a two-iteration
ADMM budget and verify finite, non-degenerate inference rather than canonical
performance.

The complete estimator was compared in eager execution and with
`torch.compile(..., backend="eager", fullgraph=True)` on both end-to-end
geometries. Outputs agree, and `torch._dynamo.explain` reports one graph and no
graph breaks for each geometry, establishing Tier A inference.

The current focused test command is:

```text
python -m pytest test/mimo_ofdm_chest/papers/lyu2026/test_estimator.py -q
```

Current result:

```text
14 passed
```

## Canonical Benchmark

```text
python -m sionna_bench.mimo_ofdm_chest.papers.lyu2026.run_benchmark
```

The canonical path uses `diffuse_num_taps=3`, `tau_noise_scale=6.55` bounded
to `[1e-6, 3.0]`, `diffuse_energy=0.807`, `admm_rho=0.08`, 18 ADMM
iterations, and `err_var_scale=1.0`. It took 57 min 41 s.

The canonical result is:

```text
Eb/N0 [dB]: -10.0, -7.5, -5.0, -2.5, 0.0, 2.5, 5.0, 7.5, 10.0, 12.5, 15.0
NMSE:       1.430708e-01, 7.314181e-02, 3.892052e-02, 2.246696e-02,
            1.451980e-02, 9.308526e-03, 6.081519e-03, 3.975991e-03,
            2.569161e-03, 1.650374e-03, 1.076263e-03
BER:        2.310776e-01, 1.776633e-01, 1.257451e-01, 8.013505e-02,
            4.610263e-02, 2.413602e-02, 1.208170e-02, 5.950283e-03,
            2.671339e-03, 9.240489e-04, 3.383063e-04
BLER:       9.639550e-01, 8.647487e-01, 7.192460e-01, 5.307540e-01,
            3.184524e-01, 1.818783e-01, 9.093915e-02, 4.464286e-02,
            2.116402e-02, 7.936508e-03, 2.976190e-03
```

Against the previous worktree output, five-SNR weighted NMSE changed from
`0.0053374035` to `0.0059030533` (+10.6%). NMSE improves below 0 dB and
regresses from 0 dB onward. BER and BLER improve at every SNR, including 33.2%
lower BER and 43.9% lower BLER at 10 dB. This disagreement with the paired
NMSE validation is retained transparently: the canonical benchmark was a
final test and was not reused to choose between parameter sets.

## Current Risks

- The canonical ADMM solve is deliberately early-stopped and has no convergence
  certificate.
- The reduced pilot-comb model differs materially from the paper's full-grid
  observation model.
- Omitting continuous support extraction and debiasing can retain shrinkage
  bias.
- Runtime remains high relative to conventional channel estimators.
