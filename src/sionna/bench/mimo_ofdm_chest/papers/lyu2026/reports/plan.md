# Implementation Plan: Lyu et al., 2026

This is the implementer plan for `papers/lyu2026`. No estimator or benchmark
runner code should be written until this plan has been reviewed in
`reports/plan_review.md` and the human has approved proceeding.

Revision note, 2026-06-15: after the initial review, the human requested an
attempt at a runnable P2-style method even if debiasing is initially omitted.
The human then clarified that P2 should be implemented through its Toeplitz
semidefinite lift rather than through a gridded sparse surrogate. This revision
therefore keeps the paper's exact full-grid P2 classified separately from a
new experimental pilot-comb Toeplitz-SDP adaptation.

## Harness Summary

The current task is channel estimation in a fixed Sionna OFDM MU-MIMO
benchmark. The relevant harness files are:

- `configs/system_config.py`: immutable `SystemConfig` schema and output slug
  construction.
- `configs/umi_3gpp_config.py`: canonical 3GPP UMi uplink configuration.
- `configs/smoke_config.py`: one-SNR, one-iteration smoke configuration.
- `evaluation/system.py`: builds the `ResourceGrid`, channel source, mapper,
  detector, LDPC chain, and calls the injected estimator.
- `evaluation/channel_estimation.py`: runs an SNR sweep, computes BER, BLER,
  and NMSE, and writes one `.npz` result file for each method.
- `evaluation/result_store.py`: stores paper-local outputs, config snapshots,
  and manifests.
- `src/pilot_groups.py`: provides `PilotGroupMixin` for grouping Sionna compact
  pilot vectors by physical pilot coordinates.

The benchmark estimator input is

```text
y  : [B, num_rx, num_rx_ant, num_ofdm_symbols, fft_size] complex
no : scalar or broadcastable noise variance
```

and the estimator output must be

```text
h_hat   : [B, num_rx, num_rx_ant, num_tx, num_streams_per_tx,
           num_ofdm_symbols, fft_size] complex
err_var : same shape as h_hat, float
```

The canonical UMi configuration is uplink with 4 single-stream UEs, a 4-by-4
single-polarization BS array, 14 OFDM symbols, 624 subcarriers, Kronecker
pilots on OFDM symbols 2 and 11, 16-QAM, code rate 0.60, and SNRs from -10 dB
to 15 dB in 2.5 dB steps. Built-in baseline outputs are under
`outputs/builtin/<config-slug>/`; paper outputs live under
`outputs/<paper_id>/<config-slug>/`. Smoke outputs use
`outputs/<paper_id>/smoke/<config-slug>/`.

First-pass paper implementations must preserve Sionna's pilot-domain
estimation pattern:

```text
pilot observations -> estimates at pilot locations
                   -> explicit interpolation/reconstruction
                   -> full resource-grid estimate
```

Existing mature implementations put the estimator in `estimator.py`, the runner
in `run_benchmark.py`, expose algorithm choices as constructor arguments, use
explicit interpolation, document benchmark adaptations, and use
`PilotGroupMixin` when pilot grouping is needed.

## Paper-To-Benchmark Mapping

Lyu et al. formulate an OFDM pilot model in Equations (6), (11), and (12).
After inverting the diagonal pilot matrices and averaging snapshots, the
estimator operates on

```text
y_tilde = A_f* alpha* + D gamma* + n_tilde,
```

a full length-`N` frequency-domain LS channel observation. Programs (P0), (P1),
and (P2) demix this full vector into a continuous-delay sparse component and a
diffuse tapped-delay component.

The repository benchmark does not expose `y_tilde` over all subcarriers. It
exposes sparse Kronecker pilot observations and expects a Sionna-compatible
estimator to return channel estimates at Sionna pilot locations before an
explicit interpolation/reconstruction step. A pilot-sampled version of HALS,
for example

```text
min 0.5 || y_pilot - M(h_s + D gamma) ||_2^2
    + tau ||h_s||_A + 0.5 lambda ||gamma||_2^2,
```

would be a different inverse problem from the paper's Programs (P1)/(P2). It
would require a missing-sample atomic norm formulation, new solver choices,
and new support-recovery rules. That is not a small Sionna interface
adaptation.

There is no separable paper component that only estimates pilot-location
channels and then leaves off-pilot interpolation to Sionna. The paper's core
contribution is the full-grid atomic sparse/diffuse reconstruction itself.

## Method Inventory

### Noiseless Atomic Sparse/Diffuse Demixing, P0

Status: `not implemented`

Paper location:

- Equations (13)-(14): atomic norm and atomic set.
- Program (P0): noiseless demixing objective.
- Program (P0, SDP): SDP form.
- Proposition 1: exact recovery under minimum-separation assumptions.

Method description:

P0 estimates the diffuse coefficients by minimizing the atomic norm of the
residual `y_tilde - D gamma`. The sparse component is then the residual itself.
The optimization assumes a full-frequency noiseless observation and is used for
exact-recovery analysis.

Paper inputs and outputs:

- Input: full-frequency `y_tilde`, known diffuse matrix `D`.
- Output: diffuse coefficients `gamma_hat`, sparse response
  `h_s_hat = y_tilde - D gamma_hat`.

Target repository abstraction:

- None for this pass. P0 is not a practical noisy Sionna pilot-domain
  estimator.

Reason not implemented:

- The method is noiseless and not the paper's benchmarked noisy estimator.
- A faithful implementation needs an SDP atomic-norm solver and full-frequency
  observations.
- Applying P0 to sparse Sionna pilot observations would change the objective
  and observation model.

What would be required:

- A reviewed full-grid or custom-interpolator design that explicitly supports
  atomic-norm reconstruction from the relevant observations.
- Solver infrastructure for the SDP representation of the atomic norm.
- Tests for exact recovery on synthetic full-grid noiseless channels satisfying
  the paper's separation conditions.

### HALS Penalized Noisy Estimator, P1

Status: `not implemented`

Paper location:

- Equation (11): full-frequency preprocessed observation model.
- Program (P1): noisy HALS objective with atomic and `l2` regularization.
- SDP reformulation following Program (P1).
- Proposition 2 and Corollary 1: optimality conditions and dual residual.
- Proposition 3: dual problem.
- Section III-D: hyperparameter selection for `tau` and `lambda`.

Method description:

P1 estimates a full-frequency sparse component `h_s` and diffuse coefficients
`gamma` by solving

```text
min_{h_s,gamma} 0.5 || y_tilde - h_s - D gamma ||_2^2
                 + tau ||h_s||_A + 0.5 lambda ||gamma||_2^2.
```

The atomic norm promotes a sparse continuous-delay representation, while the
`l2` term shrinks the diffuse tapped-delay coefficients.

Paper inputs and outputs:

- Input: full-frequency `y_tilde`, diffuse basis `D`, `tau`, and `lambda`.
- Output: biased estimates `h_s_hat`, `gamma_hat`, and
  `h_hat = h_s_hat + D gamma_hat`.

Target repository abstraction:

- Future candidate only: a custom `BaseChannelInterpolator` or reviewed
  equivalent that receives Sionna pilot-domain LS estimates and performs a
  paper-faithful reconstruction.

Reason not implemented:

- The estimator is a full-grid atomic-norm inverse problem, not a pilot-domain
  estimator with a separable built-in interpolation step.
- The repository's first-pass contract forbids bypassing Sionna pilot
  extraction with a direct raw-grid estimator.
- A sparse-pilot HALS variant would not be Program (P1); it would be a new
  missing-observation atomic-norm problem.
- The paper relies on CVX/off-the-shelf convex solvers. The repository has no
  solver infrastructure for per-sample semidefinite programs inside the Monte
  Carlo benchmark.
- The paper acknowledges that `tau` and `lambda` require fine-tuning. A
  faithful benchmark would need a documented validation sweep before canonical
  outputs are generated.
- The paper does not provide detector-facing error variances required by
  Sionna's receiver.

What would be required:

- A reviewed decision to add a full-grid reconstruction exception or a custom
  interpolator whose input/output contract can express HALS faithfully.
- A solver module, likely under `src/sionna_bench/mimo_ofdm_chest/papers/lyu2026/`, wrapping CVXPY/CVX-like SDP
  solves or another mathematically equivalent atomic-norm solver.
- A support-independent synthetic test suite for the SDP: zero-noise recovery,
  atomic-norm pass-through in simple one-path cases, and diffuse-only ridge
  limits.
- A bounded validation protocol for `tau`, `lambda`, and any Sionna
  `err_var_scale`, using validation data rather than final benchmark metrics.

### Paper Full-Grid Constrained and Debiased HALS, P2

Status: `not implemented`

Paper location:

- Program (P2): constrained HALS formulation with `||gamma||_2^2 <= E_d`.
- Equation (16): support extraction from the dual residual.
- Equation (17): debiased sparse-component projection.
- Equation (22): ridge-regression interpretation of diffuse estimation.
- Section V-A: reported HALS estimate `h_hat_HALS = h_s,db + D gamma_hat`.

Method description:

P2 replaces the diffuse `lambda` penalty by an energy constraint `E_d`. After
solving the convex program, the sparse support is estimated by locating
continuous frequencies where the dual polynomial reaches the threshold `tau`.
The sparse component is then debiased by projection onto the estimated
continuous steering-vector subspace.

Paper inputs and outputs:

- Input: full-frequency `y_tilde`, diffuse basis `D`, `tau`, and `E_d`.
- Output: `gamma_hat`, support set `T_hat`, debiased sparse response
  `h_s,db`, and full response `h_s,db + D gamma_hat`.

Target repository abstraction:

- Future candidate only: a custom reconstruction/interpolator component, not a
  Sionna `estimate_at_pilot_locations()` implementation.

Reason not implemented:

- P2 depends on the same full-grid SDP infrastructure as P1.
- Continuous support extraction from Equation (16) and debiasing by Equation
  (17) are central to the method and cannot be replaced by FFT bin selection
  without creating a surrogate estimator.
- Selecting `E_d` requires a prior or validation protocol for diffuse energy in
  the active channel scenario.
- The method still does not define detector-facing Sionna error variances.

What would be required:

- Everything required for P1.
- A robust dual-polynomial/root-search procedure for Equation (16), with tests
  for support recovery in one- and two-path synthetic cases.
- A documented `E_d` generation or tuning procedure, likely stored as an
  `aux/` calibration asset per scenario if it is estimated from the benchmark
  channel model.

### Pilot-Comb Toeplitz-SDP P2 Adaptation

Status: `implement`

Paper location:

- Equation (11): preprocessed pilot observation after dividing by known pilots.
- Equations (13)-(14): atomic norm and continuous complex sinusoid atomic set.
- Program (P2): Toeplitz SDP representation with diffuse-energy constraint.

Human-requested scope change:

This method is an experimental adaptation, not the exact paper setting. The
human requested trying a P2-style implementation under the current Sionna
benchmark by treating the pilot comb as a reduced OFDM grid: if every
`N_spacing`-th physical subcarrier carries a pilot, the estimator treats the
pilot vector as a frequency response sampled on a grid with effective
subcarrier spacing `N_spacing*SCS`. Full-grid reconstruction is then delegated
to Sionna's explicit interpolation, as in the existing RKHS first-pass
adaptation.

This adaptation does not replace the P2 objective by a finite-grid `l1`
surrogate. It solves the Toeplitz-lift SDP form of P2 on the reduced pilot-comb
grid. The continuous atomic structure enters through the Hermitian Toeplitz
PSD constraint. The sparse vector `h_s` is a free optimization variable only
in the ordinary SDP sense; it is coupled to the Toeplitz variable and scalar
lift through the PSD block constraint and penalized by `tau*(t+iota_1)`.

Method description:

For each fixed Sionna pilot group `(tx, stream, pilot OFDM symbol)`, gather the
LS-normalized pilot-channel vector over pilot subcarriers. Let its length be
`K`. Solve the reduced-grid P2 SDP

```text
min_{t,iota,h_s,gamma}
    || h_ls - h_s - D gamma ||_2^2 + tau*(t + iota_1)

subject to
    [ Toep(iota)  h_s ] >= 0
    [ h_s^H       t   ]

    ||gamma||_2^2 <= E_d.
```

Here `D gamma` is the diffuse tapped-delay component. `D` uses the same
steering-vector family as Equation (10), but on the reduced pilot-comb grid
rather than on all physical subcarriers. The estimator initially returns the
biased reconstruction `h_s + D gamma`; Equation (16) support extraction and
Equation (17) debiasing are intentionally omitted for the first diagnostic
implementation.

Target repository abstraction:

- Public class: `SparseDiffuseChannelEstimator`.
- File: `src/sionna_bench/mimo_ofdm_chest/papers/lyu2026/estimator.py`.
- Base class: `sionna.phy.ofdm.BaseChannelEstimator`.
- Mixin: `src.pilot_groups.PilotGroupMixin`.
- Pilot grouping: `_init_pilot_groups(group_over=("subcarrier",))`.
- Interpolation: explicit Sionna `"lin_time_avg"` by default.

Planned constructor:

```text
SparseDiffuseChannelEstimator(
    resource_grid,
    *,
    diffuse_num_taps=None,
    tau=0.1,
    diffuse_energy=1.0,
    err_var_scale=0.1,
    solver="SCS",
    solver_eps=1e-3,
    solver_max_iters=2000,
    interpolation_type="lin_time_avg",
    precision=None,
    device=None,
)
```

Constructor choices:

- `diffuse_num_taps`: number of diffuse grid columns in `D`. If `None`, use
  `K`, matching the reduced pilot-comb grid.
- `tau`: weight on the Toeplitz SDP atomic-norm lift term `t+iota_1`.
- `diffuse_energy`: the P2-style upper bound `E_d`.
- `err_var_scale`: standard Sionna detector-facing confidence scale applied to
  the LS observation variance fallback.
- `solver`, `solver_eps`, and `solver_max_iters`: CVXPY solver controls. The
  current local Sionna environment has CVXPY, SCS, and Clarabel installed. The
  estimator should import CVXPY lazily and raise a clear error if it is not
  available.
- `interpolation_type`: explicit Sionna interpolation type.

Implementation pseudocode:

```text
1. In __init__:
   a. initialize BaseChannelEstimator with explicit interpolation_type;
   b. initialize PilotGroupMixin over subcarriers;
   c. validate uniform pilot-comb groups and nonzero pilots;
   d. build and register D with shape [K, diffuse_num_taps] on the
      reduced pilot-comb grid.

2. In estimate_at_pilot_locations(y_pilots, no):
   a. gather pilot observations and pilot symbols by pilot group;
   b. form LS pilot observations h_ls = y_pilots / pilot_symbols;
   c. for each batch/rx/antenna/group vector, solve the CVXPY problem:
      i.   variables: h_s in C^K, gamma in C^L, iota defining a
           K-by-K Hermitian Toeplitz matrix, and scalar t >= 0;
      ii.  objective:
           ||h_ls - h_s - D gamma||_2^2 + tau*(t+iota_1);
      iii. constraints:
           [[Toep(iota), h_s], [h_s^H, t]] is Hermitian PSD,
           ||gamma||_2^2 <= diffuse_energy;
      iv.  return h_s + D gamma if the solver succeeds, otherwise mark the
           vector as failed and use the LS input only as an explicitly reported
           fallback for diagnostic runs.
   d. set grouped_err_var = err_var_scale * no / |pilot|^2;
   e. scatter grouped_h_hat and grouped_err_var back to Sionna's compact
      pilot-vector layout;
   f. let BaseChannelEstimator call the explicit Sionna interpolator.
```

Interface adaptations and deviations:

- This solves the Toeplitz SDP form of P2 on the reduced pilot-comb grid, not
  on the paper's full-subcarrier `y_tilde` vector.
- This uses the current pilot comb as the frequency grid. It is equivalent to
  pretending that the pilot subcarrier spacing is the physical spacing times
  the comb stride, as requested by the human.
- The first diagnostic version omits Equation (16) and Equation (17)
  debiasing. A future version can add gridded support selection and least
  squares debiasing if the biased reconstruction is promising.
- Hyperparameters `tau`, `diffuse_energy`, solver tolerances, and
  `err_var_scale` will be tuned empirically for an initial run. They are
  constructor arguments and must be recorded in the implementation report and
  README.
- The paper does not define detector-facing Sionna error variances, so this
  method uses the repository's standard scaled LS observation-variance
  fallback.
- The implementation is CPU/CVXPY based and not torch-compilable. It is
  expected to be slow: a single synthetic SDP at `K=64` took several seconds
  with SCS in the local environment. The current UMi pilot-group length is
  larger, and a full Monte Carlo batch contains many independent vectors.
  Initial runs should therefore be treated as feasibility diagnostics, not
  canonical benchmark generation.

Auxiliary assets:

- None for the first diagnostic implementation. Hyperparameters are passed as
  runner arguments rather than cached as canonical assets.

Initial verification:

- A small synthetic CVXPY solve that verifies the Toeplitz SDP is constructed
  and solved successfully.
- A tiny Sionna integration run if runtime permits, potentially with a reduced
  diagnostic configuration rather than the standard smoke configuration.
- Log the exact hyperparameters and solver settings used for every diagnostic
  run.

### Vanilla Atomic Norm Minimization Baseline

Status: `not implemented`

Paper location:

- Section V-A comparison against vanilla ANM.
- Statement after Equation (17) that ANM is recovered for purely sparse
  channels, for example by setting the diffuse component to zero.

Method description:

ANM is the sparse-only special case that ignores the diffuse term. It is used
as a baseline in the paper, not as the proposed hybrid estimator.

Paper inputs and outputs:

- Input: full-frequency `y_tilde` and `tau`.
- Output: sparse-only full-frequency channel estimate.

Target repository abstraction:

- Future custom reconstruction component if ANM is intentionally benchmarked.

Reason not implemented:

- It requires the same atomic-norm SDP and full-frequency observation model as
  HALS.
- It is a paper baseline rather than the proposed method.
- Implementing it from sparse pilots would be a different compressed-sensing
  problem.

What would be required:

- The same atomic-norm solver infrastructure as P1/P2.
- A reviewer-approved reason to include the paper baseline in the leaderboard
  in addition to existing Sionna baselines.

### LS and Genie-Aided Comparators

Status: `not implemented`

Paper location:

- Section V-A defines `h_hat_LS = y_tilde`.
- Program (P4) and Equation (31): genie-aided estimator with known sparse
  support.

Method description:

The LS comparator is the unstructured full-frequency preprocessed observation.
The genie-aided method uses ground-truth sparse support and diffuse structure
to benchmark parameter-estimation limits.

Paper inputs and outputs:

- LS input/output: full-frequency `y_tilde`.
- Genie input: full-frequency `y_tilde`, known diffuse basis `D`, known sparse
  support `A_f*`, and regularization `mu`.

Target repository abstraction:

- None for this pass.

Reason not implemented:

- The repository already has built-in LS baselines (`ls-nn`, `ls-lin`) in the
  Sionna pilot-domain benchmark.
- The paper's genie method requires ground-truth support, which is unavailable
  to a deployable channel estimator and inappropriate for the leaderboard.

What would be required:

- LS: no paper-specific estimator; use existing built-ins.
- Genie: a separate diagnostic experiment outside the deployable estimator
  leaderboard.

### Cramer-Rao Bound Analysis

Status: `not implemented`

Paper location:

- Section IV and Propositions 4-6.

Method description:

The paper derives CRBs for the hybrid sparse/diffuse channel parameters and
channel response.

Paper inputs and outputs:

- Inputs: the aHSD parameterization, supports, amplitudes, noise variance, and
  observation model.
- Outputs: lower bounds on parameter and channel-estimation error.

Target repository abstraction:

- None. This is analysis, not a channel estimator.

Reason not implemented:

- The task is to implement benchmarkable Sionna channel estimators.
- The CRB assumes knowledge of ground-truth parameters and supports and is not
  an estimator callable inside `evaluation/channel_estimation.py`.

What would be required:

- A separate analysis script, not a leaderboard estimator, if CRB plots are
  desired later.

## Auxiliary Asset Plan

No `aux/` assets are required for the first diagnostic pilot-comb Toeplitz-SDP
adaptation. The diffuse dictionary is a deterministic buffer constructed from
the active pilot-group length and constructor arguments. Empirical
hyperparameters and solver settings are passed through the runner and recorded
in the implementation report rather than cached as canonical assets.

If a future full-grid HALS/P2 implementation proceeds, likely assets are:

- `src/sionna_bench/mimo_ofdm_chest/papers/lyu2026/aux/<config-slug>/hals_hyperparams.json`: selected `tau`,
  `lambda` or `E_d`, and detector-facing error-variance scale, generated by a
  bounded validation sweep.
- Optional synthetic validation fixtures for SDP and support-recovery tests.
- No pretrained weights or channel statistics are required by the paper, but
  scenario-specific diffuse-energy calibration may be needed for P2.

## Recommended Next Decision

The implementer recommendation is to ask the reviewer to audit this revised
scope. If the reviewer accepts the revised plan and the human approves it, the
next implementation step is `SparseDiffuseChannelEstimator` plus a paper-local
`run_benchmark.py` for smoke and diagnostic runs. The paper's full-grid P2
with continuous support extraction and debiasing remains outside this pass
unless the benchmark contract and solver scope are changed explicitly.
