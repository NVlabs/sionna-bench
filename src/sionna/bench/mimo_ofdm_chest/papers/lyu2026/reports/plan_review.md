# Plan Review: Lyu et al., 2026

## Reviewer Verdict

The revised plan is accepted under the explicit human scope change.

The plan now correctly distinguishes three things:

- the exact paper methods P0, P1, and full-grid constrained/debiased P2;
- a human-requested pilot-comb adaptation of P2 for the Sionna benchmark;
- the omitted Equation (16) support extraction and Equation (17) debiasing.

The important correction relative to the previous review is that P2 is not a
finite-grid `l1` surrogate. Program (P2) is a finite-dimensional Toeplitz-lift
SDP representation of the continuous atomic-norm regularizer. In that SDP,
`h_s` is an optimization variable, but it is coupled to the Toeplitz variable
and scalar lift through the Hermitian PSD block constraint. The revised plan's
`Pilot-Comb Toeplitz-SDP P2 Adaptation` preserves this Toeplitz SDP structure.

Implementation may proceed for `SparseDiffuseChannelEstimator` as a diagnostic,
pilot-comb adaptation. Exact full-grid HALS/P2 remains `not implemented`.

## Harness and Contract Review

The plan correctly restates the Sionna channel-estimation contract: Sionna
extracts pilot observations, the estimator returns pilot-location estimates,
and explicit interpolation produces full-grid channel estimates and error
variances.

The plan also correctly identifies the paper-to-benchmark mismatch. Lyu et al.
form a full length-`N` preprocessed observation `y_tilde` in Equations
(11)-(12), whereas the repository benchmark provides sparse Kronecker pilot
observations. Treating each pilot comb as a reduced OFDM grid is therefore an
interface adaptation, not the exact paper observation model.

This adaptation is acceptable here only because the human explicitly requested
trying it and because the plan does not present it as exact full-grid HALS.
The interpolation choice, `"lin_time_avg"` by default, is explicit and must be
reported as a Sionna benchmark adaptation.

## Method Review

### Noiseless Atomic Sparse/Diffuse Demixing, P0

Reviewer status: accepted.

P0 is a noiseless full-grid demixing program based on the atomic norm of
`y_tilde - D gamma`. It is primarily an exact-recovery object and is not the
noisy estimator intended for this benchmark pass. The plan is right to leave
it `not implemented`.

### HALS Penalized Noisy Estimator, P1

Reviewer status: accepted.

P1 is the paper's penalized noisy HALS estimator over the full-frequency
observation `y_tilde`. It requires the Toeplitz SDP representation of the
atomic norm, `tau` and `lambda` selection, and Sionna error-variance
adaptation. The plan is right to leave exact P1 `not implemented` in this
pass.

### Paper Full-Grid Constrained and Debiased HALS, P2

Reviewer status: accepted.

The exact paper P2 consists of the constrained Toeplitz SDP, followed by
continuous support extraction from Equation (16) and debiasing by Equation
(17). The plan correctly keeps this exact method as `not implemented`, because
the proposed diagnostic method changes the observation grid and initially
returns the biased reconstruction `h_s + D gamma`.

### Pilot-Comb Toeplitz-SDP P2 Adaptation

Reviewer status: accepted as a human-approved experimental adaptation.

The revised plan is algorithmically coherent:

- it uses the Toeplitz-lift SDP form of P2, not a finite Fourier dictionary
  with an `l1` penalty;
- it applies that SDP to LS-normalized pilot-comb vectors;
- it treats the comb spacing as an effective larger subcarrier spacing;
- it returns the biased reconstruction `h_s + D gamma`;
- it delegates off-pilot reconstruction to explicit Sionna interpolation.

This remains a substantial departure from exact HALS/P2 because the paper's
`y_tilde` is full-grid and because the debiasing stage is omitted. Those
departures are justified for a first runnable diagnostic under the human's
scope change, but they must be reported plainly.

Required implementation caveats:

- Do not call the method exact HALS or exact P2 in outputs, README, or reports.
  `SparseDiffuseChannelEstimator` is acceptable if the description says
  "pilot-comb adaptation" or "biased pilot-comb P2 adaptation".
- Solver failures must not be silent. If LS fallback is used, report counts or
  at least state that the run is diagnostic and fallback-contaminated.
- `tau`, `diffuse_energy`, solver name/tolerance/iteration limit, and
  `err_var_scale` must be recorded for every run.
- For canonical outputs, `err_var_scale` must be selected by a bounded
  validation sweep and documented. For feasibility diagnostics, it may be a
  stated provisional value.
- The reduced diffuse dictionary `D` should be documented as living on the
  pilot-comb grid. Any handling of pilot-comb offsets should be consistent
  across groups; absorbing fixed offset phases into complex coefficients is
  acceptable if documented.
- CVXPY/SCS/Clarabel execution is CPU-only and not torch-compilable. This is
  acceptable for the diagnostic scope, but runtime limits should be stated in
  the implementation report.

### Vanilla Atomic Norm Minimization Baseline

Reviewer status: accepted.

The paper's ANM baseline is full-grid and not required for the repository
leaderboard. Excluding it is acceptable.

### LS and Genie-Aided Comparators

Reviewer status: accepted.

The paper's LS comparator is `h_hat_LS = y_tilde`, which is not identical to
the repository's Sionna pilot-domain LS baselines. The genie-aided comparator
uses unavailable ground-truth support. Neither should be implemented as a
deployable estimator in this pass.

### Cramer-Rao Bound Analysis

Reviewer status: accepted.

The CRB material is analysis, not a Sionna channel estimator. Excluding it is
correct.

## Divergence Ledger

### Exact full-grid P2 not implemented

- Source: interface
- Reviewer verdict: necessary
- Rationale: exact P2 uses the paper's full-frequency `y_tilde`, continuous
  support extraction, and debiasing.
- Impact on paper faithfulness: preserves faithfulness by not labeling the
  pilot-comb diagnostic as exact full-grid HALS/P2.
- Impact on benchmarking fairness: avoids overclaiming the leaderboard method.

### Reduced pilot-comb observation

- Source: interface
- Reviewer verdict: justified under human scope
- Rationale: the human requested treating the comb as a reduced OFDM grid so
  that the method can run inside the Sionna pilot-domain estimator contract.
- Impact on paper faithfulness: substantial; the observation model is no
  longer Equations (11)-(12)'s full-grid `y_tilde`.
- Impact on benchmarking fairness: acceptable if the method name and README
  clearly identify the reduced-grid adaptation.

### Toeplitz SDP retained

- Source: implementation
- Reviewer verdict: justified
- Rationale: the revised plan keeps the paper's SDP representation of the
  atomic norm instead of replacing it by a gridded sparse penalty.
- Impact on paper faithfulness: materially improves faithfulness relative to
  the previous finite-grid surrogate plan.
- Impact on benchmarking fairness: acceptable, subject to runtime disclosure.

### Debiasing omitted

- Source: implementation
- Reviewer verdict: justified for diagnostic implementation
- Rationale: Equation (16) support extraction and Equation (17) debiasing are
  central to exact HALS/P2, but the human approved first checking the biased
  reconstruction.
- Impact on paper faithfulness: substantial. The method must be described as
  biased and incomplete relative to exact P2.
- Impact on benchmarking fairness: may create high-SNR bias or error floors;
  these should be discussed if observed.

### CPU CVXPY solver path

- Source: implementation
- Reviewer verdict: justified under diagnostic scope
- Rationale: the paper itself uses off-the-shelf convex solvers such as CVX;
  CVXPY/SCS or Clarabel is a reasonable local analogue.
- Impact on paper faithfulness: acceptable for the SDP solve, modulo solver
  tolerances.
- Impact on benchmarking fairness: runtime may limit Monte Carlo depth and
  must be reported with any results.

### Empirical hyperparameter tuning

- Source: implementation
- Reviewer verdict: justified with documentation
- Rationale: the paper discusses practical selection of `tau` and `E_d`; the
  repository additionally requires tuning detector-facing confidence scales.
- Impact on paper faithfulness: acceptable if the selected values and search
  criterion are recorded.
- Impact on benchmarking fairness: acceptable for diagnostics; canonical
  results need bounded validation rather than silent tuning on final metrics.

### Detector-facing error variance fallback

- Source: interface
- Reviewer verdict: necessary
- Rationale: the paper does not define Sionna-compatible marginal
  channel-estimation error variances.
- Impact on paper faithfulness: unavoidable benchmark adaptation.
- Impact on benchmarking fairness: acceptable if `err_var_scale` is exposed,
  tuned or marked provisional, and reported.

## Required Revisions Before Coding

None required.

## Implementation Approval Boundary

Implementation may proceed only for `Pilot-Comb Toeplitz-SDP P2 Adaptation`,
implemented as `SparseDiffuseChannelEstimator`, after human approval.

The implementation is approved as a diagnostic pilot-comb adaptation of P2, not
as exact full-grid HALS/P2. P0, exact P1, exact constrained/debiased P2, ANM,
LS/genie comparators, and CRB analysis remain out of scope for this pass.
