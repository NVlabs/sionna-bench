# Final Review: Lyu et al., 2026

Review updated: 2026-08-15

## Reviewer Verdict

Accepted with caveats for diagnostic use.

`SparseDiffuseChannelEstimator` is a pilot-comb adaptation of Program (P2), not
the paper's exact full-grid HALS estimator. The implementation now states its
actual observation model directly: repeated pilot-symbol observations are
averaged per stream and receive antenna, after which independent vector
problems are solved. No cross-antenna common-support method is claimed.

## Paper Faithfulness

The test-local reduced CVXPY reference retains complex `h_s` and `gamma`, a
Hermitian Toeplitz lift, the PSD block constraint, the objective
`||h_ls - h_s - D gamma||_2^2 + tau*(t+iota_1)`, and the diffuse-energy
constraint `||gamma||_2^2 <= E_d`.

The paper's Equation (12) motivates averaging repeated observations. This is
done independently for each receive antenna. The implementation then returns
the biased `h_s + D gamma` estimate. It does not implement the paper's
continuous support extraction from Equation (16), projection debiasing from
Equation (17), or exact full-grid observation model.

The ADMM backend targets the same reduced Toeplitz variables through PSD
projection and alternating updates. It avoids atom-grid quantization, but it
is still an approximate, early-stopped solver without a conic optimality
certificate.

## Sionna Review

The estimator inherits from `BasePilotChannelEstimator`, uses Sionna's
internal `_PilotGroupMixin`, registers its fixed diffuse dictionary and pilot
quantities as buffers, and exposes ADMM and interpolation choices through
constructor arguments.
Off-pilot reconstruction remains explicit because the paper assumes
full-subcarrier observations and does not define sparse-comb interpolation.
Specifically, Equations (5)-(6) form length-`N` vectors over every subcarrier,
and Equation (12) averages those complete LS observations. The paper provides
no sampling operator or reconstruction rule for missing subcarriers, so the
current `BasePilotChannelEstimator` boundary is the appropriate Sionna
adaptation.

CVXPY is confined to the test-local reference oracle. The complete Torch ADMM
estimator, including Sionna interpolation, compiles with `fullgraph=True`.
Compiled and eager outputs agree on both tested system geometries, and
`torch._dynamo.explain` reports one graph with no graph breaks, establishing
Tier A inference.

## Verification

The focused test suite covers three reduced solver dimensions, 50 randomized
Torch-ADMM-versus-CVXPY comparisons, diffuse-only feasibility, and two
end-to-end system geometries. The system variants change comb spacing,
pilot-symbol count, UE count, and BS antenna count.

```text
python -m pytest test/mimo_ofdm_chest/papers/lyu2026/test_estimator.py -q
```

Result on 2026-08-15:

```text
14 passed
```

The final calibrated-source regeneration completed 80 solver trials without
failure, canonical-size paired solver validation, a cached ten-trial 10 dB
detector-scale study, paired detector validation, and a canonical benchmark.
Paired validation selected rounded solver parameters
`(3, 6.55, 0.807, 0.08, 18)`. Rounded detector scale `0.00558` tied incumbent
`1.0` exactly, so the strict tie rule retained `err_var_scale=1.0`.

The canonical benchmark improves BER and BLER at every SNR. Its NMSE improves
below 0 dB and regresses from 0 dB onward; five-SNR weighted NMSE is 10.6%
worse than the former worktree output despite the selected candidate winning
the independent paired validation. This is reported as a material benchmark
outcome, not used to revise the frozen selection.

## Divergence Ledger

### Reduced Pilot-Comb Observation

- Source: interface adaptation.
- Verdict: accepted for diagnostics.
- Impact: substantial, because the paper assumes observations on all
  subcarriers.

### Approximate ADMM Backend

- Source: runtime adaptation.
- Verdict: accepted for diagnostics and canonical comparison with explicit
  labeling.
- Evidence: close agreement with CVXPY/Clarabel on the bounded randomized
  reduced-SDP suite.
- Impact: no convergence or optimality certificate is available for the
  early-stopped canonical run.

### Continuous Support Recovery Omitted

- Source: approved scope limitation.
- Verdict: clearly documented.
- Impact: the returned estimate can retain sparse shrinkage bias and is not the
  paper's final HALS estimate.

### Detector-Facing Error Variance

- Source: Sionna adaptation.
- Verdict: calibrated for the final rounded solver configuration.
- Impact: the Equation (12)-aware fallback
  `err_var_scale * no * sum_g w_g^2/|pilot_g|^2` is not paper-defined. A
  ten-trial cached 10 dB study selected raw `0.0055752934`, but independent
  paired validation tied rounded `0.00558` and incumbent `1.0`; the strict tie
  rule retained `err_var_scale=1.0`.

## Final Verdict

No blocking implementation defect was found within the diagnostic scope. The
method must continue to be described as a reduced pilot-comb P2 adaptation,
not as a faithful implementation of the complete full-grid HALS estimator.
