# Plan Review

- Paper ID: `fesl2024`.
- Reviewer: independent Codex reviewer agent.
- Original review: 2026-10-25, plan version 2.
- Consolidated: 2026-11-01, preserving original decisions and later gate closure.

## Overall Gate Decision

**Architecture accepted for the approved receive-array/OFDM adaptation.**
The original review conditionally accepted one angular-domain, SNR-initialized,
deterministic iterative `DiffusionChannelEstimator`. No further mathematical
plan revision remained in version 2; implementation initially awaited human
approval and source-authority resolution. The human subsequently approved the
4×4 adaptation, inclusive matched transition and single-model training policy,
and designated arXiv:2403.03545v1 authoritative. Those gates are closed.
This consolidation is not a new algorithm or plan-approval request.

The official citation remains IEEE DOI10.1109/LWC.2024.3474570. The review used
the complete identified arXiv v1 and original release; it did not inspect the
complete final IEEE PDF. The original final-PDF condition was superseded by
the human's source instruction, not by a claim that final-PDF collation occurred.
Completed implementation/results are assessed in [final_report.md](final_report.md).

## Applicable Contracts

Applied the reviewer/channel-estimation prompt layers and task interface;
paper architecture, asset-manifest, tuning and canonical-reference contracts;
shared execution rules; CONTRIBUTING.md; and the explicit human scope.
[plan.md](plan.md) and [paper_summary.md](paper_summary.md) contain the
implementation specification and source inventory. The contribution is only
this paper directory; tests, data, weights and generated results stay external.

Original source identities are preserved: preprint SHA-256
`8b99d0a8cf36d61378cb737fd98aa015e81693aa62cf65a95f3cf845e421ddd4`,
`Diffusion_channel_est` wcl-2024 commit
`fe43f0aa7d10da06f3f37217168625bf6c197cbb`, and comparison scheduler
`diffusers-dmse` commit `63dfa20e0003a47295eb22cdb7120a04e4903aec`.
Neither reference repository is a runtime dependency.

## Harness Understanding

The plan correctly maps each active orthogonal pilot to a complete 16-entry
receive-array vector. UEs, frequencies and OFDM symbols are separate examples,
not transmit-array axes. Sionna's antenna order is `row + rows*column`;
a physical row/column FFT therefore requires the planned reorder.

`BasePilotChannelEstimator` with explicit `lin_time_avg` is appropriate:
the paper denoises fully observed spatial channels, rather than reconstructing
a sparse OFDM grid itself. The output is full-grid complex `h_hat` and original-
unit finite nonnegative `err_var` in Sionna's standard axes. Compact pilot zero
slots remain inactive. Canonical pilot power 4 gives complex LS variance
`no/4`; actual pilot noise, not numerical Eb/N0, determines diffusion SNR.

Loader/generation/construction/evaluation responsibilities are separate.
Complete system realizations are the independent split/evaluation units;
correlated users and pilot vectors do not increase that count. Built-in
baselines, explicit interpolation and transient smoke versus canonical output
rules were adequately understood.

## Method Reviews

### Angular-domain deterministic diffusion estimator

- Implementer status: `implement`; reviewer decision: **accepted** under the
  approved adaptations and source authority.
- Paper coverage and equation-linked pseudocode are adequate: actual-noise
  observation scaling, linear-SNR matching, epsilon-to-mean conversion,
  deterministic reverse order, inverse transform and restoration of units.
- The released CNN structure is retained: widths (2,32,64)/(64,43,22,2), five
  3×3 convolutions, ReLU placement, 16-dimensional sinusoidal conditioning and
  residual affine modulation; 55,025 parameters, without batch normalization.
- The fixed 100-step VP schedule and complex epsilon target with real variance
  one-half remain. No score-SDE/DDIM replacement, supervised LS-to-clean
  surrogate, stochastic sampling or oracle fallback is substituted.
- Frozen training-only pooled power implements the population-unit-power
  convention. Per-observation/test-channel normalization and inverse VP
  rescaling are excluded. Noise and output variance units remain distinct.
- The schema-2 asset plan specifies external paths, ordered pilot/geometry
  compatibility, state/schedule/dtype/checksum checks and exact checkpoint
  linkage; normal loading cannot train, generate or download implicitly.
- The approved single-model policy uses 6400/256 complete realizations and
  7,987,200/319,488 correlated derived vectors, max 200 epochs, patience 20 and
  relative delta 1e-4. These are explicit benchmark training-policy adaptations,
  not the paper's 100000/10000-matrix, 500-epoch experiment or eleven SNR models.
- Bounded optimizer selection, measured cost approval, independent confirmation
  and post-freeze detector calibration are separate stages. Later approved
  protocol amendments and selected outcomes are recorded in
  [implementation_report.md](implementation_report.md).
- Tier B compiled inference was an appropriate target, conditional at planning
  on whole-estimator eager/compiled tests; completed evidence is in the final
  audit. No full-graph or CUDA/Inductor result was assumed.

### Methods not implemented

| Method | Reviewer decision | Reason and requirement for a faithful extension |
| :-- | :-- | :-- |
| Spatial-domain DM ablation | accepted | Outside angular-domain scope; requires a separately trained spatial prior and labeled evaluation. |
| SNR-mismatch experiment | accepted | A robustness study, not the selected API; needs controlled perturbation streams and frozen estimator. |
| Alternative total-step schedules | accepted | Require compatible learned weights and a reviewed recipe; reverse-loop subsampling is not equivalent. |
| Generic library sampling/alternative targets | accepted | Not the selected deterministic WCL recipe; adding stochastic or ground-truth branches would enlarge scope. |
| Paper comparison estimators | accepted | Separate methods/assets or oracle side information; use existing benchmark results instead of adding algorithms. |

## Findings and Resolutions

**Seed overlap resolved.** Version 1's adjacent calibration/validation bases
would overlap under `seed+batch_index`. Version 2 reserves disjoint role ranges:
24330000+[0,62] and 24340000+[0,62]. Training diffusion/shuffle use separate
keyed local generators; propagation streams remain disjoint. Derived vectors
must remain grouped by complete realization across splits.

**Matched-index correction accepted.** Coefficient index k denotes paper
state k+1. Algorithm 1 therefore includes k,…,0; the release selects k then
executes k−1,…,0. Keep this intentional extra update visible, with no legacy
inference selector or claim of complete-chain release equality. Independent
correlated proper-complex Gaussian calculations matched exact LMMSE to
floating-point precision at multiple schedule indices, supporting the decision.

**Source qualification resolved by human authority.** Keep the IEEE citation
and arXiv/release distinction. No final-PDF inspection is implied.

**Resource and selection discipline accepted.** Fit-count ceilings are not
runtime bounds. Full work required measured feasibility and explicit cost
approval; retry counts, failed trials and actual costs must be disclosed.
Canonical outputs remain outside training, algorithm selection and variance
calibration. Later completed approvals do not authorize further experiments.

## Divergence Ledger

| Divergence | Source | Verdict | Rationale / paper impact | Benchmark fairness |
| :-- | :-- | :-- | :-- | :-- |
| 64×16 matrix becomes 4×4 receive snapshot | interface | necessary | Preserves spatial diffusion law, changes learned prior/axis meaning; not original-curve reproduction. | Scenario-matched training and explicit limits. |
| Physical two-axis FFT with antenna reorder | interface | justified | Correct unitary angular representation of planar array. | No hidden joint-user information. |
| Sparse-pilot LS plus `lin_time_avg` | interface | necessary | OFDM reconstruction is external to paper's spatial denoiser. | Deliberate common interpolation; full-grid results include its effect. |
| Pooled frozen training power | asset | justified | Population-power convention, not observation whitening. | Training-only statistic and complete-realization splits prevent leakage. |
| Inclusive index differs from release | implementation | justified | Authoritative Algorithm 1 and Gaussian identities support one extra update. | Fixed before tuning; no outcome-driven alternate mode. |
| Single 6400/256 bank and reviewed training budget | asset | justified | Human-approved shared-prior adaptation, not original experiment. | Common candidate data and bounded disclosed budgets. |
| Scaled-LS detector variance | interface | necessary | Added empirical confidence, not diffusion posterior variance. | Independent post-freeze BLER calibration; unchanged mean/NMSE. |
| Exact-zero-noise LS endpoint | implementation | justified | Explicit infinite-SNR limit outside ordinary positive-noise loop. | No tuned clipping or hidden noise injection. |
| External tests/assets/results | interface | justified | Contribution policy excludes generated payloads/shared edits. | Preserve executable evidence and maintainer integration boundaries. |

## Final Instructions To Implementer

The original mathematical revisions are resolved and subsequent source/human
gates were satisfied as stated above. Preserve the reviewed algorithm, method
omissions, source caveats and honest performance limits. Current selected
model/calibration and contribution readiness are governed by the final audit,
not by historical pre-coding restrictions.

Full pre-consolidation reports remain hash-verified in the authors' external
`fesl2024-finalization-v1/paper_after_cleanup/` archive against the sibling
`final_hashes.json` manifest. No experiment was repeated.

**Verdict: plan accepted for the approved adaptation; no new approval gate.**
