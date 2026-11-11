# Final Report

- Paper ID: `fesl2024`.
- Reviewer: independent Codex reviewer agent.
- Date: 2026-11-01.
- Reviewed: code, selected assets, completed selection/calibration/evaluation,
  focused verification and the consolidated paper-only contribution.

## Final Verdict

**Accepted with caveats for benchmark use and contribution preparation.**
No blocking implementation, numerical or documentation finding remains.
The original epoch 170 model and independently validated detector scale 0.623
are supported by completed evidence. All 35 focused cleanup checks pass;
formatting preserves executable behavior apart from the selected default.

Official result/leaderboard registration, global inventories and public asset
distribution are maintainer actions outside the paper-only contribution.
No staging, commit, push or PR is performed or authorized by this verdict.

## Reviewed Evidence and Contracts

Applied the reviewer/channel-estimation prompt layers, task interface,
AGENTS.md, CONTRIBUTING.md, shared execution rules, and the paper architecture,
asset, tuning and canonical-reference contracts. Read
[paper_summary.md](paper_summary.md), [plan.md](plan.md),
[plan_review.md](plan_review.md), [implementation_report.md](implementation_report.md)
and [README](../README.md). Later human approvals cover completed stages and
this documentation consolidation; no algorithm or new experiment is introduced.

The human designated arXiv:2403.03545v1 authoritative. The IEEE publication
remains cited; the complete final IEEE PDF was not inspected. Original release
wcl-2024 commit `fe43f0aa7d10da06f3f37217168625bf6c197cbb` and scheduler
comparison commit `63dfa20e0003a47295eb22cdb7120a04e4903aec` support recorded
checks, not runtime dependencies. The inclusive matched-step correction is
intentional; full-chain bit equality with the release is not claimed.

The reviewer inspected source/archives, actual test source/logs, manifests,
read-only SQLite, JSON and NPZ arrays, with small hash/statistical/AST checks.
No model, competitor, propagation, inference, detector, fitting or training
workload was executed by the reviewer.

## Findings

### Algorithm, interface and assets

The 55,025-parameter CNN, 16-dimensional sinusoidal affine conditioning,
100-step VP schedule, variance-one-half real epsilon target and deterministic
reverse means are retained. Actual LS pilot noise and frozen training power
determine normalization and matched timestep. Physical antenna order is
restored around the orthonormal array FFT; explicit `lin_time_avg` reconstructs
the OFDM grid. Independent correlated-Gaussian and reference CNN/step checks
support the mathematics without adding oracle inference.

The 64×16 paper matrix is materially adapted to physical 4×4 receive snapshots,
sparse OFDM pilots and new UMi training. These assumptions limit paper-result
reproduction; they do not substitute a different estimator. Training-only
population normalization enforces unit-average-power units without test-channel
or per-observation normalization. Construction accepts prepared tensors only;
strict external asset loading cannot generate, train or download implicitly.

The selected reference is LR1e-4/B512, 6400/256 complete realizations,
power 1.6261395394191218, validation-selected epoch 170. Ten optimizer trials did
not beat it on selection; independent confirmation of the preselected raw
winner retained it. The checkpoint diagnostic retained 170 among six tested
epochs. No rounded challenger or new checkpoint was fitted for final evaluation.

| Protected artifact | SHA-256 |
| :-- | :-- |
| Reference manifest | `ef8d88197d860f9942d085e9e2eb7af3f14088341d40bdac35bfbe3eb2c9668b` |
| Inference payload | `60b2d5e8d4c801d40a20354336e396981965583bada545c375f3eff96204ad1d` |
| Actual epoch 170 | `1e32560f9ed70138df5a29af497885c0bad4b985157c4b7df59d6bc94425d98e` |

The honest `single-reference` purpose is unchanged. Explicit runner/variance
purpose options preserve strict checks; default loading still requires the
appropriate canonical/smoke purpose and overrides cannot authorize training.

### Calibration and final evaluation

Read-only SQLite contains exactly 10 COMPLETE trials, no failed/pruned or
replacement trial. All ten log-uniform startup draws reproduce exactly from
sampler 24350000. Default TPE startup 10 is retained, not the earlier optimizer-
study startup 5 exception. Calibration uses BLER at 10 dB, 63×4 cached realizations,
seed 24330000+[0,62]. Raw winner trial 8 is 0.6229862696239178; rounding gives 0.623.
Independent paired 24340000+[0,62] validation gives BLER 0.02976190476190476
at 1 versus 0.028769841269841268 at 0.623, correctly satisfying strict adoption.
The 3.333% relative reduction is only 0.099206 percentage points absolute;
aggregate scores do not establish statistical significance or global optimality.
Selected and hashed paired records agree and bind the final result.

The final CPU 1/single-precision run uses setup seed 42/evaluation seed 2026, 11 SNRs
−10:2.5:15, 63×4 complete realizations/SNR and fingerprint
`4f774132bba964e4d31fe64a10374b020db5abd5c13a5e564b4e62bf258a4efc`.
All SNR/NMSE/both NMSE confidence-limit arrays match the scale 1 result in
shape, dtype and bytes. Final NPZ SHA-256:
`10396d3b84973308571fdfd3a78bb1419b8849712615ea2a5c972ae7ed075876`.
At 10 dB NMSE is 0.0017364497734659434, BER 0.0022061129218892157 and
BLER 0.016534391534391533. BER decreases at all 11 points; BLER decreases at 10
and ties at 15 dB. Canonical observations did not select the scale.
The full sequence took 3575.759 s; final evaluation 2251.301 s includes simulation,
detection, decoding and writing, not isolated CNN inference.

Independent comparison reproduced all 20 published NPZ/leaderboard mappings,
220 relative improvements and 231 ranks. Unchanged NMSE retains ranks
12,11,11,11,11,9,8,7,4,7,7. Diffusion nominally beats VAE/spatial GMM only
at 10 dB, by 1.1655%/1.8878%; spectral GMM/LMMSE/RKHS remain better throughout.
Published LS uses nn/lin, not matched `lin_time_avg`. CUDA/CPU metadata establish
common scientific settings, not identical realized samples or paired
cross-estimator significance. Marginal bootstrap arrays cannot supply a paired
improvement interval without the raw sufficient statistics.

### Cleanup and verification

Independent AST comparison of all five Python files against executed snapshots
is exact after removing docstrings and normalizing only default 0.623 back to 1.
No schedule, solver, arithmetic, training behavior or purpose guard changed.
Fixed-asset comparison also gives bitwise h_hat/variance equality at 0.623.
Tier B inference remains supported for the tested eager compile backend;
no CUDA/Inductor or all-geometry calibration result is claimed.

Reviewed actual external tests/logs: 29 analytical/reference/transform/noise/
Gaussian/precision/compile/mocked-integrity checks passed in 5.54 s; six final
fixed-weight/default/purpose/rounding/result-linkage checks passed in 5.63 s.
Training/data-generation fixtures were excluded. Existing saved development
inputs and protected epoch 170 weights were reused; no new training or Monte
Carlo run occurred for cleanup. Synthetic study-resume checks establish
accounting/context, not sampler-RNG recovery; exact partial stochastic-study
continuation is documented as unsupported and was refused by the external run.

Historical red flags, failures, bounded recovery and approvals remain archived:
early undertrained deficits, same-fit interrupted training/resume, evaluation-
order correction, serialization-only result recovery, interrupted reviewer
handoff and pre-observation dependency-discovery mismatch. None was hidden
by retraining on canonical data or replacing a failed calibration trial.

## Method Audit

Original iterative angular-domain diffusion: accepted for the approved 4×4/
OFDM adaptation. The surrogate rule is satisfied; no single-step approximation,
replacement objective or oracle inference is mislabeled. Pilot grouping,
explicit interpolation, prepared-state construction, precision/device and
error units follow the interface. Asset counts distinguish complete realizations
from correlated derived vectors; exact model/scale selection is documented.

Spatial ablation, SNR mismatch, alternate schedules/library targets, original
published-weight reproduction and additional comparison algorithms remain
accepted scoped omissions. Faithful extensions require their matching data/
models or a separately reviewed experiment; no surrogate is substituted.

## Divergence Ledger

| Divergence | Source | Verdict | Paper impact / rationale | Benchmark fairness |
| :-- | :-- | :-- | :-- | :-- |
| Matrix prior becomes physical receive-array prior | interface | justified | Same diffusion algorithm, new UMi geometry/distribution. | Explicit scope; no original-curve reproduction claim. |
| Sparse pilots plus `lin_time_avg` | interface | necessary | OFDM reconstruction is an interface adaptation. | Full-grid ranking includes interpolation effects. |
| Training-only population power and adapted bank/budget | asset | justified | Human-approved resource/domain convention, not test normalization. | Common candidate data; derived vectors not called independent channels. |
| Inclusive matched transition differs from release | implementation | justified | Authoritative Algorithm 1 and Gaussian identities support extra update. | Fixed before tuning, no outcome-selected mode. |
| Empirical LS variance, selected 0.623 | interface | justified | Not diffusion posterior covariance; independently calibrated. | Exact mean/NMSE invariance, finite-cohort gain qualified. |
| Retain single-reference incumbent without rounded challenger fit | asset | justified | Approved independent confirmation rejected challenger. | No relabeling or canonical-data reselection. |
| Formatting/default adoption and report consolidation | implementation | justified | AST/fixed-asset equivalence; original evidence archived. | No altered result or concealed failure. |
| External tests/assets/results, deferred global registration | interface | necessary | Contribution scope permits this paper directory only. | Maintainers own distribution/inventories/official outputs. |

## Suitability Assessment

The small shared conditional network and explicit noise matching fit reusable
Sionna pilot estimation. Results are useful against published LS but do not
establish broad VAE/GMM superiority and remain behind stronger baselines.
Other arrays require matching assets and independent detector calibration.
No further experiment is needed for this documentation consolidation.

Only five Python modules, README and five required Markdown reports belong in
the submission. Generated payloads/tests/PDF/bytecode and shared edits are
excluded. Python headers identify Benedikt Fesl and Apache-2.0; DCO sign-off
remains the human's action. No Git publication is authorized.
The complete 48-file predecessor was hash-verified in the authors' external
`fesl2024-finalization-v1/paper_after_cleanup/` archive against the sibling
`final_hashes.json` manifest. Archived reports, raw external
experiment roots, test source/logs and executed sources preserve the audit trail.

**Final verdict: accepted with caveats; no blocking implementation finding.
Official registration/inventories/distribution remain maintainer follow-up.**

## Maintainer Changes After Review (2026-11-11)

The maintainers regenerated the canonical asset with the reviewed recipe on
CUDA, recalibrated the detector-facing scale to `0.0358`, and tracked the
canonical results; the authors' training study was accepted as reported. The
verdict above refers to the authors' submission and is unchanged. Details are
in the maintainer addendum of
[implementation_report.md](implementation_report.md).
