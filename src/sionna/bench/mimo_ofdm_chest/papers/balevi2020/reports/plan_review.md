# Plan Review: Balevi et al., 2020

Reviewer role: mandatory plan reviewer before implementation.

Sources reviewed: `AGENTS.md`, `prompts/reviewer.md`,
`prompts/task_reviewer.md`, `prompts/task.md`,
`src/sionna_bench/mimo_ofdm_chest/papers/balevi2020/paper.pdf`, `reports/paper_summary.md`, and
`reports/plan.md`.

## Overall Verdict

The plan is conditionally approved. The main `DeepPriorChannelEstimator`
classification may remain `implement`, but only if the implementer revises the
plan to incorporate the conditions below and then obtains explicit human
approval before writing implementation code.

The proposed direct-call architecture is acceptable only as a narrow interface
adaptation needed to preserve Balevi et al.'s order of operations:

```text
received grid -> Eq. (16) untrained DNN fitting -> Eq. (17) denoised grid
-> LS pilot operation
```

It is not approved as evidence that Balevi et al. define a Sionna-style
full-grid reconstruction or interpolation method. The paper assumes pilots on
all subcarriers in its system model and does not specify interpolation for the
benchmark's sparse Kronecker pilot pattern. Therefore the sparse-pilot LS step
and `lin_time_avg` interpolation must be documented as benchmark adaptations.

## Method And Variant Decisions

| Method or variant | Plan classification | Reviewer decision | Rationale |
|---|---:|---:|---|
| `DeepPriorChannelEstimator` | `implement` | conditional | The DNN architecture, per-block fitting objective, and denoised-signal LS step are traceable to Eq. (10)-Eq. (17). A normal `estimate_at_pilot_locations()` implementation would discard the received-grid tensor used by Eq. (16), so direct `call(y, no)` is justified. Conditions below are required because the paper does not specify sparse-pilot interpolation, detector error variances, or non-oracle early stopping. |
| Single-antenna, single-cell massive-MIMO, and hidden-width/pilot-count simulation settings | scenario variants under main estimator | accepted | These are evaluation settings and hyperparameter choices, not distinct inference algorithms. The plan correctly treats `k`, pilot count, and antenna count as configuration or validation parameters. |
| Pilot-contamination and inpainting experiments | `not implemented` | accepted | The current benchmark does not generate neighboring-cell pilot reuse, random contamination masks, contiguous interference patches, or paper-style SIR controls. Reproducing Fig. 8 requires a scenario extension, not just an estimator class. |
| Paper LS, MMSE, and genie-aided MMSE baselines | `not implemented` | accepted | These are paper reference baselines, not Balevi's proposed estimator. The benchmark already has built-in LS and LMMSE baselines. Exact genie-aided MMSE would require oracle covariance/statistical assets and should be handled as a separate baseline task if needed. |

## Required Plan Revisions Before Approval

1. State that `FullGridChannelEstimator` is used solely to expose raw `y` for
   Eq. (16) before pilot extraction. The plan must not describe Balevi et al.
   as a native full-grid sparse-pilot reconstruction method.

2. Make interpolation policy unambiguous. The reviewed default is
   `lin_time_avg`. Nearest-neighbor interpolation is not approved for canonical
   Balevi outputs unless a later reviewed plan change justifies it.

3. Replace `crop_or_interpolate` for the DNN output with one deterministic
   policy. The paper's simulations use 64 by 64 grids, while the benchmark has
   624 effective subcarriers and 14 OFDM symbols; the implementation plan must
   say exactly how the upsampled tensor is resized and must report this as an
   interface adaptation.

4. Resolve the output-shape statement against the actual harness behavior.
   The task contract states `fft_size`, while existing direct-call estimators
   return estimates over effective subcarriers. The Balevi implementation must
   match the harness and document the chosen shape in code, README, and report.

5. Keep the fitting objective label-free. Runtime optimization may use only
   the received tensor `y` and fixed random input `Z0`; it must not use
   `h_true`, decoded bits, benchmark NMSE, or final benchmark results.

6. Replace oracle early stopping with a fixed `num_steps` or other
   preselected rule from a bounded validation sweep. The selected
   `hidden_channels`, `num_steps`, and criterion must be reported before
   canonical outputs are generated.

7. Treat `err_var_scale=1.0` as only a constructor placeholder. Canonical
   outputs require a bounded validation sweep for detector-facing confidence
   scale, with candidate grid, criterion, selected value, and result reported.

8. Explicitly disclose that the estimator fits the DNN on the benchmark's
   full received grid, including data-bearing resource elements. This follows
   the raw received-grid denoising idea, but it gives the estimator a stronger
   input surface than pilot-only baselines and may not reproduce the paper's
   exact simulation signal model.

## Divergence Ledger

| Divergence | Source | Reviewer verdict | Rationale | Impact on paper faithfulness | Impact on benchmarking fairness |
|---|---|---:|---|---|---|
| Direct `call(y, no)` via `FullGridChannelEstimator` despite the paper's all-subcarrier pilot assumption | interface | justified | Eq. (16) fits the DNN to the received tensor before LS. Sionna's pilot-domain hook would remove the central paper input. | Preserves the paper's denoise-before-LS algorithm better than pilot-domain-only adaptation. | Gives raw-grid access not available to pilot-only estimators; must be disclosed. |
| Sparse Sionna Kronecker pilots followed by explicit `lin_time_avg` interpolation | interface | necessary | The paper assumes pilots over all subcarriers and does not define interpolation for the benchmark pilot pattern. | Reduces faithfulness for the channel-grid completion part; LS at denoised pilots remains paper-traceable. | Conservative and comparable if named as an adaptation and compared to `ls-lin`/`lmmse`. |
| Fitting on data-bearing benchmark resource elements | interface | questionable | The paper describes fitting to the received OFDM tensor, but its simulations do not map cleanly to coded Sionna data REs. | Could change what structure the DNN learns compared with the paper. | Potentially stronger input than baselines; performance claims need caveats. |
| Fixed optimizer steps replacing true-channel-NMSE early stopping | paper | necessary | Paper tuning uses true-channel NMSE, which is unavailable to a deployable estimator. | Faithful objective is preserved, but stopping rule differs. | Fair if selected by predeclared validation and not final benchmark tuning. |
| DNN output resizing for 624 by 14 benchmark grid | interface | questionable | The paper uses 64 by 64 grids compatible with repeated upsampling; the benchmark dimensions are not equally convenient. | Depends on the chosen crop/resize policy; must be deterministic and documented. | May affect NMSE independently of the paper algorithm; tests and reporting needed. |
| Multi-UE LS after one shared received-grid denoising pass | interface | justified | The benchmark has four orthogonal UE streams; applying LS per Sionna pilot stream after common denoising preserves the paper's LS-type second stage. | Minor notation change from the paper's single-user LS expressions. | Fair if pilot extraction uses Sionna symbols and no user-specific labels. |
| Detector-facing `err_var` fallback with tunable scale | interface | necessary | Balevi et al. do not define Sionna marginal error variances. | Does not affect channel estimate faithfulness, but affects receiver behavior. | Must be validation-tuned and reported to avoid hidden BLER/BER tuning. |
| Omission of pilot-contamination experiments | interface | justified | Current benchmark lacks contaminated regions and SIR controls. | Does not implement one paper claim, but the plan classifies it correctly. | Avoids comparing incompatible scenarios. |
| Omission of paper MMSE/genie baselines | asset | justified | Exact paper baselines require covariance/oracle statistics and are not the proposed estimator. | Does not reduce faithfulness of the Balevi estimator. | Built-in baselines are sufficient for the current benchmark unless exact paper-curve reproduction is separately requested. |

## Proceed Statement

Implementation may proceed only after the implementer revises `reports/plan.md`
to address the required conditions above and the human explicitly approves the
reviewed plan. Without those revisions, the plan is not approved for coding.
