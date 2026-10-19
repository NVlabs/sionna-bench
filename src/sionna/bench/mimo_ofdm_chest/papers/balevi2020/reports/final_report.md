# Final Report: Balevi et al., 2020

V2 calibrated-channel addendum (2026-08-14): the final study used Sionna
commit `f1620834`, calibration ancestor `ad1609ea1`, and UMi TR 38.901 v19.2.
Independent paired validation rejected the rounded 30-trial fitting winner
(`0.0062403092`) in favor of the incumbent recipe (`0.0062212779`). A separate
10-trial 10 dB BLER calibration accepted rounded `err_var_scale=0.454` over
incumbent `0.334` (`0.03472222` versus `0.03505291`). The isolated canonical
benchmark took 2 hours 11 minutes 19 seconds and achieved five-point weighted
NMSE `0.0055496595`. All NMSE, BER, and BLER 95% intervals overlap the
preserved worktree result at every SNR. These results supersede the historical
addenda below.

RNG-corrected fitting-selection addendum (2026-08-05): repeating the 30-trial
study with the shared post-construction RNG reset produced raw winner
`num_steps=1587`, `learning_rate=0.001083708466655142`, and
`input_noise_scale=0.3852956689041489`. On an independent seed-`161803`
63-batch-per-SNR holdout, the incumbent, raw winner, and three-significant-
digit candidate scored `5.817681719894735e-3`, `5.816172850453034e-3`, and
`5.844327044435728e-3`, respectively. Selection compared the incumbent with
the rounded candidate, so the existing `num_steps=1597`,
`learning_rate=0.000698`, and `input_noise_scale=1.70` point is retained. The
corrected detector study selected raw `err_var_scale=0.38285407090201085`, but
on the independent seed-`314159` holdout incumbent `0.334` scored
`0.21055273600140045`, while the raw winner and rounded `0.383` scored
`0.21070107626868895` and `0.21070088304541565`. The incumbent is therefore
retained. The corrected 63-batch canonical benchmark achieved weighted NMSE
`5.963163e-3`, compared with `6.110896e-3` before the RNG correction and
`6.682170e-3` for the older regression reference. Its NMSE, BER, and BLER 95%
intervals overlap the pre-correction Sionna 2.0.1 intervals at all 11 SNR
points. Relative to the older reference, BER and BLER improve at every SNR,
while 15 dB NMSE improves from `3.565445e-3` to `2.447565e-3`. The run took 2
hours 37 minutes 15 seconds on one NVIDIA L40S shared with a concurrent Melgar
training study. This corrected result supersedes the benchmark figures in the
following 2026-08-04 addendum.

Sionna 2.0.1 fitting-selection addendum (2026-08-04): a fresh 30-trial Optuna
study selected raw fitting parameters `num_steps=1597`,
`learning_rate=0.0006990012178445972`, and
`input_noise_scale=1.7023411222360874`. Independent canonical-size validation
and a nearby three-significant-digit search selected `num_steps=1597`,
`learning_rate=0.000698`, and `input_noise_scale=1.70`. The rounded point's
weighted NMSE was `5.871749e-3`, compared with `6.170669e-3` for the previous
operating point on the independent holdout. This addendum supersedes the older
fitting selection below. A subsequent detector calibration selected raw
`err_var_scale=0.33356249642770297`; independent canonical-size validation
selected its three-significant-digit value `0.334`, which is now the estimator
and runner default. The 63-batch Sionna 2.0.1 benchmark then achieved weighted
NMSE `6.110896e-3`, compared with `6.682170e-3` for the pre-migration
reference. NMSE improved from 2.5 dB onward, and BER/BLER improved at every
SNR. The run completed in 2 hours 23 minutes.

Implementation-audit addendum (2026-07-15): the human-approved LS pilot-grid
adaptation was reread against Equations (10)-(17), corrected to use the
paper's bias-free 1x1 linear maps, and covered by focused architecture,
geometry, optimization-independence, and compile-boundary tests. A fresh
30-trial Optuna study selected `hidden_channels=32`, `num_steps=764`,
`learning_rate=0.0013999838509451525`, and
`input_noise_scale=0.4597935254266679`. The complete 63-batch canonical
benchmark was then regenerated. Its weighted NMSE over -10 dB through 10 dB
is `6.682170e-3`; performance is credible but has a documented high-SNR floor
of `3.565444e-3` at 15 dB.

Superseded detector-calibration addendum (2026-07-17): a separate 30-trial Optuna study
selected `err_var_scale=0.38415771061572224`. Its detector objective improved
from `0.23593089176405121` at the prior `1.0` to `0.2298122344058279`. The
improvement reproduced on the independent canonical-size holdout, from
`0.24501265888473742` to `0.23822450948390014`. The selected value is now the
historical pre-migration value. It is superseded by the Sionna 2.0.1
calibration above.

The historical raw-grid evidence is retained below because it motivated the
human-approved pilot-observation adaptation documented in
`implementation_report.md`. The current assessment and verdict apply to that
revised implementation.

Current repository status: accepted with the documented adaptations and
high-SNR red flag. The historical "not accepted" verdict below applies only to
the superseded raw-grid implementation.

Reviewer role: mandatory final reviewer after implementation.

Sources reviewed: `AGENTS.md`, `prompts/reviewer.md`,
`prompts/task_reviewer.md`, `prompts/task.md`,
`src/sionna_bench/mimo_ofdm_chest/papers/balevi2020/paper.pdf`, `reports/paper_summary.md`,
`reports/plan.md`, `reports/plan_review.md`,
`reports/implementation_report.md`, `README.md`, `estimator.py`,
`run_benchmark.py`, and `test/mimo_ofdm_chest/papers/balevi2020/test_estimator.py`.

Additional lightweight checks run for the superseded implementation:

```bash
python -m pytest -q test/mimo_ofdm_chest/papers/balevi2020/test_estimator.py
```

Result: `3 passed`.

Stored smoke result files were also inspected. No heavy benchmark was run.

## Current Overall Assessment

The current code is executable and traceable to the human-approved revised
implementation plan. It forms active LS pilot observations, fits an independent
bias-free 1x1-convolutional deep decoder to each frequency-time pilot grid,
returns the fitted pilot estimates, and delegates resource-grid completion to
the configured Sionna interpolator. The implementation and README disclose that
moving the fit from the complete received grid to the LS pilot grid reverses the
paper's denoising-before-LS order and is a material benchmark adaptation.

The current implementation is accepted for benchmark use with that explicit
qualification. Its architecture and independent batched fitting were checked
directly, its runtime fitting parameters and detector-facing variance scale
were selected on validation probes and confirmed on independent holdouts, and
the complete canonical benchmark was regenerated under Sionna 2.0.1.

## Superseded Raw-Grid Finding

The original implementation fitted the network to the complete coded received
grid. Its smoke results were a serious red flag:

```text
DeepPriorChannelEstimator, 5 dB smoke:
  NMSE = 1.01216809, BER = 0.501266, BLER = 1.0

Stored smoke baselines, same config:
  ls-nn  NMSE = 0.03762552
  ls-lin NMSE = 0.01512823
  lmmse  NMSE = 0.00678475
```

Longer fitted-DNN probes remain near unusable NMSE:

```text
num_steps=25    NMSE = 0.99830894
num_steps=250   NMSE = 0.98097722
num_steps=2000  NMSE = 0.94552907
```

The `num_steps=0` diagnostic bypass gives a credible LS-style result:

```text
num_steps=0     NMSE = 0.01014598, BER = 0.04294658, BLER = 0.25
```

This established that pilot extraction, LS division, and interpolation were
wired correctly, while fitting the network to arbitrary QAM-bearing resource
elements was not credible under the repository's observation model.

## Adaptation Decision

Balevi et al. fit the untrained DNN to a received OFDM tensor, then apply an
LS pilot operation to the generated/denoised signal. Their system model and
simulations use pilot-bearing OFDM grids and `64 x 64` time-frequency channel
blocks; they do not demonstrate fitting the DNN to a coded Sionna resource
grid containing arbitrary QAM data resource elements.

The superseded implementation applied Equation (16) to the full coded received
grid, including data-bearing resource elements. The smoke evidence showed that
this corrupted the pilot observations rather than denoising them. The approved
replacement preserves the fitted-network inference step but applies it to the
active LS pilot-observation grids. This gives the network a channel-like target
and makes the estimator reusable with sparse pilot patterns, at the cost of a
clearly documented departure from the paper's operation order.

## Validation And Test Coverage

The focused tests cover deterministic upsampling dimensions, zero-step
equivalence to Sionna LS estimation, finite and non-degenerate fitted outputs,
and fitted inference across changed pilot-grid, UE, and BS-antenna dimensions.
They compare vectorized fitting with separate optimization problems and verify
that network parameters, normalization statistics, and Adam state are not
shared. They also establish the tensor-only denoiser's compilation behavior and
compiled/eager equivalence around the explicitly disabled online-fitting
boundary. The Sionna 2.0.1 validation and benchmark results provide numerical
evidence beyond tensor-shape correctness.

## Not-Implemented Claims

The not-implemented classifications remain accepted:

- Pilot-contamination and inpainting experiments require a benchmark scenario
  with neighboring-cell pilot reuse, interference masks, and SIR controls.
- Paper MMSE and genie-aided MMSE curves are reference baselines, not the
  proposed Balevi estimator, and exact reproduction would require separate
  covariance/oracle-statistics assets.

## Divergence Ledger

| Divergence | Source | Reviewer verdict | Rationale | Impact on paper faithfulness | Impact on benchmarking fairness |
|---|---|---:|---|---|---|
| Fitting active LS pilot grids rather than the complete received grid | interface | accepted adaptation | The coded resource grid contains arbitrary QAM data outside pilot locations, for which Equation (16) produced near-random estimates. The human approved fitting channel-like LS pilot observations instead. | Reverses the paper's denoising-before-LS order while retaining observation-specific untrained-network fitting. | Produces a deployable pilot-domain estimator; the material departure is disclosed in the README and result interpretation. |
| Sparse Sionna Kronecker pilots followed by configurable interpolation | interface | necessary | The paper assumes pilots over all subcarriers and does not define interpolation for the benchmark's sparse pilot pattern. | Full-grid completion is not defined by the paper. | The canonical `lin_time_avg` choice is explicit and shared with reference estimators. |
| Superseded fitting of the full coded Sionna grid, including data REs | interface | rejected | The paper does not validate Equation (16) on arbitrary coded QAM data REs, and smoke results showed that adaptation destroyed estimator quality. | Major loss of faithfulness to the paper's simulation signal model. | Removed from the current implementation. |
| Fixed `num_steps` replacing true-channel-NMSE early stopping | paper | necessary | Paper tuning uses true-channel NMSE, which is unavailable to a deployable estimator. | The fitting objective is preserved, but the stopping rule differs. | The fixed budget was selected by a predeclared Optuna validation study and independent holdout. |
| Non-power-of-two bilinear upsampling schedule for `624 x 14` grids | interface | justified | The paper's `64 x 64` dimensions divide cleanly through repeated upsampling; the benchmark dimensions require deterministic resizing. | Keeps the deep-decoder structure while changing exact layer sizes. | May affect performance, but the policy is deterministic and tested for shape. |
| Detector-facing LS error-variance fallback with `err_var_scale` | interface | necessary | The paper does not define Sionna marginal channel-estimation error variances. | Does not change `h_hat`, but adds receiver metadata outside the paper. | The scale was selected by a separate detector-objective study and independent holdout. |
| Original unvalidated defaults for `err_var_scale`, hidden width, and step count | implementation | resolved | The original audit correctly blocked canonical claims before validation. Sionna 2.0.1 studies selected `hidden_channels=32`, `num_steps=1597`, the documented fitting parameters, and `err_var_scale=0.334`; both selections were checked on independent canonical-size holdouts. | Does not alter the estimator algorithm. | No longer blocks canonical claims; the canonical rerun uses the selected operating point. |
| `num_steps=0` bypass matches LS plumbing | implementation | justified | This is a diagnostic showing Sionna pilot extraction, LS division, and interpolation are correct. | It is not the Balevi method because it removes the fitted DNN. | Useful sanity check only; cannot be reported as Balevi performance. |
| Omission of pilot-contamination/inpainting experiments | interface | justified | The current benchmark lacks interference masks, neighboring-cell pilot reuse, and SIR controls. | Leaves one paper claim unevaluated. | Avoids comparing incompatible scenarios. |
| Omission of paper MMSE/genie-aided MMSE baselines | asset | justified | They are reference baselines and exact genie-aided reproduction needs oracle covariance/statistical assets. | Does not reduce implementation faithfulness of the proposed estimator. | Built-in LS/LMMSE baselines are sufficient for this benchmark pass. |

## Final Verdict

Accepted for canonical benchmark use with a material, human-approved
pilot-observation adaptation. The implementation retains the paper's defining
observation-specific untrained-network fit, but it is not a literal
reproduction of the paper's denoising-before-LS signal model. The selected
Sionna 2.0.1 operating point is validated, the complete estimator's Tier C
compilation boundary is documented, and the canonical result includes the
calibrated detector-facing variance scale.
