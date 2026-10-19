# Final Review Report: Wei et al., 2017

## Review Scope

This review covers the finalized Wei et al. implementation, its paper-local
documentation and reports, the shared repository contracts, relevant tests,
the Sionna 2.0.1 tuning studies, and the canonical 3GPP UMi result. The source
paper is C. Wei, Z. Zhang, and H. Liu, "Expectation-propagation based
low-complexity channel estimation for massive MIMO systems," *IEEE
International Conference on Communications (ICC)*, 2017.

## Final Verdict

**Accepted with caveats.** The implementation is a coherent and numerically
validated uplink adaptation of the paper's common-support sparse Bayesian
learning model. It must not be described as a faithful implementation of the
paper's complete LC-EM-EP receiver: the comb-pilot transform diagonalizes the
adapted observation model, so the implementation uses exact scalar Gaussian
posterior updates and EM rather than expectation-propagation messages.

No algorithmic or numerical blocker was found. The canonical smoke test,
multi-geometry tests, analytical update checks, and full-graph compilation
probe pass. The three documentation and reproducibility issues found during
review were resolved during integration: the plan now records the final
human-requested revision, the historical Optuna trial-zero configuration is
explicitly replayable, and the estimator output docstring uses
`num_effective_subcarriers`.

## Findings

### Resolved: The approved-plan record was stale

The post-approval update at the top of `reports/plan.md` described an
`ExpectationPropagationChannelEstimator`, full-grid inheritance,
middle-zero-padding reconstruction, and older defaults. The final
implementation is the pilot-domain `SBLChannelEstimator`, uses explicit Sionna
linear interpolation with time averaging, and uses the selected parameters
`num_iter=10`, `alpha_init=0.00259`, and
`posterior_variance_scale=0.755`. The README and implementation report describe
the final design correctly, but the plan no longer provided an accurate record
of the approved implementation history. A final human-requested-revision
section now records the active design while preserving the earlier update as
history.

### Resolved: The documented SBL tuning command was not an exact clean replay

The retained study evidence records the previous incumbent
`num_iter=12`, `alpha_init=1.32`, and `num_taps=156` as trial zero. The current
`tuning.py` formerly enqueued the finalized defaults instead. It now exposes
and enqueues the historical trial-zero values, so the documented command starts
from the same initial observation. The retained CSV/JSON evidence remains the
authoritative record of the completed study and independent validation.

### Resolved: One output dimension was documented incorrectly

The estimator class docstring describes the final channel-estimate dimension
as `fft_size`. The inherited Sionna contract returns
`num_effective_subcarriers`. The docstring now states the inherited contract;
the correction does not affect runtime behavior or the reported result.

## Algorithmic Audit

The implementation correctly represents the following parts of the paper
after specialization to the repository's uplink pilot model:

- A zero-mean conditionally Gaussian delay-domain prior with one precision
  parameter per delay coefficient, corresponding to Equations (3)-(5).
- Gaussian posterior means and variances corresponding to the scalarized form
  of Equations (14) and (16).
- The EM update of Equation (21), with each delay precision shared across the
  receive antennas used as observations of a common support.
- Independent processing of UE, stream, and pilot-symbol groups, followed by
  reconstruction at observed pilot subcarriers.

The omission of explicit EP messages is mathematically defensible for this
adapted model. Dividing by the uniform, equal-power comb pilots and applying the
comb-length inverse DFT produces independent scalar Gaussian observations in
the aliased delay domain. Their posterior moments have closed forms, so an EP
schedule would add no approximation benefit. The class and documentation now
use sparse Bayesian learning terminology and do not claim to implement the
paper's generic LC-EM-EP algorithm.

The following paper components are not implemented and are appropriately
excluded from the claimed method: the generic non-diagonal pilot matrix, the
downlink superposition of multiple transmit antennas, the paper's complete EP
message schedule, block-ISD and belief-propagation variants, oracle support
information, and reproduction of the paper's original channel simulation.

## Divergence Ledger

| Paper behavior | Repository behavior | Assessment |
|---|---|---|
| Downlink observations superpose channels from multiple transmit antennas. | The Sionna uplink benchmark exposes separated receive-antenna observations for each UE stream. | Necessary interface adaptation; materially simplifies inference and is documented. |
| LC-EM-EP handles a generic linear observation matrix. | Uniform comb pilots are transformed into scalar delay-domain observations and solved with exact Gaussian SBL updates plus EM. | Mathematically sound for the adapted model, but not a literal LC-EM-EP implementation. |
| Common delay support is shared across transmit antennas. | Delay precisions are shared across receive antennas. | Reasonable uplink transpose of the common-support assumption. |
| The paper uses its own full-frequency pilot model and discrete delay basis. | The estimator returns denoised pilot-domain values and delegates unobserved resource elements to Sionna's `lin_time_avg` interpolation. | Necessary for the benchmark's sparse comb pattern; interpolation performance is part of the reported method. |
| The paper uses fixed simulation parameters, including `L=128`. | The canonical implementation uses the complete 156-coefficient comb-delay representation and tunes iterations and initialization on Sionna 2.0.1 data. | Defensible scenario adaptation; selected values are documented and validated. |
| The paper does not define Sionna detector-facing error variances. | Marginal posterior variances are reconstructed, interpolated, and multiplied by a calibrated scale of `0.755`. | Required interface adaptation; scale retention was independently validated. |

## Verification

The paper-specific test suite passes with **20 tests**. It covers multiple
comb lengths and antenna counts, changed UE and OFDM-symbol geometries,
nonuniform and incomplete comb rejection, comb-transform identities, exact EM
and posterior updates, high-SNR reconstruction, finite outputs, and
non-negative variances.

The canonical smoke result was reproduced exactly:

- BER: `6.25e-2`
- BLER: `3.75e-1`
- NMSE: `8.851104e-3`

The eager estimator and `torch.compile(..., fullgraph=True, backend="eager")`
agree on the checked alternative geometry. Dynamo captured one graph with no
graph breaks, supporting the documented Tier A classification for online
inference.

The calibrated Sionna 2.0.1 algorithm study completed 50 trials over five SNR
points with three batches of four channels per SNR. Paired canonical-size
validation selected the three-significant-digit setting `num_iter=10`,
`alpha_init=0.00259`, and `num_taps=156`. The separate 10-trial 10 dB
detector-facing study did not reproduce a gain over the incumbent on paired
validation, so retaining `posterior_variance_scale=0.755` is justified.

The canonical output is finite and monotonically improves with SNR. Its NMSE
curve is

```text
[4.313531e-2, 3.054678e-2, 2.117557e-2, 1.459569e-2,
 1.011717e-2, 7.072674e-3, 4.970072e-3, 3.486605e-3,
 2.423694e-3, 1.663590e-3, 1.134483e-3]
```

for Eb/N0 values from -10 dB to 15 dB. The canonical file hash matches the
README and implementation report:
`9ca84a263f13787cfc31e6bd7271a4d0ce8c64d098b6577c6e79a9b482b59ebb`.

## Conclusion

The implementation is suitable for inclusion as
`SBLChannelEstimator`, provided it continues to be presented as an uplink,
comb-specialized sparse Bayesian adaptation rather than as a reproduction of
the paper's full LC-EM-EP receiver. The observed results, compile behavior,
geometry support, selected hyperparameters, and detector-facing variance
handling are supported by the available evidence. The review findings were
resolved without changing estimator inference or the canonical benchmark
result.
