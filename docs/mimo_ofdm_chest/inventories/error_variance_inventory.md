# Detector-Facing Error-Variance Inventory

This document records how every implemented channel estimator constructs the
`err_var` passed to Sionna's detector. Interpolation of a variance field does
not affect its classification below.

## Classification

- **Scalar calibration** means the paper does not provide a
  detector-facing uncertainty model, or the implementation deliberately adds
  an empirical scale factor. The scale factor must be tuned (for BLER performance) only after the
  estimator, interpolation, hyperparameters, and trained assets are fixed.
- **Derived** means the variance is defined by the implemented mathematical model.

## Shared Calibration Protocol

Scalar detector-facing fallbacks are calibrated only after estimator code,
algorithmic hyperparameters, and trained assets are frozen. Under the canonical
detector-calibration protocol, each calibration uses:

- BLER at 10 dB SNR;
- 10 seeded Optuna TPE trials over the logarithmic interval `[1e-3, 1]`;
- 63 Monte Carlo batches of four channels, matching one canonical benchmark
  SNR point;
- cached channel estimates and unscaled variances whenever the scale cannot
  affect `h_hat`; and
- a paired comparison of the winner (rounded to three significant digits)
  and current value on another 63 batches of four channels at 10 dB.

The selected value must improve on the paired validation; otherwise retain the simpler
prior value. Record both the Optuna and paired validation results. When the scale only
multiplies `err_var`, use `build_cached_detector_scale_objective()` to reuse
the same frozen estimates and unscaled variances across candidates. The scale
may change BER and BLER but must leave `h_hat` and NMSE unchanged.

## Inventory

| Estimator | Unscaled variance source | Current scalar | Classification | Action |
| :- | :- | :-: | :- | :- |
| Built-in LS (linear and nearest-neighbor) | Sionna LS uses `no/abs(pilot)^2` at pilots and propagates it through the selected built-in interpolator. | None | **Derived.** | No scalar calibration. |
| Built-in LMMSE | Sionna's `LMMSEChannelEstimator` derives posterior error variance from the frequency, time, and spatial covariance matrices and observation noise. | None | **Derived.** | No scalar calibration. |
| Perfect CSI | Exact channel with zero error variance. | None | **Derived.** | No scalar calibration. |
| `LSCombEstimator` (all Coleri variants) | Sionna LS pilot variance is averaged across each pilot group, expanded across subcarriers, interpolated over OFDM symbols, and multiplied by the shared `err_var_scale`. This is a detector-facing fallback, not a variance derived for the nonlinear interpolation. | `err_var_scale=0.868` | **Scalar Calibration.** | The calibrated-Sionna detector-calibration study produced raw winner `0.4540521216`. On an independent canonical-size paired validation, rounded `0.454` scored mean BLER `0.0579530423`, while incumbent `0.868` scored `0.0558035714`. The incumbent was retained for all four methods. |
| `RKHSChannelEstimator` | Constant per-receive-antenna fallback `err_var_scale * no` on reconstructed pilot-bearing symbols, followed by time interpolation. | `err_var_scale=0.00102` | **Scalar Calibration.** | The canonical 10-trial 10 dB BLER study selected raw `0.00557529`. Rounded `0.00558` and incumbent `0.00102` both scored `0.0148810` on paired canonical-size validation, so the strict tie rule retained the incumbent. |
| `TruncatedChannelEstimator` (LS, diagonal LMMSE, LMMSE) | Fallback `chest_err_const * no/abs(pilot)^2`; it does not use the covariance/truncation posterior. | `chest_err_const=0.308` | **Scalar Calibration.** | The calibrated-Sionna detector-calibration study gave equal weight to all three methods and produced raw winner `0.4540521216`. On independent paired validation, rounded `0.454` scored mean BLER `0.0333994709`, while current `0.308` scored `0.0332892416`; current `0.308` was retained. |
| `SpatialGMMChannelEstimator` | Marginal covariance of the Gaussian-mixture conditional posterior, including within-component posterior covariance and between-component posterior-mean uncertainty. | None | **Derived.** | No scalar calibration. |
| `SpectralGMMChannelEstimator` | Full-subcarrier marginal covariance of the Gaussian-mixture conditional posterior, then time interpolation. | None | **Derived.** | No scalar calibration. |
| `SBLChannelEstimator` | Sum of scalar Gaussian delay-domain posterior variances, returned at pilot subcarriers and propagated by Sionna linear interpolation. | `posterior_variance_scale=0.755` | **Calibrated posterior-derived approximation.** | The canonical 10-trial 10 dB BLER study selected raw `0.0649144`. Paired canonical-size validation scored incumbent `0.755` at `0.0195106` and rounded candidate `0.0649` at `0.0198413`, so the incumbent was retained. |
| `SparseDiffuseChannelEstimator` | For the Equation (12) pilot-symbol average with weights `w_g`, fallback `err_var_scale * no * sum_g w_g^2/abs(pilot_g)^2`, followed by explicit Sionna interpolation. This reduces to `err_var_scale * no/(G*abs(pilot)^2)` for `G` equal-power pilots. | `err_var_scale=1.0` | **Scalar Calibration.** | The calibrated-source ten-trial 10 dB BLER study selected raw `0.0055752934`. Independent canonical-size paired validation gave identical BLER `0.0201719577` for rounded `0.00558` and incumbent `1.0`, so the strict tie rule retained `1.0`. |
| `DeepPriorChannelEstimator` | Fallback `err_var_scale * no/abs(pilot)^2` from its internal LS pilot operation, returned for both zero-step and fitted estimates. | `err_var_scale=0.454` | **Scalar Calibration.** | The calibrated-source 10-trial 10 dB BLER study selected raw `0.4540521216`. On an independent canonical-size paired validation, rounded `0.454` scored `0.0347222222` and incumbent `0.334` scored `0.0350529101`, so `0.454` was selected. |
| `DNNChannelEstimator` (Melgar Model 2) | Sionna LS pilot variance multiplied by `err_var_scale`, then processed by the configured interpolator; the DNN supplies no uncertainty. | `err_var_scale=0.454` | **Scalar Calibration.** | The ten-trial 10 dB BLER study selected raw `0.4540521216`, rounded to `0.454`. Paired canonical-size validation scored `0.454` at BLER `0.0760582011` and the rounded incumbent `0.273` at `0.0780423280`, so `0.454` was selected. |
| `InterpolatedDNNChannelEstimator` | Sionna LS-linear full-grid variance, unchanged by the DNN except for `err_var_scale`. | `err_var_scale=0.341` | **Scalar Calibration.** | The ten-trial 10 dB BLER study selected raw `0.0055752934`, rounded to `0.00558`. Paired canonical-size validation scored the rounded candidate at BLER `0.0208333333` and the rounded incumbent `0.341` at `0.0201719577`, so the incumbent was retained. |
| `ReEsNetChannelEstimator` | Sionna LS pilot variance, bilinearly resized to the full grid independently of the learned channel reconstruction. | `err_var_scale=0.234` | **Scalar Calibration.** | The ten-trial 10 dB BLER study proposed rounded `0.224`. The candidate and rounded incumbent `0.234` tied at paired BLER `0.03042328`, so the strict-improvement rule retained the incumbent. |
| `VAEChannelEstimator` | Marginal posterior covariance from the VAE decoder covariance and LS observation noise, transformed back to the antenna domain. | `err_var_scale=0.541` | **Scalar Calibration.** | The ten-trial 10 dB study proposed raw `0.4540521216`, rounded to `0.454`. The candidate and rounded incumbent `0.541` tied at paired BLER `0.0211640212`, so the strict-improvement rule retained the incumbent. |

## Calibration Sets

The estimators requiring a scalar calibration study are:

- Coleri `LSCombEstimator` variants;
- Delfeld `RKHSChannelEstimator`;
- all three Edfors `TruncatedChannelEstimator` variants;
- Lyu `SparseDiffuseChannelEstimator`;
- Balevi `DeepPriorChannelEstimator`;
- both Melgar estimators;
- Li `ReEsNetChannelEstimator`;
- the selected Baur VAE method.

Every selected canonical method with a current inference asset has completed
calibration using the pre-release Sionna source that subsequently shipped in
Sionna 2.1.
