# Implementation Report

- Paper ID: `edfors1996`
- Implementer: Codex
- Date: 2026-06-04
- Revision note: updated after refactoring `TruncatedChannelEstimator` to
  use Sionna's private `_PilotGroupMixin`, Sionna-style class documentation,
  renamed variants,
  the shared `evaluation.evaluate_channel_estimator` runner path, and an
  explicit task-contract error-variance scale. Updated again after Optuna
  tuning of the LS delay-truncation length, which is reused for all three
  variants. Updated on 2026-07-09 after moving the covariance lifecycle to
  `asset.py`, injecting the prepared matrix into the estimator, and rechecking
  the transform, covariance, truncation, and confidence paths against the paper.
  Updated on 2026-08-04 after correcting complete RNG reset for Sionna 2.0.1,
  regenerating covariance, repeating both tuning studies, and replacing the
  canonical outputs.

## Implemented Files

- `src/sionna_bench/mimo_ofdm_chest/papers/edfors1996/__init__.py`
- `src/sionna_bench/mimo_ofdm_chest/papers/edfors1996/asset.py`
- `src/sionna_bench/mimo_ofdm_chest/papers/edfors1996/estimator.py`
- `src/sionna_bench/mimo_ofdm_chest/papers/edfors1996/run_benchmark.py`
- `src/sionna_bench/mimo_ofdm_chest/papers/edfors1996/tuning.py`

## Implemented Methods

### `TruncatedChannelEstimator (LMMSE)`

Implements the paper's Estimator A on each Sionna nonzero pilot-frequency
vector:

- LS pilot observations are formed with
  `LSChannelEstimator.estimate_at_pilot_locations`, corresponding to Chapter
  3's back-rotated LS observation.
- A unitary IDFT maps each pilot vector to the cyclic time-domain coefficient
  vector used in Fig. 3.2.
- The implementation selects the `L` coefficients with largest empirical
  time-domain channel power.
- It applies the full selected-tap LMMSE matrix
  `R_L (R_L + sigma2 I)^(-1)`, matching Appendix A's Estimator A.
- Excluded taps are zeroed, and a unitary DFT maps back to the pilot-frequency
  vector.

### `TruncatedChannelEstimator (diag. LMMSE)`

Implements the paper's Estimator B on the same pilot-domain vectors:

- Uses the same IDFT, tap ordering, and tap selection as Estimator A.
- Replaces the full selected covariance matrix with scalar diagonal weights
  `sigma_i2 / (sigma_i2 + noise)`, matching Appendix A's Estimator B.
- The canonical run keeps 64 of 156 pilot-domain taps, selecting taps by
  empirical IDFT-domain power. This tap count was selected by optimizing the
  `LS` variant only and is then reused for all three implemented variants. The
  all-tap Estimator B setting discussed in the paper remains available with
  `--num-taps 156`.

### `TruncatedChannelEstimator (LS)`

Implements the paper's Estimator C:

- Uses the same IDFT and tap selection.
- Copies selected time-domain LS coefficients unchanged and zeros excluded
  coefficients.
- With an explicit all-tap override `L=M_p`, this reduces to LS on the pilot
  vector followed by the configured Sionna interpolation.

## Supported System Geometry

The online methods support changed FFT sizes, OFDM-symbol counts, pilot-symbol
positions, UE counts, and BS receive-array dimensions. Pilot processing forms
one frequency vector for each fixed `(tx, stream, pilot OFDM symbol)` tuple.
All active vectors must be nonempty and have one common length because the DFT
and registered pilot-group tensors are rectangular. `num_taps` must not exceed
that length, and the injected covariance must have the corresponding square
shape. Ragged active pilot groups fail during construction with
`_PilotGroupMixin` currently requires equal-length groups; incompatible ordered
pilot coordinates or receive-array geometry fail during covariance-asset
validation. The permanent tests cover two supported geometries and one
intentional ragged-group rejection.

## Algorithm And Confidence Audit

The implementation was rechecked against Chapter 3, Figure 3.2, and Appendix
A. The paper's unitary matrix convention is implemented with matching
`ifft(..., norm="ortho")` and `fft(..., norm="ortho")` calls. The covariance
asset accumulates `E[f f^H]` after that same IDFT. Sorting `diag(R_delay)` by
decreasing power and applying the resulting index order to both the covariance
and observation vector is equivalent to the paper's assumed coefficient
permutation.

For Estimator A, the code applies
`R_L (R_L + sigma2 I)^(-1)`. Its left solve computes the algebraically
equivalent `(R_L + sigma2 I)^(-1) R_L`; the factors commute because the second
is a shifted copy of the first. Estimator B uses only the selected diagonal
powers, and Estimator C uses the identity on the selected coefficients.
Excluded coefficients are zeroed before the matching unitary DFT.

The returned `err_var` is deliberately separate from these paper equations.
It is `chest_err_const` times Sionna's LS observation variance at pilots,
followed by the same configured interpolation used for estimates. It is
therefore a detector-confidence fallback, not an implementation of the
paper's estimation-error covariance. The Sionna 2.0.1 shared aggregate
detector-facing calibration selected `chest_err_const=0.308` across all three
methods with the canonical `lin_time_avg` policy held fixed.

## Auxiliary Asset

The paper-local `asset.py` owns covariance path selection, manifest validation,
loading, estimation, and atomic persistence. Explicit preparation and tuning
use `estimate_covariance()` with an external asset root, while normal benchmark
execution uses strict `load_covariance()` and cannot estimate or write a
missing asset. Both handles return an inference-ready matrix and its selected
manifest path for direct injection and result linkage. `TruncatedChannelEstimator`
has no dependency on repository paths or generation settings.

The validated internal package-local copy of the Sionna 2.0.1 covariance was
stored at:

`src/sionna_bench/mimo_ofdm_chest/papers/edfors1996/assets/3gpp-umi/delay-covariance/it1000-bs64-seed42/`

It contains:

- Files:
  - `time_covariances.npz`
  - `manifest.json`

The asset stores one empirical IDFT-domain covariance matrix pooled across all
`(tx, stream, pilot OFDM symbol)` groups. It is generated from the active Sionna
UMi benchmark channel by sampling true channels, gathering each group's comb
pilot subcarriers, applying the same unitary IDFT used at inference, and
accumulating `E[f f^H]` into a shared covariance. This matches the benchmark
channel symmetry: the second-order statistics should not depend on transmit
index, stream index, or pilot-bearing OFDM symbol.

The Sionna 2.0.1 asset was regenerated from 1,000 batches of 64 complete
four-UE channels (64,000 system realizations), with seed `42`, on an NVIDIA
L40S. Generation took 79.34 seconds. The NPZ payload is 195,850 bytes with
SHA-256
`eea2a270eb004f475b3a6753efde8845ddee88bd3802131fbf3e52a0d4eb426f`;
its covariance and tap-power arrays have shapes `[156, 156]` (`complex64`) and
`[156]` (`float32`). The manifest records the exact ordered pilot groups,
pooled covariance scope, unitary-IDFT convention, recipe, and compatibility.
Its SHA-256 is
`88cce34a7c00fb023410ba4aea3fe2df16919a2adcb3ffbf7618dfd484c5955e`.
Generation resets Python, NumPy, Torch, CUDA, and Sionna's explicit RNGs to
the recorded seed; the Sionna reset is required to reproduce Sionna 2.0.1
channel and topology draws.

## Deviations From Paper

- The paper assumes known transmitted symbols on all OFDM tones of one symbol.
  The benchmark exposes only pilots, so the implementation applies the paper's
  DFT-domain transforms to each compact pilot-frequency vector. This is the
  reviewed first-pass interface adaptation.
- The paper does not specify reconstruction from sparse pilots to all
  time-frequency resources. The estimator exposes Sionna's standard
  `interpolation_type` and custom `interpolator` interface and uses
  `interpolation_type="lin_time_avg"` as its canonical default. The paper does
  not prescribe this OFDM-time policy.
- The paper derives covariance from its analytical channel model and PDP. The
  implementation estimates covariance empirically from the active benchmark UMi
  channel, pools all pilot groups, and caches it under `assets/`.
- The paper treats the retained coefficient count `L` as a
  complexity/performance parameter. The canonical benchmark uses one shared
  `num_taps=64` value for all three variants. This value was selected by
  Optuna tuning of the `LS` variant's weighted NMSE and then reused unchanged
  for `LMMSE` and `diagonal LMMSE`.
- The paper does not define the Sionna detector-facing `err_var` after the
  benchmark adaptation. The implementation uses the task-contract fallback
  `chest_err_const*no/|pilot|^2`. A formal aggregate detector-facing study over
  all three methods selected the shared default
  `chest_err_const=0.308`.

## Verification

Current checks include:

- `python -m pytest -q test/mimo_ofdm_chest/papers/edfors1996`
- `python -m pytest -q`
- `python -m py_compile src/sionna_bench/mimo_ofdm_chest/papers/edfors1996/*.py`
- an isolated `run_benchmark(..., smoke=True)` integration run with a temporary
  covariance asset root and output directory; and
- one temporary `run_tuning()` Optuna trial using the shared tuning helper.

The permanent Edfors tests generate tiny covariance assets for two changed
system geometries and run all three methods with a shared tap count. They also
verify that ragged active pilot-frequency groups fail during construction with
the documented equal-length-group message. The transform-domain tests
establish the following analytical limits:

- retaining all taps in Estimator C equals Sionna LS with the same linear
  interpolation;
- retaining all taps in Estimator A equals Sionna's `LMMSEChannelEstimator` for
  the corresponding frequency covariance;
- a complex non-diagonal delay covariance produces the direct
  frequency-domain LMMSE map, checking covariance orientation;
- retaining one tap preserves only the covariance-ranked strongest
  coefficient;
- a noiseless channel supported on the retained coefficients is recovered
  exactly; and
- `chest_err_const` remains a scalar multiplier after Sionna interpolation.

Covariance loading and estimation are completed before estimator construction.
After replacing the group-noise `.item()` conversions with scalar-tensor
operations, all three complete online methods pass
`torch.compile(..., fullgraph=True)`. A diagnostic `torch._dynamo.explain` run
on the alternate geometry reports one graph and zero graph breaks for each
method, so Edfors online inference is Tier A.

Current runner status:
`sionna_bench.mimo_ofdm_chest.papers.edfors1996.run_benchmark` delegates metric
evaluation and `.npz` writing to `evaluation.evaluate_channel_estimator`. It
prepares the covariance through `asset.py`, injects it into each estimator,
and no longer runs built-in baselines or creates combined plots.

The full default benchmark was rerun after the estimator rename and covariance
cache cleanup and again after formal detector-facing calibration. The default
output directory now contains the calibrated `TruncatedChannelEstimator (...)`
files, a current configuration snapshot, and scenario-local `results.json`.

A final `python -m sionna_bench.mimo_ofdm_chest.papers.edfors1996.run_benchmark --smoke` run passed after
the geometry documentation and detector-facing calibration were complete. All
three methods produced finite NMSE, BER, and BLER at 5 dB. The transient smoke
result directory and its five-batch covariance asset were removed afterward.

Aggregate tap-count tuning:

```bash
python -m sionna_bench.mimo_ofdm_chest.papers.edfors1996.tuning \
  --study taps \
  --asset-root /path/to/external/assets \
  --output-dir /path/to/tuning/edfors1996/taps_v2
```

Result: passed. Fifty Optuna trials minimized linear-SNR-weighted NMSE for the
`LS` method over `-10`, `-5`, `0`, `5`, and `10` dB. Every trial used three
batches of four channels per SNR with sampler seed `123` and probe seed
`4242`. The search took 15.88 seconds and selected `num_taps=61` with objective
`8.155232729752004e-03`. A paired canonical-size validation used 63 batches of
four channels per SNR and independent seed `314159`; the existing 64-tap value
scored `8.89008178325196e-03`, while 61 taps scored
`9.036960406574467e-03`. The independently validated `num_taps=64` is
therefore retained and reused for `LMMSE`, `diagonal LMMSE`, and `LS`. The
validation took 23.09 seconds.

Aggregate detector-facing calibration:

```bash
python -m sionna_bench.mimo_ofdm_chest.papers.edfors1996.tuning \
  --study variance \
  --asset-root /path/to/external/assets \
  --output-dir /path/to/tuning/edfors1996/variance_scale_v2
```

Result: passed. The study held `num_taps=64`, the covariance asset, and all
estimate-producing code fixed. Ten TPE trials sampled `chest_err_const`
logarithmically over `[1e-3, 1]` and minimized mean codeword BLER across all
three methods at 10 dB. Each trial used 63 batches of four channels, sampler
seed `456`, and probe seed `271828`. The search took 56.45 seconds and selected
raw `0.4540521216492649` with objective `0.03196649029982363`. A paired
canonical-size validation with independent seed `161803` scored the existing
`0.308` at `0.033289241622574954` and rounded candidate `0.454` at
`0.0333994708994709`. The existing `chest_err_const=0.308` was therefore
retained. Validation took 20.40 seconds.

Command:

```bash
python -m sionna_bench.mimo_ofdm_chest.papers.edfors1996.run_benchmark
```

Result: passed. The latest run loaded the existing pooled covariance asset,
evaluated all three variants with the shared LS-optimized `num_taps=64` over
the full Eb/N0 grid with `chest_err_const=0.308` and
`interpolation_type="lin_time_avg"`, and saved `.npz` outputs.
`run_benchmark.py` uses `64` as its default tap count.

Default output files:

- `outputs/edfors1996/3gpp-umi/TruncatedChannelEstimator (LMMSE).npz`
- `outputs/edfors1996/3gpp-umi/TruncatedChannelEstimator (diag. LMMSE).npz`
- `outputs/edfors1996/3gpp-umi/TruncatedChannelEstimator (LS).npz`

Validated internal package-local Sionna 2.0.1 covariance copy:

- `src/sionna_bench/mimo_ofdm_chest/papers/edfors1996/assets/3gpp-umi/delay-covariance/it1000-bs64-seed42/`
- Its `time_covariances` array has shape `(156, 156)`, `tap_powers` has shape
  `(156,)`, all entries are finite, and the Hermitian residual is `0.0`.
- Metadata records `covariance_scope=pooled_all_pilot_groups`,
  `array_format=single_matrix_2d`, `num_groups=8`, equal group lengths of 156,
  and `subcarrier_groups`; it no longer records the old per-group
  `(tx, stream, symbol)` metadata.

Default NMSE:

| Estimator | -10 | -7.5 | -5 | -2.5 | 0 | 2.5 | 5 | 7.5 | 10 | 12.5 | 15 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `TruncatedChannelEstimator (LMMSE)` | `5.1921e-02` | `3.6025e-02` | `2.5091e-02` | `1.7623e-02` | `1.2583e-02` | `9.2228e-03` | `7.0165e-03` | `5.5984e-03` | `4.7125e-03` | `4.1772e-03` | `3.8636e-03` |
| `TruncatedChannelEstimator (diag. LMMSE)` | `6.2524e-02` | `4.4315e-02` | `3.1366e-02` | `2.2192e-02` | `1.5765e-02` | `1.1329e-02` | `8.3372e-03` | `6.3867e-03` | `5.1670e-03` | `4.4347e-03` | `4.0083e-03` |
| `TruncatedChannelEstimator (LS)` | `1.4514e-01` | `8.2985e-02` | `4.8173e-02` | `2.8598e-02` | `1.7590e-02` | `1.1400e-02` | `7.9193e-03` | `5.9622e-03` | `4.8618e-03` | `4.2431e-03` | `3.8953e-03` |

## Red Flags And Residual Risks

- The LS-tuned `L=64` benchmark confirms that `diagonal LMMSE` does not
  reproduce the paper's expected best tradeoff under this pilot-domain adaptation;
  `LMMSE` is the strongest of the three variants by NMSE over most
  of the sweep.
- Relative to the prior canonical run, median NMSE ratios are within roughly
  one percent for every method, and every new NMSE, BER, and BLER point lies
  inside the corresponding prior 95% confidence interval.
- The `L=64` truncation still introduces a visible high-SNR floor in
  the Edfors variants, especially relative to the previous all-tap sanity
  checks.
- Estimator C is equivalent to LS followed by the configured interpolation
  only in the explicit all-tap setting; default `LS` zeros the excluded taps.
- The implemented methods are pilot-domain adaptations. They should not be
  described as the literal all-tone receiver analyzed by the paper.
