# Implementation Plan

Approved basis: plan version 2, 2026-10-25, followed by the human-authorized
reference fit, bounded studies, calibration, and final evaluation. This is a
consolidated record of that design and its approved amendments; the original
reviewed documents remain externally archived. It does not authorize another
experiment. Completed outcomes are in
[implementation_report.md](implementation_report.md).

## Applicable Contracts

- `AGENTS.md` and implementer/channel-estimation prompts: equation-linked
  method inventory, separate review, explicit approval, and reporting of
  adaptations and red flags.
- `CONTRIBUTING.md`: one paper directory with original source, README, and
  Markdown reports; assets, generated results, tests, and shared inventories
  remain outside this contribution.
- `paper_architecture.md`: pilot-domain estimator, explicit interpolation,
  prepared-state construction, separate asset lifecycle, compilation checks.
- `asset_manifests.md`: strict schema-2 compatibility, recipes, tensor metadata,
  checksums, external generation, and result-to-manifest linkage.
- `tuning_protocol.md`: fixed data, disjoint seeds, weighted NMSE selection,
  prior feasibility/cost approval, independent validation, and post-freeze
  detector calibration. Approved new-family exceptions are recorded below.
- `canonical_benchmark_references.md` and shared `todo.md` execution rules:
  transient development observations, selection/evaluation separation,
  mathematical, geometry, dtype/device, and smoke checks.

## Source and Harness Summary

The authoritative algorithm source is arXiv:2403.03545v1 by explicit human
instruction; IEEE remains the publication citation. The arXiv forward model
is Eqs. (2)–(3), reverse posterior Eqs. (5)–(7), SNR Eq. (8), LS Eq. (10),
matching Eq. (11), and deterministic composition Eq. (12)/Algorithm 1.
The released `wcl-2024` code supplies operational CNN/schedule/training details.
Source identities and final-publication access limits are in
[paper_summary.md](paper_summary.md). The later arXiv instruction closed the
original conditional final-IEEE-PDF gate.

The UMi benchmark provides complex received grids
`y: [B,R,A,S,Ffft]` and scalar or leading-prefix noise variance. Outputs are
complex `h_hat` and real `err_var`, both `[B,R,A,U,L,S,Feffective]`.
Canonical non-batch dimensions are `[1,16,4,1,14,624]`: one 4×4 BS array,
four single-stream UEs, 14 symbols, 52 PRBs, pilot symbols 2 and 11.
Each complete realization supplies 1,248 active receive vectors; each UE has
156 active pilot subcarriers on each pilot symbol, with pilot power four.
Noise must be isotropic across antennas within a receive vector.

Use `DiffusionChannelEstimator(BasePilotChannelEstimator, _PilotGroupMixin)`.
The paper estimates a fully observed spatial channel, so denoising belongs in
`estimate_at_pilot_locations()`. Explicit `lin_time_avg` interpolation fills
the OFDM grid; a custom interpolator retains standard precedence. The shared
evaluator owns LMMSE detection, LDPC decoding, NMSE/BER/codeword BLER and
complete-realization confidence intervals. Built-ins are LS-nearest,
LS-linear, LMMSE, and perfect CSI. Matched-interpolation LS is useful for
isolating denoising; existing competitor models must not be regenerated.

## Paper Method Inventory

| Method | Status | Scope and faithful requirements |
| :-- | :-- | :-- |
| Angular deterministic diffusion, 100-step recipe | Implement | Algorithm 1 with new scenario-matched weights and the specification below. |
| Spatial-domain ablation | Not implemented | User scope excludes it; needs a separately trained identity-transform prior. |
| ±3 dB SNR-mismatch experiment | Not implemented | Needs a controlled side-information robustness experiment with separate reporting. |
| Other total-step schedules | Not implemented | Requires schedule-specific training, manifests, and validation. |
| Stochastic sampling, direct clean/mean prediction, other library targets | Not implemented | Not the selected released WCL recipe; requires a separately specified algorithm and loss. |
| Paper LS/Scov/genie/GMM/GMM-Kron/Score comparisons | Not implemented | Separate methods need corresponding assets, statistics, or oracle information; use existing benchmark results only. |
| Later single-step extension | Not implemented | Explicitly outside this contribution. |

## Algorithm Specification

### Representation and CNN

For each active pilot use the full length-`A=rows*columns` receive vector.
Sionna orders antennas as `row + rows*column`; reshape to `[columns,rows]`
and transpose to `[rows,columns]` before `fft2(norm="ortho")`. Invert by
`ifft2`, transpose, and flatten. Do not stack users or pilot frequencies into
a transmit-array axis. No FFT shift, oversampling, or covariance whitening.

Freeze `channel_power = mean_training(abs(h)**2)` in float64. Divide clean
training vectors by its square root. This enforces population unit power,
without per-observation or test-channel normalization. Real/imaginary network
channels are `[M,2,rows,columns]`, with no extra sqrt(2) input factor.

Preserve the original CNN:

```text
features = Conv3x3(32→64)(ReLU(Conv3x3(2→32)(x)))
f[j] = exp(-log(10000)*j/7), j=0,...,7
embedding = concatenate(sin(t*f), cos(t*f))
scale, shift = split(Linear(16→128)(embedding), 64)
features = features*(1+scale[...,None,None]) + shift[...,None,None]
features = ReLU(Conv3x3(64→43)(features))
features = ReLU(Conv3x3(43→22)(features))
epsilon_hat = Conv3x3(22→2)(features)
```

All convolutions have bias, stride one, zero same padding. There is no
activation after the second pre-convolution or final output, and no pooling,
batch norm, clipping, or clean-channel residual shortcut. Parameter count is
55,025. Model timestep `t` is zero-based; the paper calls this state `t+1`.

### Schedule and Training Objective

Construct fixed coefficients in float64, then convert to inference precision
and register them as buffers:

```text
T = 100
beta = linspace(1-10000/10001, 0.1, T)
alpha = 1-beta
alpha_bar[t] = product(alpha[:t+1])
schedule_snr[t] = alpha_bar[t]/(1-alpha_bar[t])
a[t] = 1/sqrt(alpha[t])
b[t] = beta[t]/sqrt(alpha[t]*(1-alpha_bar[t]))
```

For clean normalized angular `x0`, sample uniform `t` and real noise of
variance 1/2 on each component:

```text
epsilon = randn_like(x0)/sqrt(2)
xt = sqrt(alpha_bar[t])*x0 + sqrt(1-alpha_bar[t])*epsilon
loss = mean((cnn(xt,t)-epsilon)**2)
```

Use unweighted epsilon-MSE and Adam with default moments, zero weight decay,
and constant learning rate. No supervised LS-to-clean surrogate, stochastic
reverse noise, velocity target, EMA, or posterior-variance head is introduced.

### Online Estimation

```text
1. Let Sionna extract compact received pilots.
2. Compute LS pilot estimates and v_ls = no/abs(pilot)**2.
3. Gather active receive vectors with _PilotGroupMixin(group_over=()).
   Require isotropic antenna noise within each vector.
4. v = v_ls/channel_power; rho = 1/v.
   Angular observation = physical_fft2(h_ls/sqrt(channel_power)).
5. k = first argmin(abs(schedule_snr-rho)).
   state = real_imag(angular_observation)/sqrt(1+v).
   Use actual v, not the rounded schedule variance; add no random noise.
6. For t=k,k-1,...,0 in each matched-index bucket:
       state = a[t]*state - b[t]*cnn(state,t)
7. Convert to complex, inverse FFT, restore antenna order, multiply by
   sqrt(channel_power). Do not undo the observation factor sqrt(1+v).
8. Scatter active estimates and err_var_scale*v_ls into compact pilot slots.
9. Sionna applies the configured interpolator to both tensors.
```

Algorithm 1 includes the matched transition. The inspected author release
passes `k` to `reversed(range(k))`, executing only `k-1,...,0`; the approved
implementation intentionally corrects that discrepancy and exposes no legacy
index mode. Exact zero noise uses LS directly. Finite out-of-range noise maps
to the nearest schedule endpoint, with first-index tie handling.

For verification, a proper-complex Gaussian prior `CN(0,C)` gives oracle
noise prediction `sqrt(1-alpha_bar[t])*solve(D_t,xt)` with
`D_t=alpha_bar[t]*C+(1-alpha_bar[t])*I`. At a schedule-matched variance the
inclusive chain telescopes to `C @ solve(C+v*I,y)`, including non-diagonal C.
The oracle is test-only. Between schedule points, discretization mismatch is
expected; finite-network exact MMSE is not claimed.

## Assets and Training

One fixed bank contains 6,400 independent training and 256 validation system
realizations, with all derived vectors from a realization kept together.
It supplies 7,987,200/319,488 correlated examples per epoch. One shared
noise-conditional model covers the entire SNR range. This approved new-family
adaptation differs from the named supervised families' per-SNR model banks.

```text
generate_training_data:
  require an explicit external destination
  sample the unmodified configured UMi channel with disjoint train/val seeds
  retain true active-pilot receive vectors; compute training-only power
  save arrays and schema-2 manifest atomically; strictly reload and validate

train_model:
  strictly load the fixed dataset and validate compatibility/checksums
  initialize a fresh original CNN and fixed schedule
  each epoch: shuffle deterministic example identities, sample t/noise by
    role/epoch/example-addressed streams, apply the epsilon-MSE Adam updates
  validate using frozen validation t/noise
  relative improvement controls patience; absolute minimum loss selects state
  restore selected state; persist model/manifest; reload via the public loader

load_model:
  select the requested external artifact or optional read-only package asset
  verify schema, purpose, geometry, pilots, transforms, schedule, normalization,
    ordered state keys, array shapes/dtypes, finiteness, sizes and hashes
  return prepared state, channel power, and selected manifest path
  fail actionably on missing assets; never generate or download implicitly
```

Training permits 200 epochs, patience 20, relative minimum improvement `1e-4`.
Model initialization, shuffle/noise, validation noise, and propagation streams
are separate; minibatch size does not change per-example noise/timesteps.
Complete-epoch checkpoints preserve Adam, absolute epoch, best state and
stopping state; an interrupted partial epoch is replayed. Dataset generation,
loading, training, inference, and evaluation remain separate operations.

## Approved Selection and Calibration

After bounded feasibility and human cost approval, the optimizer study used
10 fresh fits, LR log `[1e-5,1e-3]` and batch sizes `{128,256,512,1024,2048}`.
The original LR1e-4/B512 reference stayed outside new-trial accounting.
Approved amendments set TPE startup five, cap work at ten fits with zero
replacements/216 active hours, reuse the honest single-reference bank, and
confirm only the exact raw winner without an additional rounded fit.
These exceptions supersede the initial proposal's B128 reference and rounded
challenger refit; they do not change the diffusion architecture or loss.

Selection uses cached observations at `{-10,-5,0,5,10}` dB, three batches of
four channels per SNR and normalized weights proportional to `10**(SNR/10)`.
Independent confirmation uses 63×4 realizations per SNR and selects a challenger
only for strictly lower weighted NMSE. The exact tested checkpoint is retained;
canonical observations never choose parameters or checkpoints.

After estimator/assets/interpolation freeze, tune only the empirical variance
scale: ten TPE trials on log `[1e-3,1]`, codeword BLER at 10 dB, 63×4 cached
realizations. Round to three significant digits and compare against the
current scale on a separate paired 252-realization cohort. Adopt only strict
BLER improvement; a tie retains the incumbent. Then run the final eleven-SNR
evaluation once with the frozen selected scale.

| Role | Seed |
| :-- | --: |
| Training / validation propagation | 24250000 / 24260000 |
| Fresh model initialization | 24270000 |
| Epoch/example diffusion and shuffle | 24280000 |
| Fixed validation timestep/noise | 24290000 |
| Algorithm sampler / cached probes / paired confirmation | 24300000 / 24310000 / 24320000 |
| Variance calibration / independent validation / sampler | 24330000 / 24340000 / 24350000 |
| Canonical setup / evaluation | 42 / 2026 |

Propagation streams reserve disjoint role ranges; the detector sets use
`base+[0,62]`. Training RNG keys include role, seed, epoch, and example identity.
Freeze source/dependencies/hardware and data hashes before observations.
Shared `_bounded_study` restores counts/context, not sampler RNG: exact generic
partial stochastic-study continuation is unsupported. The reviewed external
optimizer controller replays its completed-trial prefix; the separate variance
controller refuses to restart partial calibration studies.

## Files and Verification

`estimator.py` owns CNN/online estimation; `asset.py` owns strict loading and
explicit generation/training; `tuning.py` owns study entry points;
`run_benchmark.py` owns construction, standard evaluation and result linkage;
`__init__.py` exports the estimator. Full outputs follow the shared result-store
contract. Public assets and official inventory/leaderboard integration are
maintainer follow-up outside the paper-only contribution.

Focused verification covers physical antenna order/FFT round trips, complex
noise normalization, released-CNN/scheduler parity, correlated Gaussian
conditioning, inclusive reverse counts, zero-noise limits, output geometry,
single/double precision, eager/compiled agreement, strict manifest and purpose
guards, selected-weight identity, and variance-scale invariance. Temporary
smoke checks establish integration, not final performance. Training fixtures
are excluded from final documentation-cleanup verification; existing results
are preserved without new model execution.
