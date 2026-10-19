# Implementation Plan: Balevi et al., 2020

Post-review revision on 2026-06-24: the reviewed raw received-grid fitting
plan was implemented and smoke-tested, but it produced near-random channel
estimates because the coded Sionna resource grid contains many data-bearing
resource elements. After human review, the implementation was changed to fit
the untrained network on active LS pilot-observation images instead. The
sections below document the original reviewed plan; the implementation report
records the approved revision and its smoke result.

This is the implementer plan for `papers/balevi2020`. No estimator code should
be written until this plan has been reviewed in `reports/plan_review.md` and
the human has approved proceeding.

## Harness Summary

The benchmark evaluates Sionna OFDM channel estimators inside `evaluation/`.
`evaluation.system.OFDMChannelEstimationSystem` builds a `ResourceGrid`,
generates LDPC-coded QAM symbols, samples a channel through a `ChannelSource`,
calls the injected estimator as

```text
h_hat, err_var = estimator(y, no)
```

and then runs Sionna's LMMSE detector and LDPC decoder. Metrics are computed in
`evaluation.channel_estimation`: BER, BLER, and channel NMSE. Result files are
`.npz` files with `ebno_dbs`, `nmse`, `ber`, and `bler`.

The canonical config is `configs/umi_3gpp_config.py`:

- `num_prb=52`, hence `fft_size=624`;
- `num_ofdm_symbols=14`;
- Kronecker pilots on OFDM symbols `(2, 11)`;
- four UEs with one stream each;
- one BS receiver with a `4 x 4` single-polarized array, hence
  `num_rx_ant=16`;
- UMi uplink, pathloss and shadow fading disabled, UE speed 3 km/h;
- Eb/N0 grid from `-10` dB to `15` dB in `2.5` dB steps;
- `batch_size=4`, `max_mc_iter=63`, `precision="single"`, and `device="cuda"`.

The smoke config uses the same physical setup but runs only the 5 dB point,
with `batch_size=2` and `max_mc_iter=1`.

The estimator input is

```text
y  : [B, num_rx, num_rx_ant, num_ofdm_symbols, fft_size] complex
no : scalar or broadcastable AWGN variance
```

and the estimator must return

```text
h_hat   : [B, num_rx, num_rx_ant, num_tx, num_streams_per_tx,
           num_ofdm_symbols, num_effective_subcarriers] complex
err_var : same shape as h_hat, float
```

The task contract names the last output axis `fft_size`, but Sionna's
`BaseChannelEstimator` and the existing direct-call estimators operate after
`RemoveNulledSubcarriers` and therefore return estimates on the effective
subcarrier grid. For the current config there are no nulled subcarriers, so
`num_effective_subcarriers == fft_size == 624`; the implementation should still
write code against `resource_grid.num_effective_subcarriers`.

Sionna's default `BaseChannelEstimator` extracts pilot observations first,
then calls `estimate_at_pilot_locations()`, and then interpolates. Balevi et
al.'s estimator does not fit this order: Equations (10)-(17) denoise the
received OFDM block before the LS pilot operation. Preserving the paper
therefore requires direct access to the raw received grid `y`.

Built-in baselines are `ls-nn`, `ls-lin`, `lmmse`, and `perfect`, stored under
`outputs/builtin/<config-slug>/`. Paper outputs are stored under
`outputs/<paper_id>/<config-slug>/`; smoke outputs use
`outputs/<paper_id>/smoke/<config-slug>/`.

Existing paper implementations use one public estimator class in
`estimator.py`, a paper-local `run_benchmark.py` that calls
`evaluate_channel_estimator`, explicit method names for leaderboard outputs,
Sionna-style constructor arguments, and README/report documentation of
adaptations. Structured pilot access should use `PilotGroupMixin`.

## Paper-To-Benchmark Mapping

The paper's method is a received-grid denoiser followed by LS. The target
implementation should preserve that order:

```text
raw received grid y
-> remove nulled subcarriers
-> real-stack antenna and complex axes
-> fit untrained DNN to y by Eq. (16)
-> denoised received grid y_star by Eq. (17)
-> gather Sionna pilot observations from y_star
-> LS divide by known Sionna pilot symbols, matching Eqs. (2)-(7)
-> explicit interpolation over the Sionna resource grid
```

The paper assumes pilots on all subcarriers of one pilot OFDM symbol in the
system model. The benchmark instead has a Sionna Kronecker pilot pattern with
two pilot-bearing OFDM symbols and only every fourth active pilot subcarrier
per UE stream. The planned adaptation is:

- denoise the full received grid exactly as the paper proposes;
- apply LS only at the active nonzero Sionna pilot resource elements;
- use explicit Sionna linear interpolation with OFDM-time averaging
  (`"lin_time_avg"`) to fill the channel grid.

This interpolation is not claimed as part of Balevi et al.'s paper. It is a
benchmark adaptation required by the sparse multi-UE pilot pattern.

The proposed estimator should inherit from
`src.full_grid_channel_estimator.FullGridChannelEstimator` and implement
`call(y, no)` directly solely to expose raw `y` to the Equation (16)
optimization before pilot extraction. This is a narrow interface adaptation,
not a claim that Balevi et al. specify native full-grid sparse-pilot channel
reconstruction. If this direct-call adaptation is rejected, the method should be
marked `not implemented` under the current contract because a normal
`estimate_at_pilot_locations()` implementation would discard the paper's
central input.

## Method Inventory

### DeepPriorChannelEstimator

Status: `implement`, with the direct-call architecture limited to preserving
raw received-grid DNN fitting before LS.

Paper locations:

- Equations (1)-(7): received pilot model, LS estimator, and MMSE reference.
- Equation (10): received OFDM tensor over antenna, subcarrier, and OFDM
  symbol axes.
- Equations (11)-(15): untrained DNN parameterization.
- Equation (16): per-block squared-error fitting objective.
- Equation (17): denoised received signal generated by the optimized network.
- Section V-A/B: six hidden layers, Adam optimizer, learning rate `0.01`, and
  hidden width `k` selected by simulation.

Concise method description:

Fit a small untrained deep-decoder/DIP network to the current received OFDM
grid. The network's output is a denoised version of the received grid. Channel
estimates are then obtained by applying LS to denoised pilot observations.

Target repository abstraction:

- Public class: `DeepPriorChannelEstimator`.
- File: `src/sionna_bench/mimo_ofdm_chest/papers/balevi2020/estimator.py`.
- Base class: `FullGridChannelEstimator`.
- Mixin: `PilotGroupMixin`.
- Output method name: `DeepPriorChannelEstimator`.
- Runner: `src/sionna_bench/mimo_ofdm_chest/papers/balevi2020/run_benchmark.py`.

Planned constructor:

```text
DeepPriorChannelEstimator(
    resource_grid,
    *,
    hidden_channels=16,
    num_hidden_layers=6,
    num_steps=250,
    learning_rate=0.01,
    input_noise_scale=1.0,
    seed=42,
    time_avg=True,
    err_var_scale=1.0,
    precision=None,
    device=None,
)
```

Constructor choices:

- `hidden_channels` is the paper's `k`. Section V selects `k=16` for
  `M=64`. The benchmark has 16 BS receive antennas, so `k=16` is the planned
  first default.
- `num_hidden_layers=6` follows Section V-A.
- `num_steps` replaces the paper's oracle early stopping. The paper selects
  epochs by true-channel NMSE in simulation, which is unavailable at runtime.
  A fixed step count must be selected by a bounded validation run, not by final
  benchmark metrics.
- `learning_rate=0.01` follows Section V-A.
- `time_avg=True` selects Sionna's linear interpolation with OFDM-time
  averaging, i.e. the `"lin_time_avg"` policy. This is the only approved
  canonical interpolation policy for Balevi outputs in this plan. A diagnostic
  `"lin"` probe may be added later only as a documented noncanonical run.
  Nearest-neighbor interpolation is not approved here.
- `err_var_scale` exposes the required detector-facing confidence scale
  because the paper does not define Sionna error variances. The value `1.0` is
  only a constructor placeholder until a bounded confidence-scale validation
  sweep selects a canonical value.

Fixed tensors and modules:

- Register the effective-subcarrier indices from the resource grid.
- Use `PilotGroupMixin._init_pilot_groups(group_over=("subcarrier",),
  active_only=False)` or the Sionna `_pilot_ind` directly for compact pilot
  gathering. If grouped active pilots are needed for clean interpolation input,
  use `PilotGroupMixin` rather than reimplementing pilot coordinates.
- Register the fixed random input tensor `Z0` for a given resource-grid shape,
  hidden width, seed, and precision. Shape:
  `[1, hidden_channels, F0, T0]` where `F0` and `T0` are the first entries of
  the deterministic upsampling schedule described below.
- Instantiate the untrained deep decoder network inside the estimator. The
  weights are reset for every received batch or sample before runtime fitting.

Network architecture:

Use a PyTorch module corresponding to Equations (11)-(14):

```text
input z0: [examples, hidden_channels, F0, T0]
for i = 0, ..., num_hidden_layers-2:
    z = Conv2d(hidden_channels, hidden_channels, kernel_size=1)(z)
    z = bilinear_upsample_to_scheduled_size(z)
    z = ReLU(z)
    z = BatchNorm2d(hidden_channels, affine=True)(z)
last hidden layer:
    z = Conv2d(hidden_channels, hidden_channels, kernel_size=1)(z)
    z = ReLU(z)
    z = BatchNorm2d(hidden_channels, affine=True)(z)
output:
    y_hat_real = Conv2d(hidden_channels, 2*num_rx_ant, kernel_size=1)(z)
```

The paper's `64 x 64` simulations divide evenly by `2**5`; the benchmark's
`624 x 14` grid does not. The deterministic resizing policy is:

```text
num_upsampling_layers = num_hidden_layers - 1
size_j = (
    ceil(num_effective_subcarriers / 2**(num_upsampling_layers - j)),
    ceil(num_ofdm_symbols / 2**(num_upsampling_layers - j)),
)
for j = 0, ..., num_upsampling_layers
size_last = (num_effective_subcarriers, num_ofdm_symbols)
```

Each hidden upsampling layer uses
`torch.nn.functional.interpolate(..., size=size_{j+1}, mode="bilinear",
align_corners=False)`. This preserves the paper's repeated bilinear upsampling
idea while ending exactly on the Sionna effective resource grid. No final
cropping and no learned transposed convolution are used. This non-power-of-two
size schedule is an interface adaptation and must be documented in the code,
README, and implementation report.

The paper writes the tensor as `[spatial, frequency, time]`. PyTorch will use
`[channels, frequency, time]`. The benchmark's received tensor should be
converted as:

```text
y_eff = remove nulled subcarriers from y
y_eff: [B, num_rx, num_rx_ant, num_ofdm_symbols, num_effective_subcarriers]
move to [B*num_rx, num_rx_ant, num_effective_subcarriers, num_ofdm_symbols]
real-stack to [B*num_rx, 2*num_rx_ant, num_effective_subcarriers,
               num_ofdm_symbols]
```

The paper fits one network per received signal. To keep that behavior, the
implementation should fit independently for each `B*num_rx` example. A first
implementation can loop over examples for simplicity. If runtime is too high,
a batched implementation with independent per-example networks can be
considered later, but it must not share weights across unrelated channel
realizations unless that change is reviewed.

The target tensor includes all received resource elements, including
data-bearing OFDM symbols. This follows Balevi et al.'s received-grid denoising
order but gives the estimator access to a stronger input surface than
pilot-only estimators. The README and implementation report must disclose this
explicitly.

`call(y, no)` pseudocode:

```text
remove nulled subcarriers:
    y_eff = self._remove_nulled_scs(y)

form real target tensor for each batch/rx example:
    target = real_stack(y_eff)

for each example:
    reset untrained network weights using deterministic seed + call counter
    z0 = fixed random input buffer expanded to one example
    optimizer = Adam(network.parameters(), lr=learning_rate)
    for step in range(num_steps):
        y_fit = network(z0)
        loss = mean_square_error(y_fit, target_example)       # Eq. (16)
        update network by Adam
    denoised_example = network(z0).detach()                   # Eq. (17)

convert denoised real tensor back to complex y_denoised:
    [B, num_rx, num_rx_ant, num_ofdm_symbols, num_effective_subcarriers]

flatten OFDM-symbol/subcarrier axes and gather pilot observations with
Sionna's compact pilot indices:
    y_pilots_denoised = gather(y_denoised, self._pilot_ind)

compute LS estimates:
    h_ls = y_pilots_denoised / pilot_symbols                  # Eqs. (2)-(7)
    ls_err = no / |pilot_symbols|^2
    use zeros where Sionna pilot symbols are zero

interpolate:
    h_hat, err_var = LinearInterpolator(self._pilot_pattern, time_avg=True)
    # Sionna "lin_time_avg" policy; no nearest-neighbor fallback.

scale and clamp detector-facing error variance:
    err_var = err_var_scale * err_var
```

The implementation must not fit to `h_true`, benchmark metrics, decoded bits,
or any label. Runtime fitting uses only the received grid `y`, as in Equation
(16).

Required interface adaptations:

- Uplink instead of the paper's target-cell notation: the benchmark received
  grid is used as the paper's received signal tensor.
- Four UEs instead of the paper's single-user LS notation: after denoising the
  common received grid, pilot extraction and LS division are performed for each
  Sionna tx/stream using known orthogonal pilot symbols.
- Sparse Kronecker pilot pattern instead of pilots on all subcarriers: explicit
  Sionna `"lin_time_avg"` interpolation fills missing subcarriers and OFDM
  symbols.
- Fixed optimizer-step count instead of true-channel-NMSE early stopping.
- Sionna detector-facing `err_var` fallback with tunable `err_var_scale`.
- Non-power-of-two time-frequency grid sizes: deterministic scheduled bilinear
  interpolation replaces the paper's exact repeated factor-two sizes.
- Full-grid fitting includes data-bearing received-grid elements; this is
  inherited from the paper's raw received-grid fitting idea but must be treated
  as a benchmarking caveat.

Auxiliary assets:

None. The paper's estimator is untrained and fits weights per received grid.
No offline checkpoint or covariance asset is required. If later validation
selects canonical values for `num_steps`, `hidden_channels`, or
`err_var_scale`, those values should be documented in the README and
implementation report rather than stored as model assets.

Validation and tuning plan:

- Smoke integration first with small `num_steps` such as 2-5 to verify shapes,
  finite loss, finite `h_hat`, finite nonnegative `err_var`, and result-file
  writing.
- Bounded validation before canonical outputs:
  - `hidden_channels`: `{8, 16, 32}`;
  - `num_steps`: `{25, 50, 100, 250}` if runtime permits;
  - `err_var_scale`: `{0.03, 0.1, 0.3, 1.0, 3.0, 10.0}`;
  - fixed validation SNR subset such as `{-5, 5, 15}` or a documented smoke
    subset, selected before seeing the final benchmark.
- Select `num_steps` and `hidden_channels` by validation NMSE/runtime tradeoff.
  Select `err_var_scale` by validation BLER/BER after the estimator parameters
  are fixed.
- Canonical result files must not be written with the constructor placeholder
  `err_var_scale=1.0` unless that value is selected by the validation sweep.
  The selected candidate grid, criterion, value, and validation result must be
  recorded in `reports/implementation_report.md` and the README.

Focused tests after approval:

- network output shape test for the benchmark resource-grid dimensions;
- LS equivalence test: with `num_steps=0` or a bypass flag, the estimator's
  pilot-domain LS values should match Sionna LS on the same pilot observations;
- real/complex stacking round-trip test;
- nonnegative `err_var` shape/dtype test.

Faithfulness risks:

- The interpolation step is outside the paper.
- Fixed-step runtime fitting is an approximation to the paper's oracle
  early-stopping procedure.
- Fitting one network per received batch element may be very slow in the full
  benchmark.
- The paper's pilot-contamination claim may not translate to the current UMi
  benchmark because no explicit inter-cell contamination regions are injected.
- The scheduled bilinear upsampling sizes are deterministic but not identical
  to the paper's exact factor-two `64 x 64` setting.

Benchmarking risks:

- The estimator sees the entire received grid, including data-bearing resource
  elements. This follows the paper's received-grid denoising idea, but it is a
  stronger input usage than estimators that only use pilot observations.
- Runtime may make a full benchmark impractical without a reduced canonical
  setting.
- High-SNR performance may be limited by interpolation rather than by received
  grid denoising.

### Pilot-Contamination/Inpainting Experiments

Status: `not implemented`.

Paper locations:

- Section II and Equations (1), (3), and (5): pilot contamination terms.
- Section IV discussion after Equation (35): inpainting interpretation.
- Section V-B.3 and Figure 8: random and contiguous contamination experiments.

Reason:

The current benchmark does not include neighboring-cell pilot reuse or explicit
interference patches in selected time-frequency resource elements. The same
`DeepPriorChannelEstimator` can run on contaminated received grids if such a
scenario is added, but reproducing the paper's pilot-contamination experiments
requires a channel/interference source that is outside the current evaluation
contract.

What would be required:

- a reviewed multi-cell or synthetic interference extension to the benchmark;
- configurable random and contiguous interference masks over OFDM resource
  elements;
- documented SIR controls matching Section V-B.3;
- separate baseline outputs generated under that contaminated scenario.

### Paper MMSE And Genie-Aided MMSE Baselines

Status: `not implemented`.

Paper locations:

- Equation (7): LS and MMSE linear estimators.
- Figures 4, 6, 8: comparisons against LS, MMSE, and genie-aided MMSE.

Reason:

These are reference baselines, not the proposed Balevi estimator. The
repository already provides built-in `ls-nn`, `ls-lin`, and `lmmse` baselines
for the active Sionna benchmark. A genie-aided MMSE baseline would require
true channel statistics or oracle covariance information beyond the current
paper implementation scope.

What would be required:

- a separate baseline plan if exact reproduction of the paper's MMSE curves is
  desired;
- covariance/statistical assets or oracle channel statistics for genie-aided
  MMSE;
- output naming that makes clear these are baselines, not Balevi estimator
  variants.
