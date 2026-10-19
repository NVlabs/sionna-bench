# Implementation Plan: Li et al., 2020

Plan version: 2026-06-23 revision after FullGridChannelEstimator contract
update.

## Harness Summary

The benchmark evaluates Sionna OFDM channel estimators inside `evaluation/`.
`evaluation.system.OFDMChannelEstimationSystem` builds a `ResourceGrid` with a
Kronecker pilot pattern, maps LDPC-coded QAM symbols, samples the configured
channel, calls the injected channel estimator as

```text
h_hat, err_var = estimator(y, no)
```

and then uses Sionna's LMMSE detector and LDPC decoder to compute BER, BLER,
and NMSE. `evaluation.channel_estimation.evaluate_channel_estimator()` builds
the resource grid, instantiates the estimator, runs all SNR points from the
`SystemConfig`, and stores `ebno_dbs`, `nmse`, `ber`, and `bler` in `.npz`
files.

The full 3GPP UMi config is `configs/umi_3gpp_config.py`. It uses:

- `num_prb=52`, hence `fft_size=624`;
- `num_ofdm_symbols=14`;
- Kronecker pilots on OFDM symbols `(2, 11)`;
- Sionna's Kronecker `PilotPattern.mask` marks all 624 effective subcarriers
  on both pilot OFDM symbols for each tx/stream, so the compact pilot vector
  length is `1248`;
- only 312 entries per tx/stream have nonzero pilot symbols: 156 active
  subcarriers on each of the two pilot OFDM symbols. Sionna's
  `LSChannelEstimator` returns zero LS estimates and zero error variance for
  zero-valued pilot entries, so the paper-level pilot observations `Y_p, X_p`
  correspond to these nonzero entries;
- four UEs, one stream per UE;
- a `4 x 4` single-polarized BS array, hence 16 receive antennas;
- UMi uplink, pathloss and shadow fading disabled, UE speed 3 km/h;
- Eb/N0 grid from `-10` dB to `15` dB in `2.5` dB steps;
- `batch_size=4`, `max_mc_iter=63`, `precision="single"`, `device="cuda"`.

The smoke config uses the same physical setup but only one 5 dB SNR point,
`batch_size=2`, and `max_mc_iter=1`.

Implemented channel estimators must return:

```text
h_hat   : [B, num_rx, num_rx_ant, num_tx, num_streams_per_tx,
           num_ofdm_symbols, fft_size] complex
err_var : same shape as h_hat, float
```

The current task contract permits two estimator architectures. The default is
Sionna's `BaseChannelEstimator` pattern, where the estimator implements
`estimate_at_pilot_locations()` and Sionna applies an explicit interpolation
or custom `BaseChannelInterpolator`. A paper may instead use
`src.full_grid_channel_estimator.FullGridChannelEstimator` and implement
`call(y, no)` directly when the paper explicitly treats selected pilot
resource elements as a low-resolution observation grid and the proposed
algorithm itself reconstructs the full channel grid.

Li et al. explicitly formulate the task as super-resolution from pilot
observations to the full OFDM channel image. Therefore the reviewed target
architecture should be `FullGridChannelEstimator`, not an LS estimator with a
custom interpolator hidden behind Sionna's pilot/interpolation split.

Built-in baselines live under `src/sionna_bench/mimo_ofdm_chest/papers/builtin/` and currently include
`ls-nn`, `ls-lin`, `lmmse`, and `perfect`. Paper outputs live under

```text
outputs/<paper_id>/<config-slug>/
outputs/<paper_id>/smoke/<config-slug>/
```

where `<config-slug>` is produced by `SystemConfig.config_slug()`.

## Method Inventory

### Method: `ReEsNetChannelEstimator`

- Status: `implement`
- Paper location: Section II-A, Equation (3); Section III-A, Equation (7);
  Section III-C, Equation (8); Figure 2 row two.
- Concise method description: compute LS channel estimates at pilot positions,
  represent them as a two-channel low-resolution complex image, and apply the
  ReEsNet residual CNN with transposed-convolution upsampling to estimate the
  full OFDM channel grid.
- Paper inputs: pilot observations `Y_p`, pilot symbols `X_p`, and trained
  ReEsNet weights.
- Paper output: full-grid channel estimate `H_hat`.
- Target repository abstraction: `ReEsNetChannelEstimator`, implemented as a
  `FullGridChannelEstimator` with `PilotGroupMixin`. This is appropriate
  because Li et al. do not merely define pilot-location estimation; ReEsNet is
  the paper's native interpolation/super-resolution algorithm from sparse
  pilot observations to a full OFDM channel image.

#### Implementation Description

The implementation should use the same full-grid pattern as the existing RKHS,
EP, and spectral-GMM estimators:

```text
ReEsNetChannelEstimator.__init__(resource_grid, cfg, ...)
    FullGridChannelEstimator.__init__(
        resource_grid=resource_grid,
        precision=precision,
        device=device,
    )
    PilotGroupMixin._init_pilot_groups(
        group_over=("subcarrier", "symbol"),
        active_only=True,
    )
    initialize fixed pilot-image metadata
    load ReEsNet weights from aux/
    build ReEsNet module on self.device
    store err_var_scale
```

Initialization should:

```text
validate/recover the rectangular pilot image layout:
    fixed coordinates are (tx, stream)
    varying coordinates are nonzero pilot subcarrier and pilot OFDM symbol
    Sionna mask geometry is 624 x 2
    active nonzero pilot geometry is N_pf x N_pn = 156 x 2
register fixed tensors:
    group subcarrier coordinates
    group OFDM-symbol coordinates
    tx/stream group coordinates
    group-to-output tx/stream slots
    pilot values and pilot powers
    effective-subcarrier indices
load a ReEsNet state dict from aux/
```

`ReEsNet` should follow Section III-C. The paper's `N_pf x N_pn` image is
interpreted as the nonzero pilot observation image, not the full Sionna mask
with zero-valued orthogonal pilot placeholders:

```text
input: [num_examples, 2, N_pf, N_pn]

features = Conv2d(2, 16, kernel_size=3, padding=1)(input)
for block in 1..4:
    residual = features
    features = Conv2d(16, 16, kernel_size=3, padding=1)(features)
    features = ReLU(features)
    features = Conv2d(16, 16, kernel_size=3, padding=1)(features)
    features = features + residual
features = Conv2d(16, 16, kernel_size=3, padding=1)(features)
features = ConvTranspose2d(...)(features)
output = Conv2d(16, 2, kernel_size=3, padding=1)(features)
```

The transposed convolution must be configured to output exactly
`[num_effective_subcarriers, num_ofdm_symbols]`, i.e. `156 x 2 -> 624 x 14`
for the active benchmark observations. The plan is to choose stride and output
padding from the fixed benchmark pilot-image and full-grid sizes, and to pass
`output_size` during the forward pass so the output dimensions are
unambiguous. The exact stride/padding formula must be documented in code.

`ReEsNetChannelEstimator.call(y, no)` should:

```text
y_eff = remove nulled subcarriers from y
flatten OFDM-symbol/subcarrier dimensions
gather y_pilots with BaseChannelEstimator's pilot indices
compute LS pilot estimates using Sionna's LSChannelEstimator logic:
    h_ls = y_pilots / pilot_symbols
    ls_err = no / |pilot_symbols|^2
group h_ls with PilotGroupMixin active groups
reshape each tx/stream group to the Li pilot image:
    [B*num_rx*num_rx_ant*num_groups, N_pf, N_pn]
split complex values into real/imag channels:
    network_input = [examples, 2, N_pf, N_pn]
with torch.no_grad():
    network_output = ReEsNet(network_input)
combine real/imag channels into complex full-grid estimates:
    [examples, num_effective_subcarriers, num_ofdm_symbols]
transpose to Sionna output order:
    [examples, num_ofdm_symbols, num_effective_subcarriers]
reshape/scatter by tx/stream group to:
    [B, num_rx, num_rx_ant, num_tx, num_streams_per_tx,
     num_ofdm_symbols, num_effective_subcarriers]
construct detector-facing err_var with the same shape
return h_hat, err_var
```

The paper does not define Sionna detector-facing variances. The implementation
must expose `err_var_scale` and tune it by a bounded validation sweep before
canonical benchmark outputs are generated. The direct full-grid architecture
should still keep this fallback simple: gather the LS pilot error variances,
arrange them in the same `N_pf x N_pn` image convention, interpolate or
upsample them deterministically to the full grid without using learned
variance predictions, and return `err_var_scale * err_var_fallback`. The exact
fallback rule must be documented because it is an interface adaptation, not a
quantity defined in Li et al.

#### Training Asset Plan

The method requires trained weights. Assets should live under:

```text
aux/<aux_slug>/li2020_reesnet/<cache-key>/
    reesnet_params.npz
    manifest.json
```

The cache key must include:

- config slug and scenario hash;
- a pilot mask hash and nonzero pilot-value hash;
- Sionna mask dimensions (`624 x 2` for the active pilot OFDM symbols);
- active nonzero pilot image dimensions (`156 x 2` for each tx/stream);
- pilot OFDM-symbol indices and active subcarrier offsets for every tx/stream;
- ReEsNet architecture parameters (`num_res_blocks=4`, `num_filters=16`);
- optimizer, learning rate, optimizer batch size, epochs, patience, and seed;
- train/validation channel counts;
- training SNR grid;
- channel source and speed distribution used for training;
- loss function (`l1`);
- selected `err_var_scale` when a canonical result is generated.

The estimator must not silently train a canonical asset during normal
benchmark evaluation. The runner should expose an explicit training mode, and
normal benchmark mode should load a matching cache or fail with a clear
message describing the missing asset key. Smoke mode may train a tiny temporary
asset as an integration check.

Training-data generation should mirror Equation (3) and Equation (8):

```text
for each independent channel batch:
    sample UMi channel H from the active config
    for each training SNR in training_snr_dbs:
        gather true pilot channels H_p according to the Sionna pilot pattern
        generate LS pilot observations:
            H_p^LS = H_p + complex Gaussian noise with variance no/|pilot|^2
        form ReEsNet input image from H_p^LS
        form target image from full-grid H
optimize L1(H_hat, H) with Adam
select best checkpoint by independent validation L1 loss
```

The paper trains one network over multiple SNRs from `0` to `20` dB in `5` dB
steps and randomly selects user speed from `0` to `50` km/h when generating
training samples. The benchmark adaptation should expose two modes:

- paper-style training, if feasible, where the UMi channel source samples
  speeds over the paper's range and uses the paper's SNR grid;
- benchmark-specialized training, where the fixed active config is used,
  including UE speed 3 km/h and `cfg.snr_db_range()`.

If canonical outputs use benchmark-specialized fixed-speed training, the README
and implementation report must state that the paper's scenario-generalized
random-speed training mechanism was not reproduced. Smoke training is only an
integration test. Canonical training must use an independent validation
holdout and a bounded sanity sweep over learning rate and optimizer batch
size before writing final outputs.

#### Interface Adaptations

- The paper is downlink and single-antenna; the benchmark is uplink MU-MIMO.
  The implementation will train one shared network across all UEs, receive
  antennas, and streams, treating each tx/stream/antenna channel image as an
  example. This follows the paper's goal of one reusable network but is not
  the paper's exact experiment.
- The benchmark's Sionna mask has all 624 effective subcarriers marked on the
  two pilot OFDM symbols for each tx/stream, but only 156 subcarriers per
  pilot OFDM symbol carry nonzero pilots for a given tx/stream. The
  implementation uses the nonzero entries as the paper-level pilot
  observations. It must document this mask-versus-observation distinction and
  prove it with a layout test.
- The benchmark has active nonzero `N_pf=156` pilot subcarriers per stream and
  `N_pn=2` pilot OFDM symbols, while the paper evaluates 24 or 48 pilots on a
  `72 x 14` grid. Endpoint dimensions and transposed-convolution scaling are
  therefore benchmark-specific.
- The paper's pilot-symbol coordinates are described as sparse/even pilot
  grids. The benchmark's two pilot OFDM symbols are fixed at indices `(2, 11)`.
  The compact two-sample time input does not explicitly encode those absolute
  indices; the network is tied to the fixed pilot pattern through training and
  cache metadata.
- The paper uses L1 loss. Use L1 for the implemented ReEsNet asset, not MSE.
- The paper uses Adam with learning rate `0.001`, minibatch 128, and 100
  epochs. The implementation may expose these as defaults or tuned
  hyperparameters, but canonical values must be selected by validation and
  documented.
- Long training must be initiated by `run_benchmark.py --prepare-assets` or an
  equivalent explicit runner option, not hidden inside ordinary estimator
  construction during a benchmark evaluation.

#### Tests And Verification

Before canonical benchmark output:

- add a lightweight shape/dtype test for `ReEsNet` and
  `ReEsNetChannelEstimator`;
- add a pilot-layout test that asserts:
  - Sionna's mask has 624 marked subcarriers on symbols `(2, 11)`;
  - each tx/stream has 312 nonzero pilot entries;
  - the nonzero pilot entries reshape to `[156, 2]`;
  - active subcarrier coordinates are monotone within each symbol and have
    the expected Kronecker stream offset;
- test that `call(y, no)` returns the required Sionna channel-estimator output
  shape and finite nonnegative `err_var`;
- test cache metadata matching for a tiny smoke asset if feasible;
- run `python -m sionna_bench.mimo_ofdm_chest.papers.li2020.run_benchmark --smoke`;
- compare smoke and canonical outputs against `ls-nn`, `ls-lin`, and `lmmse`
  as sanity checks;
- run a bounded `err_var_scale` validation sweep, for example over
  `[0.01, 0.03, 0.1, 0.3, 1.0, 3.0]`, selecting by held-out detector metric
  or a documented validation criterion that is not the final benchmark result;
- run a non-smoke training attempt with validation logs before any quality
  claim.

#### Risks

- ReEsNet is learned and can fail silently if the training set is too small or
  the transposed-convolution coordinate convention is poor.
- The network does not guarantee pilot pass-through.
- Detector-facing error variances are not defined by the paper.
- Full benchmark training may be expensive because outputs are full-grid
  channel images across 16 receive antennas and 4 UEs.

### Method: `ChannelNet`

- Status: `not implemented`
- Paper location: Section III-B and Figure 2 row one.
- Reason: ChannelNet is a baseline from reference [7], not the proposed method
  of Li et al. The letter only summarizes it and does not provide the full
  SRCNN/DnCNN architecture and training recipe needed for a faithful local
  implementation.
- What would be required: read and review Soltani et al. [7], implement its
  SRCNN and DnCNN architectures, reproduce its separate training procedure,
  and create separate `aux/` assets. This should be treated as a separate
  paper implementation rather than folded into Li et al.

### Method: Conventional LS With Linear Interpolation

- Status: `not implemented`
- Paper location: Section II-A, Equation (3).
- Reason: this is a conventional baseline already available as the repository
  built-in `ls-lin`. It is not Li et al.'s paper contribution.
- What would be required: no paper-local implementation is needed; use
  `python -m sionna_bench.mimo_ofdm_chest.papers.builtin.run_benchmark` for built-in LS baselines.

### Method: LMMSE

- Status: `not implemented`
- Paper location: Section II-B, Equations (4)-(5).
- Reason: this is a conventional baseline already available as the repository
  built-in `lmmse`. It requires channel covariance/statistics and is not the
  proposed ReEsNet method.
- What would be required: no paper-local implementation is needed unless a
  later task explicitly asks for a Li-specific reproduction of the conventional
  baseline.

## Proposed Files

If the reviewed plan is approved, implementation should add:

- `src/sionna_bench/mimo_ofdm_chest/papers/li2020/estimator.py`
  - `ReEsNet`
  - `ResidualBlock`
  - `ReEsNetChannelEstimator`
- `src/sionna_bench/mimo_ofdm_chest/papers/li2020/training.py`
  - training-pair generation
  - cache loading/saving
  - train/validation loop
- `src/sionna_bench/mimo_ofdm_chest/papers/li2020/run_benchmark.py`
  - smoke/full training parameters
  - explicit asset-preparation mode
  - benchmark result writing through `evaluate_channel_estimator`
- `tests/test_li2020_estimator.py`
  - lightweight shape/dtype/cache tests
- `src/sionna_bench/mimo_ofdm_chest/papers/li2020/README.md`
  - after implementation
- `src/sionna_bench/mimo_ofdm_chest/papers/li2020/reports/implementation_report.md`
  - after implementation

No implementation code should be written until `reports/plan_review.md` exists
and the reviewed plan has explicit human approval.
