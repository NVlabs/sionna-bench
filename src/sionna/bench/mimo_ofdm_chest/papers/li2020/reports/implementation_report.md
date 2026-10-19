# Implementation Report: Li et al., 2020

## Approved Plan

Implementation followed the revised `reports/plan.md` accepted in
`reports/plan_review.md`. The human approved the reviewed plan before coding.

The implemented method is `ReEsNetChannelEstimator`, a full-grid Sionna
adaptation of Li et al.'s ReEsNet estimator. `ChannelNet`, paper-local LS, and
paper-local LMMSE remain out of scope as approved in the plan review.

## Implemented Files

- `src/sionna_bench/mimo_ofdm_chest/papers/li2020/estimator.py`
- `src/sionna_bench/mimo_ofdm_chest/papers/li2020/asset.py`
- `src/sionna_bench/mimo_ofdm_chest/papers/li2020/tuning.py`
- `src/sionna_bench/mimo_ofdm_chest/papers/li2020/run_benchmark.py`
- `src/sionna_bench/mimo_ofdm_chest/papers/li2020/README.md`
- `test/mimo_ofdm_chest/papers/li2020/test_asset.py`
- `test/mimo_ofdm_chest/papers/li2020/test_estimator.py`

## Implementation Notes

`ReEsNetChannelEstimator` inherits from `BaseChannelEstimator` and
`_PilotGroupMixin`. It initializes active pilot groups with
`group_over=("subcarrier", "symbol")`, which produces four `(tx, stream)`
groups. For the active benchmark, each group contains 312 nonzero pilot
resource elements, reshaped as a `156 x 2` pilot image.

Explicit asset preparation and tuning call `asset.train_model()`, while normal
benchmark execution calls strict `asset.load_model()`. Both paths return state
dictionaries, optional per-SNR noise variances, and ordered manifest paths.
The estimator therefore has no persistence,
manifest, data-generation, or training responsibility. At runtime,
`call(y, no)` selects the ReEsNet weights matching the current noise variance
and then follows Sionna's pilot-gathering semantics:

1. remove nulled subcarriers;
2. flatten OFDM symbol and subcarrier dimensions;
3. gather compact pilot observations with Sionna's `_pilot_ind`;
4. compute LS pilot estimates via `LSChannelEstimator.estimate_at_pilot_locations`;
5. gather active nonzero pilot groups with `_PilotGroupMixin`;
6. feed real/imaginary `156 x 2` pilot images to ReEsNet;
7. reshape the predicted `624 x 14` full-grid images, using the mixin-guaranteed
   `(tx, stream)` group order, to Sionna's
   `[B, num_rx, num_rx_ant, num_tx, num_streams_per_tx, T, F]` output shape.

Training and checkpoint I/O are outside estimator construction and compiled
inference. With one mixed-SNR state, the complete estimator compiles as one
full graph and is Tier A. The canonical per-SNR bank is Tier B: the Python
selection of one prepared `ModuleList` entry is isolated in
`_select_model_index()` with `torch.compiler.disable`, after which the selected
ReEsNet and the remaining tensor path compile. `torch._dynamo.explain` reports
one graph and no breaks for one state, and two graphs with the one explicit
selection break for two states.

`ReEsNet` implements the paper's residual CNN structure:

- `Conv2d(2, 16, kernel_size=3, padding=1)`;
- four residual blocks, each with two `3 x 3` convolutions and one ReLU;
- one post-block `3 x 3` convolution;
- one transposed convolution with kernel and stride `(4, 7)`, mapping
  `156 x 2` to `624 x 14`;
- final `Conv2d(16, 2, kernel_size=3, padding=1)`.

A focused reread of Section III-C, Figure 2, and EDSR reference [13] confirms
the residual path: ReEsNet retains EDSR's batch-normalization-free residual
blocks, local block skips, post-body convolution, and global feature skip
before late upsampling. Li et al. explicitly replace EDSR's pixel shuffle with
a transposed convolution to permit unequal frequency and OFDM-time scaling.

The paper does not state that transposed convolution's kernel, padding, or
output-padding parameters. The implementation therefore uses the minimal
integer-scale choice `kernel_size=stride` with zero padding and output
padding. It exactly realizes the canonical anisotropic factors `(4, 7)` with
no overlap between transposed-convolution kernels; neighboring context is
still processed by the residual body and final `3 x 3` convolution. This is a
defensible but explicit adaptation. It restricts supported grids to integer
axis-wise scale factors and yields 28,658 parameters for the canonical adapted
network, so the paper's approximate 53k count is not claimed to be reproduced.

The asset path stores checkpoints as:

```text
src/sionna_bench/mimo_ofdm_chest/papers/li2020/assets/<scenario_id>/reesnet/<artifact-id>/
    reesnet_params.npz
    manifest.json
```

The asset metadata includes the scenario identity, exact ordered pilot
coordinates, complex values and powers, architecture settings, training SNRs,
training speed mode, optimizer settings, validation settings, the SNR
augmentation rule, and seed. Detector-facing `err_var_scale` is deliberately
excluded because it changes only inference confidence, not trained weights.

Loading uses the global version-2 envelope and its explicit `compatibility` and
`generation_recipe` sections. The loader
derives the expected state-dictionary keys and tensor shapes from a ReEsNet
with the requested input and output dimensions, then requires exact NPZ array
names, shapes, and `float32` dtypes.

By default, the training routine stores one ReEsNet checkpoint per benchmark
SNR, analogous to the repository's Melgar DNN setup. Each SNR-specific
checkpoint is trained from `train_batches * train_batch_size` independently
sampled channel realizations at that SNR. The previous mixed-SNR asset mode is
still available through `--mixed-snr-weights`; in that mode, each sampled
channel batch generates one noisy LS pilot image for every requested training
SNR.

Normal benchmark evaluation does not silently train a missing canonical
checkpoint. Persistent training is available only through `--prepare-assets`
with an explicit external asset root; `--smoke` may train a tiny ignored asset.

The final calibrated-Sionna bank uses 400 training batches of 16 channels and
16 validation batches of 16 channels at each SNR, at most 200 epochs,
patience 20, relative minimum improvement `1e-4`, optimizer batch size 256,
learning rate `0.000186`, L1 loss, and paper-style uniform 0-50 km/h training
speeds. The fixed bank contains
6,400 training and 256 validation channels. It is stored in `float16` and
converted to `float32` for optimization; model parameters, forward computation,
loss evaluation, and optimization remain in `float32`. These choices are
recorded in each manifest's `generation_recipe`.

Ten full five-SNR trials selected optimizer batch size 256 and rounded learning
rate `0.000186`; paired validation retained that recipe. After freezing all 11
states, a ten-trial BLER calibration at 10 dB tied rounded scale `0.224` and
the existing `0.23432889192166198`. The strict-better rule retained the latter
as the estimator and runner default.

## Deviations From The Paper

- The paper is single-antenna downlink; the benchmark is uplink MU-MIMO. The
  implementation treats each UE stream and BS receive antenna channel image
  as one ReEsNet example and shares one network across them.
- The paper's experimental channel grid is `72 x 14`; the benchmark grid is
  `624 x 14`.
- The paper's pilot settings are 24 or 48 pilots. The benchmark provides a
  `156 x 2` active nonzero Kronecker pilot image per stream.
- The compact time axis uses the two pilot-bearing OFDM symbols as adjacent
  image columns, although their absolute indices are `(2, 11)`.
- The unspecified transposed-convolution geometry is instantiated as
  `kernel_size=stride` with zero padding. This supports different integer
  scale factors on the two axes but not arbitrary noninteger grid ratios.
- The default benchmark runner trains one ReEsNet weight set per benchmark
  SNR. The paper reports one network trained over a coarser SNR set, which is
  still available as the explicit `--mixed-snr-weights` mode.
- The paper does not provide weights, so a local benchmark-trained `assets/`
  asset is required.
- The paper does not define detector-facing error variances. The
  implementation uses a scaled, bilinearly upsampled LS pilot variance image
  as an interface fallback.

## Verification

Commands run:

```bash
python -m py_compile src/sionna_bench/mimo_ofdm_chest/papers/li2020/estimator.py src/sionna_bench/mimo_ofdm_chest/papers/li2020/asset.py src/sionna_bench/mimo_ofdm_chest/papers/li2020/tuning.py src/sionna_bench/mimo_ofdm_chest/papers/li2020/run_benchmark.py test/mimo_ofdm_chest/papers/li2020/test_asset.py test/mimo_ofdm_chest/papers/li2020/test_estimator.py
python -m pytest -q test/mimo_ofdm_chest/papers/li2020
python -m sionna_bench.mimo_ofdm_chest.papers.li2020.tuning --study variance
python -m sionna_bench.mimo_ofdm_chest.papers.li2020.run_benchmark
python -m sionna_bench.mimo_ofdm_chest.papers.li2020.run_benchmark --smoke --prepare-assets --device cpu --precision single --validation-batches 0 --asset-root assets/smoke
python -m sionna_bench.mimo_ofdm_chest.papers.li2020.run_benchmark --smoke --asset-root assets/smoke --device cpu --precision single
```

Results:

- Python compilation: passed.
- Focused Li tests: `22 passed`.
- The estimator tests cover the paper's residual blocks and long skip,
  super-resolution dimensions, ordered pilot grids, exact recovery by a
  constructed linear/pilot-consistent state, and independent stream/receive-
  antenna processing over three system geometries.
- Complete eager and compiled inference agree for both one mixed-SNR state
  (`fullgraph=True`) and a per-SNR state bank (`fullgraph=False`).
- Two temporary one-SNR fixtures train, persist, reload without retraining,
  and run ReEsNet models for `48 x 8` and `72 x 10` full grids.
- Temporary-asset corruption tests reject changed pilot/full-grid
  compatibility, missing state keys, and incorrectly shaped or typed state
  tensors. Checkpoint loading requires only the explicit compatibility and
  generation-recipe metadata.
- Architecture smoke: the runner trained a one-epoch checkpoint under a
  temporary asset root, injected it into the estimator, and completed the
  one-SNR smoke evaluation.
- Smoke asset-preparation run with one SNR-specific checkpoint, one channel, and one
  epoch:
  - training validation L1 after the one smoke epoch: `5.018962e-01`
- Smoke evaluation using that asset:
  - BER: `4.785406e-01`
  - BLER: `1.000000e+00`
  - NMSE: `9.999772e-01`
- Canonical evaluation with the calibrated scale completed for all eleven
  Eb/N0 points. At 5 dB it reports NMSE `4.464318e-03`, BER
  `1.017580e-02`, and BLER `8.201058e-02`; at 15 dB it reports NMSE
  `2.504530e-03`, BER `8.196531e-04`, and BLER `8.597884e-03`.

The smoke model is intentionally tiny and is only an integration check. Its
metrics are poor and should not be interpreted as method quality.

## Canonical Status

The trained per-SNR assets, detector-facing calibration, canonical output, and
leaderboard entry are current. The remaining repository-wide regression and
release checks are tracked in the root `todo.md` rather than as Li-specific
implementation work.
