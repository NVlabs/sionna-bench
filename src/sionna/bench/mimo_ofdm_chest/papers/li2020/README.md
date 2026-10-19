# Implementation of the Estimator from Li et al., 2020

Publication: L. Li, H. Chen, H.-H. Chang, and L. Liu, "Deep Residual Learning Meets OFDM Channel Estimation," IEEE Wireless Communications Letters, vol. 9, no. 5, pp. 615-618, 2020.

## Abstract

Li et al., 2020 formulate OFDM channel estimation as image super-resolution. The implemented estimator first forms LS estimates at active nonzero pilot resource elements, interprets those estimates as a low-resolution complex image with real and imaginary channels, and then applies the paper's residual convolutional ReEsNet to reconstruct the full OFDM channel grid.

The implementation is adapted to the repository's Sionna uplink MU-MIMO benchmark. The paper's single-antenna downlink setting is not reproduced directly; instead, ReEsNet weights are shared across UE streams and BS receive antennas. The canonical asset contains one prepared weight set per benchmark SNR, and the estimator selects the matching network from the Sionna noise variance.

## Implemented Classes

`ReEsNetChannelEstimator` in `estimator.py` is a Sionna
`BaseChannelEstimator` that implements the full-grid `call()` contract
directly. It uses Sionna's internal `_PilotGroupMixin` to construct the active
pilot grid consumed by ReEsNet.

Main classes and methods:

- `ResidualBlock`: implements the paper's two-convolution residual block.
- `ReEsNet`: implements the Section III-C residual CNN with four residual blocks, a transposed-convolution upsampling layer, and a final two-channel output convolution.
- `ReEsNetChannelEstimator.call()`: gathers Sionna pilot observations, computes LS pilot estimates corresponding to Equation (3), forms the active pilot image, applies ReEsNet's Equation (7) mapping, and returns full-grid Sionna channel estimates.
- `ReEsNetChannelEstimator._init_pilot_image_metadata()`: validates the active benchmark pilot image layout and registers fixed pilot-coordinate tensors.
- `asset.train_model()`: trains mixed-SNR or per-SNR ReEsNet checkpoints under an explicit external asset root and strictly reloads every written checkpoint.
- `asset.load_model()`: strictly loads compatible optional package-local or external checkpoints for evaluation without training or writing.
- `tuning.run_tuning()`: uses the shared Optuna evaluation path to tune ReEsNet training parameters.
- `run_benchmark.prepare_assets()`: explicitly trains the selected model bank and exits without evaluation.
- `run_benchmark.run_benchmark()`: strictly loads prepared model states, injects them into the estimator, and invokes the shared channel-estimator evaluation.

The implemented computation corresponds to:

- Equation (3): LS estimates at active pilot positions;
- Equation (7): learned mapping from pilot image to full channel image;
- Equation (8): L1 training objective;
- Figure 2, row two: ReEsNet residual convolutional architecture.

The architecture follows the adaptation from EDSR in reference [13]. Its
residual blocks omit batch normalization, each block has its own local skip,
and the post-block convolution is added to the first feature map through the
global feature skip shown in both papers. Li et al. replace EDSR's pixel
shuffle with a late transposed convolution so frequency and OFDM-symbol axes
can use different upsampling factors; the implementation preserves that
ordering and uses factors derived from the resource-grid dimensions.

Benchmark outputs are named `ReEsNetChannelEstimator`.

## Modifications and Omissions

The benchmark uses Sionna Kronecker pilots. For each UE stream, Sionna's pilot mask marks all effective subcarriers on the two pilot-bearing OFDM symbols, but only every fourth subcarrier carries a nonzero pilot symbol for that stream. The implementation uses only those nonzero pilot observations as Li et al.'s `Y_p, X_p` data.

The active benchmark pilot image has shape `156 x 2`, corresponding to 156 active pilot subcarriers and two pilot-bearing OFDM symbols. ReEsNet upsamples this to the full `624 x 14` effective OFDM grid.

Li et al. do not specify the transposed convolution's kernel size, padding, or
output padding. This implementation uses the minimal anisotropic construction
`kernel_size=stride=(4, 7)` with zero padding for the canonical grid. More
generally, each output dimension must be an integer multiple of the matching
pilot-grid dimension. This produces the exact requested size without
overlapping transposed-convolution kernels; the low-resolution residual body
and final `3 x 3` convolution still mix neighboring pilot cells and output
blocks. The choice is a conservative implementation of the stated learned
upsampling operation, not a recovered detail of the authors' unavailable
code. Consequently, the canonical adapted network has 28,658 parameters and
should not be claimed to reproduce the paper's approximate 53k parameter
count, which depends on its unspecified upsampler and different pilot grid.

The implementation derives both image dimensions from the resource grid. Its
supported-geometry tests cover grids with 48, 72, and 96 effective
subcarriers; 8, 10, and 12 OFDM symbols; one, two, and three pilot-bearing
symbols; one, three, and four UEs; and two, four, and twelve BS receive
antennas. Each UE-stream/receive-antenna channel image is processed as an
independent ReEsNet batch example while sharing the same trained weights.
Separate one-epoch fixtures train and reload checkpoints for the two smaller
geometries entirely under temporary asset roots.

The paper evaluates a `72 x 14` single-antenna downlink grid with 24 or 48 pilots. The benchmark dimensions, uplink direction, four streams, and 16 BS receive antennas are therefore interface adaptations.

The paper does not provide pretrained weights or its private simulator dataset.
Generate candidate weights with the paper-local training routine and an
explicit external `--asset-root`. Public distributions record the selected
ReEsNet bank's identities and recipes but omit its checkpoint payloads. An
internal distribution may optionally place a complete bank beneath the
paper-local `assets/` tree.

Training, checkpoint selection, and manifest validation live in `asset.py`.
`ReEsNetChannelEstimator` accepts only prepared state dictionaries and their
optional SNR-selection noise variances, so estimator construction performs no
repository I/O or offline training.

With one mixed-SNR state, the complete estimator is Tier A: it compiles as one
full graph. The canonical per-SNR state bank is Tier B. Selecting the matching
prepared state from the Python `ModuleList` is explicitly compiler-disabled,
while the selected ReEsNet, LS pilot processing, and full-grid assembly remain
compiled. Eager and `torch.compile(..., fullgraph=False)` outputs agree.

Checkpoint loading compares the version-2 manifest's explicit `compatibility`
and `generation_recipe` sections. Compatibility covers the exact pilot
grid, full output grid, and network architecture. The loader also verifies the
complete state key sequence and every persisted tensor's array name, shape,
and `float32` dtype before returning a state dictionary. For per-SNR weights,
the returned manifest paths have the same order as the states and their Sionna
noise variances; benchmark results record that ordered asset bank.

The paper trains one network over SNRs from 0 dB to 20 dB in 5 dB steps and samples user speed from 0 to 50 km/h. The runner can reproduce the speed sampling with `--paper-style-training`. The default benchmark path instead trains separate weights for each configured benchmark SNR, analogous to the repository's Melgar DNN setup, so each network sees `train_batches * train_batch_size` independently sampled channels at its own SNR. Use `--mixed-snr-weights` to train or load one network over multiple SNRs.

The enlarged canonical training corpus stores generated inputs and targets
in `float16` to keep the finite dataset resident without reducing its channel
count. Each optimizer minibatch is converted to `float32`; model parameters,
forward computation, loss evaluation, and optimization therefore retain the
original training precision. Training tensors remain on the configured device,
while the independent validation corpus is stored on CPU and transferred in
minibatches.

The paper does not define detector-facing Sionna error variances. This
implementation upsamples the LS pilot error variance grid with deterministic
bilinear interpolation and returns `err_var_scale` times that fallback. The
scale is an inference-only confidence calibration and is not part of
checkpoint identity, so changing it reuses the same trained weights.

ChannelNet is not implemented because it is prior work from Soltani et al., not Li et al.'s proposed estimator, and the letter does not fully specify the separate SRCNN/DnCNN reproduction. Paper-local LS and LMMSE baselines are also not duplicated because the repository already provides built-in `ls-lin` and `lmmse` baselines.

## Detector-Facing Error Variance

After the final checkpoint bank was frozen, ten logarithmic trials searched
`err_var_scale` over `[1e-3, 1]` using BLER at 10 dB. Paired validation used
the canonical 63 batches of four channels with seed `161803`. The rounded
candidate `0.224` and incumbent `0.23432889192166198` tied at BLER
`0.03042328`; the strict-better rule therefore retained the incumbent. Scaling
changes detector confidence, BER, and BLER, but not `h_hat` or NMSE.
The TPE study used sampler seed `456` and probe seed `271828`; it ran 10
requested trials: 10 completed, zero failed, and zero were pruned. Its raw
winner was `0.22435190778458997`, with BLER `0.025793650793650792`.

Reproduce the calibration with:

```bash
python -m sionna.bench.mimo_ofdm_chest.papers.li2020.tuning --study variance
```

The study summary and trials are stored in the external regeneration record.

## Benchmark Run

Prepare a temporary smoke asset, then evaluate it:

```bash
python -m sionna.bench.mimo_ofdm_chest.papers.li2020.run_benchmark \
  --smoke --prepare-assets --asset-root assets/smoke
python -m sionna.bench.mimo_ofdm_chest.papers.li2020.run_benchmark \
  --smoke --asset-root assets/smoke
```

On machines without a CUDA runtime, use:

```bash
python -m sionna.bench.mimo_ofdm_chest.papers.li2020.run_benchmark \
  --smoke --prepare-assets --asset-root assets/smoke \
  --device cpu --precision single
python -m sionna.bench.mimo_ofdm_chest.papers.li2020.run_benchmark \
  --smoke --asset-root assets/smoke --device cpu --precision single
```

The final calibrated-Sionna study used source commit
`f16208347246d8d17ec69628dfb856b19287b80c` and the explicit UMi TR 38.901
v19.2 channel model. It generated one deterministic data bank with 6,400
training and 256 validation channels at each of the eleven canonical SNRs.
The network processes every UE/receive-antenna pair independently, so
each SNR-specific model sees 409,600 training and 16,384 validation grids.
The data/model seed is `42`, the validation seed is `100045`, and the fixed
data manifest SHA-256 is
`255dbe564889f42798c34b443d3a4f28868d06c41e7f3972091fefb320e9730f`.
Its payload records are:

| File | Size (bytes) | SHA-256 |
| :- | -: | :- |
| `train-pilots.npy` | 511,180,928 | `b5f3787a88a9e6b0d42a115d1204a4c7e0ef2e7734fa3b1fc87c4d220d3aa0c4` |
| `train-targets.npy` | 14,313,062,528 | `72469c0a6522cb5c8ffd2439bebfdf9b5fd7176e5d719eea1157ce9500671fd6` |
| `train-pilot-powers.npy` | 255,590,528 | `655af09af58d120542b5aa7004f4b8df7bf02fdf7cdb84a780090848da44a222` |
| `validation-pilots.npy` | 20,447,360 | `2c2edbf642d13afc520ea1749bc18323acfa33ea894e70d7c955d6b862256e61` |
| `validation-targets.npy` | 572,522,624 | `9ef8a21a2cc8001a01ce9e1c4b15b4e04219d67c800375dd8ae495e18ff2221d` |
| `validation-pilot-powers.npy` | 10,223,744 | `1907c712b5632b9c97b976d4649fb37d563c11cd1c145408c03918f933560d7d` |

Every candidate used the full recipe: at most 200 epochs, patience 20,
relative minimum improvement `1e-4`, best-checkpoint restoration, paper-style
speed sampling, and the paper's L1 loss. Ten Optuna trials trained five
SNR-specific models on the same fixed bank and searched learning rate in
`[1e-5, 1e-3]` and optimizer batch size in `{256, 512, 1024, 2048}`. The
weighted-NMSE probe used `{-10, -5, 0, 5, 10}` dB, sampler seed `123`, and
probe seed `4242`, with exact weights
`[0.1, 0.31622776601683794, 1.0, 3.1622776601683795, 10.0]`. The TPE study
ran 10 requested trials: 10 completed, zero failed, and zero were pruned.

The raw winner used learning rate `0.0001856957298624054` and batch size 256.
Paired validation over 63 batches of four independent channels at every
tuning SNR retained the rounded learning rate `0.000186`: its objective was
`0.00445086712`, versus `0.00446277144` for the prior recipe.

Generate the final model bank under an external asset root with:

```bash
python -m sionna.bench.mimo_ofdm_chest.papers.li2020.run_benchmark \
  --prepare-assets --asset-root "$ASSET_ROOT" \
  --training-data-manifest "$TRAINING_DATA_MANIFEST" \
  --train-batches 400 --train-batch-size 16 \
  --validation-batches 16 --validation-batch-size 16 \
  --epochs 200 --optimizer-batch-size 256 --learning-rate 0.000186 \
  --patience-epochs 20 --relative-min-delta 1e-4 \
  --paper-style-training
```

The fixed training-data manifest was generated once with
`generate_training_data()` using the same channel counts, all eleven canonical
SNRs, paper-style speed sampling, seed 42, and an explicit external asset root.
The exact generation call was:

```python
manifest = generate_training_data(
    build_resource_grid(umi_3gpp_config),
    umi_3gpp_config,
    train_batches=400,
    train_batch_size=16,
    validation_batches=16,
    validation_batch_size=16,
    training_snr_dbs=list(umi_3gpp_config.snr_db_range()),
    paper_style_training=True,
    seed=umi_3gpp_config.seed,
    asset_root=Path(DATA_ASSET_ROOT),
)
```

The training study and paired validation are reproduced with:

```bash
python -m sionna.bench.mimo_ofdm_chest.papers.li2020.tuning \
  --study training \
  --output-dir "$TRAINING_STUDY_DIR" \
  --asset-root "$TRIAL_ASSET_ROOT" \
  --training-data-manifest "$TRAINING_DATA_MANIFEST"

python -m sionna.bench.mimo_ofdm_chest.papers.li2020.tuning \
  --study training-validation \
  --output-dir "$TRAINING_STUDY_DIR" \
  --asset-root "$TRIAL_ASSET_ROOT" \
  --training-data-manifest "$TRAINING_DATA_MANIFEST"
```

The selected 11-SNR bank is not included in the public package. After it has
been generated, canonical evaluation loads it without additional training:

| SNR (dB) | Best/stopped epoch | Manifest SHA-256 | Checkpoint SHA-256 |
| -: | :-: | :- | :- |
| -10 | 139/159 | `ccb253374bca3b3f4d34eb3176215b884781b5d67defe787336e35f2b42d856b` | `32cb8484b5703bc7ce05e001e5a04171086debc8988bdf3562efa311adafcc47` |
| -7.5 | 106/126 | `a1caa742ea85b1754761c6855082171f39680849b895fd97612c0783cc01539d` | `779d94dee88a350e7cde4d79947bc3420b6c4a87891a3eda52c63df04541aaa7` |
| -5 | 161/181 | `509f3931a5e7b9113e89a7b2c8b73bcd1d069b9cb0d69981e6140d4c361007cf` | `2355e112ccc8c9e03d27ebd0519d802b3218f8f4ec912cdd4a318af4aa11151d` |
| -2.5 | 126/146 | `bbcafd97ccd87b3be5328bfae02bc6a578c0566dda9b1ffdf4555f3592ac3897` | `b558ce5fb2e30e1c63c14f5ceab4e3ca0903d6f5ccf79404ad2413be2dd82b11` |
| 0 | 126/146 | `6b300c476fac70e4a20f196fc2a9243bc1e9e399aae33bb121b93e0319c2d4a3` | `c5936de3d2c763964b0ad825f5f9559547016d42f65782a5d43b5b2470be3902` |
| 2.5 | 64/84 | `b100d20eca425dd1187e7a3ac1b95c9222f7df717286ac1a350725a136f22429` | `7440a408a296d162924b7cccdc087065f025d54c666d84b4b7f4cbc5badcd5a4` |
| 5 | 64/84 | `9b01969b93e33850d9b0225969106a2d4d76226d46f7f0a540be3a1a4a926d54` | `460ac68fd11d97ded25b71b7fac17f0d67b34e083bd5e2857f90a973ed64722f` |
| 7.5 | 64/84 | `7d029060367ca6d6f83b67d6622c00664219c10e02b176235eb76e8d3243c244` | `4f351c2392708b23f5ad72532fa1a6acfba694529f9bc15dead9f5666ce48b76` |
| 10 | 64/84 | `5cddedd0558871202c2f8b45c3c0dbc8f39b7f9ec30997386dd7a66a58a8008a` | `a9d35b2a99bbafd62aae9f84f94d86c112459933938e5b291fbb1382be9ce0f0` |
| 12.5 | 64/84 | `0c809f2c2dd37cf045a294b6ad5f8dff1fd9e37bf1c8a416e277da1377eb3321` | `5c8dc9ee6e6f3befef537866d55ee6f9339ebf8da4ac2ea1ede9cd51974f5534` |
| 15 | 64/84 | `83f1ae7d287c949090115046535b0f697dfc15dde0add7f806bce34d4e661078` | `753f4edfa685f781768f90866618f2cb965ac4d2f05fe64233fd6f19b1e6bb72` |

Every checkpoint payload is 120,490 bytes.

```bash
python -m sionna.bench.mimo_ofdm_chest.papers.li2020.run_benchmark \
  --asset-root "$ASSET_ROOT"
```

Detector-facing calibration searched ten logarithmic scales over `[1e-3, 1]`
at 10 dB. Paired validation tied the rounded candidate `0.224` and incumbent
`0.23432889192166198` at BLER `0.03042328`; the strict-better rule therefore
retained `0.23432889192166198`.

The complete accepted run took 133,828 seconds (37.17 hours) on an NVIDIA
L40S: 208.85 seconds for data generation, 105,310 seconds for the algorithm
study, 13,344 seconds for paired training validation, 14,894 seconds for the
final bank, 28.56 seconds for variance search/validation, and 42.45 seconds
for the benchmark.

The canonical weighted NMSE is `0.00427391`, versus `0.00498370` in the
calibrated comparison worktree, a 14.2% point-estimate improvement. All
per-SNR NMSE confidence intervals overlap. The benchmark comparison was not
used to revise the selected parameters.

The canonical result is stored in
`outputs/mimo_ofdm_chest/li2020/3gpp-umi/ReEsNetChannelEstimator.npz`.

The retained preflight records Sionna Bench commit
`5012974748d4b8aa6426e563d8fed1380eba3abc` and tracked-diff SHA-256
`7271ee02fc968085f4eeb2bb4d92326aae7336f04c912fc6642ee31319172ce2`.
The environment was Python 3.12.3, PyTorch `2.11.0+cu126`, CUDA 12.6,
driver `580.126.09`, NumPy 2.4.4, SciPy 1.17.1, and Optuna 4.9.0 on an
NVIDIA L40S and AMD EPYC 7R13. The retained `pip-freeze.txt` has SHA-256
`b981919b35253baa1f4cbc244763e09292be6a503d42480d3f9003e397884082`.
Canonical evaluation used seed `2026`.
