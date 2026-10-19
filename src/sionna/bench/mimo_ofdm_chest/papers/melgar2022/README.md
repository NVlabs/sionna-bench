# Implementation of the Estimators from Melgar et al., 2022

Publication: A. Melgar, A. de la Fuente, L. Carro-Calvo, O. Barquero-Perez, and E. Morgado, "Deep Neural Network: An Alternative to Traditional Channel Estimators in Massive MIMO Systems," IEEE Transactions on Cognitive Communications and Networking, vol. 8, no. 2, pp. 657-671, 2022.

## Abstract

This implementation covers the Model 2 and Model 3 DNN estimators from Melgar
et al., 2022. Model 2 obtains LS channel estimates at pilot resource elements,
denoises the real and imaginary parts of each pilot-subcarrier vector with a
trained fully connected DNN, and then uses a configured Sionna interpolator for
the remaining OFDM resource elements. Model 3 first forms an LS-linear
full-grid estimate and then denoises full-grid frequency blocks with a trained
DNN.

## Implemented Classes

### `DNN`

`DNN` is the real-valued feed-forward network used for both real and imaginary
parts of a channel vector. It uses hidden layers
`1024, 512, 256, 128, 64, 64, 64`, ReLU activations, and a linear output layer.
The paper instead uses `1000, 500, 256, 128, 32, 16, 8`; the wider
64-dimensional bottleneck is an explicit benchmark-domain adaptation discussed
below.

### `DNNChannelEstimator`

`DNNChannelEstimator` is a Sionna `BasePilotChannelEstimator`. It uses Sionna's
internal `_PilotGroupMixin` to group nonzero pilot subcarriers by transmit
stream and pilot-bearing OFDM symbol, applies LS pilot estimation, denoises
each grouped pilot vector with `DNN`, and delegates full resource-grid
reconstruction to a Sionna interpolator. The default is
`interpolation_type="lin_time_avg"`; callers may select another supported
interpolation type or inject a custom `BaseChannelInterpolator`.

The estimator constructor receives prepared model states. When multiple
SNR-specific states are supplied, it selects the matching network from the
current noise variance. The runner prepares one state per configured SNR by
default; `--mixed-snr-weights` selects a single model trained over an SNR
mixture.

Its main methods are:

- `__init__`: initializes Sionna's channel-estimator base class, builds pilot
  groups, and installs prepared denoiser states.
- `_denoise_values`: applies the real-valued DNN to the real and imaginary
  parts of grouped complex LS pilot estimates.
- `estimate_at_pilot_locations`: computes LS pilot estimates, applies DNN
  denoising to the grouped pilot vectors, scatters the denoised values back to
  Sionna's pilot-vector layout, and returns scaled LS pilot error variances.

Benchmark outputs are named `DNNChannelEstimator`.

### `InterpolatedDNNChannelEstimator`

`InterpolatedDNNChannelEstimator` is a Sionna `BaseChannelEstimator` that
implements the full-grid `call()` contract directly. It first runs Sionna LS
estimation with linear interpolation over the full effective resource grid,
then splits every OFDM symbol into fixed-length frequency blocks and applies
`DNN` separately to real and imaginary parts.

Its main methods are:

- `__init__`: initializes the full-grid estimator, configures an internal
  LS-linear estimator, and installs prepared Model 3 denoiser states.
- `_denoise_blocks`: applies the real-valued DNN to full-grid frequency
  blocks.
- `call`: computes LS-linear full-grid estimates, applies Model 3 DNN
  denoising, and returns the refined full-grid estimates.

Benchmark outputs for this estimator are named
`InterpolatedDNNChannelEstimator`.

Both estimators use the internal `_DNNBank` module to install prepared states
and select the state whose training noise variance matches the current input.

### `asset.py`

`asset.py` owns Model 2 and Model 3 dataset generation, RMSprop training with
MSE loss, best-validation checkpoint selection, manifest validation, and model
persistence. Its method-specific `train_model2()` and `train_model3()` functions
write checkpoints only below an explicitly selected external asset root. The
strict `load_model2()` and `load_model3()` functions load optional
package-local checkpoints or checkpoints from an explicit asset root and never
train or write. Public distributions omit checkpoint payloads.
Canonical execution uses the matching method-specific loader;
asset generation is an explicit `run_benchmark.py --prepare-assets` step.
Estimators never perform file I/O or training.
Model 2 manifests identify the exact ordered active pilot-group coordinates,
values, and powers. Model 3 manifests identify the complete resource-grid and
pilot layout consumed by Sionna's LS-linear interpolation. Loading also
validates the ordered DNN state keys and every tensor's expected shape and
`float32` dtype before returning a state dictionary. Both loaders treat the
explicit `compatibility` and `generation_recipe` sections as authoritative.
Both model types use the global asset-manifest schema version 2. Ordered state
keys and training history live below `metadata`, and per-SNR manifest paths are
returned in the same order as the model states and noise-variance grid.

### `tuning.py`

`tuning.py` exposes `run_tuning()`. It uses the shared Optuna
`tune_channel_estimator()` lifecycle either to prepare trial-specific DNN
assets and tune RMSprop settings against weighted NMSE or to calibrate the
detector-facing variance scale with frozen canonical checkpoints.

### Compilation

Dataset generation, model training, checkpoint validation, and cache loading
run through `asset.py` before estimator construction and are not reachable from
online inference. With one mixed-SNR model state, both complete estimator paths
compile as a single graph. The canonical per-SNR configuration is Tier B:
selecting one prepared model from the Python `ModuleList` is an explicitly
compiler-disabled operation, while the selected DNN, LS estimation, and
interpolation remain in the compiled path. Eager and compiled outputs are
covered by regression tests for both configurations and both estimators.

## Modifications And Omissions

The following modifications have been made compared to the paper:

- The paper's Model 2 example uses 48 pilot resources per block. The benchmark
  pilot pattern has 156 pilot subcarriers per transmit stream and
  pilot-bearing OFDM symbol, so the input and output dimension is adapted to
  156.
- The paper's Model 3 example uses 72-subcarrier blocks. The benchmark has
  624 effective subcarriers, so the default Model 3 block length is adapted to
  78, giving eight equal blocks per OFDM symbol.
- The DNN is trained on the active Sionna 3GPP UMi benchmark channel rather
  than the paper's MATLAB 5G Toolbox CDL-A dataset.
- The current hidden-layer widths are the reviewed h64x3 diagnostic
  architecture `1024, 512, 256, 128, 64, 64, 64`, not the exact architecture
  `1000, 500, 256, 128, 32, 16, 8` stated in Section III-D. Thus, the minimum
  hidden dimension is 64 for both methods, compared with 8 in the paper.
- The same real-valued DNN is applied independently to real and imaginary
  channel vectors and is shared across receive antennas, transmit streams,
  pilot symbols, and frequency blocks. This follows Sections III-B and III-D;
  spatial correlation is deliberately not presented to the network.
- The paper does not define detector-facing channel-estimation error
  variances. Model 2 scales Sionna's LS pilot variance before interpolation;
  Model 3 scales the LS-linear full-grid variance. The independently calibrated
  defaults are documented below and do not belong to checkpoint identity.
- For Model 2, a configurable Sionna interpolator reconstructs off-pilot
  resource elements and the DNN is only applied at pilot locations. The paper
  specifies 2D linear interpolation, available through
  `interpolation_type="lin"`; the repository-standard and canonical default is
  `"lin_time_avg"`, which averages the pilot-symbol estimates in time. Model 2
  checkpoint compatibility therefore does not depend on the selected runtime
  interpolator. For Model 3, Sionna's `"lin"` interpolation is fixed before
  the DNN post-filter, matching the paper's processing order.
- The paper's specialized training uses one model per combination of SNR,
  delay spread, and Doppler frequency. The canonical implementation uses one
  model per SNR while sampling the full configured UMi channel distribution;
  it does not condition separately on delay spread or Doppler. The optional
  mixed-SNR mode corresponds only approximately to the paper's generalized
  model because it mixes SNRs within that one UMi scenario.
- MSE on separate real and imaginary vectors, RMSprop, ReLU hidden
  activations, and a linear output match Section III-D. The implementation
  additionally uses independent validation channels, best-validation
  checkpoint selection, and configurable early stopping.

The following paper variants are not implemented here:

- Model 1 is not implemented because it estimates individual full-grid channel
  coefficients from pilot blocks and coordinates, which would require a
  separate target-coordinate DNN design.

## Detector-Facing Error Variance

The calibrated Sionna 2.0.1 Model 2 and Model 3 assets use separate
`err_var_scale` values because their DNNs act at different stages and produce
different residual errors.

After freezing the final Model 2 bank, ten Optuna TPE trials searched
`err_var_scale` logarithmically over `[1e-3,1]` using codeword BLER at 10 dB,
63 batches of four channels, sampler seed `456`, and probe seed `271828`.
The raw winner `0.4540521216492649` rounded to `0.454`. Independent paired
validation with seed `161803` scored `0.454` at BLER
`0.07605820105820106`, compared with `0.07804232804232804` for the rounded
incumbent `0.273`, so `0.454` was selected.

After freezing the final Model 3 bank, ten Optuna TPE trials searched
`err_var_scale` logarithmically over `[1e-3,1]` using codeword BLER at 10 dB,
63 batches of four channels, sampler seed `456`, and probe seed `271828`.
All trials tied at BLER `0.02513227513227513`; the raw winner
`0.005575293412686262` rounded to `0.00558`. Independent paired validation
with seed `161803` scored the rounded candidate at BLER
`0.020833333333333332` and incumbent `0.3405776426701012` at
`0.02017195767195767`. The strict-lower-wins rule therefore retained the
incumbent. Scaling changes detector confidence and BER/BLER, but not `h_hat`
or NMSE.

The calibration entry points are:

```bash
python -m sionna.bench.mimo_ofdm_chest.papers.melgar2022.tuning --study variance --method model2
python -m sionna.bench.mimo_ofdm_chest.papers.melgar2022.tuning --study variance --method model3
```

## Asset Generation

### Sionna 2.0.1 Model 2 Training Study

Model 2 was retuned with calibrated Sionna commit
`f16208347246d8d17ec69628dfb856b19287b80c`, including calibration commit
`ad1609ea1`, and explicit UMi TR 38.901 version 19.2. The run used an NVIDIA
L40S with driver `580.126.09`, AMD EPYC 7R13, Python 3.12.3, PyTorch
`2.11.0+cu126`, CUDA 12.6, and cuDNN 91002. The dependency snapshot has
SHA-256 `58ea855b22bd1e339e10dd9bd8a4e2079e6b232257e83e87aa030ec4f5bfc1e2`.

One deterministic bank supplied 6,400 training and 256 validation channels
at all eleven canonical SNRs. Every channel contributes 256 real pilot-vector
examples, giving 1,638,400 training and 65,536 validation examples per model.
The fixed-data manifest SHA-256 is
`ea6883bee8506a7d3b0c75036547c38017fd3ffc145fc91ef71287d32f28b727`.

Optuna's `TPESampler(seed=123)` completed 10 requested trials with no failed
or pruned trials. Every trial trained fresh models at
`{-10,-5,0,5,10}` dB for at most 200 epochs, patience 20, relative minimum
improvement `1e-4`, and best-checkpoint restoration. The search covered
learning rate `[1e-5,1e-3]` logarithmically and optimizer batch size
`{256,512,1024,2048}`. Its weighted-NMSE probe used three batches of four
channels per SNR, seed `4242`, and weights
`[0.1,0.31622776601683794,1.0,3.1622776601683795,10.0]`.

The 14:07:06 study selected raw learning rate
`7.01799283113844e-05`, batch size 256, and objective
`0.004693918396679923`. After rounding, paired validation used 63 batches of
four channels per SNR with seed `314159`; learning rate `7.02e-05` and batch
size 256 scored `0.008802547101945238`, versus `0.012489951946848734` for
the prior recipe, and was selected. Paired validation took 3:41:02. The final
eleven-model bank took 6:12:29, occupies 39,270,824 bytes, and passed the exact
asset audit recorded by SHA-256
`fcaf264a77df167481deb0be0da792684ceb5433781b87db2612cf259c050d0e`.

The final audit exposed a bookkeeping defect: payloads already contained the
restored best-validation states, but manifest `train_mse` and `val_mse` fields
had been measured on the last in-memory epoch. All 11 payloads were therefore
reloaded and evaluated against the fixed validation bank. Every result matched
`best_val_mse` exactly, payload hashes remained unchanged, the manifests were
corrected, and the source was fixed to restore the best state before computing
final metrics. The verification record has SHA-256
`e9538675cbaa7f9e39db1ff15e9a843ab4cbb85fddd28175d69cb20b6e13cc05`.

| Eb/N0 (dB) | Best/stopped epoch | Runtime (s) | Validation MSE | Manifest SHA-256 |
| -: | :-: | -: | -: | :- |
| -10 | 115/135 | 1976.6 | `0.0580133046859` | `dbd7efe9e9bb8a8d77d15c76ff4083bfdf948a2b438545f735e3134495ecb7ff` |
| -7.5 | 124/144 | 1797.4 | `0.0384416969923` | `a68e0d8bb315cd6bab064abd3304ed97a2cae0551641ff758ac3df6896a52353` |
| -5 | 86/106 | 1515.7 | `0.0264361244746` | `27f74dd4799f66974be5a15cd03e95704a5f817901b66406164dcc891311d172` |
| -2.5 | 124/144 | 2108.2 | `0.0182142360852` | `e4da507fe8c97a0a3d0f82466f6c040bede551cf8b6b65eabaec21ac0d8aa2b3` |
| 0 | 127/147 | 1913.7 | `0.0136327644189` | `a961b4a20c3bccc15f3202597bf376e3871056ddb4cd3b48d8e78ce2c161eb21` |
| 2.5 | 149/169 | 2457.8 | `0.00970092320289` | `fc5ddfc92f4869eb4ffc7955594d99e1907c0056e60b2e23977ea58b39da2ab6` |
| 5 | 99/119 | 1744.4 | `0.00880826532077` | `17e4e09970594975f8f9ad96e8e162e5198439d88cc33c59d806af98472035be` |
| 7.5 | 196/200 | 2623.8 | `0.00669370171351` | `6dd8228da077949806370b99484c425fb50b614f9e4868b8f1e59555a0878af5` |
| 10 | 149/169 | 1880.8 | `0.00579949038533` | `637c60cd8d6bab8005c1ad1eafe4b209574b1ccb7cbc746ec2ea47fad807fdd8` |
| 12.5 | 196/200 | 2155.1 | `0.00596959057909` | `64cdee527d393b0dd659b7ac4ddbd2545d4860c91294d951be5413c96d86aa6d` |
| 15 | 196/200 | 2154.1 | `0.00430104910181` | `181a8b02f74669c008d64fa0b80ca167c1acc910687844f33bc97697c51f16a9` |

### Sionna 2.0.1 Model 3 Training Study

Model 3 was retuned under calibrated Sionna commit
`f16208347246d8d17ec69628dfb856b19287b80c`, including calibration commit
`ad1609ea1`, with explicit UMi TR 38.901 version 19.2. The run used an NVIDIA
L40S with driver `580.126.09`, AMD EPYC 7R13, Python 3.12.3, PyTorch
`2.11.0+cu126`, CUDA 12.6, and cuDNN 91002. The dependency snapshot has
SHA-256 `d6cce4c165d0d806ce9e12e80ab32e6d2ea0fd87806027ba9a955e1c0dc29f5a`.

One deterministic bank supplied 6,400 training and 256 validation channels at
all eleven canonical SNRs. Each eight-channel generation batch retained at
most 4,096 real frequency-block examples, so one model used 3,276,800
training and 131,072 validation examples. The fixed-data manifest SHA-256 is
`da52a76d051517397c86730a870b2466c897daaa456453a7278af6b2cb500a31`.

Optuna's `TPESampler(seed=123)` completed 10 requested trials with zero failed
or pruned trials. Every trial trained fresh models at
`{-10,-5,0,5,10}` dB for at most 200 epochs, patience 20, relative minimum
improvement `1e-4`, and best-checkpoint restoration. The search covered
learning rate `[1e-5,1e-3]` logarithmically and optimizer batch size
`{256,512,1024,2048}`. Its weighted-NMSE probe used three batches of four
channels per SNR, seed `4242`, and weights
`[0.1,0.31622776601683794,1.0,3.1622776601683795,10.0]`.

The 11:01:56 study selected raw learning rate
`3.168916025578695e-05`, batch size 512, and objective
`0.004829016614663942`. After rounding, paired validation used 63 batches of
four channels per SNR with seed `314159`; learning rate `3.17e-05` and batch
size 512 scored `0.004890910391028793`, versus `0.005637904627207314` for
the prior recipe, and was selected. Paired validation took 2:11:48. The final
eleven-model bank took 3:48:10 and passed an exact asset audit with SHA-256
`40595b56e5c74b9ca5f8a8b151576a53d01f2558b40003e57fd1f5018d438691`.

The first search attempt is retained externally but rejected: it saved the
best state while evaluating final manifest metrics on the unrestored last
state. The corrected implementation restores `best_state` before computing
metadata, and none of the rejected attempt's assets or runtime enters the
selected canonical bank.

| Eb/N0 (dB) | Best/stopped epoch | Runtime (s) | Validation MSE | Manifest SHA-256 |
| -: | :-: | -: | -: | :- |
| -10 | 68/88 | 932.8 | `0.0776076202209` | `58e6f1017ee16fd83108870f18acc10e9c3d4894cab7030602f4d8398963c831` |
| -7.5 | 60/80 | 841.7 | `0.0499930339746` | `a4508c77364384045b5b447a265e00d064d4dd2c8e1380a0069455b7bde4ecfc` |
| -5 | 85/105 | 1109.0 | `0.0314795837188` | `a51fc3e2733032fd5b26511ba0476eb0ee0e7bcc47c8e4ccbfd6d37a4fadb31d` |
| -2.5 | 58/78 | 819.6 | `0.0193806166450` | `2b27a21a269a62b23dafcbedd4166b2004b4e1433a95f66fc7d5a02e96130d38` |
| 0 | 104/124 | 1310.5 | `0.0118010239914` | `b8da39b117bba75ead806b08a12a05ad65f1a4cf665c09f5422874446d6e4652` |
| 2.5 | 97/117 | 1228.3 | `0.00712674273512` | `f790ac10ba52f5e550dcd9d6516f9e9017198489a25d5dbf82ccca5d7d02ef31` |
| 5 | 145/165 | 1737.6 | `0.00425264659600` | `1e2e6dbc525049021fb69bbef272c0eb74e2c3449f36edfaab82e523780e2d3f` |
| 7.5 | 94/114 | 1204.5 | `0.00258626273045` | `c380402e3214bc2fc3ec6fc163f7e70931b4808fcc605a02a65c39bbcc9d6e01` |
| 10 | 155/175 | 1840.0 | `0.00152158289431` | `a4cf61ee22b14c13324a55c375f3935805c2770d7857f430698a340f30d00640` |
| 12.5 | 119/139 | 1468.9 | `0.000931875481724` | `496ebf15f1f149c3aeaeed5a6d167de34cbc7774a001add68d06d877b6db7ee9` |
| 15 | 92/112 | 1179.4 | `0.000612023710393` | `95df245d04aaeedd66d1b2daa13de36e79aff7c9bad12e3d0f828eb63032afbd` |

The study is invoked with `--study training`; independent recipe validation
uses `--study training-validation`. Data/model seed is 42, validation seed
100045, and canonical evaluation seed 2026.

### Asset Preparation Commands

The selected canonical Model 2 checkpoints are generated with:

```bash
python -m sionna.bench.mimo_ofdm_chest.papers.melgar2022.run_benchmark \
  --prepare-assets \
  --asset-root assets/generated \
  --method model2 \
  --train-batches 800 \
  --train-batch-size 8 \
  --validation-batches 32 \
  --validation-batch-size 8 \
  --epochs 200 \
  --optimizer-batch-size 256 \
  --learning-rate 0.0000702 \
  --patience-epochs 20 \
  --min-delta 0.0001 \
  --training-data-manifest "$TRAINING_DATA_MANIFEST"
```

The selected canonical Model 3 checkpoints are generated with:

```bash
python -m sionna.bench.mimo_ofdm_chest.papers.melgar2022.run_benchmark \
  --prepare-assets \
  --asset-root "$FINAL_ASSET_ROOT" \
  --training-data-manifest "$TRAINING_DATA_MANIFEST" \
  --method model3 \
  --block-len 78 \
  --train-batches 800 \
  --train-batch-size 8 \
  --validation-batches 32 \
  --validation-batch-size 8 \
  --epochs 200 \
  --optimizer-batch-size 512 \
  --learning-rate 0.0000317 \
  --patience-epochs 20 \
  --min-delta 0.0001 \
  --max-examples-per-batch 4096
```

Both commands only generate checkpoint assets; they do not run or write a
benchmark result.

The run used `configs.umi_3gpp_config`, i.e. the Eb/N0 grid from `-10` dB to
`15` dB in `2.5` dB steps, `batch_size=4`, `num_mc_batches=63`, four UEs, a `4x4`
BS antenna array, 52 PRBs, and pilot-bearing OFDM symbols `(2, 11)`.

Each final Model 2 and Model 3 checkpoint uses 6,400 training and 256
validation system-channel realizations. Model 2 produces 1,638,400 training
and 65,536 validation pilot-vector examples per SNR; Model 3 produces
3,276,800 retained training and 131,072 validation frequency-block examples
per SNR. The paper reports 800,000 training and 50,000 validation data points
drawn from 20 full channel realizations, so its data-point count must not be
compared directly with this implementation's independently sampled channel
count.

## Benchmark Run

After preparing the selected checkpoints under an external asset root, the
strict benchmark path loads them without training or writing assets. The two
canonical methods are run separately from that same root:

```bash
python -m sionna.bench.mimo_ofdm_chest.papers.melgar2022.run_benchmark \
  --method model2 \
  --asset-root "$FINAL_ASSET_ROOT" \
  --err-var-scale 0.454

python -m sionna.bench.mimo_ofdm_chest.papers.melgar2022.run_benchmark \
  --method model3 \
  --asset-root "$FINAL_ASSET_ROOT" \
  --err-var-scale 0.341
```

Each run uses 63 batches of four channels at every canonical SNR and evaluation
seed 2026. The Model 2 benchmark completed in 59.47 seconds and writes
`outputs/mimo_ofdm_chest/melgar2022/3gpp-umi/DNNChannelEstimator.npz`. Its NMSE values are:

```text
[4.849086e-02, 3.257865e-02, 2.343215e-02, 1.666506e-02,
 1.332787e-02, 1.031226e-02, 9.751628e-03, 8.078198e-03,
 7.200751e-03, 7.612416e-03, 5.452624e-03]
```

Its five-SNR weighted NMSE is `0.008809663267045341`, versus
`0.011443368547024409` in the previous canonical reference. All pointwise
NMSE confidence intervals overlap, while BER and BLER improve materially over
most of the curve. The selection was completed before this benchmark was
inspected.

The final Sionna 2.0.1 Model 3 NMSE values are:

```text
[9.207146e-02, 5.896127e-02, 3.686300e-02, 2.295963e-02,
 1.406740e-02, 8.612731e-03, 5.168823e-03, 3.137101e-03,
 1.853556e-03, 1.124965e-03, 7.306132e-04]
```

The Model 3 benchmark completed in 44.11 seconds. Its five-SNR weighted NMSE
is `0.004788725857026059`, versus
`0.005556121071827339` in the calibrated comparison worktree, a 13.81%
reduction. NMSE confidence intervals no longer overlap from 5 through 15 dB;
BER and BLER are lower at every SNR.

Model 2's high-SNR points remain a known weakness. Training logs show that at
12.5 dB and 15 dB the best pilot-domain validation MSE of the DNN is already
worse than the LS input MSE, so longer training alone is unlikely to remove
the high-SNR error floor. A future revision should consider an
identity-preserving or residual denoising design.

One plausible contributor is the relative bottleneck size. Model 2 maps each
156-dimensional real pilot vector through hidden layers whose minimum width is
64, whereas Model 3 maps each 78-dimensional real frequency block through the
same minimum width. Model 2 therefore imposes substantially stronger
compression when the desired high-SNR mapping approaches the identity. This
has not been isolated experimentally as the definite cause of the error floor.
