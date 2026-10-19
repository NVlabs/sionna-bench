# Canonical Asset Inventory

This document records the canonical inference assets used by methods that
require fitted channel statistics or trained model parameters. It distinguishes
the assets selected by the canonical runners from external training datasets,
intermediate tuning candidates, and rejected experiments. Public distributions
do not include the payloads. Their recorded identities, generation recipes,
payload metadata, checksums, and validation evidence make the selected assets
auditable and reproducible.

Regenerated assets use
`<asset_root>/<scenario_id>/<paper_id>/<asset_type>/<artifact_id>/`. Every
usable artifact directory contains a versioned `manifest.json` with
compatibility, generation-recipe, payload shape/dtype metadata, file sizes,
and SHA-256 checksums. A private distribution may copy a complete canonical
asset beside its implementation, but public runners do not assume such a copy
exists.
The complete contract is documented in `docs/mimo_ofdm_chest/contracts/asset_manifests.md`.

## Storage And Promotion Policy

Generation and fitting always write beneath an explicit external asset root.
Assets selected by the tuning protocol are strictly validated against their
manifests and identified from scenario-local `results.json`. The result record
stores the asset type, artifact ID, and manifest checksum, not a path to a
distributed payload.

All assets inventoried below are canonical selections generated with Sionna
source commit `f16208347246d8d17ec69628dfb856b19287b80c`. That pre-release
checkout still reported version 2.0.1 and contains the calibrated channel
models and channel-estimation API subsequently released in Sionna 2.1. Exact
historical versions remain recorded with each asset; current regeneration uses
the supported Sionna 2.1 release. Strict loaders validated the selected
external or private package-local copies. Public source and wheel distributions
omit the payloads; users regenerate them with the commands below and pass the
same external asset root to evaluation.

Asset manifests use the global asset schema version 2 with explicit
`compatibility`, `generation_recipe`, `files`, and optional nested `metadata`
sections.
Scenario-local `results.json` entries identify the selected asset type,
artifact ID, and exact manifest checksum. Built-in LS results have no asset
reference; built-in LMMSE and each Edfors method reference only the covariance
used for that evaluation.

## Canonical Asset Recipes

### Built-In LMMSE Covariance

The selected canonical asset was generated with:

```bash
python -m sionna.bench.mimo_ofdm_chest.papers.builtin.run_benchmark \
  --prepare-assets \
  --estimator lmmse \
  --cov-num-iters 1000 \
  --cov-batch-size 64 \
  --asset-root "${RUN_ROOT}/assets" \
  --device cuda:0
```

- External asset path:
  `<asset_root>/3gpp-umi/builtin/lmmse-covariance/default/`.
- Manifest identity: scenario `3gpp-umi`, asset type `lmmse-covariance`,
  artifact `default`, global asset schema version 2, manifest SHA-256
  `05c1f57b08bf3c22c6f84bdcea24849e4a809864070f497d17dd83521d03dc56`.
- Recipe: 1,000 independently sampled batches of 64 single-UE UMi channels
  (64,000 channel realizations in total), using a random single-sector topology
  per batch and component seed 42. The channel, grid, topology, and array
  parameters are recorded in the manifest.
- The canonical runner loads the compatible `default` artifact above. Its
  asset-generation defaults are `cov_num_iters=1000` and
  `cov_batch_size=64`, matching the selected artifact's manifest-recorded
  64,000-channel recipe.
- Reproducibility record: Sionna Bench source
  `5012974748d4b8aa6426e563d8fed1380eba3abc` plus tracked-diff SHA-256
  `27293ab390306a6b31fb640e628f8f7a3a484c64967bcee676314eb70912bc70`,
  Sionna source `f16208347246d8d17ec69628dfb856b19287b80c`, and calibrated
  channel-model commit `ad1609ea1`. The run explicitly used UMi TR 38.901
  version `19.2`, setup seed 42, evaluation seed 2026, Python 3.12.3,
  PyTorch 2.11.0+cu126, NumPy 2.4.4, and SciPy 1.17.1. A complete
  preflight record is retained at `tuning/mimo_ofdm_chest/builtin/preflight.txt`. The complete
  `pip freeze` snapshot is retained at `tuning/mimo_ofdm_chest/builtin/pip-freeze.txt`, with
  SHA-256 `8a5e065fdd478673bbb0c3bd5ce59745a3fb3d9fa93955c733054ab1cc075ee8`.
- Hardware: NVIDIA L40S GPU, driver 580.126.09, and AMD EPYC 7R13 CPU.
- Covariance-generation runtime: 52.64 seconds wall-clock.
- Payloads: frequency covariance `[624, 624]`, OFDM-time covariance `[14, 14]`,
  and receive-space covariance `[16, 16]`, all `complex64` and 3,119,008 bytes
  in total.
- Payload checksums:
  - `freq_cov.npy`: `ba63dd008cacb9a9fd5757776a8da96030b1e6b1c48a054e3b99601481e73760`;
  - `time_cov.npy`: `777032e1c9148d555ff2fca4227346fe8d158851bd4dfbd1812be41a62291264`;
  - `space_cov.npy`: `d9a5f282d6ebb977c5835b927ae74e1788378e8986dde22773b574d5d1f20232`.
- Validation: manifest checksum/size validation, exact compatibility reload,
  finite and Hermitian array checks, and built-in unit tests passed.
- Canonical benchmark runtime: 32 minutes 56.67 seconds for LS nearest-neighbor,
  LS linear, and memory-bounded LMMSE together. LMMSE uses
  `batch_chunk_size=1` and dominates this runtime.
- Canonical benchmark command:

  ```bash
  python -m sionna.bench.mimo_ofdm_chest.papers.builtin.run_benchmark \
    --estimator ls-nn,ls-lin,lmmse \
    --asset-root "${RUN_ROOT}/assets" \
    --output-dir "${RUN_ROOT}/outputs" \
    --device cuda:0
  ```

- Canonical benchmark validation reproduced every NMSE, BER, BLER,
  and confidence-interval array bit-identically against the previous
  canonical reference generated with the calibrated TR 38.901 channel
  model.

### Edfors Delay Covariance

The selected canonical Edfors covariance was generated with:

```bash
python -m sionna.bench.mimo_ofdm_chest.papers.edfors1996.run_benchmark \
  --prepare-assets \
  --cov-num-iters 1000 \
  --cov-batch-size 64 \
  --asset-root "${RUN_ROOT}/assets"
```

The selected asset has:

- external asset path
  `<asset_root>/3gpp-umi/edfors1996/delay-covariance/it1000-bs64-seed42/`;
- manifest identity: scenario `3gpp-umi`, asset type `delay-covariance`,
  artifact `it1000-bs64-seed42`, global asset schema version 2, with manifest
  SHA-256 `88cce34a7c00fb023410ba4aea3fe2df16919a2adcb3ffbf7618dfd484c5955e`;
- recipe: 1,000 batches of 64 complete four-UE channels (64,000 system channel
  realizations), pooled over all ordered pilot groups with component seed 42;
- hardware and runtime: NVIDIA L40S, 79.34 seconds;
- payload: `time_covariances.npz`, 195,850 bytes, SHA-256
  `eea2a270eb004f475b3a6753efde8845ddee88bd3802131fbf3e52a0d4eb426f`;
- arrays: covariance `[156, 156]` `complex64` and tap powers `[156]`
  `float32`; and
- validation: exact manifest/recipe reload, finite and Hermitian covariance,
  transformed-estimator tests, and the Edfors test suite passed.

### Koller Spatial GMM

The calibrated-Sionna study generated one deterministic bank of 6,400
complete four-UE channels with seed `22022` and reused it for all eight
spatial/spectral component-count candidates. The spatial study exhaustively
evaluated K5, K10, K15, and K20 with 15 EM iterations and covariance
regularization `0.001`; weighted NMSE selected K20.

```bash
python -m sionna.bench.mimo_ofdm_chest.papers.koller2022.run_benchmark \
  --prepare-assets \
  --method spatial \
  --gmm-components 20 \
  --train-batches 100 \
  --train-batch-size 64 \
  --em-iters 15 \
  --covariance-reg 0.001 \
  --model-seed 22022 \
  --asset-root "$ASSET_ROOT"
```

The canonical artifact is
`k20-tb100-bs64-em15-reg0p001-seed22022`. Its manifest SHA-256 is
`d7be491c1caa51c302b1e68743145254e40276277c9b00a96ff6a90c2cd2f2a2`;
its float32 arrays have shapes `[20]`, `[20, 32]`, and `[20, 32, 32]`.
The 4,705-second study selected objective `0.00432931`. The canonical
benchmark took 42.50 seconds and achieved weighted NMSE `0.00423842`,
compared with `0.00429603` in the previous canonical reference.

### Koller Spectral GMM

The spectral study used the same fitting bank and exhaustive K5/K10/K15/K20
choice. Weighted NMSE selected K15.

```bash
python -m sionna.bench.mimo_ofdm_chest.papers.koller2022.run_benchmark \
  --prepare-assets \
  --method spectral \
  --gmm-components 15 \
  --train-batches 100 \
  --train-batch-size 64 \
  --em-iters 15 \
  --covariance-reg 0.001 \
  --model-seed 22022 \
  --asset-root "$ASSET_ROOT"
```

The canonical artifact is
`k15-tb100-bs64-em15-reg0p001-seed22022`. Its manifest SHA-256 is
`07cc19084a9e294e182fd0705429fcafccda7bfaeb4a27006ed0122e4fcaad92`.
It contains float32 weights `[15]`, means `[15, 1248]`, and 15 separately
stored covariance arrays `[1248, 1248]`, totaling 93,534,346 bytes including
the manifest. The 224.09-second study selected objective `0.00157878`. The
canonical benchmark took 705.39 seconds and achieved weighted NMSE
`0.00158260`, compared with `0.00174428` in the previous canonical reference.

### Melgar Model 2 DNN

The canonical Model 2 study used calibrated Sionna commit
`f16208347246d8d17ec69628dfb856b19287b80c` and one deterministic bank of
6,400 training and 256 validation channels at every canonical SNR. Each
channel contributes 256 real pilot-vector examples, giving 1,638,400 training
and 65,536 validation examples per model. The data-manifest SHA-256 is
`ea6883bee8506a7d3b0c75036547c38017fd3ffc145fc91ef71287d32f28b727`.

Ten seeded TPE trials trained five SNR-specific networks each and selected raw
learning rate `7.01799283113844e-05`, optimizer batch size 256, and weighted
NMSE `0.004693918396679923`. Canonical-sized paired validation selected the
rounded recipe `7.02e-5`/256 with objective `0.008802547101945238`, versus
`0.012489951946848734` for the prior recipe.

The complete selected bank is equivalent to running:

```bash
python -m sionna.bench.mimo_ofdm_chest.papers.melgar2022.run_benchmark \
  --prepare-assets \
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
  --training-data-manifest "$TRAINING_DATA_MANIFEST" \
  --asset-root "$FINAL_ASSET_ROOT"
```

The eleven-model bank trained in 6:12:29 and occupies 39,270,824 bytes. Its
exact asset audit has SHA-256
`fcaf264a77df167481deb0be0da792684ceb5433781b87db2612cf259c050d0e`.
All saved payloads independently reproduced their recorded best-validation
metrics. Strict loading verifies state keys, shapes, dtypes, payload checksums,
and finite tensors. Detector calibration selected `err_var_scale=0.454`. The
59.47-second canonical benchmark achieved five-SNR weighted NMSE
`0.00880966`, versus `0.01144337` in the previous canonical reference.

### Melgar Model 3 DNN

The canonical Model 3 study used calibrated Sionna commit
`f16208347246d8d17ec69628dfb856b19287b80c`, including channel-model commit
`ad1609ea1` and explicit UMi TR 38.901 version 19.2. One deterministic bank
provides 6,400 training and 256 validation channels at every canonical SNR.
Each eight-channel batch retains at most 4,096 real frequency-block examples,
giving 3,276,800 training and 131,072 validation examples per model. The data
manifest SHA-256 is
`da52a76d051517397c86730a870b2466c897daaa456453a7278af6b2cb500a31`.

Ten seeded TPE trials trained fresh models at `{-10,-5,0,5,10}` dB with at
most 200 epochs, patience 20, relative minimum improvement `1e-4`, and
best-checkpoint restoration. The search covered learning rate `[1e-5,1e-3]`
and optimizer batch size `{256,512,1024,2048}`. It completed with no failed or
pruned trials in 11:01:56. The raw winner was learning rate
`3.168916025578695e-05`, batch size 512, and weighted-NMSE objective
`0.004829016614663942`. Canonical-sized paired validation selected rounded
learning rate `3.17e-5` and batch size 512 with objective
`0.004890910391028793`, versus `0.005637904627207314` for the freshly trained
prior recipe.

The complete selected bank is equivalent to running:

```bash
python -m sionna.bench.mimo_ofdm_chest.papers.melgar2022.run_benchmark \
  --prepare-assets \
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
  --max-examples-per-batch 4096 \
  --training-data-manifest "$TRAINING_DATA_MANIFEST" \
  --asset-root "$FINAL_ASSET_ROOT"
```

The eleven-model bank trained in 3:48:10. Its exact asset audit has SHA-256
`40595b56e5c74b9ca5f8a8b151576a53d01f2558b40003e57fd1f5018d438691`.
Every manifest uses schema version 2 and records one SNR-specific artifact with
the `n78-tb800-bs8-vb32-vbs8-ep200-mb512-lr3p17e-05-...` recipe. Strict
loading verifies all state keys, shapes, dtypes, payload checksums, and finite
tensors.

Detector calibration retained `err_var_scale=0.3405776426701012`: the rounded
Optuna candidate `0.00558` scored paired BLER `0.0208333333`, while the
incumbent scored `0.0201719577`. The canonical benchmark took 44.11 seconds
and achieved weighted NMSE `0.00478873,` 13.81% below the previous canonical
reference.

### Li ReEsNet

The selected ReEsNet bank contains one checkpoint for every canonical SNR
from -10 dB through 15 dB. Each model used the same deterministic 6,400-channel
training and 256-channel validation bank, with SNR-specific noise, seed 42,
and validation seed 100045. The fixed data manifest SHA-256 is
`255dbe564889f42798c34b443d3a4f28868d06c41e7f3972091fefb320e9730f`.

The final recipe uses 200 maximum epochs, patience 20, relative minimum
improvement `1e-4`, optimizer batch size 256, learning rate `0.000186`,
best-checkpoint restoration, paper-style speed sampling, and the paper's L1
loss. The complete eleven-SNR model bank is equivalent to running:

```bash
python -m sionna.bench.mimo_ofdm_chest.papers.li2020.run_benchmark \
  --prepare-assets \
  --asset-root "$ASSET_ROOT" \
  --training-data-manifest "$TRAINING_DATA_MANIFEST" \
  --train-batches 400 \
  --train-batch-size 16 \
  --validation-batches 16 \
  --validation-batch-size 16 \
  --epochs 200 \
  --optimizer-batch-size 256 \
  --learning-rate 0.000186 \
  --patience-epochs 20 \
  --relative-min-delta 0.0001 \
  --paper-style-training
```

Best/stopped epochs were `139/159`, `106/126`, `161/181`, `126/146`,
`126/146`, and `64/84` for each remaining SNR from 2.5 dB through 15 dB.
The eleven manifests and payloads occupy 1,734,356 bytes and pass exact
compatibility, state-key, shape, dtype, checksum, and strict-load validation.

Ten full five-SNR trials plus paired validation selected the rounded learning
rate and optimizer batch size. Detector-facing calibration retained
`err_var_scale=0.23432889192166198` under the strict-better rule. The complete
accepted run took 133,828 seconds (37.17 hours) on an NVIDIA L40S. The
canonical benchmark took 42.45 seconds and achieved weighted NMSE
`0.00427391`, compared with `0.00498370` in the previous canonical reference.

### Baur VAE-Noisy

The canonical calibrated-Sionna study generated one deterministic bank of 6,400 training
and 256 validation channels at every canonical SNR using seed 42 and
validation seed 100045. The fixed-data manifest SHA-256 is
`35b7728ac44e738ba696df25fdbf5e7f8b69887934268c185c7c8caeee1f9718`.
Ten full five-SNR trials selected latent dimension 16, learning rate
`0.000230`, free-bits threshold `0.875`, and optimizer batch size 512. The
selected recipe scored paired weighted NMSE `0.0044366964`, versus
`0.0045868521` for the freshly trained prior recipe.

The complete eleven-SNR model bank is equivalent to running:

```bash
python -m sionna.bench.mimo_ofdm_chest.papers.baur2022.run_benchmark \
  --prepare-assets \
  --asset-root "$FINAL_ASSET_ROOT" \
  --training-data-manifest "$TRAINING_DATA_MANIFEST" \
  --train-batches 400 \
  --train-batch-size 16 \
  --validation-batches 16 \
  --validation-batch-size 16 \
  --epochs 200 \
  --optimizer-batch-size 512 \
  --learning-rate 0.000230 \
  --free-bits 0.875 \
  --latent-dim 16 \
  --patience-epochs 20 \
  --relative-min-delta 0.0001
```

Training the final bank took 30:53:17. Its 11 manifests and checkpoint
payloads occupy 15,780,168 bytes and have aggregate audit SHA-256
`17c4f440bee3ac63216e71f9bf96bea80e43b2085229b5359bb5845d6c0d3edf`.
Strict loading verifies all 517 state tensors, shapes, dtypes, payload
checksums, and finite values. Detector calibration retained
`err_var_scale=0.5405613145449313` after the rounded candidate `0.454` tied
the incumbent at paired BLER `0.0211640212`. The 44.66-second canonical
benchmark achieved five-SNR weighted NMSE `0.0042578149`; all pointwise NMSE
confidence intervals overlap those of the previous canonical reference.

## Canonical Asset Inventory

The canonical runners select the following assets:

| Scenario and asset type | Count | Classification |
| :- | -: | :- |
| 3GPP UMi built-in LMMSE covariance | 1 | canonical |
| 3GPP UMi Edfors delay covariance | 1 | canonical |
| 3GPP UMi Koller spectral GMM | 1 | canonical |
| 3GPP UMi Koller spatial GMM | 1 | canonical |
| 3GPP UMi Melgar Model 2 DNN | 11 | canonical per-SNR bank |
| 3GPP UMi Melgar Model 3 DNN | 11 | canonical per-SNR bank |
| 3GPP UMi Li ReEsNet | 11 | canonical per-SNR bank |
| 3GPP UMi Baur VAE-noisy | 11 | canonical per-SNR bank |

Selection is recorded through scenario-local result linkage after explicit
validation. Regeneration, trial assets, fixed training data, and rejected
experiments remain under an external asset root. An installed public package
therefore remains read-only and contains no canonical payloads.
