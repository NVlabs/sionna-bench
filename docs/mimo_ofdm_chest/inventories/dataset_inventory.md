# Dataset And Statistical-Sample Inventory

This document records the amount of generated channel data used by methods
that require learned weights or fitted channel statistics. It distinguishes
complete independently sampled system-channel realizations from the derived
vectors or grids presented to an optimizer. Derived examples from the same
system realization share propagation conditions and are therefore not counted
as independent channels.

Each SNR-specific neural checkpoint is trained independently, and its effective
dataset size is therefore the per-SNR count reported below. The current asset
generators reset the same training and validation seeds for every SNR-specific
checkpoint. Consequently, the clean channel realizations are reused across
SNRs while the applied noise level changes. Bank totals count channel/SNR
exposures; they are neither additional examples available to one model nor
additional unique propagation-channel realizations.

The inventory reflects the final calibrated-Sionna regeneration completed on
2026-08-24 using pre-release source commit
`f16208347246d8d17ec69628dfb856b19287b80c`. That checkout reported version
2.0.1 and contains the channel-model changes subsequently released in Sionna
2.1. Manifest `generation_recipe` and `metadata.history` fields remain
authoritative for an individual asset.

## Asset Dataset Inventory

| Method | Asset status | Independent channel data | Derived fitting or training samples |
| :- | :- | :- | :- |
| Built-in LMMSE | Canonical covariance asset. | 1,000 batches x 64 = **64,000** channel realizations. | Frequency, OFDM-time, and receive-space covariance sums are accumulated over the corresponding channel axes; no finite example tensor is retained. |
| `TruncatedChannelEstimator` (LS, diagonal LMMSE, and LMMSE) | Final calibrated-Sionna delay-covariance asset shared by all three methods. | 1,000 batches x 64 = **64,000** complete four-UE system-channel realizations. | Delay-domain covariance statistics are pooled over all eight ordered pilot groups; no finite example tensor is retained. |
| `SpatialGMMChannelEstimator` | Canonical K20 asset. | 100 batches x 64 = **6,400** complete four-UE system-channel realizations. | Each channel contributes 1,248 receive-array vectors, giving **7,987,200** complex 16-element vectors for the shared GMM fit. |
| `SpectralGMMChannelEstimator` | Canonical K15 asset. | The same **6,400** complete system-channel realizations used by the spatial fit. | Each channel contributes eight pilot-symbol/UE groups for each of 16 receive antennas, giving **819,200** complex 624-subcarrier vectors for the shared GMM fit. |
| `DNNChannelEstimator` (Melgar Model 2) | Canonical per-SNR bank. | Each independently trained SNR model uses **6,400** training and **256** validation channels. The same clean-channel draws are reused at all SNRs. The eleven-model bank records **70,400** training and **2,816** validation channel/SNR exposures, but still only 6,400 and 256 distinct clean draws. | Each channel contributes 256 real pilot-vector examples. Per model: **1,638,400** training and **65,536** validation examples; eleven-model bank: **18,022,400** and **720,896** example/SNR exposures. |
| `InterpolatedDNNChannelEstimator` (Melgar Model 3) | Canonical per-SNR bank. | Each independently trained SNR model uses **6,400** training and **256** validation channels. The same clean-channel draws are reused at all SNRs. The eleven-model bank records **70,400** training and **2,816** validation channel/SNR exposures, but still only 6,400 and 256 distinct clean draws. | The generator caps every eight-channel batch at 4,096 real frequency-block examples. Per model: **3,276,800** training and **131,072** validation examples; eleven-model bank: **36,044,800** and **1,441,792** example/SNR exposures. |
| `ReEsNetChannelEstimator` | Canonical per-SNR bank. | Each independently trained SNR model uses **6,400** training and **256** validation channels. The eleven-model bank records **70,400** training and **2,816** validation channel/SNR exposures from the same clean draws. | Each channel contributes 64 independently processed UE/receive-antenna grids. Per model: **409,600** training and **16,384** validation grids; eleven-model bank: **4,505,600** and **180,224** grid/SNR exposures. |
| `VAEChannelEstimator` (`VAE-noisy`) | Canonical per-SNR bank. | Each independently trained SNR model uses **6,400** training and **256** validation channels. The eleven-model bank records **70,400** training and **2,816** validation channel/SNR exposures from the same clean draws. | Each channel contributes 1,248 receive-array pilot observations. Per model: **7,987,200** training and **319,488** validation examples; eleven-model bank: **87,859,200** and **3,514,368** example/SNR exposures. |

## Evidence About Dataset Adequacy

The asset-selection protocol uses disjoint training and
validation splits, and checks selected hyperparameters on deterministic
channel realizations that are independent of both splits. The completed
replacement assets therefore provide evidence against gross overfitting to
the generated training channels. The protocol does **not** establish that the
chosen dataset sizes are sufficient for performance saturation.

In particular, the large derived-example counts above must not be treated as
independent-channel counts. Examples pooled from different receive antennas,
pilot groups, or frequency blocks of one system realization share propagation
conditions. No systematic ablation study on the size of channel datasets has been
performed. There is therefore no claim that the chosen dataset sizes are optimal.

## Data Used Inside Optuna Trials

For data-driven Optuna studies, every candidate trains or fits its own asset
before it is evaluated on the independent probe described in
`docs/mimo_ofdm_chest/contracts/tuning_protocol.md`. The training stream uses a fixed
asset seed and the probe uses a distinct fixed seed, so candidates see common
training and evaluation channels.

| Study | Training and validation data generated per trial |
| :- | :- |
| Koller spatial or spectral component count | One GMM fit from the same **6,400** channel realizations for each of K5, K10, K15, and K20. The raw channel bank is generated once and reused by both methods. |
| Melgar Model 3 training | Five independent tuning-SNR models, each using the same 6,400 clean training and 256 clean validation draws with SNR-specific noise: **32,000** training and **1,280** validation channel/SNR exposures per trial. |
| Melgar Model 2 training | Five independent tuning-SNR models, each using the same 6,400 clean training and 256 clean validation draws with SNR-specific noise: **32,000** training and **1,280** validation channel/SNR exposures per trial. |
| Li ReEsNet training | Five independent tuning-SNR models, each using the same 6,400 clean training and 256 clean validation draws: **32,000** training and **1,280** validation channel/SNR exposures per trial. |
| Baur VAE training | Five independent tuning-SNR models, each using the same 6,400 clean training and 256 clean validation draws: **32,000** training and **1,280** validation channel/SNR exposures per trial. |

Final canonical neural-model banks contain 11 SNR-specific checkpoints from
-10 dB through 15 dB. Training Optuna objectives use the five points
`{-10,-5,0,5,10}` dB; after selecting and independently validating a recipe,
all eleven canonical checkpoints are generated with that same recipe.

## Methods Without A Persistent Dataset

Built-in LS and perfect CSI, all `LSCombEstimator` methods,
`RKHSChannelEstimator`, `SBLChannelEstimator`, and
`SparseDiffuseChannelEstimator` have no fitted statistical or learned
inference asset. `DeepPriorChannelEstimator` also has no persistent training
dataset: it fits a separate untrained network directly to each received pilot
grid during inference.

## Reproducibility Sources

- `docs/mimo_ofdm_chest/contracts/asset_manifests.md` defines the meaning of
  `generation_recipe`, compatibility metadata, payload records, and training
  history.
- `asset_inventory.md` records
  generation commands, hardware, runtimes, payload sizes, and validation for
  regenerated assets.
- Paper-local READMEs document method-specific example construction and any
  deviation from the source paper's dataset.
- Each selected asset's `manifest.json` records its exact channel counts,
  seeds, SNR labels, training controls, and, for neural assets, observed
  training history.
