# Asset Manifests and Result Linkage

This document defines the contract for persistent assets and explains how benchmark
results identify the assets used for inference. Asset storage and result
storage are separate concerns:

- `sionna.bench.mimo_ofdm_chest.asset_store` selects asset directories and validates
  `manifest.json` files.
- `sionna.bench.mimo_ofdm_chest.benchmarking.result_store` owns scenario-local `config.json`,
  `results.json`, and result NPZ files.

## Identifiers

Three human-readable identifiers determine asset paths:

- `scenario_id` identifies the channel and system scenario, for example
  `3gpp-umi`.
- `asset_type` selects a paper-specific payload contract, for example
  `spectral-gmm` or `reesnet`.
- `artifact_id` identifies one concrete generated asset within that type, for
  example `k17-tb32-bs64-em15-reg0p001-seed42`.

Identifiers use lowercase ASCII letters, digits, and hyphens. A path that is
already occupied by a different identity or generation recipe is not silently
reused. Generation functions refuse an occupied destination unless replacement
is explicitly requested.

## Asset Locations

### Optional Package-Local Assets

An internal or private distribution may include complete, read-only inference
assets below the paper that owns them:

```text
src/sionna/bench/mimo_ofdm_chest/papers/<paper_id>/assets/
  <scenario_id>/
    <asset_type>/
      <artifact_id>/
        manifest.json
        <payload files>
```

The built-in baselines use the same layout under
`src/sionna/bench/mimo_ofdm_chest/papers/builtin/assets/`. Paper loaders select this path with
`paper_asset_dir()` when no external root is supplied.

Public source and wheel distributions may omit all payloads from these
directories. Importing `sionna.bench` and constructing estimators from prepared
tensors remain valid without them. A loader that cannot find its selected
asset raises `FileNotFoundError` with the commands required to generate the
asset externally and evaluate it from that root. It never downloads, trains,
fits, or writes implicitly.

An asset directory is usable only when both its manifest and every payload
listed by that manifest are present. Orphaned manifests are not retained in a
runtime `assets/` tree merely as documentation.

### External Assets

Training, fitting, covariance estimation, tuning, smoke runs, and tests write
only beneath an explicit external root:

```text
<asset_root>/
  <scenario_id>/
    <paper_id>/
      <asset_type>/
        <artifact_id>/
          manifest.json
          <payload files>
```

Paper loaders and generators select this path with `external_asset_dir()`.
Tests use temporary roots; they never write into package-local `assets/`.

## Asset Schema Version 2

Every usable asset directory contains exactly one `manifest.json`. The common
envelope is:

```json
{
  "asset_schema_version": 2,
  "asset_type": "spectral-gmm",
  "scenario_id": "3gpp-umi",
  "artifact_id": "k15-tb100-bs64-em15-reg0p001-seed22022",
  "compatibility": {
    "estimator_mode": "spectral",
    "real_vector_convention": "real-parts-then-imaginary-parts",
    "vector_axis": "effective-subcarrier",
    "vector_dimensions": {
      "complex": 624,
      "real_stacked": 1248
    },
    "pilot_groups": {
      "coordinate_order": [
        "tx",
        "stream",
        "symbol",
        "subcarrier"
      ],
      "coordinates": {
        "shape": [
          8,
          156,
          4
        ],
        "dtype": "int64",
        "canonical_sha256": "a00fbc7de9fe58bd9bdb5f0b0177ec3782ed1e970d16862cd0795beadd915006"
      },
      "values": {
        "shape": [
          8,
          156
        ],
        "dtype": "complex64",
        "encoding": "[real, imaginary]",
        "canonical_sha256": "e3c62bbeda337671421651e1904dead920ef241a8d9c246fb2d46703880a09bb"
      },
      "powers": {
        "shape": [
          8,
          156
        ],
        "dtype": "float32",
        "canonical_sha256": "8afb79a7c2f685201d1204b8f1e98aa90a551bd9e3ec411b740d7137884b6505"
      }
    }
  },
  "generation_recipe": {
    "num_components": 15,
    "train_batches": 100,
    "train_batch_size": 64,
    "em_iters": 15,
    "covariance_reg": 0.001,
    "seed": 22022
  },
  "files": [
    {
      "path": "gmm_params.npz",
      "size_bytes": 75450,
      "sha256": "c5fd94e72ac6a81e8f9c8073197266ba9b04f22355672e90f804353d89bfb637",
      "format": "npz",
      "arrays": {
        "weights": {
          "shape": [
            15
          ],
          "dtype": "float32"
        },
        "means": {
          "shape": [
            15,
            1248
          ],
          "dtype": "float32"
        }
      }
    },
    <other files>
  ],
  "metadata": {
    "payload_layout": "split-covariances",
    "parameter_arrays": [
      "weights",
      "means",
      "covariances"
    ],
    "fitting_data": {
      "asset_type": "gmm-fitting-data",
      "artifact_id": "tb100-bs64-seed22022",
      "manifest_sha256": "02d6593b6f0aa736ee28b3edea2e44e28699173e4140565e76f578ea372170d1"
    }
  }
}
```

The common fields mean:

- `asset_schema_version` versions this shared envelope. All asset types use
  version 2, and `load_asset_manifest()` rejects every other version.
- `asset_type`, `scenario_id`, and `artifact_id` are the validated identity.
  The artifact identifier equals the containing directory name.
- `compatibility` contains facts that must match before the payload can be
  used for inference: architecture, dimensions, tensor conventions, channel
  assumptions, and exact pilot layout where applicable.
- `generation_recipe` records how the payload was produced: sample counts,
  training or fitting settings, SNRs, and random seeds. It is generation
  evidence and is not a substitute for inference compatibility.
- `files` lists every required payload by relative path, format, byte size,
  and SHA-256 checksum. NPY records contain one `array` description; NPZ
  records contain an `arrays` mapping. Each array description records its
  shape and NumPy dtype.
- `metadata` is optional asset-specific information that is neither identity
  nor compatibility. Neural checkpoints use it for ordered state keys and
  training history.

Large pilot-coordinate or value tensors are represented in `compatibility` by
shape, dtype, encoding, and a canonical SHA-256 rather than expanded JSON.
Paper loaders recompute these identities from the requested resource grid.

`build_asset_manifest()` constructs this envelope and records NPY/NPZ
structure. `load_asset_manifest()` validates the common fields, exact expected
compatibility, path containment, payload existence, file sizes, and checksums.
Paper-specific loaders additionally validate array names, shapes, dtypes,
state-dict structure, and semantic assumptions before returning tensors.

## Active Payload Contracts

| Asset type | Payload contract |
| :- | :- |
| `lmmse-covariance` | `freq_cov.npy`, `time_cov.npy`, and `space_cov.npy`; complex square covariance matrices matching the configured frequency, OFDM-time, and receive-array dimensions. |
| `delay-covariance` | `time_covariances.npz` containing `time_covariances` and `tap_powers`; compatibility includes the ordered pilot-subcarrier geometry. |
| `spatial-gmm` | `gmm_params.npz` containing float32 `weights`, `means`, and `covs`; compatibility identifies spatial processing, the real/imaginary stacking convention, receive-vector dimensions, and exact pilot groups. |
| `spectral-gmm` | `gmm_params.npz` containing float32 `weights` and `means`, plus one `gmm_covariance_<index>.npy` per component; compatibility identifies spectral processing, full-subcarrier dimensions, and exact pilot groups. |
| `melgar-model2` | `dnn_params.npz`; `metadata` contains ordered state keys and history. Compatibility records Model 2 architecture and exact active pilot groups. |
| `melgar-model3` | `dnn_params.npz`; `metadata` contains ordered state keys and history. Compatibility records Model 3 architecture, block dimensions, and the complete LS-interpolation input layout. |
| `reesnet` | `reesnet_params.npz`; `metadata` contains ordered state keys and history. Compatibility records pilot-grid, full-grid, and architecture dimensions. |
| `vae-noisy` | `vae_params.npz`; `metadata` contains ordered state keys and history. Compatibility records the clean-label VAE method, receive-vector geometry, architecture, covariance parameterization, and exact pilot groups. |
| `vae-real` | Same payload structure as `vae-noisy`, with compatibility identifying the noisy-label method and likelihood convention. |

Payload layouts not accepted by the current paper-specific loaders are not
part of this contract.

## Asset Lifecycle

An asset-backed paper exposes separate strict loading and generation handles:

```python
load_model(...)
train_model(..., asset_root=...)

load_covariance(...)
estimate_covariance(..., asset_root=...)

load_model(...)
fit_model(..., asset_root=...)
```

Loading never samples channels, trains, fits, writes, or modifies an asset.
Generation requires an external root, writes payloads and `manifest.json`
atomically, and then invokes the loader before returning. This separation is a
safety measure: newly generated assets remain external, cannot overwrite
package files, and are validated through the same loader used for inference.
A reviewed asset may additionally be copied into a private package as a
complete package-local asset, but public distributions do not require or imply
such a copy. Both paths return inference-ready tensors and the selected
manifest path, or an ordered list of paths for a per-SNR model bank.

Estimator constructors receive only inference-ready values. They do not know
about repository paths or manifests.

## Scenario-Local Results

Canonical results use:

```text
outputs/mimo_ofdm_chest/<paper_id>/<scenario_id>/
  config.json
  results.json
  <result_name>.npz
```

Smoke results use:

```text
outputs/mimo_ofdm_chest/<paper_id>/smoke/<scenario_id>/
  config.json
  results.json
  <result_name>.npz
```

`config.json` is the estimator-independent system and evaluation snapshot. Its
current `config_schema_version` is 4. It records the OFDM system, pilot
pattern, users and arrays, MCS, channel source, SNR range, Monte Carlo budget,
setup and evaluation seeds, precision, and runtime placement. Its
`config_fingerprint` excludes the `runtime` section and prevents incompatible
runs from sharing a scenario directory.

A canonical 3GPP UMi snapshot has the following structure:

```json
{
  "config_schema_version": 4,
  "scenario_id": "3gpp-umi",
  "system": {
    "ofdm": {
      "subcarrier_spacing": 15000.0,
      "num_prb": 52,
      "num_ofdm_symbols": 14,
      "cyclic_prefix_length": 44
    },
    "pilot_pattern": {
      "type": "kronecker",
      "ofdm_symbol_indices": [2, 11]
    },
    "multi_user": {
      "num_ue": 4,
      "num_streams_per_tx": 1,
      "direction": "uplink"
    },
    "arrays": {
      "ue": {
        "polarization": "single",
        "polarization_type": "V",
        "antenna_pattern": "omni"
      },
      "bs": {
        "num_rows": 4,
        "num_cols": 4,
        "polarization": "single",
        "polarization_type": "V",
        "antenna_pattern": "38.901"
      }
    },
    "mcs": {
      "num_bits_per_symbol": 4,
      "coderate": 0.6,
      "max_k_per_cb": 8448
    }
  },
  "channel_source": {
    "type": "umi",
    "parameters": {
      "carrier_frequency": 2600000000.0,
      "o2i_model": "low",
      "enable_pathloss": false,
      "enable_shadow_fading": false,
      "ue_speed": 0.8333333333333333
    }
  },
  "evaluation": {
    "snr_db_start": -10.0,
    "snr_db_stop": 15.0,
    "snr_db_step": 2.5,
    "batch_size": 4,
    "num_mc_batches": 63,
    "setup_seed": 42,
    "evaluation_seed": 2026
  },
  "numeric": {
    "precision": "single"
  },
  "runtime": {
    "device": "cuda:0",
    "output_dir": "outputs/mimo_ofdm_chest"
  },
  "config_fingerprint": "<lowercase-sha256>"
}
```

`results.json` is the authoritative inventory of completed result payloads:

```json
{
  "results_schema_version": 1,
  "scenario_id": "3gpp-umi",
  "config_fingerprint": "<lowercase-sha256>",
  "results": {
    "SpectralGMMChannelEstimator": {
      "file": "SpectralGMMChannelEstimator.npz",
      "estimator_class": "sionna.bench.mimo_ofdm_chest.papers.koller2022.estimator.SpectralGMMChannelEstimator",
      "constructor_arguments": {
        "method": "spectral"
      },
      "assets": [
        {
          "asset_type": "spectral-gmm",
          "artifact_id": "k17-tb32-bs64-em15-reg0p001-seed42",
          "manifest_sha256": "<lowercase-sha256>"
        }
      ],
      "updated_at": "YYYY-MM-DD HH:MM:SS"
    }
  }
}
```

After the evaluation pipeline has written the NPZ payload, the paper runner in
`run_benchmark.py` calls `record_result()` with the estimator class,
JSON-serializable constructor arguments, and the manifest paths returned by its
asset loader. The result store validates each manifest and stores only its
asset type, artifact ID, and exact manifest checksum. This record identifies
the asset used for the completed benchmark; it is not a filesystem link and
does not imply that the manifest or payload is distributed with the result.
Full compatibility, generation, payload, and training details are recorded in
the canonical asset evidence and in the manifest when it is available.

Leaderboard generation reads `results.json`, rejects listed result files that
are missing or have an incompatible corresponding `config.json`, and ignores
unlisted result NPZ files. It does not require the original asset manifest or
payload. There is no second result inventory outside the scenario directory.

## Safe Writes

Asset manifests and their NPY/NPZ payloads are written through temporary
siblings and atomically replace their targets. Asset generators build complete
temporary artifact directories and move them into place only after payload and
manifest creation succeeds. A directory without a valid schema manifest (with version 2)
is not a usable asset.

Existing artifacts are not made compatible by renaming a field or changing a
version number. Any later migration or regeneration decision must preserve and
verify payload bytes explicitly or regenerate the payload through its recorded
command and validation procedure.
