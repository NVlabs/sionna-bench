# Persistence Migration Todo

This checklist tracks the repository-wide migration to the persistence APIs in
`sionna_bench.mimo_ofdm_chest.persistence`, `sionna_bench.mimo_ofdm_chest.asset_store`, and
`sionna_bench.mimo_ofdm_chest.benchmarking.result_store`. It deliberately introduces no
compatibility shims for the removed APIs. Each phase should leave the migrated
scope internally consistent and covered by focused tests.

## Migration Contracts

- Canonical results live under
  `outputs/mimo_ofdm_chest/<paper_id>/<scenario_id>/`; smoke results live under
  `outputs/mimo_ofdm_chest/<paper_id>/smoke/<scenario_id>/`.
- Each result directory contains one `config.json`, one scenario-local
  `results.json`, and the result NPZ payloads listed by `results.json`.
- `record_result()` is called only after the corresponding NPZ payload exists.
- JSON-serializable estimator constructor arguments are recorded directly in
  `results.json`. Prepared tensors and model states are represented by paths to
  the exact asset manifests selected for inference.
- Bundled inference assets live below the owning paper package. Training,
  fitting, covariance estimation, tuning, and tests write only below an
  explicitly supplied external asset root.
- Asset loaders validate the common manifest envelope through
  `load_asset_manifest()` and retain responsibility for payload keys, shapes,
  dtypes, state-dict structure, and semantic compatibility.
- Asset preparation functions return, or otherwise make available to their
  runner, the path of every manifest whose payload is injected into an
  estimator.
- The retired ray-tracing channel source must not remain in source exports,
  tests, documentation, configuration examples, outputs, or asset policy.

## 0. Freeze the Shared APIs

- [x] Remove stale ray-tracing channel-source and `file_metadata` imports and
      exports from the benchmark package; verify that importing
      `sionna_bench.mimo_ofdm_chest.benchmarking` succeeds.
- [x] Remove the remaining retired channel-source references from tests, repository
      documentation, TODOs, regression references, and asset examples.
- [x] Define `asset_schema_version` as the global version of the common
      asset-manifest envelope. Version 2 identifies the new
      `asset_schema_version`/`generation_recipe`/`metadata` format.
      Builders always write version 2 and loaders reject every other version.
- [x] Use `asset_schema_version` terminology consistently in the shared asset
      API, validation messages, and common documentation. Paper-specific
      legacy arguments and tests are migrated in their later phases.
- [x] Update imports of canonical JSON and SHA-256 helpers to
      `sionna_bench.mimo_ofdm_chest.persistence`; do not re-export persistence helpers from
      `configs` merely to preserve old imports.
- [x] Finish the small shared-module cleanup: import ordering, line wrapping,
      comments, and docstrings in `persistence.py`, `asset_store.py`, and
      `result_store.py`.
- [x] Run `git diff --check`, `py_compile` for the shared modules, and a direct
      package-import check.

## 1. Shared Persistence Tests

### `persistence.py`

- [x] Create focused tests for canonical JSON ordering, finite-number
      rejection, path serialization, and stable SHA-256 fingerprints.
- [x] Test `sha256_file()` against a known byte sequence.
- [x] Move atomic JSON, NPY, and NPZ replacement tests out of the old asset
      store tests and verify that failed replacement preserves the prior file
      and removes temporary files.

### `asset_store.py`

- [x] Rewrite `test/mimo_ofdm_chest/infrastructure/test_asset_store.py` around the new public
      API: `build_asset_manifest()`, `load_asset_manifest()`,
      `paper_asset_dir()`, and `external_asset_dir()`.
- [x] Replace the legacy manifest fixture with the final
      `asset_schema_version`, `generation_recipe`, and nested `metadata`
      structure. Remove `config_fingerprint`, `training_recipe`, `provenance`,
      and legacy reference-normalization tests.
- [x] Test bundled and external path construction independently.
- [x] Test required fields, identifier validation, supported schema versions,
      exact compatibility matching, optional metadata validation, duplicate
      file paths, path containment, missing payloads, size mismatches, and
      checksum mismatches.
- [x] Test generated NPY and NPZ records for their declared formats, keys,
      shapes, and dtypes without turning private implementation details into a
      public API.

### `result_store.py` and `SystemConfig`

- [x] Rename `test/mimo_ofdm_chest/infrastructure/test_result_identity.py` to
      `test_result_store.py` and rewrite it around
      `prepare_paper_result_dir()` and `record_result()`.
- [x] Retain tests showing that runtime-only configuration changes do not
      change the fingerprint, while system and evaluation changes do.
- [x] Remove obsolete ray-tracing configuration-fingerprint tests without
      replacement.
- [x] Test canonical and smoke directory layouts, invalid `smoke` values,
      invalid paper slugs, unclaimed nonempty directories, and incompatible
      existing `config.json` files.
- [x] Test default and explicit result names, perfect-CSI entries with no
      estimator class, missing NPZ rejection, constructor-argument
      canonicalization, incremental result updates, schema rejection, and
      configuration mismatch rejection.
- [x] Test that asset-manifest paths are validated and reduced to
      `asset_type`, `artifact_id`, and the exact manifest SHA-256.
- [x] Confirm that no old paper-level `manifest.json` behavior remains in the
      result-store tests.

## 2. Leaderboard Result Discovery

- [x] Change `leaderboard/mimo_ofdm_chest/generate_data.py` to discover scenario-local
      `results.json` files instead of paper-level `manifest.json` files.
- [x] Treat `results.json` as the authoritative result inventory: reject
      listed missing NPZ files and ignore or report unlisted NPZ files according
      to one documented policy.
- [x] Validate `results_schema_version`, `scenario_id`, and agreement with the
      neighboring `config.json` fingerprint before loading metric arrays.
- [x] Update `test/mimo_ofdm_chest/test_leaderboard_data.py` fixtures and
      rejection tests to the new structure.
- [x] Regenerate leaderboard data from a temporary mixed canonical/smoke tree
      and verify that smoke outputs remain excluded from canonical scenarios.

## 3. Assetless Benchmark Runners

For every runner in this phase:

- replace `prepare_paper_dir()` with `prepare_paper_result_dir()`;
- remove `update_paper_manifest()`;
- retain one explicit dictionary of the persistable constructor arguments
  passed to the estimator;
- call `record_result()` immediately after each successful evaluation; and
- add a focused smoke test that inspects the resulting `config.json`,
  `results.json`, and NPZ path.

- [x] Migrate Coleri 2002, recording one entry for each interpolation method
      and its method-specific low-pass arguments.
- [x] Migrate Delfeld 2025, recording all canonical RKHS ranks, iteration
      counts, regularization values, and variance scale.
- [x] Migrate Wei 2017, including interpolation, tap count, iteration count,
      initialization, and posterior-variance scale.
- [x] Migrate Balevi 2020, including the complete online-fitting recipe; do
      not represent its observation-specific online fit as a persistent asset.
- [x] Migrate Lyu 2026, including all ADMM, sparse, diffuse, interpolation, and
      variance arguments used by the canonical estimator.
- [x] Run all five runner smoke checks against temporary output directories.

## 4. Builtin and Edfors Covariance Assets

### Builtin

- [x] Move covariance batch count, covariance batch size, and LMMSE order out
      of `SystemConfig` into explicit runner defaults and CLI overrides.
- [x] Remove the covariance loader's legacy `supported_schema_versions`
      argument and rely on the global asset-manifest version 2 enforced by
      `load_asset_manifest()`. Ensure newly written covariance manifests use
      the shared builder without a paper-specific schema-version argument.
- [x] Change bundled and external covariance path selection to the separate
      `paper_asset_dir()` and `external_asset_dir()` contracts.
- [x] Replace `load_manifest()` plus `validate_manifest_envelope()` with one
      `load_asset_manifest()` call and migrate manifest construction to
      `generation_recipe` and nested `metadata`.
- [x] Remove `_covariance_asset_reference()`; return or expose the selected
      `manifest.json` path instead.
- [x] Record each builtin result separately. LMMSE references its covariance
      manifest; LS and perfect CSI have no asset manifest.
- [x] Update builtin covariance tests for the final schema, loading failures,
      external generation, and exact result linkage.

### Edfors 1996

- [x] Move covariance-generation defaults out of `SystemConfig` and preserve
      CLI overrides as explicit asset-generation arguments.
- [x] Remove the covariance loader's legacy `supported_schema_versions`
      argument and rely on the global asset-manifest version 2. Ensure newly
      written covariance manifests do not select a paper-specific schema
      version.
- [x] Migrate bundled/external paths, manifest construction, and manifest
      loading as for builtin covariance assets.
- [x] Remove `_covariance_asset_reference()` and pass the selected manifest
      path to `record_result()` for all three methods.
- [x] Record the common covariance generation recipe as asset metadata and the
      method, tap count, interpolation choice, and confidence scale as result
      constructor arguments.
- [x] Update Edfors asset tests and run all three smoke methods from one
      temporary compatible covariance asset.

## 5. Koller GMM Assets

- [x] Remove `_SCHEMA_VERSIONS` and all paper-specific `schema_version` and
      `supported_schema_versions` arguments. Both spatial and spectral GMM
      manifests use the global asset-manifest version 2; their different
      payload contracts remain identified by `asset_type` and validated by
      the Koller loader.
- [x] Migrate spatial and spectral GMM manifests to the global
      `asset_schema_version` 2 envelope and `generation_recipe` field.
- [x] Replace old manifest loading/validation and split bundled versus external
      asset path selection.
- [x] Remove `_model_asset_reference()` and make `load_model()` and
      `fit_model()` provide the selected manifest path alongside the inference
      tensors.
- [x] Keep tensor-valued weights, means, and covariances out of
      `constructor_arguments`; record method-specific scalar arguments and the
      selected manifest path.
- [x] Rewrite Koller asset tests for both methods, temporary fitting, malformed
      manifests and payloads, and result linkage.
- [x] Run spatial and spectral smoke benchmarks from temporary fitted assets.

## 6. Neural Model Assets

Apply the following rules to Melgar, Li, and Baur:

- use separate bundled and external asset path functions;
- remove all legacy `schema_version` and `supported_schema_versions`
  arguments and rely on the global asset-manifest version 2;
- replace `training_recipe` with `generation_recipe` and `extra_fields` with
  nested `metadata`;
- replace separate load/validate calls with `load_asset_manifest()`;
- retain paper-specific state-key, shape, dtype, and architecture validation;
- return an ordered list of selected manifest paths, including one path per
  SNR-specific model; and
- pass those paths to `record_result()` while keeping model states and tensors
  out of `constructor_arguments`.

### Melgar 2022

- [x] Remove Model 2 and Model 3 loader assumptions about legacy schema
      version 1; both model asset types use the global version 2 envelope.
- [x] Migrate both Model 2 and Model 3 asset formats and loader tests.
- [x] Preserve per-SNR manifest ordering exactly as model-state and noise-grid
      ordering.
- [x] Record architecture, interpolation/block settings, weight mode, and
      detector-facing scale for each result.
- [x] Run one tiny temporary training/load round trip and one smoke benchmark
      for each method.

### Li 2020

- [x] Remove the ReEsNet loader's legacy `supported_schema_versions={1}`
      argument and use the global version 2 envelope.
- [x] Migrate the ReEsNet manifest, metadata, path selection, and loader tests.
- [x] Preserve per-SNR manifest ordering and record architecture, interpolation,
      and error-variance settings in the result entry.
- [x] Run a tiny temporary training/load round trip and smoke benchmark.

### Baur 2022

- [x] Remove both VAE loaders' legacy `supported_schema_versions={1}`
      arguments and use the global version 2 envelope.
- [x] Migrate both VAE training variants at the asset layer even though only
      the selected canonical method is placed on the leaderboard.
- [x] Preserve per-SNR manifest ordering and record covariance mode,
      architecture, optimization-independent estimator arguments, and
      detector-facing scale.
- [x] Update malformed-checkpoint and compatibility tests to the final schema.
- [x] Run tiny temporary training/load checks for both asset variants and a
      smoke benchmark for the selected canonical estimator.

## 7. Remove the Legacy API Completely

- [x] Verify with `rg` that source and tests contain no uses of
      `prepare_paper_dir`, `update_paper_manifest`, `load_manifest`,
      `validate_manifest_envelope`, `asset_manifest_reference`,
      `asset_manifest_references`, legacy `asset_dir`, `training_recipe=`, or
      `extra_fields=`.
- [x] Verify that atomic writers and hash helpers are imported from
      `sionna_bench.mimo_ofdm_chest.persistence`, not indirectly through `asset_store`.
- [x] Verify that no paper runner reconstructs a manifest path or compact asset
      reference independently of its asset loader and `record_result()`.
- [x] Run package API tests and import every paper runner and asset module.

## 8. Documentation and Existing Artifacts

- [x] Rewrite `docs/asset_manifests.md` to the final asset schema, terminology,
      bundled/external layouts, and scenario-local result linkage.
- [x] Remove descriptions of provenance, `config_fingerprint` in asset
      manifests, `training_recipe`, paper-level result manifests, and the old
      compact-reference helpers.
- [x] Update `docs/paper_architecture.md`,
      `docs/mimo_ofdm_chest/inventories/asset_inventory.md`, README, and relevant TODO/report
      commands to the new APIs and remove the retired channel source.
- [x] Update paper READMEs where their asset-generation commands or manifest
      descriptions changed.
- [ ] Decide whether existing canonical outputs and package-local assets are migrated
      mechanically or regenerated. Document the decision per artifact and do
      not silently relabel an old manifest as the new schema.
- [ ] If outputs are migrated mechanically, create scenario-local
      `results.json` files from the previous authoritative inventories and
      verify every NPZ checksum/name before deleting old paper-level manifests.
- [ ] If assets are migrated mechanically, preserve payload bytes, recompute
      manifest checksums, and run each paper-specific loader before accepting
      the migrated asset.

## 9. Final Verification

- [ ] Run all shared infrastructure, builtin, and paper asset tests.
- [ ] Run the complete unit-test suite in the supported Python environment.
- [ ] Run representative assetless, covariance-backed, GMM-backed, and
      neural-model smoke benchmarks with temporary outputs.
- [ ] Regenerate leaderboard data and inspect all scenario, paper, method,
      constructor-argument, and asset-reference entries.
- [ ] Run `git diff --check`, compile all package modules, build the wheel, and
      test imports plus one bundled-asset load from the installed wheel.
- [ ] Review the final diff for accidental asset payload changes, output churn,
      stale compatibility code, and undocumented persistence behavior.


## At some point: clean up and standardize tuning
## At some point: clean up all prompts and docs
