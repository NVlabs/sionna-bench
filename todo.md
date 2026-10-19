# Channel Estimator Cleanup and Validation TODO

This checklist orders the remaining work so that behavioral coverage exists
before readability or compilation refactors are attempted. Work one paper at
a time after the shared test infrastructure is in place. Keep canonical
benchmark outputs unchanged unless a correctness fix or explicitly approved
retuning requires a rerun.

Completed phases below retain historical paths and version references. Current
contracts and inventories are indexed in `docs/README.md`; revalidate unchecked
items against them before treating an older instruction as current.

## Definition of Done

For each estimator or configured variant:

- [ ] It passes the canonical smoke test.
- [ ] It passes supported-system-variant tests covering more than one resource
      grid size, pilot layout, UE count, and BS antenna count.
- [ ] Unsupported geometries fail early with a specific, accurate message.
- [ ] Shapes, dtypes, devices, finite outputs, and non-negative error variances
      are tested.
- [ ] At least one numerical or analytical check provides evidence beyond
      shape correctness.
- [ ] Data-driven estimators can train a tiny temporary model for at least one
      SNR and then run inference on it.
- [ ] `torch.compile(..., fullgraph=False)` can execute the inference path, or
      the smallest unavoidable region is explicitly disabled and documented.
- [ ] Compiled and eager outputs agree within precision-appropriate tolerances.
- [ ] The implementation has been reread against the paper and existing
      reports; equations, terminology, adaptations, and omitted methods remain
      accurate.
- [ ] Constructor arguments, buffers, comments, and Sionna-style docstrings
      remain clear after any refactor.
- [ ] The paper README and implementation report describe supported geometry,
      compilation limitations, and material deviations.
- [ ] `docs/mimo_ofdm_chest/inventories/estimator_inventory.md` reflects the final preparation/inference
      split, compile tier, and graph-break causes after the paper is cleaned up.
- [ ] Any estimator that obtains detector-facing error variances only by
      multiplying the incoming noise or LS error variance by a scalar has that
      scale calibrated with Optuna after all other implementation, tuning, and
      training work for the paper is final.

## Phase 0: Freeze and Inventory Current Behavior

- [x] Record the starting commit and worktree state before cleanup.
  - Starting commit: `8777b1c631e1f84ccae3139bda1998ca4000ffba`
    (`Refine leaderboard selection and layout`).
  - The worktree had no tracked modifications. Untracked entries were
    `.vscode/`, `package-lock.json`, `package.json`,
    `src/sionna_bench/mimo_ofdm_chest/papers/lyu2026/tuning.py`, `test_int.ipynb`, and this
    `todo.md`.
- [x] Add an inventory to the main README covering every estimator variant,
      constructor controls, required assets, training entry points, and
      supported pilot geometry.
- [x] Ensure asset loading and training happen outside the compiled inference
      path, and identify any estimator that performs either operation during
      `call()`.
  - The audit is recorded in `docs/estimator_inventory.md`. Persistent asset
    I/O and offline training are outside every estimator's inference method.
    Balevi intentionally fits an untrained network during inference because
    that optimization is the estimator; its outer fitting path is a documented
    compilation exception.
- [x] Use the tracked canonical benchmark outputs as numerical regression
      references. Treat smoke runs as transient integration checks rather than
      retained artifacts.
  - The reference set and comparison policy are recorded in
    `docs/regression_references.md`. The original 18 tracked 3GPP UMi result
    files and three migrated built-in results have
    the expected metric keys, common 11-point SNR grid, shapes, and finite
    values.
- [x] Remove or explicitly retain incomplete experimental files only after
      identifying whether they contain unique work.
  - Removed all current paper-local smoke/probe outputs, top-level built-in
    smoke outputs, smoke-only assets, the superseded time-interpolator scratch
    notebook, pytest caches, and the redundant npm installation. The dashboard
    uses its tracked vendored Chart.js bundle and does not use `node_modules`.
  - Retained `src/sionna_bench/mimo_ofdm_chest/papers/lyu2026/tuning.py` and the ignored Lyu Optuna databases
    because they contain unfinished, unique tuning work. Resolve them in the
    Lyu-specific phase as already listed below.
  - The top-level canonical built-in outputs were subsequently validated and
    moved without rerunning. Retained all non-smoke assets until their
    paper-specific manifest migration determines the canonical payloads.
  - Retained ignored training/tuning logs and study databases until each
    paper's report and selected asset evidence have been audited. Remove
    redundant logs, stale PID files, and superseded studies in the respective
    paper phase.
  - Left `.vscode/` untouched as local editor configuration rather than a
    repository experiment.

## Phase 1: Shared Asset and System-Variant Test Infrastructure

Current transition state:

- Result identity and storage have been migrated globally to explicit scenario
  IDs and configuration fingerprints.
- The common asset path and manifest primitives are implemented and tested.
- Persistent covariance, GMM, and neural-network payloads now use versioned
  manifests under `assets/`. Paper-specific geometry and malformed-
  payload tests remain part of each paper phase.

### Completed work

#### Asset manifest foundation

- [x] Add `docs/asset_manifests.md` and link it from the main README.
- [x] Define the common `manifest.json` envelope:
      `asset_schema_version`, `asset_type`, `scenario_id`, `artifact_id`,
      `compatibility`, `generation_recipe`, `files`, and optional nested
      `metadata`.
- [x] Define one global asset schema version for the shared envelope. Keep
      asset-specific payload contracts identified by `asset_type` and validated
      by paper-specific loaders.
- [x] Define the legacy policy for unversioned assets: validate and migrate
      known canonical assets explicitly, otherwise reject them with a clear
      message. Do not silently treat missing versions as current.
- [x] Define explicit human-readable `scenario_id` and `artifact_id` values.
      Configuration fingerprints belong to scenario-local result metadata,
      not asset manifests or directory names.
- [x] Define the normalized target structure and fingerprint inputs for
      canonical output `config.json` snapshots.

##### Global configuration and result identity

- [x] Add explicit `scenario_id` and channel-source identity to every scenario
      configuration.
- [x] Add canonical JSON serialization and full SHA-256 configuration
      fingerprints. Include system, channel-source, evaluation, and numeric
      settings while excluding runtime placement and display-only names.
- [x] Store canonical results under `outputs/<paper_id>/<scenario_id>/`, write normalized
      schema-versioned `config.json` snapshots, and reject an occupied scenario
      path or manifest entry with a different fingerprint before evaluation.
- [x] Remove `SystemConfig.config_slug()` and replace its active result and
      tuning uses with scenario IDs and configuration fingerprints. Keep the
      exactly compatible `legacy_asset_slug()` only for assets awaiting
      per-paper migration.
- [x] Add separate `paper_asset_dir()` and `external_asset_dir()` constructors
      for optional package-local read-only and external writable asset paths.
- [x] Migrate canonical 3GPP UMi result payloads without rerunning. Verify
      identical SHA-256 digests before and after the move and update leaderboard
      discovery for explicit scenario IDs.
- [x] Add focused tests for fingerprint inputs, legacy asset compatibility,
      human-readable asset paths, normalized snapshots, and result/manifest
      collision rejection.

##### Shared asset-manifest infrastructure

- [x] Add small shared helpers for SHA-256 and file-size recording, common
      asset-manifest validation, and atomic JSON writes. Keep paper-specific
      tensor and compatibility semantics out of the common helper.
- [x] Define artifact selection identity separately from compatibility in the
      common manifest contract:
  - compatibility covers architecture, tensor interpretation, dimensions, and
    exact pilot/system geometry required for inference;
  - generation recipe covers optimizer, data volume, SNRs, seed, and other
    requested generation settings;
  - optional metadata covers asset-specific history and state structure.
- [x] Validate common manifest fields plus listed-file existence, byte sizes,
      and checksums before exposing an asset to a paper-specific loader.
- [x] Add focused common tests for valid manifests, unknown schema versions,
      missing fields and files, size/checksum failures, path traversal,
      compatibility mismatches, and interrupted atomic JSON replacement.
- [x] Introduce the common helper without migrating every paper at once.
      Migrate and test each asset type in its paper-specific phase below.

#### Asset-directory migration without regeneration

- [x] Adopt the bundled/external asset path helpers and common manifest
      validator one paper at a
      time. Assign artifact IDs, migrate payloads, update loaders, and remove
      that paper's dependence on `legacy_asset_slug()` only in its dedicated
      paper phase.
- [x] Inventory every directory under `aux/`, classify it as canonical,
      transient, obsolete, or uniquely experimental, and map every retained
      asset to its documented `assets/<scenario_id>/<owner>/<asset_type>/`
      destination.
- [x] Before each paper migration, record the proposed source-to-destination
      mapping and reject destination collisions, missing metadata, or unknown
      legacy formats before moving files. Do not add a permanent migration
      framework for this one-time work.
- [x] For each retained asset, record payload sizes and SHA-256 digests before
      migration, validate its legacy contents, create the new `manifest.json`,
      move the unchanged payload, and verify identical sizes and digests
      afterward.
- [x] Disable automatic asset generation and training during migration
      verification. Prove that each migrated canonical asset supports load-only
      estimator inference before deleting its old location.
- [x] Update all loaders, runners, configurations, documentation, and
      `.gitignore` rules from `aux/` to `assets/`.
- [x] Migrate canonical output directories from hash-suffixed configuration
      slugs to `outputs/<scenario_id>/`, write the normalized `config.json`,
      and reject collisions whose stored fingerprint differs.
- [x] Remove the old `aux/` tree and obsolete slug helpers only after `rg`
      finds no live references and all canonical assets pass checksum and
      load-only validation.
- [x] Do not migrate smoke/probe assets into `assets/`; regenerate them only in
      temporary test directories when needed and remove them afterward.

#### Shared helpers

- [x] Add a reusable helper module under `test/` that can:
  - build small `SystemConfig` and `ResourceGrid` variants;
  - create the corresponding channel source and evaluation system;
  - instantiate an estimator through a paper-provided factory;
  - run one or a few NMSE-only batches at a selected SNR;
  - validate output shape, dtype, device, finiteness, and error variance;
  - compare against a zero estimate and, where meaningful, LS interpolation;
  - use temporary asset/cache directories for training-based methods.
- [x] Keep paper-specific factories and expectations in paper-specific test
      files. Do not create a central registry containing every constructor and
      every paper-specific exception.
- [x] Add pytest markers such as `system_variants`, `training`, `compile`, and
      `slow`, and document commands for running each layer.
- [x] Ensure all test configurations are deterministic. Require tests that
      generate assets or other files to use `tmp_path` and explicit output
      roots instead of canonical `assets/`, `outputs/`, or `tuning/`
      directories.

#### Shared system variants

- [x] Define a small baseline with one UE, one or two BS antennas, a small
      multiple-of-12 subcarrier grid, and one pilot-bearing OFDM symbol.
- [x] Define an alternate grid with a different number of subcarriers and OFDM
      symbols, three UEs, four BS antennas, and shifted pilot-symbol positions.
- [x] Define a wider integration case with four UEs and a nontrivial BS array,
      three pilot-bearing OFDM symbols, and a non-power-of-two BS antenna count,
      while keeping runtime suitable for marked tests.
- [x] Exercise at least two supported pilot layouts. Vary pilot-symbol
      positions and comb spacing through UE count; add a custom pattern only
      where the estimator contract claims to support it.

#### Detector-facing error-variance audit

- [x] Audit every estimator for outputs of the form `scale * no`,
      `scale * no / pilot_power`, or `scale * ls_err_var`, including equivalent
      broadcasting or interpolation. Maintain an explicit list of estimators
      that require scalar calibration and those whose variances are derived
      directly from an estimator posterior or covariance model.

### Remaining shared asset distribution work

- [x] Support optional read-only package-local assets while keeping training,
      regeneration, and test roots explicit and external. Public distributions
      omit persistent inference payloads.
- [x] Audit the exact built-in and Edfors covariance recipes selected by the
      canonical runners against their READMEs, manifests, and result outputs.
- [x] Audit the exact Koller, Melgar, Li, and Baur recipes selected by every
      remaining UMi asset-backed result before pruning historical local assets.
      Resolve differences between runner defaults, paper READMEs, manifests,
      and canonical outputs, especially for Melgar, Li, and Baur.
- [x] Add an explicit `--prepare-assets` runner mode for regenerating an asset
      into an external root. Normal benchmark execution loads a package-local
      or explicitly selected external asset without fitting or training and
      reports both preparation and evaluation commands when it is absent.
- [x] Retain descriptive asset lifecycle operations in `asset.py`: normal
      runners use strict `load_covariance()` or `load_model()` functions, while
      `--prepare-assets` uses `estimate_covariance()`, `fit_model()`, or
      `train_model()` with an explicit root.
- [x] Apply strict package-local loading and external preparation to the built-in
      and Edfors covariance assets; validate package-local loading and explicit
      external regeneration for both asset types.
- [x] Apply the same strict package-local-loading and external-preparation contract
      to the selected Koller GMMs, Melgar DNNs, Li ReEsNet, and Baur VAE after
      their final assets have been regenerated. Test each public lifecycle
      handle: complete package-local assets may load, while a missing asset
      fails without writing and explains how to use an explicit external
      `asset_root`.
- [x] Keep smoke assets transient and cheap. Asset-backed smoke checks first
      prepare a tiny asset under an external or temporary root and then invoke
      the strict benchmark path against that root.
- [x] Maintain `docs/mimo_ofdm_chest/inventories/asset_inventory.md` as a generation catalog with
      the exact command, generated path, effective recipe, seed, expected
      hardware, approximate runtime, and disk usage for each asset type.
- [x] Record the complete generation recipe, training/fitting history where
      applicable, and payload checksums in newly generated asset manifests.
- [x] Link the selected built-in and Edfors covariance-manifest identities and
      exact manifest checksums from scenario-local `results.json` entries.
- [x] Link the selected Koller, Melgar, Li, and Baur asset-manifest identities
      and exact manifest checksums from their scenario-local `results.json`
      entries. Regenerated
      GPU models need only be statistically consistent, not byte-identical.
- [ ] Remove covariance, GMM, and neural-model payloads and orphaned runtime
      manifests from the public release snapshot while retaining their
      generation evidence and scenario-local result identities.
- [ ] Validate the asset-free release from a clean clone: build and install the
      wheel, verify package and estimator imports, exercise an asset-free LS
      smoke run, check actionable missing-asset errors, and exercise one
      external preparation and loading path.

### Execution rules

#### Numerical acceptance

- Use broad method-specific checks rather than requiring every estimator to
  beat LS on tiny training runs.
- Require estimates to be non-degenerate and broadly useful, for example NMSE
  below the zero-estimate level where the method and tiny training budget make
  that reasonable.
- Where stochastic power is weak, compare high and low SNR on shared channels
  with a generous tolerance rather than demanding strict monotonicity.
- Prefer analytical invariants when available: pilot pass-through, polynomial
  recovery, transform round trips, limiting cases, posterior covariance
  positivity, or equivalence to a baseline configuration.

#### Optuna parameter selection

- Run tuning trials serially within a process. Deterministic probes reset
  process-global Sionna and Torch random-number generators, so threaded
  Optuna objectives cannot preserve the common-channel sampling contract.
- Reset probe RNGs after estimator and system construction so constructor
  randomness cannot alter the channel samples seen by different trials. Use
  one fixed probe seed per study and preserve the intended cross-SNR sampling
  policy when resetting it.
- Let Optuna search continuous values at full precision, but round selected
  continuous constructor parameters to at most three significant digits before
  canonical use. Preserve the raw winner in tuning evidence, validate the
  rounded candidate on the independent deterministic holdout, and document
  both values. Integer and categorical parameters remain unchanged.

#### Detector-facing error-variance calibration

- Perform this calibration only after the paper's algorithmic refactoring,
  estimator hyperparameter optimization, and any model training are complete.
  Changing the estimate, interpolation, solver, or trained weights invalidates
  the calibrated scale.
- Tune only the variance scale in this final calibration study. Freeze all
  estimator parameters and assets to the exact versions intended for the
  canonical benchmark.
- Use independent deterministic tuning channels and the complete relevant SNR
  range. Do not optimize against the existing canonical benchmark outputs or
  reuse their Monte Carlo samples.
- Use a detector-facing objective based on BER/BLER, because multiplying
  `err_var` by a scalar does not change `h_hat` or NMSE. Define one shared
  aggregate objective and statistical budget before calibrating the first
  paper, then use it consistently across methods.
- Record the Optuna search range, sampler seed, trial count, evaluation budget,
  objective, best value, and selected scale in the paper's tuning summary,
  README, and implementation report.
- Rerun the canonical benchmark after selecting the scale and inspect both
  BER/BLER and NMSE. NMSE must remain unchanged apart from Monte Carlo or
  numerical variation.

## Phase 2: Shared Compilation Test Infrastructure

### Completed work

- [x] Add a helper that runs eager inference, then
      `torch.compile(..., backend="eager", fullgraph=False)`, and compares
      outputs. Use the eager backend for stable CI coverage.
- [x] Add an optional CUDA/Inductor smoke path for local validation; do not
      require expensive compilation in the default unit-test suite.

### Execution rules

- Classify each estimator using the following compilation tiers:
  - Tier A: inference core compiles as a full graph;
  - Tier B: complete estimator call runs through `torch.compile` with a small,
    documented number of graph breaks;
  - Tier C: unsupported training, SciPy/CVXPY, or asset-loading regions are
    explicitly disabled while tensor inference subroutines compile.
- Use Tier B as the default target. Require a written reason for Tier C.
- Never include asset discovery, file I/O, offline training, or SciPy/CVXPY
  execution in the default compiled region.
- Do not replace clear tensor code with obscure code solely to reduce graph
  breaks unless the measured benefit is material.

## Phase 2.5: Package And Sionna 2.0.1 Migration

- [x] Move importable code into the `src/sionna_bench/` package and expose
      paper-local estimator imports plus selected root-level estimator exports.
- [x] Move canonical results to repository-level `outputs/<paper_id>/` and
      make `SystemConfig.output_dir` authoritative.
- [x] Establish paper-local and shared-evaluation package paths for canonical
      assets. Preserve the stale pre-regeneration snapshot only under the
      ignored root `assets/` directory.
- [x] Build a wheel, inspect its contents, install it outside the checkout, and
      verify root-level and paper-local imports and package boundaries.
- [x] Run the complete fast tests and representative compile, analytical,
      covariance-backed, GMM-backed, and neural-model smoke checks against the
      installed package.
- [ ] Complete the paper-specific Sionna 2.0.1 tuning, asset-generation, and
      README documentation tasks listed in the per-paper work order below.
- [ ] Freeze the selected algorithm, architecture, training, and inference
      hyperparameters. Then retrain or re-estimate the final covariance, GMM,
      and neural-model assets. Use the ignored stale snapshot only for
      comparison; do not commit it as a release asset.
- [ ] Reconcile regenerated assets with the final source freeze. Audit changes
      to generation-relevant code after each asset was produced, and
      regenerate any asset whose generating implementation changed materially.
- [ ] Audit and retain only assets selected by canonical runners. Verify that
      every manifest covers its payload and records the final compatibility,
      generation recipe, seed, relevant metadata, file size, and checksum. Move these
      assets into their package-local paths and remove the temporary
      package-asset ignore rule.
- [ ] After estimator code, hyperparameters, and refreshed assets are frozen,
      rerun detector-facing variance calibration where required. Update paper
      READMEs, reports, and the error-variance audit with the selected scales.
- [ ] Rerun every canonical smoke and benchmark evaluation with the migrated
      package, refreshed assets, and final variance scales. Compare against the
      recorded regression references, investigate material changes, then
      replace canonical outputs and regenerate the leaderboard.
- [ ] Build and install the final asset-bearing wheel from a clean clone.
      Verify root-level and paper-local imports, canonical asset loading,
      representative inference, and absence of outputs or temporary tuning
      artifacts from the wheel.
      Record compressed and uncompressed wheel sizes and the largest bundled
      payload, then verify that the intended package registry accepts the
      resulting wheel. A temporary wheel containing only the completed K17
      spectral GMM plus the two covariance families is already 100,064,670
      bytes; schema-2 component sharding avoids oversized Git objects but does
      not reduce the aggregate wheel size.
- [ ] Before release, require a Sionna version whose published package contains
      `BasePilotChannelEstimator`, the direct full-grid
      `BaseChannelEstimator` contract, and `_PilotGroupMixin`. Update the
      dependency lower bound and remove the temporary source-checkout Quick
      Start step only after validating that published package.
- [ ] Before public distribution, complete the project metadata in
      `pyproject.toml`: summary, README, authorship, license, classifiers, and
      project URLs. Treat authorship and licensing as explicit human decisions
      rather than inferring them from the checkout. Summary, README metadata,
      and Python/scientific-computing classifiers are complete and verified in
      a wheel; authorship, license, and project URLs remain. Reclassify
      `cvxpy` is now exposed only through the `test-lyu` test-oracle extra; no
      maintained source or tuning module imports it. Authorship, license, and
      project URLs remain unresolved human-owned metadata.

## Per-Paper Work Order

Within every paper, use this order:

Run each estimator on the shared system variants whose resource-grid
dimensions, pilot layout, and receive-array dimensions satisfy its documented
contract. For requirements such as uniform combs, rectangular active groups,
equal pilot power, or a specific number of pilot symbols, include both a
supported case and an intentional rejection case in that paper's tests. A
clear, documented rejection is valid for an unsupported variant; an estimator
does not have to run on all three shared variants.

0. Before editing code, read the relevant section of this file, the repository
   and paper READMEs, applicable contracts under `docs/`, the paper reports,
   shared interfaces and their tests, and the existing paper-specific tests.
   Report a short **Applicable contracts** list naming the requirements and
   interface assumptions that govern the work.
1. Add system-variant and analytical tests around the current behavior.
2. Reread the paper, reports, and implementation; write down findings before
   changing algorithms.
3. Fix correctness, contract, documentation, and readability issues.
4. Improve compileability with the new behavioral tests as protection. Run
   `torch._dynamo.explain` on representative estimator inputs and record the
   meaningful graph-break causes; do not assert exact graph-break counts.
5. Rerun focused tests, compile diagnostics, and canonical smoke tests.
6. If the estimator uses a scalar multiple of noise or LS error variance,
   freeze all other code, parameters, and trained assets, then perform the
   Optuna detector-facing variance calibration defined above.
7. Reassess the paper's entry in `docs/mimo_ofdm_chest/inventories/estimator_inventory.md`. Update its
   preparation/inference split, compile tier, graph-break causes, and
   classification if refactoring changed any of them.
8. Update the remaining documentation and commit that paper as one coherent
   unit.

### 1. Shared Sionna integration and built-in references

- [x] Test the repository-owned `TimeDomainInterpolator` across all three
      shared system variants. Rely on Sionna's unit suite for
      `BaseChannelEstimator` and `_PilotGroupMixin`, with paper tests retaining
      downstream integration coverage.
- [x] Move the ignored canonical built-in LMMSE and LS results from top-level
      `outputs/builtin/` into `outputs/builtin/<scenario_id>/`, validate
      them without rerunning, and track them as regression references.
- [x] Verify time interpolation for changed grid dimensions. Pilot-group
      ordering, coordinates, and gather/scatter behavior are covered by
      Sionna's unit suite and exercised through paper-specific tests.
- [x] Run built-in LS nearest-neighbor, LS linear, and LMMSE wrappers as
      reference integrations on the same small systems.
- [x] Replace generic result `namespace` arguments in `result_store.py` with
      an explicit `smoke: bool = False` contract. Remove `_with_namespace()`
      while preserving only `outputs/<scenario_id>/` and
      `outputs/smoke/<scenario_id>/`.
- [x] Update every built-in and paper benchmark runner to replace
      `output_namespace` with `smoke`, pass the CLI `--smoke` flag directly,
      and remove support for arbitrary temporary result namespaces.
- [x] Remove result-namespace arguments from asset APIs, including the ignored
      `namespace` argument of built-in covariance loading, and verify that no
      asset loader depends on result-output placement.
- [x] Add result-store tests for canonical and smoke paths and manifests, and
      update documentation and historical runnable commands that reference the
      old namespace API.
- [x] Run representative built-in, analytical, and asset-backed runner smoke
      checks and verify that they write only beneath the intended smoke result
      directories.
- [x] Migrate built-in LMMSE covariance assets to a documented manifest schema,
      asset-specific compatibility fields, checksums, and atomic writes.
- [x] Let built-in covariance generation/loading receive an `asset_root` that
      defaults to `assets/`, so tests can use `tmp_path` without touching
      canonical covariance assets.
- [x] Test that built-in covariance loading rejects changed grid/receive-array
      compatibility, missing named payloads, and incorrect array shapes or
      dtypes.
- [x] Seed covariance generation explicitly and include the seed in the
      generation recipe so regenerated assets are reproducible.
- [x] Audit LMMSE covariance compatibility fields, add cyclic-prefix length and
      fixed topology-distribution parameters that affect covariance
      generation, and keep generation settings confined to the generation
      recipe.
- [x] Align `src/sionna_bench/mimo_ofdm_chest/papers/builtin/` with `docs/paper_architecture.md`: rename
      `covariance.py` to `asset.py`, expose separate `estimate_covariance()` and
      `load_covariance()` functions, and expose `run_benchmark()` while keeping
      covariance preparation in the runner and injection into
      `LMMSEEstimatorWrapper`.
- [x] Run eager-versus-compiled checks and `torch._dynamo.explain`
      diagnostics for `TimeDomainInterpolator`, built-in LS, and built-in LMMSE
      calls. Record their compilation tiers and shared graph-break causes
      before paper-specific compilation work; Sionna owns equivalent checks
      for its base estimator and pilot-group utility.
- [x] Regenerate the built-in covariance with the corrected Sionna 2.0.1
      channel model. Record its exact command, generation recipe, seed,
      hardware, runtime, manifest identity, and payload size in
      `docs/mimo_ofdm_chest/inventories/asset_inventory.md`.
- [x] Rerun the canonical built-in benchmark with Sionna 2.0.1 and the
      regenerated covariance asset. Compare every method with its prior
      regression reference, investigate material changes, then replace the
      canonical outputs and regenerate the leaderboard.

### 2. Coleri et al., 2002

- [x] Parameterize existing analytical interpolation tests over multiple FFT
      sizes, OFDM-symbol counts, pilot-symbol positions, and UE comb spacings.
- [x] Test all four variants: second-order, low-pass, cubic spline, and
      time-domain.
- [x] Confirm documented behavior at pilot subcarriers, boundaries, and between
      pilot-bearing OFDM symbols.
- [x] Reread the interpolation formulas and recheck offset conventions,
      second-order coefficients, low-pass normalization, spline axes, and the
      time-domain circular-delay interpretation.
- [x] Align Coleri with `docs/paper_architecture.md`: retain only
      `estimator.py`, `tuning.py`, and `run_benchmark.py`, and expose the
      standard `run_tuning()` and `run_benchmark()` handles.
- [x] Compile all four inference variants to Tier A or B. Keep constructor-time
      SciPy spline-weight generation outside the compiled inference path.
- [x] Add one shared detector-facing variance-scale constructor argument for
      all four `LSCombEstimator` variants. After the estimator refactor is
      frozen, calibrate that scale using the shared detector-facing protocol
      and update the audit, documentation, and canonical results.
- [x] Run every complete estimator variant across all shared supported systems
      and validate shapes, dtypes, devices, finite outputs, and non-negative
      error variances.
- [x] Test explicit rejection of nonuniform combs, incomplete full-grid combs,
      and unequal active pilot-symbol counts per stream.
- [x] Document Coleri's supported pilot geometry, Tier A inference result, and
      constructor-time SciPy spline-weight boundary in the paper README.
- [x] Rerun the low-pass and detector-facing variance Optuna studies with
      Sionna 2.0.1. Update the paper README with each objective, search space,
      sampler and probe seeds, trial count, Monte Carlo/SNR budget, selected
      parameters, best value, command, and runtime. This paper has no
      persistent inference asset.
- [x] Rerun the canonical Coleri benchmark with Sionna 2.0.1 and the selected
      low-pass and detector-facing variance parameters. Compare every method
      with its prior regression reference, investigate material changes, then
      replace the canonical outputs and regenerate the leaderboard.
- [x] Repeat the low-pass study after the shared Sionna RNG correction so all
      trials use reproducible common channel draws. Preserve the pre-fix study,
      independently validate the selected parameters, then repeat variance
      calibration and the canonical benchmark and update all Coleri records.

### 3. Edfors et al., 1996

- [x] Let Edfors covariance generation/loading and the runner/tuning handles
      receive an `asset_root` defaulting to `assets/` for isolated tests.
- [x] Generate tiny covariance assets for at least two alternative geometries
      in temporary directories.
- [x] Exercise LS, diagonal LMMSE, and LMMSE with the same configured
      `num_taps` across changed subcarrier and antenna dimensions.
- [x] Add transform-domain limiting tests for all-tap LS and LMMSE, maximal
      supported truncation, and a delay-limited synthetic channel.
- [x] Recheck DFT ordering, covariance application, truncation, and propagated
      confidence scaling against the paper and reports.
- [x] Align Edfors with `docs/paper_architecture.md`: rename `covariance.py` to
      `asset.py`, expose separate `estimate_covariance()` and
      `load_covariance()` functions, prepare covariance assets in the runner,
      inject them into the estimator, and standardize `run_tuning()` and
      `run_benchmark()`.
- [x] Migrate Edfors covariance assets to the shared manifest envelope while
      retaining the existing exact subcarrier-group compatibility data.
- [x] Separate covariance compatibility from the Monte Carlo generation recipe;
      add checksums, array-key/shape validation, and atomic writes.
- [x] Test that Edfors covariance loading rejects changed ordered pilot
      geometry, missing NPZ keys, and incorrect covariance shape or dtype.
- [x] Remove the Edfors loader's dependence on legacy manifest `key_fields`;
      select and validate assets through `compatibility` and
      `generation_recipe` instead.
- [x] Exclude covariance estimation/loading from compilation; target Tier A or
      B for online inference.
- [x] Test intentional rejection of ragged active pilot-frequency groups with
      the specific shared equal-length-group error.
- [x] Document the supported pilot, tap-count, covariance, and receive-array
      geometry in the paper README and implementation report.
- [x] Run the final calibrated canonical smoke test and remove its transient
      result and covariance artifacts.
- [x] After all estimator and covariance work is frozen, calibrate the shared
      `chest_err_const` for LS, diagonal LMMSE, and LMMSE using one aggregate
      detector-facing study, then update the audit, documentation, and
      canonical results.
- [x] Rerun tap-count tuning, regenerate the selected covariance asset, and
      recalibrate detector-facing variance with Sionna 2.0.1. Document in the
      paper README the covariance command, recipe, seed, hardware, runtime,
      manifest identity, and payload size, plus both Optuna objectives, search
      spaces, seeds, trial/evaluation budgets, selected values, and runtimes.
- [x] Rerun the canonical Edfors benchmark with Sionna 2.0.1 and the selected
      tap count, covariance asset, and detector-facing variance scale. Compare
      every method with its prior regression reference, investigate material
      changes, then replace the canonical outputs and regenerate the
      leaderboard.

### 4. Delfeld et al., 2025

- [x] Test RKHS estimation on multiple comb spacings, grid sizes, UE counts,
      and BS antenna dimensions using small ranks and iteration counts.
- [x] Add or retain checks for the adjoint relationship between `S` and `S^H`,
      sparse-step behavior, debiasing, and low-rank RKHS reconstruction.
- [x] Recheck native subcarrier reconstruction and subsequent
      `TimeDomainInterpolator` use against the paper terminology.
- [x] Audit all shape-dependent constants and remove canonical-grid assumptions.
- [x] Align Delfeld with `docs/paper_architecture.md`: retain only
      `estimator.py`, `tuning.py`, and `run_benchmark.py`, and expose the
      standard `run_tuning()` and `run_benchmark()` handles.
- [x] Vectorize pilot-group processing and remove inference-time scalar
      extraction without introducing special-case optimization branches;
      verify that the complete estimator reaches Tier A compilation.
- [x] After all RKHS refactoring and hyperparameter tuning is frozen, calibrate
      `err_var_scale` using the shared detector-facing protocol and update the
      audit, documentation, and canonical results.
- [x] Rerun RKHS hyperparameter tuning and detector-facing variance calibration
      with Sionna 2.0.1. Update the paper README with each objective, search
      space, sampler and probe seeds, trial count, Monte Carlo/SNR budget,
      selected parameters, best value, command, and runtime. This paper has no
      persistent inference asset.
- [x] Rerun the canonical Delfeld benchmark with Sionna 2.0.1 and the selected
      RKHS and detector-facing variance parameters. Compare it with the prior
      regression reference, investigate material changes, then replace the
      canonical output and regenerate the leaderboard.
- [x] Repeat the RKHS study after the shared Sionna RNG correction so all
      trials use reproducible common channel draws. Preserve the pre-fix study,
      independently validate and round the selected parameters, then repeat
      variance calibration and the canonical benchmark, synchronize the
      estimator and runner defaults with the selected scale, and update all
      Delfeld records.

### 5. Koller et al., 2022

- [x] Let the GMM asset lifecycle handle receive an `asset_root` defaulting to
      `assets/` for isolated training and loading tests; keep repository paths
      out of the estimator constructors.
- [x] Train tiny one- or two-component spectral and spatial GMMs for alternative
      geometries using temporary caches and very small EM budgets.
- [x] Test both `SpectralGMMChannelEstimator` and
      `SpatialGMMChannelEstimator` across changed UE and BS antenna counts.
- [x] Add direct checks for real/complex stacking, component likelihoods,
      responsibilities summing to one, posterior means, and non-negative
      posterior variances.
- [x] Verify vectorized spatial group processing against a simple group loop on
      several shapes, not only the canonical geometry.
- [x] Recheck pooled-statistics assumptions, observation selection, noise
      covariance handling, and equations (13), (15), and (16).
- [x] Align Koller with `docs/paper_architecture.md`: replace
      `gmm_training.py` with `asset.py`, expose separate `fit_model()` and
      `load_model()` functions, prepare GMM parameters in the runner, inject
      them into both estimators, and standardize `run_tuning()` and
      `run_benchmark()`.
- [x] Migrate spatial and spectral GMM assets to the global asset-manifest
      schema version 2. Keep spatial parameters in a compact NPZ and store one
      bounded spectral covariance payload per component so release assets do
      not require Git LFS.
- [x] Include estimator mode, real-vector convention, vector dimensions, exact
      pilot-group coordinates, and pilot values/powers in compatibility rather
      than relying on the legacy asset slug and `num_groups`.
- [x] Replace partial metadata matching with explicit schema and identity
      validation; validate GMM weights, means, and covariance shapes before use.
- [x] Test that both GMM loaders reject incompatible pilot/receive-vector
      geometry, missing parameter keys, and invalid weight, mean, or covariance
      shapes and dtypes.
- [x] Remove both GMM loaders' dependence on legacy manifest `key_fields`;
      use the explicit `compatibility` and `generation_recipe` sections
      instead.
- [x] Add checksums and atomic writes, and explicitly migrate the selected K17
      spectral and K26 spatial canonical assets.
- [x] Keep GMM fitting and cache I/O outside compilation; target Tier A or B for
      both online conditional-mean paths.
- [x] Rerun the spatial component-count Optuna study with Sionna 2.0.1 under
      the ignored external asset root, independently retain K40 over K26,
      generate and validate the selected schema-2 spatial GMM, run an isolated
      candidate benchmark, and document its recipe, study, holdout, manifest,
      payload, runtime, and numerical comparison.
- [x] Finish the ten-trial spectral component-count study and select its Optuna
      winner without an additional independent holdout, as explicitly chosen
      for this regeneration pass. Generate the selected split schema-2
      spectral GMM and document its
      generation recipe, sample count, EM choices, seeds, hardware, runtime,
      manifest identity, payload sizes, Optuna budget, and selected component
      count alongside the completed spatial evidence.
- [x] Rerun complete candidate Koller benchmarks with Sionna 2.0.1 and the
      selected spectral and spatial GMM assets. Compare both methods with their
      prior regression references and investigate material changes without
      replacing canonical outputs.
- [ ] After explicit review of the paired K17/K40 candidate evidence, promote
      the paired assets and results atomically and regenerate the leaderboard.

### 6. Wei et al., 2017

- [x] Parameterize comb-transform, delay-estimation, and high-SNR reconstruction
      tests over multiple valid comb lengths and antenna counts.
- [x] Add integration cases with changed UE count and OFDM-symbol geometry.
- [x] Test clear rejection of nonuniform or incompatible comb geometries.
- [x] Recheck SBL posterior updates, shared-support variance updates, stopping
      behavior, and posterior variance propagation against the paper.
- [x] Keep Sionna linear interpolation clearly identified as the benchmark
      adaptation, and document that paper-faithful causal end-padding was
      rejected because it performs poorly for continuous-delay UMi channels.
- [x] Align Wei with `docs/paper_architecture.md`: retain only `estimator.py`,
      `tuning.py`, and `run_benchmark.py`, and expose the standard
      `run_tuning()` and `run_benchmark()` handles.
- [x] Remove avoidable `.item()` and Python data-dependent branches from the
      inference loop; verify that the complete estimator reaches Tier A.
- [x] After the SBL implementation and algorithmic hyperparameters are frozen,
      calibrate `posterior_variance_scale` using the shared detector-facing
      protocol and update the audit, documentation, and canonical results.
- [x] Rerun SBL hyperparameter tuning and detector-facing variance calibration
      with Sionna 2.0.1. Update the paper README with each objective, search
      space, sampler and probe seeds, trial count, Monte Carlo/SNR budget,
      selected parameters, best value, command, and runtime. This paper has no
      persistent inference asset.
- [x] Rerun the canonical Wei benchmark with Sionna 2.0.1 and the selected SBL
      and detector-facing variance parameters. Compare it with the prior
      regression reference, investigate material changes, then replace the
      canonical output and regenerate the leaderboard.
- [x] Repeat the full-Monte-Carlo SBL study after the shared Sionna RNG
      correction so all trials use reproducible common channel draws. Preserve
      the pre-fix study, independently validate the selected parameters, then
      repeat variance calibration and the canonical benchmark and update all
      Wei records.

### 7. Balevi et al., 2020

- [x] Parameterize the fitted pilot-grid path over changed pilot-grid sizes,
      UE counts, and BS antenna counts.
- [x] Verify that independently batched networks remain equivalent to separate
      optimization problems on several shapes and that no parameters,
      normalization statistics, or Adam state are shared.
- [x] Use very small runtime fitting budgets in system-variant tests and assert
      finite, non-degenerate behavior rather than canonical performance.
- [x] Recheck the deep-decoder architecture, fixed input, normalization,
      upsampling schedule, denoising objective, and pilot-observation adaptation.
- [x] Align Balevi with `docs/paper_architecture.md`: keep observation-specific
      online fitting in `estimator.py`, retain optional `tuning.py`, and expose
      the standard `run_tuning()` and `run_benchmark()` handles without a
      persistent-asset module.
- [x] Compile the denoiser forward path. Treat model construction, optimizer
      creation, autograd fitting, and `torch.func` state stacking as an
      explicitly documented Tier C outer region unless a simple Tier B path is
      demonstrated.
- [x] Rerun Balevi optimization-behavior tuning and the canonical benchmark
      after the bias-free architecture correction; update the README, reports,
      tuning summaries, canonical output, and leaderboard.
- [x] After the inference-time fitting procedure and its hyperparameters are
      frozen, calibrate `err_var_scale` using the shared detector-facing
      protocol and update the audit, documentation, and canonical results.
- [x] Rerun optimization-behavior tuning and detector-facing variance
      calibration with Sionna 2.0.1. Update the paper README with each
      objective, search space, sampler and probe seeds, trial count, Monte
      Carlo/SNR budget, selected parameters, best value, command, and runtime.
      This paper fits online and has no persistent inference asset.
- [x] Rerun the canonical Balevi benchmark with Sionna 2.0.1 and the selected
      fitting and detector-facing variance parameters. Compare it with the
      prior regression reference, investigate material changes, then replace
      the canonical output and regenerate the leaderboard.
- [x] Repeat the 30-trial Balevi optimization-behavior study after the shared
      RNG correction, using probe seed `314159` so every trial sees the same
      reproducible Sionna 2.0.1 channel draws independently of estimator seed
      `42`. Preserve the pre-fix study as historical evidence.
- [x] With the corrected fitting parameters frozen, repeat the cached
      detector-facing variance calibration and independent holdout, apply the
      three-significant-digit rule, and update the error-variance audit.
- [x] Rerun the corrected canonical Balevi benchmark, compare its complete
      metric curves against both the pre-fix Sionna 2.0.1 output and the older
      regression reference, then update the README, reports, canonical output,
      and leaderboard.

### 8. Melgar et al., 2022

- [x] Let both DNN loaders and estimator constructors receive an `asset_root`
      defaulting to `assets/` for isolated training and loading tests.
- [x] Add tiny one-SNR training fixtures for `DNNChannelEstimator` and
      `InterpolatedDNNChannelEstimator`, writing caches only under `tmp_path`.
- [x] Exercise changed subcarrier, OFDM-symbol, UE, and BS antenna dimensions;
      ensure model dimensions are derived rather than canonical constants.
- [x] Verify cache metadata rejects incompatible system geometry and accepts an
      exactly matching temporary model.
- [x] Migrate Model 2 and Model 3 checkpoints to separate documented schema
      versions using the shared manifest envelope.
- [x] Add exact pilot subcarrier coordinates and pilot values/powers to Model 2
      compatibility, and the complete LS-interpolation pilot layout to Model 3
      compatibility.
- [x] Separate architecture/geometry compatibility from the generation recipe;
      add checksums, state-key/shape validation, and atomic writes.
- [x] Test that both Melgar checkpoint loaders reject incompatible pilot/grid
      geometry, missing state keys, and incorrectly shaped or typed tensors.
- [x] Remove both Melgar checkpoint loaders' dependence on legacy manifest
      `key_fields`; use explicit `compatibility` and `generation_recipe`
      metadata.
- [x] Keep checkpoints trained on explicit SNR values reusable when only the
      later evaluation SNR sweep changes; retain the configured sweep as
      generation metadata rather than compatibility in this case.
- [x] Recheck Model 2 and Model 3 architectures, latent dimensions,
      per-SNR-weight selection, interpolation placement, loss, and training
      sample accounting against the paper.
- [x] Align Melgar with `docs/paper_architecture.md`: merge `dnn_dataset.py`
      and `dnn_training.py` into `asset.py`, expose separate `train_model()` and
      `load_model()` functions, prepare model states in the runner, inject them
      into both estimators, and add standardized `run_tuning()` and
      `run_benchmark()` handles.
- [x] Keep training and cache loading outside compilation; target Tier A or B
      for trained-model inference.
- [x] Add a detector-facing variance-scale constructor argument to
      `DNNChannelEstimator` without including it in checkpoint compatibility or
      trained-weight identity.
- [x] After both Melgar models and checkpoints are frozen, calibrate the Model 2
      and Model 3 variance scales using the shared detector-facing protocol,
      then update the audit, documentation, and canonical results.
- [x] Rerun the Model 3 training study with Sionna 2.0.1, independently retain
      the rounded learning-rate and batch-size choice, generate and strictly
      validate the complete per-SNR checkpoint bank, recalibrate its
      detector-facing variance, run an isolated candidate benchmark, and
      document the study, holdout, recipes, sample accounting, manifests,
      payloads, runtimes, and numerical comparison.
- [x] Rerun the Model 2 training study and independent holdout with Sionna
      2.0.1, generate and validate its selected per-SNR checkpoint bank,
      recalibrate detector-facing variance, and document the same complete
      training and asset evidence as for Model 3.
- [x] Rerun the complete canonical Melgar benchmark with Sionna 2.0.1, both
      selected checkpoint banks, and their calibrated variance scales. Compare
      both methods with their prior regression references, investigate
      material changes, then promote the paired assets and results atomically
      and regenerate the leaderboard.

### 9. Li et al., 2020

- [x] Let ReEsNet loading/training and estimator construction receive an
      `asset_root` defaulting to `assets/` for isolated tests.
- [x] Add tiny one-SNR ReEsNet training fixtures for at least two small resource
      grids and temporary cache locations.
- [x] Exercise changed UE and BS antenna counts and verify that antennas and
      streams are treated independently where intended.
- [x] Add tests for pilot-grid layout, super-resolution output dimensions,
      residual blocks, long skip connection, and linear/pilot consistency.
- [x] Migrate ReEsNet checkpoints to a documented schema version using the
      shared manifest envelope, preserving the existing pilot-layout and tensor
      hashes.
- [x] Remove detector-facing `err_var_scale` from trained-weight identity so
      changing confidence calibration does not require retraining.
- [x] Separate compatibility, generation recipe, and optional training
      metadata; add checksums, state-key/shape validation, and atomic writes.
- [x] Test that ReEsNet checkpoint loading rejects incompatible pilot/full-grid
      geometry, missing state keys, and incorrectly shaped or typed tensors.
- [x] Remove ReEsNet loading and estimator metadata dependence on legacy
      manifest `key_fields`; use explicit compatibility and generation-recipe
      metadata.
- [x] Reread the super-resolution architecture and reference [13] adaptation;
      verify that the implemented upsampling and skip path remain defensible.
- [x] Align Li with `docs/paper_architecture.md`: replace `training.py` with
      `asset.py`, expose separate `train_model()` and `load_model()` functions,
      prepare model states in the runner, inject them into
      `ReEsNetChannelEstimator`, and add standardized `run_tuning()` and
      `run_benchmark()` handles.
- [x] Keep training outside compilation; target Tier A or B for inference.
- [x] After ReEsNet training and all architecture work are frozen, calibrate
      `err_var_scale` using the shared detector-facing protocol and update the
      audit, documentation, and canonical results.
- [ ] Rerun ReEsNet training hyperparameter tuning with Sionna 2.0.1, generate
      the selected per-SNR checkpoint assets, and then recalibrate
      detector-facing variance. Use ten Optuna trials; for each SNR-specific
      model, use 1,024 training batches of size 16, 256 validation batches of
      size 16, at most 50 epochs, and patience 5. Update the paper README with
      generation commands and recipes, channel/sample accounting, seeds, hardware,
      per-SNR and total runtimes, manifest identities, and payload sizes, plus
      each Optuna objective, search space, trial/evaluation budget, selected
      parameters, best value, and runtime.
- [ ] Run an isolated complete Li candidate benchmark with Sionna 2.0.1, the
      selected ReEsNet checkpoints, and the recalibrated detector-facing
      variance scale. Compare it with the prior regression reference and
      investigate material changes without replacing canonical outputs.

### 10. Baur et al., 2022

- [x] Let both VAE loaders and estimator construction receive an `asset_root`
      defaulting to `assets/` for isolated training and loading tests.
- [x] Add tiny one-SNR training fixtures for both noisy and real VAE variants
      with temporary model caches.
- [x] Exercise changed BS antenna arrays first, then changed UE and grid
      dimensions where the spatial model contract permits them.
- [x] Extend covariance tests to trained outputs: symmetry, positive
      definiteness, transform round trips, and finite posterior estimates.
- [x] Migrate noisy and real VAE checkpoints to documented schema versions
      using the shared manifest envelope.
- [x] Add exact pilot coordinates and pilot values/powers to compatibility, in
      addition to receive-vector and architecture dimensions.
- [x] Separate compatibility, generation recipe, and optional training
      metadata; add checksums, state-key/shape validation, and atomic writes.
- [x] Test that both VAE checkpoint loaders reject incompatible pilot/receive
      geometry, missing state keys, and incorrectly shaped or typed tensors.
- [x] Remove both VAE checkpoint loaders' dependence on legacy manifest
      `key_fields`; use explicit compatibility and generation-recipe metadata.
- [x] Recheck encoder/decoder parameterization, ELBO terms, transform-domain
      representation, full covariance output, and both paper variants.
- [x] Align Baur with `docs/paper_architecture.md`: replace `vae_training.py`
      with `asset.py`, expose separate `train_model()` and `load_model()`
      functions, prepare model states in the runner, inject them into
      `VAEChannelEstimator`, and add standardized `run_tuning()` and
      `run_benchmark()` handles.
- [x] Keep VAE training and cache I/O outside compilation; target Tier A or B
      for encoder/decoder inference and posterior channel estimation.
- [x] After VAE training and covariance validation are frozen, calibrate
      `err_var_scale` for the selected canonical `VAE-noisy` method using the
      shared detector-facing protocol, then update the audit, documentation,
      and canonical results.
- [x] Rerun the ten-trial full-effort VAE training study with calibrated Sionna
      2.0.1 using a reusable bank of 6,400 training and 256 validation channels
      per SNR, 200 epochs, patience 20, and best-checkpoint restoration. Train
      the selected eleven-SNR `VAE-noisy` bank, recalibrate detector-facing
      variance, and document the complete study, assets, seeds, environment,
      manifests, and runtimes.
- [x] Run the isolated canonical Baur benchmark with the selected checkpoints
      and retained detector-facing variance scale, and compare all curves with
      the prior regression reference.
- [x] Finish promotion in the current worktree by placing the selected model
      bank directly under the bundled `vae-noisy` asset directory, then verify
      strict loading and the linked manifest checksums before committing it.

### 11. Lyu et al., 2026

- [x] Retain the unusually expensive Optuna setup as an explicitly bounded
      experimental search and document why its objective budget is smaller
      than for faster estimators.
- [x] Extend existing solver tests to multiple observation lengths and diffuse
      support lengths.
- [x] Add small end-to-end system variants for supported uniform-comb pilot
      layouts and changed antenna counts.
- [x] Recheck the implemented P2 constraints, Hermitian Toeplitz construction,
      ADMM updates, sparse/diffuse decomposition, pilot averaging, and all
      numerical safeguards.
- [x] Confirm that the estimator remains pilot-domain with explicit
      interpolation because the paper assumes observations on all subcarriers.
- [x] Align Lyu with `docs/paper_architecture.md`: keep the online ADMM and
      reference solvers in `estimator.py`, retain optional `tuning.py`, and
      expose the standard `run_tuning()` and `run_benchmark()` handles without
      introducing an asset module.
- [x] Verify that the complete batched Torch vector-ADMM inference path reaches
      Tier A compilation with eager-equivalent outputs.
- [x] After the sparse/diffuse solver and hyperparameters are frozen, calibrate
      `err_var_scale` using the shared detector-facing protocol and update the
      audit and documentation. Keep the calibrated result staged with the
      candidate until explicit promotion.
- [x] Complete the accepted Sionna 2.0.1 solver-study budget of 80 objective
      evaluations. Document its command, objective, search space, sampler and
      probe seeds, completed and interrupted trial counts, Monte Carlo/SNR
      budget, selected parameters, best value, and runtime in the paper README.
      This paper has no persistent inference asset.
- [x] Run an isolated complete Lyu candidate benchmark with Sionna 2.0.1 after
      freezing solver and detector-facing variance decisions. Compare it with
      the prior regression reference and document the low-SNR NMSE gains,
      high-SNR NMSE regressions, and BER/BLER improvements without replacing
      canonical outputs.
- [ ] After explicit review of the mixed Lyu candidate evidence, promote or
      reject the staged parameters and result. If promoted, replace the
      canonical output and regenerate the leaderboard.

## Phase 3: Cross-Paper Review

- [x] Run all fast unit, analytical, and compile-eager tests.
- [x] Run the marked system-variant suite across every estimator.
- [x] Run the marked tiny-training suite for Koller, Balevi, Melgar, Li, and
      Baur with caches redirected to temporary directories.
- [x] Produce a compact matrix listing supported geometries, intentional
      restrictions, compile tier, graph-break causes, test runtime, and asset
      requirements for every method.
- [x] Add a cross-paper Optuna inventory recording the parameters, trial count,
      objective-evaluation budget, and completion state of every study.
- [x] Add a cross-paper dataset inventory that distinguishes independent system
      channels from correlated derived training examples and states the limits
      of the current dataset-adequacy evidence.
- [ ] Reconcile both cross-paper inventories with the final selected studies,
      assets, manifests, and paper READMEs after all Sionna 2.0.1 regeneration
      work is complete.
- [x] Search for canonical dimension literals, cryptic paper-variable names,
      module-level private helpers, unregistered fixed tensors, and ignored
      `precision` or `device` handling.
- [ ] Reconcile every README with its estimator, runner, tuning record, output,
      and implementation report.
- [ ] Reconcile `docs/mimo_ofdm_chest/inventories/error_variance_inventory.md` with the final estimator
      constructors, selected defaults, calibration studies, READMEs, reports,
      and benchmark commands. No entry classified as requiring calibration may
      retain a missing scale or an unresolved required action.
- [x] After all paper loaders have stopped consuming legacy `key_fields`,
      remove that duplicate field from `build_asset_manifest()` and retain
      `compatibility`, `generation_recipe`, and optional `metadata` as the
      structured inputs.
- [x] Verify every persistent asset type is listed in
      `docs/mimo_ofdm_chest/contracts/asset_manifests.md` and has no undocumented schema version.
- [x] Review all deviations against the paper2sionna surrogate rule. Escalate
      any newly discovered algorithmic substitution or material faithfulness
      issue before changing the method.

## Phase 4: Final Regression

- [ ] Run transient smoke tests for every method and investigate non-finite,
      degenerate, or grossly inconsistent results.
- [ ] Rerun full benchmarks only for methods changed in a way that can affect
      numerical outputs.
- [ ] Rerun every canonical BER/BLER benchmark after the shared metric fix that
      counts each LDPC codeword separately instead of treating all codewords in
      one resource grid as a single block.
- [ ] Regenerate leaderboard data after any canonical output changes.
- [ ] Ensure generated databases, temporary assets, logs, and probe outputs are
      ignored or removed.
- [ ] Remove all generated smoke outputs and smoke manifests after validation.
- [ ] Make small, paper-scoped commits with test and compile results recorded in
      each commit summary.
