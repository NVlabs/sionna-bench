# Paper Module Architecture

This document defines the default code structure and dependency contract for
paper implementations. It applies to new papers and to existing papers as they
are reviewed. Existing implementations may temporarily differ while their
paper-specific migration remains unfinished.

## Default Layout

The default paper-local Python modules are:

```text
src/sionna/bench/mimo_ofdm_chest/papers/<paper_id>/
  estimator.py
  asset.py          # optional
  tuning.py         # optional
  run_benchmark.py
```

`README.md`, `paper.pdf`, and `reports/` remain paper-local source material but
are excluded from the wheel. Tests remain under the repository-level `test/`
tree, and MIMO-OFDM channel-estimation results remain under
`outputs/mimo_ofdm_chest/`.

Additional Python modules must have a concrete justification. For example, a
substantial independent reference solver may justify its own module. A small
helper or hypothetical future functionality does not justify another module.

## Dependency Direction

The benchmark runner orchestrates estimator construction and asset
preparation:

```text
run_benchmark.py -> asset.py
run_benchmark.py -> estimator.py
asset.py         -> estimator.py  # when training needs model definitions
estimator.py     -> no asset.py import
```

`estimator.py` must not import `asset.py`, and function-local imports must not
hide circular dependencies. Imports needed only to resolve type annotations may
be placed under `TYPE_CHECKING`. This exception does not permit `estimator.py`
to depend on runtime behavior owned by `asset.py`.

The runner obtains inference-ready tensors, parameters, or state dictionaries
from `asset.py` and passes them to the estimator constructor. This keeps the
estimator usable independently of this repository's directory layout and
prevents construction from unexpectedly performing file I/O or training.

## `estimator.py`

`estimator.py` contains:

- Sionna channel estimator and interpolator classes;
- neural-network architectures used during estimator inference;
- online numerical solvers that are part of the paper's estimation algorithm;
- constructor validation for algorithmic choices and inference inputs; and
- the estimator's forward or `call()` implementation.

An estimator constructor may accept:

- the Sionna resource grid and other Sionna interface objects;
- algorithmic hyperparameters;
- prepared covariance matrices, mixture parameters, model states, or models;
- `precision` and `device`; and
- system information genuinely required by the algorithm and unavailable from
  the Sionna interface.

It should not accept repository concerns such as `asset_root`, training batch
counts, training epochs, cache identities, or fallback asset-generation
policies. File discovery, manifest validation, and missing-model training must
occur outside the estimator.

Fixed inference tensors are registered as buffers. Prepared model states are
loaded during construction, before compiled inference begins.

Every public estimator and interpolator constructor argument is annotated.
Interface objects use the applicable Sionna types, such as ``ResourceGrid``,
``PilotPattern``, and ``Precision``. Nullable arguments use ``Optional[T]``,
while ``Union[A, B]`` is reserved for constructors that genuinely accept both
alternatives. Meaningfully optional controls have defaults. For controls
covered by the canonical tuning or detector-calibration protocol, the default
equals the rounded, independently validated value selected under
`tuning_protocol.md`. Other defaults follow their documented paper or
implementation rationale. Required prepared parameters do not have defaults.
For tensor or array inputs, the Sionna-style docstring states the expected
shape and dtype and explains any real/complex stacking or axis convention
that is not self-evident.

For configurable estimator methods, each machine-oriented method value is a
module-level ``METHOD_<NAME>`` constant, and the values are collected in an
ordered ``METHOD_NAMES`` list. Presentation-only names remain separate, for
example in a runner-local ``METHOD_LABELS`` mapping.

Pilot-domain estimators that inherit from Sionna's
``BasePilotChannelEstimator`` expose its standard ``interpolation_type`` and
optional ``interpolator`` constructor arguments. Paper estimators using this
base class default to ``interpolation_type="lin_time_avg"`` so their off-pilot
reconstruction follows one common policy; a supplied custom interpolator still
takes precedence.
Estimators that implement full-grid ``call()`` directly inherit Sionna's
``BaseChannelEstimator`` instead. An estimator whose paper algorithm is itself
the interpolation method may expose that paper-specific interpolator directly.

Paper implementations may import Sionna's private ``_PilotGroupMixin`` from
``sionna.phy.ofdm.channel_estimation`` when structured pilot access is needed.
The leading underscore is intentional: the mixin is a developer utility, not
a public component that users are expected to configure independently.

Method terminology is consistent throughout the implementation: the
constructor argument is ``method``, the stored private attribute is
``self._method``, and the runner option is ``--method``. The term ``variant``
does not represent this shared concept.

### Compiled Inference

Online estimator inference supports `torch.compile`. Tier B is the default
minimum classification, while implementations naturally capable of full-graph
execution use Tier A. Tier C is reserved for algorithmically unavoidable
regions such as observation-specific optimization that requires autograd,
SciPy, or another unsupported runtime.

- **Tier A:** the complete online inference core compiles as a full graph.
- **Tier B:** the complete estimator executes through `torch.compile` with a
  small documented compiler-disabled boundary or graph break.
- **Tier C:** an unavoidable region remains outside compilation while the
  tensor-only inference subroutine remains compilable.

Asset discovery, file I/O, dataset generation, offline fitting, and offline
training remain outside compiled inference.

Every estimator has a focused test comparing eager execution with
`torch.compile(..., backend="eager", fullgraph=False)` using
precision-appropriate tolerances. CUDA/Inductor execution is an optional local
smoke check rather than a default unit-test requirement.

`torch._dynamo.explain` provides diagnostic evidence about graph breaks.
Tests do not assert an exact graph-break count. Clear tensor code is not
replaced by obscure implementations solely to improve the compilation tier
unless the measured benefit is material.

## `asset.py`

`asset.py` contains the complete lifecycle of persistent paper assets:

- canonical asset-path selection through `sionna.bench.mimo_ofdm_chest.asset_store`;
- manifest construction and validation;
- loading and saving payloads;
- training-data or channel-sample generation;
- offline training, fitting, or statistical estimation; and
- conversion of persisted payloads into inference-ready values.

Complete package-local inference assets, when included in an internal or
private distribution, live beneath the owning paper package as
`assets/<scenario_id>/<asset_type>/<artifact_id>/`. Public distributions may
omit these payloads. An explicitly supplied external asset root uses
`<asset_root>/<scenario_id>/<paper_id>/<asset_type>/<artifact_id>/` and is the
only writable location for regeneration, training, and tests. This policy
prevents accidental writes into an installed package and accidental use of
preliminary or outdated assets for benchmarking. Installed package assets, if
present, are read-only inputs.

Paper asset functions import model definitions from `estimator.py` when
training requires them. Their return values are suitable for direct injection
into an estimator, such as a covariance tensor, GMM parameters, or one or more
model state dictionaries.

Public loading and generation handles are separate:

```python
load_model(...)
train_model(...)          # learned neural model
fit_model(...)            # fitted statistical model, such as a GMM
load_covariance(...)
estimate_covariance(...)  # Monte Carlo or analytical covariance
```

When one paper exposes multiple asset-backed methods with different canonical
generation recipes, each method has a qualified public handle, such as
`train_model2()` and `train_model3()`. This keeps the benchmark-matching values
visible and unambiguous in each function signature; common implementation code
remains private.

Loading never generates or modifies an asset. Generation functions require an
explicit external asset root, write the payload and manifest, and then call the
strict loader so newly generated and package-local assets follow the same
validation path. If the selected asset is absent, loading raises an actionable
`FileNotFoundError` that shows both the external preparation command and the
matching evaluation command. Both operations return the inference-ready
payload and the selected ``manifest.json`` path. Serialization, manifest
assembly, dataset batching, and training-loop helpers normally remain private.

Loading and generation are not combined in a ``load_or_*`` function, and the
descriptive handles are not replaced by a generic ``generate_asset()``:
training, statistical fitting, and covariance estimation have materially
different behavior. Generation refuses an occupied destination unless the
caller explicitly requests replacement with ``overwrite=True``.

## `tuning.py`

`tuning.py` contains paper-specific hyperparameter search and exposes:

```python
run_tuning(...)
```

`run_tuning()` uses
`sionna.bench.mimo_ofdm_chest.benchmarking.hyperparameter_tuning.tune_channel_estimator()` for
the shared Optuna study lifecycle and NMSE or detector-facing probe evaluation
rather than implementing a paper-local tuning loop. The generated `best_params.json`
records the sampler seed, effective probe seed, requested and total trial
counts, and measured optimization runtime alongside the objective and selected
parameters; paper-local code does not duplicate this metadata assembly.

`build_cached_detector_scale_objective()` should be used when a detector-facing
scalar provably only rescales a frozen error-variance tensor. It caches each
channel estimate and unscaled error variance once while retaining the shared
Optuna lifecycle in `tune_channel_estimator()`. The same helper provides the
independent paired validation so candidate scales share estimates and observations. The
cached boundary is documented, and caching does not cross a parameter that can
change channel estimates, trained weights, or observations.

Optuna may search continuous parameters at full floating-point precision, but
canonical constructor values do not imply unsupported numerical precision.
Every selected continuous parameter is rounded to at most three significant
digits, while the study summary and trial history retain the raw Optuna winner.
The rounded candidate is evaluated with the same deterministic independent
validation protocol used to approve the raw winner. If rounding
causes a material regression, a nearby three-significant-digit value is
selected and validated instead of retaining additional digits. Integer and
categorical parameters are unchanged. Paper READMEs and reports distinguish
the raw Optuna winner from the rounded value used by the runner and canonical
outputs.

Optuna objectives and parameter-suggestion functions normally remain private,
for example `_objective()` and `_suggest_parameters()`. Tuning may invoke asset
training when a trial changes learned parameters, while final benchmark results
remain outside the tuning objective.

When a trial fits statistics or trains weights from sampled channels, it uses
a fixed probe seed distinct from the asset-generation seed. This gives trials
common deterministic probe channels without evaluating on their training
stream. Probe RNGs are reset after constructing the estimator and evaluation
system: constructor initialization may consume a hyperparameter-dependent
number of random draws, but it does not change subsequent probe channels.
Asset generators use ``sionna.bench.mimo_ofdm_chest.benchmarking.set_random_seed()`` so the
recipe's seed controls Sionna's explicit generators as well as Python, NumPy,
and PyTorch/CUDA.

## `run_benchmark.py`

Every runner exposes:

```python
run_benchmark(...)
main()
```

`main()` delegates command-line parsing to a private `_parse_args()` helper and
construction of the CLI-overridden system configuration to a private
`_system_config_from_args()` helper. This keeps CLI-specific behavior separate
from the programmatic `run_benchmark()` interface.

`run_benchmark()`:

1. selects the system configuration and algorithmic method;
2. asks `asset.py` for any required inference assets;
3. passes those assets and algorithmic arguments to the estimator;
4. invokes the shared evaluation functions;
5. writes the result under the canonical display name; and
6. records the completed payload in the scenario-local `results.json`,
   including references to every selected persistent asset.

``sionna.bench.mimo_ofdm_chest.benchmarking.result_store.record_result()`` is called after the
NPZ payload has been written. It receives the concrete estimator class,
persistable constructor arguments, and the paths of all selected asset
manifests. The result store validates each manifest and records its asset type,
artifact ID, and exact manifest checksum without duplicating compatibility
tensors or model-state metadata.

Estimator constructor defaults are the single source of truth for canonical
inference settings. Runner CLI options for optional estimator arguments use
``None`` to mean "use the constructor default" and pass only explicit
overrides to estimator construction. Before calling ``record_result()``, the
runner uses ``resolve_constructor_arguments()`` with an explicit list of
behavior-relevant argument names to record the effective values. Constructor
defaults are not mirrored as runner constants or imported into tuning or
tests, and prepared tensors and asset-generation settings are not persisted as
constructor arguments.

Tuning follows the same rule: fixed estimator kwargs omit default-valued
arguments, and paired validation reads an incumbent value from the estimator
constructor when it needs to compare against that default. Asset training and
fitting defaults likewise live in the public function signature; tuning reads
them with ``resolve_callable_arguments()`` instead of duplicating a current
training recipe. Tests either cover the omitted-default path or pass an
explicit test-specific override.

Asset-backed runners expose ``--prepare-assets`` for persistent generation
into an explicitly supplied external ``--asset-root``. This mode prepares the
selected asset and exits without evaluating or writing benchmark results.
Normal canonical execution uses strict ``load_covariance()``, ``load_model()``,
or equivalent handles and never trains or estimates a missing persistent
asset. An asset-backed smoke check uses two explicit commands: first
``--prepare-assets`` with a cheap recipe and external root, then the normal
strict benchmark command against that same root. The benchmark function itself
does not combine loading with generation.

Estimator evaluation uses
`sionna.bench.mimo_ofdm_chest.benchmarking.channel_estimation.evaluate_channel_estimator()`
rather than a paper-local evaluation loop. Methods such as perfect CSI that do
not construct a channel estimator may use the corresponding shared system-level
evaluation entry point.

For one estimator class, the display name is its class name. Configurable
methods use `EstimatorClass (method label)`. The same display name is used as
the NPZ filename stem, returned result-dictionary key, and result-manifest
entry. Machine-oriented CLI tokens remain separate from display labels.

## Examples

A learned estimator is loaded outside the estimator class for normal
evaluation:

```python
# run_benchmark.py
model_states, noise_variances, manifest_paths = load_model(
    resource_grid,
    cfg,
    **asset_arguments,
)
constructor_arguments = {}
result = evaluate_channel_estimator(
    DNNChannelEstimator,
    cfg,
    output_path,
    estimator_kwargs={
        "model_states": model_states,
        "model_noise_variances": noise_variances,
        **constructor_arguments,
    },
)
effective_arguments = resolve_constructor_arguments(
    DNNChannelEstimator,
    ("err_var_scale",),
    constructor_arguments,
)
record_result(
    cfg,
    PAPER_SLUG,
    DNNChannelEstimator,
    constructor_arguments=effective_arguments,
    asset_manifests=manifest_paths,
)
```

The runner's separate `--prepare-assets` branch calls `train_model(...)` with
an explicit external asset root and exits before reaching
`evaluate_channel_estimator()`.

Likewise, a covariance-based estimator strictly loads the selected statistics:

```python
covariance, manifest_path = load_covariance(
    resource_grid,
    cfg,
    **asset_arguments,
)
result = evaluate_channel_estimator(
    TruncatedChannelEstimator,
    cfg,
    output_path,
    estimator_kwargs={"covariance": covariance},
)
```

Its explicit preparation branch uses `estimate_covariance(...)` with an
external asset root and does not run the benchmark.

The manifest path is passed to `record_result()` for result-to-asset linkage;
the estimator itself does not consume repository metadata.

## Exceptions

Optimization belongs in `estimator.py` when it is the online estimation
algorithm rather than offline asset preparation. Balevi et al.'s untrained
network fitting and Lyu et al.'s iterative sparse/diffuse solver are examples.
These methods do not become `asset.py` training merely because they perform
optimization.

Every deviation from the default dependency direction or module layout must
be justified in the implementation plan and recorded in the paper's
implementation report.

## Testing Consequences

Estimator tests inject small synthetic or temporary inference assets directly
and do not depend on canonical files. Asset tests use an explicit temporary
asset root and separately verify loading, malformed payload rejection, and
small generation or training paths. Runner smoke tests verify the integration
between those two independently testable layers.
