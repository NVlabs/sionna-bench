# Benchmarking Wireless Channel Estimators at Agentic Speed


In wireless communications research, there is no established practice of comparing the performance of different methods on common benchmarks.
The absence of benchmarks makes it difficult to establish the "state of the art" on any given problem.
Moreover, methods that address essentially the same task are derived for system models
that differ from each other multiple regards, such as what idealizations are made, what aspects are neglected, what conditions are assumed, etc.

For instance, when considering channel estimation, system models differ in whether or not they assume frequency-selectivity,
whether or not they consider mobility, how they model propagation, whether or not interpolation is considered,
what type of side information is available, what the geometry and characteristics of the antenna/arrays are,
how hardware non-idealities are modeled, and whether error estimates are provided.
On a surface level, it is therefore impossible to directly compare the performance of different methods.

This repository presents the first comprehensive benchmarking of wireless channel estimators for MIMO-OFDM systems.
All benchmarked methods are implemented and evaluated using Sionna.

## Methodology and Aspirations

We have selected a representative set of channel estimation methods proposed in the academic literature.
The selected methods have been reimplemented and adapted to a common contract with the help of agentic AI.
These necessary adaptations mean that we make _no_ claim of exact reproduction, neither of the method itself nor of the results presented in the corresponding paper.
Instead, our objective has been to adapt, extend, or restrict the described methods in a way that was faithful to the underlying approach and ideas.
Moreover, while we did our best to check, verify, and optimize the agentic-driven reimplementations, mistakes are always possible.

We emphasize that a "bad" performance of any given method does not at all imply any failure on part of the authors of the corresponding paper:
Our benchmark compares the raw performance of different methods without accommodating the fact that some of these methods are more computationally efficient than others,
or are much more robust to changes in distribution, etc. Moreover, the design choices of our benchmark configuration may also disadvantage certain methods, e.g.,
because it is not sufficiently "massive MIMO", or sufficiently line-of-sight, etc. Finally, it is always possible that our adaptation of the paper's method to
our benchmark contract has been done in a suboptimal way, or that our choice of hyperparameters or learning procedure was not sufficiently optimized,
or that some other mistake has been made.


## Benchmark

The canonical benchmark evaluates channel estimators in a common multi-user
MIMO-OFDM uplink and reports NMSE, BER, and codeword BLER. Corresponding Monte
Carlo batches are paired across estimators and SNR points, while training and
statistical-fitting procedures use deterministic, manifest-recorded datasets.

The complete system configuration, metric definitions, evaluation procedure,
and design qualifications are documented on the
[benchmark page](doc/benchmarks/mimo_ofdm_chest/index.rst). The implemented methods and their
supported geometries are summarized in the
[estimator documentation](doc/benchmarks/mimo_ofdm_chest/estimators/index.rst) and the
[validation and inference inventory](docs/mimo_ofdm_chest/inventories/estimator_inventory.md).

## About the Structure of this Repository

```text
sionna-bench/
|-- src/sionna/bench/              Installable Python package
|   `-- mimo_ofdm_chest/           MIMO-OFDM channel-estimation benchmark
|       |-- benchmarking/          Communication system and evaluation tools
|       |-- configs/               System configurations
|       |-- papers/<paper_id>/     Paper implementations and runners
|       |   |-- assets/            Optional package-local inference assets
|       |   |-- reports/           Implementation and review reports
|       |   |-- estimator.py
|       |   |-- asset.py           Optional asset generation and loading
|       |   |-- tuning.py          Optional hyperparameter tuning
|       |   `-- run_benchmark.py
|       `-- asset_store.py         Asset paths and manifests
|-- ext/delfeld2025/              Separate RKHS Git submodule
|-- test/mimo_ofdm_chest/          Infrastructure and paper-specific tests
|-- docs/mimo_ofdm_chest/
|   |-- contracts/                 Channel-estimation implementation contracts
|   `-- inventories/               Assets, datasets, estimators, and evidence
|-- doc/                           Sphinx documentation website
|-- prompts/                       Agent roles, tasks, and code templates
|-- outputs/mimo_ofdm_chest/       Canonical channel-estimation results
|-- tuning/mimo_ofdm_chest/        Compact channel-estimation tuning evidence
|-- scripts/mimo_ofdm_chest/       Benchmark reproduction helper
`-- leaderboard/mimo_ofdm_chest/  Standalone browser dashboard
```

Public source and wheel distributions omit trained models, fitted statistics,
and covariance payloads. Complete package-local assets may be present in
internal checkouts, while `outputs/` and `tuning/` are repository evidence. The
[documentation map](docs/README.md) links to the detailed architecture, asset,
tuning, validation, and reproducibility contracts.

## Quick Start

The commands below support macOS and Linux and assume the repository root is
the current working directory. The reference environment uses Python 3.12 and
Torch 2.11.0 with CUDA 12.6. Select a Torch build appropriate for the local CPU
or CUDA installation.

Install [uv](https://docs.astral.sh/uv/getting-started/installation/). Its
standalone installer does not require an existing Python installation; users
who already have `uv` can skip these two commands:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
```

Create and activate an isolated Python 3.12 environment. `uv` downloads a
compatible interpreter automatically when necessary:

```bash
uv venv --python 3.12
source .venv/bin/activate
```

Install Sionna Bench and its test dependencies from the repository root. This
also installs Sionna 2.1 and its Sionna RT dependency from PyPI:

```bash
uv pip install -e ".[test]"
```

Estimators are available from their paper modules and, for the primary public
classes, directly from the package root:

```python
from sionna.bench import LSCombEstimator
from sionna.bench.mimo_ofdm_chest.papers.koller2022 import SpatialGMMChannelEstimator
```

A CUDA-capable GPU is optional. The canonical configuration selects
`cuda:0`, so CPU-only users must pass `--device cpu` to paper benchmark and
asset-preparation commands. The repository-wide reproduction helper accepts
the same option and forwards it to every runner. Full evaluation and asset
generation can be substantially slower on CPU.

Hyperparameter tuning and the optional Lyu CVXPY reference tests require
additional extras. Install only the Lyu test oracle with
`uv pip install -e ".[test-lyu]"`, or install all currently defined
optional dependencies with:

```bash
uv pip install -e ".[all]"
```

Run the shared infrastructure and built-in estimator tests as a quick local
check:

```bash
python -m pytest -q test/mimo_ofdm_chest/infrastructure test/mimo_ofdm_chest/builtin
```

Run an asset-free smoke evaluation of Sionna's linear LS estimator:

```bash
python -m sionna.bench.mimo_ofdm_chest.papers.builtin.run_benchmark --smoke --estimator ls-lin --device cpu
```

Covariance, GMM, and trained neural inference assets are not included in the
public distribution. Their selected identities, generation recipes, and
validation evidence are recorded in
[`docs/mimo_ofdm_chest/inventories/asset_inventory.md`](docs/mimo_ofdm_chest/inventories/asset_inventory.md)
and the paper-local READMEs. Generate a required asset explicitly with the
paper runner's `--prepare-assets --asset-root <path>` options, then pass the
same `--asset-root <path>` to the benchmark run. Estimators themselves never
perform asset I/O, and missing assets are never downloaded or generated
implicitly.

Generate the dashboard data from the available canonical outputs and serve the
static leaderboard:

```bash
python leaderboard/mimo_ofdm_chest/generate_data.py
python -m http.server 8000 --directory leaderboard/mimo_ofdm_chest
```

Then open <http://localhost:8000>. The dashboard loads Chart.js 4.5.1 from
jsDelivr and does not require an npm installation.

## License

Except for the separate RKHS submodule, Sionna Bench is licensed under
Apache-2.0; see [LICENSE](LICENSE). Third-party dependency notices are
collected in [Licenses](Licenses).

The RKHS implementation is maintained in the separate `ext/delfeld2025` Git
submodule and is not covered by this license. Its release and license are under
separate review. The Apache-only package configuration excludes its symlinked
Python module from package discovery.
