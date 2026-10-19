# Sionna Bench

Sionna Bench is a platform for wireless-communications benchmarks built on
[Sionna](https://nvlabs.github.io/sionna/). Its first benchmark compares twenty
channel estimators from the academic literature on a common multi-user
MIMO-OFDM channel-estimation task. The implementations were developed with the
help of agentic AI and evaluated through a shared pipeline. The repository
includes an interactive leaderboard, reusable estimator implementations, and
tools for evaluating new methods. We plan to add more methods and benchmarks
and welcome scientifically grounded contributions via pull requests.

## Benchmark

The MIMO-OFDM benchmark reports normalized mean-square error (NMSE), bit error
rate (BER), and codeword block error rate (BLER). Corresponding Monte Carlo
batches are paired across estimators and SNR points, while training and
statistical-fitting procedures use deterministic, manifest-recorded datasets.

The complete system configuration, metric definitions, evaluation procedure,
and design qualifications are documented on the
[benchmark page](https://nvlabs.github.io/sionna/bench/benchmarks/mimo_ofdm_chest/index.html). The implemented
methods and their supported geometries are summarized in the
[estimator documentation](https://nvlabs.github.io/sionna/bench/benchmarks/mimo_ofdm_chest/estimators/index.html).

## Repository Structure

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
and covariance payloads. 

## Get Started

We recommend using a [Python virtual environment](https://docs.python.org/3/library/venv.html) and/or a
[Docker container](https://www.docker.com), although neither is required.

### Installation

Clone the repository and install Sionna Bench from its root directory:

```bash
git clone --recurse-submodules https://github.com/NVlabs/sionna-bench.git
cd sionna-bench
pip install -e .
```

Install optional dependencies only as needed:

```bash
pip install -e ".[test]"    # Tests
pip install -e ".[tuning]"  # Hyperparameter tuning
pip install -e ".[docs]"    # Documentation build
```

Confirm that the active environment imports Sionna Bench:

```bash
python - <<'PY'
from importlib.metadata import version
import sionna.bench

print(version("sionna-bench"))
PY
```

The command should print the installed Sionna Bench version.

### Local Leaderboard

The repository includes the numerical results needed to display the
leaderboard without rerunning the benchmark. From the repository root, run:

```bash
python -m http.server 8000 --directory leaderboard/mimo_ofdm_chest
```

Open <http://localhost:8000> in a browser. Stop the server with `Ctrl-C`.

### Adding a New Method

A new method can live in an external project or in a new module under
`src/sionna/bench/<benchmark_identifier>/papers`. It must implement the
applicable Sionna block interface and be passed to the shared evaluation
pipeline.

For the MIMO-OFDM channel-estimation benchmark, subclass Sionna's
`BasePilotChannelEstimator` or `BaseChannelEstimator`. The
[custom estimator](examples/custom_estimator.py) demonstrates pilot-location
least squares with scalar shrinkage and uses Sionna's configured interpolator
for full-grid reconstruction:

```python
class ShrinkageLSChannelEstimator(BasePilotChannelEstimator):
    """Apply a unit-variance Gaussian shrinkage prior to pilot LS estimates."""

    def estimate_at_pilot_locations(
        self,
        y_pilots: torch.Tensor,
        no: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return posterior pilot estimates and their error variances."""

        # Reuse the LS implementation, which only requires the pilot pattern
        # provided by BasePilotChannelEstimator.
        h_ls, ls_err_var = (
            LSChannelEstimator.estimate_at_pilot_locations(
                self,
                y_pilots,
                no,
            )
        )
        gain = torch.reciprocal(1.0 + ls_err_var)
        h_hat = gain * h_ls
        err_var = torch.clamp(gain * ls_err_var, min=0.0)
        return h_hat, err_var
```

The [example runner](examples/run_custom_benchmark.py) evaluates the estimator
and records its result:

```python
def evaluate_example(
    *,
    output_dir: Path = Path("outputs/mimo_ofdm_chest"),
    device: Optional[str] = None,
) -> dict[str, object]:
    """Evaluate the example estimator and register its canonical result."""
    overrides = {"output_dir": str(output_dir)}
    if device is not None:
        overrides["device"] = device
    cfg = replace(umi_3gpp_config, **overrides)

    constructor_arguments = {"interpolation_type": "lin_time_avg"}
    result_name = ShrinkageLSChannelEstimator.__name__
    result_dir = Path(prepare_paper_result_dir(cfg, "custom"))
    result = evaluate_channel_estimator(
        ShrinkageLSChannelEstimator,
        cfg,
        result_dir / f"{result_name}.npz",
        estimator_kwargs=constructor_arguments,
    )
    record_result(
        cfg,
        "custom",
        ShrinkageLSChannelEstimator,
        result_name=result_name,
        constructor_arguments=constructor_arguments,
    )
    return result
```

`evaluate_channel_estimator()` runs the common communication system and writes
the metric arrays. `record_result()` adds the estimator class and effective
algorithmic settings to the scenario-local `results.json`. Run the example
from the repository root:

```bash
python -m examples.run_custom_benchmark
```

Pass `--device cpu` if you do not have a CUDA-capable GPU. After recording a
result, refresh the dashboard data with:

```bash
python leaderboard/mimo_ofdm_chest/generate_data.py
```

Then serve the leaderboard as described above. We welcome pull requests for
new methods with scientific value, such as those described in a paper or
preprint; see [CONTRIBUTING.md](CONTRIBUTING.md) for submission requirements.

### Reproducing Results

Every implementation in `src/sionna/bench/mimo_ofdm_chest/papers` provides a
`run_benchmark.py` module. Its defaults select the published estimator and
asset-generation settings. Full runs are computationally expensive. The
default device is `cuda:0`; pass `--device cpu` to paper runners and
asset-preparation commands if needed. For example:

```bash
python -m sionna.bench.mimo_ofdm_chest.papers.delfeld2025.run_benchmark \
    --device cpu
```

Displaying the local leaderboard requires no GPU. Full evaluations,
statistical fitting, and neural-network training can take substantially
longer on CPU.

Methods with no persistent inference assets can be run directly. For example,
to regenerate the RKHS result:

```bash
python -m sionna.bench.mimo_ofdm_chest.papers.delfeld2025.run_benchmark
```

For implementations with multiple methods, choose one explicitly. For
example, to evaluate the low-pass comb interpolator:

```bash
python -m sionna.bench.mimo_ofdm_chest.papers.coleri2002.run_benchmark \
    --method low-pass
```

Results are written under `outputs/mimo_ofdm_chest/<paper>/<scenario>/`. The
NPZ contains the metric curves; neighboring `config.json` and `results.json`
record the system configuration and estimator settings.

For methods needing covariances, fitted statistical models, or trained network
states, generate the assets explicitly before evaluation. For example, to fit
and evaluate the spatial GMM:

```bash
python -m sionna.bench.mimo_ofdm_chest.papers.koller2022.run_benchmark \
    --prepare-assets \
    --asset-root generated-assets \
    --method spatial

python -m sionna.bench.mimo_ofdm_chest.papers.koller2022.run_benchmark \
    --asset-root generated-assets \
    --method spatial
```

The preparation command uses the paper module's benchmark-matching defaults,
writes a manifest beside the asset payloads, and validates the completed
asset. Paper READMEs document available methods and generation arguments.

To generate every required asset and regenerate all canonical results, run:

```bash
python scripts/mimo_ofdm_chest/reproduce_results.py \
    --all \
    --prepare-assets \
    --asset-root generated-assets
```

Omit `--prepare-assets` to reuse compatible existing assets. Use one or more
`--paper` options to run a subset, and `--dry-run` to inspect commands without
executing them:

```bash
python scripts/mimo_ofdm_chest/reproduce_results.py \
    --paper coleri2002 \
    --paper delfeld2025 \
    --dry-run
```

Use `--output-dir` to write results outside `outputs/` and `--device` to
override the execution device.

Paper modules with tunable hyperparameters provide separate tuning commands.
Tuning does not rewrite estimator defaults or canonical results. For example,
to run the RKHS algorithmic study and its paired validation:

```bash
python -m sionna.bench.mimo_ofdm_chest.papers.delfeld2025.tuning \
    --study rkhs \
    --output-dir tuning/mimo_ofdm_chest/delfeld2025/my-rkhs-study

python -m sionna.bench.mimo_ofdm_chest.papers.delfeld2025.tuning \
    --study rkhs-validation \
    --output-dir tuning/mimo_ofdm_chest/delfeld2025/my-rkhs-study
```

The search spaces and tuning budgets are documented in the paper-local tuning
docstring. After changing any outputs, refresh the leaderboard data:

```bash
python leaderboard/mimo_ofdm_chest/generate_data.py
```

## License

Except for the separate RKHS submodule, Sionna Bench is licensed under
Apache-2.0; see [LICENSE](LICENSE). Third-party dependency notices are
collected in [Licenses](Licenses).
The RKHS implementation is maintained in the separate `ext/delfeld2025` Git
submodule and is not covered by this license. 
