# Contributing to Sionna Bench

We accept contributions of new paper-derived channel estimators only. A pull
request must add one new directory at `src/sionna/bench/<benchmark_identifier>/papers/<identifier>/`
and must not change files elsewhere in the repository. The directory name
must be a stable identifier for the paper or method.

## Permitted Files

The new directory must contain:

- `README.md`: identify the source paper, explain the implemented method,
  supported system geometry, material adaptations, and how to run the estimator.
- `estimator.py`: implement the channel estimator.
- `run_benchmark.py`: integrate the estimator with the shared benchmark.
- `__init__.py`: export the public estimator class or classes.

It may also contain:

- `asset.py`: asset loading and reproducible generation, if needed.
- `tuning.py`: a reproducible hyperparameter study, if needed.
- `reports/`: Markdown reports about the implementation only.

No other files may be included in the pull request. In particular, do not
submit trained or fitted weights, generated statistics, datasets, study
databases, benchmark results, plots, or copies of the paper.

Submitted source code must be original work created by you or your employer,
and you must have authority to contribute it under Apache-2.0. Each Python
file must identify its actual copyright holder and include
`SPDX-License-Identifier: Apache-2.0`.

## Module Interfaces

`estimator.py` defines a Sionna-compatible channel estimator.
Constructors receive any prepared inference parameters; they
must not load files or start training.

`run_benchmark.py` exposes `run_benchmark()` and `main()`. It uses the shared
benchmark evaluator, writes results through the repository's result-store
interface, and records the effective constructor arguments. If assets are
needed, normal evaluation loads them strictly; a separate
`--prepare-assets` mode generates them and exits without evaluating.

`asset.py`, when present, exposes separate loading and generation functions
such as `load_model()` and `train_model()`, `fit_model()`, or
`estimate_covariance()`. Loading never generates assets. Generation requires
an explicit external asset root and records a manifest. The loader returns
inference-ready parameters and the selected manifest path.

`tuning.py`, when present, exposes `run_tuning()` and uses the repository's
shared tuning infrastructure. Describe the search space and the generation
recipe, but do not include generated study artifacts in the pull request.

`__init__.py` exports the public estimator classes from `estimator.py`.
If a contribution is accepted, maintainers will handle generation and publication
of benchmark results.

#### Signing Off Your Work

* We require that all contributors "sign-off" on their commits. This certifies that the contribution is your original work, or you have rights to submit it under the same license, or a compatible license.

  * Any contribution which contains commits that are not Signed-Off will not be accepted.

* To sign off on a commit you simply use the `--signoff` (or `-s`) option when committing your changes:
  ```bash
  $ git commit -s -m "Add cool feature."
  ```
  This will append the following to your commit message:
  ```
  Signed-off-by: Your Name <your@email.com>
  ```

* Full text of the DCO (https://developercertificate.org/):

  ```
    Developer Certificate of Origin
    Version 1.1

    Copyright (C) 2004, 2006 The Linux Foundation and its contributors.

    Everyone is permitted to copy and distribute verbatim copies of this
    license document, but changing it is not allowed.


    Developer's Certificate of Origin 1.1

    By making a contribution to this project, I certify that:

    (a) The contribution was created in whole or in part by me and I
        have the right to submit it under the open source license
        indicated in the file; or

    (b) The contribution is based upon previous work that, to the best
        of my knowledge, is covered under an appropriate open source
        license and I have the right under that license to submit that
        work with modifications, whether created in whole or in part
        by me, under the same open source license (unless I am
        permitted to submit under a different license), as indicated
        in the file; or

    (c) The contribution was provided directly to me by some other
        person who certified (a), (b) or (c) and I have not modified
        it.

    (d) I understand and agree that this project and the contribution
        are public and that a record of the contribution (including all
        personal information I submit with it, including my sign-off) is
        maintained indefinitely and may be redistributed consistent with
        this project or the open source license(s) involved.
  ```