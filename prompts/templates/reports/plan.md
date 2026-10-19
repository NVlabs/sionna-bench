# Implementation Plan

- Paper ID:
- Full citation:
- Task:
- Implementer:
- Date:

## Applicable Contracts

- `<contract>`: <applicable requirements>

## Harness Summary

- Benchmark config:
- Smoke config:
- Built-in baselines:
- Task base class or template:
- Benchmark input contract:
- Benchmark output contract:
- Metrics:
- Output layout:

## Paper Method Inventory

List every method or variant relevant to the task.

### Method: <name>

- Paper location:
- Paper equations / algorithms / statements:
- Paper inputs:
- Paper outputs:
- Required side information:
- Target repository abstraction:
- Status: `implement` | `not implemented`

#### Faithful Implementation Detail

Write near-pseudocode for the intended implementation. Cite paper equations or
statements next to each major operation.

```text
1. ...
2. ...
3. ...
```

#### Interface Adaptations

- Required shape, batching, dtype, device, or Sionna API adaptations:
- Why these adaptations preserve the paper algorithm:

#### Hyperparameter Selection

- Preliminary-probe configurations and budget:
- Preliminary-probe seed:
- Runtime-estimation method:
- Full-study approval required:

| Parameter | Classification | Search space or fixed source | Selection procedure |
| :- | :- | :- | :- |

#### Surrogate Check

- Does the plan replace any paper objective/model/solver/training/inference
  step?
- If yes, change status to `not implemented`.

#### Asset Plan

| Asset | Independent training/statistical channels | Validation channels | Derived examples | SNR coverage | Generation entry point |
| :- | -: | -: | -: | :- | :- |

- Proposed external path and optional package-local path:
- Generation, training, fitting, or estimation seeds:
- Payload files, shapes, and dtypes:
- Expected serialized size:
- Compatibility and validation checks:
- Estimated runtime:
- Required hardware:
- Future-agent resume steps:

#### Risks

- Missing paper details:
- Numerical risks:
- Faithfulness risks:
- Benchmarking fairness risks:

#### If `not implemented`

- Reason the method is not implemented in this pass:
- Required asset, method detail, infrastructure, or benchmark-contract change:
- Why implementing it under the current conditions would be misleading, if
  applicable:

## Cross-Cutting Decisions

- Shared assumptions:
- Shared helper modules:
- Planned smoke command:
- Required sanity checks:

## Questions For Reviewer

- Question:
