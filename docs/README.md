# Documentation Map

This directory indexes benchmark-specific architecture contracts, policies,
and inventories. It does not contain active work plans, agent-role prompts,
or paper-specific implementation reports.

## MIMO-OFDM Channel Estimation

The documents below belong to [`mimo_ofdm_chest/`](mimo_ofdm_chest/).

Documents are divided into two categories:

- **Contracts** define the required behavior and structure of the repository.
- **Inventories** record the current implementation state and supporting
  evidence.

When an inventory conflicts with a contract, the inventory requires correction.

## Contracts

| Document | Purpose |
| :- | :- |
| [`paper_architecture.md`](mimo_ofdm_chest/contracts/paper_architecture.md) | Defines paper-module layout, dependency direction, estimator ownership, asset APIs, tuning entry points, runner behavior, and testing boundaries. |
| [`asset_manifests.md`](mimo_ofdm_chest/contracts/asset_manifests.md) | Defines optional package-local and external asset locations, manifest schemas, identifiers, validation, result linkage, and atomic persistence. |
| [`tuning_protocol.md`](mimo_ofdm_chest/contracts/tuning_protocol.md) | Defines canonical Optuna objectives, search budgets, datasets, neural training policy, parameter selection, rounding, paired validation, and detector-variance calibration. |
| [`canonical_benchmark_references.md`](mimo_ofdm_chest/contracts/canonical_benchmark_references.md) | Defines which outputs are canonical numerical references and how they are compared after implementation changes. |

## Inventories

| Document | Purpose |
| :- | :- |
| [`estimator_inventory.md`](mimo_ofdm_chest/inventories/estimator_inventory.md) | Records supported system geometries, focused validation evidence, preparation and inference boundaries, and `torch.compile` classifications. |
| [`asset_inventory.md`](mimo_ofdm_chest/inventories/asset_inventory.md) | Records canonical inference-asset identities, generation recipes, payload metadata, and validation evidence without distributing the payloads. |
| [`dataset_inventory.md`](mimo_ofdm_chest/inventories/dataset_inventory.md) | Records independent channel counts, derived training examples, validation data, and statistical samples used to generate fitted or trained assets. |
| [`error_variance_inventory.md`](mimo_ofdm_chest/inventories/error_variance_inventory.md) | Records each estimator's detector-facing error-variance expression, classification, selected scale, and calibration evidence. |

## Related Repository Documents

- [`../README.md`](../README.md) provides the project overview, benchmark
  configuration, installation instructions, and public implementation summary.
- [`../AGENTS.md`](../AGENTS.md) defines the agent workflow, review gates, and
  contract-reading requirements.
- [`../prompts/`](../prompts/) contains agent roles, task-specific instructions,
  and implementation templates.
- Paper-local `README.md` files document individual methods, deviations,
  supported geometries, assets, tuning outcomes, and benchmark commands.
- Paper-local `reports/` directories retain implementation plans and review
  evidence.

## Maintenance

Contracts change only when repository behavior or policy changes. Inventories
are updated whenever implementation state, selected assets, validation
evidence, or canonical numerical results change.

Completed migration plans and tuning work orders are not retained as durable
documentation once their rules and evidence have been incorporated into the
contracts, inventories, and paper-local records.
