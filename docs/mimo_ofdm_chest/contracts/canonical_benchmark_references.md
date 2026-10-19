# Canonical Benchmark Reference Policy

This document defines how canonical benchmark outputs are identified and used
when channel-estimator implementations are reviewed, corrected, or refactored.
Canonical outputs provide numerical comparison points rather than exact test
thresholds. Monte Carlo variation, numerical precision, corrected defects, and
approved algorithm changes can produce legitimate differences.

## Canonical References

A benchmark result is canonical when:

- its NPZ payload is tracked directly under
  `outputs/mimo_ofdm_chest/<paper_id>/<scenario_id>/`;
- the scenario directory contains a tracked `config.json`; and
- the corresponding estimator is recorded in the scenario-local
  `results.json`.

The scenario-local `results.json` files form the authoritative inventory of
canonical results.

## Comparison Policy

Focused analytical, system-variant, and smoke tests establish implementation
correctness before canonical numerical results are compared.

For a behavior-preserving change, the complete NMSE curve is compared with the
tracked canonical result. BER and BLER are also inspected when detector-facing
behavior may have changed. Reported uncertainty is considered where available.
Unexpected differences are investigated before the canonical reference is
replaced.

A correctness fix or approved algorithm change may legitimately alter the
result curves. The corresponding paper documentation explains the expected
difference, and the updated canonical result is committed together with the
implementation change.

Documentation, storage-layout, and metadata-only changes do not require a
benchmark rerun. Canonical benchmark outputs remain outside hyperparameter
selection, model training, and detector-variance calibration.

## Transient Results

Smoke results verify that an integration executes and produces finite,
plausible outputs. Probe results answer bounded experimental questions. Neither
is a canonical benchmark reference.

Smoke and probe outputs remain untracked, are excluded from the leaderboard,
and are written beneath designated transient or temporary output directories.
