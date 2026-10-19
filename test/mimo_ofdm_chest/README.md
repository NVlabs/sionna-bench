# Test Layers

Tests are organized by ownership:

- `infrastructure/` covers shared evaluation and estimator infrastructure.
- `test_leaderboard_data.py` covers benchmark-specific leaderboard behavior.
- `builtin/` covers Sionna's built-in reference estimators.
- `papers/<paper_id>/` covers one paper implementation per directory.
- `system_variants.py` contains reusable estimator-test helpers and is not
  collected as a test module.

The benchmark-local `conftest.py` applies deterministic runtime setup to every group.

The default suite contains fast unit and analytical checks:

```bash
python -m pytest -q
```

Additional layers use these markers:

```bash
python -m pytest -q -m system_variants
python -m pytest -q -m training
python -m pytest -q -m compile
python -m pytest -q -m slow
```

The default compile checks use the stable eager backend. To additionally run
the optional CUDA/Inductor smoke test on a machine with a working CUDA
toolchain, use:

```bash
python -m pytest -q -m compile --cuda-inductor
```

`test.mimo_ofdm_chest.system_variants` provides the shared system-variant
infrastructure. A paper-specific test defines its own estimator factory rather than adding the
estimator to a central registry:

```python
def estimator_factory(resource_grid, cfg, asset_root):
    return PaperEstimator(
        resource_grid,
        cfg,
        asset_root=asset_root,
        device=cfg.device,
    )
```

Pass this factory to `build_test_system()` and evaluate it with
`run_nmse_probe()`. The probe uses the production transmitter and channel
source, skips detector/decoder work, validates the full estimator output
contract, and can evaluate an LS reference on the same observations.

Use `assert_compiled_estimator_matches_eager()` in compile-marked tests to run
the same estimator inputs eagerly and through `torch.compile` with the selected
backend and `fullgraph=False`. The default backend is `"eager"`; use
`"inductor"` only for explicit backend smoke tests. The helper compares the
complete estimator outputs and returns the eager `(h_hat, err_var)` pair for
additional method-specific checks.

Training-based tests must pass an asset root below `tmp_path`. Tests must not
write to repository-level `assets/`, `outputs/`, or `tuning/` directories. The
global autouse fixture in `conftest.py` only seeds Python, NumPy, Torch, and
Sionna for every test and restores their process-global state afterward.
Isolation of generated files is the responsibility of each test through
`tmp_path` and explicit output roots.
