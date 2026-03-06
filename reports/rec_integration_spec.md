# ESS_KAN REC Integration Spec

Date: 2026-03-06
Status: Draft for implementation

## 1. Objective
Integrate standalone ESS_KAN runtime into REC framework as an alternative ESS backend that is deterministic and lightweight inside GA+FIS loops.

## 2. Runtime Artifact Schema
Expected artifact directory:

- `export/model_ts.pt` (TorchScript model)
- `export/norm.json` (normalization metadata)
- `export/model_meta.json` (model metadata)
- `runtime_benchmark.json` (latency report, optional at runtime)

`norm.json` schema:
- `x_mean`: float[3] in order `[Iin_A, Temp_C, SoC_prev]`
- `x_std`: float[3]
- `y_mean`: float
- `y_std`: float

`model_meta.json` schema:
- `framework`: string (`efficient-kan`)
- `param_count`: int
- `input_dim`: int (must be 3)
- `output_dim`: int (must be 1)
- `features`: list[str]
- `target`: string (`SoC_next`)
- `hidden_dims`: list[int]
- `grid_size`: int
- `spline_order`: int
- `source_artifact_dir`: string

## 3. REC Config Keys
Proposed architecture config for each microgrid:

```yaml
architecture:
  mg1:
    ESS:
      model: ess_kan
      kan_artifact_dir: path/to/artifacts/run_xxxxxxxx_xxxxxx
      kan_inputs:
        use_temp: true
        temp_default_c: 25.0
      kan_runtime:
        clamp_soc: true
```

Existing ESS fields must remain accepted and unchanged:
- `Q, P_S_max, a, b, B, eta, SoE_0, V_n, SoE_min, SoE_max, Q_n`

## 4. Required Backend Interface
`ESS_KAN` backend must implement same callable interface used by framework:
- `update_SoE_ch(p_GL_S, p_GL, delta_t)`
- `update_SoE_dch(p_GL_S, delta_t)`
- `get_wear_cost(SoE_prev, p_S_k, delta_t)`

Additional constructor/runtime expectations:
- loads artifact once at init
- keeps internal `SoE` state
- enforces bounds `[SoE_min, SoE_max]`
- deterministic inference on CPU

## 5. Method Mapping
Mapping from REC method inputs to KAN runtime:

- KAN input vector is `[Iin_A, Temp_C, SoC_prev]`
- `SoC_prev` = current internal SoE before update
- `Temp_C` from framework sensor/channel if available, else `temp_default_c`
- `Iin_A` derived from power flow using framework battery conventions:
  - charging current positive toward battery
  - discharging current negative from battery
  - conversion must use same voltage/capacity assumptions already used in framework ESS equations

Output handling:
- KAN returns `SoC_next` in `[0,1]`
- apply optional clamp to `[SoE_min, SoE_max]`
- update internal state

## 6. Integration Adapter (Recommended)
Use `src/runtime_wrapper.py` class:
- `ESSKANRuntime.from_artifact_dir(artifact_dir)`
- `predict_next_soc(iin_a, temp_c, soc_prev) -> float`

Framework adapter should wrap this runtime and expose REC ESS interface methods.

## 7. Wear Cost Strategy
Two-stage approach:
1. keep existing wear cost formula from current framework (`get_wear_cost`) to avoid behavior regressions
2. optionally add KAN-informed degradation model later, behind feature flag

## 8. Validation Plan in REC Framework
Minimum checks before merge:
- API compatibility test: backend swap with no call-site changes
- Determinism test: repeated run equality for same inputs
- Performance test: mean `update_*` call time under GA loop budget
- Behavioral sanity:
  - SoE remains within bounds
  - no NaN/Inf
  - converges in same optimization loop counts as existing ESS backend

## 9. Real Test Definition
"Real test" for ESS_KAN integration is considered passed when all are true:
- standalone full training done (200 epochs max, early stopping allowed)
- unseen test metrics generated (`test_metrics.json`)
- runtime benchmark generated (`runtime_benchmark.json`)
- REC dry-run with `model: ess_kan` completes one full optimization scenario without fallback

## 10. Open Decisions
- exact formula/sign convention for `p_GL_S, p_GL -> Iin_A` in REC code path
- whether REC runtime should load TorchScript directly or via Python wrapper module
- fallback policy if artifact path missing/corrupt (hard fail vs fallback ESS)
