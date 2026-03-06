# ESS_KAN Standalone Contract (Draft)

## Goal
Train and validate a KAN-based ESS model standalone, then export a fast runtime artifact for REC integration.

## Primary task
One-step SoC estimation.

## Inputs (per timestep)
- Iin_A: battery current [A]
- Temp_C: battery temperature [degC] (optional if missing in data)
- SoC_prev: previous SoC [0..1]

## Output
- SoC_next: next SoC [0..1]

## Data source
- RW train/test MAT files, same format used in ENNC repo:
  - train keys: tr_Time, tr_Iin, tr_SoC, Temp, Ts, Cn
  - test keys: ts_Time, ts_Iin, ts_SoC, Temp, Ts, Cn

## Split policy
- Train/val split from train set only (70/30)
- Test set kept fully unseen

## Metrics
- RMSE, MAE, R2 on test set
- Residual summary by SoC bins

## Runtime target (for later REC use)
- Sub-millisecond to a few milliseconds per step on CPU
- If full KAN misses target, export compact runtime surrogate

## Deliverables
- trained model artifact
- normalization metadata
- test metrics JSON
- runtime benchmark JSON
- export artifact for integration (format TBD)

## REC Framework Compatibility (Must-Have)

- Selectable as ESS backend via: architecture.mgX.ESS.model
- Must accept existing ESS config fields:
  Q, P_S_max, a, b, B, eta, SoE_0, V_n, SoE_min, SoE_max, Q_n
- Must implement methods:
  update_SoE_ch(p_GL_S, p_GL, delta_t)
  update_SoE_dch(p_GL_S, delta_t)
  get_wear_cost(SoE_prev, p_S_k, delta_t)
- Must keep internal SoE state and enforce SoE bounds
- Must be deterministic and lightweight for GA/FIS loops
- Must load model/export artifact from config path (relative path supported)

