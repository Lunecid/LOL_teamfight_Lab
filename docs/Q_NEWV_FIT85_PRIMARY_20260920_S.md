> **T009/T010 pipeline by-product:** the 'primary' contrast below is vs PT_linear on this fit's TEST (not the scale-split primary); union TEST is not a contract cell.

# q primary table — new V OOF labels (fit85 MLP lineage)

Generated: 2026-09-21T00:18:44+09:00
**Cohort tag:** `S`
**Selected q (Q_SELECT Brier):** `logit_state`

Contract: [Q_PREDICTION_DESIGN_CONTRACT_20260920.md](Q_PREDICTION_DESIGN_CONTRACT_20260920.md)

## Q_SELECT (selection)

| Model | Brier | logloss | AUC |
|---|---:|---:|---:|
| logit_state ← selected | 0.2462 | 0.6857 | 0.5725 |
| lgbm_state (diagnostic) | 0.2474 | 0.6879 | 0.5563 |
| b_p | 0.2492 | 0.6916 | 0.5342 |
| PT | 0.2492 | 0.6916 | 0.5341 |
| constant | 0.2501 | 0.6933 | 0.5000 |

## TEST 15.16 S (after freeze)

| Model | Brier | logloss | AUC |
|---|---:|---:|---:|
| logit_state | 0.2458 | 0.6849 | 0.5767 |
| lgbm_state | 0.2475 | 0.6880 | 0.5560 |
| b_p | 0.2495 | 0.6921 | 0.5301 |
| PT | 0.2495 | 0.6921 | 0.5301 |
| constant | 0.2500 | 0.6932 | 0.5000 |

### Primary contrast: ΔBrier(`logit_state` − PT_linear)

- estimate=-0.00368  (negative ⇒ q better)
- 95% CI=[-0.00412, -0.00322]
- τ=0.001 interpretation threshold (a priori)
- PT_linear := StandardScaler+Logistic on [p_pre, time_minutes]
- Q_CAL: LGBM early_stopping only (no post-hoc q calibrator)

## B40 (new p_pre)

| Model | Brier | AUC |
|---|---:|---:|
| logit_state | 0.2481 | 0.5530 |
| lgbm_state | 0.2496 | 0.5281 |
| constant | 0.2501 | 0.5000 |
| b_p | 0.2502 | 0.5042 |
| PT | 0.2502 | 0.5038 |

### B40 contrast: ΔBrier(`logit_state` − PT_linear)

- estimate=-0.00213
- 95% CI=[-0.00266, -0.00153]
- P(Δ>0)=0.0000
