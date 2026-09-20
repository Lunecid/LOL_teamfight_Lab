# q primary table — new V OOF labels (fit85 MLP lineage)

Generated: 2026-09-21T00:20:00+09:00
**Cohort tag:** `TS`
**Selected q (Q_SELECT Brier):** `logit_state`

Contract: [Q_PREDICTION_DESIGN_CONTRACT_20260920.md](Q_PREDICTION_DESIGN_CONTRACT_20260920.md)

## Q_SELECT (selection)

| Model | Brier | logloss | AUC |
|---|---:|---:|---:|
| logit_state ← selected | 0.2432 | 0.6793 | 0.5937 |
| lgbm_state (diagnostic) | 0.2441 | 0.6812 | 0.5842 |
| b_p | 0.2471 | 0.6873 | 0.5619 |
| PT | 0.2471 | 0.6874 | 0.5616 |
| constant | 0.2500 | 0.6931 | 0.5000 |

## TEST 15.16 TS (after freeze)

| Model | Brier | logloss | AUC |
|---|---:|---:|---:|
| logit_state | 0.2432 | 0.6795 | 0.5943 |
| lgbm_state | 0.2444 | 0.6818 | 0.5821 |
| b_p | 0.2474 | 0.6879 | 0.5587 |
| PT | 0.2474 | 0.6880 | 0.5585 |
| constant | 0.2500 | 0.6931 | 0.5000 |

### Primary contrast: ΔBrier(`logit_state` − PT_linear)

- estimate=-0.00422  (negative ⇒ q better)
- 95% CI=[-0.00464, -0.00382]
- τ=0.001 interpretation threshold (a priori)
- PT_linear := StandardScaler+Logistic on [p_pre, time_minutes]
- Q_CAL: LGBM early_stopping only (no post-hoc q calibrator)

## B40 (new p_pre)

| Model | Brier | AUC |
|---|---:|---:|
| logit_state | 0.2482 | 0.5521 |
| lgbm_state | 0.2493 | 0.5322 |
| constant | 0.2500 | 0.5000 |
| b_p | 0.2501 | 0.5084 |
| PT | 0.2501 | 0.5073 |

### B40 contrast: ΔBrier(`logit_state` − PT_linear)

- estimate=-0.00197
- 95% CI=[-0.00251, -0.00136]
- P(Δ>0)=0.0000
