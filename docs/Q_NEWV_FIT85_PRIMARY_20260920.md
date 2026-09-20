# q primary table — new V OOF labels (fit85 MLP lineage)

Generated: 2026-09-20T09:33:55+09:00
**Selected q (Q_SELECT Brier):** `logit_state`

Contract: [Q_PREDICTION_DESIGN_CONTRACT_20260920.md](Q_PREDICTION_DESIGN_CONTRACT_20260920.md)

## Q_SELECT (selection)

| Model | Brier | logloss | AUC |
|---|---:|---:|---:|
| logit_state ← selected | 0.2339 | 0.6604 | 0.6462 |
| lgbm_state | 0.2383 | 0.6693 | 0.6271 |
| PT | 0.2388 | 0.6706 | 0.6217 |
| b_p | 0.2389 | 0.6706 | 0.6218 |
| constant | 0.2500 | 0.6932 | 0.5000 |

## TEST 15.16 T (after freeze)

| Model | Brier | logloss | AUC |
|---|---:|---:|---:|
| logit_state | 0.2355 | 0.6637 | 0.6403 |
| lgbm_state | 0.2393 | 0.6714 | 0.6210 |
| PT | 0.2397 | 0.6723 | 0.6164 |
| b_p | 0.2397 | 0.6723 | 0.6162 |
| constant | 0.2501 | 0.6933 | 0.5000 |

### Primary contrast: ΔBrier(`logit_state` − PT)

- estimate=-0.00422  (negative ⇒ q better)
- 95% CI=[-0.00516, -0.00325]
- τ=0.001 interpretation threshold (a priori)

## B40 (new p_pre)

| Model | Brier | AUC |
|---|---:|---:|
| logit_state | 0.2474 | 0.5669 |
| PT | 0.2497 | 0.5209 |
| b_p | 0.2497 | 0.5203 |
| lgbm_state | 0.2498 | 0.5275 |
| constant | 0.2500 | 0.5000 |
