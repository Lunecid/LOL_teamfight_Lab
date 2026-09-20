# q primary table — new V OOF labels (fit85 MLP lineage)

Generated: 2026-09-20T09:33:55+09:00  
**Selected q (Q_SELECT Brier):** `logit_state`  
**Primary sample:** 15.16 teamfight T — \(n=32{,}981\) / \(24{,}020\) matches (not the 210k corpus)

Contract: [Q_PREDICTION_DESIGN_CONTRACT_20260920.md](Q_PREDICTION_DESIGN_CONTRACT_20260920.md)  
Scope lock: [Q_RESULT_SCOPE_LOCK_20260920.md](Q_RESULT_SCOPE_LOCK_20260920.md)  
Public metrics: [Q_NEWV_FIT85_PRIMARY_TABLE_20260920.json](Q_NEWV_FIT85_PRIMARY_TABLE_20260920.json)

## Claim scope (this run)

Supported wording:

> Pre-fight state features beat a **linear** \(p_{\mathrm{pre}}+\)time baseline (`PT_linear`) at predicting the **direction** of new-\(V\) engagement ΔV on the **15.16** holdout.

Not supported by this table alone:

- “q fully controls flexible \(p_{\mathrm{pre}}\)/time information”
- “the lift transfers to major 16.x external cohorts” (see transfer doc: it does **not**)
- “Q_CAL probability calibration of q was completed” (this run: **no** post-hoc calibrator)

## Q_SELECT (selection)

| Model | Brier | logloss | AUC |
|---|---:|---:|---:|
| logit_state ← selected | 0.2339 | 0.6604 | 0.6462 |
| lgbm_state | 0.2383 | 0.6693 | 0.6271 |
| PT_linear (`PT`) | 0.2388 | 0.6706 | 0.6217 |
| b_p | 0.2389 | 0.6706 | 0.6218 |
| constant | 0.2500 | 0.6932 | 0.5000 |

## TEST 15.16 T (after freeze)

| Model | Brier | logloss | AUC |
|---|---:|---:|---:|
| logit_state | 0.2355 | 0.6637 | 0.6403 |
| lgbm_state | 0.2393 | 0.6714 | 0.6210 |
| PT_linear (`PT`) | 0.2397 | 0.6723 | 0.6164 |
| b_p | 0.2397 | 0.6723 | 0.6162 |
| constant | 0.2501 | 0.6933 | 0.5000 |

### Primary contrast: ΔBrier(`logit_state` − PT_linear)

- estimate=−0.00422  (negative ⇒ q better)
- 95% CI=[−0.00516, −0.00325]
- τ=0.001 interpretation threshold (a priori)
- ≈1.76% relative Brier reduction vs PT_linear; ΔAUC ≈ +0.0239

AUC 0.6403 is **ranking quality**, not “64% of fights correctly signed.”

## B40 (new p_pre; \(n=5{,}423\) / \(4{,}945\) matches)

| Model | Brier | AUC |
|---|---:|---:|
| logit_state | 0.2474 | 0.5669 |
| PT_linear (`PT`) | 0.2497 | 0.5209 |
| b_p | 0.2497 | 0.5203 |
| lgbm_state | 0.2498 | 0.5275 |
| constant | 0.2500 | 0.5000 |

### B40 contrast: ΔBrier(`logit_state` − PT_linear)

Already computed in `primary_table.json` / public JSON copy:

- estimate=−0.00225
- 95% CI=[−0.00440, −0.00012]
- \(P(\Delta>0)=0.016\) under match-clustered bootstrap

Point estimate and CI both favor q on this **new-V** B40 cell (not comparable row-wise to old-V B40).

## Implementation notes (affect interpretation)

| Item | This run |
|---|---|
| **PT** | `StandardScaler` + `LogisticRegression` on `[p_pre, time_minutes]` → **`PT_linear`**. No spline / interaction. |
| **Q_CAL** | Used for LightGBM **early stopping** only. **No** separate probability calibrator for logit / PT / \(b(p)\). |
| **logit vs LGBM inputs** | logit: 351 numeric + \(p_{\mathrm{pre}}\) (352). LGBM: +10 champion IDs (362). Selection = this feature/setup mix, not pure learner duel. |
| **Labels** | TRAIN = match-fold OOF MLP; Q_CAL / Q_SELECT / TEST = frozen fit85. |

## Artifact digests (local `outputs/`; JSON copies under `docs/`)

| File | SHA256-16 |
|---|---|
| `primary_table.json` | `a80340fede7b6e6a` |
| `selection_freeze.json` | `294a3e4261f9a40e` |
| `labels/TRAIN_oof_STATUS.json` | `f924a5e2be9cf217` |
| `labels/TRAIN_oof_h90_meta.json` | `14870768b7fde836` |
