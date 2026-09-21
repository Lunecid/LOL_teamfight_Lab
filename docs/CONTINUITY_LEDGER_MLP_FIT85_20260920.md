# Continuity ledger — A_MLP_expanded fit85 bundle

Generated: 2026-09-20T00:36:39+09:00
**Evaluator:** `A_MLP_expanded_evaluator.joblib`
**Fit scope:** `train_fit85_match_holdout_frac0.15_seed7_NO_full_train_refit`

Primary: change-measurement suitability of \(\widehat{V}_{\mathrm{fit85}}\). Peers (A0, corrected LR) are comparison only — not selection targets.

**RQ placement:** Journal **J-RQ1** part ② (ΔV/SVI diagnostics), same evidence for master **M-RQ2**. Does **not** close J-RQ1/M-RQ2. Question lock: [J_RQ1_SCOPE_LOCK_20260920.md](J_RQ1_SCOPE_LOCK_20260920.md).

## 1. Consecutive TEST |ΔV| (calibrated)

| interior mean | boundary mean | excess (b−i) | P(|Δ|>0.05) boundary |
|---:|---:|---:|---:|
| 0.1525 | 0.1706 | 0.0182 | 0.7442 |

## 2. Engagement pre→post (h90, n=32981)

- mean |ΔV|=0.1154; median=0.0752; P(|ΔV|>0.05)=0.6013
- frac exact 0=0.0000; frac |ΔV|<1e-3=0.0214
- V_pre→W: Brier=0.1442 AUC=0.8750
- V_post→W: Brier=0.1181 AUC=0.9149

### Sign(ΔV) agree (nonzero)

- MLP_vs_A0_wave2: 0.9151
- MLP_vs_A_LR_expanded_fit85: 0.9136

## Reading

- Boundary excess on bucket queries ≠ proof of artifact; document as diagnostic.
- Engagement V→W links training queries to application times.
- Do not prefer a peer merely for higher sign agree with A0.
