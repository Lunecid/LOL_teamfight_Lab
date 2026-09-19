# Band ledger — `shared_lgbm` (calibrated) · wave-1

**Source:** local `outputs/v_redesign_20260919/results.json` (gitignored tensors; this table is the citeable extract).  
**Epistemic:** `EXPLORATORY_V_REDESIGN_SAME_COHORT_PRIOR_TEST_EXPOSURE`  
**Evaluation map:** \(\widehat{V}_{\mathrm{final}}=g\circ f\circ T\) with PosSlopeSigmoid \(g\) fit on **V_CAL**.  
**Query sample:** `bucket_only=True` (`is_bucket_sample`), grid from 120 s — **not** every minute frame.

Do **not** read [TIMEBAND_COG_TOG_SVI_COMPARE_20260919.md](TIMEBAND_COG_TOG_SVI_COMPARE_20260919.md) AUC rows as this model’s ledger (those are old-V / local ToG-pipeline / CoG fight-outcome).

---

## V_SELECT (15.15) — selection cell

| Candidate | \(L_{\mathrm{time}}\) | Brier | logloss† | AUC |
|---|---:|---:|---:|---:|
| shared_lgbm | 0.139299 | 0.155686 | 0.464345 | 0.8533 |
| per_band_lgbm | 0.139016 | 0.155553 | 0.463874 | 0.8534 |
| legacy_logistic (compare only) | 0.137208 | 0.152929 | 0.456066 | 0.8583 |

† Wave-1 **tie-break** = overall match-weighted logloss on V_SELECT (not time-balanced logloss).  
Mechanical min-\(L_{\mathrm{time}}\) among *new* candidates = `per_band_lgbm`.  
**Provisional freeze = Choice A `shared_lgbm` (a priori)** — not selected by TEST continuity.

### shared_lgbm — V_SELECT by band

| Band | n | n_match | Brier | logloss | AUC |
|---|---:|---:|---:|---:|---:|
| t_0_10 | 37481 | 18780 | 0.2269 | 0.6442 | 0.6701 |
| t_10_20 | 37226 | 18636 | 0.1570 | 0.4743 | 0.8535 |
| t_20_30 | 26922 | 15253 | 0.0898 | 0.2895 | 0.9492 |
| t_30_inf | 6929 | 5012 | 0.0835 | 0.2736 | 0.9561 |

Calibration (shared): coef≈1.0314, intercept≈−0.0271 (ok).

---

## TEST ledger (15.16) — sealed report only

### shared_lgbm (calibrated)

| Band | n | n_match | Brier | logloss | AUC |
|---|---:|---:|---:|---:|---:|
| all | 347234 | 60190 | 0.1567 | 0.4674 | 0.8521 |
| t_0_10 | 120186 | 60190 | 0.2291 | 0.6494 | 0.6652 |
| t_10_20 | 119488 | 59825 | 0.1562 | 0.4729 | 0.8552 |
| t_20_30 | 85342 | 48440 | 0.0916 | 0.2958 | 0.9469 |
| t_30_inf | 22218 | 15915 | 0.0869 | 0.2851 | 0.9517 |

### legacy logistic (raw) on same TEST queries — baseline compare

| Band | Brier | AUC |
|---|---:|---:|
| all | 0.1530 | 0.8586 |
| t_0_10 | 0.2262 | 0.6741 |
| t_10_20 | 0.1508 | 0.8637 |
| t_20_30 | 0.0883 | 0.9505 |
| t_30_inf | 0.0874 | 0.9508 |

**Reading:** New shared LGBM tracks the same early-hard / late-easy pattern; on this slate it does **not** beat legacy logistic on \(L_{\mathrm{time}}\) or TEST Brier. Freeze is about **ΔV-compatible shared map**, not beating legacy WP Brier.

Not yet in this ledger (deferred): band ECE/reliability curves; match-clustered CIs; engagement-time \(V\to W\).

---

## Artifact names

| File | Role |
|---|---|
| `outputs/v_redesign_20260919/results.json` | Selection + TEST ledger (local) |
| `outputs/v_redesign_20260919/freeze_manifest.json` | **Final evaluator freeze** (\(T,f,g\)) |
| `outputs/v_redesign_20260919/winner_manifest.json` | Short pointer / mechanical vs provisional |
| this doc | GitHub-visible band numbers |
