# Paper cohort contract — corpus vs evaluation sample

**Status:** revised 2026-09-19 after collaborator review  
**Study corpus:** **210,000** KR matches on patches **15.14 + 15.15 + 15.16**.  
**Primary prediction sample:** **15.16 teamfight T**, \(n=32{,}981\) rows / \(24{,}020\) matches — **all models in the main prediction table use these rows only**.  
**Transfer:** 2026 API `16.x` cohorts — score-only; not used for selection.

Companions: [STRATEGIC_VALUE_LABEL_REDESIGN_20260919.md](STRATEGIC_VALUE_LABEL_REDESIGN_20260919.md),  
[SVI_MODEL_RESELECTION_TRANSFER_20260919.md](SVI_MODEL_RESELECTION_TRANSFER_20260919.md).

---

## 1. Two statements (both true)

1. The **study corpus** is 210,000 matches (15.14–15.16).  
2. The **primary prediction performance** is evaluated on the **15.16 holdout engagement sample** (T), not on pooled T.

Same corpus ≠ same evaluation sample. Never report \(p_{\mathrm{pre}}\) AUC on pooled T beside \(q-\mathrm{PT}\) ΔBrier on 15.16 as one “incremental lift.”

---

## 2. Role of each slice

| Slice | n (T) | Allowed use |
|---|---:|---|
| 15.14 TRAIN | 39,605 | Fit \(\widehat{V}\) queries / \(q\) (OOF \(\widehat{V}\) for TRAIN labels) |
| 15.15 Q_CAL / Q_SELECT | 10,390 / 10,195 | Calibrate / **select** \(q\) (never TEST) |
| **15.16 sealed eval** | **32,981** | **Main prediction table** (\(p_{\mathrm{pre}}\), \(b(p)\), PT, \(q\)) |
| Pooled T (all three) | 113,901 | Measurement: label mix, material concordance, patch strata |
| EXT KR/NA1 16.x | see §4 | Transfer section only |

TRAIN in-sample \(q\) never appears as a main lift number.  
15.14/15.15 “same pattern” ≠ independent holdout proof.

---

## 3. Main-text prediction rules

1. Every column of the primary table is computed on **identical 15.16 T rows** and the **same match weights**.  
2. B40 / B45 / \(B40^{c}\): same subset definition and weights for every model.  
3. **Primary contrast:** \(q\) vs **PT**. Secondary: \(q\) vs \(b(p)\).  
4. **Two Intro RQs only** (see redesign §3):  
   - **Axis A (predictability):** overall \(\Delta\mathrm{Brier}(q-\mathrm{PT})\) + CI.  
   - **Axis B (contested):** B40 / \(B40^{c}\) ΔBrier and  
     \(H=E[d_i\mid B40]-E[d_i\mid B40^{c}]\) with \(d_i=(Y_i-q_i)^2-(Y_i-\mathrm{PT}_i)^2\)  
     via the **same** match-clustered bootstrap (do not infer heterogeneity from sig vs non-sig).  
5. Practical threshold for “meaningful” ΔBrier: \(\tau=0.001\) (declared a priori for interpretation).  
6. Raw \(p_{\mathrm{pre}}\) vs SVI: **diagnostic** only.  
7. ΔBrier ≠ encompassing; optional \(p_\lambda\) combination is auxiliary.  
8. “MAIN_TEST-only headline 금지” = do not confuse corpus with test sample; **15.16 as headline prediction sample is required**.

Literature framing (not warrants): [RELATED_FORECAST_VALUE_LIT_20260919.md](RELATED_FORECAST_VALUE_LIT_20260919.md) — Clements–Harvey, Giacomini–White, Gneiting–Raftery; omit Fair (1996) as central.

---

## 4. Transfer (2026 collection)

| Cohort | Matches | Headline transfer? |
|---|---:|---|
| KR 16.13 | 10,064 | yes |
| NA1 16.13 | 10,000 | yes |
| KR 16.15 | 926 | yes |
| KR 16.14 pilot | 200 | no |

**Freeze for transfer:** \(\widehat{V}\), \(q\), **PT**, **\(b(p)\)**, preprocessing, calibrators. Do not refit baselines on external for the primary transfer table.

**Report separately:**

| Object | Metric |
|---|---|
| \(\widehat{V} \to W\) | AUC / calibration by time (on external matches) |
| \(q \to \mathrm{SVI}(\widehat{V})\) | vs frozen PT and \(b(p)\); all + B40 |

EUW1: 0 complete pairs — omit.

---

## 5. Measurement cite notes (pooled T)

Sources: `outputs/svi_overnight_20260919/stage_concordance/`, `stage_quiet/`.

| Claim | Value | Caveat |
|---|---|---|
| Pooled T | 113,901 rows | measurement only |
| Kill agree / disagree | 0.937 / 0.063 (sum=1.0); decided n=99,091 | match-weighted; tie share 0.130 |
| Obj / alive agree | 0.845 / 0.819 on decided | high tie shares (0.34 / 0.56) |
| Kill–SVI disagree | n=6,395; obj⇄SVI 0.36; obj-tie 0.41 | correspondence, not external truth |
| Quiet matched mean \(\|\Delta V\|\) ratio | 2.666 (median 2.881); pairs=2,361 | \(p_{\mathrm{pre}}\)×time×duration match; signed fight−quiet +0.0056 |

Pooled \(p_{\mathrm{pre}}\) AUC vs SVI remains a **diagnostic** only — never pair it with 15.16 ΔBrier as one lift.

---

## 6. Prediction cite (15.16 T — primary)

**Authority:** `outputs/svi_primary_table_20260919/` (identical rows for every model).

| Claim | Value |
|---|---|
| n | 32,981 / 24,020 |
| Primary \(q\) | **LightGBM** |
| LightGBM − PT ΔBrier | **−0.00113** [−0.00156, −0.00069] |
| LightGBM − \(b(p)\) ΔBrier | −0.00192 [−0.00245, −0.00141] |
| B40 LightGBM − PT | −0.00041 [−0.00146, +0.00066] — CI includes 0 |
| TabM − PT | −0.00105 [−0.00163, −0.00047] (does not beat LGBM) |
| TabNet − PT (point) | −0.00108 (≈ LGBM; +0.00005 vs LGBM) |
| FT-Transformer | collapsed (Brier 0.250, AUC≈0.50) |

15.16 diagnostic \(p_{\mathrm{pre}}\) AUC vs SVI: 0.664 (same rows; not a lift claim).

**Lift localization** (`outputs/svi_lift_localization_20260919/`): overall −0.00113 is not uniform —
late game (\(t{\ge}30\)) ΔBrier ≈ −0.00316 [−0.00486, −0.00150]; B40 CI still includes 0.

**State-dependent / Axis B** (`outputs/svi_state_dependent_20260919/`): \(H\) (B40 vs outside) + flexible PT + B40 info-set ablation + continuous \(\widehat{\Delta V}\) appendix — locked readings in redesign §3.

---

## 7. Forbidden

- Mixing pooled-T diagnostics with 15.16 lift in one interpretive sentence.  
- Selecting on TEST or external.  
- Refitting PT/\(b(p)\) on external in the primary transfer table.  
- Calling material concordance “independent ground-truth validation.”  
- Quiet ratios from unmatched fight vs quiet populations.  
- Treating exploratory post-exposure work as confirmatory.  
- “Overall significant + B40 non-significant ⇒ effects differ” without estimating \(H\).  
- “B40 = empty prior” / “B40 = high leverage” / “CI includes 0 ⇒ no additional information.”  
- Equating ΔBrier superiority with forecast encompassing.
