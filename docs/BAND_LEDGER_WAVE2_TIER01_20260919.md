# Band ledger — V redesign wave-2 Tier 0–1

**Source:** local `outputs/v_redesign_wave2_20260919/results.json`  
**Epistemic:** `EXPLORATORY_V_REDESIGN_SAME_COHORT_PRIOR_TEST_EXPOSURE`  
**Protocol:** [V2_CANDIDATE_MATRIX_20260919.md](V2_CANDIDATE_MATRIX_20260919.md) · \(V=g\circ f\) · `bucket_only` · α_b=1/4

**Mechanical V_SELECT winner (new eligible):** `A0_shared_logistic`  
**Legacy** still slightly better on \(L_{\mathrm{time}}\) but **not eligible** as redesign freeze.  
ΔV default remains Choice A policy until continuity on top contenders.

---

## V_SELECT (15.15)

| Rank | Candidate | \(L_{\mathrm{time}}\) | Brier | AUC | Precedent |
|---:|---|---:|---:|---:|---|
| — | legacy_logistic_raw | 0.137208 | 0.152929 | 0.8583 | old sealed V (ineligible) |
| 1 | **A0_shared_logistic** | **0.137638** | 0.153440 | 0.8573 | Maymin; Hodge LR |
| 2 | A3_shared_mlp | 0.138532 | 0.155333 | 0.8539 | Kim; CoG matched MLP |
| 3 | A2b_shared_lgbm_cog | 0.138907 | 0.155266 | 0.8541 | CoG paper LGBM hypers |
| 4 | B0_per_band_lgbm | 0.139016 | 0.155553 | 0.8534 | Choice B |
| 5 | A2_shared_lgbm_wave1 | 0.139299 | 0.155686 | 0.8533 | wave-1 LGBM 400 |
| 6 | A1_shared_rf | 0.143363 | 0.158095 | 0.8486 | Hodge RF |

## TEST (15.16) — calibrated all-query

| Candidate | Brier | AUC |
|---|---:|---:|
| legacy_logistic_raw | 0.153003 | 0.8586 |
| A0_shared_logistic | 0.153309 | 0.8580 |
| A3_shared_mlp | 0.154724 | 0.8553 |
| B0_per_band_lgbm | 0.155940 | 0.8530 |
| A2b_shared_lgbm_cog | 0.156161 | 0.8530 |
| A2_shared_lgbm_wave1 | 0.156673 | 0.8521 |
| A1_shared_rf | 0.157672 | 0.8498 |

## Reading

- Expanding beyond a single rough LGBM mattered: **re-fit logistic leads** the new slate; MLP and CoG-hyper LGBM beat wave-1 LGBM.
- Hodge RF underperforms on this StateV2 / match-equal setup (not a reason to drop it from the matrix — record as weak arm).
- Still open: Tier 2–3 history/sequence (BiGRU/Transformer/…); Tier 4 GNN (input-blocked).
- Next for freeze: ΔV continuity on **A0 / A3 / A2b** (and legacy compare) under shared \(g\circ f\).
