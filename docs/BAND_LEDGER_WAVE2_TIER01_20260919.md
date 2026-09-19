# Band ledger — V redesign wave-2 Tier 0–1

**Source:** local `outputs/v_redesign_wave2_20260919/results.json`  
**Epistemic:** `EXPLORATORY_V_REDESIGN_SAME_COHORT_PRIOR_TEST_EXPOSURE`  
**Protocol:** [V2_CANDIDATE_MATRIX_20260919.md](V2_CANDIDATE_MATRIX_20260919.md) · \(V=g\circ f\) · `bucket_only` · α_b=1/4  
**ID glossary:** [V_REDESIGN_ID_GLOSSARY_20260919.md](V_REDESIGN_ID_GLOSSARY_20260919.md)

**Mechanical V_SELECT winner (new eligible):** **A0 = shared logistic (current frame)**  
**Legacy** (old sealed logistic) still slightly better on \(L_{\mathrm{time}}\) but **not eligible** as redesign freeze.  
ΔV default remains **Choice A** (one shared model for all times) until continuity on top contenders.

| Short | Plain name |
|---|---|
| A0 | Shared logistic |
| A1 | Shared random forest |
| A2 | Shared LightGBM (wave-1 hypers) |
| A2b | Shared LightGBM (CoG hypers) |
| A3 | Shared MLP |
| B0 | Per-band LightGBM (Choice B) |
| legacy | Old sealed logistic |

---

## V_SELECT (15.15)

| Rank | ID | Plain name | \(L_{\mathrm{time}}\) | Brier | AUC |
|---:|---|---|---:|---:|---:|
| — | legacy | Old sealed logistic | 0.137208 | 0.152929 | 0.8583 |
| 1 | **A0** | **Shared logistic** | **0.137638** | 0.153440 | 0.8573 |
| 2 | A3 | Shared MLP | 0.138532 | 0.155333 | 0.8539 |
| 3 | A2b | Shared LGBM (CoG) | 0.138907 | 0.155266 | 0.8541 |
| 4 | B0 | Per-band LGBM | 0.139016 | 0.155553 | 0.8534 |
| 5 | A2 | Shared LGBM (wave-1) | 0.139299 | 0.155686 | 0.8533 |
| 6 | A1 | Shared RF | 0.143363 | 0.158095 | 0.8486 |

## TEST (15.16) — calibrated all-query

| ID | Plain name | Brier | AUC |
|---|---|---:|---:|
| legacy | Old sealed logistic | 0.153003 | 0.8586 |
| A0 | Shared logistic | 0.153309 | 0.8580 |
| A3 | Shared MLP | 0.154724 | 0.8553 |
| B0 | Per-band LGBM | 0.155940 | 0.8530 |
| A2b | Shared LGBM (CoG) | 0.156161 | 0.8530 |
| A2 | Shared LGBM (wave-1) | 0.156673 | 0.8521 |
| A1 | Shared RF | 0.157672 | 0.8498 |

## Reading

- Expanding beyond a single rough LGBM mattered: **re-fit shared logistic (A0)** leads the new slate; **shared MLP (A3)** and **CoG-hyper LGBM (A2b)** beat wave-1 LGBM.
- Hodge **RF (A1)** underperforms on this StateV2 / match-equal setup (record as weak arm).
- Tier 2–3 done: [BAND_LEDGER_WAVE3_TIER23_20260919.md](BAND_LEDGER_WAVE3_TIER23_20260919.md) — history/seq **behind A0**; Tier 4 GNN still input-blocked.
- Partial V-4 continuity done: [CONTINUITY_LEDGER_WAVE2_20260919.md](CONTINUITY_LEDGER_WAVE2_20260919.md) — A0 excess≈legacy; A0↔legacy sign agree 0.98.
- Next for freeze: full V-4 (\(D_{\mathrm{switch}}\), quiet/event) then `freeze_manifest` if A0 holds.
