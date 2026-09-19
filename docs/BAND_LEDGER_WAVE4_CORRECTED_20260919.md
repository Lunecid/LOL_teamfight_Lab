# Band ledger — V redesign wave-4 corrected inputs (Phase A/B)

**Source:** `outputs/v_redesign_wave4_corrected_20260919/results.json`  
**Input impl:** `corrected_v1` — CAT-1 · PACK-1 · CAL-1 · HIST-1  
**Epistemic:** `EXPLORATORY_V_REDESIGN_SAME_COHORT_PRIOR_TEST_EXPOSURE`  
**Contract:** [V_NEXT_RUN_EXECUTION_CONTRACT_20260919.md](V_NEXT_RUN_EXECUTION_CONTRACT_20260919.md)  
**ID glossary:** [V_REDESIGN_ID_GLOSSARY_20260919.md](V_REDESIGN_ID_GLOSSARY_20260919.md)

**Mechanical V_SELECT winner:** **A_MLP_expanded** — embedding MLP on Expanded361 (351 numeric + 10×8 champ emb → 431)  
**Close second:** **B_GRU_K5_expanded** (left-aligned history GRU) — within ~0.0002 of winner.

Do **not** mix these scores with INPUT_IMPL_v0 (wave-2/3) as equal evidence: weights are mean-one, V_CAL is calib-only, categoricals are typed.

---

## What was fixed

| ID | Fix |
|---|---|
| CAT-1 | Numeric/categorical split; LR one-hot; LGBM native cat; MLP emb-8 |
| PACK-1 | Left-aligned history for GRU `pack_padded_sequence` |
| CAL-1 | 15% match-holdout inside TRAIN for early stop; V_CAL → \(g\) only |
| HIST-1 | Current token = supervised query \(X\); past from observation store; rel-time; champ once |
| Profiles | **Expanded361** (primary) vs **Core267** (drop 94 team×time) |

**Store density note:** TRAIN/V_CAL/V_SELECT full loads equal bucket counts (no denser frames). Only **TEST** has denser non-bucket rows (~1.49M). History on SELECT therefore still uses bucket spacing; packing/current-token contracts still apply.

---

## Plain names

| ID | Plain name |
|---|---|
| A_LR_* | Logistic, numeric + TRAIN one-hot champions |
| A_LGBM_* | LightGBM, native categorical champions |
| A_MLP_* | MLP, champion embeddings (e=8) |
| B_LGBM_H3/H5 | Same LGBM hypers + history K=3/5 flat |
| B_GRU_K5 | History GRU + current skip + champ emb |

---

## V_SELECT (\(L_{\mathrm{time}}\), lower better)

| Rank | ID | Plain | \(L_{\mathrm{time}}\) | Brier | AUC |
|---:|---|---|---:|---:|---:|
| 1 | **A_MLP_expanded** | Embedding MLP, Expanded361 | **0.1383** | 0.1552 | 0.8542 |
| 2 | B_GRU_K5_expanded | History GRU K=5 | 0.1385 | 0.1547 | 0.8548 |
| 3 | A_MLP_core | Embedding MLP, Core267 | 0.1387 | 0.1556 | 0.8535 |
| 4 | A_LR_expanded | Logistic Expanded | 0.1395 | 0.1557 | 0.8531 |
| 5 | A_LGBM_expanded | Native LGBM Expanded | 0.1397 | 0.1562 | 0.8523 |
| 6 | A_LGBM_core | Native LGBM Core | 0.1398 | 0.1562 | 0.8523 |
| 7 | B_LGBM_H3_expanded | LGBM history-3 | 0.1400 | 0.1571 | 0.8505 |
| 8 | A_LR_core | Logistic Core | 0.1401 | 0.1564 | 0.8518 |
| 9 | B_LGBM_H5_expanded | LGBM history-5 | 0.1412 | 0.1585 | 0.8477 |

## Same-learner history test (question that v0 could not answer)

| Compare | \(L_{\mathrm{time}}\) | Δ |
|---|---:|---:|
| A_LGBM_expanded (current) | 0.1397 | — |
| B_LGBM_H5_expanded | 0.1412 | **+0.0015** (history worse) |
| B_LGBM_H3_expanded | 0.1400 | +0.0003 |

Under matched LGBM + typed cats + corrected history builder, **flat history does not beat current-frame LGBM**.

**GRU history** (0.1385) **does** beat current LGBM (0.1397) and nearly matches embedding MLP — so “RNN unfit” from v0 is **withdrawn**; packing fix mattered.

## Expanded vs Core

| Learner | Expanded | Core | Prefer |
|---|---:|---:|---|
| MLP | **0.1383** | 0.1387 | Expanded (small) |
| LGBM | **0.1397** | 0.1398 | ~tie |
| Logistic | **0.1395** | 0.1401 | Expanded |

Explicit team×time interactions help logistic/MLP slightly; not a large LGBM effect.

## vs wave-2 A0 (context only)

Wave-2 **A0 shared logistic** had \(L_{\mathrm{time}}=0.1376\) under INPUT_IMPL_v0 (different weight scale / categorical pipeline). Wave-4 LR is not a drop-in replacement score. Treat A0 as **legacy baseline reference**, wave-4 MLP as **best corrected-input eligible arm so far**.

## Freeze reading

- Provisional corrected-input leader: **embedding MLP (Expanded361)**.
- Keep **A0 (wave-2)** and **B_GRU_K5** as continuity / ablation peers.
- Flat H3/H5 LGBM: no freeze case.
- Next: partial/full V-4 continuity on **A_MLP_expanded** (+ A0, GRU) with shared as-of current token; then freeze_manifest.

## Artifacts

- Script: `scripts/rr20260919_v_redesign_fit_wave4_corrected.py`
- Adapters: `scripts/v_redesign_feature_adapters.py`
- Models: `outputs/v_redesign_wave4_corrected_20260919/models/`
