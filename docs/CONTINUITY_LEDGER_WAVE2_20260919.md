# Continuity ledger — wave-2 freeze contenders (partial V-4)

**Source:** `outputs/v_redesign_wave2_20260919/CONTINUITY_REPORT.md`  
**Script:** `scripts/rr20260919_v_redesign_continuity_wave2.py --with-h5`  
**Epistemic:** `EXPLORATORY_V_REDESIGN_SAME_COHORT_PRIOR_TEST_EXPOSURE`  
**Role of TEST:** diagnostic / warning only — **not** freeze selection.  
**ID glossary:** [V_REDESIGN_ID_GLOSSARY_20260919.md](V_REDESIGN_ID_GLOSSARY_20260919.md)

**Provisional freeze candidate:** **A0 = shared logistic (current frame)** — Choice A policy (one model for all times).

| Short | Plain name |
|---|---|
| A0 | Shared logistic (current frame) |
| A3 | Shared MLP (current frame) |
| A2b | Shared LightGBM, CoG hypers (current frame) |
| H5 | History-5 LightGBM (last 5 frames, flat) |
| legacy | Old sealed logistic (compare only) |

---

## Consecutive bucket |Δp| excess (calibrated, TEST)

| Model | Plain name | excess (boundary − interior) |
|---|---|---:|
| **A0** | **Shared logistic** | **0.0112** |
| legacy | Old sealed logistic | 0.0112 |
| H5 | History-5 LGBM | 0.0132 |
| A3 | Shared MLP | 0.0135 |
| A2b | Shared LGBM (CoG) | 0.0151 |

A0 matches legacy excess; A2b is the noisiest of the contenders on this (unmatched) query series.

## Engagement sign(ΔV) agree (n=32 981, nonzero)

| Pair (plain) | Agree |
|---|---:|
| Shared logistic (A0) vs **legacy** | **0.9815** |
| Shared logistic (A0) vs shared MLP (A3) | 0.9114 |
| Shared logistic (A0) vs shared LGBM-CoG (A2b) | 0.9060 |
| Shared MLP (A3) vs shared LGBM-CoG (A2b) | 0.8850 |
| Shared logistic (A0) vs history-5 LGBM (H5) | 0.7230 |

Top **current-frame** arms are highly SVI-sign-stable with each other and with legacy. **History-5** diverges more (~72%) — but H5 used a **different current-token construction** (last buckets, not `X_pre`/`X_post`; ENG-1), so do **not** read this as pure model disagreement. See [V_NEXT_RUN_EXECUTION_CONTRACT_20260919.md](V_NEXT_RUN_EXECUTION_CONTRACT_20260919.md).

## Coverage

**Partial V-4 only.** Still missing: \(D_{\mathrm{switch}}(x)\), quiet/event strata, frame-refresh-only.

## Freeze reading

- Continuity does **not** overturn **shared logistic (A0)** among *current-frame* arms.
- **Do not** freeze or reject history/RNN from INPUT_IMPL_v0 scores (PACK-1 / HIST-1 / CAT-1 open).
- Next: Step 1 manifest ✓ → typed Phase A → fixed history Phase B.
