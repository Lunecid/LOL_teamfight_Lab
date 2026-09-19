# Band ledger — V redesign wave-3 Tier 2–3 (history + sequence)

**Source:** local `outputs/v_redesign_wave3_tier23_20260919/results.json`  
**Epistemic:** `EXPLORATORY_V_REDESIGN_SAME_COHORT_PRIOR_TEST_EXPOSURE`  
**Protocol:** [V2_CANDIDATE_MATRIX_20260919.md](V2_CANDIDATE_MATRIX_20260919.md) · \(V=g\circ f\) · `bucket_only` · α_b=1/4 · K∈{3,5} history stacks  
**ID glossary:** [V_REDESIGN_ID_GLOSSARY_20260919.md](V_REDESIGN_ID_GLOSSARY_20260919.md)

**Mechanical winner (this slate only):** **H5 = history-5 LightGBM** (last 5 frames, flat)  
**Still behind reference:** **REF = A0 shared logistic (current frame only)** — best \(L_{\mathrm{time}}\) across Tier 0–3 so far.

ΔV default remains **Choice A** (shared model); TEST continuity = diagnostic not selection.

| Short | Plain name |
|---|---|
| REF / A0 | Shared logistic, current frame (no history) |
| H3 | History-3 LightGBM (last 3 frames flat) |
| H5 | History-5 LightGBM (last 5 frames flat) |
| S_* | Sequence nets on K=5 (BiGRU / BiLSTM / Transformer / TCN) |

---

## V_SELECT (15.15)

| Rank | ID | Plain name | \(L_{\mathrm{time}}\) | Brier | AUC |
|---:|---|---|---:|---:|---:|
| REF | **A0** | **Shared logistic (current frame)** | **0.137638** | 0.153440 | 0.8573 |
| 1 | H5 | History-5 LGBM | 0.138488 | 0.155099 | 0.8544 |
| 2 | H3 | History-3 LGBM | 0.138609 | 0.155173 | 0.8542 |
| 3 | S_transformer | Transformer on K=5 | 0.139787 | 0.155739 | 0.8537 |
| 4 | S_tcn | TCN on K=5 | 0.157817 | 0.188554 | 0.7622 |
| 5 | S_bigru | BiGRU on K=5 | 0.157962 | 0.188320 | 0.7626 |
| 6 | S_bilstm | BiLSTM on K=5 | 0.158649 | 0.188633 | 0.7622 |

## TEST (15.16) — calibrated all-query

| ID | Plain name | Brier | AUC |
|---|---|---:|---:|
| A0 | Shared logistic (current frame) | 0.153309 | 0.8580 |
| H3 | History-3 LGBM | 0.154450 | 0.8560 |
| H5 | History-5 LGBM | 0.154458 | 0.8559 |
| S_transformer | Transformer K=5 | 0.154934 | 0.8555 |
| S_bigru | BiGRU K=5 | 0.188307 | 0.7633 |
| S_bilstm | BiLSTM K=5 | 0.188509 | 0.7630 |
| S_tcn | TCN K=5 | 0.188563 | 0.7629 |

## Band note (early collapse)

Sequence RNNs / TCN show near-chance AUC in `t_0_10` (~0.50) while mid/late bands remain strong — early-mask / short-history underfit, not a claim that sequence is useless for ΔV.

## Reading

- **Do not conclude “history is useless” or “RNN is unfit” from INPUT_IMPL_v0.**
  Wave-3 used right-aligned stacks with `pack_padded_sequence` (PACK-1 bug) and
  mixed categorical handling. See [V_NEXT_RUN_EXECUTION_CONTRACT_20260919.md](V_NEXT_RUN_EXECUTION_CONTRACT_20260919.md).
- Under v0 scores only: best history arm **H5** still trailed **A0 shared logistic**
  on \(L_{\mathrm{time}}\); treat as **pre-fix ledger**, not learner verdict.
- Next: typed Expanded361/Core267 Phase A, then same-LGBM ± history Phase B after left-align fix.

## Artifact

- Script: `scripts/rr20260919_v_redesign_fit_wave3_tier23.py`
- Models: `outputs/v_redesign_wave3_tier23_20260919/models/`
