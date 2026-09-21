# V freeze-prep — accept collaborator post-wave4 review

**Date:** 2026-09-20  
**Reviewed commit:** `18a2c39`  
**Status:** Design direction accepted. **Not yet frozen.**  
**RQ framing:** 저널 **J-RQ1** 문구는 고정([J_RQ1_SCOPE_LOCK_20260920.md](J_RQ1_SCOPE_LOCK_20260920.md)). 현재 작업은 RQ1 **부분 ①** (\(\widehat{V}\to W\)). 석사 인용 번호는 **M-RQ2**. 질문 고정 ≠ RQ1/M-RQ2 검증 완료.

---

## 0. Agreement (what we keep)

- Corrected-input comparison is **much better** than INPUT_IMPL_v0.
- **Provisional leader:** Expanded361 embedding MLP (`A_MLP_expanded`) by **predeclared** \(L_{\mathrm{time}}\) on V_SELECT — not because overall Brier/AUC win (GRU is slightly better there).
- Do **not** restart architecture search; finish the evaluator, then connect **ΔV/SVI meaning & stability** (RQ1 part ② / M-RQ2).

---

## 1. Narrowed claims (mandatory wording)

| Too strong | Use instead |
|---|---|
| “MLP is clearly better” | “Among current candidates, MLP has the lowest V_SELECT \(L_{\mathrm{time}}\) (point estimate; single seed).” |
| “H5 history never helps” | “Under current store density + flat H3/H5 + these LGBM settings, flat history did not beat same-learner current LGBM.” |
| “GRU improved because of packing alone” | “After input/training corrections (incl. PACK-1), GRU beats current LGBM here; v0 ‘RNN unfit’ is withdrawn. Packing-only effect was not isolated.” |
| “Expanded ≡ Core” | Report point estimates only; no equivalence claim. |

---

## 2. Fit-scope lock (choose A for continuity now)

| Option | Decision |
|---|---|
| **A — use fit85 as provisional evaluator** | **LOCKED for continuity / next labels** |
| B — full TRAIN refit at stopped epoch | Deferred as `corrected_v2` (needs `best_epoch` recorded first) |

**Rule:** The model that passes continuity must be the model that produces SVI labels. No silent swap.

Identifier: `train_fit85_match_holdout_frac0.15_seed7_NO_full_train_refit`

---

## 3. Remaining blockers before final freeze

| Priority | Item | Status |
|---|---|---|
| P0 | Persist **full evaluator_bundle** (schema, mean/std, vocab, weights, \(g\), fit scope) | done (script + check) |
| P0 | New-process **reload reproducibility** on fixed queries | done (subprocess PASS) |
| P1 | DEV continuity on MLP (+ corrected LR, wave2 A0 as peers) | ledger written; interpret as diagnostic |
| P1 | **TEST** time-resolved ledger (band + per-minute raw grid) | ledgers + figures |
| P1 | **RQ1 part ② / M-RQ2:** ΔV/SVI meaning & stability on same `fit85` evaluator | **table + analyses written** → [NEWV_ENGAGEMENT_VALUE_VERIFY_20260920.md](NEWV_ENGAGEMENT_VALUE_VERIFY_20260920.md); quiet smoke optional; then RQ1 writeup (still not q) |
| P0 | V freeze close + SHA256 | [V_EVALUATOR_FREEZE_CLOSE_20260920.md](V_EVALUATOR_FREEZE_CLOSE_20260920.md) |
| P2 | History density unification (only if promoting GRU/H5) | scoped caveat for now |
| P2 | Optional: full TRAIN refit → `corrected_v2` | deferred |

---

## 4. Continuity peer set

Primary: **`A_MLP_expanded` evaluator_bundle**  
Peers: corrected Logistic Expanded, wave-2 A0, `B_GRU_K5` (history-density caveat attached)  
Not peers for freeze: flat H5 as “history helps” claim.

Shared-model candidates: do **not** require \(D_{\mathrm{switch}}\) band-model swap; focus as-of refresh / event / clock diagnostics.

---

## 5. Artifact index

- Ledger: [BAND_LEDGER_WAVE4_CORRECTED_20260919.md](BAND_LEDGER_WAVE4_CORRECTED_20260919.md)
- Reload check: [V_EVALUATOR_BUNDLE_RELOAD_CHECK_20260919.md](V_EVALUATOR_BUNDLE_RELOAD_CHECK_20260919.md)
- Export script: `scripts/rr20260919_v_export_evaluator_bundle.py`
- Bundle API: `scripts/v_redesign_evaluator_bundle.py`
