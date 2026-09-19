# V freeze-prep — accept collaborator post-wave4 review

**Date:** 2026-09-20  
**Reviewed commit:** `18a2c39`  
**Status:** Design direction accepted. **Not yet frozen.**

---

## 0. Agreement (what we keep)

- Corrected-input comparison is **much better** than INPUT_IMPL_v0.
- **Provisional leader:** Expanded361 embedding MLP (`A_MLP_expanded`) by **predeclared** \(L_{\mathrm{time}}\) on V_SELECT — not because overall Brier/AUC win (GRU is slightly better there).
- Do **not** restart architecture search; finish the evaluator.

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
| P0 | Persist **full evaluator_bundle** (schema, mean/std, vocab, weights, \(g\), fit scope) | script + check |
| P0 | New-process **reload reproducibility** on fixed queries | check doc |
| P1 | DEV continuity on MLP (+ corrected LR, wave2 A0, GRU as peers) | next |
| P1 | **TEST band-wise** ledger (not overall-only) for leader | next |
| P2 | History density unification (only if promoting GRU/H5) | scoped caveat for now |
| P2 | Optional: LR numeric-only scaling; shared Brier early-stop | version bump if changed |

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
