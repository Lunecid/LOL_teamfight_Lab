# Scale-split experiment contract — teamfight T vs skirmish S under the frozen fit85 evaluator

**Date:** 2026-09-20 · **Status:** PREDECLARED (written before any S-cohort label, fit or score exists) · **Line:** thesis / v2 experiment line (M-RQ1 scale sensitivity); **not** a journal-completion item  
**Role tag for every artifact:** `EXPLORATORY_SCALE_SPLIT_PRIOR_TEST_EXPOSURE`  
**Author decision (2026-09-20):** run under this contract first; decide journal inclusion only after the results report.

## 0. Why this document exists

The journal freeze ([JOURNAL_FINISH_LOCK_20260920.md](JOURNAL_FINISH_LOCK_20260920.md)) stops V/q performance racing and lists what must not be reopened. A cohort change is not on that list, and I1 of [COG_SUCCESSION_LOCK_20260919.md](COG_SUCCESSION_LOCK_20260919.md) §2 names "size scope (T vs N)" as a legitimate scope question. But [Q_PREDICTION_DESIGN_CONTRACT_20260920.md](Q_PREDICTION_DESIGN_CONTRACT_20260920.md) freezes the engagement scope of the primary q run, and [COMMON_RESEARCH_SPINE_20260919.md](COMMON_RESEARCH_SPINE_20260919.md) §3 keeps "주 교전 T; N은 보조·부록". A silent re-run on another cohort would violate those. This contract is the explicit addendum: it fixes cohort, labels, arms, the single primary contrast and the forbidden readings **before** execution.

## 1. What was already done (lineage, not to be re-derived)

- 2026-09-15 (`codex/research-snapshot-20260917`, commit `e0ec3d0`, `outputs/cohort_role_training_20260915/`): under the **old logistic V**, the same candidate pool was refit per cohort (T = min participation ≥ 4; N = the rest). Main TEST h90: T specialist vs pooled ΔBrier −0.00166 [−0.00235, −0.00104]; N specialist −0.00042 [−0.00057, −0.00024]; inside N, skirmish (2–3) −0.00023 [−0.00043, −0.00004], pick −0.00083. No PT baseline existed in that run. Stated conclusion: "Treat N specialization as not robust"; T/N mechanism explanation withdrawn (claim ledger X-31).
- Under the market_event label (ToG lineage, [tog_manuscript/sec_prediction.tex](tog_manuscript/sec_prediction.tex)): skirmish AUC 0.6633 vs teamfight 0.6809, with the pick−teamfight sign flipping across cuts 3/4/5 and the earlier "scale gradient" withdrawn (time_norm leak). **Not comparable** to this run (different label, corpus version, evaluator).
- Under the frozen fit85 MLP V (2026-09-20 pipeline) only T has ever been labeled and fit.

## 2. Cohorts

| Symbol | Rule (stored v3.3 participation counts, `{SET}_cohort.npz`) | Status |
|---|---|---|
| **T** | `cohort == 1` ⟺ `min(cluster_blue, cluster_red) ≥ 4` (≡ `fine == 2`) | frozen results reused as-is; **no refit** |
| **S** | `cohort == 0 & fine == 1` ⟺ `2 ≤ n_min ≤ 3` (skirmish) | new |
| pick (`fine == 0`) | excluded from this contract | — |

All rows additionally require `valid_h90 == 1`, `pre_ok == 1`, finite pre/post scores — identical to the T pipeline. Cohort membership is **post-cutoff** information: no routing, deployment or "which fights to bet on" claim.

Expected S row counts at h90 (from `cohort_manifest.json`, `e0ec3d0`): TRAIN fold0–4 = 24 030 / 25 005 / 24 520 / 24 690 / 24 804 (Σ 123 049); Q_CAL 31 302; Q_SELECT 31 059; TEST 101 205; EXT KR 16.13 15 641, NA1 16.13 16 100, KR 16.15 1 307, KR 16.14 pilot 285. Deviations must be explained in the run manifest (finite-score exclusions are expected to be few).

## 3. Labels — same evaluators as T

- Definitions unchanged: τ = first kill − 15 s, q_pre = τ − 1 ms; endpoint_h90 = min(L + 90 s, next kill − 1 ms, next engagement start − 1 ms, game end − 1 ms); ΔV̂ = V̂(S_end) − V̂(S_pre); Y = 1[ΔV̂ > 0]; ΔV̂ = 0 → Y = 0; non-finite → excluded; B40 = 0.40 ≤ p_pre ≤ 0.60 recomputed from the new p_pre.
- **TRAIN (15.14):** the five saved fold evaluators `outputs/q_newv_fit85_20260920/oof_evaluators/V_oof_fold{k}_mlp_expanded.joblib` + `bundle_oof_fold{k}.joblib` are **reused**; S rows of fold-k matches are scored by V^(−k) at pre and post. Folds are match-level, so the leave-match guarantee holds unchanged.
- **Q_CAL / Q_SELECT (15.15), TEST (15.16), EXT:** frozen fit85 bundle (`A_MLP_expanded_evaluator.joblib`, sha16 `ac459cc4397630a9`).
- **If the fold evaluators are missing on disk:** refit them exactly per [Q_PREDICTION_DESIGN_CONTRACT_20260920.md](Q_PREDICTION_DESIGN_CONTRACT_20260920.md) §2 (same script, seed 7, 15% holdout), relabel **both** T and S, and report the T reproduction rate (Y agreement and |Δp_pre| quantiles against the frozen `TRAIN_oof_h90.npz`). Anything below 100% Y agreement is a BLOCKED item for the author, not a silent substitution.

## 4. Arms (no learner re-selection; identity calibrator; same 352 inputs as the frozen q)

| Arm | Fit on | Applied to | Purpose |
|---|---|---|---|
| **q_S** | S TRAIN (OOF labels) | S | specialist |
| **q_T→S** | — (frozen `models/logit_state.joblib`) | S | transfer of the journal model |
| **q_TS** | T ∪ S TRAIN (match weights recomputed inside the union) | S and T | pooled |
| **PT_flex_S**, PT_linear_S, b(p)_S | S TRAIN; knots/C chosen on S Q_SELECT with the same grid and two-stage rule as RR1/RR12 (uniform knots) | S | cohort-specific baselines |
| `lgbm_state_S` | S TRAIN, early stop on S Q_CAL | S | diagnostic only, never primary |

The learner for q_S and q_TS is fixed a priori to the frozen T specification (`logit_state`: 351 numeric + p_pre, L2 logistic, standardized). Q_SELECT is used only to choose PT_flex knots/C, exactly as for T. No candidate is selected on TEST.

## 5. Contrasts

**Primary (one):** on 15.16 TEST **S rows**, ΔBrier(q_S − PT_flex_S), paired match-cluster percentile bootstrap, 2 000 draws, seed 7, cell weights w = 1/n_m.

**Secondary (reported regardless of sign):**
1. identical S rows: ΔBrier(q_S − q_T→S), ΔBrier(q_S − q_TS);
2. identical T rows: ΔBrier(q_TS − q_T) — does pooling cost the teamfight model anything;
3. S ∩ B40 (recomputed): ΔBrier(q_S − PT_flex_S);
4. CORP components (point values, no intervals) for q_S and PT_flex_S on S;
5. external score-only, S rows of KR/NA1 16.13 (and the two small cohorts, reported not pooled): V_pre Brier, ΔBrier(q_S − PT_flex_S), CORP — no intervals, as in RRX.

τ = 0.001 remains an interpretation threshold only.

## 6. Forbidden readings

- Comparing S and T absolute Brier/AUC as an "improvement", a "scale gradient", or "which engagements are more predictable" ([tog_manuscript/sec_prediction.tex](tog_manuscript/sec_prediction.tex) §withdrawn conclusion; [MASTER_THESIS_RESEARCH_PLAN_20260919.md](MASTER_THESIS_RESEARCH_PLAN_20260919.md) M-RQ1 interpretation limit).
- Any mechanism story for a T/S difference (claim ledger X-31 stays withdrawn).
- Promoting a secondary contrast, a λ filter, a bin or a learner because of its TEST value.
- Substituting all-N or pick results for S, or pooling the small external cohorts into a success claim.
- Touching the frozen T artifacts, the journal manuscript, or any lock document before the results report is reviewed.

## 7. Artifacts

- `outputs/q_newv_fit85_20260920_S/{labels,models}/`, `outputs/scale_split_TvsS_20260920/` (predictions, paired CIs, CORP, external).
- `docs/SCALE_SPLIT_RR0_MANIFEST_20260920.json` — keys, row counts, sha16 digests of every input and output, evaluator paths, bootstrap settings.
- `docs/SCALE_SPLIT_TvsS_RESULTS_20260920.{md,json}` — tables for §5, one "primary" row, allowed/forbidden readings copied from §6.
- Existing T artifacts and docs are never overwritten (all new paths carry the `_S` / `scale_split` suffix).

## 8. Promotion rule (after results)

If the author later decides to include S in the journal, the following sentences must be explicitly amended with a dated note, not silently: COMMON_RESEARCH_SPINE §3 "주 교전 T; N은 보조·부록"; PAPER_COHORT_CONTRACT §1/§3 "all models in the main prediction table use these rows only"; JOURNAL_FINISH_LOCK "Block manuscript on new performance — No". Until then S is thesis/v2 material.

**Decision 2026-09-21 (author):** promoted to the journal **main text** (option C, two cohort blocks). Main reporting = contract-literal **identity** calibrator (T010); the RR12 two-stage selection values (T009) are kept as sensitivity. The three sentences named above were amended with dated notes (additions only) on 2026-09-21: [COMMON_RESEARCH_SPINE_20260919.md](COMMON_RESEARCH_SPINE_20260919.md) §3, [PAPER_COHORT_CONTRACT_20260919.md](PAPER_COHORT_CONTRACT_20260919.md) §1, [JOURNAL_FINISH_LOCK_20260920.md](JOURNAL_FINISH_LOCK_20260920.md). §6 forbidden readings carry over into the manuscript unchanged.
