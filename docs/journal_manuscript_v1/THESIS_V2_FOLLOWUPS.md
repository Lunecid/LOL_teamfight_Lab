# Thesis / v2 follow-ups — preregistration-ready plans (T013, 2026-09-21)

*Method: hypothesis-generation skill (observation → question → rival candidates → discriminating predictions → analysis plan → deviation log) and experimental-design principles. Every item is **deferred**: none is part of the journal paper, none has been run, and nothing here is evidence. Labels: `candidate` = a proposition to be challenged. Locks respected: no V or q performance search for the journal (`JOURNAL_FINISH_LOCK`), no new numbers, thesis/v2 only.*

## F1. External probability adapter (deferred by `JOURNAL_FINISH_LOCK` L65–73; `WORKING_NOTES` §B)

- **Observation (reported).** On KR/NA1 16.13 T rows, ΔBrier(q − PT_flex) is +0.0026 / +0.0040 (no intervals); CORP shows q DSC still above PT_flex while ΔMCB dominates (RRX). On S rows the ordering is reversed (−0.0018 / −0.0020).
- **Research question (PICO-like).** In new-environment T rows (population), does a low-dimensional monotone adapter fit on a small held-out slice of that environment (intervention) versus the same adapter applied to PT_flex (comparator) reduce Brier (outcome), with V̂, q and PT_flex weights frozen?
- **Candidates (rivals).** (a) `candidate`: the external loss is dominated by a shift in the probability scale (adapter recovers most of the gap); (b) `candidate`: the shift is in the evaluator V̂ itself (adapter on q cannot recover; V̂→W on external matches degrades); (c) `candidate`: composition shift (positive rate, B40 share) explains the ordering without any scale shift; (d) artifact: feature-order or patch-specific feature semantics.
- **Discriminating predictions.** (a) predicts adapter gain on q larger than on PT_flex; (b) predicts V̂→W external Brier worsens with time band in the same way for q and PT_flex; (c) predicts the gap closes after reweighting to the 15.16 p_pre/time composition; (d) predicts a fail-closed feature check would trip (it did not in RRX).
- **Analysis plan (prespecified).** Adapt/evaluate split by match within each external cohort (e.g., 30/70, seed fixed), adapter family fixed a priori (positive-slope sigmoid on logit), primary = paired ΔBrier(q_adapted − PT_flex_adapted) on the evaluation slice with match-cluster bootstrap; secondary = V̂→W on the same matches; report both cohorts; small cohorts reported, not pooled. Negative control: adapter fit on shuffled labels must give no gain.
- **Not allowed to claim.** "Recalibration fixes transfer" (Forbidden list); anything from 15.16 CORP about 16.x.

## F2. Participation-count cells (2 / 3 / 4 / 5+) within the frozen design

- **Observation.** T and S each show a within-cohort lift; the definition lineage's AUC by n_min (market_event label, old learner) was non-monotone and its pick-minus-teamfight sign flipped with the cut (`sec_prediction.tex` §pred-cut). 15.16 rows by n_min (all rows, lineage joint counts): 1: 29 432; 2: 68 134; 3: 33 108; 4: 21 555; 5: 11 426.
- **Question.** Within each n_min cell, is ΔBrier(q_cohort − PT_flex_cohort) different from zero, and does the sign of the cell-to-cell difference depend on where the cut is placed?
- **Candidates.** (a) `candidate`: the within-cohort lift is roughly uniform across n_min; (b) `candidate`: the lift concentrates at the cohort boundary (n_min = 3 and 4) because the cut is a convention; (c) artifact: cell size and B40 share drive apparent differences.
- **Predictions.** (a) all four cells' intervals overlap; (b) cells 3 and 4 differ from 2 and 5+ in the same direction; (c) differences vanish after matching on p_pre and time.
- **Plan.** Re-score the existing prediction tables by exact n_min from `{SET}_cohort.npz` (no refit); paired match-cluster bootstrap per cell; predeclare that cells are descriptive and that no "gradient" wording is allowed; report cell sizes and B40 share alongside.
- **Not allowed to claim.** Any scale gradient or "which fights are more predictable" (contract §6; FINISH_LOCK Forbidden).

## F3. One shared PT_flex object across S arms

- **Observation.** Each S RR12 run re-selected PT_flex on S Q_SELECT and chose the same (4, 4, 0.01); `SCALE_SPLIT_TvsS_RESULTS` "Suggestions" asks whether a single frozen PT_flex_S object should be reused.
- **Question.** Does freezing PT_flex_S once change any reported S contrast?
- **Prediction.** No change beyond floating-point noise, because selection was deterministic on identical rows.
- **Plan.** Serialize one PT_flex_S from the `_qS_id` run and re-run `_qT_id`, `_qTS_id` with it; assert prediction tables equal; record in the manifest. Documentation hygiene, not a result.

## F4. A predeclared T-versus-S external-transfer contrast

- **Observation.** External ΔBrier orderings differ by cohort (T positive, S negative); the paper reports them side by side without a contrast (`\pending{ext-TS-sentence}`).
- **Question.** Is the difference between the cohorts' external ΔBrier larger than match-sampling variation, on the same external matches?
- **Candidates.** (a) `candidate`: the T-vs-S difference is within resampling noise; (b) `candidate`: the difference is real but explained by composition (S has more balanced-state rows, larger n); (c) `candidate`: the frozen T model overfits patch-specific teamfight features more than the S model.
- **Predictions.** (a) a paired match-cluster bootstrap over external matches yields an interval covering 0; (b) reweighting S rows to the T p_pre/time composition shrinks the difference; (c) SHAP-free ablation (drop patch-sensitive feature groups) changes T more than S. Rivals (b) and (c) predict different behaviour under reweighting.
- **Plan.** Predeclare the estimand (difference of within-cohort ΔBrier on matches that contain both cohorts' rows), the bootstrap (2 000 draws, seed 7), and the forbidden readings (no mechanism claim; X-31 stays withdrawn). Run only after the journal is submitted.

## F5. Definition-constant sensitivity under the v3.3 corpus

- **Observation.** Methods §1 states that sensitivity of secondary constants (anchor placement, shop-event exclusion, horizon, execution kills, grid error, G × D sweep) was measured on conference-era or pilot corpora and not re-run under the current corpus.
- **Question.** Do the headline contrasts (T and S primary ΔBrier) change sign or leave their intervals when G and D move within their bootstrap intervals and when the 15 s lead B changes to 10 s or 20 s?
- **Plan.** Re-detect with the two neighbouring constant settings on a fixed 20 000-match subsample (as the earlier kill-less audit did), relabel with the frozen V̂, refit nothing but PT_flex knots on Q_SELECT, and report the primary contrast per setting; predeclare that this is a robustness check of the *unit*, not of the model.
- **Not allowed to claim.** That the definition is optimal, or that the prediction results generalise beyond the tested constants.

## F6. Information-set modules (thesis M-RQ3)

- The thesis plan defines M-F0 (p_pre + time), M-F1 (+ frame features), M-F2 (+ events/history), M-F3 (full). The journal has no information-set ablation (J-RQ2's "information set" clause is unsupported; see `.ai/RQ_DECISION_BRIEF.md`).
- **Plan (thesis only).** Nested models on identical rows and identical learner; paired ΔBrier between adjacent modules; predeclare that M-F0 is "C-only logistic" and not the PT baseline; forbidden: reading module gains as feature importance or mechanism.

## Deviation log (to be kept if any item runs)

| Date | Item | Deviation from the plan above | Reason | Decided by |
|---|---|---|---|---|
| — | — | — | — | — |
