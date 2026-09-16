# Horizon sensitivity: the five h90-selected learners refit on the h60 and h120 labels (T/N)

2026-09-16. Design, implementation, execution and audit: Claude (continuing the Codex line; user decision 2026-09-16 to
keep the current data and continue the validations). GPU (cuda) authorised for the MLP families. Exploratory follow-up
after prior TEST exposure; not an untouched confirmation.

## Why

`DELTA_Q_FAIR_BALANCED_PROTOCOL_20260915.md` section 3 step 6: "h60/h120에서는 h90에서 선정한 family별 hyperparameters를
고정한 채 해당 horizon의 TRAIN 라벨로 재학습하고, 동일 Q_CAL/Q_SELECT 규칙으로 보정만 선택한다. 그 적합도 TEST를 열기
전에 마친다. 균형 상태 분석은 h90 주 분석, 다른 상한은 보조 결과다." Both `incremental_q_training_20260915` (iq: PT,
full logistic, full LightGBM) and `track_a_mlp_20260916` (ta: plain MLP, residual MLP) deferred this step. The question is
whether the learner ranking found at h90 (LightGBM >= plain MLP ~ logistic > residual MLP > PT on T; LightGBM > MLPs >
logistic > PT on N) holds when the label endpoint cap is 60 s or 120 s. The h90 column itself is not refit.

## Scope and immutable parents

New root `outputs/horizon_sensitivity_20260916`; scripts prefix `hs20260916_`; tests `tests/test_hs20260916_contracts.py`.
Python `C:/Users/todtj/anaconda3/python.exe`; CPU threads <= 4 (LightGBM n_jobs 4), cuda for the MLPs; no installs,
deletes, commits or pushes. Read-only parents: full_corpus_training (features, h60/h120 labels, schema, V manifest),
cohort_role_training (cohorts with valid_h60/valid_h120, A_<set>_h60|h120_<cohort>.npz legacy predictions, results_A.json),
incremental_q_training (h90 winners and their configurations), track_a_mlp (h90 MLP winners and their configurations).
Pure classes `iq20260915_common.LinearQBase` / `LgbmQBase` and `ta20260916_common.MlpQBase` are reused unchanged; parent
writers are never called. Failures are preserved; stages resume safely.

## Fixed scientific contract

1. Same corpus and roles as iq/ta: TRAIN 15.14, VALIDATION 15.15 (Q_CAL / Q_SELECT), TEST 15.16, external KR_16.13,
   KR_16.14_pilot, KR_16.15, NA1_16.13 separately. Cohorts T / N from the parent cohort arrays. Rows: valid_h{h} and cohort;
   the parent cohort manifest gives identical counts at h60, h90 and h120 (TRAIN T 39,605 / N 159,753; Q_CAL 10,390 /
   40,565; Q_SELECT 10,195 / 40,180; TEST 32,981 / 130,595; externals as in iq). Any mismatch blocks fitting.
2. Labels Y_h60 = 1[delta_h60 > 0] and Y_h120 = 1[delta_h120 > 0] from the parent labels unchanged (endpoint
   e(h) = min(last kill + h, next kill - 1 ms, next engagement start - 1 ms, game end - 1 ms)); TRAIN p_pre / labels from the
   own-match-excluded OOF V (provenance verified). Inputs: the same 352 ridge columns (PT: p_pre_V and time_minutes only).
3. **Configurations are fixed at the h90 winners** and copied into protocol.json before any fit: PT and logistic C, LightGBM
   leaves / min_child_samples, MLP width / dropout, per cohort (iq: pt_C1 T/N; logit_C0.001 T, logit_C0.01 N; lgbm_L15_M100
   T/N; ta: mlp_W128_D0.3 T/N; resmlp_W128_D0.1 T/N). Internal stopping (LightGBM iterations, MLP epochs) is re-determined
   on the horizon's labels by the unchanged stop10 rule; it is part of the fitting procedure, not a validation choice.
4. Only the calibration (raw / sigmoid / isotonic, fit on cohort Q_CAL) is selected per family and horizon on cohort
   Q_SELECT by Brier, then log loss, then name. Three candidates per family; no selection inside subsets.
5. Equal total training weight per match in every fit subset (fc20260915_common.weights). Seeds 7, 42, 123 for LightGBM and
   the MLPs (raw mean before calibration); prediction definition for the MLPs as in Track A (CPU float64, one thread).

## Freeze, evaluation, comparators

One freeze covering both horizons (all 5 families x 2 cohorts x 2 horizons) before any TEST/external array or any parent
TEST prediction is opened; sealed access gated on THIS run's frozen manifest. The freeze records the iq / ta / cohort-role
frozen manifest hashes and the byte hashes of the cohort-role A_<set>_h{60,120}_<cohort>.npz files.

Per horizon, set and cohort: cells all, B40, B45, pre-time bins, B40 x time (frozen p_pre and pre time; identical across
horizons). Named models: the five refit winners (`pt_winner`, `logit_winner`, `lgbm_winner`, `mlp_winner`, `resmlp_winner`),
an `overall_winner` alias (same rule over the 15 joint candidates of the horizon and cohort), and the cohort-role legacy
references at that horizon (`old_A_specialist` = spec_<A_specialists[cohort_h{h}].chosen>, `old_pooled`, `old_p_pre_spline`,
`old_p_pre_logistic`, `old_constant`) joined by exact (match, s_ms) keys.

Predeclared contrasts (a - b; negative Brier / log loss = a better), paired match bootstrap 1,000 replicates, seed 20260915,
on all / B40 / B45 cells with >= 30 matches and both classes:

- PRIMARY: `lgbm_winner - pt_winner` (does the full-information tree keep its gain over the strengthened baseline at the
  other caps);
- planned: `logit_winner - pt_winner`; `lgbm_winner - logit_winner`; `mlp_winner - lgbm_winner`; `resmlp_winner -
  lgbm_winner`; `mlp_winner - logit_winner`.

The h90 values quoted next to h60 / h120 in the report come from the iq and ta evaluation results (same rows, same cells)
and are descriptive: labels differ between horizons, so no cross-horizon interval is computed.

## Gates and deliverables

Tests: per-horizon loader and counts, fixed-configuration registry equal to the parent frozen winners, three-candidate
calibration-only selection with ties, family dispatch to the reused classes, bundle reload identity, legacy join per
horizon, sealed gate keyed on this run's freeze, scripts writing only under this root, parent preservation. Order: integrity
snapshot -> contract tests -> protocol.json -> TRAIN-only smoke (h60, fit / freeze / evaluation on pseudo rows) ->
contract tests again -> full fits (h60 then h120) -> freeze -> sealed evaluation (both horizons) -> snapshot after ->
post-run checks -> Korean REPORT.md and DEFINITION_AND_EVIDENCE.md.

Save protocol.json, source hashes, command / environment / failure logs, status.json, registry, stop records, selections,
frozen manifest, bundles, raw / seed / calibrated predictions, metrics and paired intervals, validation.json with real
passed / failed checks. Keep an honest result if the ranking changes with the cap; do not choose a cap from TEST.
