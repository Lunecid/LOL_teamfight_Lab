# Balanced-state explanations (RQ3): exact 7-group Shapley of the final common-input q on all-T versus B40-T

2026-09-16. Design, implementation, execution and audit: Claude (continuing the Codex line; user decision to keep the
current data and continue). No new fitting. Exploratory follow-up after prior TEST exposure; not an untouched confirmation.

## Why

`DELTA_Q_FAIR_BALANCED_PROTOCOL_20260915.md` RQ3 and section 6: "이전 256사례 SHAP 안에서 B40만 남겨 전체 균형 집단 설명이라고
부르지 않는다. 새 hash 기준으로 h90 T 전체와 B40에서 각각 최대 256개 사례를 정하고 겹치는 사례를 표시한다. ... 배경은 동일한
cohort TRAIN 최대 128사례로 고정한다. 공통-input 최종 선정 q는 기존 q_pre_only_schema.json의 shap_groups를 352열에 제한해
재사용한다: time_and_observation_age 2, health_mana_other 21, economy_and_experience 60, combat_and_survival 68, objectives 176,
structures 24, prior_win_probability 1개. ... 128 coalitions의 final calibrated probability group Shapley를 사용한다." The
collaboration e-mail draft promises SHAP explanations in the balanced subset. Track A and the horizon sensitivity closed
RQ1; this stage closes RQ3.

## Scope and immutable parents

New root `outputs/balanced_shap_20260916`; scripts prefix `bsh20260916_`; tests `tests/test_bsh20260916_contracts.py`.
Python `C:/Users/todtj/anaconda3/python.exe`, CPU only (torch threads 1 for the MLP prediction definition). Read-only
parents: full_corpus_training (features, h90 labels, schema with shap_groups), cohort_role_training (cohorts),
incremental_q_training (frozen LightGBM winners = the five-family Q_SELECT winners of both cohorts per the Track A freeze,
and their evaluation predictions), track_a_mlp (frozen plain-MLP winners and evaluation predictions). Reused pure code:
`fc20260915_shap.group_exact_shapley`, `iq20260915_common` / `ta20260916_common` bundle prediction and classes. Parent
writers are never called; failures preserved.

## Fixed contract

1. Explained models, h90: (a) `final_q` = the common-input final selected q per cohort, i.e. the iq LightGBM winner
   (T `lgbm_L15_M100__raw`, N `lgbm_L15_M100__sigmoid`), which the Track A freeze records as the five-family Q_SELECT winner
   in both cohorts; (b) `plain_mlp` = the Track A plain-MLP winner (T `mlp_W128_D0.3__raw`, N `mlp_W128_D0.3__sigmoid`),
   the neural learner that tied LightGBM on T. Bundles and calibrations are read from the parent frozen manifests and
   hash-checked; nothing is refit.
2. Explained quantity: the final selected calibrated probability q. Group value function v(S) = mean over background
   rows of f(x_S, background_notS); exact Shapley over 2^7 = 128 coalitions (`fc20260915_shap.group_exact_shapley`).
3. Groups: `shap_groups` of the parent schema restricted to the 352 ridge inputs, in the fixed order
   prior_win_probability (1), time_and_observation_age (2), economy_and_experience (60), combat_and_survival (68),
   objectives (176), structures (24), health_mana_other (21); partition asserted and its mapping hash frozen before TEST is
   opened. `unknown_objective_team_count` stays in health_mana_other as in the parent classification.
4. Cells: `all` = all h90-valid TEST rows of the cohort; `B40` = rows with 0.40 <= p_pre_V <= 0.60 (frozen p_pre only).
   Explained rows per cell: the first 256 by sha256('bsh20260916_shap_explain:<cohort>:<cell>:<match>:<s>'), independent
   of any model output; overlap between the two cells reported. Background per cohort: the first 128 cohort TRAIN h90-valid
   rows by sha256('bsh20260916_shap_background:<cohort>:<match>:<s>'), the same background for both cells and both models.
   T is primary; N is reported as secondary with the same rule.
5. Reported per model, cohort and cell: mean absolute and mean signed group contribution (unweighted and match-weighted),
   a 1,000-replicate row bootstrap (seed 20260915) of the mean absolute contributions (explanation-sample variability
   only), the rank order of groups, pre-time bands, and 8 hash-selected local cases (`bsh20260916_shap_case`) with y,
   q, base value, p_pre and the seven contributions. all-versus-B40 differences of mean absolute contributions are
   descriptive (different rows; no paired interval).
6. Checks: additivity phi.sum + base = f(x) (< 1e-8), full coalition = f(x), unique empty-coalition value, bitwise reload
   reproduction of the first three rows, deterministic row selection, and phi.sum + base equal to the frozen evaluation
   prediction of the same rows in the parent evaluation predictions joined by (match, s_ms) (< 1e-8). Sealed access to
   TEST and parent TEST predictions is gated on THIS run's frozen manifest (written after the protocol and the smoke).

Order: integrity snapshot -> contract tests -> protocol.json -> TRAIN-only smoke (pseudo rows, small sizes, same code)
-> contract tests again -> freeze (group map, bundles, hash tags) -> sealed explanation -> snapshot after -> post-run
checks -> Korean REPORT.md and DEFINITION_AND_EVIDENCE.md.

Interpretation limits (protocol section 6): interventional replacement creates artificial mixed states and breaks derived
relations (differences, x time, aggregates); contributions describe the fitted models on sampled rows, are not causal
win factors, and magnitudes of different models or cells are not ranked against each other beyond the descriptive tables.
