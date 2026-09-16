# How the label works: exact column decomposition of Delta-logit of the frozen value model V (h90)

2026-09-16. Design, implementation, execution and audit: Claude (continuing the Codex line). No fitting. Exploratory
follow-up after prior TEST exposure. Answers item 2 of `COLLABORATOR_CRITIQUE_RESPONSE_20260915.md` section 7 ("주 V의
작동 원리: Δlogit 열별 합산, 시간 관련 항, 정적 항 상쇄") and the "V 작동 원리" gap in `TOG_END_TO_END_READINESS_AUDIT_20260915.md`.

## The identity being audited

The frozen final V is a scikit-learn Pipeline: ColumnTransformer (numeric block of 351 state columns, one-hot block of the
10 champion-id columns; 1,733 transformed columns) followed by LogisticRegression (C = 0.01) with the identity calibrator
(`selection_v.json`: raw). For an engagement row i with pre state S_pre and endpoint state S_e,

  Delta-logit_i = beta^T (t(S_e) - t(S_pre)) = sum_j c_ij,  c_ij = beta_j (t(S_e)_j - t(S_pre)_j),

and Y_h90 = 1[V(S_e) - V(S_pre) > 0] = 1[Delta-logit_i > 0] because the sigmoid is strictly monotone. The champion one-hot
columns are identical at both ends, so their c_ij are exactly zero (static terms cancel). Each transformed column maps to
one original state name (`numeric__<name>`) or to the champion block, so the 1,733 terms can be summed into the seven
groups of the balanced-SHAP stage (prior_win_probability is absent from V; champion_identity is a separate block expected
to be 0) and, separately, into "time-related" columns (time_minutes, time_minutes_sq and every `_x_time` interaction);
the time-related share is reported next to the group shares, never added to them (the classifications overlap).

## Scope and parents

New root `outputs/v_mechanism_20260916`; scripts prefix `vd20260916_`; tests `tests/test_vd20260916_contracts.py`. Read-only
parents: full_corpus_training (frozen `models/v/v_final_raw.joblib`, `extract/MAIN/states/chunk_*.npz` with `e_X_pre` and
`e_X_post_h90`, labels with p_pre / p_post_h90 / Y_h90 / snapshot timestamps), cohort_role_training (cohorts). Streaming
over the 1,050 chunks; nothing is refit; only per-row group sums, a few flags and per-column aggregates are kept.

## What is computed (all h90-valid rows, no sampling)

Sets: MAIN_VALIDATION (adapter `final`, not sealed; development pass and smoke on chunks 370-429, since the chunks are
ordered TRAIN, VALIDATION, TEST) and MAIN_TEST (sealed until this run's freeze; primary). For each row: Delta-logit; the eight block sums (7 groups + champion block); the time-related sum;
|Delta V| = |p_post - p_pre|; flags: cohort T / N, same-frame (post snapshot timestamp equal to the pre snapshot), pre-time
band, |Delta V| stratum (<= 0.5 pp, 0.5-1, 1-2, > 2 pp).

Reported per set, cohort and stratum: n; mean |Delta-logit|; mean |g_k| and share of sum |g_k| per group; the time-related
share of sum_j |c_j|; the sign-deciding group (largest |g_k| whose sign equals the sign of Delta-logit) frequencies;
the share of rows where one group alone exceeds |Delta-logit| (the others net to the opposite sign); the 25 transformed
columns with the largest mean |c_j| (name, group, mean signed c_j).

## Checks (all exact or < 1e-9)

p_pre and p_post_h90 recomputed from the stored states with the frozen adapter equal the parent label arrays exactly;
sum_k g_ik = Delta-logit_i; logit(p_post) - logit(p_pre) = Delta-logit; champion block = 0; sign(Delta-logit) = Y_h90 on
every valid row; rows joined to the parent labels and cohorts 1:1 by (match, s); counts equal the cohort manifest; parent
files unchanged (integrity snapshot); sealed access only after this run's freeze.

Not done here: V(S(L)) (the state at the last kill is not stored), so the pre -> L -> e split of the critique response
stays open; any causal reading of the terms; any change to V, labels or q.
