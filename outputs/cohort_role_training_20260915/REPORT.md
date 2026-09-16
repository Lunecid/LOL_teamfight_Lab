# Teamfight / non-teamfight q specialists and role-aware q representation (2026-09-15)

Generated 2026-09-15 15:02:14 KST from saved artifacts. Design: docs/CLAUDE_COHORT_ROLE_TRAIN_20260915.md (Codex). Implementation and execution: Claude Opus 5. Output root: outputs/cohort_role_training_20260915. Read-only source: outputs/full_corpus_training_20260915 (pre states, keys, labels, V, pooled q).

**Role of this work.** Exploratory follow-up on model-defined labels (Y = 1[V(endpoint) - V(q_pre) > 0], estimated probability improvement, not ground truth). TEST 15.16 and every external set were already evaluated for the pooled q in the completed full run (and earlier work), so all TEST numbers here are **frozen re-evaluations with prior exposure, not a fresh confirmatory test**. Every new choice (cohort specialists, role models, arms, ablations) was frozen before this task opened any TEST/external label. Cohort membership uses post-cutoff participation and is not a live input.

## Key results (h90 primary; all TEST numbers carry prior exposure)

1. **Teamfight specialist (A).** On the 32,981 identical TEST T rows the T specialist (ridge_raw) beats the pooled q: Brier -0.00166 [-0.00235, -0.00104], log loss -0.00465 [-0.00711, -0.00264], AUC +0.0061 [+0.0036, +0.0088]; its Q_SELECT Brier was also lower (0.229194 vs pooled 0.230738). The same direction holds at h60/h120 on TEST. On external sets at h90 the T differences are not significant (KR 16.13 point estimate favours the specialist; NA1 16.13 Brier point estimate favours pooled), and at h60 the NA1 16.13 T specialist is significantly worse in Brier and log loss, so the T gain does not transfer reliably to 16.xx.
2. **Non-teamfight specialist (A).** On TEST N rows: Brier -0.00042 [-0.00057, -0.00024], log loss -0.00003 [-0.00114, +0.00130] (n.s.), AUC +0.0016 [+0.0006, +0.0026]. This gain is small and not supported on validation: on N Q_SELECT the pooled q was better (0.242411 vs specialist 0.242560; diagnostic, not used for selection). The h60 N specialist (economic_sigmoid) is worse in AUC on TEST. Treat N specialization as not robust.
3. **Oracle-cohort routing** (post-cutoff membership, not deployable): TEST Brier -0.00051 [-0.00068, -0.00033] vs pooled.
4. **Role representation (C), N cohort.** With the same LightGBM base and calibration family, TEST N: role - participant Brier -0.00071 [-0.00098, -0.00046], draft control - participant -0.00017 [-0.00030, -0.00006], role - draft control -0.00054 [-0.00082, -0.00024]. On external N rows role - participant Brier is -0.00147 [-0.00217, -0.00076] in KR 16.13 (better) and -0.00161 [-0.00231, -0.00081] in NA1 16.13 (better); the two small sets are n.s. So role organization of the same participant information adds signal beyond slot order and beyond direct draft indicators for non-teamfight engagements, as ranked on Q_SELECT before TEST.
5. **Role representation (C), T cohort.** No TEST arm pair is significant (role - participant -0.00017 [-0.00056, +0.00025]); the overall T winner (role_lgbm_raw) does not beat the A specialist ridge on TEST (+0.00007 [-0.00045, +0.00065]).
6. **Model class.** The new full-feature LightGBM is the within-arm winner everywhere; in N the overall winner beats the A specialist ridge by -0.00350 [-0.00390, -0.00307] Brier on TEST, a mix of model class and representation effects.
7. **Role-block ablations (T, role ridge).** TEST Brier degradation when a block is removed: TOP +0.00005 [-0.00019, +0.00032]; JUNGLE +0.00062 [+0.00031, +0.00090]; MIDDLE +0.00054 [+0.00027, +0.00079]; BOTTOM +0.00027 [+0.00001, +0.00056]; UTILITY +0.00021 [-0.00006, +0.00048]. Removing TOP or UTILITY (and, negligibly, BOTTOM) lowered Q_SELECT Brier, and external results are inconsistent, so blocks are partly redundant with each other and with global totals; this is a model ablation, not causal lane importance.
8. **Role estimates.** Draft-only role posteriors agree with the weak TRAIN proxy at 0.979 (OOF; not accuracy). On independent external raw teamPosition (post-freeze) participant accuracy is KR_16.13 0.950, KR_16.14_pilot 0.917, KR_16.15 0.922, NA1_16.13 0.925; unseen champions are harder (section 5).
9. **Explanations (descriptive).** Mean |group Shapley| on the final probability: T role_TOP 0.0156, role_JUNGLE 0.0178, role_MIDDLE 0.0134, role_BOTTOM 0.0158, role_UTILITY 0.0107, global_context 0.0892, role_uncertainty 0.0009; N role_TOP 0.0113, role_JUNGLE 0.0247, role_MIDDLE 0.0170, role_BOTTOM 0.0314, role_UTILITY 0.0141, global_context 0.0290, role_uncertainty 0.0008. Global context dominates T; in N (more than half of the explained rows start before 10 min) the BOTTOM and JUNGLE role groups are comparable to global context (256-row descriptive sample per cohort).

## 1. Stage status

| Stage | State | Evidence |
|---|---|---|
| Read-only integrity snapshot before / after | executed; verified (unchanged) | integrity/snapshot_before.json, snapshot_after.json, snapshot_diff.json |
| A1 scale restoration + cohort manifest (before fitting) | executed (2026-09-15 14:26:26); verified | cohorts/cohort_manifest.json |
| B0 draft fields + weak role-supervision provenance (before fitting) | executed (2026-09-15 14:31:00) | draft/draft_manifest.json, role_supervision_provenance.json |
| Contract tests (TRAIN-only / synthetic) | run 1 failed (5 fixture/code defects fixed), run 2 and run 3 passed (34 tests, gate run 3 at 2026-09-15 14:42:27) | contract_tests/run1-3.txt, result.json |
| Protocol (predeclared rules and pairs) | executed (2026-09-15 14:35:41) | protocol.json |
| TRAIN-only smoke runs (1/8 TRAIN match subset, pseudo roles; code paths only) | executed; run 1 of arms failed (ridge reload identity), fixed | smoke_train_only/, logs/smoke_* |
| A2 T/N specialist q (h90/h60/h120) | executed | selection/q_specialist_*.json |
| B1 role models (final + 5 OOF) | executed; converged | role_models_manifest.json |
| C arms (T, N; h90) + T ablations | executed | selection/arms_*_h90.json, ablations_T_h90.json |
| Freeze | executed (2026-09-15 14:47:42); 123 files hashed | frozen_manifest.json |
| Frozen evaluation (TEST + 4 external sets) | executed; frozen hashes unchanged | eval/results_A.json, results_C.json, hashes_*.json |
| Role reliability vs external raw teamPosition (post-freeze) | executed | eval/role_reliability_external_raw.json |
| Role-arm group SHAP | executed; checks pass | shap/shap_summary.json |
| Post-freeze ridge optimizer diagnostic (TRAIN/VALIDATION only) | executed | diagnostics/ridge_optimizer_path_diagnostic.json |
| Post-run verification | pass (42 checks) | validation.json |

## 2. Scale definition restored: v3.3 participation, teamfight cut 4

Authoritative scale block (D:/LOL_Project/fusion_2615/corpus_shards_v33/manifest.json): `{'pick_max': 1, 'skirmish_min': 2, 'teamfight_min': 4}`; detector `{'TF2_KILL_CLUSTER_GAP_MS': 13700, 'CLUSTER_MAX_DIAMETER': 4264.0, 'TF2_VALIDITY_RADIUS': 1600.0, 'TF2_ENGAGE_PRE_KILL_MS': 15000, 'FIGHT_HORIZON_SEC': 35}` (G 13.7 s, D 4,264 u, R 1,600 u, B 15 s; M 2). The old ENGAGEMENT_SCALE_DEFINITION.md (v2, cut 3) was not used.

* **Participation count** = stored `cluster_blue` / `cluster_red` (gameplay/fights.py `det_cluster_*`): champions among the killers, victims and assisters of the engagement's kills plus actors of other timeline events within 3,000 u of the anchor between the start and the last kill (+ tail), structure/monster events excluded, shop events included (`TF2_EXCLUDE_SHOP_INTERACTIONS` False), position-less events at the actor's interpolated position; a merged engagement keeps its earlier candidate's counts. It is **not** the pre-cutoff presence count (`present_*`, diagnostic only).
* **Recovery:** all 32 shards joined by exact (match, engage_ts = s) key to every frozen label row: 566,452 of 566,452 shard rows used exactly once, 566,452 label rows, valid rows h60/h90/h120 = 566,104; label (s, L) equal the parent exposure (s, L) for every match; no shard omission, so no main-corpus re-detection was needed for counts. Old market_event labels, y and X were not read.
* **Negative counts:** 59 main rows carry -1. `_fight_to_ref_row` stores `int(count or -1)`. The frozen detector was re-run on those 59 matches: every row re-detected once at the same (s, L) with raw integer count 0 on the -1 side (59 proven, 0 unknown), so -1 is read as 0. External sets: EXT_KR_16.13 1 (proven 1), EXT_KR_16.14_pilot 0 (proven 0), EXT_KR_16.15 0 (proven 0), EXT_NA1_16.13 5 (proven 5). No unknown-scale row remains in any set; the unknown category is implemented but empty.
* **Detector checks (TRAIN-only fixture):** 400 hash-selected 15.14 matches re-detected: exposure tuples equal for 400/400 matches; participation and presence counts equal the shards for 1070/1070 engagements; settings {'TF2_KILL_CLUSTER_GAP_MS': 13700, 'CLUSTER_MAX_DIAMETER': 4264.0, 'TF2_VALIDITY_RADIUS': 1600.0, 'TF2_ENGAGE_PRE_KILL_MS': 15000, 'TF2_MIN_PER_TEAM': 2, 'TF2_INTERACTION_RADIUS': 3000.0, 'TF2_EXCLUDE_SHOP_INTERACTIONS': False, 'FIGHT_HORIZON_SEC': 35, 'FEATURE_VERSION': 'featV7_schema_pruned_status_runes_bans_spells_styles_bin5s'}.
* **External sets:** the same detector re-run on the already-adapted caches regenerated the participation fields; exposure tuples equal the stored external exposures bitwise and label keys equal the re-detected keys for all four sets (EXT_KR_16.13: 10064 matches, 25,443 rows, EXT_KR_16.14_pilot: 200 matches, 475 rows, EXT_KR_16.15: 926 matches, 2,219 rows, EXT_NA1_16.13: 10000 matches, 26,549 rows).

E = all engagement rows; T = {min(blue, red) >= 4}; N = E \ T (known scale). T and N are disjoint and their union is the known-scale E in every set.

| Set / role | rows | valid h90 | T (valid) | N (valid) | pick <=1 (valid) | skirmish 2-3 (valid) | T share | cut 3 teamfight (DIAG, all rows) | cut 5 (DIAG) | presence min >= 4 (DIAG) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MAIN_TRAIN fold0 | 39,219 | 39,197 | 7,870 | 31,327 | 7,297 | 24,030 | 0.201 | | | |
| MAIN_TRAIN fold1 | 40,202 | 40,167 | 7,850 | 32,317 | 7,312 | 25,005 | 0.195 | | | |
| MAIN_TRAIN fold2 | 39,796 | 39,777 | 7,860 | 31,917 | 7,397 | 24,520 | 0.198 | | | |
| MAIN_TRAIN fold3 | 40,005 | 39,980 | 8,037 | 31,943 | 7,253 | 24,690 | 0.201 | | | |
| MAIN_TRAIN fold4 | 40,258 | 40,237 | 7,988 | 32,249 | 7,445 | 24,804 | 0.199 | | | |
| **MAIN_TRAIN total** | 199,480 | | 39,605 (all rows) | 159,875 | 36,774 | 123,101 | 0.199 | 79,571 | 13,627 | 3,993 |
| MAIN_VALIDATION Q_CAL | 50,981 | 50,955 | 10,390 | 40,565 | 9,263 | 31,302 | 0.204 | | | |
| MAIN_VALIDATION Q_SELECT | 50,412 | 50,375 | 10,195 | 40,180 | 9,121 | 31,059 | 0.202 | | | |
| MAIN_VALIDATION V_CAL | 50,397 | 50,372 | 10,191 | 40,181 | 9,199 | 30,982 | 0.202 | | | |
| MAIN_VALIDATION V_SELECT | 51,502 | 51,468 | 10,539 | 40,929 | 9,413 | 31,516 | 0.205 | | | |
| **MAIN_VALIDATION total** | 203,292 | | 41,315 (all rows) | 161,977 | 37,072 | 124,905 | 0.203 | 82,455 | 14,502 | 4,439 |
| MAIN_TEST TEST | 163,680 | 163,576 | 32,981 | 130,595 | 29,390 | 101,205 | 0.202 | | | |
| **MAIN_TEST total** | 163,680 | | 32,981 (all rows) | 130,699 | 29,457 | 101,242 | 0.201 | 66,089 | 11,426 | 3,357 |
| EXT_KR_16.13 EXTERNAL | 25,443 | 25,430 | 5,202 | 20,228 | 4,587 | 15,641 | 0.205 | | | |
| **EXT_KR_16.13 total** | 25,443 | | 5,202 (all rows) | 20,241 | 4,597 | 15,644 | 0.204 | 10,677 | 1,724 | 542 |
| EXT_KR_16.14_pilot EXTERNAL | 475 | 475 | 101 | 374 | 89 | 285 | 0.213 | | | |
| **EXT_KR_16.14_pilot total** | 475 | | 101 (all rows) | 374 | 89 | 285 | 0.213 | 185 | 28 | 11 |
| EXT_KR_16.15 EXTERNAL | 2,219 | 2,219 | 507 | 1,712 | 405 | 1,307 | 0.228 | | | |
| **EXT_KR_16.15 total** | 2,219 | | 507 (all rows) | 1,712 | 405 | 1,307 | 0.228 | 973 | 160 | 48 |
| EXT_NA1_16.13 EXTERNAL | 26,549 | 26,538 | 5,312 | 21,226 | 5,126 | 16,100 | 0.200 | | | |
| **EXT_NA1_16.13 total** | 26,549 | | 5,312 (all rows) | 21,237 | 5,133 | 16,104 | 0.200 | 10,754 | 1,738 | 569 |

Main corpus at cut 4: 113,901 teamfight rows of 566,452 detected engagements. The manuscript's 109,829 teamfights refer to the market_event-labelled population (532,547 rows, draws dropped); the deltaV population here is every detected engagement, so the counts differ by construction and were not assumed equal. Cut 3 / cut 5 and presence counts are diagnostics only; the main cut stays 4.

## 3. A: independent T and N q pools (selection on cohort Q_SELECT, before TEST)

Candidate pool, parameters and the rule (match-weighted Q_SELECT Brier, then log loss, then name) are the unchanged full-run ones (`fc20260915_fit_q.fit_candidates`); fits use all eligible cohort TRAIN rows (held-out-fold labels, p_pre_V), calibrators the cohort Q_CAL rows. Pooled reference = frozen full-run choice (h90 ridge_isotonic, h60 ridge_sigmoid, h120 ridge_isotonic).

**h90** Q_SELECT Brier (chosen *):

| candidate | T | N |
|---|---:|---:|
| constant | 0.249981 | 0.249959 |
| p_pre_logistic | 0.231175 | 0.246614 |
| p_pre_spline | 0.230661 | 0.245691 |
| ridge_raw | 0.229194 * | 0.242621 |
| ridge_sigmoid | 0.229239 | 0.242627 |
| ridge_isotonic | 0.229614 | 0.242560 * |
| economic_raw | 0.230411 | 0.243005 |
| economic_sigmoid | 0.230412 | 0.242788 |
| economic_isotonic | 0.231380 | 0.242983 |
| pooled frozen q on the same rows (diagnostic, not a candidate) | 0.230738 | 0.242411 |
| TRAIN / Q_CAL / Q_SELECT rows | 39,605 / 10,390 / 10,195 | 159,753 / 40,565 / 40,180 |

**h60** Q_SELECT Brier (chosen *):

| candidate | T | N |
|---|---:|---:|
| constant | 0.249991 | 0.249950 |
| p_pre_logistic | 0.230490 | 0.246466 |
| p_pre_spline | 0.229909 | 0.245566 |
| ridge_raw | 0.228734 * | 0.242688 |
| ridge_sigmoid | 0.228792 | 0.242696 |
| ridge_isotonic | 0.229284 | 0.242787 |
| economic_raw | 0.229756 | 0.242846 |
| economic_sigmoid | 0.229746 | 0.242591 * |
| economic_isotonic | 0.230995 | 0.242617 |
| pooled frozen q on the same rows (diagnostic, not a candidate) | 0.229861 | 0.242519 |
| TRAIN / Q_CAL / Q_SELECT rows | 39,605 / 10,390 / 10,195 | 159,753 / 40,565 / 40,180 |

**h120** Q_SELECT Brier (chosen *):

| candidate | T | N |
|---|---:|---:|
| constant | 0.249979 | 0.249960 |
| p_pre_logistic | 0.231185 | 0.246565 |
| p_pre_spline | 0.230654 | 0.245622 |
| ridge_raw | 0.229082 * | 0.242580 |
| ridge_sigmoid | 0.229133 | 0.242586 |
| ridge_isotonic | 0.229508 | 0.242504 * |
| economic_raw | 0.230351 | 0.242994 |
| economic_sigmoid | 0.230367 | 0.242781 |
| economic_isotonic | 0.231306 | 0.243054 |
| pooled frozen q on the same rows (diagnostic, not a candidate) | 0.230680 | 0.242337 |
| TRAIN / Q_CAL / Q_SELECT rows | 39,605 / 10,390 / 10,195 | 159,753 / 40,565 / 40,180 |

## 4. A: pooled vs specialist on identical rows (frozen; equal match weights; paired match bootstrap 1000)

Differences are specialist minus pooled (negative Brier / log loss = specialist better; positive AUC = specialist better). Rows of different cohorts are never compared as an improvement.

### h90

| Set | cohort | rows (matches) | spec. | pooled AUC / Brier / LL | specialist AUC / Brier / LL | dBrier [95% CI] | dLogLoss [95% CI] | dAUC [95% CI] | cal. slope pooled / spec. | ECE pooled / spec. | exact 0/1 (opp.) pooled / spec. |
|---|---|---:|---|---|---|---|---|---|---|---|---|
| MAIN_TEST | T | 32,981 (24,020) | ridge_raw | 0.6641 / 0.2298 / 0.6528 | 0.6702 / 0.2282 / 0.6481 | -0.00166 [-0.00235, -0.00104] | -0.00465 [-0.00711, -0.00264] | +0.0061 [+0.0036, +0.0088] | 1.049 / 1.002 | 0.0090 / 0.0045 | 6 (2) / 0 (0) |
| MAIN_TEST | N | 130,595 (54,182) | ridge_isotonic | 0.5968 / 0.2431 / 0.6796 | 0.5984 / 0.2427 / 0.6796 | -0.00042 [-0.00057, -0.00024] | -0.00003 [-0.00114, +0.00130] | +0.0016 [+0.0006, +0.0026] | 0.865 / 0.912 | 0.0079 / 0.0038 | 8 (2) / 12 (7) |
| EXT_KR_16.13 | T | 5,202 (3,859) | ridge_raw | 0.6307 / 0.2385 / 0.6727 | 0.6385 / 0.2375 / 0.6690 | -0.00096 [-0.00299, +0.00123] | -0.00369 [-0.01182, +0.00312] | +0.0078 [-0.0012, +0.0163] | 0.834 / 0.790 | 0.0468 / 0.0341 | 1 (1) / 0 (0) |
| EXT_KR_16.13 | N | 20,228 (8,642) | ridge_isotonic | 0.5881 / 0.2449 / 0.6831 | 0.5880 / 0.2447 / 0.6839 | -0.00023 [-0.00084, +0.00036] | +0.00079 [-0.00150, +0.00412] | -0.0001 [-0.0034, +0.0034] | 0.750 / 0.754 | 0.0231 / 0.0205 | 0 (0) / 2 (1) |
| EXT_KR_16.14_pilot | T | 101 (80) | ridge_raw | 0.6514 / 0.2339 / 0.6616 | 0.6585 / 0.2317 / 0.6570 | -0.00223 [-0.01585, +0.01137] | -0.00458 [-0.03541, +0.02567] | +0.0071 [-0.0487, +0.0617] | 0.931 / 0.918 | 0.0628 / 0.0588 | 0 (0) / 0 (0) |
| EXT_KR_16.14_pilot | N | 374 (165) | ridge_isotonic | 0.5562 / 0.2494 / 0.6922 | 0.5722 / 0.2464 / 0.6862 | -0.00309 [-0.00767, +0.00097] | -0.00608 [-0.01562, +0.00244] | +0.0160 [-0.0054, +0.0411] | 0.484 / 0.640 | 0.0492 / 0.0437 | 0 (0) / 0 (0) |
| EXT_KR_16.15 | T | 507 (377) | ridge_raw | 0.6099 / 0.2399 / 0.6725 | 0.6109 / 0.2403 / 0.6734 | +0.00038 [-0.00604, +0.00641] | +0.00091 [-0.01308, +0.01394] | +0.0010 [-0.0244, +0.0290] | 0.748 / 0.687 | 0.0391 / 0.0461 | 0 (0) / 0 (0) |
| EXT_KR_16.15 | N | 1,712 (764) | ridge_isotonic | 0.6022 / 0.2433 / 0.6798 | 0.6122 / 0.2407 / 0.6745 | -0.00255 [-0.00468, -0.00028] | -0.00527 [-0.00968, -0.00053] | +0.0100 [-0.0027, +0.0226] | 0.834 / 0.998 | 0.0246 / 0.0258 | 0 (0) / 0 (0) |
| EXT_NA1_16.13 | T | 5,312 (3,955) | ridge_raw | 0.6117 / 0.2436 / 0.6937 | 0.6136 / 0.2457 / 0.6877 | +0.00204 [-0.00024, +0.00432] | -0.00607 [-0.02260, +0.00757] | +0.0020 [-0.0066, +0.0107] | 0.604 / 0.592 | 0.0506 / 0.0591 | 7 (3) / 0 (0) |
| EXT_NA1_16.13 | N | 21,226 (8,753) | ridge_isotonic | 0.5912 / 0.2451 / 0.6878 | 0.5904 / 0.2446 / 0.6842 | -0.00050 [-0.00109, +0.00016] | -0.00366 [-0.00935, +0.00108] | -0.0008 [-0.0044, +0.0024] | 0.657 / 0.723 | 0.0208 / 0.0168 | 13 (5) / 12 (2) |

Inside N (N specialist vs pooled on the same pick or skirmish rows):

| Set | subset | rows | pooled Brier | N specialist Brier | dBrier [95% CI] | dLogLoss [95% CI] |
|---|---|---:|---:|---:|---|---|
| MAIN_TEST | pick | 29,390 | 0.2443 | 0.2435 | -0.00083 [-0.00115, -0.00052] | -0.00059 [-0.00215, +0.00156] |
| MAIN_TEST | skirmish | 101,205 | 0.2429 | 0.2427 | -0.00023 [-0.00043, -0.00004] | +0.00018 [-0.00115, +0.00163] |
| EXT_KR_16.13 | pick | 4,587 | 0.2462 | 0.2450 | -0.00123 [-0.00246, -0.00004] | -0.00247 [-0.00505, +0.00005] |
| EXT_KR_16.13 | skirmish | 15,641 | 0.2452 | 0.2451 | -0.00008 [-0.00071, +0.00054] | +0.00113 [-0.00147, +0.00482] |
| EXT_KR_16.14_pilot | pick | 89 | 0.2632 | 0.2551 | -0.00804 [-0.01484, -0.00128] | -0.01820 [-0.03368, -0.00285] |
| EXT_KR_16.14_pilot | skirmish | 285 | 0.2444 | 0.2427 | -0.00175 [-0.00682, +0.00267] | -0.00269 [-0.01324, +0.00691] |
| EXT_KR_16.15 | pick | 405 | 0.2435 | 0.2356 | -0.00796 [-0.01200, -0.00436] | -0.01629 [-0.02450, -0.00874] |
| EXT_KR_16.15 | skirmish | 1,307 | 0.2436 | 0.2428 | -0.00082 [-0.00320, +0.00176] | -0.00184 [-0.00700, +0.00347] |
| EXT_NA1_16.13 | pick | 5,126 | 0.2463 | 0.2443 | -0.00203 [-0.00319, -0.00076] | -0.01276 [-0.03125, -0.00232] |
| EXT_NA1_16.13 | skirmish | 16,100 | 0.2453 | 0.2451 | -0.00018 [-0.00093, +0.00060] | -0.00347 [-0.00985, +0.00176] |

Oracle-cohort routing (**membership needs post-cutoff participation; NOT a deployable live model**) vs pooled on all known rows:

| Set | rows | routed Brier | pooled Brier | dBrier [95% CI] | dLogLoss [95% CI] | pooled E baseline reproduces full run |
|---|---:|---:|---:|---|---|---|
| MAIN_TEST | 163,576 | 0.2394 | 0.2400 | -0.00051 [-0.00068, -0.00033] | -0.00080 [-0.00170, +0.00005] | True |
| EXT_KR_16.13 | 25,430 | 0.2422 | 0.2426 | -0.00038 [-0.00095, +0.00032] | -0.00022 [-0.00236, +0.00256] | True |
| EXT_KR_16.14_pilot | 475 | 0.2445 | 0.2480 | -0.00344 [-0.00825, +0.00119] | -0.00686 [-0.01698, +0.00282] | True |
| EXT_KR_16.15 | 2,219 | 0.2408 | 0.2428 | -0.00197 [-0.00415, +0.00033] | -0.00408 [-0.00871, +0.00075] | True |
| EXT_NA1_16.13 | 26,538 | 0.2440 | 0.2440 | -0.00003 [-0.00067, +0.00063] | -0.00349 [-0.00868, +0.00143] | True |

### h60

| Set | cohort | rows (matches) | spec. | pooled AUC / Brier / LL | specialist AUC / Brier / LL | dBrier [95% CI] | dLogLoss [95% CI] | dAUC [95% CI] | cal. slope pooled / spec. | ECE pooled / spec. | exact 0/1 (opp.) pooled / spec. |
|---|---|---:|---|---|---|---|---|---|---|---|---|
| MAIN_TEST | T | 32,981 (24,020) | ridge_raw | 0.6665 / 0.2292 / 0.6502 | 0.6722 / 0.2277 / 0.6470 | -0.00154 [-0.00219, -0.00093] | -0.00318 [-0.00459, -0.00183] | +0.0057 [+0.0031, +0.0083] | 1.056 / 1.014 | 0.0132 / 0.0074 | 0 (0) / 0 (0) |
| MAIN_TEST | N | 130,595 (54,182) | economic_sigmoid | 0.5962 / 0.2433 / 0.6798 | 0.5915 / 0.2435 / 0.6799 | +0.00014 [-0.00019, +0.00049] | +0.00011 [-0.00059, +0.00082] | -0.0047 [-0.0071, -0.0020] | 0.891 / 0.958 | 0.0142 / 0.0039 | 0 (0) / 0 (0) |
| EXT_KR_16.13 | T | 5,202 (3,859) | ridge_raw | 0.6325 / 0.2374 / 0.6677 | 0.6409 / 0.2371 / 0.6681 | -0.00034 [-0.00227, +0.00178] | +0.00039 [-0.00388, +0.00511] | +0.0084 [-0.0002, +0.0166] | 0.874 / 0.796 | 0.0321 / 0.0368 | 0 (0) / 0 (0) |
| EXT_KR_16.13 | N | 20,228 (8,642) | economic_sigmoid | 0.5880 / 0.2451 / 0.6835 | 0.5806 / 0.2451 / 0.6833 | +0.00006 [-0.00083, +0.00093] | -0.00019 [-0.00205, +0.00168] | -0.0074 [-0.0135, -0.0014] | 0.771 / 0.842 | 0.0252 / 0.0159 | 0 (0) / 0 (0) |
| EXT_KR_16.14_pilot | T | 101 (80) | ridge_raw | 0.6480 / 0.2338 / 0.6605 | 0.6557 / 0.2331 / 0.6601 | -0.00069 [-0.01534, +0.01484] | -0.00034 [-0.03420, +0.03348] | +0.0077 [-0.0505, +0.0651] | 0.937 / 0.867 | 0.0863 / 0.0552 | 0 (0) / 0 (0) |
| EXT_KR_16.14_pilot | N | 374 (165) | economic_sigmoid | 0.5534 / 0.2496 / 0.6926 | 0.5529 / 0.2469 / 0.6867 | -0.00276 [-0.00966, +0.00398] | -0.00595 [-0.02059, +0.00806] | -0.0005 [-0.0469, +0.0480] | 0.480 / 0.679 | 0.0361 / 0.0507 | 0 (0) / 0 (0) |
| EXT_KR_16.15 | T | 507 (377) | ridge_raw | 0.6126 / 0.2395 / 0.6715 | 0.6147 / 0.2400 / 0.6729 | +0.00055 [-0.00571, +0.00641] | +0.00139 [-0.01210, +0.01444] | +0.0021 [-0.0238, +0.0296] | 0.764 / 0.700 | 0.0311 / 0.0387 | 0 (0) / 0 (0) |
| EXT_KR_16.15 | N | 1,712 (764) | economic_sigmoid | 0.5978 / 0.2439 / 0.6810 | 0.5676 / 0.2472 / 0.6876 | +0.00329 [-0.00007, +0.00640] | +0.00666 [-0.00050, +0.01314] | -0.0302 [-0.0522, -0.0075] | 0.835 / 0.685 | 0.0349 / 0.0195 | 0 (0) / 0 (0) |
| EXT_NA1_16.13 | T | 5,312 (3,955) | ridge_raw | 0.6162 / 0.2429 / 0.6800 | 0.6175 / 0.2453 / 0.6871 | +0.00233 [+0.00014, +0.00460] | +0.00710 [+0.00208, +0.01233] | +0.0013 [-0.0069, +0.0096] | 0.698 / 0.603 | 0.0524 / 0.0599 | 0 (0) / 0 (0) |
| EXT_NA1_16.13 | N | 21,226 (8,753) | economic_sigmoid | 0.5913 / 0.2451 / 0.6838 | 0.5812 / 0.2454 / 0.6838 | +0.00027 [-0.00067, +0.00125] | +0.00003 [-0.00206, +0.00212] | -0.0101 [-0.0163, -0.0040] | 0.716 / 0.796 | 0.0213 / 0.0133 | 0 (0) / 0 (0) |

Inside N (N specialist vs pooled on the same pick or skirmish rows):

| Set | subset | rows | pooled Brier | N specialist Brier | dBrier [95% CI] | dLogLoss [95% CI] |
|---|---|---:|---:|---:|---|---|
| MAIN_TEST | pick | 29,390 | 0.2446 | 0.2441 | -0.00047 [-0.00116, +0.00022] | -0.00131 [-0.00277, +0.00017] |
| MAIN_TEST | skirmish | 101,205 | 0.2431 | 0.2436 | +0.00049 [+0.00009, +0.00087] | +0.00086 [+0.00002, +0.00164] |
| EXT_KR_16.13 | pick | 4,587 | 0.2463 | 0.2460 | -0.00032 [-0.00207, +0.00135] | -0.00082 [-0.00451, +0.00269] |
| EXT_KR_16.13 | skirmish | 15,641 | 0.2453 | 0.2454 | +0.00010 [-0.00092, +0.00110] | -0.00023 [-0.00243, +0.00196] |
| EXT_KR_16.14_pilot | pick | 89 | 0.2602 | 0.2581 | -0.00211 [-0.01641, +0.01103] | -0.00564 [-0.03585, +0.02196] |
| EXT_KR_16.14_pilot | skirmish | 285 | 0.2460 | 0.2424 | -0.00366 [-0.01142, +0.00337] | -0.00737 [-0.02382, +0.00682] |
| EXT_KR_16.15 | pick | 405 | 0.2434 | 0.2433 | -0.00008 [-0.00547, +0.00542] | +0.00037 [-0.01090, +0.01174] |
| EXT_KR_16.15 | skirmish | 1,307 | 0.2444 | 0.2489 | +0.00451 [+0.00091, +0.00802] | +0.00877 [+0.00113, +0.01612] |
| EXT_NA1_16.13 | pick | 5,126 | 0.2463 | 0.2471 | +0.00087 [-0.00090, +0.00268] | +0.00137 [-0.00252, +0.00532] |
| EXT_NA1_16.13 | skirmish | 16,100 | 0.2451 | 0.2452 | +0.00008 [-0.00101, +0.00119] | -0.00042 [-0.00279, +0.00189] |

Oracle-cohort routing (**membership needs post-cutoff participation; NOT a deployable live model**) vs pooled on all known rows:

| Set | rows | routed Brier | pooled Brier | dBrier [95% CI] | dLogLoss [95% CI] | pooled E baseline reproduces full run |
|---|---:|---:|---:|---|---|---|
| MAIN_TEST | 163,576 | 0.2400 | 0.2401 | -0.00006 [-0.00038, +0.00024] | -0.00023 [-0.00092, +0.00039] | True |
| EXT_KR_16.13 | 25,430 | 0.2424 | 0.2424 | -0.00001 [-0.00087, +0.00076] | -0.00007 [-0.00197, +0.00154] | True |
| EXT_KR_16.14_pilot | 475 | 0.2451 | 0.2482 | -0.00304 [-0.00998, +0.00367] | -0.00639 [-0.02163, +0.00818] | True |
| EXT_KR_16.15 | 2,219 | 0.2457 | 0.2430 | +0.00271 [-0.00002, +0.00553] | +0.00554 [-0.00016, +0.01146] | True |
| EXT_NA1_16.13 | 26,538 | 0.2448 | 0.2441 | +0.00078 [-0.00004, +0.00166] | +0.00160 [-0.00021, +0.00352] | True |

### h120

| Set | cohort | rows (matches) | spec. | pooled AUC / Brier / LL | specialist AUC / Brier / LL | dBrier [95% CI] | dLogLoss [95% CI] | dAUC [95% CI] | cal. slope pooled / spec. | ECE pooled / spec. | exact 0/1 (opp.) pooled / spec. |
|---|---|---:|---|---|---|---|---|---|---|---|---|
| MAIN_TEST | T | 32,981 (24,020) | ridge_raw | 0.6645 / 0.2297 / 0.6524 | 0.6704 / 0.2281 / 0.6481 | -0.00154 [-0.00220, -0.00090] | -0.00435 [-0.00686, -0.00233] | +0.0059 [+0.0034, +0.0085] | 1.049 / 0.992 | 0.0088 / 0.0041 | 6 (2) / 0 (0) |
| MAIN_TEST | N | 130,595 (54,182) | ridge_isotonic | 0.5978 / 0.2429 / 0.6794 | 0.5991 / 0.2425 / 0.6793 | -0.00035 [-0.00051, -0.00018] | -0.00010 [-0.00116, +0.00123] | +0.0013 [+0.0003, +0.0023] | 0.871 / 0.911 | 0.0069 / 0.0045 | 9 (3) / 9 (6) |
| EXT_KR_16.13 | T | 5,202 (3,859) | ridge_raw | 0.6270 / 0.2394 / 0.6778 | 0.6358 / 0.2383 / 0.6709 | -0.00116 [-0.00327, +0.00108] | -0.00688 [-0.01764, +0.00154] | +0.0088 [-0.0001, +0.0174] | 0.794 / 0.761 | 0.0444 / 0.0343 | 2 (2) / 0 (0) |
| EXT_KR_16.13 | N | 20,228 (8,642) | ridge_isotonic | 0.5879 / 0.2451 / 0.6836 | 0.5879 / 0.2448 / 0.6828 | -0.00033 [-0.00095, +0.00027] | -0.00077 [-0.00206, +0.00045] | +0.0000 [-0.0034, +0.0034] | 0.741 / 0.763 | 0.0231 / 0.0195 | 0 (0) / 1 (0) |
| EXT_KR_16.14_pilot | T | 101 (80) | ridge_raw | 0.6462 / 0.2368 / 0.6673 | 0.6620 / 0.2322 / 0.6585 | -0.00460 [-0.02054, +0.01097] | -0.00887 [-0.04398, +0.02602] | +0.0158 [-0.0483, +0.0774] | 0.859 / 0.902 | 0.0781 / 0.0537 | 0 (0) / 0 (0) |
| EXT_KR_16.14_pilot | N | 374 (165) | ridge_isotonic | 0.5525 / 0.2498 / 0.6930 | 0.5731 / 0.2474 / 0.6884 | -0.00244 [-0.00736, +0.00159] | -0.00457 [-0.01467, +0.00394] | +0.0207 [-0.0028, +0.0488] | 0.467 / 0.579 | 0.0462 / 0.0449 | 0 (0) / 0 (0) |
| EXT_KR_16.15 | T | 507 (377) | ridge_raw | 0.6031 / 0.2418 / 0.6762 | 0.6079 / 0.2414 / 0.6760 | -0.00036 [-0.00717, +0.00619] | -0.00028 [-0.01478, +0.01374] | +0.0048 [-0.0221, +0.0322] | 0.683 / 0.656 | 0.0397 / 0.0540 | 1 (0) / 0 (0) |
| EXT_KR_16.15 | N | 1,712 (764) | ridge_isotonic | 0.6012 / 0.2433 / 0.6798 | 0.6091 / 0.2416 / 0.6764 | -0.00165 [-0.00366, +0.00053] | -0.00337 [-0.00758, +0.00105] | +0.0079 [-0.0041, +0.0194] | 0.835 / 0.941 | 0.0269 / 0.0216 | 0 (0) / 0 (0) |
| EXT_NA1_16.13 | T | 5,312 (3,955) | ridge_raw | 0.6099 / 0.2443 / 0.6954 | 0.6138 / 0.2458 / 0.6877 | +0.00144 [-0.00082, +0.00367] | -0.00763 [-0.02408, +0.00644] | +0.0039 [-0.0048, +0.0127] | 0.596 / 0.590 | 0.0526 / 0.0582 | 7 (3) / 0 (0) |
| EXT_NA1_16.13 | N | 21,226 (8,753) | ridge_isotonic | 0.5916 / 0.2450 / 0.6876 | 0.5904 / 0.2446 / 0.6826 | -0.00038 [-0.00097, +0.00026] | -0.00504 [-0.01051, -0.00103] | -0.0012 [-0.0047, +0.0021] | 0.664 / 0.740 | 0.0204 / 0.0188 | 12 (5) / 5 (0) |

Inside N (N specialist vs pooled on the same pick or skirmish rows):

| Set | subset | rows | pooled Brier | N specialist Brier | dBrier [95% CI] | dLogLoss [95% CI] |
|---|---|---:|---:|---:|---|---|
| MAIN_TEST | pick | 29,390 | 0.2441 | 0.2433 | -0.00083 [-0.00118, -0.00049] | -0.00132 [-0.00237, +0.00005] |
| MAIN_TEST | skirmish | 101,205 | 0.2427 | 0.2426 | -0.00016 [-0.00036, +0.00003] | +0.00023 [-0.00102, +0.00161] |
| EXT_KR_16.13 | pick | 4,587 | 0.2463 | 0.2453 | -0.00106 [-0.00228, +0.00015] | -0.00221 [-0.00480, +0.00035] |
| EXT_KR_16.13 | skirmish | 15,641 | 0.2454 | 0.2451 | -0.00028 [-0.00095, +0.00037] | -0.00074 [-0.00212, +0.00063] |
| EXT_KR_16.14_pilot | pick | 89 | 0.2618 | 0.2558 | -0.00591 [-0.01314, +0.00087] | -0.01359 [-0.02880, +0.00112] |
| EXT_KR_16.14_pilot | skirmish | 285 | 0.2452 | 0.2438 | -0.00143 [-0.00672, +0.00326] | -0.00189 [-0.01311, +0.00776] |
| EXT_KR_16.15 | pick | 405 | 0.2437 | 0.2367 | -0.00694 [-0.01055, -0.00329] | -0.01420 [-0.02179, -0.00655] |
| EXT_KR_16.15 | skirmish | 1,307 | 0.2433 | 0.2434 | +0.00010 [-0.00219, +0.00259] | +0.00012 [-0.00474, +0.00539] |
| EXT_NA1_16.13 | pick | 5,126 | 0.2464 | 0.2447 | -0.00171 [-0.00290, -0.00045] | -0.01201 [-0.03026, -0.00151] |
| EXT_NA1_16.13 | skirmish | 16,100 | 0.2452 | 0.2451 | -0.00016 [-0.00088, +0.00061] | -0.00522 [-0.01137, -0.00097] |

Oracle-cohort routing (**membership needs post-cutoff participation; NOT a deployable live model**) vs pooled on all known rows:

| Set | rows | routed Brier | pooled Brier | dBrier [95% CI] | dLogLoss [95% CI] | pooled E baseline reproduces full run |
|---|---:|---:|---:|---|---|---|
| MAIN_TEST | 163,576 | 0.2394 | 0.2398 | -0.00041 [-0.00060, -0.00023] | -0.00074 [-0.00160, +0.00009] | True |
| EXT_KR_16.13 | 25,430 | 0.2423 | 0.2429 | -0.00055 [-0.00114, +0.00012] | -0.00215 [-0.00426, -0.00010] | True |
| EXT_KR_16.14_pilot | 475 | 0.2454 | 0.2491 | -0.00372 [-0.00961, +0.00146] | -0.00735 [-0.01998, +0.00392] | True |
| EXT_KR_16.15 | 2,219 | 0.2417 | 0.2430 | -0.00129 [-0.00342, +0.00105] | -0.00252 [-0.00702, +0.00239] | True |
| EXT_NA1_16.13 | 26,538 | 0.2440 | 0.2442 | -0.00010 [-0.00074, +0.00055] | -0.00530 [-0.00977, -0.00112] | True |

## 5. B: role supervision, meta-role classifier and team role posteriors

**Supervision source (weak).** Original raw match-detail files for the main corpus are not on disk under the project-configured paths (core/config.py DETAIL_DIR default does not exist; the legacy path holds a later 15.18-15.22 collection with no main-corpus IDs; docs/DATA_AVAILABILITY.md states the cache is the sole copy). Per the specification, cached `role_slots` of 15.14 TRAIN matches are used only as an explicitly **weak** annotation: `core/roles.get_role_slots_from_detail` maps teamPosition (individualPosition fallback, which in practice never rescues an empty teamPosition because Riot reports "Invalid") and silently fills unmatched slots by participant ID. Teams whose slot sequence in JSON key order reveals an ID fill are excluded: 184 of 149,346 TRAIN teams excluded, 149,162 used (745,810 participants). Whole-team or trailing-slot fills cannot be detected and remain. role_slots of VALIDATION/TEST/external matches were never read.
Available raw 15.14 detail outside the project (user Downloads, 2 distinct TRAIN matches): KR_77... teamPosition equals cached slot role 10/10, empty teamPosition 0; KR_77... teamPosition equals cached slot role 10/10, empty teamPosition 0. Two matches cannot estimate a fallback rate.

**Classifier (fixed, no tuning).** Inputs per participant: champion ID (one-hot, `handle_unknown=ignore`) and the two selected summoner spells (unordered multi-hot); multinomial LogisticRegression(C=1, lbfgs, max_iter 2000); one row per eligible participant, equal total weight per match. Final model on all eligible TRAIN teams, 5 OOF models on the existing TRAIN folds (TRAIN q rows use their own held-out fold model).

| model | rows | fit matches | champion vocab | spell vocab | lbfgs iterations | converged |
|---|---:|---:|---:|---:|---:|---|
| final | 745,810 | 74,672 | 171 | 9 | 94 | True |
| oof_fold0 | 598,775 | 59,948 | 171 | 9 | 79 | True |
| oof_fold1 | 594,765 | 59,550 | 171 | 9 | 85 | True |
| oof_fold2 | 597,065 | 59,778 | 171 | 9 | 76 | True |
| oof_fold3 | 597,245 | 59,799 | 171 | 9 | 96 | True |
| oof_fold4 | 595,390 | 59,613 | 171 | 9 | 93 | True |

OOF agreement with the **weak proxy itself** (not validated accuracy): posterior argmax 0.9789, classifier argmax 0.9437, team exact assignment 0.9468; per role TOP 0.954, JUNGLE 1.000, MIDDLE 0.950, BOTTOM 0.994, UTILITY 0.996.

**Team posterior.** 120 one-to-one assignments weighted by the product of participant role probabilities (clipped at 1e-12 inside logs), logsumexp-normalized and marginalized. This imposes one player per role as a modeling assumption; flex picks and lane swaps remain uncertain. Row and column sums were checked (max deviation ~1e-15); identical unseen drafts give uniform 0.2 marginals; permutation equivariance is tested.

| Set | matches | mean assignment entropy (nats) | mean max marginal | unseen champion participants | teams with any unseen champion |
|---|---:|---:|---:|---:|---:|
| MAIN_TRAIN (OOF) | 74,673 | 0.1206 | 0.9824 | 0.0000 | - |
| MAIN_VALIDATION (final) | 74,748 | 0.1182 | 0.9827 | 0.0000 | - |
| MAIN_TEST (final) | 60,579 | 0.1300 | 0.9806 | 0.0000 | 0.0000 |
| EXT_KR_16.13 (final) | 10,064 | 0.2306 | 0.9643 | 0.0174 | 0.0853 |
| EXT_KR_16.14_pilot (final) | 200 | 0.3327 | 0.9477 | 0.0215 | 0.1075 |
| EXT_KR_16.15 (final) | 926 | 0.2989 | 0.9533 | 0.0151 | 0.0740 |
| EXT_NA1_16.13 (final) | 10,000 | 0.2869 | 0.9544 | 0.0189 | 0.0925 |

**Role reliability on independent annotations (post-freeze diagnostic).** External raw detail `teamPosition` (Riot position assignment, not observed spatial lane), teams with five distinct valid positions:

| Set | participants | posterior argmax acc. | classifier argmax acc. | mean P(true role) | posterior log loss | team exact | seen / unseen champion acc. (n) | proxy (get_role_slots) = teamPosition on valid teams | invalid teams: detectable / undetectable fill |
|---|---:|---:|---:|---:|---:|---:|---|---:|---|
| KR_16.13 | 100,580 | 0.9503 | 0.8796 | 0.9354 | 0.1505 | 0.8808 | 0.9515 (98,833) / 0.8832 (1,747) | 1.0000 | 10 / 2 |
| KR_16.14_pilot | 2,000 | 0.9170 | 0.8385 | 0.8972 | 0.2639 | 0.7975 | 0.9198 (1,957) / 0.7907 (43) | 1.0000 | 0 / 0 |
| KR_16.15 | 9,250 | 0.9223 | 0.8569 | 0.9068 | 0.2371 | 0.8173 | 0.9248 (9,110) / 0.7571 (140) | 1.0000 | 2 / 0 |
| NA1_16.13 | 99,925 | 0.9245 | 0.8478 | 0.9090 | 0.2315 | 0.8209 | 0.9265 (98,042) / 0.8216 (1,883) | 1.0000 | 10 / 5 |

Per-role recall (posterior argmax vs teamPosition): KR_16.13: TOP 0.920, JUNGLE 1.000, MIDDLE 0.894, BOTTOM 0.953, UTILITY 0.984; KR_16.14_pilot: TOP 0.895, JUNGLE 1.000, MIDDLE 0.828, BOTTOM 0.875, UTILITY 0.988; KR_16.15: TOP 0.899, JUNGLE 1.000, MIDDLE 0.846, BOTTOM 0.892, UTILITY 0.974; NA1_16.13: TOP 0.881, JUNGLE 1.000, MIDDLE 0.843, BOTTOM 0.929, UTILITY 0.970. Raw sha256 mismatches: KR_16.13 0, KR_16.14_pilot 0, KR_16.15 0, NA1_16.13 0; draft vs raw champion/spell mismatches: KR_16.13 0, KR_16.14_pilot 0, KR_16.15 0, NA1_16.13 0. These are later patches/regions than TRAIN, so they measure transfer of the draft-to-role mapping, not the in-distribution accuracy on 15.14-15.16.

## 6. C: controlled h90 representation comparison

Arms: **participant** (frozen ridge set, 352 numeric: ridge C=.01 and a NEW full-feature LightGBM with the economic-model parameters), **draft control** (participant set + per-slot champion one-hot + unordered spell multi-hot, encoders fit on cohort TRAIN), **role** (191 global features + p_pre_V + role-weighted blue/red features 2x5x16 + role differences 5x16 + difference x time for gold/xp/level 5x3 + team role-uncertainty 2x7 = 461; no participant-slot block, no direct champion/spell indicators). Feature sets differ; this is not a parameter-count-only comparison. TRAIN rows use OOF role posteriors. Each base has raw/sigmoid/isotonic variants.

**T** Q_SELECT Brier (within-arm winner *; overall choice **role_lgbm_raw**; ranking ['role_lgbm_raw', 'participant_lgbm_raw', 'draft_lgbm_raw', 'A_specialist:ridge_raw']):

| arm | features | ridge raw | ridge sigmoid | ridge isotonic | LGBM raw | LGBM sigmoid | LGBM isotonic | ridge fit s / iter | LGBM fit s |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|
| participant | 352 | 0.229194 | 0.229239 | 0.229614 | 0.228550 * | 0.228618 | 0.229191 | 1.6 / 81 | 4.3 |
| draft | 1641 | 0.229845 | 0.229801 | 0.230643 | 0.228635 * | 0.228676 | 0.228963 | 1.9 / 119 | 4.5 |
| role | 461 | 0.229018 | 0.229062 | 0.229591 | 0.228032 * | 0.228091 | 0.228904 | 0.9 / 91 | 6.5 |
| A specialist (ridge_raw) | | 0.229194 | | | | | | | |

**N** Q_SELECT Brier (within-arm winner *; overall choice **role_lgbm_sigmoid**; ranking ['role_lgbm_sigmoid', 'draft_lgbm_sigmoid', 'participant_lgbm_sigmoid', 'A_specialist:ridge_isotonic']):

| arm | features | ridge raw | ridge sigmoid | ridge isotonic | LGBM raw | LGBM sigmoid | LGBM isotonic | ridge fit s / iter | LGBM fit s |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|
| participant | 352 | 0.242637 | 0.242643 | 0.242605 | 0.239743 | 0.239280 * | 0.239300 | 4.0 / 109 | 10.7 |
| draft | 1789 | 0.242174 | 0.242178 | 0.242097 | 0.239435 | 0.238830 * | 0.238958 | 10.1 / 159 | 13.5 |
| role | 461 | 0.241760 | 0.241765 | 0.241725 | 0.238863 | 0.238537 * | 0.238553 | 5.7 / 140 | 20.9 |
| A specialist (ridge_isotonic) | | 0.242560 | | | | | | | |

TEST and external paired comparisons (a minus b; fixed frozen models; the overall-vs-A pair mixes base model class and representation, so the controlled representation effect is read from the arm-winner pairs, which here share the LightGBM base and calibration family):

| Set | cohort | rows | pair | dBrier [95% CI] | dLogLoss [95% CI] | dAUC [95% CI] |
|---|---|---:|---|---|---|---|
| MAIN_TEST | T | 32,981 | role_winner - participant_winner | -0.00017 [-0.00056, +0.00025] | -0.00035 [-0.00120, +0.00054] | +0.0006 [-0.0011, +0.0023] |
| MAIN_TEST | T | 32,981 | draft_winner - participant_winner | +0.00004 [-0.00016, +0.00025] | +0.00008 [-0.00034, +0.00055] | -0.0001 [-0.0010, +0.0007] |
| MAIN_TEST | T | 32,981 | role_winner - draft_winner | -0.00021 [-0.00063, +0.00023] | -0.00044 [-0.00134, +0.00051] | +0.0007 [-0.0011, +0.0025] |
| MAIN_TEST | T | 32,981 | overall_winner - A_specialist | +0.00007 [-0.00045, +0.00065] | -0.00005 [-0.00119, +0.00128] | -0.0007 [-0.0029, +0.0014] |
| MAIN_TEST | T | 32,981 | A_specialist - pooled | -0.00166 [-0.00235, -0.00104] | -0.00465 [-0.00711, -0.00264] | +0.0061 [+0.0036, +0.0088] |
| MAIN_TEST | N | 130,595 | role_winner - participant_winner | -0.00071 [-0.00098, -0.00046] | -0.00149 [-0.00203, -0.00096] | +0.0035 [+0.0020, +0.0052] |
| MAIN_TEST | N | 130,595 | draft_winner - participant_winner | -0.00017 [-0.00030, -0.00006] | -0.00035 [-0.00060, -0.00010] | +0.0014 [+0.0006, +0.0022] |
| MAIN_TEST | N | 130,595 | role_winner - draft_winner | -0.00054 [-0.00082, -0.00024] | -0.00114 [-0.00172, -0.00054] | +0.0021 [+0.0004, +0.0039] |
| MAIN_TEST | N | 130,595 | overall_winner - A_specialist | -0.00350 [-0.00390, -0.00307] | -0.00856 [-0.01010, -0.00726] | +0.0196 [+0.0170, +0.0220] |
| MAIN_TEST | N | 130,595 | A_specialist - pooled | -0.00042 [-0.00057, -0.00024] | -0.00003 [-0.00114, +0.00130] | +0.0016 [+0.0006, +0.0026] |
| EXT_KR_16.13 | T | 5,202 | role_winner - participant_winner | -0.00080 [-0.00190, +0.00034] | -0.00170 [-0.00406, +0.00068] | +0.0030 [-0.0020, +0.0076] |
| EXT_KR_16.13 | T | 5,202 | draft_winner - participant_winner | -0.00031 [-0.00085, +0.00023] | -0.00068 [-0.00182, +0.00046] | +0.0013 [-0.0010, +0.0035] |
| EXT_KR_16.13 | T | 5,202 | role_winner - draft_winner | -0.00050 [-0.00161, +0.00055] | -0.00103 [-0.00338, +0.00123] | +0.0017 [-0.0027, +0.0065] |
| EXT_KR_16.13 | T | 5,202 | overall_winner - A_specialist | -0.00261 [-0.00509, -0.00041] | -0.00656 [-0.01228, -0.00137] | +0.0046 [-0.0040, +0.0133] |
| EXT_KR_16.13 | T | 5,202 | A_specialist - pooled | -0.00096 [-0.00299, +0.00123] | -0.00369 [-0.01182, +0.00312] | +0.0078 [-0.0012, +0.0163] |
| EXT_KR_16.13 | N | 20,228 | role_winner - participant_winner | -0.00147 [-0.00217, -0.00076] | -0.00310 [-0.00456, -0.00165] | +0.0080 [+0.0040, +0.0123] |
| EXT_KR_16.13 | N | 20,228 | draft_winner - participant_winner | +0.00012 [-0.00014, +0.00041] | +0.00026 [-0.00028, +0.00088] | -0.0005 [-0.0024, +0.0012] |
| EXT_KR_16.13 | N | 20,228 | role_winner - draft_winner | -0.00159 [-0.00231, -0.00086] | -0.00337 [-0.00490, -0.00185] | +0.0085 [+0.0042, +0.0130] |
| EXT_KR_16.13 | N | 20,228 | overall_winner - A_specialist | -0.00399 [-0.00507, -0.00290] | -0.00971 [-0.01358, -0.00657] | +0.0220 [+0.0158, +0.0282] |
| EXT_KR_16.13 | N | 20,228 | A_specialist - pooled | -0.00023 [-0.00084, +0.00036] | +0.00079 [-0.00150, +0.00412] | -0.0001 [-0.0034, +0.0034] |
| EXT_KR_16.14_pilot | T | 101 | role_winner - participant_winner | -0.00640 [-0.01677, +0.00324] | -0.01450 [-0.03721, +0.00682] | +0.0200 [-0.0276, +0.0717] |
| EXT_KR_16.14_pilot | T | 101 | draft_winner - participant_winner | -0.00176 [-0.00565, +0.00238] | -0.00408 [-0.01236, +0.00488] | +0.0091 [-0.0135, +0.0300] |
| EXT_KR_16.14_pilot | T | 101 | role_winner - draft_winner | -0.00464 [-0.01620, +0.00536] | -0.01042 [-0.03503, +0.01103] | +0.0109 [-0.0398, +0.0642] |
| EXT_KR_16.14_pilot | T | 101 | overall_winner - A_specialist | -0.00644 [-0.02106, +0.00801] | -0.01473 [-0.04748, +0.01753] | +0.0210 [-0.0400, +0.0837] |
| EXT_KR_16.14_pilot | T | 101 | A_specialist - pooled | -0.00223 [-0.01585, +0.01137] | -0.00458 [-0.03541, +0.02567] | +0.0071 [-0.0487, +0.0617] |
| EXT_KR_16.14_pilot | N | 374 | role_winner - participant_winner | -0.00058 [-0.00641, +0.00511] | -0.00103 [-0.01301, +0.01101] | +0.0062 [-0.0310, +0.0397] |
| EXT_KR_16.14_pilot | N | 374 | draft_winner - participant_winner | -0.00022 [-0.00224, +0.00168] | -0.00050 [-0.00468, +0.00347] | +0.0036 [-0.0119, +0.0176] |
| EXT_KR_16.14_pilot | N | 374 | role_winner - draft_winner | -0.00036 [-0.00622, +0.00552] | -0.00053 [-0.01259, +0.01166] | +0.0026 [-0.0331, +0.0359] |
| EXT_KR_16.14_pilot | N | 374 | overall_winner - A_specialist | -0.00431 [-0.01203, +0.00399] | -0.00892 [-0.02462, +0.00858] | +0.0353 [-0.0111, +0.0793] |
| EXT_KR_16.14_pilot | N | 374 | A_specialist - pooled | -0.00309 [-0.00767, +0.00097] | -0.00608 [-0.01562, +0.00244] | +0.0160 [-0.0054, +0.0411] |
| EXT_KR_16.15 | T | 507 | role_winner - participant_winner | +0.00241 [-0.00110, +0.00599] | +0.00529 [-0.00198, +0.01284] | -0.0089 [-0.0246, +0.0086] |
| EXT_KR_16.15 | T | 507 | draft_winner - participant_winner | +0.00068 [-0.00096, +0.00228] | +0.00167 [-0.00189, +0.00519] | -0.0022 [-0.0094, +0.0060] |
| EXT_KR_16.15 | T | 507 | role_winner - draft_winner | +0.00173 [-0.00174, +0.00546] | +0.00361 [-0.00378, +0.01156] | -0.0068 [-0.0233, +0.0094] |
| EXT_KR_16.15 | T | 507 | overall_winner - A_specialist | -0.00353 [-0.01096, +0.00413] | -0.00816 [-0.02452, +0.00824] | +0.0134 [-0.0153, +0.0437] |
| EXT_KR_16.15 | T | 507 | A_specialist - pooled | +0.00038 [-0.00604, +0.00641] | +0.00091 [-0.01308, +0.01394] | +0.0010 [-0.0244, +0.0290] |
| EXT_KR_16.15 | N | 1,712 | role_winner - participant_winner | -0.00039 [-0.00300, +0.00209] | -0.00087 [-0.00631, +0.00435] | +0.0000 [-0.0144, +0.0149] |
| EXT_KR_16.15 | N | 1,712 | draft_winner - participant_winner | +0.00039 [-0.00064, +0.00142] | +0.00086 [-0.00125, +0.00297] | -0.0016 [-0.0077, +0.0045] |
| EXT_KR_16.15 | N | 1,712 | role_winner - draft_winner | -0.00078 [-0.00343, +0.00172] | -0.00172 [-0.00726, +0.00355] | +0.0017 [-0.0123, +0.0171] |
| EXT_KR_16.15 | N | 1,712 | overall_winner - A_specialist | +0.00100 [-0.00309, +0.00494] | +0.00190 [-0.00664, +0.01042] | -0.0051 [-0.0282, +0.0183] |
| EXT_KR_16.15 | N | 1,712 | A_specialist - pooled | -0.00255 [-0.00468, -0.00028] | -0.00527 [-0.00968, -0.00053] | +0.0100 [-0.0027, +0.0226] |
| EXT_NA1_16.13 | T | 5,312 | role_winner - participant_winner | +0.00026 [-0.00087, +0.00144] | +0.00050 [-0.00193, +0.00306] | -0.0017 [-0.0064, +0.0029] |
| EXT_NA1_16.13 | T | 5,312 | draft_winner - participant_winner | -0.00004 [-0.00060, +0.00045] | -0.00016 [-0.00139, +0.00088] | -0.0004 [-0.0023, +0.0018] |
| EXT_NA1_16.13 | T | 5,312 | role_winner - draft_winner | +0.00030 [-0.00084, +0.00146] | +0.00066 [-0.00176, +0.00316] | -0.0013 [-0.0061, +0.0034] |
| EXT_NA1_16.13 | T | 5,312 | overall_winner - A_specialist | -0.00666 [-0.00924, -0.00432] | -0.01643 [-0.02219, -0.01101] | +0.0131 [+0.0047, +0.0209] |
| EXT_NA1_16.13 | T | 5,312 | A_specialist - pooled | +0.00204 [-0.00024, +0.00432] | -0.00607 [-0.02260, +0.00757] | +0.0020 [-0.0066, +0.0107] |
| EXT_NA1_16.13 | N | 21,226 | role_winner - participant_winner | -0.00161 [-0.00231, -0.00081] | -0.00339 [-0.00485, -0.00174] | +0.0086 [+0.0040, +0.0128] |
| EXT_NA1_16.13 | N | 21,226 | draft_winner - participant_winner | +0.00028 [-0.00004, +0.00057] | +0.00060 [-0.00007, +0.00120] | -0.0013 [-0.0031, +0.0006] |
| EXT_NA1_16.13 | N | 21,226 | role_winner - draft_winner | -0.00189 [-0.00261, -0.00110] | -0.00399 [-0.00548, -0.00235] | +0.0099 [+0.0053, +0.0142] |
| EXT_NA1_16.13 | N | 21,226 | overall_winner - A_specialist | -0.00360 [-0.00479, -0.00240] | -0.00933 [-0.01376, -0.00597] | +0.0181 [+0.0117, +0.0247] |
| EXT_NA1_16.13 | N | 21,226 | A_specialist - pooled | -0.00050 [-0.00109, +0.00016] | -0.00366 [-0.00935, +0.00108] | -0.0008 [-0.0044, +0.0024] |

Metrics of the chosen candidates (AUC / Brier / log loss / calibration slope / ECE):

| Set | cohort | role winner | participant winner | draft winner | A specialist | pooled |
|---|---|---|---|---|---|---|
| MAIN_TEST | T | 0.6695 / 0.2282 / 0.6481 / 1.051 / 0.0064 | 0.6688 / 0.2284 / 0.6484 / 1.065 / 0.0063 | 0.6688 / 0.2284 / 0.6485 / 1.060 / 0.0075 | 0.6702 / 0.2282 / 0.6481 / 1.002 / 0.0045 | 0.6641 / 0.2298 / 0.6528 / 1.049 / 0.0090 |
| MAIN_TEST | N | 0.6180 / 0.2392 / 0.6710 / 1.010 / 0.0033 | 0.6145 / 0.2399 / 0.6725 / 1.013 / 0.0056 | 0.6159 / 0.2397 / 0.6722 / 1.011 / 0.0074 | 0.5984 / 0.2427 / 0.6796 / 0.912 / 0.0038 | 0.5968 / 0.2431 / 0.6796 / 0.865 / 0.0079 |
| EXT_KR_16.13 | T | 0.6431 / 0.2349 / 0.6625 / 0.883 / 0.0120 | 0.6401 / 0.2357 / 0.6642 / 0.855 / 0.0158 | 0.6414 / 0.2354 / 0.6635 / 0.868 / 0.0146 | 0.6385 / 0.2375 / 0.6690 / 0.790 / 0.0341 | 0.6307 / 0.2385 / 0.6727 / 0.834 / 0.0468 |
| EXT_KR_16.13 | N | 0.6100 / 0.2407 / 0.6742 / 0.907 / 0.0108 | 0.6019 / 0.2422 / 0.6773 / 0.864 / 0.0121 | 0.6015 / 0.2423 / 0.6776 / 0.848 / 0.0136 | 0.5880 / 0.2447 / 0.6839 / 0.754 / 0.0205 | 0.5881 / 0.2449 / 0.6831 / 0.750 / 0.0231 |
| EXT_KR_16.14_pilot | T | 0.6795 / 0.2252 / 0.6423 / 1.192 / 0.0373 | 0.6595 / 0.2316 / 0.6568 / 0.960 / 0.0658 | 0.6686 / 0.2299 / 0.6527 / 1.012 / 0.0938 | 0.6585 / 0.2317 / 0.6570 / 0.918 / 0.0588 | 0.6514 / 0.2339 / 0.6616 / 0.931 / 0.0628 |
| EXT_KR_16.14_pilot | N | 0.6076 / 0.2420 / 0.6773 / 0.816 / 0.0481 | 0.6014 / 0.2426 / 0.6783 / 0.806 / 0.0455 | 0.6050 / 0.2424 / 0.6778 / 0.811 / 0.0571 | 0.5722 / 0.2464 / 0.6862 / 0.640 / 0.0437 | 0.5562 / 0.2494 / 0.6922 / 0.484 / 0.0492 |
| EXT_KR_16.15 | T | 0.6243 / 0.2368 / 0.6653 / 0.844 / 0.0666 | 0.6332 / 0.2343 / 0.6600 / 0.910 / 0.0322 | 0.6310 / 0.2350 / 0.6617 / 0.899 / 0.0421 | 0.6109 / 0.2403 / 0.6734 / 0.687 / 0.0461 | 0.6099 / 0.2399 / 0.6725 / 0.748 / 0.0391 |
| EXT_KR_16.15 | N | 0.6071 / 0.2417 / 0.6765 / 0.852 / 0.0218 | 0.6070 / 0.2421 / 0.6773 / 0.852 / 0.0273 | 0.6054 / 0.2425 / 0.6782 / 0.822 / 0.0237 | 0.6122 / 0.2407 / 0.6745 / 0.998 / 0.0258 | 0.6022 / 0.2433 / 0.6798 / 0.834 / 0.0246 |
| EXT_NA1_16.13 | T | 0.6268 / 0.2390 / 0.6712 / 0.752 / 0.0292 | 0.6284 / 0.2388 / 0.6707 / 0.756 / 0.0297 | 0.6280 / 0.2387 / 0.6706 / 0.760 / 0.0277 | 0.6136 / 0.2457 / 0.6877 / 0.592 / 0.0591 | 0.6117 / 0.2436 / 0.6937 / 0.604 / 0.0506 |
| EXT_NA1_16.13 | N | 0.6085 / 0.2410 / 0.6748 / 0.882 / 0.0121 | 0.6000 / 0.2426 / 0.6782 / 0.828 / 0.0129 | 0.5986 / 0.2429 / 0.6788 / 0.805 / 0.0146 | 0.5904 / 0.2446 / 0.6842 / 0.723 / 0.0168 | 0.5912 / 0.2451 / 0.6878 / 0.657 / 0.0208 |

**Leave-one-role-block-out ablations (T only).** Preselected full role ridge: role_ridge_raw (calibration raw); each block drop removes 51 columns (blue/red role features, differences, phase interactions) and refits the ridge and calibrators before TEST; global context remains, so information redundant with other roles or global totals can remain. This is a model ablation, not causal importance.

| Set | dropped block | own Q_SELECT choice | Q_SELECT Brier (drop / full) | TEST-set dBrier [95% CI] (drop - full) | dLogLoss [95% CI] | dAUC [95% CI] |
|---|---|---|---|---|---|---|
| MAIN_TEST | role_TOP | drop_role_TOP_ridge_raw | 0.228724 / 0.229018 | +0.00005 [-0.00019, +0.00032] | +0.00008 [-0.00045, +0.00068] | -0.0004 [-0.0014, +0.0005] |
| MAIN_TEST | role_JUNGLE | drop_role_JUNGLE_ridge_raw | 0.229445 / 0.229018 | +0.00062 [+0.00031, +0.00090] | +0.00126 [+0.00056, +0.00190] | -0.0025 [-0.0036, -0.0013] |
| MAIN_TEST | role_MIDDLE | drop_role_MIDDLE_ridge_raw | 0.229395 / 0.229018 | +0.00054 [+0.00027, +0.00079] | +0.00112 [+0.00050, +0.00168] | -0.0022 [-0.0032, -0.0012] |
| MAIN_TEST | role_BOTTOM | drop_role_BOTTOM_ridge_raw | 0.229012 / 0.229018 | +0.00027 [+0.00001, +0.00056] | +0.00059 [+0.00001, +0.00122] | -0.0012 [-0.0022, -0.0001] |
| MAIN_TEST | role_UTILITY | drop_role_UTILITY_ridge_raw | 0.228841 / 0.229018 | +0.00021 [-0.00006, +0.00048] | +0.00045 [-0.00014, +0.00105] | -0.0008 [-0.0019, +0.0003] |
| EXT_KR_16.13 | role_TOP | drop_role_TOP_ridge_raw | 0.228724 / 0.229018 | +0.00083 [+0.00009, +0.00162] | +0.00183 [+0.00015, +0.00360] | -0.0020 [-0.0050, +0.0008] |
| EXT_KR_16.13 | role_JUNGLE | drop_role_JUNGLE_ridge_raw | 0.229445 / 0.229018 | +0.00077 [+0.00000, +0.00160] | +0.00159 [-0.00011, +0.00340] | -0.0041 [-0.0074, -0.0009] |
| EXT_KR_16.13 | role_MIDDLE | drop_role_MIDDLE_ridge_raw | 0.229395 / 0.229018 | +0.00067 [-0.00015, +0.00151] | +0.00171 [-0.00003, +0.00354] | -0.0008 [-0.0040, +0.0022] |
| EXT_KR_16.13 | role_BOTTOM | drop_role_BOTTOM_ridge_raw | 0.229012 / 0.229018 | +0.00016 [-0.00071, +0.00104] | +0.00033 [-0.00157, +0.00229] | -0.0008 [-0.0043, +0.0027] |
| EXT_KR_16.13 | role_UTILITY | drop_role_UTILITY_ridge_raw | 0.228841 / 0.229018 | +0.00009 [-0.00057, +0.00080] | +0.00020 [-0.00133, +0.00179] | -0.0007 [-0.0037, +0.0021] |
| EXT_KR_16.14_pilot | role_TOP | drop_role_TOP_ridge_raw | 0.228724 / 0.229018 | -0.00483 [-0.00873, -0.00093] | -0.01067 [-0.01979, -0.00170] | +0.0181 [-0.0009, +0.0386] |
| EXT_KR_16.14_pilot | role_JUNGLE | drop_role_JUNGLE_ridge_raw | 0.229445 / 0.229018 | -0.00402 [-0.01047, +0.00222] | -0.00871 [-0.02446, +0.00541] | +0.0163 [-0.0114, +0.0444] |
| EXT_KR_16.14_pilot | role_MIDDLE | drop_role_MIDDLE_ridge_raw | 0.229395 / 0.229018 | +0.00372 [-0.00105, +0.00927] | +0.00914 [-0.00161, +0.02151] | -0.0148 [-0.0417, +0.0096] |
| EXT_KR_16.14_pilot | role_BOTTOM | drop_role_BOTTOM_ridge_raw | 0.229012 / 0.229018 | -0.00176 [-0.00729, +0.00313] | -0.00374 [-0.01659, +0.00723] | +0.0053 [-0.0175, +0.0307] |
| EXT_KR_16.14_pilot | role_UTILITY | drop_role_UTILITY_ridge_raw | 0.228841 / 0.229018 | +0.00095 [-0.00387, +0.00600] | +0.00054 [-0.01141, +0.01235] | -0.0076 [-0.0306, +0.0147] |
| EXT_KR_16.15 | role_TOP | drop_role_TOP_ridge_raw | 0.228724 / 0.229018 | -0.00122 [-0.00361, +0.00108] | -0.00248 [-0.00781, +0.00254] | +0.0093 [-0.0008, +0.0192] |
| EXT_KR_16.15 | role_JUNGLE | drop_role_JUNGLE_ridge_raw | 0.229445 / 0.229018 | -0.00047 [-0.00280, +0.00206] | -0.00142 [-0.00673, +0.00405] | +0.0003 [-0.0104, +0.0106] |
| EXT_KR_16.15 | role_MIDDLE | drop_role_MIDDLE_ridge_raw | 0.229395 / 0.229018 | -0.00142 [-0.00356, +0.00060] | -0.00269 [-0.00739, +0.00190] | +0.0100 [+0.0014, +0.0194] |
| EXT_KR_16.15 | role_BOTTOM | drop_role_BOTTOM_ridge_raw | 0.229012 / 0.229018 | +0.00008 [-0.00280, +0.00288] | +0.00012 [-0.00602, +0.00611] | -0.0007 [-0.0130, +0.0115] |
| EXT_KR_16.15 | role_UTILITY | drop_role_UTILITY_ridge_raw | 0.228841 / 0.229018 | -0.00024 [-0.00258, +0.00228] | -0.00052 [-0.00587, +0.00507] | +0.0036 [-0.0066, +0.0135] |
| EXT_NA1_16.13 | role_TOP | drop_role_TOP_ridge_raw | 0.228724 / 0.229018 | +0.00014 [-0.00069, +0.00100] | +0.00050 [-0.00146, +0.00254] | +0.0012 [-0.0017, +0.0042] |
| EXT_NA1_16.13 | role_JUNGLE | drop_role_JUNGLE_ridge_raw | 0.229445 / 0.229018 | -0.00071 [-0.00164, +0.00027] | -0.00200 [-0.00411, +0.00025] | +0.0002 [-0.0034, +0.0036] |
| EXT_NA1_16.13 | role_MIDDLE | drop_role_MIDDLE_ridge_raw | 0.229395 / 0.229018 | +0.00103 [+0.00023, +0.00177] | +0.00234 [+0.00055, +0.00406] | -0.0020 [-0.0048, +0.0009] |
| EXT_NA1_16.13 | role_BOTTOM | drop_role_BOTTOM_ridge_raw | 0.229012 / 0.229018 | +0.00049 [-0.00048, +0.00149] | +0.00101 [-0.00123, +0.00322] | -0.0027 [-0.0064, +0.0007] |
| EXT_NA1_16.13 | role_UTILITY | drop_role_UTILITY_ridge_raw | 0.228841 / 0.229018 | +0.00003 [-0.00078, +0.00080] | -0.00007 [-0.00193, +0.00164] | -0.0016 [-0.0045, +0.0015] |

**Ridge optimizer-path diagnostic (post-freeze, TRAIN/VALIDATION only, not used for any choice).** The participant-arm ridge and the A-specialist ridge share rows, inputs, weights and C but were fit through different array paths with sklearn's default lbfgs tolerance (1e-4):

| cohort | Q_SELECT abs(production A - production arm) mean / p99 / max | lbfgs iter (A / arm) | tight-tol refits: A path vs arm path max | production A vs tight max | production arm vs tight max | Q_SELECT Brier production A / arm / tight |
|---|---|---|---|---|---|---|
| T | 2.04e-08 / 8.62e-08 / 3.86e-07 | 81 / 81 | 7.57e-06 | 1.16e-02 | 1.16e-02 | 0.229194 / 0.229194 / 0.229211 |
| N | 8.14e-04 / 4.42e-03 / 2.92e-02 | 115 / 109 | 1.62e-05 | 4.60e-02 | 5.70e-02 | 0.242621 / 0.242637 / 0.242622 |

## 7. Role-arm explanations (exact 7-group Shapley on the final calibrated probability)

Model = each cohort's frozen **role-arm winner** (this explains the role model whether or not it is the overall winner). Groups: five role groups (blue and red role features, differences and phase interactions of that role), global context (all non-participant state features, time, p_pre_V), role uncertainty; 128 coalitions; 256 hash-selected TEST rows per cohort, 128 hash-selected cohort TRAIN background rows (OOF role posteriors). Interventional replacement breaks derived relations and role ambiguity remains, so these are descriptive associations with generated labels, not causal lane importance (Lundberg & Lee 2017 framework).

**T: role_lgbm_raw** (overall h90 winner role_lgbm_raw); 256 of 32981 eligible TEST cohort rows (0.78%); descriptive sample; background 128 rows; base value 0.4955; additivity 2.2e-16, phi+base = selected calibrated prediction 2.2e-16, reload identical True, 4,194,304 model evaluations in 24.5 s.

| group | columns | mean abs phi [bootstrap 95% CI over explained rows] | mean abs (match-weighted) | mean signed | mean abs, start 0-10 min (n=18) | mean abs, start 10-20 min (n=88) | mean abs, start 20-30 min (n=119) | mean abs, start 30+ min (n=31) |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| role_TOP | 51 | 0.0156 [0.0142, 0.0173] | 0.0156 | -0.00008 | 0.0088 | 0.0129 | 0.0180 | 0.0180 |
| role_JUNGLE | 51 | 0.0178 [0.0162, 0.0193] | 0.0178 | -0.00104 | 0.0170 | 0.0167 | 0.0194 | 0.0154 |
| role_MIDDLE | 51 | 0.0134 [0.0121, 0.0147] | 0.0134 | +0.00165 | 0.0088 | 0.0121 | 0.0149 | 0.0138 |
| role_BOTTOM | 51 | 0.0158 [0.0146, 0.0170] | 0.0158 | -0.00168 | 0.0137 | 0.0153 | 0.0162 | 0.0175 |
| role_UTILITY | 51 | 0.0107 [0.0095, 0.0119] | 0.0107 | +0.00111 | 0.0148 | 0.0089 | 0.0106 | 0.0138 |
| global_context | 192 | 0.0892 [0.0826, 0.0962] | 0.0892 | +0.00029 | 0.0254 | 0.0717 | 0.1098 | 0.0966 |
| role_uncertainty | 14 | 0.0009 [0.0007, 0.0010] | 0.0009 | +0.00003 | 0.0010 | 0.0009 | 0.0009 | 0.0008 |

* Local case (start 25.9 min, Y=1, q=0.5565, base 0.4955); largest signed groups: global_context +0.0334, role_TOP +0.0274, role_BOTTOM -0.0257. Draft role estimates (champion, estimated role, max marginal): blue (assignment entropy 0.01): Sion TOP 1.00, Viego JUNGLE 1.00, Orianna MIDDLE 1.00, Draven BOTTOM 1.00, Nautilus UTILITY 1.00; red (assignment entropy 1.02): Yone TOP 0.66, Khazix JUNGLE 1.00, Hwei MIDDLE 0.57, Heimerdinger BOTTOM 0.57, Leona UTILITY 1.00.
* Local case (start 34.82 min, Y=0, q=0.6440, base 0.4955); largest signed groups: global_context +0.1521, role_UTILITY +0.0229, role_BOTTOM -0.0213. Draft role estimates (champion, estimated role, max marginal): blue (assignment entropy 0.00): KSante TOP 1.00, RekSai JUNGLE 1.00, Vex MIDDLE 1.00, Smolder BOTTOM 1.00, Braum UTILITY 1.00; red (assignment entropy 0.00): Gnar TOP 1.00, Lillia JUNGLE 1.00, Azir MIDDLE 1.00, Sivir BOTTOM 1.00, Nautilus UTILITY 1.00.

**N: role_lgbm_sigmoid** (overall h90 winner role_lgbm_sigmoid); 256 of 130595 eligible TEST cohort rows (0.20%); descriptive sample; background 128 rows; base value 0.5120; additivity 3.3e-16, phi+base = selected calibrated prediction 3.3e-16, reload identical True, 4,194,304 model evaluations in 28.6 s.

| group | columns | mean abs phi [bootstrap 95% CI over explained rows] | mean abs (match-weighted) | mean signed | mean abs, start 0-10 min (n=135) | mean abs, start 10-20 min (n=77) | mean abs, start 20-30 min (n=39) | mean abs, start 30+ min (n=5) |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| role_TOP | 51 | 0.0113 [0.0100, 0.0126] | 0.0113 | -0.00166 | 0.0103 | 0.0118 | 0.0134 | 0.0138 |
| role_JUNGLE | 51 | 0.0247 [0.0225, 0.0270] | 0.0248 | -0.00224 | 0.0226 | 0.0267 | 0.0290 | 0.0171 |
| role_MIDDLE | 51 | 0.0170 [0.0155, 0.0185] | 0.0170 | +0.00051 | 0.0146 | 0.0185 | 0.0226 | 0.0145 |
| role_BOTTOM | 51 | 0.0314 [0.0289, 0.0340] | 0.0314 | -0.00077 | 0.0281 | 0.0377 | 0.0318 | 0.0194 |
| role_UTILITY | 51 | 0.0141 [0.0127, 0.0155] | 0.0141 | -0.00216 | 0.0139 | 0.0121 | 0.0179 | 0.0203 |
| global_context | 192 | 0.0290 [0.0252, 0.0332] | 0.0290 | -0.00441 | 0.0102 | 0.0366 | 0.0752 | 0.0579 |
| role_uncertainty | 14 | 0.0008 [0.0006, 0.0009] | 0.0007 | +0.00007 | 0.0009 | 0.0006 | 0.0004 | 0.0004 |

* Local case (start 2.69 min, Y=0, q=0.4985, base 0.5120); largest signed groups: role_UTILITY -0.0459, role_MIDDLE +0.0139, role_BOTTOM +0.0138. Draft role estimates (champion, estimated role, max marginal): blue (assignment entropy 0.60): Azir MIDDLE 0.71, Viego JUNGLE 1.00, Galio TOP 0.71, Lucian BOTTOM 1.00, Seraphine UTILITY 1.00; red (assignment entropy 0.03): Poppy TOP 1.00, RekSai JUNGLE 1.00, Akali MIDDLE 1.00, Yunara BOTTOM 1.00, Soraka UTILITY 1.00.
* Local case (start 18.09 min, Y=1, q=0.3808, base 0.5120); largest signed groups: global_context -0.0698, role_BOTTOM -0.0488, role_UTILITY -0.0135. Draft role estimates (champion, estimated role, max marginal): blue (assignment entropy 0.03): KSante TOP 0.99, Graves JUNGLE 1.00, Akali MIDDLE 0.99, Smolder BOTTOM 1.00, Pyke UTILITY 1.00; red (assignment entropy 0.01): Olaf TOP 1.00, Sylas JUNGLE 1.00, Galio MIDDLE 1.00, Ashe BOTTOM 1.00, Amumu UTILITY 1.00.

All eight local cases per cohort (signed group values, per-participant role marginals and confidence) are in shap/shap_summary.json.

## 8. Verification record

- [pass] readonly_prior_full_run_sha256_unchanged
- [pass] readonly_prior_full_run_external_cache_stat_unchanged
- [pass] readonly_original_repo_py_sha256_unchanged
- [pass] readonly_worktree_py_sha256_unchanged
- [pass] readonly_workspace_prior_scripts_sha256_unchanged
- [pass] readonly_main_cache_dir_stat_unchanged
- [pass] readonly_shard_sha256_unchanged
- [pass] cohort_manifest_before_protocol_and_fits
- [pass] contract_tests_passed_before_full_fits
- [pass] role_provenance_before_role_models
- [pass] all_selections_before_freeze
- [pass] test_external_labels_opened_only_after_freeze
- [pass] frozen_artifacts_unchanged_now
- [pass] frozen_artifacts_unchanged_through_evaluation
- [pass] cohort_join_one_to_one_566452
- [pass] cohort_partition_all_sets
- [pass] detector_train_fixture_pass
- [pass] external_redetection_bitwise
- [pass] specialist_serialization_and_fit_origins
- [pass] specialist_predictor_sets_equal_frozen_full_run
- [pass] role_models_converged
- [pass] role_oof_generator_is_own_heldout_fold_model
- [pass] role_oof_self_match_exclusion_zero
- [pass] arms_T_reload_identity_select
- [pass] arms_T_ridge_converged
- [pass] arms_N_reload_identity_select
- [pass] arms_N_ridge_converged
- [pass] pooled_E_baseline_reproduces_full_run_metrics
- [pass] C_MAIN_TEST_T_reload_identity_within_1e-12
- [pass] C_MAIN_TEST_N_reload_identity_within_1e-12
- [pass] C_EXT_KR_16.13_T_reload_identity_within_1e-12
- [pass] C_EXT_KR_16.13_N_reload_identity_within_1e-12
- [pass] C_EXT_KR_16.14_pilot_T_reload_identity_within_1e-12
- [pass] C_EXT_KR_16.14_pilot_N_reload_identity_within_1e-12
- [pass] C_EXT_KR_16.15_T_reload_identity_within_1e-12
- [pass] C_EXT_KR_16.15_N_reload_identity_within_1e-12
- [pass] C_EXT_NA1_16.13_T_reload_identity_within_1e-12
- [pass] C_EXT_NA1_16.13_N_reload_identity_within_1e-12
- [pass] independent_metric_reconstruction_lt_1e-10
- [pass] bootstrap_point_estimates_equal_evaluate_lt_1e-10
- [pass] shap_checks
- [pass] smoke_outputs_have_no_eval_or_test_predictions

## 9. Deviations, disclosures and limitations

- Prior exposure: TEST 15.16 and all four external sets were evaluated for the pooled q (and V) in the completed full run; KR 16.13 also in earlier pilot work. Results are exploratory re-evaluations.
- Labels are generated by a fitted V (TRAIN: held-out-fold V); Y is estimated probability improvement, not a won fight or human judgement. Labels and V were not changed.
- Cohort membership (participation count) is post-cutoff information. Specialist-vs-pooled comparisons are within identical cohort rows; oracle routing is not deployable.
- Role supervision is a weak proxy (cached role_slots) because raw main detail is unavailable; undetectable participant-ID fills can remain. OOF agreement with that proxy is not accuracy. External teamPosition reliability is a post-freeze transfer diagnostic on later patches/regions.
- Role posteriors come only from champion and spells; they are estimated meta roles, not observed lanes. The one-player-per-role assignment is a modeling assumption; flex picks and swaps stay uncertain.
- Role weights express role uncertainty, not lane importance; no hand-set lane values or lane reward entered labels or models.
- The full-feature participant LightGBM is a new model; arm differences include base-model class effects where arms choose different bases (not the case for the chosen arm winners here, which are all LightGBM in each cohort).
- Ridge fits use sklearn default lbfgs tolerance (1e-4), as in the frozen full run (pooled ridge candidates included). Identical designs fit through different array paths differ (N: mean 8e-4, max 0.029 on Q_SELECT rows), and default-tolerance fits sit up to ~0.06 (single rows) from a tight-tolerance optimum while aggregate Q_SELECT Brier changes by < 2e-5 (section 6 diagnostic). The frozen models were not refit; small ridge-vs-ridge differences should be read with this optimizer noise in mind.
- Isotonic-calibrated candidates can output exact 0/1 probabilities (counts reported per cell); probabilities were not repaired using TEST.
- Bootstrap intervals cover test-sample (match) variability only; training, calibration, selection and role-model variability are not included. Many cells are reported; no multiplicity adjustment was made.
- SHAP covers 256 of the eligible TEST rows per cohort (coverage stated) with a 128-row background; group attributions are descriptive.
- Smoke runs used TRAIN rows only (1/8 match subset, pseudo Q_CAL/Q_SELECT from TRAIN folds). In the role-model smoke, pseudo-validation TRAIN rows received OOF (not final) posteriors; smoke results were not used for any choice.
- Implementation defects found and fixed before full fits: StateV2 participant slots are numbered 0-9; a forbidden-token check matched "UTILITY_"; a test fixture was wrong; ridge reload identity requires the same featurize-then-predict path; a Downloads timeline file was mistaken for detail. Fixes are logged in commands.txt.
- draft/draft_manifest.json describes slots as "1-5 Blue, 6-10 Red" in ordinal terms; array index and StateV2 names are 0-4 Blue, 5-9 Red.
- Atakhan/FEAT_UPDATE absence and unseen champions in 16.xx external sets (full-run report) also affect these models; role vocabulary unseen-champion rates are listed in section 5.
- The old manuscript was not edited. No file in the original repo or the worktree changed (integrity snapshot; mtime scan). Side effects in the workspace: new bytecode caches scripts/__pycache__/cr20260915_*.pyc, fc20260915_shap.cpython-313.pyc and tests/__pycache__/test_cr20260915_contracts*.pyc (pytest and in-process imports); runtime_repo/ under this output root. Nothing was deleted.

Literature scope (as verified in the specification): Maymin (2021), Smart kills and worthless deaths, JQAS, DOI 10.1515/jqas-2019-0096 (tables 6/7: role-specific associations of performance bundles with match outcome) motivates role-aware analysis only, not this classifier, causal effects, the 90 s horizon or predicted gains. Lee & Ramler (2017), Identifying and Evaluating Successful Non-meta Strategies in League of Legends, FDG, DOI 10.1145/3102071.3102081, treats role/composition inference with spells and end-of-match items; end-of-match items are not allowed here and this classifier is not a reproduction. Lundberg & Lee (2017), arXiv:1705.07874, is the attribution framework; the grouping and background budget are operational choices. Cut 4 and the participation rule are the local v3.3 manifest/manuscript convention with reported cut 3/4/5 sensitivity, not a universal threshold from a paper.

## 10. Artifacts

protocol.json; status.json and status/; commands.txt; logs/; integrity/ (snapshots, diff); cohorts/ (cohort_manifest.json, <set>_cohort.npz, negative_count_provenance.json, detector_train_fixture.json); draft/ (draft npz, draft_manifest.json); role_supervision_provenance.json; contract_tests/; smoke_train_only/; selection/ (q_specialist_*, arms_*, ablations_T_h90, summaries); q_fit_metrics/; predictions/ (specialist trainval); models/q_specialist, models/arms, models/ablations; role_models/ + role_models_manifest.json; role_outputs/; frozen_manifest.json; label_access_log.jsonl; eval/ (results_A.json, results_C.json, role_reliability_external_raw.json, predictions/, hashes_*.json); shap/; diagnostics/; validation.json; DEFINITION_AND_EVIDENCE.md. Scripts: scripts/cr20260915_*.py; tests: tests/test_cr20260915_contracts.py.