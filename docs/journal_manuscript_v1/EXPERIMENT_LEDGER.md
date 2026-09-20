# Experiment ledger — every line of work behind the journal paper (T013, 2026-09-21)

*One page per reader question: what was run, what it is for in the paper, the headline numbers with their source fields, and what may not be said. Nothing here is recomputed; every number is copied from the named document. Status vocabulary follows the source documents (done / CLOSED / frozen / exploratory / withdrawn / superseded / thesis-v2). Role in paper: **main** = main-text table, **secondary** = main-text sentence or table row, **supp** = supplementary material, **cite-only** = mentioned as lineage, **none** = not in the journal.*

## 0. Reading order and the one-line story

Corpus (210 000 KR matches, patches 15.14–15.16) → engagement detector v3.3 (566 452 detected, 566 104 valid at h90) → cohorts T (113 901 rows) and S → frozen evaluator V̂ (fit85 MLP) → direction label Y = 1[ΔV̂ > 0] → predictor q vs PT_flex, one primary contrast per cohort → verification packs (RR3–RR6b) → external score-only (RRX) → scale split (SS). Everything after the evaluator freeze carries a "prior TEST exposure" tag and is reported as exploratory.

## 1. Evaluator lineage (V̂)

| Line | What it is | Status | Role | Headline (source → field/row) | Not to claim |
|---|---|---|---|---|---|
| V-1 / V-2 / V-3 / V-4 contracts (`V_REDESIGN_CONTRACT_20260919.md`, `V1_TASK_CONTRACT`, `V2_CANDIDATE_MATRIX`, `V_DYNAMIC_FRAME_CONTRACT`) | Estimand, 361-d expanded StateV2, candidate matrix, band reporting rules | locked 2026-09-19 | cite-only (Methods §2) | α_b = 1/4; expanded = 351 numeric + 10 champion IDs (`V_FEATURE_MANIFEST_RUNTIME`) | "V-4 proved model-switch artifact" (V1 L7) |
| Wave 1–3 band ledgers (`BAND_LEDGER_SHARED_LGBM`, `WAVE2_TIER01`, `WAVE3_TIER23`) | Early horse race on INPUT_IMPL_v0 | superseded by wave 4 | none | A0 shared logistic L_time 0.137638 (wave 2; legacy reference only) | mixing v0 scores with corrected wave-4 scores (WAVE4 L12) |
| **Wave 4 corrected** (`BAND_LEDGER_WAVE4_CORRECTED_20260919.md` L44–54) | 9 arms on corrected inputs, selection by L_time on V_SELECT | frozen basis | supp (selection table) | A_MLP_expanded L_time 0.1383 / Brier 0.1552 / AUC 0.8542; B_GRU_K5 0.1385 / 0.1547 / 0.8548 | "MLP clearly better" (single seed; GRU within 0.0002; L85–86) |
| **Freeze** (`V_EVALUATOR_FREEZE_CLOSE_20260920.md`; `A_MLP_expanded_evaluator_meta_20260919.json`) | fit85 MLP, PosSlopeSigmoid g on V_CAL, no full-TRAIN refit | PROVISIONAL FREEZE for ΔV/SVI | main (Methods §2) | bundle sha256 AC459CC4…B5204E3; census train 424 160 (fit 360 479 / stop 63 681), V_CAL 106 580, V_SELECT 108 558, TEST 347 234; calibration coef 0.9076 / intercept 0.0640 | "MLP globally best"; "SVI = true fight winner"; using fit85 SVI on TRAIN as q labels without OOF (L48–52) |
| **TEST band ledger** (`TEST_BAND_LEDGER_MLP_FIT85_20260920.md` L5–12; `REVIEW_RESPONSE_RR6A_CORP` rows V_to_W_timeline) | V̂→W on 15.16 timeline by band | reporting only | main (Results §1) | overall Brier 0.1552 / AUC 0.8542 / n 347 234 / 60 190 matches; t_2_10 0.2283 / 0.6642; t_10_20 0.1544 / 0.8580; t_20_30 0.0891 / 0.9496; t_30_inf 0.0856 / 0.9544; CORP MCB 0.0002 / DSC 0.0948 | retuning per-band models from TEST |
| Continuity ledger (`CONTINUITY_LEDGER_MLP_FIT85_20260920.md` L15–27) | ΔV̂ continuity, engagement pre/post V→W, peer sign agreement | diagnostic | secondary (Results §1, §2.4) | V_pre→W Brier 0.1442 / AUC 0.8750; V_post→W 0.1181 / 0.9149; sign agree vs peers 0.9151 / 0.9136 | boundary excess = artifact proof; preferring a peer for higher agreement |
| Bundle reload check (`V_EVALUATOR_BUNDLE_RELOAD_CHECK_20260919.md`) | bundle vs live path parity | PASS (max abs 0) | supp | — | — |

## 2. Label and cohort lineage (2026-09-15 build, copied under `docs/lineage_20260915/`)

| Line | What it is | Status | Role | Headline (source) | Not to claim |
|---|---|---|---|---|---|
| Endpoint rule (`engagement_labels_v3_rules.py`; `LABEL_ENDPOINT_RULE_AND_EXAMPLES_20260914.md`) | e_h = min(L + 1000h, K_next − 1, S_next − 1, T_end − 1); validity flags | reference copy | main (Methods §1b) | 566 104 / 566 452 valid; 348 overlap exclusions, all non-teamfight; follow-up mean 45.493 s, median 39.440 s; cap reached 14.286 %; next-kill stop ≈ 78.9 % (README L97–110; computed under the 2026-09-15 logistic V, L112) | "h90 is a 90 s window" |
| Cohort rule (`cr20260915_common.py` `scale_classes`; `cohort_manifest.json`) | cohort == 1 ⟺ min(cluster_blue, cluster_red) ≥ 4 ≡ fine == 2; S = cohort == 0 & fine == 1 | reference copy | main (Methods §1, §5) | T rows TRAIN 39 605 / VAL 41 315 / TEST 32 981 / pooled 113 901; per-role T matches (28 989 Σ folds; 7 550; 7 469; 7 448; 7 632; 24 020); S rows TEST 101 205 (101 242 detected) | pooled T = prediction n; presence count as participation |
| Corpus census (README L82–86) | matches / engagements / valid / T / N per patch | reference copy | main (Methods §5) | TRAIN 74 673 matches, 199 480 eng; VAL 74 748, 203 292; TEST 60 579, 163 680; total 210 000 / 566 452 / 566 104 | 109 829 teamfights of the market_event lineage ≠ 113 901 (README L115) |

## 3. Predictor q and transfer (frozen T pipeline, 2026-09-20)

| Line | What it is | Status | Role | Headline (source → field) | Not to claim |
|---|---|---|---|---|---|
| **Q design contract** (`Q_PREDICTION_DESIGN_CONTRACT_20260920.md`) | OOF labels for TRAIN (5 match folds), frozen bundle elsewhere; learner set; Q_SELECT selection | EXECUTION LOCK; scope amended 2026-09-21 (T ∪ S) | main (Methods §4, §5) | — | leaving fit85 p_pre in q inputs while labels are OOF (L89) |
| **Primary table** (`Q_NEWV_FIT85_PRIMARY_20260920.md` L37–48, 56–68; `Q_NEWV_FIT85_PRIMARY_TABLE_20260920.json`) | logit_state selected on Q_SELECT; TEST 15.16 T | frozen | secondary (continuity vs PT_linear) | TEST logit_state 0.2355 / 0.6637 / 0.6403; ΔBrier vs PT_linear −0.00422 [−0.00516, −0.00325] (800 draws); B40 −0.00225 [−0.00440, −0.00012] | "AUC 0.64 = 64 % of fights correctly signed"; "pure learner duel" (352 vs 362 inputs) |
| OOF status (`Q_NEWV_FIT85_OOF_STATUS_20260920.json`, `OOF_META_SLIM`) | label integrity for TRAIN | COMPLETE, PASS | supp | n 39 605; n_exact_zero 1; folds 5 (held matches 14 628–15 017) | — |
| **Transfer 16.x vs PT_linear** (`Q_NEWV_FIT85_TRANSFER_16X_20260920.md` L19–30) | score-only q on KR/NA1 16.13, KR 16.15, pilot | frozen; superseded for headline by RRX (PT_flex) | supp (continuity) | KR 16.13 +0.00274, NA1 +0.00342, KR 16.15 −0.00301, pilot −0.00320 | averaging small cohorts into "transfer maintained"; cause identification (L11–13) |
| Scope lock (`Q_RESULT_SCOPE_LOCK_20260920.md`) | interpretation umbrella | INTERPRETATION LOCK | cite-only | three-question table L26–28 | 210k as prediction n; AUC vs old-V q as improvement; "Q_CAL calibration completed" |

## 4. Review-response packs (all 2026-09-20, tag EXPLORATORY_REVIEW_RESPONSE_PRIOR_TEST_EXPOSURE)

| Pack | What it is | Status | Role | Headline (source → field) | Not to claim |
|---|---|---|---|---|---|
| RR0 (`REVIEW_RESPONSE_RR0_MANIFEST_20260920.json`) | keys, weights, digests | done | supp | OOF n 39 605; TEST parity 32 981; 12 digests (logit_state d6e8fc31310a5a58; TEST_h90 0a2e977b231acf85) | — |
| RR1 addendum (`…_RR1_EXECUTION_ADDENDUM_20260920.md`) | design vs as-run (uniform knots, non-normalised weights, two-stage) | EXECUTION LOCK | main (Methods §0, §4.2) | PT_flex (4, 4, 0.01); b_spline (6, 0.01) | "weighted-quantile knots"; "complete removal of initial-edge effects" |
| **RR12** (`…_RR12_RESULTS_20260920.{md,json}`) | q vs PT_flex, B40, bins, H | done | **main** (Results §3.1–§3.2, Fig. 2) | all-T −0.00373 [−0.00461, −0.00279]; B40 −0.00214 [−0.00422, −0.00003]; vs PT_linear −0.00422 [−0.00515, −0.00323]; H +0.00181 [−0.00060, +0.00422]; 20 narrow bins (md only) | "clear ≥ 0.001 B40 gain"; cherry-merging bins; p_gt0 as p-value |
| RR3 quiet + close-out (`…_RR3_QUIET`, `…_RR3_CLOSEOUT`) | matched quiet windows | done | secondary (Results §2.2) | E[|ΔV|_fight − |ΔV|_quiet] 0.0633 [0.0612, 0.0654]; matched 9 140 / 32 981 (0.277); L > 120 s unmatchable 9.2 % | causal fight effect; old-V quiet reuse; labelling the whole table match-weighted |
| RR4 margin (`…_RR4_MARGIN_20260920.md`) | λ·s_Q and absolute cut-offs; triad | CLOSED | secondary (Results §2.1, §4) | λ 0/0.25/0.5/1: ≈ −0.004, CIs exclude 0; s_Q 64.9 % p-only; 10-bin triad | selecting λ on TEST; "small ΔV labels are fine" |
| RR5a/5b + RR6b (`…_RR5_RR6B_20260920.md`) | material correspondence; next elite objective; horizon flips | CLOSED | secondary (Results §2.3, §2.4) | kill agree 0.904 / 0.928; Blue share 0.604 (6 004/9 934) vs 0.396 (4 072/10 277); flips 0.007–0.019; same-endpoint 0.59–0.81 | q accuracy; independent fight-winner accuracy; converting none→Red |
| RR6a CORP (`…_RR6A_CORP_20260920.md`, 34 rows) | CORP decomposition V→W and q→SVI | CLOSED (948b36a) | main (Results §1, §3.3) | q MCB 0.0009 / DSC 0.0154 vs PT_flex 0.0005 / 0.0113 (md prints 0.00086 / 0.00053 in manuscript trace) | "calibration is fine"; TEST-fit isotonic as upgrade; pooling stages |
| RRX external (`…_RRX_EXTERNAL_20260920.md`) | dual-stage score-only on 16.x T | CLOSED | main (Results §5) | KR 16.13 +0.0026, NA1 +0.0040, KR 16.15 +0.0006, pilot −0.0085; no intervals | validating EXT labels by V_pre Brier; fitted EXT recalibrator |
| Evidence matrix (`RESPONSE_EVIDENCE_MATRIX_20260920.md`) | tracker | JOURNAL FREEZE | cite-only | intervals exist for RR12, RR3, RR4 only (L43) | 13 forbidden items (L45–58) |

## 5. Scale split (T vs S, 2026-09-20/21, tag EXPLORATORY_SCALE_SPLIT_PRIOR_TEST_EXPOSURE)

| Line | What it is | Status | Role | Headline (source → field) | Not to claim |
|---|---|---|---|---|---|
| Contract (`SCALE_SPLIT_EXPERIMENT_CONTRACT_20260920.md`) | predeclared cohorts, arms, one primary contrast, forbidden readings; §8 decision C (2026-09-21) | predeclared; promoted to main text | main (Methods §1, §4) | — | §6 list (scale gradient; mechanism; TEST selection; substitution) |
| T007–T010 execution (`.ai/reports/T007–T010.md`, reviews) | cohort-parametrised pipeline; S labels; fits; identity rerun | DONE, reviewed | supp (manifest) | S census TEST 101 205 / Q_CAL 31 302 / Q_SELECT 31 059 / TRAIN 123 049; S∩T keys 0; fold held sets 14 628–15 017; RR0 digests match | — |
| **Results** (`SCALE_SPLIT_TvsS_RESULTS_20260920.{md,json}`) | Tables 1–4; identity main, selection sensitivity | done | **main** (Results §3.1, §3.2, §3.3, §3.4, §5) | q_S − PT_flex_S −0.00335 [−0.00379, −0.00288] (sens. −0.00346); q_S − q_T→S −0.00194; q_S − q_TS −0.00024 [−0.00039, −0.00009]; q_TS − q (T) +0.00071 [+0.00018, +0.00126]; S∩B40 −0.00210 [−0.00266, −0.00149]; H_S +0.00178 [+0.00096, +0.00259]; EXT S −0.0018 / −0.0020 / −0.0033 / +0.0035 | ranking cohorts; T/S mechanism (X-31); pooled or pick substitution |
| Prior-lineage T/N split (old logistic V, `e0ec3d0:outputs/cohort_role_training_20260915/REPORT.md`; quoted in the contract L13) | specialist vs pooled ΔBrier by class | superseded | cite-only or none | T −0.00166 [−0.00235, −0.00104]; N −0.00042; pick −0.00083; skirmish −0.00023 [−0.00043, −0.00004]; h60 skirmish sign flips | "N specialization robust"; any mechanism |

## 6. Withdrawn or superseded lines (never in the journal as results)

| Line | Why withdrawn / superseded | Where recorded |
|---|---|---|
| market_event engagement-winner prediction (CoG-extension `docs/tog_manuscript/`) | different target; leak history (time_norm, anchor); scale-gradient conclusion withdrawn (cut-dependent sign) | `sec_prediction.tex` §pred-audit, §pred-cut; `ACADEMIC_PACKAGE_AUDIT_20260920.md` L101–103 |
| Old-V (logistic) SVI numbers: LGBM−PT −0.00113, B40 −0.00041, kill agree 0.937, transfer "holds" | evaluator replaced by fit85; `EXPERIMENT_INVENTORY_COHORT_20260919.md` §G item 5 ("transfer holds") is contradicted by the new-V transfer doc and carries no supersession banner | `PAPER_COHORT_CONTRACT` §5–§6 banners (T004); `Q_NEWV_FIT85_TRANSFER_16X` L11 |
| Direction AUC 0.583 (stacking / |ΔV|-weighted sign model) | superseded by frozen q 0.6403 | `ACADEMIC_PACKAGE_AUDIT_20260920.md` L102 |
| "RNN unfit" (wave-3 v0) | withdrawn after input fix | `BAND_LEDGER_WAVE4_CORRECTED` L66 |
| T/N mechanism explanation (claim ledger X-31) | withdrawn; the ledger file itself is not in this branch (back-references only: contract L13/L64, results L96, FINISH_LOCK L43, Outline F7) | cite the withdrawal via `SCALE_SPLIT_EXPERIMENT_CONTRACT` §6 |
| TabNet 0.6188, SAINT 0.6546 (CoG-extension learners) | sign bug; leak-exposed batches | `reviewer_response_matrix.md` R2-5b/5c, X-2 |

## 7. Deferred to thesis / v2 (not in this paper; see `THESIS_V2_FOLLOWUPS.md`)

External probability adapter; Foster–Stine path diagnostics; multi-horizon rationality on full timelines; Brill–Yurko–Wyner match-resampled V uncertainty; q_λ combinations; per-participant-count cells; shared PT_flex object across S arms; a predeclared T/S external-transfer contrast; definition-constant sensitivity under the v3.3 corpus.

## 8. Known documentation discrepancies to fix before submission (documentation only)

1. `EXPERIMENT_INVENTORY_COHORT_20260919.md` §G says transfer "holds" (old V); add a supersession banner pointing to `Q_NEWV_FIT85_TRANSFER_16X_20260920.md` and RRX.
2. Claim ledger X-31 is cited in seven files but absent from this branch; either copy the ledger from `e0ec3d0` into `docs/lineage_20260915/` or cite the contract §6 as the withdrawal record.
3. `WORKING_NOTES.md` §D is one revision behind `.ai/RQ_DECISION_BRIEF.md` (missing the 2026-09-21 scale-split section).
4. CORP q/PT_flex MCB prints as 0.0009 / 0.0005 in `RR6A` (4 dp) but 0.00086 / 0.00053 in the manuscript trace (5 dp); both are the same field — state the precision once.
