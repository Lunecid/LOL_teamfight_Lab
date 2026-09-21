# G0 Integrity Status — Phase A+B (2026-09-21)

**overall:** `PARTIAL_PHASE_B`  
**generated_at_utc:** 2026-09-21T07:25:23.653494+00:00  
**tasks:** T015 (Phase A) · T016 (Phase B)  

| ID | Status | Reason |
|---|---|---|
| `G0.1_provenance_record` | **PARTIAL** | git HEAD, script sha16, and package versions recorded in this audit; runtime imported worktree modules are not hashed (scripts still sys.path-insert engagement-state-value — see worktree_inserts). |
| `G0.2_evaluator_hash_after_finalize` | **PASS** | Phase B: on-disk OOF evaluator SHA16 recorded after load; finalize unit tests remain from Phase A. |
| `G0.3_s_reuse_and_write_guard` | **PASS** | S requires --reuse-evaluators; OOF dumps guarded for S/reuse. |
| `G0.4_join_key_integrity` | **PASS** | join_label_engagement_indices fails on missing/duplicate keys; full-corpus join is Phase B. |
| `G0.5_fold_holdout_match_set` | **INCOMPLETE** | reuse mode loads V-bucket fold match sets in code, but verifying held-out match set hashes against stored OOF fits requires outputs/ (AGENTS.md: do not open in Phase A). |
| `G0.6_bundle_reload_parity` | **PARTIAL** | Loaded all five OOF evaluator+bundle joblibs and verified required keys / n_eng_labeled; full score-vs-saved prediction parity not run (needs engagement X). |
| `G0.7_headline_reaggregate` | **PASS** | Recomputed match-weighted ΔBrier from frozen prediction_table.npz; compared to docs print. |
| `G0.8_missing_data_policy` | **PASS** | This audit marks data-dependent checks INCOMPLETE rather than emitting a fake PASS; FAIL artifacts are not written as normal results. |
| `G0.F_future_info_invariance` | **INCOMPLETE** | Needs fixed query metadata + feature rebuild with post-query events stripped (Phase B). |
| `unit_tests_provenance` | **PASS** |  |

## Phase B digests (RR0)

status: **PASS**

- `rr12_prediction_table`: **MATCH** (`outputs/review_response_rr12_20260920/prediction_table.npz`)
- `rr12_paired_ci`: **MATCH** (`outputs/review_response_rr12_20260920/paired_ci.json`)
- `rr12_baseline_selection`: **MATCH** (`outputs/review_response_rr12_20260920/baseline_selection.json`)
- `rr12_PT_flex`: **MATCH** (`outputs/review_response_rr12_20260920/models/PT_flex.joblib`)
- `rr12_PT_linear`: **MATCH** (`outputs/review_response_rr12_20260920/models/PT_linear.joblib`)
- `rr12_b_spline`: **MATCH** (`outputs/review_response_rr12_20260920/models/b_spline.joblib`)
- `logit_state`: **MATCH** (`outputs/q_newv_fit85_20260920/models/logit_state.joblib`)
- `primary_table`: **MATCH** (`outputs/q_newv_fit85_20260920/primary_table.json`)
- `selection_freeze`: **MATCH** (`outputs/q_newv_fit85_20260920/selection_freeze.json`)
- `TRAIN_oof_STATUS`: **MATCH** (`outputs/q_newv_fit85_20260920/labels/TRAIN_oof_STATUS.json`)
- `TRAIN_oof_h90_meta`: **MATCH** (`outputs/q_newv_fit85_20260920/labels/TRAIN_oof_h90_meta.json`)
- `TEST_h90`: **MATCH** (`outputs/q_newv_fit85_20260920/labels/TEST_h90.npz`)

## Headline re-aggregation

- **T**: ΔBrier=-0.00373234 (docs print -0.00373); n=32981/24020 matches; **PASS**
- **S_identity**: ΔBrier=-0.00335262 (docs print -0.00335); n=101205/49730 matches; **PASS**

## OOF evaluator reload

- fold0: PASS sha16=`CD52698D7A1CE4E4` n_eng_labeled=7870
- fold1: PASS sha16=`DD31CC0DC3365ECA` n_eng_labeled=7850
- fold2: PASS sha16=`D5EDA9FAF9B21CDE` n_eng_labeled=7860
- fold3: PASS sha16=`A994E851B53118F8` n_eng_labeled=8037
- fold4: PASS sha16=`5D8445D5AE2ADB7F` n_eng_labeled=7988
