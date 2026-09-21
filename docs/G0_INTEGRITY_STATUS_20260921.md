# G0 Integrity Status — Phase A (2026-09-21)

**overall:** `PARTIAL_PHASE_A`  
**generated_at_utc:** 2026-09-21T07:18:41.890014+00:00  
**contract:** docs/SUPPLEMENTARY_EXPERIMENT_DESIGN_20260921.md#3  
**task:** .ai/tasks/T015.md  

Phase A does **not** open `outputs/` or large NPZ/joblib (`AGENTS.md`). INCOMPLETE items require Phase B with an explicit data-root work order.

| ID | Status | Reason |
|---|---|---|
| `G0.1_provenance_record` | **PARTIAL** | git HEAD, script sha16, and package versions recorded in this audit; runtime imported worktree modules are not hashed (scripts still sys.path-insert engagement-state-value — see worktree_inserts). |
| `G0.2_evaluator_hash_after_finalize` | **PASS** | finalize_fold_evaluator unit-tested; on-disk frozen evaluator re-hash is Phase B. |
| `G0.3_s_reuse_and_write_guard` | **PASS** | S requires --reuse-evaluators; OOF dumps guarded for S/reuse. |
| `G0.4_join_key_integrity` | **PASS** | join_label_engagement_indices fails on missing/duplicate keys; full-corpus join is Phase B. |
| `G0.5_fold_holdout_match_set` | **INCOMPLETE** | reuse mode loads V-bucket fold match sets in code, but verifying held-out match set hashes against stored OOF fits requires outputs/ (AGENTS.md: do not open in Phase A). |
| `G0.6_bundle_reload_parity` | **INCOMPLETE** | Requires loading frozen joblib bundles from outputs/. |
| `G0.7_headline_reaggregate` | **INCOMPLETE** | Requires frozen weights + prediction tables under outputs/; docs RR0 manifests present for later digest compare only. |
| `G0.8_missing_data_policy` | **PASS** | This audit marks data-dependent checks INCOMPLETE rather than emitting a fake PASS; FAIL artifacts are not written as normal results. |
| `G0.F_future_info_invariance` | **INCOMPLETE** | Needs fixed query metadata + feature rebuild with post-query events stripped (Phase B). |
| `unit_tests_provenance` | **PASS** | pytest rc=0; collected 12 items

tests\test_q_provenance_fixes.py ............                            [100%]

============================= 12 passed in 0.21s ============================== |

## Phase B required for

- G0.5_fold_holdout_match_set
- G0.6_bundle_reload_parity
- G0.7_headline_reaggregate
- G0.F_future_info_invariance
- G0.1 runtime imported-module byte hashes
- G0.2 on-disk frozen evaluator re-hash

## Provenance snapshot

- git_head: `55982c6a4a605199e48d9ef05c49eeba154758cd`
- packages: `{"python": "3.13.5", "platform": "Windows-11-10.0.26200-SP0", "numpy": "2.1.3", "sklearn": "1.6.1", "lightgbm": "4.6.0", "joblib": "1.4.2", "torch": "2.10.0+cu128"}`

### Script sha16

- `scripts/rr20260920_q_train_oof_mlp_folds.py`: `629B24E7109376CE`
- `scripts/rr20260920_review_response_rr12.py`: `62F0C8B7EE677278`
- `scripts/rr20260920_q_build_newv_labels.py`: `B420C8D26CCD48B1`
- `scripts/rr20260920_q_newv_primary_fit.py`: `EF8856957EAB4EA1`
- `scripts/g0_integrity_audit.py`: `1B687C81D5E9DE5C`

### Worktree / path-insert hits (known)

- `scripts/rr20260920_q_train_oof_mlp_folds.py:118:wt = data_root / "worktrees" / "engagement-state-value"`
- `scripts/rr20260920_q_train_oof_mlp_folds.py:120:sys.path.insert(0, str(data_root / "scripts"))`
- `scripts/rr20260920_q_train_oof_mlp_folds.py:121:sys.path.insert(0, str(wt))`
- `scripts/rr20260920_review_response_rr12.py:66:wt = data_root / "worktrees" / "engagement-state-value"`
- `scripts/rr20260920_review_response_rr12.py:68:sys.path.insert(0, str(data_root / "scripts"))`
- `scripts/rr20260920_review_response_rr12.py:69:sys.path.insert(0, str(wt))`
- `scripts/rr20260920_q_build_newv_labels.py:49:wt = data_root / "worktrees" / "engagement-state-value"`
- `scripts/rr20260920_q_build_newv_labels.py:51:sys.path.insert(0, str(data_root / "scripts"))`
- `scripts/rr20260920_q_build_newv_labels.py:52:sys.path.insert(0, str(wt))`
- `scripts/rr20260920_q_newv_primary_fit.py:50:wt = data_root / "worktrees" / "engagement-state-value"`
- `scripts/rr20260920_q_newv_primary_fit.py:52:sys.path.insert(0, str(data_root / "scripts"))`
- `scripts/rr20260920_q_newv_primary_fit.py:53:sys.path.insert(0, str(wt))`
- `scripts/g0_integrity_audit.py:92:elif "worktrees" in line and "engagement-state-value" in line:`
