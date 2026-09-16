# Execute P1: independent V2 training and calibration

You are the implementation/execution worker. Codex owns research design. Implement and RUN this bounded task; do not merely propose a plan. Keep the final report concise with artifact paths and checks. Do not read the entire conversation or manuscript.

Workspace: C:/Users/todtj/문서/LOL_Teamfight
Worktree/library: worktrees/engagement-state-value
Python: C:/Users/todtj/anaconda3/python.exe
Raw cache (READ ONLY): D:/LOL_Project/cache/match_cache_fresh_v3_engage_status13
Old V artifacts (READ ONLY): worktrees/engagement-state-value/outputs/temporal_winprob_v3_buckets
New output ONLY: outputs/independent_v2_participant_order
New implementation: scripts/train_independent_v2.py and, if needed, a new versioned model module under the worktree. Do not modify the frozen v1 modules, old artifacts, original Pycharm repository, or Claude's existing training queue. No git commit/push, deletion, installation, or secret inspection.

Read only the necessary implementation references:
- worktrees/engagement-state-value/gameplay/state_value_v2.py
- worktrees/engagement-state-value/scripts/run_temporal_winprob_v3_buckets.py
- worktrees/engagement-state-value/train/state_value_experiment.py
- worktrees/engagement-state-value/train/temporal_winprob.py
- old V protocol.json, sampled_minutes.json, results.json, expanded_training_cv.json.

Fixed protocol:
1. State version is objective_history_v2_participant_order. Build all new states with state_value_v2.StateBuilder and validate with state_matrix. Correct soul name parsing and team+participant ID ordering are already implemented. Do not restore postgame role_slots. Test suite: tests/test_state_value_v2_contract.py.
2. Reuse EXACT old protocol.json fit/calibrate/select/test match IDs and EXACT sampled_minutes.json queries. Report set/row equality. All engagement match IDs must stay out of every V partition. outputs/state_value_v2_fix/states.npz is the ENGAGEMENT sample, NOT V training data; do not use it to train/calibrate/select V.
3. Read per-match raw .npz, .meta.json and .events.json to construct states at each specified query. Final Blue winner is the independent target, obtained separately from GAME_END. Never include final winner or end-of-match duration in a feature. Use public node feature names from core.config. Keep snapshot_age_s as audit metadata and exclude it from V features, as before. Preserve the legacy 7-feature Maymin family for comparison; primary remains expanded.
4. Keep old model family and regularization protocol: inspect old expanded_training_cv.json to determine whether the original run fixed C or selected among .001/.01/.1/1 with grouped 3-fold CV. Reuse the same rule. Do not introduce hyperparameter search or change training weights compared with the original without reporting a blocking mismatch. Raw/sigmoid/isotonic calibrators fit only on calibrate. Select by select-partition log loss, as in the original V protocol. Test never selects models or calibrators.
5. Save a versioned inference adapter checking StateV2/schema before vectorization. Do not label old v1 models as V2. Check old model objects' assumptions about champion feature naming; participant_slotN_champion_id is categorical, not a continuous champion ranking. If a helper cannot safely accept the new schema, implement an equivalent local helper, preserving the declared family.
6. Evaluate one-query-per-test-match metrics (AUC/Brier/log loss, calibration intercept/slope and bins), and compare with stored old V probabilities on exactly matched match/time keys. For time/objective diagnostics use the SAME stored full test grid keys from independent_time_curves.npz, rebuilding V2 states. Report time bands 2–10,10–20,20–30,30+ and relevant objective histories with sample counts. Distinguish soul teamId=0 diagnostic from owned souls. No causal effect claims. Use equal total evaluation weight per match, within each stratum.
7. At least 500 paired test-match bootstrap replicates for primary V2 minus V1 AUC/Brier/log loss. Models fixed within bootstrap; report this uncertainty limit. No engagement label generation or q training in this P1 task.
8. Validation: zero match leakage, identical query membership, finite states/predictions, state queries never use future frames/events, all six soul kinds recognized when present, state construction invariant to deleting/permuting role_slots (reuse tests plus actual samples), and unchanged hashes for legacy source/models. Do not claim all-patch generalization. 15.16 and 26.13 have already been examined; no untouched external test claim.

Output contract:
- protocol.json with state/model schema, exact splits/sampling source hashes, exclusions and settings.
- status.json updated as extraction/training/evaluation progresses; errors.json if incomplete.
- new states (checkpoint for restart), model files and inference adapter, predictions with match/time/winner and model version, results.json, validation.json, comparison_v1_v2.json.
- REPORT.md: before/after table, what changed, tests, limitations, and next-stage readiness. References: Maymin 2021 DOI10.1515/jqas-2019-0096; Decroos et al. 2019 DOI10.1145/3292500.3330758; Van Calster et al. 2019 DOI10.1186/s12916-019-1466-7. These support value/probability validation, not our exact state schema or 90-second endpoint.
- Short final response with paths, actual result counts/metrics, remaining issues. Do not report success without running and checking output.

Resource discipline: CPU only, at most 4 worker threads, avoid nested BLAS oversubscription, no GPU competition with Claude's other experiment. Save extraction checkpoints. Stop and report a material specification conflict; repair ordinary implementation bugs and retry within scope. Do not weaken checks to pass.
