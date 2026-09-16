# Retained missing-team counter: source clarification

This clarification does not change the frozen176/185 feature decision, models, results or frozen protocol.

`unknown_objective_team_count` is a mixed counter retained in A and B. The legacy implementation in `worktrees/engagement-state-value/gameplay/state_value.py` increments it for unidentified elite-monster teams, unidentified building/plate teams and unidentified soul teams. However, the actual V2 wrapper in `gameplay/state_value_v2.py` removes unassigned `DRAGON_SOUL_GIVEN` events before calling that legacy builder and tracks them separately as `unassigned_soul_events`.

Therefore descriptions of the counter in the frozen feature evidence/common helper that list the legacy unknown-soul branch should not be read as saying those filtered soul events enter this V2 model feature. Its remaining mixed monster/structure semantics still justify retaining it verbatim under the fixed named-channel ablation. TRAIN prevalence was measured, not assumed zero:1041/424160 bucket states (0.245426%).

This is a documentation clarification discovered during Codex source review, not a post-result feature change or claim that all objective proxies have been removed. Frozen source/evidence files are preserved for provenance. The main findings document uses the V2-specific description.
