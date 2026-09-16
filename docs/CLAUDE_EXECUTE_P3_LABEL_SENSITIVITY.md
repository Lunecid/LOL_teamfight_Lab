# P3: frozen-A engagement label and window sensitivity

Codex design; Claude Opus 5 implementation and execution. User authorized overnight work until 2026-09-15 08:10 KST. Implement AND RUN this bounded exploratory experiment. Do not launch q training within P3.

## Context and preserved decision

P2 completed with validation pass and 19 tests. Codex independently recomputed 4,986 test match metrics (outputs/temporal_winprob_v3/codex_metric_audit.json). SELECT chose A=frozen P1 expanded V2, not the history RNN B. A model SHA256 7ca9dd1403ad35c6f0d043bc595dbded28a0d90c2024346f499f22732a3af53c. This is a frozen exploratory valuation candidate, not proven objective causal value. B remains a sensitivity comparator only; no retraining or selection on engagements.

Use inherited TEAMFIGHT_WORKSPACE, relative paths/Join-Path, Python C:/Users/todtj/anaconda3/python.exe, library worktrees/engagement-state-value, raw read-only cache D:/LOL_Project/cache/match_cache_fresh_v3_engage_status13. New outputs only outputs/engagement_labels_v3_sensitivity; new implementation scripts only. Frozen V1/P1/P2 modules and artifacts, original repository, existing user GPU queue unchanged. CPU <=4, no GPU/install/secrets/deletion/commits. Hash frozen inputs before/after. Continue routine fixes without weakening checks.

## Cohort and endpoints: no redefinition

Read scripts/event_boundary_cif.py, scripts/regenerate_state_v2.py, original source chain used by the latter, outputs/postkill_objective_delay_full/exposures.csv, outputs/state_value_v2_fix/rows.csv and P2 endpoint diagnostics. Authoritative cohort is the EXACT (match,s) set and order in state_value_v2_fix rows: expected 26,693 rows/9,198 matches. No silent dropping or new detection. Trace and document historic inclusion selection (in particular event_boundary_cif selects old valid h=120 rows); this may be availability/selection bias and must be disclosed rather than called all engagements. Preserve the cohort for fair comparison. Census exclusions from the parent exposures where reconstructable; do not expand the cohort based on new outcomes.

For each row, s=stored start, L=stored last kill from unique exposure join; q_pre=s-1ms. Confirm first kill=s+15000ms against raw events; if simultaneous raw kills, flag ambiguity, never choose an arbitrary position to exclude/relabel rows. Recover next eligible engagement start from original exposures and game end from source/raw GAME_END, independently check compatibility.

For h in {60,90,120} seconds:
  next_kill = first raw CHAMPION_KILL timestamp strictly greater than L (absence means infinity)
  endpoint_h = min(L+1000*h, next_kill-1ms, next_eligible_start-1ms, game_end-1ms).
Keep timestamp precision and closed inclusion <=query exactly as the existing B_next_kill rule. Handle absent next engagement with the original sentinel semantics. Store every tied terminal reason, not an arbitrary one. Require endpoint>=L and >q_pre; invalid cases must be reported and must block success, not silently dropped. An objective acquisition is NOT itself a terminal event. The state valuation covers both teams' observed global state, with no new spatial exclusion. Separate after-last-kill event attribution from events during the engagement.

Audit equality of reconstructed B90 endpoints to ALL existing state_value_v2_fix endpoints before scoring. Any mismatch is a material issue: report exact keys/reasons; do not overwrite reference data. Do not use older validate_windows.py's A-style next-engagement-only boundary by accident. Existing definitions s/G/D/R/participant rules remain unchanged.

## Values and labels

Build StateV2 causally with no future frames/interpolation/postgame roles. Use the SAME frozen A for pre and post; preserve query/snapshot/state/model version/hash metadata. Primary delta_A=p_post_A-p_pre_A, Y_A=1(delta_A>0). Exact zero is non-improvement Y=0, reported separately. Near-zero bands abs(delta)<=0.005/0.01/0.02 are sensitivity descriptions, not new label thresholds. Value is an estimated probability-point change, not monetary value or causal effect. No manual objective weights.

B90 A probabilities/states must match P2 endpoint outputs to numerical tolerance explicitly declared before scoring (prefer exact equality, diagnose operation-order rounding separately); key ordering must be exact. Score frozen B ensemble at all three endpoints with P2's history adapter for diagnostic model sensitivity, clearly not selected target. Do not mix A and B endpoints in one delta.

Outputs should include one row per (match,s,h) with source identifiers, patch, L, all boundary timestamps/reasons, duration, source age, p_pre/post per model, delta and label, missingness/validity flags and hashes. Separate input feature matrix (PRE ONLY, StateV2 + p_pre_A) from outcome metadata/post-state; never put L, endpoint, end reason, future membership/participants, game end or final winner in q features. Save a schema manifest explicitly marking input vs target vs audit-only. Champion IDs remain categorical, snapshot_age audit-only, no positions added.

## Analyses and verification

1. All three windows on the same cohort; positive/zero/near-zero counts, delta distributions, effective followup, terminal causes, availability and stale snapshot rates. Report row-weighted AND equal-total-weight-per-match rates distinctly.
2. Paired window sign flips 60vs90/90vs120/60vs120; model A vs B sign disagreement at each window. Compare older V1 values only where exact keys/windows match; never call prediction-label disagreement an error rate against truth. Compute >=1000 paired match bootstrap CIs for main sign-flip and model-disagreement rates, fixed models; state no training uncertainty included.
3. Breakdowns by existing patch/time bands and raw objective occurrences (baron, all dragon elements, elder, herald, grubs/horde, Atakhan, owned souls; teamId0 diagnostic not owned). Counts and broad sparse-cell uncertainty, no optimization of h/model on these outcomes. Report simultaneous event occurrences. Larger windows do not imply causal acquisition by the engagement.
4. Independent raw-event audit: no additional CHAMPION_KILL in (L,endpoint]; endpoint not crossing next eligible engagement/game end; last-kill validation; monotone endpoint_60<=endpoint_90<=endpoint_120; p_pre identical for all h; duplicated endpoints imply same post prediction; delta exact formula and label exact threshold; source snapshot<=query. Include examples where new kill truncates, next engagement truncates, horizon wins, game end truncates, ties and simultaneous events if present; mark synthetic examples as synthetic, never invent actual cases.
5. Confirm zero match overlap with any V fit/calibrate/select/test partition. Source keys and B90 reference identity; frozen model/module hash checks; deterministic rerun sample and schema guards. Any unresolved boundary/source mismatch blocks downstream P4.
6. References: reuse docs/TEMPORAL_WINPROB_LITERATURE_ALIGNMENT_20260915.md and existing endpoint definition/evidence docs. Cite Maymin state-value rationale and current operational B-rule as OUR event-based design, not a rule proved by that paper. Competing-event analysis references may motivate diagnostics but do not establish that all postkill objectives are caused by the fight. If exact original evidence cannot be recovered, say so; no invented citation.

Write protocol.json before execution, checkpoints/status.json, full logs, validation.json, results.json, labels and pre-only features/schema, source provenance, REPORT.md, DEFINITION_AND_EVIDENCE.md and TIMELINE_EXAMPLES.md. Preserve B90 as working primary and 60/120 as sensitivity; do not pick a new horizon by test results. Clearly label artifacts exploratory candidate labels, not production replacement or validated ground truth. End by stating readiness/remaining scientific limits for P4. Monitor full run until complete or an external blocker; no unverified completion claims.
