# Full-corpus label validity diagnostics — 2026-09-15

User authorizes continuing the ToG audit. Codex designs; Claude Opus 5 implements AND RUNS the bounded study below to completion, preserving failures and fixing routine bugs. Read docs/TOG_END_TO_END_READINESS_AUDIT_20260915.md, the full/cohort protocols, and applicable local instructions. This is exploratory follow-up after prior test exposure, not a new confirmatory test. Do not stop at scripts or a launched PID.

## Scope and resources

New outputs: outputs/label_validity_full_20260915. New implementation scripts prefixed lv20260915_, optional new tests. Parent full_corpus_training_20260915 and cohort_role_training_20260915, raw D: caches/shards, original Pycharm repo and manuscript are READ ONLY. No changes to old scripts/models/labels/logs, GPU queue, installs, deletes, commits/pushes, credentials or external messages. CPU total <=4, no GPU. Python C:/Users/todtj/anaconda3/python.exe. Work from TEAMFIGHT_WORKSPACE. No subagents necessary.

Use the entire designated corpus, all existing patch roles and external sets. No match reduction to pilots. Saved extraction chunks already contain V bucket states and pre/post states plus event count summaries; reuse them rather than repeat full raw decoding. Raw logs only for case packets and sanity fixtures. Loading helpers may write outcome access logs into the PARENT: avoid this. Implement a read-only wrapper that preserves existing role access checks while writing new access logs only to the new root. Do not globally redirect parent data paths incorrectly.

Primary frozen V, labels, E/T/N rules, cutoff, h90 and h60/h120 remain unchanged. Never choose new h/features/models by TEST performance. No q retraining in this study; q-vs-alternative-label scores are diagnostic estimand sensitivity, not new q validation.

## Protocol before fitting

Save protocol.json, source/parent hash inventory, exact feature lists and analysis choices before any new fits or inspecting newly generated TEST sensitivity scores. Existing results have already been seen; say so. Run TRAIN-only or synthetic meaningful contracts before full fitting. Fail if key joins are not exact. Preserve primary output metadata and invalid cases; no silent exclusions.

## A. Two fixed alternative simple valuation models

Retain primary A expanded logistic C=.01 and its final/5 OOF adapters untouched. Train exactly two comparators, no grid search:

1. B_reg: identical expanded features/preprocessing/logistic family to A, C=.1. This measures regularization sensitivity, not a new architecture claim.
2. B_econ: snapshot economic/resource logistic C=.01, feature columns fixed by suffix rules: time_minutes, time_minutes_sq; all participant_slot*_totalGold_norm,curGold_norm,level_norm,xp_norm,laneCS_norm,jgCS_norm; all 10 participant_slot*_champion_id as categorical one-hot. No explicit objective/kill/death/survival/hp/mana/history/age channels. Report this is a deliberately narrower state valuation comparator; object gains can still enter later observed resources, so do not call it a no-objective causal ablation.

Use exact primary TRAIN bucket query keys (424160/74168 matches), equal match weighting, TRAIN-only fit preprocessing. Fit each final and five OOF models on existing folds. Every TRAIN engagement's pre and post must use an adapter excluding its own match, including preprocessing. Same existing V_CAL/V_SELECT roles, raw vs positive-slope sigmoid family and same selection order (logloss then Brier/name) as primary. Apply selected family uniformly across fold adapters with fold-specific calibration if selected. Record memberships, convergence, fit/model hashes. Raise iteration cap only for convergence; do not tune C or features. No q-label-based V selection.

Freeze BOTH alternative families and chosen calibrators before scoring TEST/external W or generated labels. Evaluate their W probabilities on saved main/external bucket query keys: AUC/Brier/logloss/calibration and broad game-time bands. This is necessary context: disagreement with a much poorer V is not evidence the primary is wrong.

Generate both alternative pre/post probabilities and delta/Y at all three EXISTING endpoints for all sets, keyed (match,s). Use the same adapter at both ends. Write all predictions as NPZ with source/fold/model metadata. Confirm no primary changes and exact pre/endpoint/key identity.

## B. Full label agreement and observed-outcome diagnostics

For A vs each B report row-weighted and equal-match-weighted disagreement, positive/exact-zero rates, continuous delta difference, and unchanged-endpoint horizon disagreement separately. Main h90 E, T, N and each external E/T/N; 1000 paired match bootstrap for primary disagreement rates (fixed models; no training uncertainty). Do not use binomial independent-row uncertainty. Use denominator/empty/sparse indicators.

Stratify h90 by primary abs(delta) bins [0,.005],(.005,.01],(.01,.02],>.02; pre V probability bands [0,.2),[.2,.4),[.4,.6),[.6,.8),[.8,1]; time 2–10/10–20/20–30/30+ with an explicit <2 bin if present; ending reason; whether post_snapshot>L; frame age; cohort. These are diagnostics not new label exclusion thresholds or error bars for V.

Compare primary Y with observed event/resource directions on the SAME fixed endpoints:
- During engagement kills and full (pre,endpoint] kill differences Blue−Red separately, define the interval boundaries exactly. Blue kills minus Red kills, with ties kept as separate category; don't turn zero into a loss.
- Change in observed totalGold team difference and XP team difference between pre/post. State normalized units must be traced to source normalizers. If no reliable raw conversion, keep units explicitly normalized and never present as raw gold. Stale same-frame values distinguished; zero is not a tie in true unseen economy.
- Objective acquisitions in (pre,endpoint] and (L,endpoint] separately: Baron, each elemental dragon, elder, herald, grubs, Atakhan, owned soul; unknown teams/soul0 diagnostic only. Shared/contested acquisitions and combinations may overlap; don't sum subgroup rows as disjoint.

Use existing extraction event counters/StateV2 differences where exact. If unavailable, document limits rather than pretending counters have different temporal meaning. Include counts, sign tables, small-delta proportion and A/B disagreements by objective. No fixed gold exchange weights, no causal objective uplift. Do not compute semantic 'accuracy' using these heuristics as truth.

Saved q predictions can be scored against alternative Y on exact h90 main T/N keys, clearly a diagnostic label-dependence table for a model trained on A. No new model ranking/selection, no claim alternative q performance ceiling. Primary scores must reproduce audited values.

## C. Blinded human review packet, not automated ground truth

Build a deterministic TRAIN-only packet so future human interpretation can inform development without silently tuning on test. 120 unique-match cases maximum: 40 T small abs(A delta)<=.01, 40 T where A disagrees with either B, 20 T observed gold/kill-vs-A disagreement with an objective acquisition, 20 N reference; priority assignment and hash seed20260915, remove duplicates, report shortages and exact inclusion probabilities/denominators. This is stratified case finding, not a representative accuracy sample.

For each case include actual timestamps s/K/L/pre/e, scale, pre/post observed resource summaries with source frame times, both teams' champions, raw kill/objective/structure timeline, and post-L followup context to e. Show unavailable data honestly. Use raw caches for selected matches only. No player/account IDs beyond pseudonymous case IDs in the reviewer view; preserve match mapping in separate private key file in output root.

Reviewer-facing Markdown (Korean labels if practical) must hide A/B probabilities, delta, generated Y, selection stratum, q and W. Separate analysis/key file contains these. Review form asks (a) observed short-term exchange: Blue/Red/neutral/insufficient, (b) strategic advantage over shown interval same choices, (c) endpoint appears to truncate continuing fight/new encounter/unclear, (d) what observation is missing, (e) confidence ordinal. Instructions explicitly say team strategic judgment isn't observed game-winning causal truth. Expert names/ratings empty; never fill with AI judgments. Raw event packet cannot establish positioning/skill accuracy absent telemetry. Record human review UNPERFORMED.

## D. Scientific record and validation

Write status.json with regular real progress, commands/logs/failed runs, REPORT.md in Korean or clear English, DEFINITION_AND_EVIDENCE.md, validation.json and publication tables. Record before (primary labels fixed, unknown alternative stability and semantic ratings) and after (exact computed findings, limitations and unperformed expert review).

Contracts: all data roles; OOF own-match exclusion; preprocessing membership; same-model endpoints; no forbidden future fields; no mutation of frozen artifacts; exact cohort joins and unchanged valid masks; zero-delta/tie logic; event interval counting against raw TRAIN fixtures; normalization source; reload predictions; independently recomputable metrics/bootstrap; reviewer blind-field absence. Verify results and retain unfavorable findings. Do not claim construct validity from numeric tests.

References: reuse audited sources from the ToG audit. Maymin2021 state win probability/change rationale; KimCoG2020 probability-quality motivation (not reproduction); Jacobs/Wallach2021 construct vs measurement distinction; current event-boundary/CIF references support methodological handling, NOT90seconds or true fight attribution. The two alternative feature/C settings, strata and120cases are our prespecified diagnostic choices. Papers do not establish them. End with readiness for human review and targeted follow-up, not a claim validated real fight labels.
