# Approved manuscript integration contract — 16 September 2026

Codex owns scientific design, audit and acceptance; Claude implements this documentation-only revision. The user requested integration of today's evidence into a new Delta V manuscript, reviewer response matrix and collaborator specification, after auditing positional assumptions and excessive conclusions.

## Scope and preservation

- Write only new files under `docs/tog_delta_v_20260916/` and `outputs/manuscript_integration_20260916/`. Do not edit original manuscripts, parent findings, scripts, data, predictions, models, frozen manifests or previous deliverables.
- This is an exploratory working manuscript, not a completed submission, publication, new experiment, causal claim or live deployment. Do not fabricate author affiliations, acknowledgments, repository releases or ethical approval.
- No training, package installations, GPU, raw-data upload, credentials/account identifiers, email, commits or publication. Existing Claude transfer authorization covers this specification, relevant code, aggregates and selected scholarly text. Do not read raw corpus records.
- Read the integration audit inputs placed in the new output directory by Codex. They supersede interpretive claims in earlier findings, not their numerical results. Preserve contradictory historical text as history.

## Deliverables

1. `docs/tog_delta_v_20260916/README.md`: Korean navigation/status, scope, correction summary, completed vs open work, version relationship.
2. `manuscript.md`: coherent English journal working draft with title, abstract, introduction/contributions, narrowly supported related work, full methods, results, interpretation, limitations, conclusion and references. All main sections must be populated; no stale market_event methods/metrics mixed into the current target. Old labels belong only in explicit lineage.
3. `reviewer_response_matrix.md`: English issue-level response to **prior CoG paper 118 feedback informing a journal revision**, explicitly distinguish rejected 118 from accepted/published 308 and from future ToG review. Include original concern, implemented change, evidence, draft section, remaining condition, and status completed/partial/open. Include balanced states and exchange-value coefficients from the collaborator email separately.
4. `collaborator_specification.md`: updated English collaboration document retaining accurate CoG lineage, definitions, worked hypothetical label example, input/output distinction, tables and completed/planned status. Use the previous 20260915 specification as structure/source, not as up-to-date truth.
5. `claim_evidence_ledger.md`: source path/JSON key, scope, value and allowable claim; correction ledger for every overclaim below. Distinguish published precedent, empirical estimate and operational choice.
6. `references.bib` containing only verified, actually used sources. Markdown citations resolve to reference IDs; include source links and exact support limits. Optional LaTeX only if existing tooling allows a faithful complete build without installing dependencies; do not sacrifice the coherent Markdown draft for a broken skeleton.
7. `outputs/manuscript_integration_20260916/writer_receipt.json`: delivered files, checks, source hashes, unresolved limitations, no experiment rerun. Root will independently validate.

## Authoritative evidence to read

- `docs/TOG_END_TO_END_READINESS_AUDIT_20260915.md` and `docs/COLLABORATOR_RESEARCH_SPECIFICATION_20260915.md`.
- `docs/COLLABORATOR_CRITIQUE_RESPONSE_20260915.md` and the incremental-q specification/report/results.
- Findings, REPORT.md, eval/results.json (or mechanism/SHAP summary), validation.json and report_manifest.json for `outputs/{track_a_mlp,horizon_sensitivity,balanced_shap,v_mechanism,champion_class,definition_dev}_20260916/`.
- `outputs/incremental_q_training_20260915/eval/results.json` and the saved independent verification in `outputs/claude_dispatch_incremental_q/`.
- Original manuscript at `C:/Users/todtj/PycharmProjects/LOL_teamfight/docs/tog_manuscript/`, read-only. Reviewer source at `C:/Users/todtj/.codex/attachments/68cfc3ee-c632-4cff-a469-80304ef81c41/pasted-text.txt`, read-only. Both are already authorized relevant research material.
- Codex audit handoff files in the new output directory (position, citation, manuscript map). If those are absent, report missing rather than inventing content.

## Frozen scientific definitions

W is actual final Blue match victory. V(S(t)) estimates P(W=1 | state available by t); current selected V is raw logistic. Delta_i = V(S(e_i))-V(S(pre_i)); Y_i=1[Delta_i>0], exact zero is non-improvement. q(X_pre) estimates P(Y=1 | pre-state), not final match-win probability, delta magnitude, causal effect or fight occurrence.

Retrospective kill-conditioned engagement: first kill K, onset s=K-15s, pre=s-1ms, last kill L. Endpoint e_i(h)=min(L+h, next global kill-1ms, next eligible engagement onset-1ms, match end-1ms). Verify exact implementation inclusion rules and overlaps against the authoritative audit. Primary h90; h60/h120 sensitivity. Objectives do NOT stop the interval. Same V/calibration at both ends; no terminal 0/1 forcing. Next onset is retrospective, so this is not a fully observable live stopping rule. Exclude overlap when next onset<=L. Global next kill may truncate unrelated fighting/pursuit.

G=13.7s and D=4264u are pooled estimates (including TEST patch) with operational choices inside their construction. R=1600 and onset lead15 are not universal learned constants. TRAIN-only G14.0,D4285 is sensitivity, not silent replacement. Do not claim all combat; kill-free engagements are excluded. T and N are disjoint: T at least4 recorded cluster participants per side; N other known scale. State uses latest frame at/before query plus causal event history; no future interpolation. Avoid claiming actual per-role slots.

Split: 15.14 TRAIN74673,15.15 VAL74748,15.16 TEST60579 =210000. Valid rows566104, T113901,N452203, overlap exclusions348. Train valid T39605,N159753; val41315/161855; test32981/130595. External21190 in four KR/NA cohorts, EUW absent. Five-fold own-match-excluded TRAIN V labels; disjoint V_CAL/V_SELECT/Q_CAL/Q_SELECT in VAL. Report V eligibility separately. Prior TEST exposure means all follow-ups exploratory, not untouched confirmatory. Freezing within a follow-up does not change that history.

## Required corrections and restrained conclusions

### Champion class and positions

Parent state builder sorts participants by `(team_id, participant_id)`; class code assigns TOP/JG/MID/BOT/SUP to slot indices. Champion tag frequencies alone do not validate per-match lane roles. Call paired features **slot-index pairs under a positional proxy assumption**, never proven true-role interactions. Integrate the independent positional audit and its exact coverage; do not silently reuse weak role estimates as truth. Tag/static-identity/class-state findings can remain within their representation limits; true role-matchup conclusions remain unestablished.

T LGBM class_pairs-base Brier +0.00013 CI[-0.00009,+0.00035]: no detected benefit in tested setup, not equivalence or absence of learnable information. N +0.00027 CI[+0.00004,+0.00047] worse. Draft AUC near chance is not proof of zero information. Fixed logit C as width grows and limited LGBM budgets constrain interpretation; overfitting is a possible explanation, not proven cause. Track B graphs, finer classes and other learners were NOT tested and are not ruled out. Avoid “nothing to learn”, “graph cannot help”, “unique compositions make synergy unlearnable”, “state absorbs all class effects”. Preserve numerical reproduction caveat: base LightGBM matches parent exactly; logistic has floating-point/solver-path differences, N isotonic can amplify to ~0.0118. Same-run arm comparisons are not automatically immune to numerical uncertainty.

### Increment beyond initial match-win estimate

Comparator PT includes p_pre and time splines/tensor interactions, not p_pre alone. Main TEST T Brier PT0.2295183, full logit0.2284049, LGBM0.2283912; LGBM-PT -0.001127 CI[-0.001588,-0.000647]. N PT0.2450398,logit0.2426897,LGBM0.2386881. T B40 p_pre[.4,.6]: LGBM-PT -0.000435 CI[-0.001442,+0.000601] inconclusive; logit-PT -0.0015685 CI[-0.0030559,-0.0001268] favors logit in an exploratory contrast. Do not generalize the LGBM B40 result to all full-input models. Report B40 row counts and secondary/multiple-comparison caveats. Old specialist is a different budget/history, not the winner of matched Track A selection.

### Matched tabular MLP, horizon and definitions

Track A adds plain/residual MLP with same352 inputs, not the entire CoG eight-model lineup. T plain-LGBM +0.00007 CI[-0.00046,+0.00057] is inconclusive, residual-LGBM +0.00061 CI[+0.00012,+0.00110] worse. N LGBM better than MLPs. Early best epoch alone is not proof of overfitting. Five tabular families total include PT/logit/LGBM/plain/resMLP. FT/TabNet/SAINT and full sequence/graph adapters remain unexecuted.

h60/h90/h120 labels differ: no TEST horizon optimization, no cross-h Brier superiority claim. Broad comparison pattern persists but exact rankings do not (T h60 plain point estimate leads). TRAIN-only boundary sensitivity does not remove original TEST dependence; D4285 is **outside** pooled D95%CI[4256.0475,4273.9677], although G14 lies within temporal stability plateau. Report changed matches3.3–4.4%, removed/added exposure denominators separately, not vague “2% changed”. T refit-frozen -0.00007 CI[-0.00023,+0.00010] inconclusive; N and some external cells worsen. Do not call those equivalence or dismiss differences as mere refit variance without replicated fits.

### V mechanism and SHAP

Delta logit=beta^T(z_post-z_pre), z is actual preprocessing incl interactions. Static additive champion terms cancel from Delta logit/sign, but can still affect p_pre and Delta V magnitude through sigmoid; dynamic effects through game state remain possible. Group mean absolute logit contributions are not shares of total absolute delta nor causal exchange prices. Only pre/post verified, no last-kill intermediate decomposition.

SHAP is grouped interventional explanation of calibrated q using128 TRAIN background and256 selected rows/cell, seven groups/128coalitions. Exact coalition sum conditional on sampled rows/background is not full-population inference. Masked p_pre not recomputed from masked states; correlated features can generate off-manifold states. Relative objective/economic share increase in B40 mostly reflects diminished p_pre contribution, not increased absolute or causal effect. Different learners distribute attribution differently. Descriptive row resampling is not match-bootstrap population uncertainty. Include SHAP as interpretability evidence with these limits.

## References and review mapping

Use verified bibliographic handoff: Kim2020 for confidence-aware MOBA prediction precedent (its loss not implemented); Hodge for live Dota2 win prediction, not a LoL replication; Maymin for state-conditioned valuation precedent, not exact replication/causal identification; Halfaker for sessionization analogy, not identical threshold algorithm; Schubert/Drachen/Mahlmann for encounter analyses, not proof of our constants; Lundberg/Lee for attribution principles, not causal explanation.

All assumptions/results must have local evidence anchors or primary references appropriate to the claim. Research choices need explicit rationale and sensitivity, not a fabricated citation proving the choice. Label semantic validity remains limited without independent outcome criterion; user declined manual review, so do not make human review a new mandatory blocker. Address alternative valuation/ablation already run where supported, while noting that learned V still induces a model-defined label.

## Acceptance

- All three narrative deliverables agree on definitions, split/counts, primary h90, target, matched inputs, exploratory status and open work.
- Tables extracted/checked against JSON evidence; no copying of disproven interpretive claims. Each numerical claim linked to source/JSON path in ledger. No fabricated p-values, CIs, significance or completed tests.
- All used citations resolve with verified metadata and narrow support. All local links resolve.
- Original manuscript and experiment evidence hash inventory unchanged; no experiment launch.
- Deliverable status clearly states this is an integrated draft awaiting remaining scientific/submission tasks, not submission-ready.
