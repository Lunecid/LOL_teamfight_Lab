# DeltaV manuscript integration map (2026-09-16)

This is a bounded, read-only exploration handoff. The original draft and all
experiment outputs were left unchanged; no training, raw-data access, or
compiler installation was performed. The approved contract is
docs/CLAUDE_MANUSCRIPT_INTEGRATION_20260916.md:L1-L76.

## 1. Relevant files and symbols

| Area | File(s) and anchors | Finding |
|---|---|---|
| Integration contract | docs/CLAUDE_MANUSCRIPT_INTEGRATION_20260916.md:L5-L76 | The writer must create only the six narrative/bibliography files under docs/tog_delta_v_20260916/ plus writer_receipt.json; the result is an exploratory working draft, not a submission, new experiment, causal claim, or deployment. |
| Immutable original wrapper | C:/Users/todtj/PycharmProjects/LOL_teamfight/docs/tog_manuscript/main.tex:L1-L57; macros.tex:L1-L10 | LaTeX wrapper is an old CoG/market_event-oriented draft. It inputs intro, related, background, definition, label, prediction, learners, limit, limitations, availability, and a missing sec_conclusion.tex. It uses \bibliography{../references/definition_refs}. |
| Immutable original sections | .../sec_intro.tex; sec_definition.tex; sec_label.tex; sec_prediction.tex; sec_learners.tex; sec_limit.tex; sec_limitations.tex; sec_availability.tex | Core label/prediction/learner/limit/limitation sections contain old market_event text and old headline metrics. They are lineage/context only and should not be copied as current DeltaV results. Background and related work are selectively reusable after scope/citation edits. |
| Original review matrix | .../docs/tog_manuscript/reviewer_response_matrix.md:L1-L55,L58-L327 | This matrix is explicitly for rejected CoG submission 118. Its quoted review text is not stored; most rows paraphrase a plan. The old status tally and v3.3 market_event block are stale. All issue IDs below come from this matrix. |
| Review lineage | docs/REVIEW_LINEAGE_AND_BOUNDARY_PLAN_20260914.md:L7-L20,L22-L37 | 118 was rejected despite the meta recommendation; 308 was a later accepted auxiliary submission. 118 source: C:/Users/todtj/.codex/attachments/68cfc3ee-c632-4cff-a469-80304ef81c41/pasted-text.txt. 308 source: C:/Users/todtj/.codex/attachments/74026d44-604f-40fe-ab45-5507d0b8bb8f/pasted-text.txt. |
| Current audit | docs/TOG_END_TO_END_READINESS_AUDIT_20260915.md:L18-L43,L45-L70,L72-L175,L223-L247 | Current question is q=P(Y=1 | pre), with W final Blue win, V(S(t))=P(W=1 | state by t), Delta=V(post)-V(pre), and Y=1[Delta>0]. The original manuscript is not yet integrated with this target. |
| Sept16 evidence | outputs/track_a_mlp_20260916/, horizon_sensitivity_20260916/, balanced_shap_20260916/, v_mechanism_20260916/, champion_class_20260916/, definition_dev_20260916/ | Each root has a report, protocol, frozen manifest, validation, and report manifest. Validation counts are respectively 115, 91, 36, 24, 91, and 53. Use each report plus its JSON result; do not infer untested learners or endpoints. |
| Existing handoff inputs | outputs/manuscript_integration_20260916/aggregate_verification.json; numeric_evidence.json/.md; position_audit.md; citation_audit.md; source_snapshot.json | These are already present for the writer. At inspection, docs/tog_delta_v_20260916/ did not yet exist; it is the isolated destination. |

## 2. Data and evidence flow

1. Timeline events are converted by gameplay.fights.detect_fights,
   _fight_to_ref_row, and analyse_match into retrospective engagement rows.
   The current detector uses K=first kill, s=K-15 seconds, pre=s-1 ms,
   L=last kill, and e=min(L+h, next global kill-1 ms, next eligible onset-1
   ms, match end-1 ms). The main cap is h=90 s; h=60/120 are sensitivity
   endpoints. Objective acquisition does not stop the interval.
2. StateV2 is built from the latest frame at or before the query with causal
   history. Its participant order is sorted by (team_id, participant_id);
   V uses 361 inputs and q uses the current 352-input feature family. V is
   fit/calibrated/selected with own-match exclusion, then frozen for DeltaV.
3. The q endpoint predicts the direction of the model-derived change, not
   final victory probability, Delta magnitude, a causal effect, or live
   fight/onset detection. The frozen main split is 15.14 train, 15.15
   validation, 15.16 test; later/regional cohorts are external and already
   exposed to prior project work.
4. Main counts are 74,673/74,748/60,579 matches by split (210,000 total);
   valid rows are 113,901 T and 452,203 N, with 348 overlap exclusions.
   External data cover 21,190 matches across four KR/NA partitions; EUW
   is absent. The writer should use the current report JSONs for exact metric
   values and confidence intervals.
5. Findings add orthogonal evidence: h60/h90/h120 endpoint sensitivity;
   same-input plain/residual MLP versus logistic/LightGBM; grouped V
   logit-mechanism decomposition; grouped, interventional sampled SHAP for q;
   champion-class arms; and a separate last-observed-position ablation.
   These are exploratory after existing TEST exposure and do not create a
   fresh confirmatory test set.

The positional audit is especially relevant to class wording. StateV2 slot
order is participant order, while the class code names those slots
TOP/JG/MID/BOT/SUP by convention. Stored ordering disagreements with cached
role slots are 1,255/74,673 TRAIN (1.68%), 1,251/74,748 validation (1.67%),
1,042/60,579 TEST (1.72%), and 3,548/210,000 overall (1.69%). These rates
are order statistics, not per-slot role accuracy. Therefore write
“slot-index pairs under a positional proxy assumption,” never “true-role
interactions” or “same-role matchup.”

## 3. Reviewer lineage and issue map

The new matrix should say explicitly that these are 118 issues informing a
journal revision. 308 is accepted auxiliary lineage, not the source of the
118 issue IDs, and no separate future ToG review was found.

| New response area | Original IDs | Current evidence / status boundary |
|---|---|---|
| Abstract, contribution framing, background, self-contained exposition | R1-1, R1-2, R1-3, R1-6, R3-6 | Rewrite around DeltaV/q and bounded scope. Community-impact and broad contribution claims remain citation/interpretation work, not measured evidence. |
| Definition, label, constants, kill-free scope, API limits | R1-4, R1-5, R1-7, R1-8, R4-1, R4-2, R4-3 | Use the frozen detector and DeltaV definitions. Definition-dev supports a TRAIN-only G/D sensitivity (pooled G=13.7,D=4264 versus TRAIN-only about G=14.0,D=4285), but it does not establish equivalence or remove the operational choice. |
| Hand-set weights, raw/learned label alternatives, sensitivity | R2-2, R2-3, R2-4 | Current primary Y is model-defined DeltaV direction. Keep collaborator exchange-value coefficients in a separate operational subsection/spec; do not call them causal prices or ground truth. Monetary/independent semantic validation remains partial/open. |
| Model fairness, same inputs, architecture coverage | R2-5a, R2-5b, R2-5c, R2-6, R2-7, R2-10, R3-2, R3-3 | Track A is a bounded 352-input comparison with a small fixed budget and three seeds. It supports only the tested MLP/logistic/LightGBM comparison. FT-Transformer, TabNet, SAINT, short-sequence Transformer, attention, and CNN probes remain unexecuted. |
| Temporal resolution, window, and technical motivation | R2-8, R2-9, R3-1, R3-4 | h60/h90/h120 and V-mechanism results support a sensitivity/notation discussion. h90 is practical main choice, not a universal optimum; Brier values across horizons are different endpoints. |
| Attribution and explanation | R3-5 (also R3-1) | Use V delta-logit identity and balanced grouped SHAP. Contributions are descriptive; SHAP is conditional on sampled rows/background and 128 coalitions, not population, causal, or economic attribution. |
| Limitations, public release, and reproducibility | R1-9, R2-1 | Rewrite limitations and future work. Availability links, release DOI, and data licence are unresolved author/submission items; do not invent them. |
| Historical pipeline/leakage and position concerns | X-1, X-2, X-3, X-4 | Keep old CoG/market_event and leak-exposed SAINT as explicit lineage only. Do not transfer .6699 market_event or old SHAP/stacking claims. Position ablation is a simple 363/383-input LightGBM sensitivity (15.16 AUC +0.0031, CI [-0.0029,+0.0093]); it does not prove positional utility or role accuracy. |

Do not copy the original matrix’s old “done/running/writing/pending” tally.
The new matrix must add, for every row, original concern, implemented change,
evidence, draft section, remaining condition, and completed/partial/open status.
Where a verbatim response is needed, retrieve it from the 118 or 308
attachment above rather than treating the matrix paraphrases as quotations.

## 4. Rewrite versus preserve

| Surface | Treatment |
|---|---|
| manuscript.md | Write as one coherent English working draft with title, abstract, introduction/contributions, narrowly supported related work, background, full definitions/methods, q/V results, interpretation, limitations, conclusion, and references. All sections must be populated; use local evidence anchors. |
| Old intro/label/prediction/learners/limit/limitations | Rewrite substantially. Their market_event label, 532,547/191,940 headline, old architecture claims, and old causal/economic language are not current DeltaV evidence. |
| Definition/background/related work | Preserve only audited detector/formal notation, LoL/API explanations, and verified precedent framing. Update counts, split, h endpoint, state ordering, and source limits. |
| Availability | Reuse structure, but replace old release/count claims and leave unresolved release/licence/anonymous-link facts open. |
| collaborator_specification.md | Use docs/COLLABORATOR_RESEARCH_SPECIFICATION_20260915.md as structure/source: retain accurate CoG lineage, definitions, worked hypothetical label example, input/output distinction, tables, and planned/completed status. Reconcile it with the current contract and keep exchange coefficients separate from DeltaV Y. |
| claim_evidence_ledger.md | For every numerical or strong interpretive claim record source path/JSON key, scope/value, and allowable wording. Add correction entries for role assumptions, class non-equivalence, V/SHAP attribution, G/D, horizon, external exposure, and architecture limits. |
| references.bib | Create a small verified bibliography for sources actually cited. Existing docs/references/definition_refs.bib is a source pool, not a reason to copy all entries. Use outputs/manuscript_integration_20260916/citation_audit.md for verified facts and bounded citation roles. |
| README.md and writer_receipt.json | README should state lineage, scope, evidence roots, and open conditions. Receipt should list files/checks/hashes and explicitly say no rerun/PDF compilation. |

## 5. Constraints, risks, and build status

- Preserve the frozen 15.14/15.15/15.16 split and distinguish validation,
  external, and already-exposed TEST evidence. Do not report the external
  cohorts as untouched confirmatory tests.
- Keep pooled G=13.7/D=4264 as an operational setting and report the
  TRAIN-only sensitivity separately. D=4285 lies outside the pooled D 95% CI
  [4256.0475, 4273.9677]; this is not an equivalence result.
- Class frequencies and class-pair results are exploratory slot features.
  The primary T class_pairs-base contrast is +0.00013 Brier with CI
  [-0.00009,+0.00035]; do not convert this into “no information,” “nothing to
  learn,” or a universal graph/role conclusion.
- The V mechanism identity is exact for the evaluated pre/post rows, but
  group mean absolute logit contributions are not shares of total DeltaV,
  causal values, or exchange prices. Balanced SHAP uses interventional masking
  with a finite background sample; masked p_pre is not recomputed.
- Track A, horizon, class, mechanism, SHAP, and definition reports are
  evidence for the specified scopes only. Do not generalize limited learner
  budgets or a h90 ranking to all architectures, patches, or deployments.
- Existing output handoffs include aggregate verification, numeric evidence,
  positional audit, and citation audit. The writer should preserve their
  source paths and use the report manifests/frozen manifests rather than
  displaying raw identifiers.
- Original main.tex documents “latexmk -pdf main.tex” and assumes a build from
  docs/tog_manuscript. Current environment has pandoc, but latexmk, pdflatex,
  xelatex, lualatex, tectonic, bibtex, biber, and make are unavailable on PATH.
  The contract requires Markdown, so no install or compiler chase is warranted;
  receipt status should say LaTeX/PDF was not built.

## 6. Isolated destination

Create only:

    docs/tog_delta_v_20260916/
      README.md
      manuscript.md
      reviewer_response_matrix.md
      collaborator_specification.md
      claim_evidence_ledger.md
      references.bib

The writer receipt belongs at
outputs/manuscript_integration_20260916/writer_receipt.json. This map and the
existing audit/evidence handoffs in that output directory are inputs, not
replacement manuscript deliverables. Original C:/Users/todtj/PycharmProjects/
LOL_teamfight/docs/tog_manuscript files, source findings, scripts, manifests,
models, and raw data remain immutable.
