# Reviewer-defense dossier — journal_manuscript_v1 (IEEE Transactions on Games)

**Date:** 2026-09-21 · **Status:** internal preparation memo, not manuscript text · **Base:** `docs/journal_manuscript_v1/` after T012, `JOURNAL_FINISH_LOCK_20260920.md`, `RESPONSE_EVIDENCE_MATRIX_20260920.md`
**Rules:** no new numbers; every figure is transcribed from the cited document and field. Where an answer needs work the freeze does not contain, the entry says *would need* and defers to thesis/v2. Nothing here proposes a claim on the Forbidden lists (F1–F8 in `01_OUTLINE_AND_CLAIM_EVIDENCE.md`; `JOURNAL_FINISH_LOCK` "Forbidden manuscript claims"; `SCALE_SPLIT_EXPERIMENT_CONTRACT` §6).

Method: `reviewer-defense` (vulnerability scan → ranked questions → allocation → negative-results framing → edits) and `blindspot-pass` (landmines, hidden context, what good looks like, questions to ask) in §5.

---

## 1. Weakness table

Severity: **high** = possible rejection reason; **med** = major-revision item; **low** = polish. Fix: *text* (wording), *artifacts* (numbers exist in `docs/`), *new work* (not this paper).

| # | Type | Weakness | Sev. | Where | Fix |
|---|---|---|---|---|---|
| W1 | Experimental | Primary contrasts carry prior TEST exposure; no confirmatory test exists; PT_flex execution deviated from design (uniform knots, un-normalized weights, two-stage) | high | `02_METHODS` §0, §4.2; RR1 addendum | text (disclosed once, early) |
| W2 | Technical | Label is model-defined (\(Y_{\mathrm{SVI}}=\mathbf 1[\Delta\widehat V>0]\)); all predictability claims conditional on fit85 and the endpoint rule | high | `02_METHODS` §2–§3; `04_DISCUSSION` §3 | text + artifacts (OOF two-path table; RR5/RR6b) |
| W3 | Experimental | EXT, CORP, \(V\to W\), RR5, RR6b carry **no intervals**; "lift does not hold" is a point ordering | high | `02_METHODS` §6; `03_RESULTS` §5 | text now; intervals = new work |
| W4 | Presentation | Absolute lift small (ΔBrier ≈ −0.004 vs UNC ≈ 0.25); reads as a null unless framed as verification | high | `03_RESULTS` §3.1; `04_DISCUSSION` §1; `WORKING_NOTES` §A | text |
| W5 | Technical | logit \(q\) (352 inputs) vs LightGBM (362) is not same-input; J-RQ2's learner / information-set clause unsupported | med | `01_OUTLINE` C9; `02_METHODS` §4.1; `WORKING_NOTES` §D | text + author RQ decision |
| W6 | Experimental | T B40 upper bound −0.00003; \(H\) covers zero; three printed upper bounds across docs | med | `03_RESULTS` §3.2; `.ai/CONTEXT.md` | text (RR12 field cited) |
| W7 | Presentation | Two cohorts side by side invite the cross-cohort comparison the paper refuses | med | `03_RESULTS` §3.1; `04_DISCUSSION` §5 | text (heterogeneity sentence; §pred-cut) |
| W8 | Technical | S uses contract-literal identity although the RR12 rule chose a sigmoid for every S model | med | `02_METHODS` §4.2; `03_RESULTS` §3.1 | text (sensitivity row exists) |
| W9 | Experimental | Single-seed evaluator (GRU within ≈0.0002 on V_SELECT); intervals exclude fitting and label-model uncertainty | med | `02_METHODS` §2, §6 | text; multi-seed = new work |
| W10 | Experimental | Definition sensitivity (G×D, presence gate, window) measured only on conference-era `market_event` corpora | med | `02_METHODS` §1 last para; matrix R1-5 | text; re-run = new work |
| W11 | Presentation | RQ wording undecided; `\pending{pick-exclusion-wording}`, `\pending{ext-TS-sentence}` open | med | `01_OUTLINE`; `.ai/RQ_DECISION_BRIEF.md` | author decision, text |
| W12 | Presentation | Intro, Abstract, Related Work, Conclusion, figures absent; `sec_learners` pendings could re-lock the frame | med | `README.md`; `00_MIGRATION` | text (next pass) |
| W13 | Experimental | Training corpus = KR Master-and-above; raw 15.x data no longer held | med | `DATA_AVAILABILITY.md` §4–§5 | text (limitation + availability) |
| W14 | Technical | Cause of external loss unidentified (patch / region / \(q\) calibration / \(V\)); no adapter validated | med | `Q_NEWV_FIT85_TRANSFER_16X` L13 | text; adapter = thesis/v2 |

---

## 2. Anticipated reviewer questions (ranked by likelihood)

**Q1. The improvement over PT_flex is ≈0.004 Brier. Why is this a contribution?**
*Motivation:* W4; journals press "so what".
*Answerable:* Yes — `03_RESULTS` §3.1 (T −0.00373 [−0.00461, −0.00279]; S −0.00335 [−0.00379, −0.00288]); `04_DISCUSSION` §2.
*Draft response:* The contribution is the measurement object and the verification stack, not the size of the lift: a match-linked direction label under a frozen evaluator, a flexible \((p_{\mathrm{pre}},t)\) stress baseline, and an explicit test of where the lift survives (small-|ΔV| filters, balanced states, later patches, another region). The interval-excluding lift recurs under the same fixed procedure in both cohorts, and we state that it is small and not an operational tip engine.

**Q2. The test patch was used in earlier analyses. How is this not post-hoc?**
*Motivation:* W1.
*Answerable:* Yes — `02_METHODS` §0; `REVIEW_RESPONSE_RR1_EXECUTION_ADDENDUM_20260920.md`; `REVIEW_RESPONSE_RR0_MANIFEST_20260920.json`; `SCALE_SPLIT_EXPERIMENT_CONTRACT` header and §5.
*Draft response:* Every primary contrast is labeled exploratory in Methods §0. Selection of \(q\), PT_flex knots and calibrators was done on 15.15 Q_SELECT and weights were frozen; the S block ran under a predeclared internal contract fixing cohort, arms, one primary contrast and forbidden readings before any S label existed. No candidate was selected on TEST; λ slices are labeled post-hoc and none is promoted.

**Q3. \(Y_{\mathrm{SVI}}\) is defined by \(\widehat V\) and \(q\) receives \(p_{\mathrm{pre}}=\widehat V(x_{\mathrm{pre}})\). Is \(q\) learning anything except \(\widehat V\)'s own structure?**
*Motivation:* W2.
*Answerable:* Yes — `02_METHODS` §2, §5 "Two evaluator paths"; §4.2 (spline tensor product in \(p_{\mathrm{pre}},t\)); `03_RESULTS` §2.3–§2.4 (RR5a agreement 0.904; peer evaluators ≈0.91).
*Draft response:* PT_flex already contains any smooth function of \(p_{\mathrm{pre}}\) and time, so the contrast isolates information beyond the evaluator's snapshot value; TRAIN labels come from match-fold evaluators that never saw the fold's matches, and the frozen bundle never labels its own training matches. Every claim remains conditional on fit85 and the endpoint rule (Discussion §3); correspondence with kill nets and next objectives shows SVI is not arbitrary, not that \(q\) is accurate.

**Q4. The lift disappears on KR/NA1 16.13. Doesn't that undercut the headline, and why no intervals?**
*Motivation:* W3, W14.
*Answerable:* Partly — `REVIEW_RESPONSE_RRX_EXTERNAL_20260920.md` summary table (ΔBrier +0.0026 / +0.0040; ΔMCB 0.0045 / 0.0063; ΔDSC 0.0019 / 0.0023). *Would need:* match-cluster bootstrap on the RRX predictions and a cause decomposition — thesis/v2, not this paper.
*Draft response:* The external result is half the thesis: scoped predictability plus its transfer limit. On the main 16.13 cohorts the 15.16 ordering reverses at the point-estimate level, with discrimination still favouring \(q\) and a larger miscalibration component dominating; we computed no intervals there and say so, and we do not claim that recalibration would repair transfer.

**Q5 (CoG-R2.2 carry-over; collaborator C7). Only a logistic model? Where are deep baselines, and why is LightGBM not on the same inputs?**
*Answerable:* Yes — `02_METHODS` §4.1, §8; `Q_NEWV_FIT85_PRIMARY_20260920.md` L27–28 (Q_SELECT Brier logit 0.2339 vs lgbm 0.2383); F4. *Would need:* same-input learner ladder — thesis/v2.
*Draft response:* This version does not rank learners: the question is whether pre-state carries information beyond \((p_{\mathrm{pre}},t)\), and a regularized logistic model on 351 numeric features plus \(p_{\mathrm{pre}}\) answers it. The LightGBM candidate used champion IDs as extra categorical inputs and lost on Q_SELECT; we state that no same-input comparison is made and no architecture claim follows.

**Q6 (collaborator: CI coverage). Which numbers carry uncertainty?**
*Answerable:* Yes — `02_METHODS` §6; `01_OUTLINE` Uncertainty column; matrix "Uncertainty coverage".
*Draft response:* Match-cluster percentile bootstrap intervals (2 000 draws, seed 7) accompany every paired ΔBrier contrast and the matched quiet contrast. CORP components, \(V\to W\), horizon, next-objective and external tables are point values, each marked "no interval", and conclusions from them are worded as observed orderings.

**Q7. B40 (T): the interval touches zero and \(H\) covers zero. What is claimed?**
*Answerable:* Yes — `03_RESULTS` §3.2 (−0.00214 [−0.00422, −0.00003]; \(H\) 0.00181 [−0.00060, 0.00422]); F2.
*Draft response:* Only that a small exploratory improvement is observed in the balanced band and that we cannot conclude balanced states are harder. We claim no gain of at least 0.001 and do not read B40 as evidence about unobserved combat execution.

**Q8 (CoG-R1.4 carry-over). The detector constants look arbitrary; do results depend on them?**
*Answerable:* Partly — `02_METHODS` §1 constants table (G, D data-derived with CIs; per-patch re-estimation inside the plateau); `tog_manuscript/reviewer_response_matrix.md` R1-5. *Would need:* re-detection under the SVI pipeline — thesis/v2.
*Draft response:* Every constant is tabulated with value and source class; G and D are estimated from kill data with intervals and re-estimated per patch. Sensitivity of the winner-label lineage to G×D and the presence gate was measured on conference-era corpora and is cited as lineage, not re-run under the frozen evaluator.

**Q9. You print T and S side by side but refuse to compare them.**
*Answerable:* Yes — `03_RESULTS` §3.1 (rows 32 981 vs 101 205; positive rate 0.498 vs 0.511; B40 share 16.4 % vs 31.3 %); `sec_prediction.tex` §pred-cut; contract §6.
*Draft response:* The cohorts differ in size, base rate and balance share, and in the definition lineage the class gap changes sign with the teamfight cut, so absolute scores are not on a common scale. The only cross-cohort statements are the predeclared transfer and pooling contrasts on identical rows (Results §3.4), which say nothing about the relative predictability of the two scales.

**Q10 (collaborator: RQ structure). J-RQ2 asks about learner choice and information sets; neither is tested.**
*Answerable:* Status only — `WORKING_NOTES` §D / `.ai/RQ_DECISION_BRIEF.md` (option C recommended); author decision pending.
*Draft response (after decision):* The questions are measurement (J-RQ1), predictability beyond a flexible function of initial win probability and time (J-RQ2), and scope of that lift across balanced states, small-change filters, cohorts and later environments (J-RQ3); learner comparison is outside this version.

**Q11 (collaborator: units, denominators). Is ΔV in probability or percentage points? What are the RR5b denominators?**
*Answerable:* Yes — `02_METHODS` §3 (0.70→0.60 is −0.10, i.e. −10 pp); `03_RESULTS` §2.3 (6 004 of 9 934; 4 072 of 10 277; unweighted; 8 112 game-ended windows); §1 (347 234 queries, 60 190 matches).
*Draft response:* ΔV̂ is a probability difference; percentage points appear only when multiplied by 100 and labeled. Next-objective shares are unweighted counts among decided windows, neither win rates nor \(q\) accuracy.

**Q12 (CoG-R2.1 carry-over). The old label had hand-set gold weights. What does the new label decide by rule?**
*Answerable:* Yes — `02_METHODS` §1b (follow-up mean 45.493 s; 14.286 % reach the cap; ≈78.9 % stop at the next kill), §3 (ΔV̂=0→0; one exact zero in TRAIN); `03_RESULTS` §2.4 (flips 0.007–0.019; same-endpoint 0.59–0.81).
*Draft response:* The gold label is demoted to CoG continuity; the primary label is the sign of the evaluator change to a rule-defined endpoint capped at the next kill, next engagement or match end. We report that "h90" is not a 90 s window, that horizon agreement is inflated by shared endpoints, and that exact zeros are coded 0.

**Q13 (CoG-R4.1 carry-over). What about fights without kills?**
*Answerable:* Yes — `02_METHODS` §1 last paragraph; matrix R4-1 (lineage count, 20 000 matches, v3.3).
*Draft response:* The detector is kill-anchored, so kill-less exchanges are not instances and not the target; the lineage quantified them on a pilot corpus, and this paper carries that as a scope limitation.

**Q14. Training data are KR Master-and-above. Does anything generalize?**
*Answerable:* Yes — `DATA_AVAILABILITY.md` §5; `sec_limitations.tex` "What the corpus covers"; `03_RESULTS` §5 (NA1 16.13).
*Draft response:* The corpus is a high-rank single-region collection with tier as a collection property; the external panel adds a later patch on the same region and on NA1, and the transfer limit is the generalization result. No claim is made beyond these environments.

**Q15. Why the identity calibrator for S when your own rule chose a sigmoid?**
*Answerable:* Yes — `02_METHODS` §4.2; `03_RESULTS` §3.1 (identity −0.00335; sensitivity −0.00346 [−0.00382, −0.00307]).
*Draft response:* The contract fixed identity a priori and defines \(q_{T\to S}\) as the frozen T model applied unchanged; the selection variant is a sensitivity row with the same sign and similar size.

**Q16. Single seed for \(\widehat V\) and \(q\); do intervals cover fitting variance?**
*Answerable:* Yes — `02_METHODS` §2, §6. *Would need:* multi-seed refits — thesis/v2.
*Draft response:* Intervals describe match resampling under fixed evaluator, labels and selection; fitting and label-model uncertainty are excluded and listed as limitations. Retuning or seed-averaging \(V\) would redefine the label, so it is deferred.

**Q17. Your AUC is 0.64; the conference winner task reached ≈0.67.**
*Answerable:* Yes — `README.md` rule 2; `00_MIGRATION`. *Response:* different estimands on different rows; not comparable, never tabulated together.

---

## 3. Main text vs appendix allocation

| Item | Place | Justification |
|---|---|---|
| Methods §0 epistemic status; one paragraph on RR1 deviations | Main | Required disclosure; addendum detail to appendix |
| Constants G, D, R, B, M; full 20-row table | Main (5 rows) / Appendix | R1.4 needs data-derived constants visible; the rest is reference |
| Table 5b engagement roles (T, S) | Main | Splits and label paths are the credibility backbone |
| Table 5a evaluator census | Appendix | Measurement context |
| Results §1 \(V\to W\) (overall, early band, pre/post) | Main, compact | Three numbers establish the evaluator |
| §2.1 triad by \(p_{\mathrm{pre}}\) | Main, one figure | Motivates binary SVI |
| §2.2 RR3 quiet contrast | Appendix | Coverage ≈28 %; not ATT; easily over-read |
| §2.3 RR5a/RR5b correspondence | Appendix | Correspondence only; F5 risk in main |
| §2.4 RR6b horizons, peer evaluators | Appendix + one main sentence | Stability check |
| §3.1 primary panel T and S | Main | The paper |
| §3.2 B40 T and S | Main | Key conditional; honesty must stay visible |
| §3.3 CORP tables | Appendix; ΔMCB/ΔDSC columns stay in the EXT table | Diagnostic; needed only to read EXT |
| §3.4 transfer/pooling (4 rows) | Main | Only permitted cross-cohort statements |
| S secondary rows, Table 1b sensitivity, lgbm diagnostic | Appendix | Support §3.4 and Q15 |
| §4 λ·\(s_Q\) sensitivity | Appendix + one main sentence | Post-hoc by definition |
| §5 EXT KR/NA1 16.13 (T and S rows) | Main | Transfer limit is half the thesis |
| KR 16.15, 16.14 pilot rows | Appendix or footnote | Small; reported, not pooled |
| 2026-09-15 N-cohort lineage | Appendix only, if at all | Old evaluator; not comparable (§4d) |
| §6 summary-of-answers table | Main | Answer map for journal readers |

---

## 4. Negative-results framing (draft sentences)

**(a) External transfer loss on T.** "On the two main external cohorts (KR 16.13, n = 5 202; NA1 16.13, n = 5 312) the 15.16 Brier advantage of \(q\) over PT_flex is not observed: ΔBrier is +0.0026 and +0.0040 (point estimates; no interval computed) while \(V_{\mathrm{pre}}\) Brier remains 0.1511 / 0.1514. The decomposition places the reversal in a larger miscalibration component (ΔMCB 0.0045 / 0.0063) with discrimination still favouring \(q\) (ΔDSC 0.0019 / 0.0023). We report this as the transfer limit of the frozen procedure; the cause is not identified here and no adapter was fitted or validated." [RRX summary table; `Q_NEWV_FIT85_TRANSFER_16X` L13]

**(b) B40 interval touching zero on T.** "Within the balanced band (n = 5 423 from 4 945 matches) the paired contrast is −0.00214 with interval [−0.00422, −0.00003], whose upper end lies at zero to the reported precision, and \(H\) = 0.00181 [−0.00060, 0.00422] covers zero. We describe this as a small exploratory improvement that neither establishes a gain of at least 0.001 nor shows that balanced states are harder." [RR12 `bootstrap.B40_q_minus_PT_flex`, `bootstrap.H_B40_minus_outside`]

**(c) Pooling T∪S costing T.** "A single model fitted on the union of T and S training rows (162 654 rows, 65 465 matches) improves neither cohort: on S rows \(q_S - q_{TS}\) = −0.00024 [−0.00039, −0.00009], and on T rows \(q_{TS} - q\) = +0.00071 [+0.00018, +0.00126], so pooling costs the teamfight model a small, interval-excluding amount. Cohort-specific models are retained; the contrast describes two fitted models on identical rows and carries no mechanism." [SS `table2a_identity`; `.ai/reports/logs/T009_union.txt`]

**(d) N-cohort specialization, 2026-09-15 lineage (appendix only, if at all).** "In an earlier lineage under a previous logistic evaluator and without a \((p_{\mathrm{pre}},t)\) baseline, refitting the same candidate pool per cohort gave a teamfight specialist −0.00166 [−0.00235, −0.00104] against a pooled model, whereas the non-teamfight specialist gained −0.00042 [−0.00057, −0.00024] (skirmish rows −0.00023 [−0.00043, −0.00004]); that lineage judged non-teamfight specialization not robust and withdrew its mechanism explanation. Because evaluator, labels and baselines differ, these figures are lineage context and are not placed beside the fit85 results." [`SCALE_SPLIT_EXPERIMENT_CONTRACT` §1; `.ai/CONTEXT.md`]

---

## 5. Blindspot pass — unknown-unknowns for a first journal submission

Starting point inferred: one rejected 8-page conference submission, no prior journal submission, evidence package closed, spine mid-draft.

### Landmines

1. **Double-anonymous review.** ToG reviews double-anonymously for submissions from 2025-01-01 (`DATA_AVAILABILITY.md` §6). The CoG submission number, its reviews, repository URL, GitHub owner and machine paths all break anonymity. *Action:* run `scripts/prepare_anonymous_release.py` on the release commit, resolve its manual-review lines (§7), strip "CoG 2026" references, and describe continuity without identifiers.
2. **The availability statement describes the wrong paper.** `sec_availability.tex` / `DATA_AVAILABILITY.md` inventory the v3.3 `market_event` package; the freeze artifacts (fit85 bundle `ac459cc4397630a9`, five OOF evaluators, `logit_state` weights, RR12/RRX/SS predictions, `SCALE_SPLIT_RR0_MANIFEST`) are not listed. *Action:* rebuild the inventory around the RR0 and scale-split manifests; decide whether joblib bundles are released.
3. **Raw data gone; API retention unchecked** (`DATA_AVAILABILITY.md` §4, §2.3). *Action:* phrase reproducibility as "from released derived tables" plus "from match identifiers while Riot serves them"; never promise full re-collection.
4. **Code lineage split across branches.** The 2026-09-15 cohort-role code and `outputs/` are not on this branch; only six rule files are copied to `docs/lineage_20260915/` (`.ai/CONTEXT.md`). *Action:* fix a release commit containing both lineages, or state which parts are copies.
5. **Riot terms and player data.** Derived tables are neither expressly authorised nor forbidden; data licence and written confirmation are open (`\pending{data-licence}`). The 2026 collector stores rank/tier/LP snapshots per PUUID in SQLite state (`ACQUISITION_2026.md` L52). *Action:* add an ethics/data statement for both collections — public API, no human-subject interaction, no player identifiers in released tables, per-PUUID state kept only for acquisition and not analysed — and settle the licence before submission.
6. **Patch labels.** API 15.14–15.16 = public 25.14–25.16 (`DATA_AVAILABILITY.md` §5); 16.x = 26.x (`ACQUISITION_2026.md` L57–58). *Action:* state the mapping once; confirm 16.13 collection dates from the collector state.
7. **"Predeclared" is not "preregistered".** The scale-split contract is internal, dated by commit, not externally time-stamped. *Action:* write "predeclared internal contract (commit `78695a6`)"; never "preregistered".

### Hidden context

- Three RQ sentences are LOCKED (`JOURNAL_RESEARCH_PLAN` §6, `J_RQ1_SCOPE_LOCK` §1, `COG_SUCCESSION_LOCK` §5); any change needs an author decision and dated history lines.
- `docs/tog_manuscript/` is read-only lineage; its `sec_learners` pendings must not be completed for this version (`00_MIGRATION`).
- Zotero returned 0 hits for Maymin / CORP; the keys live in local `.bib` files (`ACADEMIC_PACKAGE_AUDIT` §5). *Action:* run citation checks against the local bibs; invent no DOIs.

### What good looks like

- `DATA_AVAILABILITY.md` §3: every statement about terms quotes the clause and read date and says what the terms do *not* say — the discipline the freeze paper's availability section should keep.
- `02_METHODS` §9 evidence trace: one row per manuscript number with file and field. A supplementary "number ledger" built from it pre-empts Q6 and Q11.

### Questions an experienced submitter would ask

1. *Current ToG page budget and supplementary policy?* Not recorded; `DATA_AVAILABILITY.md` §6 names the page but no limit. **Action:** read it and size §3 to it.
2. *Does IEEE require disclosure of AI assistance in writing?* Not recorded in the repo. **Action:** check the IEEE Author Center policy and add the acknowledgement it requires; this workflow used AI agents for drafting.
3. *Which figures exist?* None. **Action:** plan three or four (triad by \(p_{\mathrm{pre}}\) bin; ΔBrier forest plot for T and S contrasts; endpoint-rule schematic; reliability/CORP diagram), each traceable to a JSON field, colour-blind-safe, not colour-only, legible at column width.
4. *Can the CoG submission be cited?* It was not published (`DATA_AVAILABILITY.md` §6 item 3). **Action:** no self-citation.

---

## 6. Suggested text edits

| # | File § | Proposed sentence |
|---|---|---|
| E1 | `02_METHODS` §0, end of first paragraph | "Readers should treat every interval in this paper as descriptive of match-sampling variation under a fixed, previously exposed design, not as a confirmatory test." |
| E2 | `02_METHODS` §4.1, after the 352-vs-362 sentence | "Selection between the two candidates was made on 15.15 Q_SELECT by match-weighted Brier (logistic 0.2339 vs LightGBM 0.2383; `Q_NEWV_FIT85_PRIMARY_20260920.md` L27–28), not on TEST; no learner ranking is claimed." |
| E3 | `02_METHODS` §1, resolve `\pending{pick-exclusion-wording}` | "Picks (\(n_{\min}\le 1\)) are excluded a priori: the predeclared scale-split contract covers the two multi-champion classes only, no fit85-lineage result exists for picks, and none is inferred." (author to confirm) |
| E4 | `02_METHODS` §5, after Table 5b | "Source matches were collected from the Korean server's Master-and-above ladders; tier is a collection property and is not stored per match (`DATA_AVAILABILITY.md` §5). The 16.13 cohorts' tier constraint is to be stated by the authors." |
| E5 | `03_RESULTS` §5 heading | "External score-only evaluation (dual stage; point estimates, no intervals)" |
| E6 | `03_RESULTS` §5, resolve `\pending{ext-TS-sentence}` | "On the S rows the ordering is the reverse of the T rows; both are point estimates without intervals, and we draw no conclusion from the difference between them." (author to confirm or drop) |
| E7 | `04_DISCUSSION` §6, two bullets | "**Fitting uncertainty:** \(\widehat V\) and \(q\) are single-seed fits; intervals cover match resampling only." / "**External cause:** whether the 16.13 loss reflects patch, region, \(q\) calibration or \(V\) measurement is not identified (`Q_NEWV_FIT85_TRANSFER_16X` L13)." |
| E8 | Conclusion (next pass), last sentence, per `WORKING_NOTES` §C | "Under the frozen evaluator, engagement definition and evaluated procedures, limited and environment-sensitive pre-information about the direction of estimated win-probability change exists in both cohorts; the ceiling of public telemetry has not been found." |
