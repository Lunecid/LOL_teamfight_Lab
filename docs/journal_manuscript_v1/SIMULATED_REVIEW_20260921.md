# Simulated multi-perspective peer review — journal_manuscript_v1

| Item | Value |
|---|---|
| Target venue (simulated) | IEEE Transactions on Games (journal; full review mode: EIC + 3 peer reviewers + Devil's Advocate) |
| Date | 2026-09-21 |
| Reviewed | `README.md`, `01_OUTLINE_AND_CLAIM_EVIDENCE.md`, `02_METHODS_CORE.md`, `03_RESULTS.md`, `04_DISCUSSION.md`; `WORKING_NOTES.md` (context only). Author-side constraints: `docs/JOURNAL_FINISH_LOCK_20260920.md` (thesis, Forbidden list), `docs/SCALE_SPLIT_EXPERIMENT_CONTRACT_20260920.md` §6, `.ai/CONTEXT.md` (numeric ledger). |
| Not yet written | Abstract, Introduction, Related Work, Conclusion / Limitations / Availability, figures, LaTeX port. Concerns about these are tagged **[missing section]** and are not defects of the existing text. |
| Ground rules | Numbers are quoted only as they appear in the drafts; none are computed here. Analyses absent from `docs/` are labeled **would require new work (thesis/v2)**. Forbidden claims (scale gradient, T/S mechanism, best architecture, "recalibration fixes transfer", AUC ceiling) are not requested. The two `\pending{...}` markers (`02_METHODS_CORE.md` §1; `03_RESULTS.md` §5) are known author-wording items, not errors. |
| Status of this document | Simulated review. Scores and the Accept/Major label are **not** submission forecasts (README rule). |

---

## Editor summary

**Fit.** The manuscript belongs in ToG: it is an esports-analytics / applied-forecasting study on public League of Legends telemetry, with no human-subject component. The object is unusual and defensible — the *direction* of a frozen win-probability evaluator's change over an engagement interval, predicted from pre-fight public state beyond flexible functions of initial win probability and time — and the authors report a verification stack rather than a leaderboard.

**What the existing text does well.** Epistemic status is stated up front (`02_METHODS_CORE.md` §0). Every number carries an artifact pointer (§9). Absence of uncertainty is stated where it applies (external cohorts, CORP, V→W). The Forbidden list (`04_DISCUSSION.md` §5) pre-empts the common overreadings, and the T/S design stays within-cohort.

**Why it cannot be assessed as a submission yet.** Abstract, Introduction, Related Work and Conclusion do not exist, so originality and positioning are unassessable and the reader meets the study through a claim ledger. Independently, the panel converges on three issues in the existing text: (i) the headline lift is small (all-T ΔBrier −0.00373 on a Brier of 0.2355; UNC≈0.25) and does not hold on the only cohorts without prior exposure (T rows of KR/NA1 16.13: +0.0026 / +0.0040, no intervals); (ii) the external "does not hold" reading rests on point-estimate orderings only, while `04_DISCUSSION.md` §1 says the advantage "disappears"; (iii) the meaning of "prediction from pre-state" needs a plain statement that the prediction time τ = first kill − 15 s is anchored retrospectively.

**Devil's Advocate CRITICAL items** are adjudicated in the consolidated list below (both are recorded as *unresolved in the current draft, fixable by revision*; neither is a foundation collapse).

**Simulated decision:** Major Revision (see scores).

---

## Reviewer 1 — Methods and statistics

1. **[major] Exploratory status needs a concrete timeline, not a label.** `02_METHODS_CORE.md` §0; `03_RESULTS.md` header. "Prior TEST exposure" is declared, but the reader cannot see *which* 15.16 quantities were seen before the flexed baselines, CORP and RRX were designed. *Fix:* one dated table (from the RR0 manifest / RR1 addendum in `docs/`) listing, per analysis, predeclared vs post-exposure and what had been inspected. No new experiments.

2. **[major] External result is an untested ordering; wording must match everywhere.** `03_RESULTS.md` §5 states this correctly ("observed ordering of point estimates, not a tested difference"), but `03_RESULTS.md` §6 ("Not on main KR/NA1 16.13") and `04_DISCUSSION.md` §1 ("disappears") read as tested findings. *Fix:* use one formula ("observed ordering without interval") in §6, Discussion §1 and, later, the Abstract. Match-cluster intervals for the EXT ΔBrier cells would require new work (thesis/v2); the authors should say so in Limitations.

3. **[major] "Beyond p,t" mixes information set and learner.** `02_METHODS_CORE.md` §4.1–4.2. q is L2 logistic on 352 inputs; PT_flex is spline-logistic on two, so the contrast confounds *more information* with *different functional form*. *Fix (wording):* state whether the 351 features contain time or p-like summaries, and which way the form handicap runs (q gets p_pre linearly, without the spline basis). A nested model (q inputs + PT_flex basis) would require new work (thesis/v2).

4. **[major] PT_flex winner sits at the grid corner.** `02_METHODS_CORE.md` §4.2: grid `n_knots_p ∈ {4, 6} × C ∈ {0.01, 0.1, 1}`, winner (4, 4, 0.01) — the most regularized corner on both axes. This means the baseline's optimum may lie outside the grid. *Fix:* say this explicitly, and note which way it biases the q − PT_flex contrast. Extending the grid would require new work (thesis/v2).

5. **[major] Reference rows are missing from the primary table.** `03_RESULTS.md` §3.1. Methods §4.2 says constant and p-only spline baselines exist in the RR12 table; `02_METHODS_CORE.md` §9 lists the T positive rate 0.498 from `TEST_all.constant.p_pos`. *Fix:* add the constant and p-only rows (already computed, transcription only) so the reader can form a skill score against UNC≈0.25 and see how much of the gap is p alone.

6. **[minor] Interval coverage statement belongs in Results, not only Methods.** `02_METHODS_CORE.md` §6 says intervals reflect match-sampling variation only, "not model-fitting or label-model uncertainty", and the evaluator candidates are "single-seed point estimates" (§2). *Fix:* repeat this in one sentence at the top of `03_RESULTS.md` §3.

7. **[minor] Sample flow.** `02_METHODS_CORE.md` §5: intermediate counts (detected → T → valid → finite scores) "are not recorded". *Fix:* transcribe them if the manifests in `docs/` hold them; otherwise move the honest sentence to Limitations (reconstruction would require new work, thesis/v2).

8. **[minor] Weighted AUC.** `02_METHODS_CORE.md` §6 calls AUC a "weighted average under w". *Fix:* one sentence on how match weights enter the pair counting.

9. **[minor] Dangling learner mention.** `02_METHODS_CORE.md` §4.1 names the LightGBM candidate (362 inputs) without a Q_SELECT value. *Fix:* print it from the RR12 JSON if present, or move the sentence to a footnote; never as a ranking (Forbidden F4).

10. **[missing section] Availability.** Internal paths and sha16 digests are provenance, not a public archive. *Fix:* plan an Availability statement (code, sealed feature order, engagement keys, predictions) within Riot API terms.

---

## Reviewer 2 — Games / esports domain

1. **[missing section] Related Work is absent, so positioning is unassessable.** `01_OUTLINE_AND_CLAIM_EVIDENCE.md` §Section outline item 2 names the intended anchors (Maymin WP-change; encounter detection; proper scores / direction vs mean as object separation). The domain reader will also expect the in-game win-prediction literature for MOBAs and the authors' own CoG lineage. No citations are proposed here; the authors should fill the outline they already have.

2. **[major] Engagement-definition robustness is asserted from other corpora.** `02_METHODS_CORE.md` §1: sensitivity of secondary constants "was measured on conference-era or pilot corpora ... not re-run under the current corpus"; the presence gate uses M = 2 and "at M = 3 the corpus shrinks to 76 290 engagements". For a journal, the teamfight/skirmish units are the study's foundation. *Fix:* state in Limitations, in one paragraph, which constants were re-estimated on the present corpus (G, D per patch are — §1 "Per-patch re-estimation") and which were not. A re-run of the G×D or M sweep on the current corpus would require new work (thesis/v2).

3. **[major] Match selection is under-described.** `02_METHODS_CORE.md` §5: "source matches (210 000 KR, patches 15.14–15.16)". Queue type, rank tier, game mode and collection dates are not stated; the external cohorts add region NA1 and patches 16.13–16.15 without a calendar gap. *Fix:* transcribe these from the cohort contract if recorded; if not recorded, say so. Domain readers will read the 15.16 → 16.13 gap as a meta shift and need the dates to judge it.

4. **[minor] Scale-cut evidence should be visible in the paper.** `04_DISCUSSION.md` §5 justifies Forbidden F6 by "the pick-minus-teamfight AUC difference changes sign with the class cut (3 / 4 / 5)" with the evidence in read-only `sec_definition.tex`. *Fix:* one sentence in Methods §1 that names the lineage result, so the refusal to rank scales is grounded in the paper itself. No new numbers.

5. **[minor] Pick exclusion.** `02_METHODS_CORE.md` §1 `\pending{pick-exclusion-wording}`; `04_DISCUSSION.md` §6 ("about 18% of 15.16 engagements"). Known author item; state it as scope, not as a data limitation.

6. **[minor] Operational reading is missing.** `03_RESULTS.md` §3.1 says q is not "a strong operational tip engine". A coach or broadcast analyst still needs one paragraph on what AUC 0.6403 (T) / 0.5767 (S) means in practice, and what the next-objective correspondence (Blue-credited share 0.604 after SVI+ vs 0.396 after SVI−, `03_RESULTS.md` §2.3) does and does not say. Wording only; keep it under Forbidden F5.

7. **[minor] Side convention.** `02_METHODS_CORE.md` §2 fixes blue = 1; SVI is blue-perspective. *Fix:* one sentence on side symmetry of features and target.

---

## Reviewer 3 — ML forecasting and calibration

1. **[major] Prediction time is retrospectively anchored.** `02_METHODS_CORE.md` §1b: τ = t_first kill − B, so the forecaster is scored only at instants known, after the fact, to be 15 s before a kill. Legitimate, but a *conditional, post-hoc anchored* forecast, not a streaming one; onset detection is not part of the task. *Fix:* say so in Methods §1b and Limitations. Wording only.

2. **[major] Calibration is measured but never shown.** `02_METHODS_CORE.md` §6 lists ECE among weighted metrics; `03_RESULTS.md` shows only CORP MCB/DSC points. A forecasting reader expects reliability diagrams (or binned tables) for q vs PT_flex on 15.16 and on KR/NA1 16.13. *Fix:* if reliability tables exist in the RR6a / RRX artifacts under `docs/`, include them; otherwise this would require new work (thesis/v2).

3. **[major] Describe the shift, without promising a fix.** `03_RESULTS.md` §5: on KR/NA1 16.13 T rows q MCB is 0.0065 / 0.0100 while DSC stays 0.0082 / 0.0064. The reader will want to know whether the miscalibration is a base-rate (intercept) shift or a slope problem. *Fix:* report, per external cell, the match-weighted positive rate and the mean of q — descriptive only, consistent with Forbidden F3. If these are not in the RRX documents, they would require new work (thesis/v2).

4. **[minor] Print UNC per cell.** `02_METHODS_CORE.md` §6 gives BS = MCB − DSC + UNC, but UNC is only stated as "≈0.25" for T. *Fix:* print UNC for each CORP cell (T, S, EXT) so BS_iso = UNC − DSC can be read; note it is an in-sample diagnostic.

5. **[minor] Calibrator-selection margins.** `02_METHODS_CORE.md` §4.2: identity beat the sigmoid on Q_SELECT for T; for S the two-stage rule chose the sigmoid for every model and the primary moves from −0.00335 to −0.00346. *Fix:* report the Q_SELECT margins between identity and sigmoid (from `calibrator_choice` if present). This helps the reader see how thin the choice was.

6. **[minor] Explain the B40 baseline AUCs.** `03_RESULTS.md` §3.2: PT_flex AUC 0.5248 (T) and 0.4999 (S) inside B40. This is expected — p_pre is nearly constant in the band, so the baseline has almost only t — but should be said, or it looks like a bug.

7. **[minor] Uniform columns.** Log loss appears in §3.1 but not in §3.2 or §5. *Fix:* same columns in every block, or say why not.

8. **[minor] Interpretation threshold.** The scale-split contract uses τ = 0.001 as an interpretation threshold; the manuscript never states one, yet calls −0.00214 "small" and −0.00373 "limited". *Fix:* state the threshold once in Methods §6 and use it consistently.

---

## Devil's Advocate

### Strongest counter-argument

Strip the verification apparatus and the paper says this: a 352-feature logistic regression, trained and tuned inside a pipeline that had already seen the test patch, lowers Brier by 0.00373 against a two-feature spline on 15.16 (Brier 0.2355 vs 0.2392; positive rate 0.498; UNC≈0.25), and on the only cohorts it had never seen — T rows of KR/NA1 16.13 — its Brier is worse (+0.0026 / +0.0040). A skeptic will call this a carefully documented near-null result. Worse, the target Y_SVI is the sign of a change in a *model's* output, and q is fed the same 351 numeric features that the evaluator V̂ consumes (`02_METHODS_CORE.md` §2, §4.1). The most parsimonious reading of the 15.16 lift is therefore that q learned how V̂ responds to feature changes, not how the game unfolds. The correspondence checks (kill-axis agreement 0.904; next-objective shares 0.604 / 0.396) validate SVI, not q — nothing in the draft ties q's predictions to an observable event. Finally, the S cohort entered the main text on 2026-09-21, after the T external failure was known; its external ordering is favorable (−0.0018 / −0.0020, no intervals). The authors forbid comparing T and S, but a reader will still see S as a cushion. The thesis sentence — "limited, environment-sensitive pre-information exists" — is hard to falsify as written.

### Issue list

**CRITICAL (fixable by revision; not a foundation collapse)**

- **DA-1 [major, flagged CRITICAL by criterion 4: stronger counter-narrative].** q may predict the evaluator rather than the game. *Anchor:* `02_METHODS_CORE.md` §2 (Expanded361 inputs to V̂) and §4.1 (q inputs = the same 351 numeric features + p_pre). *Fix:* (a) say the shared-input fact plainly in Discussion §3; (b) any check that q's output tracks an observable outcome (kill differential, next-objective credit) would require new work (thesis/v2) and must not be framed as a mechanism (Forbidden F7). Confidence 4/5.
- **DA-2 [major, flagged CRITICAL by criterion 2: logic chain].** The transfer conclusion has no uncertainty behind it while the in-patch conclusion does; the paper's central asymmetry ("lift exists / lift does not transfer") is half-tested. *Anchor:* `03_RESULTS.md` §5 ("No bootstrap intervals were computed for the external cohorts"). *Fix:* as Reviewer 1 item 2; state in the Abstract-to-be that transfer is an observed ordering. Confidence 4/5.

**MAJOR**

- **DA-3.** Timeline disclosure for the S cohort. *Anchor:* `02_METHODS_CORE.md` §0 (S added 2026-09-21 under a contract dated 2026-09-20); `03_RESULTS.md` §5 S block. *Fix:* state the order of events (T external result → S contract → S results → promotion decision) in Methods §0. Keep T as the headline; do not let S's external ordering soften the T transfer statement.
- **DA-4.** Falsifiability. *Anchor:* `WORKING_NOTES.md` §C (thesis last word), `04_DISCUSSION.md` §1. *Fix:* state what result would have led the authors to say "no usable pre-information" (e.g., the interpretation threshold of Reviewer 3 item 8), so the conclusion is a finding, not a framing.

**MINOR**

- **DA-5.** `03_RESULTS.md` §6 says "beats PT_flex on Brier" for both cohorts, while the claim map calls the S external block "supported as observed ordering" (C22). The two registers should match.

### Alternative explanations not yet addressed

- Early-band evaluator noise: V̂ is weak in [2,10) (Brier 0.2283, AUC 0.6642, `03_RESULTS.md` §1); labels there may be partly evaluator noise. The λ·s_Q filter (§4) addresses magnitude, not time. A per-band ΔBrier would require new work (thesis/v2); an acknowledging sentence does not.
- Endpoint mechanics: ≈78.9% of intervals stop at the next kill and 14.286% reach the cap (`02_METHODS_CORE.md` §1b), so the "post" state is often the instant before the next kill. State this as a label property in Discussion §3.

### Missing stakeholder perspectives

Players and teams whose public data form the corpus (state the data-use terms); analysts who might misuse a 0.64-AUC signal as a live tip (already disclaimed).

---

## Consolidated revision list (prioritized)

**Editor's adjudication of DA CRITICAL items:** DA-1 and DA-2 are recorded as *unresolved in the current draft*; both are addressable by wording and transcription (items 1 and 3 below), not by new experiments. They block a simulated Accept but not a Major Revision.

1. **Write the missing sections** (Abstract, Introduction, Related Work, Conclusion) so that every headline sentence is checkable against `03_RESULTS.md` §§3–5 and does not exceed `04_DISCUSSION.md` §5. [missing section]
2. **Add a predeclared-vs-post-exposure timeline table** in Methods §0, including the S-cohort order of events. (R1-1, DA-3)
3. **Unify the transfer wording** to "observed ordering without interval" in Results §6, Discussion §1 and the Abstract. (R1-2, DA-2)
4. **State the shared-input fact** (q and V̂ read the same 351 numeric features) and its consequence for interpretation in Discussion §3, without a mechanism claim. (DA-1)
5. **Make "pre-state" precise**: retrospectively anchored τ; onset detection not part of the task. (R3-1)
6. **Add the constant and p-only rows** to the primary tables and print UNC per CORP cell. (R1-5, R3-4)
7. **State the interpretation threshold** (contract τ = 0.001) once and apply it to every "small / limited" sentence. (R3-8, DA-4)
8. **Describe the PT_flex grid corner** and the direction of its bias; note the information-vs-form confound of the "beyond p,t" contrast. (R1-3, R1-4)
9. **Transcribe match-selection details** (queue, tier, dates, calendar gap 15.16 → 16.13) and the engagement-definition sensitivity scope into Methods/Limitations. (R2-2, R2-3)
10. **Calibration evidence**: reliability tables and external positive rate / mean q if they exist in `docs/`; otherwise state in Limitations that they would require new work (thesis/v2). (R3-2, R3-3)

Minor items (11+): weighted AUC, uniform columns, B40 baseline-AUC sentence, LightGBM footnote, side symmetry, operational reading, scale-cut sentence, Availability statement.

---

## Scores (skill rubric, 0–100; ordinal, not cardinal)

| Dimension | Weight | Score | Basis |
|---|---|---|---|
| Originality | 0.20 | 64 | Adequate: new measurement object and verification framing; incremental over the CoG lineage; cannot rise without Related Work. |
| Methodological rigor | 0.25 | 68 | Adequate-to-strong: clean roles, out-of-fold labels, match-cluster bootstrap, execution addendum; held down by prior TEST exposure, no EXT/CORP intervals, sensitivity not re-run. |
| Evidence sufficiency | 0.25 | 56 | Well-traced empirical evidence; external half untested; no literature base yet. |
| Argument coherence | 0.15 | 66 | The chain V→W, ΔV/SVI, q→SVI, transfer is clear; problem → gap → RQ is missing; RQ wording (three locked questions vs two-plus-transfer outline) is an open author decision, not scored as an error. |
| Writing quality | 0.15 | 60 | Precise but ledger prose (RR IDs, digests, paths inline); needs journal prose and figures. |
| **Weighted** | | **62.7** | 12.8 + 17.0 + 14.0 + 9.9 + 9.0 → Major Revision band (50–64). |

Optional dimensions: *Literature integration* — not scored (section not written). *Significance and impact* — 62 (relevant problem; narrow practical reach by the authors' own account; methodological value for esports forecasting if the verification stack is presented as reusable).

---

**Recommendation:** Major Revision (simulated). Methods/Results/Discussion are honest and traceable; the revision is writing, disclosure and transcription from existing artifacts plus the four missing sections. No new experiments are needed; items marked "would require new work (thesis/v2)" belong in Limitations.
