# Session change report and merge quiz (T001–T013, 2026-09-20 → 2026-09-21)

*Method: change-quiz skill. Read the report, then take the quiz before merging `claude/happy-heisenberg-h93ypi` into `feature/fight-boundary-pipeline`. Answers are at the bottom, folded.*

## Context

The session started from `21391b2` (manuscript spine) with a collaborator's 14-point technical review. Goal: make the journal drafts correct and reproducible, then answer the author's question "should 2–3-player fights be trained separately?" without breaking the journal freeze. The session ended with the skirmish cohort promoted into the main text and a full writing package.

## What changed, by intent

**1. Correctness of the drafts (T001–T006).** SVI expanded correctly; B40 defined as 0.40–0.60; ΔV̂ units; OOF label path; RR4 exclusion wording; RR5b denominators; CORP moved out of the "uncertainty" column; Methods transcribed to reproducible level (constants table, endpoint rule, cohort rule, per-role match counts, bootstrap settings, evidence trace); Discussion memos moved to `WORKING_NOTES.md`; lineage copies under `docs/lineage_20260915/` (byte-identical to `e0ec3d0`).

**2. Operating model (AGENTS.md, CLAUDE.md, .ai/).** Work runs by task specs in `.ai/tasks/`, reports in `.ai/reports/`, reviews in `.ai/reports/*_review.md`; Cursor executes code tasks (data lives on the author's machine), Claude designs and reviews.

**3. Scale-split experiment (contract + T007–T010).** A predeclared contract (2026-09-20) defined S = `cohort == 0 & fine == 1`, one primary contrast, forbidden readings. T007 parametrised the pipeline (default T byte-identical); T008 built S labels with the same fold evaluators (held sets from V-bucket rows); T009 fitted and scored; T010 re-scored with the contract-literal identity calibrator after review found the RR12 two-stage rule had chosen sigmoid for S.

**4. Promotion and locks (T011).** Author chose option C (main text, two cohort blocks). Dated, additions-only amendments in PAPER_COHORT_CONTRACT §1, COMMON_RESEARCH_SPINE, JOURNAL_FINISH_LOCK (answers table, Forbidden +3, "what not to reopen" +1), Q_PREDICTION_DESIGN_CONTRACT, SCALE_SPLIT contract §8, RESPONSE_EVIDENCE_MATRIX (row SS). Five hand-edited docs with lossy encoding were restored from `aa2249f`.

**5. Manuscript integration (T012).** Results §3.1 two blocks; §3.4 transfer/pooling; per-cohort B40, CORP, EXT; Methods S definition, arms, calibrator policy, roles, evidence rows; Discussion additions; claim rows C18–C22, F6–F8; two `\pending` author items.

**6. Writing package (T013, this night).** Ledger, abstract/intro/related-work/conclusion briefs, reviewer defense, simulated review, self-review, figure plan + 3 SVGs, venue checklist, thesis follow-ups, this report.

## How it interacts with existing paths

- **Default-T runs are unchanged.** `--cohort`, `--cohort-tag`, `--calibrator select`, no `--out-suffix` reproduce the frozen T artifacts; RR0 digests are re-checked by the scale-split manifest generator (`T_frozen_rr0_digests_match`).
- **Guards now exist.** rr12 refuses a non-default `--q-model` without `--out-suffix`; rrx refuses `--cohort-tag S` without a suffix; both would otherwise overwrite frozen docs.
- **The manifest generator is the integrity oracle.** `ss20260920_scale_split_manifest.py --check-integrity --with-t009` recomputes the T008 integrity block and exits 2 on any drift.
- **Locks were extended, never edited.** Every amendment is an added line with a date; the original sentences still stand and the new lines say they take precedence for S only.
- **Forbidden readings grew.** Scale gradient, T/S mechanism, and pooled/pick/small-EXT substitution are now forbidden in JOURNAL_FINISH_LOCK and in Discussion §5.

## Intuition (three mental-model updates)

1. **"Identity calibrator" is a contract term, not a default.** The RR12 script's two-stage rule is a *selection*; for T it happened to choose identity, for S it chose sigmoid. The paper reports the contract-literal identity and keeps the selection as sensitivity.
2. **The fold held-set is the V-bucket partition, not the engagement-role set.** Engagement rows are a strict subset of the matches held out from each fold evaluator; the leave-match guarantee is checked against the larger set.
3. **Two cohorts, two primaries, zero comparisons.** Every cross-cohort statement is either a predeclared secondary contrast on identical rows (transfer, pooling) or forbidden.

## Quiz (7 questions; answer before merging)

1. Recall: which rows of the paper are labelled by out-of-fold evaluators rather than the frozen bundle, and why does that matter for the S cohort?
2. Prediction: a colleague runs `rr20260920_review_response_rr12.py --cohort-tag T --q-model outputs/q_newv_fit85_20260920_TS/models/logit_state.joblib` with no other flags. What happens?
3. Recall: the S primary contrast is −0.00335 [−0.00379, −0.00288]. What is the −0.00346 number that appears next to it, and which one may be called "primary"?
4. Prediction: someone deletes the "2026-09-21 개정" row from COMMON_RESEARCH_SPINE because "the old row is wrong now". What rule does that break, and what should be done instead?
5. Recall: why were `Q_NEWV_FIT85_PRIMARY_20260920_S.md` and the three `_S_q*.md` docs restored from `aa2249f`, and what must a Windows editor do to avoid a repeat?
6. Prediction: a reviewer writes "teamfights are more predictable than skirmishes since q's AUC is 0.640 vs 0.577". Where in the manuscript is the prepared answer, and what is the one-sentence reply?
7. Recall: the abstract draft says the teamfight lift "does not hold" externally. What kind of evidence is that, and which caption word must accompany every external number?

<details>
<summary>Answers</summary>

1. 15.14 TRAIN rows (both T and S) are labelled by V̂^(−k) fold evaluators (match-level folds); Q_CAL/Q_SELECT/TEST/EXT by the frozen fit85 bundle. S reuses the *same* fold evaluators, so its TRAIN matches must lie inside each fold's held-out set (checked: 14 628–15 017 matches per fold, S ⊂ V-held).
2. It exits: "non-default --q-model requires --out-suffix (would overwrite frozen outputs)". Without the guard it would have overwritten `outputs/review_response_rr12_20260920/` and `docs/REVIEW_RESPONSE_RR12_RESULTS_20260920.{md,json}`.
3. −0.00346 is the RR12 two-stage calibrator-selection variant (sigmoid on every S model); only the identity value −0.00335 is "primary" (contract §4 literal); the variant is a sensitivity row (Table 2b).
4. AGENTS.md: lock/contract numeric rows are never deleted or modified, only annotated. The correct move is what T011 did: add a dated row that states it takes precedence, leaving the original in place.
5. A hand edit on Windows saved them with a lossy (cp949) encoding, turning Δ, −, —, ← into "??"; they were restored from `aa2249f` and the intended edits (banner; one line removed) re-applied in UTF-8. Editors must save as UTF-8 (the generated `_id.md` files were fine because the script wrote them).
6. `REVIEWER_DEFENSE.md` (anticipated question) and Discussion §5 / Outline row F6. Reply: the cohorts differ in size, positive rate and balanced-state share, absolute scores are not on a common scale, and the definition lineage's cut-dependent sign reversal shows that such a ranking is an artefact of where the class boundary is placed; the paper reports within-cohort lifts only.
7. A point-estimate ordering without intervals (no bootstrap was run for external cohorts); every external number carries "no interval" / "score-only" in its caption and text.

</details>

Pass = you answered 6 of 7 without looking. Fail = re-read `.ai/reports/T009_review.md`, `.ai/reports/T010_review.md` and `03_RESULTS.md` §3.1–§3.4, then retake with fresh variants (ask the session).
