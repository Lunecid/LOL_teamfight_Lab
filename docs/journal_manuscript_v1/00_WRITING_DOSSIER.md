# Writing dossier — everything needed to finish the journal paper (T013, 2026-09-21 night)

*Master index for the author. Prepared while the author was away, so every item here is a proposal or a record; nothing changes a decision the author has not made. Reading order for the morning: this page → `SESSION_CHANGE_REPORT_AND_QUIZ.md` → the three decisions in §2 → the briefs in §4.*

## 1. Where the paper stands (commit `42bb2c6` + this package)

| Part | File | State | What is left |
|---|---|---|---|
| Outline + claim–evidence map | `01_OUTLINE_AND_CLAIM_EVIDENCE.md` | complete (C1–C22, F1–F8) | RQ table status note pending author decision |
| Methods | `02_METHODS_CORE.md` §0–§9 | complete, reproducible level, two cohorts | `\pending{pick-exclusion-wording}`; seven trace rows to add (see `SELF_REVIEW` §1); shorten to ≈ 2 300 words for the page budget |
| Results | `03_RESULTS.md` §1–§6 | complete, two cohorts | `\pending{ext-TS-sentence}`; unify "does not hold / disappears" wording; add constant and p-only rows if kept |
| Discussion | `04_DISCUSSION.md` §1–§6 | complete | state the shared-input fact (q and V̂ read the same 351 numerics) in §3 |
| Abstract | `05_ABSTRACT_DRAFT.md` | **draft, two variants** (counts corrected 2026-09-21 by script; both under 200 whitespace tokens; external and pooling sentences limited to the settings evaluated) | pick A or B (B recommended; A depends on the EXT sentence decision) |
| Introduction | `06_INTRODUCTION_BRIEF.md` | **draft paragraphs P1–P7** | confirm; decide how the CoG version is cited under double-anonymous review |
| Related Work | `07_RELATED_WORK_BRIEF.md` | **draft, four topics, verified source pools** | decide whether the econometrics bib (17 entries) enters the journal bib |
| Conclusion / Limitations / Availability / Statements | `08_CONCLUSION_AVAILABILITY_BRIEF.md` | **draft** | availability statement must describe the freeze artifacts, not the CoG-extension package (blindspot, see §3) |
| Figures | `figures/fig1_pipeline.svg`, `fig2_delta_brier_bins_TS.svg`, `fig3_cohort_roles.svg` + `FIGURE_TABLE_PLAN.md` | drawn from docs/ values | approve; EPS export and reliability diagram need outputs/ (T014) |
| Venue | `VENUE_CHECKLIST_ToG.md` | verified 2026-09-21 (10 pages incl. refs; abstract 150–200 words; 2–5 keywords; double-anonymous) | page budget enforcement at LaTeX stage |
| LaTeX | none yet | — | T014 (port to IEEEtran skeleton from `docs/tog_manuscript/main.tex`) |

## 2. Decisions only the author can make (ordered by how much they block)

1. **RQ structure** (`.ai/RQ_DECISION_BRIEF.md`; options A/B/C). Blocks the Introduction's question paragraph and the outline table. Whichever option: J-RQ3's scope list must include engagement scale (T, S), and J-RQ2's "learner choice and information set" clause is not supported by the frozen design (logit 352 vs LightGBM 362 inputs). Recommendation on file: option C. The author's collaborator (2026-09-21) agreed with a C-based layout and proposed these roles, still a recommendation and not a decision: RQ1 = what the defined win-probability change and its direction reflect, and how stable they are under changes of definition and evaluator; RQ2 = whether, within each cohort, pre-fight state adds predictive gain beyond the flexible baseline in initial win probability and time; RQ3 = how far that gain is observed in balanced states, under small-change exclusion, across engagement scales (application and transfer), and on other patches and regions.
2. **Two `\pending` items.** (a) Pick-exclusion sentence (Methods §1). (b) Whether the EXT side-by-side sentence stays (Results §5); this also selects abstract variant A or B.
3. **Availability and anonymity package.** The current availability text inventories the CoG-extension (`market_event`) release; the freeze paper needs its own list (fit85 bundle sha16 `ac459cc4397630a9`, OOF fold evaluators, `logit_state` weights, RR and SS prediction tables, label npz). Also: whether the accepted 4-page CoG paper can be cited in third person without de-anonymising. (`REVIEWER_DEFENSE.md` §5; `08_…_BRIEF.md`.)
4. **Table/figure allocation** for the 10-page limit (`FIGURE_TABLE_PLAN.md`; `REVIEWER_DEFENSE.md` §3).
5. **AI-assistance disclosure** wording and placement (`08_…_BRIEF.md`).

## 3. What the reviews found (three independent passes on the current drafts)

| Pass | File | Verdict | Items that change text now |
|---|---|---|---|
| Simulated 5-reviewer review (academic-paper-reviewer skill) | `SIMULATED_REVIEW_20260921.md` | weighted 62.7 → "Major Revision" band (work-priority signal, not a forecast) | write the four missing sections; add a predeclared-vs-post-exposure timeline in Methods §0; unify transfer wording to "observed ordering, no interval"; state the shared-input fact; make "pre-state" precise (τ is retrospectively anchored); add constant/p-only rows and UNC per CORP cell; apply τ = 0.001 wording once; say PT_flex sits at the most-regularised grid corner; transcribe match-selection details |
| Reviewer defense + blindspot pass | `REVIEWER_DEFENSE.md` | 14 weaknesses, 17 ranked questions with evidence-cited draft answers, main/appendix allocation, negative-results sentences, 8 text edits | landmines: double-anonymous stripping of CoG identifiers; availability statement describes the wrong package; raw data no longer held; 2026-09-15 code on another branch; Riot terms and the collector's per-player state; patch-label mapping 15.x ↔ 25.x; "predeclared" ≠ "preregistered" |
| Self-review + claim audit + evidence grading | `SELF_REVIEW_20260921.md` | not ready (spine only); keep 14 / weaken 5 / revise 3 claim rows | weaken C11 (T B40 upper bound −0.00003 is within seed noise of zero), C19–C21 wording; C20 "pooling helps neither" is calibrator-dependent (sigmoid variant q_S − q_TS interval touches 0); Methods §6 interval inventory must add the SS intervals; seven untraced numbers (see its §1); print `bootstrap_fraction_positive` as "no draw exceeded zero", never "p = 0"; no multiplicity adjustment across fourteen intervals — say so |

## 4. Writing order (proposal) and word budget for 10 pages

| Step | Section | Source brief | Budget |
|---|---|---|---|
| 1 | Decide §2.1–§2.2 | — | — |
| 2 | Methods trim + seven trace rows + timeline table | `02_METHODS_CORE.md`, `SELF_REVIEW` §1, `SIMULATED_REVIEW` item 2 | 2 300 w |
| 3 | Results wording unification + captions | `03_RESULTS.md`, `FIGURE_TABLE_PLAN.md` | 2 000 w + 4 tables + 2 figures |
| 4 | Introduction | `06_INTRODUCTION_BRIEF.md` P1–P7 | 700 w |
| 5 | Related Work | `07_RELATED_WORK_BRIEF.md` | 600 w |
| 6 | Discussion additions | `04_DISCUSSION.md` + review items | 800 w |
| 7 | Conclusion, Limitations, Availability, statements | `08_CONCLUSION_AVAILABILITY_BRIEF.md` | 400 w |
| 8 | Abstract (written last) | `05_ABSTRACT_DRAFT.md` | 150–200 w |
| 9 | LaTeX port, figures to EPS, page check | `VENUE_CHECKLIST_ToG.md`, T014 | — |
| 10 | Final adversarial pass with `REVIEWER_DEFENSE.md` questions and the Forbidden list | — | — |

## 5. Numbers ledger (single source for the writing pass)

`EXPERIMENT_LEDGER.md` lists every experiment line with status, role in the paper, headline numbers and the exact source field. Rules: every number in prose resolves to a field in that ledger or in Methods §9; nothing is recomputed; the three lineages (fit85/SVI journal spine; 2026-09-15 old-V; market_event CoG-extension) are never tabulated together.

## 6. Guardrails that every sentence must pass

- `JOURNAL_FINISH_LOCK` Forbidden list (8 items incl. the 2026-09-21 additions) and `SCALE_SPLIT_EXPERIMENT_CONTRACT` §6.
- `ECONOMETRICS_LIT_APPLICATION_LOCK` "Forbidden slides from lit" (no DM/GW renaming; no "first LoL WP-change study"; no ΔV = returns).
- Terminology fixed: engagement; teamfight T / skirmish S; direction of change; pre-fight state; spline baseline in win probability and time (PT_flex).
- Every external number: "score-only, no interval". Every T-vs-S sentence: within-cohort or predeclared secondary only.

## 7. Deferred (thesis / v2) and cleanup

- `THESIS_V2_FOLLOWUPS.md`: six preregistration-ready plans (external adapter; participation-count cells; shared PT_flex object; predeclared T/S external contrast; definition-constant sensitivity; information-set modules).
- Documentation cleanup (no numbers change): supersession banner on `EXPERIMENT_INVENTORY_COHORT_20260919.md` §G; copy or cite the X-31 claim ledger; `WORKING_NOTES.md` §D synced with the RQ brief (done in T013); CORP precision stated once.

## 8. Task records

`.ai/tasks/T013.md` (this package), `.ai/tasks/T014.md` (Cursor: figures from outputs/, LaTeX port, page check), `.ai/reports/T013.md`.
