# Journal manuscript v1 (freeze spine)

**Date:** 2026-09-20  
**Status:** First writing package under [JOURNAL_FINISH_LOCK_20260920.md](../JOURNAL_FINISH_LOCK_20260920.md)  
**Preserved prior draft:** `docs/tog_manuscript/` (CoG-extension / engagement-winner spine) — **do not overwrite**

## Contents

| File | Role |
|---|---|
| [00_MIGRATION_OLD_VS_FREEZE.md](00_MIGRATION_OLD_VS_FREEZE.md) | What stays, what is demoted, what is new |
| [01_OUTLINE_AND_CLAIM_EVIDENCE.md](01_OUTLINE_AND_CLAIM_EVIDENCE.md) | Section outline + claim↔artifact map |
| [02_METHODS_CORE.md](02_METHODS_CORE.md) | V / SVI / q / splits / evaluation |
| [03_RESULTS.md](03_RESULTS.md) | Reader order: \(V\) → ΔV/SVI → \(q\) → conditional/EXT |
| [04_DISCUSSION.md](04_DISCUSSION.md) | Meaning, dependence, transfer limits |
| [WORKING_NOTES.md](WORKING_NOTES.md) | Internal writing memo (journal-fit, deferred work, next-pass instructions, RQ decision brief) — not manuscript text |

**Not in this package yet (next writing pass):** Introduction, Abstract, Conclusion, Related Work rewrite, LaTeX port. Those wait until Methods/Results/Discussion claims are stable.

## Rules (from author instruction + freeze)

- No new RQ, model, label, or experiment.
- Do not compare `market_event` winner AUC and SVI \(q\) AUC as the same task.
- Exploratory / prior TEST exposure stated in Methods from draft 1.
- CORP = score decomposition on an evaluation sample, not a causal mechanism.
- TRIPOD+AI = selective reporting checklist for prediction-time availability, splits, discrimination, calibration, external evaluation, uncertainty — not a clinical primary standard for this game study.
- Simulated Accept/Major labels are **not** submission forecasts.

## Change log

- T001 — technical corrections (SVI naming, B40 definition, percentage-point units, OOF evaluator path, RR4 exclusion wording, RR5b denominators, uncertainty column) — see `.ai/reports/T001.md`
- T002 — Methods transcription (engagement constants, prediction/outcome times, role tables with counts, evaluator and q procedures, weighting and bootstrap, evidence-trace table; three `\pending` items) — see `.ai/reports/T002.md`
- T003 — journal-fit memo, deferred work and next-pass instructions moved from 04_DISCUSSION.md to WORKING_NOTES.md — see `.ai/reports/T003.md`
- T005 — `\pending` items resolved from the 2026-09-15 lineage (cohort rule, h90 endpoint rule, per-role match counts) with byte-identical copies under `docs/lineage_20260915/` — see `.ai/reports/T005.md`
- T006 — detector constants table (value + source class) in Methods §1 — see `.ai/reports/T006.md`
- T004 — lineage banners on PAPER_COHORT_CONTRACT §5–§6, 15.15 remainder note, uncertainty-coverage line on RESPONSE_EVIDENCE_MATRIX — see `.ai/reports/T004.md`
- T011 (2026-09-21) — scale-split promotion (author option C): dated additions-only amendments to PAPER_COHORT_CONTRACT §1, COMMON_RESEARCH_SPINE, JOURNAL_FINISH_LOCK, Q_PREDICTION_DESIGN_CONTRACT, SCALE_SPLIT contract §8, RESPONSE_EVIDENCE_MATRIX (row SS) — see `.ai/tasks/T011.md`
- T012 (2026-09-21) — skirmish cohort S integrated as a second block in Methods §0/§1/§4/§5/§6/§9, Results §3.1–§3.4/§5/§6, Discussion §1/§5/§6 and the claim–evidence map (C18–C22, F6–F8); all S numbers from `docs/SCALE_SPLIT_TvsS_RESULTS_20260920.json` (identity) — two `\pending` items for author wording — see `.ai/tasks/T012.md`
- T013 (2026-09-21 night) — writing package: `00_WRITING_DOSSIER.md` (index), `EXPERIMENT_LEDGER.md`, drafts 05–08 (abstract, introduction, related work, conclusion/availability), `REVIEWER_DEFENSE.md`, `SIMULATED_REVIEW_20260921.md`, `SELF_REVIEW_20260921.md`, `FIGURE_TABLE_PLAN.md` + `figures/` (3 SVG from docs/ values), `VENUE_CHECKLIST_ToG.md`, `THESIS_V2_FOLLOWUPS.md`, `SESSION_CHANGE_REPORT_AND_QUIZ.md`; no change to 01–04 — see `.ai/reports/T013.md`
