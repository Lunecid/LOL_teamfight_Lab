# Stale-claims inventory (W4, 2026-09-11, revised after the fact-check)

Purpose: the released repository must not contradict the ToG manuscript. This file lists every place in
`docs/` and `README.md` where a retracted or superseded claim appears. For each hit it gives the document
type and what was done about it.

## Method

- Searched with ripgrep over `docs/` (recursive, including `docs/references/` and `docs/tog_manuscript/`) and
  over `README.md`, at 16:27 on 2026-09-11, **after** this pass's edits.
- Entries give a section anchor first and a line number second. Other agents keep editing some of these files
  (`CLAUDE_TOG_PAPER_PLAN.md`, `README.md`, the manuscript files), so for those files **the anchor is
  authoritative** and the line number is as of 16:27.
- Patterns: `948,?369`; `one million|million engagements|~ ?1 ?M|1M|≈ ?1 ?M|100만`; `0\.7842`; `0\.0677`;
  `0\.0652`; `0\.493`; `89\.[13]`; `4\.9 ?%`; `decompos|규모에 따라 분해`; `21\.1|36\.9|42\.0`.
- Also searched: the rounded forms `.784`, `−.068`, `−.065`; the kill-less and sweep phrases `0.008`,
  `brief pass`, `0.15 per match`, `경기당 0.15`, `2.03`; and the scale-gradient wording `규모 무관`,
  `규모와 무관`, `규모 기울기는 없`, `기울기가 없`, `규모별 차이 없음`, `기울기 소멸`, `사라진다`,
  `does not change predictability`, `no scale gradient`.
- `4\.9 ?%` also matches 14.9 % and 54.9 %; those hits are listed as non-matches. The LaTeX form `4.9\%` in
  `definition_section_draft.tex` was checked separately.

**Document types.**

- *current-state*: read as a description of the repository or definition as it is now.
- *dated log*: a dated research or audit record.
- *plan*: a planning document.
- `DEFINITION_EVIDENCE.md` is a dated log, but three of its sections are read as current: §14 and §17
  (`corpus_shards_v33/manifest.json` cites §17 as "the definition"), and §22, which its status note names as
  current. Hits in those sections were treated as current-state.

**Actions.**

- EDITED: changed in this pass (only files owned by W4).
- BANNER: covered by a file-top banner or status note in a W4 file.
- MARKED: the surrounding text already presents the value as retracted, superseded or qualified.
- LEFT: dated log, left as it is.
- FLAG: not owned by W4; the file's owner needs to act, and a suggested fix is given.
- NON-MATCH: the pattern matched a different quantity.

Owned by W4: `docs/CONSTANTS_JUSTIFICATION.md`, `docs/DEFINITION_EVIDENCE.md`,
`docs/ENGAGEMENT_SCALE_DEFINITION.md`, `docs/RELEASE_REPORT_V3.md`, and this file.

---

## 1. `948,369` (v2 corpus size)

| file (anchor):line | type | context | action |
|---|---|---|---|
| README.md ("Retraction"):20 | current-state | cited inside the retraction section as the withdrawn v2 claim | MARKED |
| docs/CLAUDE_TOG_PAPER_PLAN.md (§0 table):20 | plan | "v0.2 claims vs current" row, replaced by 532,547 | MARKED |
| docs/CLAUDE_TOG_PAPER_PLAN.md (§4 never-write list):328 | plan | "948,369 · 약 100만 교전" | MARKED |
| docs/DATA_MANIFEST.md ("Canonical results"):19 | current-state | headline row: 948,369 eng, tf .7842, pick-tf -.0677 | **FLAG**: move to "Superseded (audit trail only)" and note the `time_norm` leak |
| docs/DEFINITION_EVIDENCE.md (status note):20 | dated log | withdrawal notice | EDITED |
| docs/DEFINITION_EVIDENCE.md (§4):248 | dated log | v2 class-table source line | BANNER |
| docs/DEFINITION_EVIDENCE.md (§17 shares note):885 | current-state | "*was:*" note under the v3.3 shares | EDITED |
| docs/DEFINITION_EVIDENCE.md (§19):1167 | dated log | v2 comparison row | BANNER |
| docs/DEFINITION_EVIDENCE.md (§22.1 table):1333 | current-state | v2 comparison row, retracted by the §22.1 bullets | MARKED |
| docs/ENGAGEMENT_DEFINITION_V3.md (§6):117 | current-state | struck-through v2 row | MARKED |
| docs/RELEASE_REPORT_V3.md (§5.1 table):156 | current-state | definition-change row | EDITED earlier: note at 158-162 says the rows are pre-leak-fix and the v2 values are retracted |
| docs/RELEASE_REPORT_V3.md (§5.1 note):161 | current-state | that note | EDITED (withdrawal notice) |
| docs/RELEASE_REPORT_V3.md (§5.3 table):191 | current-state | v2 row | EDITED: labelled "누수 특징; retracted" |

## 2. "one million" / "~1M" engagements

The earlier FLAGs on README.md:13 ("~1M") and README.md:148 ("approximately one million") are withdrawn: at
16:27 the README contains neither phrase.

| file (anchor):line | type | context | action |
|---|---|---|---|
| docs/AUDIT.md ("Paper-text edits for the 308 camera-ready"):91, :94 | dated log | CoG camera-ready wording decision | LEFT |
| docs/CLAUDE_TOG_PAPER_PLAN.md (§4 never-write list):328 | plan | "약 100만" | MARKED |
| docs/tog_manuscript/reviewer_response_matrix.md (R2-1 row):62 | current-state | evidence column says README.md:7-13 "still presents ... ~1M validated engagements" | **FLAG** (low): the README no longer says this; update the evidence column |

## 3. `0.7842` (v2 teamfight AUC), including `.784`

| file (anchor):line | type | context | action |
|---|---|---|---|
| README.md ("Retraction"):21 | current-state | withdrawn claim quoted in the retraction | MARKED |
| docs/CLAUDE_TOG_PAPER_PLAN.md (§0 table):19 | plan | retracted v0.2 claim | MARKED |
| docs/DATA_MANIFEST.md ("Canonical results"):19 | current-state | "tf .7842" | **FLAG** (same row as §1) |
| docs/DEFINITION_EVIDENCE.md (status note):20 | dated log | ".784" | EDITED (withdrawal notice) |
| docs/DEFINITION_EVIDENCE.md (§4):256 | dated log | v2 class table | BANNER |
| docs/DEFINITION_EVIDENCE.md (§19):1167 | dated log | "0.784 (42%)" | BANNER |
| docs/DEFINITION_EVIDENCE.md (§22.1 table):1333 | current-state | "0.784" | MARKED |
| docs/ENGAGEMENT_DEFINITION_V3.md (§6):117 | current-state | struck-through row | MARKED |
| docs/RELEASE_REPORT_V3.md (§5.3 table):191 | current-state | v2 row | EDITED (labelled retracted) |

## 4. `−0.0677` (v2 pick − teamfight gap), including `−.068`

| file (anchor):line | type | context | action |
|---|---|---|---|
| README.md ("Retraction"):21 | current-state | withdrawn claim | MARKED |
| docs/CLAUDE_TOG_PAPER_PLAN.md (§0 table):19 | plan | retracted v0.2 claim | MARKED |
| docs/CLAUDE_TOG_PAPER_PLAN.md (§2, "§5 Prediction"):156 | plan | "철회된 −0.068" | MARKED |
| docs/CLAUDE_TOG_PAPER_PLAN.md (§4 never-write list):321 | plan | "pick−tf −0.0677 → 누수" | MARKED |
| docs/DATA_MANIFEST.md ("Canonical results"):19 | current-state | "pick-tf -.0677" | **FLAG** (same row as §1) |
| docs/DEFINITION_EVIDENCE.md (status note):20 | dated log | "−.068" | EDITED (withdrawal notice) |
| docs/DEFINITION_EVIDENCE.md (§4):258 | dated log | "−.068 (−.070~−.065)" | BANNER |
| docs/DEFINITION_EVIDENCE.md (§19):1167 | dated log | "−0.068 [−0.070, −0.065]" | BANNER |
| docs/DEFINITION_EVIDENCE.md (§22.1 table):1333 | current-state | "−0.068" | MARKED |
| docs/ENGAGEMENT_DEFINITION_V3.md (§6):117 | current-state | struck-through row | MARKED |
| docs/TOG_EXTENSION_PLAN.md (file-top banner):6 | plan (superseded) | banner cites −0.068 as retracted | MARKED (see §14 for a FLAG on the same banner) |
| docs/RELEASE_REPORT_V3.md (§5.3 table):191 | current-state | v2 row | EDITED (labelled retracted) |
| docs/tog_manuscript/reviewer_response_matrix.md ("Evidence that must not be offered"):159 | current-state | never-offer list | MARKED |

## 5. `−0.0652` (cross-patch "replication" of the retracted gap)

| file (anchor):line | type | context | action |
|---|---|---|---|
| docs/CLAUDE_TOG_PAPER_PLAN.md (§4 never-write list):326 | plan | "−0.0652 교차 패치 재현" | MARKED |
| docs/DATA_MANIFEST.md ("Canonical results"):20 | current-state | "replication: 47,759 eng/10,612 matches, pick-tf -.0652" | **FLAG**: move to Superseded; it replicates a leak-affected gap |
| docs/DEFINITION_EVIDENCE.md (§4):258, (§19):1167 | dated log | "−.065" is the upper CI bound of the v2 gap | NON-MATCH |

## 6. `89.3 %` (Data Dragon coverage argument for R = 1,800 u)

| file (anchor):line | type | context | action |
|---|---|---|---|
| docs/CONSTANTS_JUSTIFICATION.md (SUPERSEDED banner, item 2):17 | current-state (superseded) | withdrawal notice | EDITED |
| docs/CONSTANTS_JUSTIFICATION.md (body):48, :55 | same | "89.3% cast within 1,800 u"; validity-radius row | BANNER |
| docs/DEFINITION_EVIDENCE.md (status note):21 | dated log | withdrawal notice | EDITED |
| docs/DEFINITION_EVIDENCE.md (§3.3):195 | dated log | coverage bullet | BANNER |
| docs/DEFINITION_EVIDENCE.md (§9):360, :381 | dated log | "R은 스킬 사거리 89.3%를 덮는 1,800 u"; "왜 1,800 u인가" | BANNER |
| docs/CLAUDE_TOG_PAPER_PLAN.md (§0 table):26 | plan | "강등" row | MARKED |
| docs/CLAUDE_TOG_PAPER_PLAN.md (§4 never-write list):327 | plan | withdrawn R basis | MARKED |
| docs/CLAUDE_TOG_PAPER_PLAN.md (§5 item 3):343 | plan | task list | MARKED |
| docs/tog_manuscript/reviewer_response_matrix.md ("Evidence that must not be offered"):157 | current-state | never-offer list | MARKED |

## 7. `89.1 %` (per-patch coverage used for R and in the drift rule)

| file (anchor):line | type | context | action |
|---|---|---|---|
| docs/CONSTANTS_JUSTIFICATION.md (SUPERSEDED banner, item 2):19 | current-state (superseded) | withdrawal notice | EDITED |
| docs/DEFINITION_EVIDENCE.md (status note):21 | dated log | withdrawal notice | EDITED |
| docs/DEFINITION_EVIDENCE.md (§10 estimator table, R and B/M rows):416-417 | current-state | rows now give R = 1,600 u and B = 15 s as rule anchors, with wiki source, check date and patch note, and "*was:*" notes | EDITED |
| docs/DEFINITION_EVIDENCE.md (§10 drift rule):424-426 | current-state | R-coverage criterion skipped for rule-anchored specs | EDITED |
| docs/DEFINITION_EVIDENCE.md (§10 "R에 대한 주의"):438, note 439-442 | dated log | superseded note | EDITED |
| docs/DEFINITION_EVIDENCE.md (§10 first-run table):460-463 | dated log (2026-09-08) | R-coverage column | LEFT (the note at 439-442 says so) |
| docs/DEFINITION_EVIDENCE.md (§14 verdict):646, note 647-649 | current-state | regenerated spec has no coverage criterion | EDITED |
| docs/DEFINITION_EVIDENCE.md (§14 current definition):703-715, "*was:*" at 709 | current-state | R = 1,600 u, B = 15 s with spec values | EDITED |
| docs/DEFINITION_EVIDENCE.md (§15):750 | dated log | "669 스펠 89.1%는 … 보조 근거로 강등" | MARKED |

## 8. Kill-less share ("4.9 % of engagements")

**Status: the v3.3 re-measurement has finished.** Source:
`D:/LOL_Project/fusion_2615/features/tog_revision/killless_grid/summary.json` (written 2026-09-11 11:25:37, code
at commit 5600f8b, recorded in commit dc243cf). It covers nine settings, 20,000 matches each, seed 7, run with
explicit `--radius`, `--min-per-team`, `--min-duration` and `--grace-ms` flags.

- **Teamfight gate**, row `r1600_t4_dG_g15`: R 1,600 u, at least 4 alive per side, at least 13.7 s, 15 s grace.
  591 of 19,155 proximity encounters have no kill: 591/19,155 = 3.09 %, and 591/20,000 = 0.02955 per match.
- **Sensitivity**: R 1,200 u gives 2.00 % (141/7,049, `r1200_t4_dG_g15`) and R 2,000 u gives 3.47 %
  (1,166/33,592, `r2000_t4_dG_g15`). A 10 s grace gives 4.34 % (832/19,155, `r1600_t4_dG_g10`). With 20 s
  instead of 13.7 s the share is 2.79 % (351/12,602, `r1600_t4_d20_g15`).
- **Denominator**: these are shares of proximity encounters, not of engagements (the file's
  `denominator_warning`). The same file's `corpus_reference`, 0.572 teamfight-class engagements per match
  (109,829/191,940), is a scale reference and must not be used as a divisor.
- **The superseded 4.9 %** comes from `D:/LOL_Project/fusion_2615/features/killless/r1800_d20_t3.json`: 305 of
  6,195 proximity encounters in 2,000 matches, with R 1,800 u, a 10 s grace, 3 per side and at least 20 s.

| file (anchor):line | type | context | action |
|---|---|---|---|
| docs/references/definition_section_draft.tex (kill-less sentence of the definition text):37 | current-state (manuscript draft) | "no kill amount to 4.9\% of engagements" | **FLAG**: replace with the teamfight-gate share of proximity encounters above, and state the denominator |
| docs/ENGAGEMENT_DEFINITION_V3.md (§1 table, row 1):27 | current-state | "킬 없는 encounter(4.9%)는 범위 밖 명시" | **FLAG**: replace with the v3.3 teamfight-gate share of proximity encounters, or drop the number |
| docs/DATA_MANIFEST.md ("Canonical results"):25 | current-state | "`killless/*.json` meta-review bound (4.9% at teamfight scale)" | **FLAG**: mark as v2 (R 1,800 u, 10 s grace) and add `tog_revision/killless_grid/summary.json` as the v3.3 grid |
| docs/DEFINITION_LITERATURE.md (§5, validity table):206 | current-state | "(4.9% 상한)" | **FLAG**: v3.3 value and its denominator |
| docs/DEFINITION_LITERATURE.md (§8, sentence list):268 | current-state | "우리 4.9%" | **FLAG**: same |
| docs/DEFINITION_EVIDENCE.md (status note):22-24 | dated log | now points to the v3.3 grid via the §17 note | EDITED |
| docs/DEFINITION_EVIDENCE.md (§5):288 | dated log | "4.9% (경기당 0.15개 vs … 2.03개)" | BANNER (the status note names it) |
| docs/DEFINITION_EVIDENCE.md (§9):372 | dated log | "맞선 킬 없는 encounter는 4.9%다" | BANNER |
| docs/DEFINITION_EVIDENCE.md (§14 current definition):712 | current-state | names the draft's kill-less sentence by content and points to the v3.3 grid; replaces a stale "line 36" reference | EDITED |
| docs/DEFINITION_EVIDENCE.md (§17 "주장하지 않는 것"):907-916 | current-state | v3.3 teamfight-gate share, sensitivity, denominator; 4.9 % kept as "*was:*" at 914 | EDITED |
| docs/ENGAGEMENT_SCALE_DEFINITION.md (STALE banner):16-18 | current-state (stale) | names the v2 table, the pairing and "brief passes" | EDITED |
| docs/ENGAGEMENT_SCALE_DEFINITION.md (kill-less table):192 | same | v2 row | BANNER + STALE note 194-246 |
| docs/ENGAGEMENT_SCALE_DEFINITION.md (STALE note, item 1):200-209 | same | 0.15 and 2.03 now sourced; 4.9 % = 305/6,195 | EDITED |
| docs/ENGAGEMENT_SCALE_DEFINITION.md (STALE note, v3.3 re-measurement):215-246 | same | grid table, scanner details, denominator | EDITED |
| docs/ENGAGEMENT_SCALE_DEFINITION.md (original paragraph):250-254 | same | struck through and tagged [STALE] | EDITED |
| docs/ENGAGEMENT_SCALE_DEFINITION.md (closing bullet):268-270 | same | "separate planned experiment" tagged [STALE] | EDITED |
| docs/RELEASE_REPORT_V3.md (§4 definition table, 킬 ≥ 1 row):136 | current-state | v3.3 teamfight-gate numbers; 4.9 % as "*was:*" | EDITED |
| docs/CLAUDE_TOG_PAPER_PLAN.md (§3.1 table and sensitivity):81-92 | plan | v3.3 grid, matches the artefact | MARKED |
| docs/CLAUDE_TOG_PAPER_PLAN.md (§3.1 "분모를 명시한다"):98-99 | plan | denominator warning; it cites `definition_section_draft.tex:36`, now line 37 | MARKED; **FLAG** (low): refer to the sentence by content |
| docs/tog_manuscript/reviewer_response_matrix.md ("Evidence that must not be offered"):155 | current-state | never-offer list | MARKED |
| docs/DEFINITION_EVIDENCE.md (§3.2 table):162 | — | base-region pair share 4.9% | NON-MATCH |
| docs/DEFINITION_EVIDENCE.md (§3.2):170 | — | 14.9% | NON-MATCH |
| docs/DEFINITION_EVIDENCE.md (§14):686, (§18):962 | — | 54.9% | NON-MATCH |
| docs/ENGAGEMENT_WINNER_DEFINITION.md:143 | — | 54.9% | NON-MATCH |

## 9. "0.15 per match vs 2.03 teamfight-class engagements per match"

The two numbers count different things:

- **0.15** = 305/2,000 = 0.1525 kill-less proximity encounters per match (`features/killless/r1800_d20_t3.json`).
- **2.03** = 416,956/205,884 = 2.025 teamfight-class engagements per match under the v2 rule `n_min ≥ 3`
  (`D:/LOL_Project/fusion_2615/features/scale_decomposition.json`, `by_participation_scale.teamfight.n` and
  `n_matches`; v2 detector, 993,484 engagements). The earlier version of this file said no artefact records
  2.03; that was wrong.

| file (anchor):line | type | context | action |
|---|---|---|---|
| docs/DEFINITION_EVIDENCE.md (status note):22 | dated log | names the pairing as history | EDITED |
| docs/DEFINITION_EVIDENCE.md (§5):288 | dated log | original pairing | BANNER |
| docs/ENGAGEMENT_SCALE_DEFINITION.md (STALE banner):16-17 | current-state (stale) | names the pairing | EDITED |
| docs/ENGAGEMENT_SCALE_DEFINITION.md (STALE note, item 1):200-209 | same | unlike-quantities objection kept; both numbers now sourced | EDITED |
| docs/ENGAGEMENT_SCALE_DEFINITION.md (original paragraph):251-252 | same | struck through | EDITED |

## 10. `0.493` ("predictability inversely related to importance")

| file (anchor):line | type | context | action |
|---|---|---|---|
| docs/CLAUDE_TOG_PAPER_PLAN.md (§4 never-write list):323 | plan | retracted | MARKED |

No other hit.

## 11. "decomposes by scale"

`decompos` also matches script and artefact names (`run_scale_decomposition.py`, `scale_decomposition_*.json`).
Those carry no claim and are not listed; rerun `rg -n decompos docs README.md` to see them.

| file (anchor):line | type | context | action |
|---|---|---|---|
| docs/ENGAGEMENT_SCALE_DEFINITION.md (STALE banner):7 | current-state (stale) | withdrawal notice | EDITED |
| docs/ENGAGEMENT_SCALE_DEFINITION.md (intro):23 | same | "The ToG extension's headline is a decomposition by engagement scale" | BANNER |
| docs/ENGAGEMENT_SCALE_DEFINITION.md (shop-event sensitivity):85 | same | "class gaps are ~0.05 AUC" | BANNER |
| docs/TOG_EXTENSION_PLAN.md (title, banner, "Thesis", "Equitable deep-learning comparison"):1, 4, 18, 23, 33, 86 | plan (superseded, banner) | title, thesis, "teamfight highest" | MARKED |
| docs/CLAUDE_TOG_PAPER_PLAN.md (§0 table):19; (§4 never-write list):321 | plan | "규모에 따라 분해된다" as retracted | MARKED |
| docs/tog_manuscript/reviewer_response_matrix.md ("How to read this file"):35 | current-state | states the retraction | MARKED |
| docs/ENGAGEMENT_WINNER_DEFINITION.md ("What the audit found"):52, 54 | current-state (v2-era label audit) | label policy "entangled with the scale decomposition" | **FLAG**, together with §15 item 1 |
| docs/ENGAGEMENT_WINNER_DEFINITION.md ("Adopted primary…", "Open items"):179, 190 | current-state | to-dos to re-score the v2 "full-corpus decomposition" | **FLAG** (low): mark as v2 to-dos |

## 12. `21.1 / 36.9 / 42.0` (v2 class shares, teamfight = `n_min ≥ 3`)

| file (anchor):line | type | context | action |
|---|---|---|---|
| docs/ENGAGEMENT_SCALE_DEFINITION.md (STALE banner):11 | current-state (stale) | withdrawal notice | EDITED |
| docs/ENGAGEMENT_SCALE_DEFINITION.md (shop-event sensitivity):77 | same | "21.1/37.0/42.0 to 22.0/36.7/41.3" | BANNER |
| docs/ENGAGEMENT_SCALE_DEFINITION.md (class-rule table):107-109 | same | 42.0 / 36.9 / 21.1 % | BANNER |
| docs/CLAUDE_TOG_PAPER_PLAN.md (§4 never-write list):329 | plan | retracted | MARKED |

## 13. Threshold-sweep "AUC spread 0.008"

Source: `D:/LOL_Project/fusion_2615/features/thresholds/*.json`: 553 matches, six one-factor settings on the v2
detector, 2,242-2,783 labelled rows per setting, AUC 0.593-0.601. The replacement, a G × D sensitivity against
the v3.3 headline, is `\pending{gd_sensitivity_v33}`.

| file (anchor):line | type | context | action |
|---|---|---|---|
| docs/CONSTANTS_JUSTIFICATION.md (SUPERSEDED banner, item 3):29-34 | current-state (superseded) | withdrawal notice with pending marker | EDITED |
| docs/CONSTANTS_JUSTIFICATION.md (body):67 | same | "AUC spread 0.008" | BANNER |
| docs/ENGAGEMENT_SCALE_DEFINITION.md (STALE banner):14 | current-state (stale) | withdrawal notice | EDITED |
| docs/ENGAGEMENT_SCALE_DEFINITION.md (recap):42 | same | "AUC moves 0.008" | BANNER |
| docs/DEFINITION_EVIDENCE.md (status note):24 | dated log | withdrawal notice | EDITED |
| docs/DEFINITION_EVIDENCE.md (§3.1):114 | dated log | "AUC 변동 0.008" | BANNER |
| docs/DATA_MANIFEST.md ("Canonical results"):24 | current-state | "R1 threshold sweep (AUC spread .008)" | **FLAG**: add "553 matches, v2 detector, one factor at a time; not against the v3.3 headline" |
| docs/TOG_EXTENSION_PLAN.md ("Equitable deep-learning comparison" table, R1-2 row):93 | plan (superseded) | "done" row | MARKED |
| docs/RELEASE_REPORT_V3.md (§5.3 table):183; docs/DEFINITION_EVIDENCE.md (§22.1 table):1325 | — | "−0.008" is a v3.3 pick − teamfight value | NON-MATCH |

## 14. "No scale gradient" / "규모 무관" without the cut

`docs/CLAUDE_TOG_PAPER_PLAN.md` §4 (line 322) forbids writing "규모 기울기가 없다" or "pick−tf = 0", because the
pick − teamfight gap depends on the teamfight cut.

- **Citable value**: at the adopted cut `n_min ≥ 4`, pick − teamfight is −0.002 [−0.007, +0.003]. Source:
  `features/scale_decomposition_v33_market_event.json`; class AUCs 0.67887 − 0.68089 = −0.0020;
  `bootstrap.pick_minus_teamfight` 2.5 % / 97.5 % = −0.0065 / +0.0026, 1,000 replicates.
- **Cut 3 / 4 / 5 values** stay `\pending{scale_cut_sensitivity}`. The artefact
  `tog_revision/A6-definition-sensitivity/scale_cut_sensitivity_v33.json` (`participation.cuts.{3,4,5}.gaps.pick_minus_teamfight`,
  400 replicates) gives +0.0107 [+0.0069, +0.0146], −0.0020 [−0.0065, +0.0027] and −0.0106 [−0.0171, −0.0043].
  The plan §2 table gives +0.0108 [+0.0070, +0.0147], −0.0019 [−0.0065, +0.0028] and −0.0105 [−0.0170, −0.0041].
  The two do not agree, and the artefact is being re-verified.

| file (anchor):line | type | context | action |
|---|---|---|---|
| docs/RELEASE_REPORT_V3.md (§1 item 2):22-30 | current-state | "0.670(규모 무관)" → qualified at `n_min ≥ 4`, pending markers for G × D and the cut values | EDITED |
| docs/RELEASE_REPORT_V3.md (§5.2):172, note 174-176 | current-state | v3 leak-affected gradient; note already qualified | MARKED |
| docs/RELEASE_REPORT_V3.md (§5.3 note):193-200 | current-state | new note: cut dependence, source and computation of −0.002 | EDITED |
| docs/RELEASE_REPORT_V3.md (§5.4 reading 4):255-256 | current-state | "규모 무관성은 모델이 만든다" → "채택 컷 n_min ≥ 4에서 클래스 간 평평함은 모델이 만든다" | EDITED |
| docs/RELEASE_REPORT_V3.md (§5.4 "읽는 법"):275-277 | current-state | "규모 기울기는 없고" → value at `n_min ≥ 4` | EDITED |
| docs/RELEASE_REPORT_V3.md (§8 item 1):333-335 | current-state | "0.670·규모 무관으로 갱신" → value at the cut plus pending cut values | EDITED |
| docs/RELEASE_REPORT_V3.md (§5.4 reading 1):252 | — | "그 격차가 사라진다" is the FT vs MLP gap | NON-MATCH |
| docs/DEFINITION_EVIDENCE.md (status note):25-28 | dated log | §22 statements hold at `n_min ≥ 4` only | EDITED |
| docs/DEFINITION_EVIDENCE.md (§22 reading):1311, 1313, 1318 | current-state | "전체 코퍼스에서 0", "규모와 무관하게", "'차이 없음'이 결론" → inline qualifiers | EDITED |
| docs/DEFINITION_EVIDENCE.md (§22.1 bullets):1335, 1345 | current-state | "규모 기울기는 없다", "규모와 무관하다" → inline qualifiers | EDITED |
| docs/DEFINITION_EVIDENCE.md (§22.1 closing note):1347-1362 | current-state | cut dependence, source, pending marker, further corrections | EDITED |
| docs/DEFINITION_EVIDENCE.md (§19):1179-1180, 1187-1188 | dated log (v3, leak-affected) | "teamfight가 가장 예측 가능 … 라벨에 무관"; presence-teamfight "가장 예측 가능" | BANNER (the status note says §19 is leak-affected) |
| docs/ENGAGEMENT_DEFINITION_V3.md (§0):20 | current-state | "규모별 차이 없음(pick − teamfight −0.002)" | **FLAG**: add "at `n_min ≥ 4`" |
| docs/ENGAGEMENT_DEFINITION_V3.md (§6):106 | current-state | "규모 기울기는 … v3.3에서 사라진다" | **FLAG**: same |
| docs/ENGAGEMENT_DEFINITION_V3.md (§7):130 | current-state | "헤드라인 0.670, 규모 무관" | **FLAG**: same |
| docs/ENGAGEMENT_DEFINITION_V3.md (§9):163 | current-state | "규모 기울기 소멸" | **FLAG**: same |
| README.md ("Retraction" heading):17 | current-state | "engagement scale does not change predictability" | **FLAG** (low): the "Clean result" bullet of the same section (lines 33-36) states the cut and that the sign depends on it; the heading does not |
| docs/TOG_EXTENSION_PLAN.md (file-top banner):5-6 | plan (superseded) | "−0.0019 [−0.0065, +0.0028] at n_min ≥ 4" | **FLAG** (low): the upper bound +0.0028 matches neither the main-loop artefact (+0.0026) nor the cut artefact (+0.0027) |
| docs/CLAUDE_TOG_PAPER_PLAN.md (§2, "§5 Prediction", cut table):150-152 | plan | reference values above | **FLAG** (low): not reproduced by the current artefact; keep them out of the manuscript until the re-verification settles |
| docs/CLAUDE_TOG_PAPER_PLAN.md (§2, "§5 Prediction"):154; (§4 never-write list):322 | plan | the rule itself | MARKED |
| docs/tog_manuscript/reviewer_response_matrix.md ("Evidence that must not be offered"):159 | current-state | "no scale gradient stated without the cut" | MARKED |

---

## 15. Adjacent findings

Edited only in W4 files; the rest are recorded.

1. **"Teamfight is the most predictable class" is still stated as live** at docs/ENGAGEMENT_WINNER_DEFINITION.md
   ("What the audit found"):68 and ("Adopted primary at the time of writing"):178. **FLAG**.
   docs/DEFINITION_EVIDENCE.md §19 (1179-1180, 1187-1188) is dated and bannered. docs/RELEASE_REPORT_V3.md §1:23
   uses the phrase only to report the retraction.
2. **§17 participation shares (EDITED).** docs/DEFINITION_EVIDENCE.md §17 (lines 877-887) now gives 19.1 / 60.3 /
   20.6 %: 101,798 / 320,878 / 109,829, each divided by 532,547, from
   `features/scale_decomposition_v33_market_event.json` `by_participation_scale`. The three counts sum to
   532,505. The v2 values 21.9 / 55.7 / 22.4 % are kept as "*was:*". The "83 % of picks are 1v2/1v3" and the
   peak-and-valley reading are still v2, marked `\pending{participation_joint_v33}`. The same v3.3 shares
   replace the v3 18.9 / 60.7 / 20.4 % in docs/RELEASE_REPORT_V3.md §4:141.
3. **RELEASE_REPORT_V3.md housekeeping (EDITED).**
   - The pilot and audit subsection is renumbered §5.5 (line 282).
   - Roadmap item 3 says "§5.4 참조" (line 337); it said "9절 참조".
   - The §3.3 evidence row is tagged as pre-leak-fix pilot (line 81).
   - No document cites RELEASE_REPORT "§5.4" for the pilot (rg at 16:27).
4. **§17 gate comparison and prediction figures (EDITED).** "(1,600, 15) 0.660 vs (1,800, 10) 0.669", "0.658 vs
   0.659" and "0.660 (초반 0.61, 중반 0.70, 후반 0.80)" come from the §16 retraining (commit c850f72, 2026-09-09
   02:04), before the leak fix (5b5f9c8, 09:33). A note at DEFINITION_EVIDENCE.md §17:899-905 says so and adds
   `\pending{presence_gate_v33}` and `\pending{gd_sensitivity_v33}`.
5. **§22 presence-teamfight AUC (EDITED).** The v3.3 value in the §22 table was 0.700. The artefact gives 0.69949
   (`by_presence_scale.teamfight.auc`), which is 0.699 at three decimals. Its "2.0%" is 11,477/566,452 = 2.03 %
   of the common population, or 2.16 % of the 532,547 labelled rows.
6. **Game-rule anchors patch-checked (EDITED).** League of Legends Wiki pages were checked on 2026-09-11:
   - "Experience (champion)": 1,600 u for champion deaths; the patch history lists minion-radius changes only.
   - "Kill": 15 s on Summoner's Rift, 20 s on Howling Abyss; no change listed.
   - Neither page records a change to 1,600 u or 15 s, including for patches 15.14-15.16.
   - Written into: CONSTANTS_JUSTIFICATION.md banner item 2; DEFINITION_EVIDENCE.md §10 table (416-417) and §15
     "Patch check" (737-742); RELEASE_REPORT_V3.md §3.2 (71) and §4 (139).
7. **TabNet footnote (EDITED).** docs/RELEASE_REPORT_V3.md §5.4 note (236-244) now cites `@arik2021tabnet`
   (AAAI 2021). It states that the quoted passage was read in arXiv:1908.07442v5 (9 Dec 2020), PDF p. 5, and adds
   `\pending{tabnet_rerun}`.
8. **Kill-less scanner details** (`scripts/run_killless_encounters.py` at 5600f8b; recorded in
   ENGAGEMENT_SCALE_DEFINITION.md 221-226):
   - The minimum duration is rounded to whole 5 s frames. 13.7 s becomes 3 frames, 10 s from first to last frame
     timestamp; 20 s becomes 4 frames.
   - A kill counts if it falls within [first frame − grace, last frame + grace].
   - Manuscript wording such as "sustained for at least 13.7 s" should reflect this.
9. **Unclear sentence, not edited.** docs/RELEASE_REPORT_V3.md §5.4 reading 4 (258-259) says the lead-only
   gradient has the opposite sign to the retracted v2 gradient. Both have teamfight above pick: lead-only
   pick − teamfight = 0.6017 − 0.6718 = −0.0701, and v2 −0.0677. The author should check what is meant.
10. **The 3,000 u interaction radius has no anchor.** Its only justification in CONSTANTS_JUSTIFICATION.md was the
    withdrawn coverage calculation (91.4 %). The SUPERSEDED banner records this.

## 16. Pending markers in W4 files (as of 16:27)

| key | where | needs |
|---|---|---|
| `gd_sensitivity_v33` | CONSTANTS_JUSTIFICATION.md:34; RELEASE_REPORT_V3.md:23; DEFINITION_EVIDENCE.md:905 | G × D sensitivity measured against the v3.3 headline |
| `scale_cut_sensitivity` | RELEASE_REPORT_V3.md:29, :196, :334; DEFINITION_EVIDENCE.md:1354 | pick − teamfight at `n_min ≥ 3, 4, 5` from the re-verified cut-sensitivity artefact |
| `tabnet_rerun` | RELEASE_REPORT_V3.md:244, :338 | TabNet test AUC on the v3.3 patch holdout with the sparsity term added to the loss |
| `ctx_sweep_v33` | RELEASE_REPORT_V3.md:300 | context-window sweep 15/30/60/120 s on v3.3 features |
| `participation_joint_v33` | DEFINITION_EVIDENCE.md:887 | v3.3 joint participation distribution (1v2/1v3 share of picks, diagonal peaks and valley) |
| `presence_gate_v33` | DEFINITION_EVIDENCE.md:904 | (1,600 u, 15 s) vs (1,800 u, 10 s) presence gate on v3.3 features |

Filled in this pass:

- `killless_v33`, from `tog_revision/killless_grid/summary.json`.
- `participation_dist_v33`, from `scale_decomposition_v33_market_event.json` `by_participation_scale`.

## 17. Dated logs left unchanged

- `docs/AUDIT.md`: CoG pre-submission code-to-paper audit.
- `docs/TOG_EXTENSION_PLAN.md`: superseded plan, already bannered (see the §14 FLAG on its banner CI).
- `docs/CoG2026_Paper.md`: CoG technical report; no hits for the searched patterns.
- The dated sections of `docs/DEFINITION_EVIDENCE.md` (§§1-9, the first-run table in §10, §19, the §22.1 table
  rows): covered by the status note. Values are not altered, except the §22 presence-teamfight AUC rounding
  (§15 item 5).

## 18. Summary of FLAGs for other owners

| file | anchor (line at 16:27) | fix |
|---|---|---|
| README.md | "Retraction" heading (17) | Qualify the heading with the cut, as the body already does |
| docs/DATA_MANIFEST.md | "Canonical results" (19, 20, 24, 25) | Move the v2 headline, the −.0652 replication and the 0.008 sweep to "Superseded"; mark `killless/*.json` as v2 and add `tog_revision/killless_grid/summary.json` |
| docs/ENGAGEMENT_DEFINITION_V3.md | §0 (20), §1 (27), §6 (106), §7 (130), §9 (163) | Qualify the scale wording with `n_min ≥ 4`; replace the 4.9 % with the v3.3 proximity-encounter share and its denominator |
| docs/references/definition_section_draft.tex | kill-less sentence (37) | v3.3 teamfight-gate share of proximity encounters, not "of engagements" |
| docs/DEFINITION_LITERATURE.md | §5 (206), §8 (268) | Same as above |
| docs/ENGAGEMENT_WINNER_DEFINITION.md | "What the audit found" (52, 54, 68), "Adopted primary" (178, 179), "Open items" (190) | "Teamfight most predictable" and the v2 decomposition to-dos |
| docs/CLAUDE_TOG_PAPER_PLAN.md | §3.1 (98); §2 "§5 Prediction" cut table (150-152) | Refer to the draft sentence by content (it is now line 37); the cut table is not yet reproduced by the artefact |
| docs/tog_manuscript/reviewer_response_matrix.md | R2-1 row (62) | The README no longer says "~1M" |
| docs/TOG_EXTENSION_PLAN.md | file-top banner (5-6) | CI upper bound +0.0028 is not in the main-loop artefact (+0.0026) |
