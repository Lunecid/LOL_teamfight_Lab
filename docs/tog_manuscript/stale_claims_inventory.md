# Stale-claims inventory (W4; first pass 2026-09-11, revised 2026-09-14)

Purpose: the released repository must not contradict the ToG manuscript. This file lists every place in
`docs/` and `README.md` where a retracted or superseded claim appears. For each hit it gives the document
type and what was done about it.

## Method

- Searched with grep over `docs/` (recursive, `*.md` and `*.tex`, including `docs/references/` and
  `docs/tog_manuscript/`) and over `README.md`, on 2026-09-14, **after** this revision's edits.
- Entries give a section anchor first and a line number second. Other agents keep editing some of these files
  (`CLAUDE_TOG_PAPER_PLAN.md`, `README.md`, the manuscript files), so for those files **the anchor is
  authoritative** and the line number is as of 2026-09-14 05:04. Of the non-W4 files cited here, only
  `docs/tog_manuscript/sec_limitations.tex` changed after the 2026-09-11 16:27 search (modified 2026-09-14 05:03,
  still being edited). It was searched again at 05:04 and has no hit for any pattern.
- Patterns: `948,?369`; `one million|million engagements|~ ?1 ?M|1M|≈ ?1 ?M|100만`; `0\.7842`; `0\.0677`;
  `0\.0652`; `0\.493`; `89\.[13]`; `4\.9 ?%` (and the LaTeX form `4.9\%`); `decompos|규모에 따라 분해`;
  `21\.1|36\.9|42\.0`.
- Also searched: the rounded forms `.784`, `−.068`, `−.065`; the kill-less and sweep phrases `0.008`,
  `brief pass`, `0.15 per match`, `경기당 0.15`, `2.03`; the scale-gradient wording `규모 무관`,
  `규모와 무관`, `규모 기울기는 없`, `기울기가 없`, `규모별 차이 없음`, `기울기 소멸`, `사라진다`,
  `does not change predictability`, `no scale gradient`, `가장 예측 가능`, `most predictable`; and, new in
  this revision, the remaining items of `docs/CLAUDE_TOG_PAPER_PLAN.md` §4: `52 ?%|commitment gap`,
  `TreeSHAP|Table VI`, `융합|vision fusion`, `본질적으로 예측 불가|inherently unpredictable`,
  `승패 예측을 개선`, `확률 라벨`, `tier|티어|player-level` (results in §18).
- `4\.9 ?%` also matches 14.9 % and 54.9 %; those hits are listed as non-matches.

**Document types.**

- *current-state*: read as a description of the repository or definition as it is now.
- *dated log*: a dated research or audit record.
- *plan*: a planning document.
- `DEFINITION_EVIDENCE.md` is a dated log, but three of its sections are read as current: §14 and §17
  (`corpus_shards_v33/manifest.json` cites §17 as "the definition"), and §22, which its status note names as
  current. Hits in those sections were treated as current-state.

**Actions.**

- EDITED: changed by W4 (only files owned by W4).
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
| docs/DEFINITION_EVIDENCE.md (§17 shares note):887 | current-state | "*was:*" note under the v3.3 shares | EDITED |
| docs/DEFINITION_EVIDENCE.md (§19):1182 | dated log | v2 comparison row | BANNER |
| docs/DEFINITION_EVIDENCE.md (§22.1 table):1348 | current-state | v2 comparison row, retracted by the §22.1 bullets | MARKED |
| docs/ENGAGEMENT_DEFINITION_V3.md (§6):117 | current-state | struck-through v2 row | MARKED |
| docs/RELEASE_REPORT_V3.md (§5.1 table):157 | current-state | definition-change row | EDITED: the note at 159-163 says the rows are pre-leak-fix and the v2 values are retracted |
| docs/RELEASE_REPORT_V3.md (§5.1 note):162 | current-state | that note | EDITED (withdrawal notice) |
| docs/RELEASE_REPORT_V3.md (§5.3 table):192 | current-state | v2 row | EDITED: labelled "누수 특징; retracted" |

## 2. "one million" / "~1M" engagements

The FLAGs from the first pass on README.md:13 ("~1M") and README.md:148 ("approximately one million") are
withdrawn: the README contains neither phrase (checked again on 2026-09-14).

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
| docs/DEFINITION_EVIDENCE.md (§19):1182 | dated log | "0.784 (42%)" | BANNER |
| docs/DEFINITION_EVIDENCE.md (§22.1 table):1348 | current-state | "0.784" | MARKED |
| docs/ENGAGEMENT_DEFINITION_V3.md (§6):117 | current-state | struck-through row | MARKED |
| docs/RELEASE_REPORT_V3.md (§5.3 table):192 | current-state | v2 row | EDITED (labelled retracted) |

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
| docs/DEFINITION_EVIDENCE.md (§19):1182 | dated log | "−0.068 [−0.070, −0.065]" | BANNER |
| docs/DEFINITION_EVIDENCE.md (§22.1 table):1348 | current-state | "−0.068" | MARKED |
| docs/ENGAGEMENT_DEFINITION_V3.md (§6):117 | current-state | struck-through row | MARKED |
| docs/TOG_EXTENSION_PLAN.md (file-top banner):6 | plan (superseded) | banner cites −0.068 as retracted | MARKED |
| docs/RELEASE_REPORT_V3.md (§5.3 table):192 | current-state | v2 row | EDITED (labelled retracted) |
| docs/RELEASE_REPORT_V3.md (§5.4 reading 4, note):280 | current-state | −0.0677 quoted as the v2 comparison value in the 2026-09-14 note | EDITED (see §15 item 9) |
| docs/tog_manuscript/reviewer_response_matrix.md ("Evidence that must not be offered"):159 | current-state | never-offer list | MARKED |

## 5. `−0.0652` (cross-patch "replication" of the retracted gap)

| file (anchor):line | type | context | action |
|---|---|---|---|
| docs/CLAUDE_TOG_PAPER_PLAN.md (§4 never-write list):326 | plan | "−0.0652 교차 패치 재현" | MARKED |
| docs/DATA_MANIFEST.md ("Canonical results"):20 | current-state | "replication: 47,759 eng/10,612 matches, pick-tf -.0652" | **FLAG**: move to Superseded; it replicates a leak-affected gap |
| docs/DEFINITION_EVIDENCE.md (§4):258, (§19):1182 | dated log | "−.065" is the upper CI bound of the v2 gap | NON-MATCH |

## 6. `89.3 %` (Data Dragon coverage argument for R = 1,800 u)

| file (anchor):line | type | context | action |
|---|---|---|---|
| docs/CONSTANTS_JUSTIFICATION.md (SUPERSEDED banner, item 2):17 | current-state (superseded) | withdrawal notice | EDITED |
| docs/CONSTANTS_JUSTIFICATION.md (body):50, :57 | same | "89.3% cast within 1,800 u"; validity-radius row | BANNER |
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
| docs/DEFINITION_EVIDENCE.md (§10 estimator table, R and B/M rows):416-417 | current-state | rows give R = 1,600 u and B = 15 s as rule anchors, with wiki source, check date and patch note, and "*was:*" notes | EDITED |
| docs/DEFINITION_EVIDENCE.md (§10 drift rule):424-426 | current-state | R-coverage criterion skipped for rule-anchored specs | EDITED |
| docs/DEFINITION_EVIDENCE.md (§10 "R에 대한 주의"):438, note 439-442 | dated log | superseded note | EDITED |
| docs/DEFINITION_EVIDENCE.md (§10 first-run table):460-463 | dated log (2026-09-08) | R-coverage column | LEFT (the note at 439-442 says so) |
| docs/DEFINITION_EVIDENCE.md (§14 verdict):646, note 647-649 | current-state | regenerated spec has no coverage criterion | EDITED |
| docs/DEFINITION_EVIDENCE.md (§14 current definition):703-715, "*was:*" at 709 | current-state | R = 1,600 u, B = 15 s with spec values | EDITED |
| docs/DEFINITION_EVIDENCE.md (§15 "Data Dragon 주의"):752 | dated log | "669 스펠 89.1%는 … 보조 근거로 강등" | MARKED |

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
| docs/tog_manuscript/sec_definition.tex (kill-less "Frequency" paragraph):396-398 | current-state (manuscript) | "305 of 6,195 proximity encounters on 2,000 matches (4.9\%)" under the conference-version constants, with the right denominator | MARKED |
| docs/tog_manuscript/sec_definition.tex (same paragraph):400-401 | current-state (manuscript) | `\pending{killless}{v3.3 grid: ... --radius 1600 --grace-ms 15000 on 2,000 matches ...}`: the result now exists, and the run description is wrong (20,000 matches, explicit `--min-per-team`/`--min-duration`/`--grace-ms`) | **FLAG** (new): fill from `killless_grid/summary.json` as above; `sec_limitations.tex` already cites the grid |
| docs/ENGAGEMENT_DEFINITION_V3.md (§1 table, row 1):27 | current-state | "킬 없는 encounter(4.9%)는 범위 밖 명시" | **FLAG**: replace with the v3.3 teamfight-gate share of proximity encounters, or drop the number |
| docs/DATA_MANIFEST.md ("Canonical results"):25 | current-state | "`killless/*.json` meta-review bound (4.9% at teamfight scale)" | **FLAG**: mark as v2 (R 1,800 u, 10 s grace) and add `tog_revision/killless_grid/summary.json` as the v3.3 grid |
| docs/DEFINITION_LITERATURE.md (§5, validity table):206 | current-state | "(4.9% 상한)" | **FLAG**: v3.3 value and its denominator |
| docs/DEFINITION_LITERATURE.md (§8, sentence list):268 | current-state | "우리 4.9%" | **FLAG**: same |
| docs/DEFINITION_EVIDENCE.md (status note):22-24 | dated log | points to the v3.3 grid via the §17 note | EDITED |
| docs/DEFINITION_EVIDENCE.md (§5):288 | dated log | "4.9% (경기당 0.15개 vs … 2.03개)" | BANNER (the status note names it) |
| docs/DEFINITION_EVIDENCE.md (§9):372 | dated log | "맞선 킬 없는 encounter는 4.9%다" | BANNER |
| docs/DEFINITION_EVIDENCE.md (§14 current definition):711-714 | current-state | names the draft's kill-less sentence by content and points to the v3.3 grid | EDITED |
| docs/DEFINITION_EVIDENCE.md (§17 "주장하지 않는 것"):921-929 | current-state | v3.3 teamfight-gate share, sensitivity, denominator; 4.9 % kept as "*was:*" at 928 | EDITED |
| docs/ENGAGEMENT_SCALE_DEFINITION.md (STALE banner):19-21 | current-state (stale) | names the v2 table, the pairing, "brief passes" and the radius-insensitivity sentence | EDITED |
| docs/ENGAGEMENT_SCALE_DEFINITION.md (kill-less table):209 | same | v2 row | BANNER + STALE note 211-263 |
| docs/ENGAGEMENT_SCALE_DEFINITION.md (STALE note, item 1):217-226 | same | 0.15 and 2.03 sourced; 4.9 % = 305/6,195 | EDITED |
| docs/ENGAGEMENT_SCALE_DEFINITION.md (STALE note, v3.3 re-measurement):232-263 | same | grid table, scanner details, denominator | EDITED |
| docs/ENGAGEMENT_SCALE_DEFINITION.md ("Two things follow"):267-271 | same | 4.9 % tagged "[v2 constants; v3.3 teamfight gate: 3.09 %]"; the pairing and "brief passes" struck through | EDITED |
| docs/ENGAGEMENT_SCALE_DEFINITION.md ("The share is insensitive to the radius"):273-280 | same | v2 claim (2 per side); tagged STALE with the v3.3 teamfight-gate radius range 2.00-3.47 % | EDITED (2026-09-14) |
| docs/ENGAGEMENT_SCALE_DEFINITION.md (closing bullets):286-293 | same | "1,800 units (the detector's gate value)" tagged as v2; "separate planned experiment" tagged [STALE] | EDITED |
| docs/RELEASE_REPORT_V3.md (§4 definition table, 킬 ≥ 1 row):137 | current-state | v3.3 teamfight-gate numbers; 4.9 % as "*was:*" | EDITED |
| docs/CLAUDE_TOG_PAPER_PLAN.md (§3.1 table and sensitivity):81-96 | plan | v3.3 grid, matches the artefact | MARKED |
| docs/CLAUDE_TOG_PAPER_PLAN.md (§3.1 "분모를 명시한다"):98-99; (§5 item 2):342 | plan | denominator warning and task; both cite `definition_section_draft.tex:36`, now line 37 | MARKED; **FLAG** (low): refer to the sentence by content |
| docs/tog_manuscript/reviewer_response_matrix.md ("Evidence that must not be offered"):155 | current-state | never-offer list | MARKED |
| docs/DEFINITION_EVIDENCE.md (§3.2 table):162 | — | base-region pair share 4.9% | NON-MATCH |
| docs/DEFINITION_EVIDENCE.md (§3.2):170 | — | 14.9% | NON-MATCH |
| docs/DEFINITION_EVIDENCE.md (§14):686, (§18):976 | — | 54.9% | NON-MATCH |
| docs/ENGAGEMENT_WINNER_DEFINITION.md:143 | — | 54.9% | NON-MATCH |

## 9. "0.15 per match vs 2.03 teamfight-class engagements per match"

The two numbers count different things:

- **0.15** = 305/2,000 = 0.1525 kill-less proximity encounters per match (`features/killless/r1800_d20_t3.json`).
- **2.03** = 416,956/205,884 = 2.025 teamfight-class engagements per match under the v2 rule `n_min ≥ 3`
  (`D:/LOL_Project/fusion_2615/features/scale_decomposition.json`, `by_participation_scale.teamfight.n` and
  `n_matches`; v2 detector, 993,484 engagements). The first pass of this file said no artefact records 2.03;
  that was wrong.

| file (anchor):line | type | context | action |
|---|---|---|---|
| docs/DEFINITION_EVIDENCE.md (status note):22 | dated log | names the pairing as history | EDITED |
| docs/DEFINITION_EVIDENCE.md (§5):288 | dated log | original pairing | BANNER |
| docs/ENGAGEMENT_SCALE_DEFINITION.md (STALE banner):19-20 | current-state (stale) | names the pairing | EDITED |
| docs/ENGAGEMENT_SCALE_DEFINITION.md (STALE note, item 1):217-226 | same | unlike-quantities objection kept; both numbers sourced | EDITED |
| docs/ENGAGEMENT_SCALE_DEFINITION.md ("Two things follow"):268-269 | same | struck through | EDITED |
| docs/DEFINITION_EVIDENCE.md (§22.1 further corrections):1387 | — | "2.03 %" is the presence-teamfight share of the common population | NON-MATCH |

## 10. `0.493` ("predictability inversely related to importance")

| file (anchor):line | type | context | action |
|---|---|---|---|
| docs/CLAUDE_TOG_PAPER_PLAN.md (§4 never-write list):323 | plan | retracted | MARKED |

No other hit.

## 11. "decomposes by scale"

`decompos` also matches script and artefact names (`run_scale_decomposition.py`, `scale_decomposition_*.json`)
and the memmap description in `DATA_MANIFEST.md`/`README.md`. Those carry no claim and are not listed.

| file (anchor):line | type | context | action |
|---|---|---|---|
| docs/ENGAGEMENT_SCALE_DEFINITION.md (STALE banner):7 | current-state (stale) | withdrawal notice | EDITED |
| docs/ENGAGEMENT_SCALE_DEFINITION.md (intro):26 | same | "The ToG extension's headline is a decomposition by engagement scale" | BANNER |
| docs/ENGAGEMENT_SCALE_DEFINITION.md (shop-event sensitivity):88 | same | "class gaps are ~0.05 AUC" | BANNER |
| docs/TOG_EXTENSION_PLAN.md (title, banner, "Thesis", "Equitable deep-learning comparison"):1, 4, 18, 23, 33, 86 | plan (superseded, banner) | title, thesis, "teamfight highest" | MARKED |
| docs/CLAUDE_TOG_PAPER_PLAN.md (§0 table):19; (§4 never-write list):321 | plan | "규모에 따라 분해된다" as retracted | MARKED |
| docs/tog_manuscript/reviewer_response_matrix.md ("How to read this file"):35 | current-state | states the retraction | MARKED |
| docs/ENGAGEMENT_WINNER_DEFINITION.md ("What the audit found"):52, 54 | current-state (v2-era label audit) | label policy "entangled with the scale decomposition" | **FLAG**, together with §15 item 1 |
| docs/ENGAGEMENT_WINNER_DEFINITION.md ("Adopted primary…", "Open items"):179, 190 | current-state | to-dos to re-score the v2 "full-corpus decomposition" | **FLAG** (low): mark as v2 to-dos |

## 12. `21.1 / 36.9 / 42.0` (v2 class shares, teamfight = `n_min ≥ 3`)

| file (anchor):line | type | context | action |
|---|---|---|---|
| docs/ENGAGEMENT_SCALE_DEFINITION.md (STALE banner):11-14 | current-state (stale) | withdrawal notice; also names the "reported at `n_min ≥ 3`" rule at 131-133 | EDITED |
| docs/ENGAGEMENT_SCALE_DEFINITION.md (shop-event sensitivity):80 | same | "21.1/37.0/42.0 to 22.0/36.7/41.3" | BANNER |
| docs/ENGAGEMENT_SCALE_DEFINITION.md (class-rule table):110-112 | same | 42.0 / 36.9 / 21.1 % | BANNER |
| docs/CLAUDE_TOG_PAPER_PLAN.md (§4 never-write list):329 | plan | retracted | MARKED |
| docs/references/definition_refs.bib:591, :599; docs/tog_manuscript/references_audit.md:184 | — | DOI `10.1016/j.inffus.2021.11.011` | NON-MATCH |

## 13. Threshold-sweep "AUC spread 0.008"

Source: `D:/LOL_Project/fusion_2615/features/thresholds/*.json`: 553 matches, six one-factor settings on the v2
detector, 2,242-2,783 labelled rows per setting, AUC 0.593-0.601. The replacement, a G × D sensitivity against
the v3.3 headline, is `\pending{gd_sensitivity_v33}` (no v3.3 output on 2026-09-14; only smoke runs under
`tog_revision/A6-definition-sensitivity/smoke/`).

| file (anchor):line | type | context | action |
|---|---|---|---|
| docs/CONSTANTS_JUSTIFICATION.md (SUPERSEDED banner, item 3):31-36 | current-state (superseded) | withdrawal notice with pending marker | EDITED |
| docs/CONSTANTS_JUSTIFICATION.md (body):69 | same | "AUC spread 0.008" | BANNER |
| docs/ENGAGEMENT_SCALE_DEFINITION.md (STALE banner):17-18 | current-state (stale) | withdrawal notice | EDITED |
| docs/ENGAGEMENT_SCALE_DEFINITION.md (recap):45 | same | "AUC moves 0.008" | BANNER |
| docs/DEFINITION_EVIDENCE.md (status note):24 | dated log | withdrawal notice | EDITED |
| docs/DEFINITION_EVIDENCE.md (§3.1):114 | dated log | "AUC 변동 0.008" | BANNER |
| docs/DATA_MANIFEST.md ("Canonical results"):24 | current-state | "R1 threshold sweep (AUC spread .008)" | **FLAG**: add "553 matches, v2 detector, one factor at a time; not against the v3.3 headline" |
| docs/TOG_EXTENSION_PLAN.md ("Equitable deep-learning comparison" table, R1-2 row):93 | plan (superseded) | "done" row | MARKED |
| docs/RELEASE_REPORT_V3.md (§5.3 table):184; docs/DEFINITION_EVIDENCE.md (§22.1 table):1340 | — | "−0.008" is a v3.3 pick − teamfight value | NON-MATCH |
| docs/CLAUDE_TOG_PAPER_PLAN.md (§7):245, 250 | — | "+0.0081" is a CI bound | NON-MATCH |

## 14. "No scale gradient" / "규모 무관" without the cut

`docs/CLAUDE_TOG_PAPER_PLAN.md` §4 (line 322) forbids writing "규모 기울기가 없다" or "pick−tf = 0", because the
pick − teamfight gap depends on the teamfight cut.

- **Main-loop value at the adopted cut `n_min ≥ 4`**: −0.002 [−0.007, +0.003]. Source:
  `features/scale_decomposition_v33_market_event.json`; class AUCs 0.67887 − 0.68089 = −0.0020;
  `bootstrap.pick_minus_teamfight` 2.5 % / 97.5 % = −0.0065 / +0.0026, 1,000 replicates. The same key's
  bootstrap `mean` is −0.0019.
- **Cut 3 / 4 / 5 values (filled 2026-09-14)**: +0.0108 [+0.0070, +0.0147], −0.0019 [−0.0065, +0.0028] and
  −0.0105 [−0.0170, −0.0041]. Source: `tog_revision/A6-definition-sensitivity/scale_cut_sensitivity_v33.json`,
  `participation.cuts.{3,4,5}.gaps.pick_minus_teamfight` (`point`, `ci_2.5`, `ci_97.5`), 400 replicates, written
  2026-09-11 16:22:40. These equal the plan's §2 reference values to four decimals, and the file's
  `reproduces_main_loop_table.reproduced_under` is `["zero"]`.
- **Why the first pass and the fact-check saw different numbers.** The file reads the 42 rows whose
  participation count is −1 either as zero (pick; the primary `participation.cuts`) or as unclassified
  (`participation_other_negative_count_reading.cuts`). The second reading gives +0.0107 [+0.0069, +0.0146],
  −0.0020 [−0.0065, +0.0027] and −0.0106 [−0.0171, −0.0043]. Those are also the primary values of the pre-rewrite
  copy `superseded/scale_cut_sensitivity_v33.20260911_1127.json`. The first pass of this file quoted that set.
- **Why the cut-4 values differ from the main loop**: the main loop leaves the 42 rows unclassified and uses 1,000
  replicates.

| file (anchor):line | type | context | action |
|---|---|---|---|
| docs/RELEASE_REPORT_V3.md (§1 item 2):20-31 | current-state | "0.670(규모 무관)" qualified at `n_min ≥ 4`; cut values filled | EDITED |
| docs/RELEASE_REPORT_V3.md (§5.2):173, note 175-177 | current-state | v3 leak-affected gradient; note already qualified | MARKED |
| docs/RELEASE_REPORT_V3.md (§5.3 note):194-218 | current-state | cut dependence, source of −0.002, cut values and both readings | EDITED |
| docs/RELEASE_REPORT_V3.md (§5.4 reading 4):273-274 | current-state | "규모 무관성은 모델이 만든다" → "채택 컷 n_min ≥ 4에서 클래스 간 평평함은 모델이 만든다" | EDITED |
| docs/RELEASE_REPORT_V3.md (§5.4 "읽는 법"):300-302 | current-state | "규모 기울기는 없고" → value at `n_min ≥ 4` | EDITED |
| docs/RELEASE_REPORT_V3.md (§8 item 1):358-361 | current-state | "0.670·규모 무관으로 갱신" → value at the cut plus cut values | EDITED |
| docs/RELEASE_REPORT_V3.md (§5.4 reading 1):270 | — | "그 격차가 사라진다" is the FT vs MLP gap | NON-MATCH |
| docs/DEFINITION_EVIDENCE.md (status note):25-28 | dated log | §22 statements hold at `n_min ≥ 4` only | EDITED |
| docs/DEFINITION_EVIDENCE.md (§22 reading):1326, 1328, 1333 | current-state | "전체 코퍼스에서 0", "규모와 무관하게", "'차이 없음'이 결론" → inline qualifiers | EDITED |
| docs/DEFINITION_EVIDENCE.md (§22.1 bullets):1350, 1360 | current-state | "규모 기울기는 없다", "규모와 무관하다" → inline qualifiers | EDITED |
| docs/DEFINITION_EVIDENCE.md (§22.1 closing note):1362-1388 | current-state | cut dependence, source, cut values and both readings, further corrections | EDITED |
| docs/DEFINITION_EVIDENCE.md (§19.1):1230 | — | "불일치 비율은 규모와 무관" is the label-disagreement rate by class | NON-MATCH |
| docs/DEFINITION_EVIDENCE.md (§19):1194-1195, 1202-1203 | dated log (v3, leak-affected) | "teamfight가 가장 예측 가능 … 라벨에 무관"; presence-teamfight "가장 예측 가능" | BANNER (the status note says §19 is leak-affected) |
| docs/DEFINITION_EVIDENCE.md (§18 market_event bullet):1092-1094 | dated log (pre-leak-fix pilot) | "'teamfight가 가장 예측 가능'은 모든 라벨에서 유지된다" | EDITED (2026-09-14): inline tag, pilot committed in bd8a8c8 at 05:04 before the leak fix 5b5f9c8 at 09:33 |
| docs/ENGAGEMENT_DEFINITION_V3.md (§0):20 | current-state | "규모별 차이 없음(pick − teamfight −0.002)" | **FLAG**: add "at `n_min ≥ 4`" |
| docs/ENGAGEMENT_DEFINITION_V3.md (§6):106 | current-state | "규모 기울기는 … v3.3에서 사라진다" | **FLAG**: same |
| docs/ENGAGEMENT_DEFINITION_V3.md (§7):130 | current-state | "헤드라인 0.670, 규모 무관" | **FLAG**: same |
| docs/ENGAGEMENT_DEFINITION_V3.md (§9):163 | current-state | "규모 기울기 소멸" | **FLAG**: same |
| README.md ("Retraction" heading):17 | current-state | "engagement scale does not change predictability" | **FLAG** (low): the "Clean result" bullet of the same section (lines 33-36) states the cut and that the sign depends on it; the heading does not |
| README.md ("Retraction", "Clean result"):33-34 | current-state | "−0.0019 (95 % CI −0.0065 to +0.0026)" | **FLAG** (low): −0.0019 is the main-loop bootstrap mean; the point estimate is −0.0020. Either quote the point, or say "bootstrap mean" |
| docs/TOG_EXTENSION_PLAN.md (file-top banner):5-6 | plan (superseded) | "−0.0019 [−0.0065, +0.0028] at n_min ≥ 4" | MARKED. This equals the cut-sensitivity artefact's primary cut-4 value. The first pass's FLAG, which said it matched neither artefact, was wrong. |
| docs/CLAUDE_TOG_PAPER_PLAN.md (§2, "§5 Prediction", cut table):150-152 | plan | reference values | MARKED: reproduced by the current cut-sensitivity artefact (first-pass FLAG withdrawn) |
| docs/CLAUDE_TOG_PAPER_PLAN.md (§2, "§5 Prediction"):154; (§4 never-write list):322 | plan | the rule itself | MARKED |
| docs/tog_manuscript/reviewer_response_matrix.md ("Evidence that must not be offered"):159 | current-state | "no scale gradient stated without the cut" | MARKED |

---

## 15. Adjacent findings

Edited only in W4 files; the rest are recorded.

1. **"Teamfight is the most predictable class" is still stated as live** at docs/ENGAGEMENT_WINNER_DEFINITION.md
   ("What the audit found"):68 and ("Adopted primary at the time of writing"):178. **FLAG**.
   docs/DEFINITION_EVIDENCE.md §19 (1194-1195, 1202-1203) is dated and bannered. §18 (1092-1094) now has an inline
   tag. docs/RELEASE_REPORT_V3.md §1:25 uses the phrase only to report the retraction.
2. **§17 participation shares and joint distribution (EDITED).**
   - docs/DEFINITION_EVIDENCE.md §17 (884-888) gives 19.1 / 60.3 / 20.6 %: 101,798 / 320,878 / 109,829, each
     divided by 532,547, from `features/scale_decomposition_v33_market_event.json` `by_participation_scale`. The
     three counts sum to 532,505; the other 42 rows have a participation count of −1.
   - The v2 values 21.9 / 55.7 / 22.4 % are kept as "*was:*". The same v3.3 shares replace the v3
     18.9 / 60.7 / 20.4 % in docs/RELEASE_REPORT_V3.md §4:142.
   - New on 2026-09-14 (890-901): the "83 % of picks are 1v2/1v3" and the peak-and-valley reading now have v3.3
     values from `tog_revision/A6-definition-sensitivity/scale_participation_v33.json`,
     `populations.labelled.participation`:
     - 85,616/101,840 = 84.1 % of picks are 1v2 or 1v3.
     - The diagonal counts are 2,217 / 78,274 / 40,551 / 32,835 / 38,474, with peaks at 2v2 and 5v5 and a
       trough at 4v4 of depth 0.147.
     - 4v4 lies below 3v3 and 5v5 in all 1,000 replicates.
3. **RELEASE_REPORT_V3.md housekeeping (EDITED).**
   - The pilot and audit subsection is numbered §5.5 (line 307).
   - Roadmap item 3 says "§5.4 참조" (line 363); it said "9절 참조".
   - The §3.3 evidence row is tagged as a pre-leak-fix pilot (line 82).
   - No document cites RELEASE_REPORT "§5.4" for the pilot.
4. **§17 gate comparison and prediction figures (EDITED).** "(1,600, 15) 0.660 vs (1,800, 10) 0.669", "0.658 vs
   0.659" and "0.660 (초반 0.61, 중반 0.70, 후반 0.80)" come from the §16 retraining (commit c850f72, 2026-09-09
   02:04), before the leak fix (5b5f9c8, 09:33). A note at DEFINITION_EVIDENCE.md §17:913-919 says so and adds
   `\pending{presence_gate_v33}` and `\pending{gd_sensitivity_v33}`.
5. **§22 presence-teamfight AUC (EDITED).** The v3.3 value in the §22 table was 0.700. The artefact gives 0.69949
   (`by_presence_scale.teamfight.auc`), which is 0.699 at three decimals (line 1306). Its "2.0%" is
   11,477/566,452 = 2.03 % of the common population, or 2.16 % of the 532,547 labelled rows.
6. **Game-rule anchors patch-checked (EDITED).** League of Legends Wiki pages were checked on 2026-09-11 and
   re-checked on 2026-09-14 (https://wiki.leagueoflegends.com/en-us/Experience_(champion),
   https://wiki.leagueoflegends.com/en-us/Kill):
   - "Experience (champion)": 1,600 u for champion deaths, 1,500 u for minions. The patch history lists
     minion-radius changes only (V4.11: 1,250 → 1,400; V25.S1.1: 1,400 → 1,500).
   - "Kill": 15 s on Summoner's Rift, 20 s on Howling Abyss; no change listed.
   - Neither page records a change to 1,600 u or 15 s, including for patches 15.14-15.16.
   - Written into: CONSTANTS_JUSTIFICATION.md banner item 2 (with URLs); DEFINITION_EVIDENCE.md §10 table (416-417)
     and §15 "Patch check" (737-744, with URLs); RELEASE_REPORT_V3.md §3.2 (72) and §4 (140).
7. **TabNet footnote (EDITED).** docs/RELEASE_REPORT_V3.md §5.4 note (254-262) cites `@arik2021tabnet`
   (AAAI 2021). It states that the quoted passage was read in arXiv:1908.07442v5 (9 Dec 2020), PDF p. 5, and adds
   `\pending{tabnet_rerun}`.
8. **Kill-less scanner details** (`scripts/run_killless_encounters.py` at 5600f8b, lines 87 and 98; recorded in
   ENGAGEMENT_SCALE_DEFINITION.md 238-243):
   - The minimum duration is converted with `round(min_duration_s * 1000 / 5000)` frames: 13.7 s → 3 frames
     (10 s from first to last frame timestamp), 20 s → 4 frames (15 s).
   - A kill counts if it falls within [first frame − grace, last frame + grace].
   - Manuscript wording such as "sustained for at least 13.7 s" should reflect this.
9. **Sign sentence in RELEASE_REPORT_V3.md §5.4 reading 4 (EDITED 2026-09-14: note added, sentence kept).**
   Lines 276-277 say the lead-only gradient has the opposite sign to the retracted v2 gradient. Both put teamfight
   above pick: lead-only pick − teamfight = 0.6017 − 0.6718 = −0.0701, and v2 −0.0677. The note (278-284) records
   what does differ in sign: the gain of the full model over lead-only is +0.0782 in pick and +0.0090 in teamfight
   (`model_comparison_v33_patch.json`). The `time_norm` leak gain was +0.0026 in pick and +0.0491 in teamfight
   (`leak_ablation_v33.json`). The author should confirm which comparison was meant.
10. **The 3,000 u interaction radius has no anchor.** Its only justification in CONSTANTS_JUSTIFICATION.md was the
    withdrawn coverage calculation (91.4 %). The SUPERSEDED banner records this.

## 16. Pending markers in W4 files (as of 2026-09-14)

| key | where | needs |
|---|---|---|
| `gd_sensitivity_v33` | CONSTANTS_JUSTIFICATION.md:36; RELEASE_REPORT_V3.md:23; DEFINITION_EVIDENCE.md:919 | G × D sensitivity measured against the v3.3 headline |
| `tabnet_rerun` | RELEASE_REPORT_V3.md:262, :364 | TabNet test AUC on the v3.3 patch holdout with the sparsity term added to the loss |
| `ctx_sweep_v33` | RELEASE_REPORT_V3.md:325 | context-window sweep 15/30/60/120 s on v3.3 features |
| `presence_gate_v33` | DEFINITION_EVIDENCE.md:918 | (1,600 u, 15 s) vs (1,800 u, 10 s) presence gate on v3.3 features |

On 2026-09-14 none of these had a v3.3 output: `tog_revision/A1-deep-baselines`, `A3-temporal-windows` and the
G × D / presence-gate folders hold only `smoke/` runs.

Filled:

- `killless_v33` (2026-09-11), from `tog_revision/killless_grid/summary.json`.
- `participation_dist_v33` (2026-09-11), from `scale_decomposition_v33_market_event.json` `by_participation_scale`.
- `scale_cut_sensitivity` (2026-09-14), from `scale_cut_sensitivity_v33.json` `participation.cuts`, whose values equal
  the reference to four decimals (§14). Filled at RELEASE_REPORT_V3.md:29-31, :198-213, :359-360 and
  DEFINITION_EVIDENCE.md:1369-1380.
- `participation_joint_v33` (2026-09-14), from `scale_participation_v33.json` (§15 item 2). Filled at
  DEFINITION_EVIDENCE.md:890-901.

## 17. Dated logs left unchanged

- `docs/AUDIT.md`: CoG pre-submission code-to-paper audit.
- `docs/TOG_EXTENSION_PLAN.md`: superseded plan, already bannered.
- `docs/CoG2026_Paper.md`: CoG technical report; no hits for the searched patterns.
- The dated sections of `docs/DEFINITION_EVIDENCE.md` (§§1-9, the first-run table in §10, §19, the §22.1 table
  rows): covered by the status note. Values are not altered, except the §22 presence-teamfight AUC rounding
  (§15 item 5). §18 has an inline tag only (§14 table).

## 18. Other items on the plan's never-write list (`CLAUDE_TOG_PAPER_PLAN.md` §4, lines 320-335)

| item | file (anchor):line | type | action |
|---|---|---|---|
| "cutoff 이후 참여자 52 %" | docs/ENGAGEMENT_SCALE_DEFINITION.md ("How the two definitions relate"):159-162 | current-state (stale) | EDITED (2026-09-14). The banner item is at 15-16 and a STALE note at 166-178. The 52 % / −0.89 come from a 1,366-engagement v2 sample. DEFINITION_EVIDENCE.md §4:272 reports 35.9 % on the full v2 corpus (not re-checked). The v3.3 smaller-side gap is negative for 194,468/532,547 = 36.5 %, mean −0.34 (`scale_participation_v33.json`, `crosstab_presence_nmin_rows_participation_nmin_cols`). |
| same | docs/DEFINITION_EVIDENCE.md (§4):272 | dated log | BANNER (v2 full corpus, 35.9 %; not the 52 %) |
| CoG Table VI TreeSHAP ranking | docs/AUDIT.md:54, :120 | dated log | LEFT |
| same | docs/DATA_MANIFEST.md ("Canonical results"):22 | current-state | **FLAG** (new): "`shap_mlex.json` attribution (tower distance top)" is the retracted ranking, whose top feature is the anchor leak; move it to Superseded |
| same | docs/CLAUDE_TOG_PAPER_PLAN.md (§4):324, (§6):378, (§7):393, (§8):415; docs/tog_manuscript/reviewer_response_matrix.md (R3-5 row):86, (:108), ("must not be offered"):154; docs/TOG_EXTENSION_PLAN.md (R1-4/R3 row):95 | plan / current-state | MARKED (each presents it as not to be revived, or is a to-do) |
| deep-model CIs of `TOG_EXTENSION_PLAN.md` | docs/TOG_EXTENSION_PLAN.md ("Equitable deep-learning comparison"):78-81 | plan (superseded) | MARKED (the file-top banner names the five paired CIs) |
| same | docs/DATA_MANIFEST.md ("Canonical results"):21 | current-state | **FLAG** (new): "`deep_mlex_fast.json` / `deep_mlex_tokens.json` fair DL table (LGBM .7311 > FT .7193 > …)" is the random-split, old-label table; move it to Superseded |
| same | docs/tog_manuscript/reviewer_response_matrix.md ("must not be offered"):150-153 | current-state | MARKED |
| replay-vision fusion figures | docs/TOG_EXTENSION_PLAN.md (file-top banner):11 | plan (superseded) | MARKED; no numeric fusion claim found elsewhere (hits for `융합` in CLAUDE_TOG_PAPER_PLAN.md 244, 258 and 392 are about state reconstruction or the deletion decision) |
| "결정적 교전은 본질적으로 예측 불가능" | docs/CLAUDE_TOG_PAPER_PLAN.md (§4):331 | plan | MARKED; no other hit |
| "교전 예측이 승패 예측을 개선한다" | docs/RELEASE_REPORT_V3.md (§1 item 5):40, (§8 item 4):366; docs/ENGAGEMENT_DEFINITION_V3.md:139 | current-state | NON-MATCH: each states the layer as a planned question, not as a result. The plan (§4:332) records the answer as null. When that layer is written up, these lines should say so |
| "확률 라벨이 골드 라벨보다 잘 예측된다" | docs/CLAUDE_TOG_PAPER_PLAN.md (§4):333 | plan | MARKED; no other hit |
| tier / player-level claims | docs/CLAUDE_TOG_PAPER_PLAN.md (§4):334, (§2 "§8 Discussion"):280-281; README.md:83; docs/tog_manuscript/sec_limitations.tex (sampling-frame and player-level paragraphs):340-351; reviewer_response_matrix.md:48 | plan / current-state | MARKED: each states tier as a collection filter and player-level analysis as not possible. No live tier claim found |

## 19. Summary of FLAGs for other owners

| file | anchor (line on 2026-09-14) | fix |
|---|---|---|
| docs/tog_manuscript/sec_definition.tex | kill-less "Frequency" paragraph (400-401) | Fill `\pending{killless}` from `tog_revision/killless_grid/summary.json` (teamfight gate 591/19,155 = 3.09 % of proximity encounters, 20,000 matches); the marker's run description (2,000 matches, two flags) is wrong |
| docs/references/definition_section_draft.tex | kill-less sentence (37) | v3.3 teamfight-gate share of proximity encounters, not "of engagements" |
| docs/DATA_MANIFEST.md | "Canonical results" (19, 20, 21, 22, 24, 25) | Move the v2 headline, the −.0652 replication, the old deep-learning table, the `shap_mlex.json` ranking and the 0.008 sweep to "Superseded"; mark `killless/*.json` as v2 and add `tog_revision/killless_grid/summary.json` |
| docs/ENGAGEMENT_DEFINITION_V3.md | §0 (20), §1 (27), §6 (106), §7 (130), §9 (163) | Qualify the scale wording with `n_min ≥ 4`; replace the 4.9 % with the v3.3 proximity-encounter share and its denominator |
| docs/DEFINITION_LITERATURE.md | §5 (206), §8 (268) | Same kill-less fix as above |
| docs/ENGAGEMENT_WINNER_DEFINITION.md | "What the audit found" (52, 54, 68), "Adopted primary" (178, 179), "Open items" (190) | "Teamfight most predictable" and the v2 decomposition to-dos |
| README.md | "Retraction" heading (17); "Clean result" (33-34) | Qualify the heading with the cut; −0.0019 is the bootstrap mean, the point estimate is −0.0020 |
| docs/CLAUDE_TOG_PAPER_PLAN.md | §3.1 (98), §5 item 2 (342) | Refer to the draft sentence by content (it is now line 37) |
| docs/tog_manuscript/reviewer_response_matrix.md | R2-1 row (62) | The README no longer says "~1M" |

Withdrawn FLAGs from the first pass: README.md 13 and 148 (§2); TOG_EXTENSION_PLAN.md banner CI and the
CLAUDE_TOG_PAPER_PLAN.md §2 cut table (§14), both of which match the current cut-sensitivity artefact.
