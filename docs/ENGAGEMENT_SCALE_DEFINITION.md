# Engagement scale: pick, skirmish, teamfight

> **STALE (2026-09-11): v2 document, kept for provenance.** It was written for the v2 detector
> (kill gap 18 s, diameter 4,000 u, presence radius 1,800 u, lead 10 s) and the v2 class rule
> teamfight = `n_min ≥ 3`. Do not cite any of the following as current:
>
> - the premise in the next paragraph that the ToG headline is a decomposition by engagement
>   scale, and the "class gaps ~0.05 AUC" in the shop-event section. The scale gradient is
>   retracted: it came from the `time_norm` feature leak (`docs/DEFINITION_EVIDENCE.md`
>   section 22);
> - the class shares (21.1 / 36.9 / 42.0 %) and counts in the class-rule table, and the rule that
>   results are reported at `n_min ≥ 3` with `n_min ≥ 4` as the robustness check. The v3.3
>   corpus uses new detector constants and teamfight = `n_min ≥ 4` (`scale.teamfight_min` = 4
>   in `D:/LOL_Project/fusion_2615/corpus_shards_v33/manifest.json`);
> - the commitment gap "negative for 52% of engagements, mean −0.89" (`docs/CLAUDE_TOG_PAPER_PLAN.md`
>   §4 lists it as never-write; see the note in that section);
> - "AUC moves 0.008" from the threshold sweep (553 matches, v2 detector, one factor at a time;
>   `D:/LOL_Project/fusion_2615/features/thresholds/*.json`);
> - the kill-less table and its 4.9 % (v2 presence gate), the pairing "0.15 per match vs 2.03
>   teamfight-class engagements", the "brief passes" reading, and "the share is insensitive to
>   the radius". The v3.3 kill-less grid replaces them (see the note in that section).
>
> Current definition: `docs/ENGAGEMENT_DEFINITION_V3.md` and `docs/tog_manuscript/sec_definition.tex`.
> List of stale claims across `docs/`: `docs/tog_manuscript/stale_claims_inventory.md`.

The ToG extension's headline is a decomposition by engagement scale, so the
class definition has to be exact, justified, and honest about what it can and
cannot be used for. This is the reference; the paper's definitions section
follows it.

Measurements below come from a seeded 300-match sample of the paper corpus
(patches 15.14–16), 1,366 detected engagements.

## What an engagement is (recap)

A kill cluster: CHAMPION_KILL events grouped when consecutive kills fall
within `TF2_KILL_CLUSTER_GAP_MS` (18 s), then split spatially when the cluster
spans more than `CLUSTER_MAX_DIAMETER` (4,000 units). The **anchor** is the
position of the cluster's first kill. The **prediction cutoff** is
`engage_ts = first_kill_ts − TF2_ENGAGE_PRE_KILL_MS` (10 s). A candidate
survives only if at least `TF2_MIN_PER_TEAM` (2) *alive* champions per team
stand within `TF2_VALIDITY_RADIUS` (1,800 units) of the anchor at the cutoff.

Both detection thresholds were swept (12/18/24/30 s; 3,000/4,000/5,000 units):
engagement counts move ±12% while AUC moves 0.008 and the label's positive
rate stays at 50.5–50.9%, so the operating point sits on a plateau.

## Two count definitions

### A. Observed participation — `det_cluster_blue` / `det_cluster_red`

Champions per team that appear in the resolved fight:

- every killer, victim, and assister of the cluster's kills, **plus**
- actors of non-kill events inside `[engage_ts, last_kill_ts + TF2_TAIL_BUFFER_MS]`
  (tail buffer defaults to 0, so the window opens at the *prediction cutoff*
  and closes at the last kill) and within `TF2_INTERACTION_RADIUS` (3,000
  units) of the anchor. `CHAMPION_KILL` is handled above;
  `ELITE_MONSTER_KILL`, `BUILDING_KILL` and `TURRET_PLATE_DESTROYED` are
  excluded so the post-fight outcome window does not double-count them.

Which event types can actually contribute (measured over 25 matches): only
`CHAMPION_KILL` and the objective/building types carry positions, so every
remaining contributor is located by interpolating the actor's own 5 s
position. In descending frequency those are `ITEM_PURCHASED`, `WARD_PLACED`,
`ITEM_DESTROYED`, `SKILL_LEVEL_UP`, `LEVEL_UP`, `WARD_KILL`, `ITEM_SOLD`,
`ITEM_UNDO`, `CHAMPION_SPECIAL_KILL`.

`LEVEL_UP`, `SKILL_LEVEL_UP`, `ITEM_DESTROYED` (consumables) and the ward
events are genuine combat signals. The shop events (`ITEM_PURCHASED`,
`ITEM_SOLD`, `ITEM_UNDO`) are not: they fire at the fountain and are normally
excluded by the radius test, but a fight inside the base can count a shopping
player as a participant.

**Measured** (`run_shop_event_sensitivity.py`, 3,000 matches detected twice,
14,209 engagements in both runs; switch: `TF2_EXCLUDE_SHOP_INTERACTIONS`):
excluding shop events removes a participant from **1.7% of team-sides**
(mean 0.017 per side, never more than 2) and moves **1.59% of engagements**
one class down — 132 skirmish→pick, 92 teamfight→skirmish, 2 teamfight→pick.
Class shares shift from 21.1/37.0/42.0 to 22.0/36.7/41.3
(pick/skirmish/teamfight). Every transition is downward, as expected for a
rule that only ever removes participants, and the engagement set itself is
untouched (14,211 vs 14,210 detected).

The default stays `False` so the published detector is reproduced, and the
1.6% figure is reported as the definition's contamination bound. The
inflation is real but an order of magnitude too small to affect the scale
decomposition, whose class gaps are ~0.05 AUC with intervals of ±0.002.

**Known only after the fight resolves.** It also inherits Match-V5's event
sparsity: the timeline carries no damage events, so a champion who fought but
neither scored, died, assisted, nor triggered a positioned event is invisible.
Read it as *"champions appearing in the fight's event record"*, not as
*"champions who fought"*.

### B. Presence at the cutoff — `det_present_blue` / `det_present_red`

Alive champions per team within 1,800 units of the anchor **at the prediction
cutoff**, read off the 5 s dense position grid. Available to a predictor: it
uses positions at t ≤ cutoff. The anchor location still comes from the
conditioning kill, exactly like the corpus's existence does — this is
kill-conditioned retrospective analysis, and the paper says so.

## Class rule

With `n_min = min(blue, red)`:

| class | rule | share, full corpus (993,484 engagements) |
|---|---|---|
| teamfight | `n_min ≥ 3` | 42.0% (416,956) |
| skirmish | `n_min = 2` | 36.9% (366,948) |
| pick | `n_min ≤ 1` | 21.1% (209,527) |

Presence classes over the same corpus: teamfight 184,981, skirmish 808,503,
pick 0. Mean participation asymmetry by class: pick 1.67, skirmish 0.90,
teamfight 0.56.

The cutoff is reported rather than tuned. The joint distribution of
(smaller, larger) participant counts, in % of the corpus:

| min \ max | 1 | 2 | 3 | 4 | 5 | row |
|---|---|---|---|---|---|---|
| **1** | 0.1 | 9.6 | 8.1 | 2.4 | 1.2 | 21.5 |
| **2** | – | 14.5 | 16.1 | 6.2 | 1.0 | 37.8 |
| **3** | – | – | 8.1 | 9.7 | 2.8 | 20.6 |
| **4** | – | – | – | 6.2 | 8.0 | 14.2 |
| **5** | – | – | – | – | 5.9 | 5.9 |

Alternative teamfight cutoffs, for the sensitivity table: `n_min ≥ 3` gives
41/38/22 (teamfight/skirmish/pick); `n_min ≥ 4` gives 20/21/59; `n_min ≥ 5`
gives 6/14/80. Results are reported at `n_min ≥ 3` with `n_min ≥ 4` as the
robustness check — a five-per-side fight is only 5.9% of engagements, so
reserving "teamfight" for it would leave the class too small to analyse.

### Asymmetry is reported separately

`min()` deliberately collapses a 5v1 collapse and a 1v1 duel into the same
class, so `|blue − red|` is carried as its own covariate. It is 1.77 on
average within picks against 0.57 within teamfights, and 22% of all
engagements differ by two or more participants. Asymmetry is what makes a
pick a pick; the class label alone does not capture it.

## How the two definitions relate

Cross-tab of the 1,366 engagements (rows = presence class, columns =
participation class):

| presence \ participation | pick | skirmish | teamfight | total |
|---|---|---|---|---|
| skirmish (2 per side present) | 265 | 493 | 363 | 1,121 |
| teamfight (3+ per side present) | 29 | 23 | 193 | 245 |

Presence never yields a pick: detection already requires two alive per team,
so **a pick is not an engagement where one player was alone — it is one where
bodies were present and only one side traded.** That reframing matters for the
result that vision features predict picks best: what vision reads is health
and position of players who were there but did not commit.

The commitment gap (present − participating) is **negative for 52% of
engagements, mean −0.89**: more champions end up in the fight's event record
than stood inside the radius ten seconds earlier. Fight size is largely
settled *after* the cutoff. This is a finding, not a defect — it is direct
evidence for why pre-onset prediction saturates where it does, and it answers
the meta-reviewer's question about what "counts" as a teamfight.

> **STALE (2026-09-14): do not cite the 52 % or the −0.89.** They come from the 1,366-engagement
> v2 sample above, and the paragraph does not say whether the gap is taken on the smaller side or per
> team. `docs/CLAUDE_TOG_PAPER_PLAN.md` §4 lists "cutoff 이후 참여자 52 %" as a sentence that must
> not be written. On the full v2 corpus, `docs/DEFINITION_EVIDENCE.md` §4 reports the smaller-side gap
> as negative for 35.9 % (mean −0.28) and the per-team gap as negative for 43.8 % (mean −0.48). That
> section is a dated log, and these values were not re-checked in this pass. Under v3.3 the smaller-side
> gap (presence `n_min` − participation `n_min`) is negative for 194,468 of 532,547 labelled engagements,
> 36.5 % (194,468/532,547). It is zero for 41.7 % and positive for 21.8 %, with mean −0.34. Source:
> `D:/LOL_Project/fusion_2615/features/tog_revision/A6-definition-sensitivity/scale_participation_v33.json`,
> `populations.labelled.crosstab_presence_nmin_rows_participation_nmin_cols`. Each share is the sum of
> the cells with presence below, equal to or above participation, divided by 532,547. The 42 rows with a
> participation count of −1 are read as zero. A negative gap is common (36.5 %) but is not a majority,
> so "largely settled after the cutoff" overstates it.

## Usage rules (binding)

1. **Participation scale is the primary reporting axis**, always labelled as a
   post-hoc subgroup: it is unknown at the prediction cutoff and must never be
   presented as an operational stratifier or fed to a model.
2. **Presence scale is a pre-fight covariate.** It may enter models and
   operational framing. It is a weak proxy for eventual scale (see cross-tab),
   and that weakness is reportable.
3. Neither is a feature in the released baselines; both are metadata.
4. Any AUC broken out by class carries its bootstrap interval over match
   clusters, and the accompanying `|blue − red|` asymmetry.

## Kill-less encounters: what the corpus omits

The corpus is kill-anchored, so engagements resolved without a kill are absent
entirely. `run_killless_encounters.py` bounds the omission by scanning the 5 s
position grid for sustained localized clusters under the detector's own
validity condition minus the kill requirement — each alive champion is tried
as an anchor, and a frame counts when some anchor has `min_per_team` alive
champions of *both* sides within the radius. 2,000 matches per setting:

| radius | duration | per team | encounters/match | kill-less share |
|---|---|---|---|---|
| 1,800 | 10 s | 2 | 13.28 | **12.2%** |
| 1,200 | 10 s | 2 | 10.34 | 12.0% |
| 2,500 | 10 s | 2 | 14.03 | 12.7% |
| 1,800 | 5 s | 2 | 15.87 | 16.0% |
| 1,800 | 20 s | 2 | 9.71 | 8.5% |
| 1,800 | 10 s | **3** | 5.26 | **7.7%** |
| 1,800 | 20 s | **3** | 3.10 | **4.9%** |

> **STALE (2026-09-11): v2 constants, and two statements below are unsupported.** Every row in
> the table above ran with R = 1,800 u and a 10 s grace (`radius` 1800, `grace_ms` 10000 in
> `D:/LOL_Project/fusion_2615/features/killless/*.json`), i.e. the v2 presence gate, not the
> v3.3 rule anchors (R = 1,600 u, B = 15 s). Two statements in the next paragraph are struck
> through and must not be cited:
>
> 1. *"0.15 per match, against 2.03 teamfight-class engagements per match"* pairs unlike
>    quantities. 0.15 is 305 kill-less proximity encounters (at least 3 alive per side within
>    R of a common anchor for at least 20 s) over 2,000 matches (`r1800_d20_t3.json`;
>    305/2,000 = 0.1525). 2.03 counts post-hoc participation-class engagements, not proximity
>    encounters: 416,956 teamfight-class engagements (v2 rule `n_min ≥ 3`) over 205,884
>    matches, 416,956/205,884 = 2.025 (`D:/LOL_Project/fusion_2615/features/scale_decomposition.json`,
>    `by_participation_scale.teamfight.n` and `n_matches`). That file is the v2 detector run of
>    993,484 engagements behind the class-rule table above, and `docs/DATA_MANIFEST.md` lists it
>    under "Superseded". The 4.9 % is likewise a share of 6,195 proximity encounters
>    (305/6,195 = 4.92 %), not of engagements.
> 2. *"brief passes rather than resolved fights"* is an interpretation, not a measurement. The
>    16.0 % setting (2 per side, at least 5 s) and the 4.9 % setting (3 per side, at least 20 s)
>    differ in both duration and party size, and nothing in these runs classifies what the
>    kill-less encounters were.
>
> **v3.3 re-measurement (finished 2026-09-11).** Source:
> `D:/LOL_Project/fusion_2615/features/tog_revision/killless_grid/summary.json`, written
> 2026-09-11 11:25:37 from code at commit 5600f8b; the results were recorded in commit dc243cf.
> The grid has nine settings, 20,000 matches each, seed 7. Every command in the file passes
> explicit `--radius`, `--min-per-team`, `--min-duration` and `--grace-ms` flags, because the
> script's argument defaults are still the v2 values. Two details of `scripts/run_killless_encounters.py`
> matter when reading the table:
>
> - An encounter has a kill when a CHAMPION_KILL falls between its first and last active frame,
>   widened by the grace on both sides.
> - The minimum duration is applied as a whole number of 5 s grid frames: 13.7 s rounds to
>   3 frames and 20 s to 4.
>
> | row in `rows[]` | R | alive per side | min. duration | grace | encounters | kill-less | share of encounters | kill-less per match |
> |---|---|---|---|---|---|---|---|---|
> | `r1600_t4_dG_g15` (teamfight gate) | 1,600 u | ≥ 4 | 13.7 s | 15 s | 19,155 | 591 | **3.09 %** | 0.02955 |
> | `r1600_t4_d20_g15` | 1,600 u | ≥ 4 | 20 s | 15 s | 12,602 | 351 | 2.79 % | 0.01755 |
> | `r1200_t4_dG_g15` | 1,200 u | ≥ 4 | 13.7 s | 15 s | 7,049 | 141 | 2.00 % | 0.00705 |
> | `r2000_t4_dG_g15` | 2,000 u | ≥ 4 | 13.7 s | 15 s | 33,592 | 1,166 | 3.47 % | 0.0583 |
> | `r1600_t4_dG_g10` | 1,600 u | ≥ 4 | 13.7 s | 10 s | 19,155 | 832 | 4.34 % | 0.0416 |
>
> Share = kill-less / encounters (for example 591/19,155 = 3.09 %). Per match = kill-less / 20,000
> (591/20,000 = 0.02955). The file's fields are `killless_share_of_encounters` and
> `killless_per_match`. The remaining four settings, with 2 or 3 alive per side, are in the same
> file.
>
> **Denominator.** These are shares of **proximity encounters**, not of engagements. The file's
> `denominator_warning` states that proximity is not commitment, so a per-match kill-less rate
> must not be divided by engagement counts. The file's `corpus_reference` gives 0.572
> teamfight-class engagements per match (109,829/191,940, from
> `features/scale_decomposition_v33_market_event.json`) as a scale reference only, not as a
> divisor.

Two things follow. **The omission is modest and shrinks exactly where the
question was aimed:** the meta-reviewer asked about *teamfights* without
kills, and at teamfight scale (3+ per side) sustained for 20 s only 4.9% of
encounters end without a kill **[v2 constants; under v3.3 constants the same 3 per side for 20 s gives 3.27 % (1,607/49,111, row `r1600_t3_d20_g15`) and the teamfight gate 3.09 %; see note]** — ~~0.15 per match, against 2.03 teamfight-class
engagements per match that the corpus does capture~~ **[STALE, see note]**. ~~Loosening the duration to
5 s triples the kill-less share (16.0%), confirming that most kill-less
"encounters" are brief passes rather than resolved fights.~~ **[STALE, see note]**

**The share is insensitive to the radius** (12.0–12.7% across 1,200–2,500
units) and sensitive to duration and party size, which is the expected
signature: a proximity threshold decides *how many* encounters exist, while
duration and size decide *which* of them are fights.
**[STALE 2026-09-14: v2 rows with 2 per side and a 10 s grace. At the v3.3 teamfight gate the
kill-less share moves from 2.00 % at 1,200 u to 3.09 % at 1,600 u and 3.47 % at 2,000 u (rows
`r1200_t4_dG_g15`, `r1600_t4_dG_g15` and `r2000_t4_dG_g15` of `killless_grid/summary.json`), so the
share is not insensitive to the radius there. The v3.3 grid has no radius variation at 2 per side.]**

This is an upper bound on what kill-anchored detection misses, not a corpus
extension: proximity is not commitment, and without a kill there is no
outcome to label, so these encounters remain outside the prediction task.

- Presence at radii other than 1,800 units (the detector's gate value) has not
  been measured; a 3,000-unit presence count would likely track participation
  better and is worth one sweep.
  **[STALE 2026-09-14: 1,800 u was the v2 gate; the v3.3 gate is R = 1,600 u
  (`TF2_VALIDITY_RADIUS` 1600.0 in `corpus_shards_v33/manifest.json`).]**
- Kill-less encounters remain outside the corpus entirely; quantifying their
  prevalence from the 5 s position grid is a separate planned experiment.
  **[STALE 2026-09-11: measured under v3.3 constants; see the v3.3 note in the kill-less section.]**
