# Engagement scale: pick, skirmish, teamfight

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
player as a participant. **Open sensitivity check:** recompute the classes
with shop events excluded from `extra_pids` and report how many engagements
change class.

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

## Open items

- Presence at radii other than 1,800 units (the detector's gate value) has not
  been measured; a 3,000-unit presence count would likely track participation
  better and is worth one sweep.
- Kill-less encounters remain outside the corpus entirely; quantifying their
  prevalence from the 5 s position grid is a separate planned experiment.
