# Who won the engagement? — the label, audited

Companion to `ENGAGEMENT_SCALE_DEFINITION.md`. The scale classes say what
kind of engagement happened; this document says what "winning" one means,
what every candidate definition actually computes, and which one the ToG
extension should report as primary. Measurements: 25,995 fights from the
5,000-match dump run (class semantics), and the seeded 553-match pilot
(2,551 engagements; label behaviour and AUC, `label_ablation_pilot.*`).

## The label space has three axes

Every scheme in `gameplay/labels.py` is a point in the same design space:

| axis | options in code |
|---|---|
| **currency** | kills only · kills + survivors · gold-weighted composite · attention-weighted event value |
| **time scope** | cluster kills only `[first_kill, last_kill]` · full label window `[engage_ts, horizon_end]` (cutoff + ≥30 s, extended to fight end) |
| **tie policy** | `drop` · seeded coin (`random`, the current default) · fixed side |

### What each scheme computes

- **`micro_win`** — kill count difference, cluster kills only. Ties when
  kills are equal.
- **`kill_survival`** — `1.0·kill_diff + 0.3·alive_diff` with survivors read
  at the last kill. Survivor margin breaks most kill ties.
- **`weighted`** — gold/objective composite over the *full window*, so it
  includes the immediate aftermath (conversion), not just the exchange.
  Continuous → effectively never ties.
- **`attention_value_win` (Eq.3, the CoG label)** — softmax-attention over
  event values in the *full window*; bounty/shutdown terms upweight
  underdog reversals by design. Continuous → ties ≈ 0.3%.

Two schemes score only the exchange; two also score its conversion. That
scope difference, not just the weights, separates them.

## What the audit found

**1. Ties are decided by a seeded coin, and the coin is class-asymmetric.**
`LABEL_TIE_STRATEGY = "random"` (a P1 decision to avoid side bias) means
tied engagements get a deterministic pseudo-random label:

| scheme | coined labels | by class (pick / skirmish / teamfight) | AUC all → decided only |
|---|---|---|---|
| Eq.3 | 0.3% | 0.0 / 0.5 / 0.2 | 0.5946 → 0.5943 |
| micro_win | **14.6%** | 2.8 / **17.4** / **18.4** | 0.6520 → **0.6808** |
| kill_survival | 9.8% | 2.3 / 11.9 / 11.8 | 0.6618 → 0.6786 |
| weighted | 0.0% | 0 / 0 / 0 | 0.6746 → 0.6746 |

Kill-currency labels flip a coin on one in six or seven skirmishes and
teamfights (1-for-1 and 2-for-2 trades) and almost never on picks. So under
kill labels, per-class AUC is mechanically deflated *differently per class*
— the label policy is entangled with the scale decomposition. Dropping ties
instead would distort class *composition* the same asymmetric way. The
published full-corpus decomposition used Eq.3 (0.3% coined), so it is clean
of this effect.

**2. Draws are real, and most of them are breakable.** Of kill-tied fights
in the dumps: 38–43% are broken by survivor margin, roughly another third
by a gold swing beyond ±200 g, leaving 27–41% of draws (≈ 5% of all
engagements) with equal kills, equal survivors, and no meaningful gold
movement — genuine draws by any material criterion.

**3. Class semantics validate** (25,995 fights): picks are median-1-kill
(74.8% single kill, zero duration), skirmishes median 2 kills and 3 s,
teamfights median 4 kills, 12.4 s, and the largest gold swings (median
509 g). The classes mean what their names say.

**4. Which claims are label-robust.** Teamfight is the most predictable
class under *every* scheme (0.623 / 0.699 / 0.710 / 0.710 on the pilot).
The pick-vs-skirmish ordering is **not** robust — kill-currency labels rank
skirmish above pick, value/gold labels rank pick at or above skirmish — and
must not be a headline claim.

**5. Eq.3's difficulty is not tie noise.** It has the fewest ties and the
lowest AUC; the hardness comes from the attention weighting itself, which
deliberately upweights upsets (the attention↔material axis established by
the weight-perturbation sweep).

## Structures are not peripheral — measured

A first draft of this document proposed a kills→survivors→gold ordering,
implicitly demoting objectives and turrets to their gold value. The dumps
say that demotion is not free:

- **Objective/tower events occur in 25.3% of fight windows** (37.0% with
  plates) — these phenomena are in a quarter of engagements, not a tail.
- **In 7.54% of all fights the kill winner and the structure winner are
  opposite sides** (picks 6.99%, skirmishes 6.45%, teamfights 8.74%). Any
  *ordering* of kills vs structures decides those 1,959 labels by fiat — a
  lexicographic rule is not assumption-free, it is an extreme weighting.
- In the tiebreak chain kills → survivors → structures(+plates) → gold ±200:
  15.7% are kill-tied, 9.4% still tied after survivors, structures break
  32.2% of those, and **4.0% remain genuine draws** (pick 3.2 / skirmish
  5.5 / teamfight 3.2).

## Recommendation for the ToG extension

A weighted composite was considered and rejected: its coefficients are
exactly the arbitrariness the review objected to, and perturbation sweeps
defend a choice rather than remove it. The resolution is to stop choosing.

**Primary label — market-priced material outcome (`weighted` with the
researcher coefficients removed: `W_KILL=0, W_OBJ=0, W_GOLD=1`, i.e. the
sign of the window team-gold-swing difference).**

The game's economy already prices every phenomenon this label must weigh:
kills pay 300 g plus bounties, turrets pay local and global gold, plates
pay 175 g, and monsters pay kill gold — all of it accrues into the
timeline's team gold. The paper's sentence is: *we set no weights; the
relative value of kills, turrets and objectives is the game's own
exchange rate, and the label reads it.*

Measured on the pilot (2,551 engagements):

- **Ties: 0.** No coin, no drops, no class-asymmetric distortion.
- **OOF AUC 0.6915** — above every researcher-weighted scheme (Eq.3
  0.5946, micro 0.6520, kill_survival 0.6618, weighted 0.6746).
- Agreement 90.6% with the weighted composite: the hand coefficients were
  mostly redundant with gold — removing them changes ~9% of labels and
  *raises* AUC.
- Per class: pick 0.7105 / skirmish 0.6591 / teamfight 0.7073.

**Stated limitation:** gold under-prices pure-buff value (Baron/Elder
buffs, dragon souls, tempo). The structure-explicit robustness columns
cover that gap, and the limitation section says so.

## Adopted primary: `market_lex` (market verdict + dead-zone refinement)

Implemented as `LABEL_TYPE="market_lex"` (commit 99c29fb): the gold swing
decides when it exceeds `LABEL_GOLD_DEADZONE` (300 g = one base kill
bounty); inside the dead zone — where any gold sign would be noise —
discrete facts refine in a fixed order (cluster kills → survivors →
structures), and an engagement even on all of them is a genuine draw. The
ordering only ever adjudicates materially-even fights, so the 7.5%
kill-vs-structure conflict set is priced by gold before the ordering is
consulted. Pilot (2,551): at ε=300 the market decides 54.9%, facts refine
39.5%, genuine draws 5.61%; ε∈{150,600} shifts the split (76/20/3.7 and
27/66/6.6) — sensitivity column for the paper. Agreement: weighted 94.3%,
micro 93.1%, Eq.3 89.9%. **OOF AUC 0.6879** (pick 0.686 / skirmish 0.645 /
teamfight 0.720), statistically indistinguishable from pure gold (0.6915)
with the noise-sign weakness removed.

**Robustness labels, reported alongside:**

- **Pure gold swing** (ε=0): the zero-constant end of the family.
- **Eq.3** for CoG continuity — the *attention-value* outcome whose
  unpredictability is a finding, not the material target.
- **micro_win under tie-drop** as the simplest-possible column.

**Class-ordering honesty under the label family:** teamfight is at or near
the top under every label; skirmish is at or near the bottom under every
label; **pick is the label-sensitive class** (0.576 → 0.711 across the five
schemes — its kill outcome is noisy, its gold consequence is predictable).
Scale claims in the paper are stated at that robust level.

**Secondary labels, reported alongside:**

- **Eq.3** for CoG continuity — reframed as what it is: an
  *attention-value* outcome whose unpredictability is a finding (the
  "excitement component is the unpredictable component"), not the primary
  material target.
- **micro_win under tie-drop** as the simplest-possible robustness column.

**Binding usage rules:**

1. The seeded coin never decides a primary-label row. It may appear only in
   a robustness appendix.
2. Any per-class AUC under a kill-currency label carries its class-wise
   coin/drop rate next to it.
3. Scale claims are stated at the label-robust level: teamfights are the
   most predictable class; pick-vs-skirmish ordering is label-dependent.
4. Headline numbers (full-corpus decomposition, deep-baseline table) are
   re-scored under the primary label once it lands; Eq.3 columns stay for
   comparison.

## Open items

- Perturbation sweep on the primary's coefficients (`W_KILL`, `W_GOLD`,
  `W_OBJ` at ×0.5/×2) on the pilot, mirroring the Eq.3 sweep.
- Implement `lexicographic` in `gameplay/labels.py` as the robustness
  column (compose existing pieces; the ±200 g constant gets a {100, 200,
  400} sensitivity line).
- Re-score the full-corpus decomposition and the deep-baseline table with
  `weighted` as primary; keep the Eq.3 columns for comparison.
- Note for the write-up: dump-based structure counts use the post-fight
  outcome window, which approximates but does not equal the label window;
  recompute exactly if a reviewer asks.
