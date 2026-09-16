# League of Legends engagement prediction — research specification

**Methods, completed results and remaining journal work. Research status as of 16 September 2026.**

This document updates the specification of 15 September 2026
([COLLABORATOR_RESEARCH_SPECIFICATION_20260915.md](../COLLABORATOR_RESEARCH_SPECIFICATION_20260915.md)), which
supplied its structure. The earlier version is retained as a dated record and should be read as *structure and
source*, not as current truth: several of its "planned" items are now executed (§16), and two of its wordings are
corrected here (§6.3 on participant slots, §16.2 on balanced states). Numerical results of the earlier version are
not altered.

Prepared for academic collaboration. It contains no raw match records, player account identifiers, credentials or
machine-specific private data. Performance figures are quoted from retained reports; exact predictions, manifests
and validation records remain local. Every number resolves through
[claim_evidence_ledger.md](claim_evidence_ledger.md).

---

## 1. Purpose and current position

We study whether the global game state observed **before** an engagement predicts which team will improve its
estimated probability of winning the match during a defined engagement and follow-up interval. The system
connects engagement construction, strictly pre-engagement state extraction, a separately trained match-win value
model, outcome-label generation, engagement prediction, and model explanation.

The pipeline has been run on a designated corpus of 210,000 matches with patch-based splits, plus external patch
and region evaluations. Since the previous version, six further studies have completed: a reinforced baseline
comparison, a matched-input learner comparison including neural families, an endpoint-cap sensitivity study, a
training-patch-only definition sensitivity study, a champion-class representation study, and two analyses of the
label mechanism and of the predictor's attribution in balanced states.

The main-test teamfight predictor reaches AUC ≈0.669 (LightGBM on 352 matched inputs) for the **direction** of
model-estimated win-probability change. The separate match-win value model reaches AUC 0.858621 for **eventual
match victory**. These evaluate different targets and must never be compared as two estimates of the same
quantity.

Our evidence supports predictive modelling of an explicitly defined strategic outcome. It does not establish that
the generated label is an independent judgement of fight victory, that the engagement caused the measured change,
or that anything has been deployed live. The complete CoG model lineup has **not** been retrained on the new
target.

---

## 2. Research lineage and what changed since CoG

The authoritative **CoG 308** paper (accepted and published) reported a binary positive-exchange outcome derived
from a manually constructed event score. Three label lineages must be kept separate and are never merged in this
document: the published CoG 308 exchange score of §2.1; the later intermediate `market_event` experiments in the
earlier codebase, whose implementation is not re-inspected here and which we therefore do **not** equate with the
published score; and the current `ΔV` target. Labels, counts and performance figures are preserved per lineage.
The earlier **paper 118** (rejected) and an extended technical report contain additional analyses and diagnostic
models. They are separate versions; 118-specific diagnostics are
identified as such. The extended report's list of more than 25 supported architectures is not a list of 25
published comparison results.

### 2.1 The original exchange label (reference only)

Each post-onset event received a sign for the benefiting team, an event value and an importance prior. A softmax
with `β = 2` converted the priors into within-window weights, and the weighted signed sum was thresholded at zero.

`Y_CoG = 1[ Σ_u α_u σ_u v_u > 0 ]`, `α_u = exp(2 p_u) / Σ_v exp(2 p_v)`

| Descriptor in CoG Table I | Event value coefficient | Importance prior |
|---|---:|---:|
| Kill indicator | 1.00 | 0.25 |
| Normalised shutdown | 1.60 | 0.30 |
| Normalised streak | 0.35 | 0.15 |
| Normalised assists | 0.20 | 0.10 |
| Normalised bounty | 0.30 | 0.20 |
| Objective tier | 1.10 | 0.35 |
| Lane priority | 0.25 | 0.15 |
| Special event bonus | `s(u)` | `s(u)` |

The original implementation clips descriptors to [0, 1], log-scales shutdown and bounty, divides streak by 10 and
assists by 4, and uses objective tiers including plate 0.35, dragon 0.75, inhibitor/Atakhan 0.85 and
Baron/Elder/soul/Nexus 1.0; special bonuses include first blood 0.20 and multi-kill 0.25, with ace handled
separately, and near-zero ties below 1e-8 resolved by a deterministic coin flip.

**These coefficients are documentation of the old design. They are not used in the current label**, and the
concern raised in the collaboration e-mail was answered by changing the valuation formulation rather than by
tuning them (see §16.4 and [reviewer_response_matrix.md](reviewer_response_matrix.md) §6.2).

### 2.2 What changed and what was retained

| Component | CoG 308 | Current Delta V study |
|---|---|---|
| Outcome valuation | Manually weighted signed event exchange | Change in a learned state-based match-win estimate |
| Prediction target | Sign of exchange score | Sign of `ΔV`; the continuous change is retained for analysis |
| Temporal kill gap | Fixed 18 s | Estimated ≈13.7 s |
| Spatial separation | Fixed 4,000 u | Estimated ≈4,264 u |
| Onset offset | First kill − 10 s | First kill − 15 s |
| Presence radius | 1,800 u in the published definition | 1,600 u operationally |
| Outcome endpoint | 30–60 s onset-based rule with ace truncation | Event-based stopping with a 90 s cap after the last kill |
| Predictive information | Multiple pre-onset representations | Primary `q` uses 352 numeric pre-state features |
| Population and split | 206,442 retained matches under the original filtering | All 210,000 designated matches keep patch roles; task-specific eligibility reported |

These changes alter the cohort, the target and the representation together, so differences between old and new
AUC values are **not** a measured improvement from any single intervention.

### 2.3 Published CoG performance (historical record)

| CoG 308 model and input view | Published TEST AUC |
|---|---:|
| LightGBM engineered tabular | 0.675 |
| MLP matched tabular input | 0.626 |
| Bi-GRU macro sequence | 0.581 |
| Layered Fusion | 0.581 |
| Transformer macro sequence | 0.576 |
| Cross-Attn event and player | 0.571 |
| ST-GNN temporal graph | 0.569 |
| GraphSAGE player graph | 0.569 |

These are the original three-seed figures on patch 15.16 under the old exchange label. The released-code
correction record reports approximately 0.669 for LightGBM after localisation corrections; published and
corrected-release results are distinct records. The old MLP comparison used the same ~2,980-dimensional
engineered representation as LightGBM, and a mistakenly routed macro-sequence diagnostic was corrected in the
released code. Per-model subsampling in the original recorded runs means that execution budget is not evidence
that a matched full-corpus comparison has already been performed. **None of these numbers is comparable with the
Delta V results in §11.**

---

## 3. Quantities and estimand

| Symbol | Definition | Role |
|---|---|---|
| `W` | Actual final Blue match victory | Supervision and evaluation for `V` |
| `S(t)` | State from information observed no later than `t` | Value-model input |
| `V` | Estimated probability of eventual Blue match victory | Valuation function |
| `E` | Retrospectively identified eligible kill-conditioned engagements | Study population |
| `T` | Engagements with recorded cluster participation of at least four per side | Primary teamfight subgroup |
| `N` | Known-scale engagements in `E` outside `T` | Other-engagement subgroup (disjoint from `T`) |
| `ΔV` | Value at the defined endpoint minus value before onset | Continuous operational outcome |
| `Y` | Indicator that `ΔV > 0` | Current `q` training label |
| `q` | Probability that `Y = 1` given pre-engagement information | Engagement prediction |

`p_t = V(S(t)) ≈ P(W = 1 | S(t))`, `q(X_pre) ≈ P(Y = 1 | X_pre)`.

The target is improvement **direction**, not magnitude. A `q` output of 0.70 means an estimated 70% probability of
a Blue-improvement label under this definition: not a 70% final win probability, not a 70-point gain, not a
causal effect, and not a prediction that a fight will occur. Because the population is located retrospectively,
the estimand is conditional on an engagement having been identified; occurrence detection and live routing are
separate, unbuilt tasks.

---

## 4. Data coverage and validation separation

### 4.1 Main corpus

| Patch role | Raw matches | Detected engagements | Valid ΔV rows | T rows | N rows |
|---|---:|---:|---:|---:|---:|
| TRAIN 15.14 | 74,673 | 199,480 | 199,358 | 39,605 | 159,753 |
| VALIDATION 15.15 | 74,748 | 203,292 | 203,170 | 41,315 | 161,855 |
| TEST 15.16 | 60,579 | 163,680 | 163,576 | 32,981 | 130,595 |
| Total | 210,000 | 566,452 | 566,104 | 113,901 | 452,203 |

The same valid-row masks serve the 60, 90 and 120 s caps. The 348 excluded rows have a next-engagement onset at or
before the current last kill. Stored match identifiers do not intersect across the three roles. Using the full
corpus means retaining its designated roles; it does not mean fitting on all 210,000 matches.

**Value-model eligibility is reported separately.** 208,603 main matches are eligible (1,396 excluded for no
eligible query time, 1 for missing or conflicting final-winner information). The final value model is fitted on
424,160 state queries from 74,168 eligible TRAIN matches, including matches without qualifying engagements.
Queries lie on a one-minute grid from minute two, before termination and within the observed frame range, with one
eligible query per five-minute bucket chosen by hash and equal total weight per match. That is within-match
sampling for computational control, not a small-match pilot, and not an omission of the final two minutes.

### 4.2 Validation roles and cross-fitting

Patch 15.15 is partitioned by match hash into V_CAL 18,579, V_SELECT 18,890, Q_CAL 18,710 and Q_SELECT 18,569 raw
matches, with task-specific eligibility applied afterwards, so value calibration and selection use different
matches from `q` calibration and selection. TRAIN engagement labels use five-fold value models excluding the
labelled match; the same out-of-fold adapter evaluates both endpoints of a training engagement and supplies its
pre-engagement value. Validation, test and external labels use the final value model. `q` preprocessing and base
fitting use TRAIN; calibrators use Q_CAL; model and calibration choices use Q_SELECT. Within any reported cell
each represented match carries equal total weight, and primary intervals resample matches rather than treating
engagements or state queries as independent.

### 4.3 External cohorts

| External cohort | Raw matches | Valid ΔV rows | T rows | N rows |
|---|---:|---:|---:|---:|
| KR API 16.13 | 10,064 | 25,430 | 5,202 | 20,228 |
| KR API 16.14 pilot | 200 | 475 | 101 | 374 |
| KR API 16.15 | 926 | 2,219 | 507 | 1,712 |
| NA1 API 16.13 | 10,000 | 26,538 | 5,312 | 21,226 |
| Total | 21,190 | 54,662 | 11,122 | 43,540 |

No EUW1 evaluation is available. Public season labels and API patch identifiers are not interchangeable. These
sets have prior inspection or incompletely established access histories; none is an untouched confirmatory
cohort, and the 200-match KR 16.14 set is explicitly a pilot.

---

## 5. Deriving and applying the engagement definition

The boundary pipeline estimates useful temporal and spatial boundaries from event relationships. It does **not**
discover a unique natural definition of a fight: the operational population still depends on researcher-selected
gates, event coverage and observation semantics.

| Definition | Current value or rule | Evidence class and qualification |
|---|---|---|
| Temporal gap `G` | 13.7246 s, operationally 13.7 | Empirical estimate; KDE valley between activity regimes; bootstrap interval 12.1601–15.6675 s |
| Temporal stability | ≈10–18 s plateau | Stability analysis, not a guarantee for every game phase |
| Spatial boundary `D` | 4263.8688 u, operationally 4264 | Empirical estimate; champion-sharing crossing at 0.5; interval 4256.0475–4273.9677 u |
| Presence radius `R` | 1,600 u | Game-rule motivation (champion-death experience range), operationally fixed. **Not a learned constant** |
| Provisional onset `s` | First kill − 15 s | Operational offset informed by assist-time scale and implementation precedent. **Not a learned constant** |
| Presence gate | At least two per team under the configured rule | Presence and retrospective participation are different quantities |
| Teamfight subgroup | Known minimum team cluster count ≥ 4 | v3.3 scale convention, not an estimated optimum |
| `q` cutoff | `s − 1 ms` | Makes the boundary convention explicit |

The pooled estimation record uses 208,141 matches, 10,417,458 temporal gaps and 5,460,008 temporally close kill
pairs. Some steps use seeded computational samples; we do not claim that every bootstrap replicate refits every
observed gap. Spatial crossing interpolates between 750 u bins, so a narrow interval does not establish equally
fine physical resolution. Patch-specific gap estimates are ≈13.964 / 13.490 / 13.725 s and distance estimates
≈4285.26 / 4265.27 / 4241.47 u.

Participation includes recorded killers, victims, assists and related event actors; it is not an exhaustive
annotation of everyone physically fighting. `T` and `N` are disjoint and are defined by these recorded counts;
unknown-scale rows would fall outside both, and none occurred in the retained evaluation. Kill-free pressure,
zoning and objective contests outside the detector remain outside the target population.

**Split limitation and its measured size.** Pooled `G`/`D` estimation included patches 15.14, 15.15 and 15.16.
Model train/test separation is intact, but definition estimation is not independent of the test patch. §16.3
reports the completed development-patch-only sensitivity study, which bounds the consequence without removing the
history.

---

## 6. Observable state and predictive inputs

### 6.1 Schemas

StateV2 combines the latest frame no later than the query time with events observed no later than that time:
total and current gold, level, experience, health and mana ratios, survival, lane and jungle CS, plus kills,
deaths, structures, plates, objective ownership and acquisitions, elapsed-time features and selected
interactions. Some acquisition and death histories are proxies, not exact live buff-state measurements.

| Schema | Size before categorical expansion | Use |
|---|---:|---|
| StateV2 | 362 columns | State construction, including observation age |
| Primary `V` input | 361 columns | Excludes `snapshot_age`; 351 numeric plus 10 champion slots |
| Primary numeric `q` input | 352 columns | Excludes champion IDs and `snapshot_age`; includes `p_pre_V` |
| Economic `q` comparator (historical) | 149 columns | Restricted representation; not an equal-information tree comparison |

The one-hot expanded dimension of `V` must not be confused with its 361 original columns. The 352 `q` features map
to seven retained groups: prior win probability 1, time and observation age 2, economy and experience 60, combat
and survival 68, objectives 176, structures 24, health/mana/other 21. The time-group name does not reintroduce the
excluded snapshot-age feature, and the mixed unknown-objective-team diagnostic sits in health/mana/other, so that
group is not purely health. Four inhibitor columns are classified into the combat group by a name-matching order
in the grouping code rather than by a deliberate game interpretation; moving them would give combat 64 and
structures 28 with the same 352 total, would not change any `q` prediction, but would change group attributions,
so existing explanations cannot simply be relabelled.

### 6.2 Input/output distinction

No `q` input contains the realised endpoint, the endpoint reason, the engagement duration, the future participant
count, the post state, the actual `ΔV`, the final match winner, or future-interpolated coordinates. Detection and
scale classification **do** use future kills, positions and participation; prediction does not. The primary `q`
does not include position: a previous limited-sample comparison gave an AUC difference of +0.00312 with interval
[−0.00287, +0.00933], and decay or Kalman extrapolation did not beat last-observation carry-forward. Those are
pilot findings, not a claim that position is unimportant, and retrospective position interpolation for detection
is separate from prediction-input construction.

### 6.3 Participant order — correction to the previous version

Base participant order is `(team_id, participant_id)`, not a post-game lane assignment. StateV2 explicitly
replaces any incoming role-slot metadata with that order, and the V2 contract test confirms that deleting,
permuting or mutating role metadata does not change V2 values. Consequently:

- Slot indices are a deterministic identifier order. Where a slot is named `TOP/JG/MID/BOT/SUP`, that naming is a
  **frozen mapping assumption**, not a validated per-match role.
- Any paired feature built from two slots is a **slot-index pair under a positional proxy assumption**, not a
  proven lane or role matchup, and class frequencies are frequencies **by participant-order slot**.
- In the stored main corpus the cached role-slot ordering disagrees with the StateV2 participant order in 1.69%
  of matches (3,548 of 210,000). Agreement in the remaining 98.3% does not establish per-slot role correctness,
  because the cached map is itself weak supervision derived from `teamPosition`/`individualPosition` with
  participant-ID fills whose accuracy was never validated.

The minimal check that would upgrade this wording is a read-only raw-detail audit over the frozen 15.14–15.16
population joining each StateV2 slot's participant ID to raw `teamPosition`, reporting slot-wise agreement,
missing and duplicate roles, fill cases and coverage. It has not been run. Full analysis:
[position_audit.md](../../outputs/manuscript_integration_20260916/position_audit.md).

Separately, the role-representation models of the earlier cohort study use a distinct 461-column representation
with role-weighted aggregates and assignment uncertainty; those belong to an information-representation
comparison and must not be mixed into the identical-352-input learner ranking.

---

## 7. The changing match-win estimate

The primary value model is a logistic model on expanded StateV2 with `C = 0.01`, pre-declared from exploratory
work rather than selected by a new search. Numeric median imputation, standardisation and one-hot vocabularies are
fitted only on the relevant TRAIN partition. The recorded solver is liblinear, maximum 1500 iterations, seed 7,
with convergence diagnostics retained. Raw probabilities and a positive-slope sigmoid calibration were compared on
V_CAL/V_SELECT by log loss then Brier; the selected primary value model uses **raw** probabilities.

Fitted parameters remain fixed during a match: as observations change, the state is rebuilt and the same model is
evaluated again, and time features can alter its output between recorded events. This is a changing
state-conditioned estimate, not online retraining, and not a guarantee of continuously observed physical state. We
do not force the terminal value to the actual winner's 0 or 1 at an engagement endpoint.

The motivation for valuing outcomes through a changing match-win estimate follows the direction of Maymin's
analysis [R7]; our public-timeline states, simple value model and engagement-specific endpoint are a separate
implementation. The value model is a supporting component of this engagement study, not a claim to have solved
game-independent strategic valuation.

### 7.1 Objective representation

The state includes Baron, six elemental dragon types, ownership-confirmed soul, Elder Dragon, Rift Herald, Void
Grubs and Atakhan through counts, histories, elapsed times and interactions. The value model learns associations
between those features and final victory. There is **no** current main-label formula that adds a separate
invariant economic score to a separately weighted strategic score, and no causal per-objective exchange price is
extracted: the same acquisition is conditioned on match time, existing advantage and accumulated objectives.

Unassigned soul events are not credited to an arbitrary team; unknown ownership is handled diagnostically.
FEAT_UPDATE and OBJECTIVE_BOUNTY are not explicit state channels. Atakhan and FEAT_UPDATE events were not observed
in the evaluated external data, so those channels are untested under transfer, and unseen champion identities
affect approximately 18–25% of external value queries.

---

## 8. Endpoint and label construction

Let `K` be the first kill, `s = K − 15 s`, `L` the last kill, `J` the first subsequent champion kill anywhere on
the map, `S_next` the next eligible engagement's onset and `T_end` match termination; missing candidates are
infinite.

`t_pre = K − 15 s − 1 ms`, `e(h) = min{ L + h, J − 1 ms, S_next − 1 ms, T_end − 1 ms }`

`ΔV(h) = V(S(e(h))) − V(S(t_pre))`, `Y(h) = 1[ΔV(h) > 0]`, with `h = 90 s` primary and 60/120 s as sensitivity.

The same model and calibration are used at both endpoints of a row. The administrative cap `L + h` is inclusive;
competing timestamps are excluded by one millisecond, and events sharing an excluded timestamp are not artificially
ordered. Overlaps with `S_next ≤ L` are excluded. Acquiring an objective does **not** stop the window, and several
acquisitions can be reflected before the endpoint; conversely the next global kill does stop it even when it is a
distant or pursuit kill, so the rule is conservative and does not identify a natural combat boundary perfectly.
`S_next` is itself retrospectively derived, so this is not a live stopping rule. Returning to base or regrouping is
not an endpoint because no reliable observation rule was established. Exactly zero change is coded `Y = 0` (Blue
non-improvement) and tracked separately from negative changes; no exact zeros occur in the retained valid labels,
and because of this convention the zero class is not strictly identical to Red improvement.

### 8.1 Why the primary cap is 90 s

The data do not support "every objective follows a fight within 60 s": among 149,002 engagements with a first
acquisition within 120 s, 55,734 (37.4%) had it after 60 s — a conditional denominator that does not describe
every engagement. In a pilot competing-event analysis with the event-based endpoint, cumulative first-objective
incidence before an additional kill was 11.93% at 60 s, 13.21% at 90 s and 13.75% at 120 s, so 90 → 120 s adds
≈0.538 pp. Together with the sensitivity results this motivated a practical 90 s cap. Competing-risk methodology
motivates the analysis; it does not choose 90.

In the stored main TEST h90 labels, follow-up after the last kill averages 45.493 s (median 39.440 s); only
14.286% reach the cap; 129,036 of 163,576 rows stop at the next kill. Label disagreement is 1.545% for h60 vs
h90, 0.608% for h90 vs h120 and 1.909% for h60 vs h120. These endpoint sensitivities are different experiments
from changing `V`.

### 8.2 A hypothetical worked example

All values below are illustrative arithmetic, not an observed case.

| Time | Event and role |
|---|---|
| 19:44.999 | Input cutoff; `V` estimates Blue match-win probability 35% |
| 19:45 | Defined engagement onset |
| 20:00 | First kill |
| 20:20 | Last kill of this engagement |
| 20:50 | Blue acquires Baron |
| 21:15 | Next eligible engagement begins |
| 21:30 | Next engagement's first kill |
| 21:50 | Original last-kill-plus-90-second cap |

With no earlier competing event, the endpoint is 21:14.999. Baron is included; the next engagement is excluded. If
the value model gives 47% at the endpoint, `ΔV = 0.47 − 0.35 = +0.12`, i.e. +12 percentage points, and `Y = 1`.
Blue remains below a 50% final-win estimate but improved relative to the start, and the whole 12-point change
cannot be attributed to Baron alone.

| Hypothetical pre | Post | Change | Label | Interpretation |
|---:|---:|---:|---:|---|
| 35% | 47% | +12 pts | 1 | Still behind but improved |
| 65% | 59% | −6 pts | 0 | Still ahead but deteriorated |
| 49% | 50% | +1 pt | 1 | Small improvement |
| 49% | 69% | +20 pts | 1 | Larger improvement, same binary label |

The training pair for the first row is the pre-state and `Y = 1`. The later Baron event, the post probability and
`ΔV` are not predictors. A `q` output of 0.62 means predicted probability 0.62 of the positive label. If Blue
loses the match much later, this engagement's label is not retrospectively changed.

---

## 9. Training the predictor and preventing leakage

`q` is supervised binary probability estimation with weighted binary cross-entropy plus model-specific
regularisation; match weights equalise each match's total contribution within the fitting subset and are not
proportional to the realised `|ΔV|`.

`L_q = −(1/Σ w_i) Σ_i w_i { Y_i log q_i + (1 − Y_i) log(1 − q_i) }`

Consequently a +1 pp and a +20 pp change share a label and, with equal weights and predictions, contribute
equally. Magnitude regression, a neutral band or a magnitude-weighted loss would change the target or the
objective and have not been adopted.

Raw, sigmoid and isotonic `q` calibrations are fitted on Q_CAL and compared on Q_SELECT by match-weighted Brier,
then log loss, then candidate name. The `q` sigmoid is an unconstrained logistic calibration on clipped logits and
must not be confused with the positive-slope value-model calibrator.

Cross-fitting prevents direct use of a training match's own final outcome in the `V` that labels it. It does not
make the label independent of `V`, remove retrospective case selection, or eliminate every statistical
dependence. Because `p_pre` is part of the label construction, a baseline that already knows the pre-engagement
estimate is essential; §11.1 therefore reports increments against a reinforced version of it rather than against
`p_pre` alone.

---

## 10. Completed value-model results

Main TEST evaluation: 347,234 state queries from 60,190 eligible matches, equal total match weighting.

| Metric against actual final match victory `W` | Primary `V` |
|---|---:|
| AUC | 0.858621 |
| Brier | 0.153045 |
| Log loss | 0.456742 |
| Calibration slope | ≈0.976 |

AUC in the 2–10 minute window is ≈0.6741, so the overall figure is not equally strong at every time point. Across
external sets, value AUC is ≈0.84–0.86 with calibration slopes ≈0.81–0.87, indicating probability-calibration
degradation under transfer. A nine-timepoint SimpleRNN value comparator has main-test log loss ≈0.4745; that
limited comparison concerns `V`, not the quality of deep engagement predictors.

---

## 11. Completed engagement results

All cells: main TEST 15.16, primary cap h90, identical rows within a cohort, equal match weighting, paired match
bootstrap of 1,000 replicates (seed 20260915). `ΔBrier = a − b`; negative favours `a`. Intervals hold fitted
models fixed and carry no multiple-comparison adjustment. These match-bootstrap metric intervals are a different
object from the descriptive **row**-bootstrap intervals over the 256 explained rows in §12, and from the
boundary-estimator bootstrap intervals for `G` and `D` in §5; the three are not interchangeable, and intervals
quoted from cited literature belong to those authors' designs. An interval containing zero is reported as
inconclusive: **no equivalence or non-inferiority test was run anywhere in this study**.

"Predeclared" below means a contrast was fixed in the relevant follow-up's own protocol and frozen manifest before
sealed predictions were read. No external preregistration record exists, and prior TEST exposure preceded these
follow-ups, so every result here is exploratory.

### 11.1 Increment over the reinforced baseline

The comparator **PT** is a tensor-product spline of `p_pre_V` with game time (quantile knots, 48 design columns)
with regularised logistic fitting — it is **not** `p_pre` alone.

| Predictor (T: 32,981 rows / 24,020 matches) | Brier | AUC |
|---|---:|---:|
| Constant | 0.249996 | 0.500000 |
| PT baseline | 0.229518 | 0.664795 |
| Full logistic (352 inputs) | 0.228405 | 0.669415 |
| Full LightGBM (352 inputs) | 0.228391 | 0.669125 |
| Prior cohort specialist (different budget/history) | 0.228171 | 0.670188 |

| Contrast | T all | T B40 (4,949 rows / 4,570 matches) |
|---|---|---|
| LightGBM − PT (predeclared primary) | −0.00113 [−0.00159, −0.00065] | −0.00043 [−0.00144, +0.00060] |
| Logistic − PT | −0.00111 [−0.00172, −0.00051] | −0.00157 [−0.00306, −0.00013] |
| LightGBM − logistic | −0.00001 [−0.00049, +0.00045] | +0.00113 [−0.00002, +0.00245] |
| New LightGBM − prior specialist | +0.00022 [−0.00031, +0.00076] | +0.00173 [+0.00036, +0.00315] |

For `N` (130,595 rows / 54,182 matches): PT 0.245040, logistic 0.242690, LightGBM 0.238688; LightGBM − PT
−0.00635 [−0.00682, −0.00592]; in `N` B40 (37,675 rows) −0.00630 [−0.00703, −0.00555].

**Reading.** Full-state models add information beyond a flexible function of the prior estimate and the clock, by
about one thousandth of a Brier score on teamfights and about six on other engagements. In balanced teamfights the
primary contrast is inconclusive while the logistic contrast favours the full-input logistic; the two must be
reported together, and neither generalises to "full-input models do (not) help in balanced states". The prior
specialist's slightly better point estimate comes from a different candidate pool and selection history and is not
the winner of the matched comparison.

### 11.2 Matched-input learner families (completed; previously planned)

Identical 352 columns, identical split, weighting, calibration and selection. Six candidate configurations per
family: six `C` values for the regularised logistic, three widths × two dropout rates for each MLP, three leaf
counts × two minimum-child sizes for LightGBM. The **stochastic** families — LightGBM and the two MLPs —
additionally average three seeds (7/42/123); the logistic is a single deterministic fit per candidate and is
**not** a three-seed ensemble. Network optimisation: AdamW (lr 1e-3, weight decay 1e-4), batch 512, ≤100 epochs,
patience 10.

| Cohort / cell | Contrast | ΔBrier [95%] |
|---|---|---|
| T all | residual MLP − LightGBM (primary) | +0.00061 [+0.00012, +0.00110] |
| T all | plain MLP − LightGBM | +0.00007 [−0.00046, +0.00057] |
| T all | plain MLP − logistic | +0.00006 [−0.00032, +0.00043] |
| T all | residual − plain MLP | +0.00054 [+0.00017, +0.00089] |
| N all | residual MLP − LightGBM (primary) | +0.00110 [+0.00079, +0.00145] |
| N all | plain MLP − LightGBM | +0.00109 [+0.00075, +0.00143] |
| N all | plain MLP − logistic | −0.00291 [−0.00328, −0.00249] |
| N B40 | residual MLP − LightGBM | +0.00185 [+0.00123, +0.00241] |

Five tabular families now exist on this target: PT, full logistic, full LightGBM, plain MLP, residual MLP. Width
128 was selected in all four network cases; the selected residual MLP reached its best epoch at epoch 1 on all
three teamfight seeds, which is consistent with rapid overfitting under the fixed optimisation budget but is not
proof of it. On teamfights the logistic/LightGBM/plain-MLP contrasts have intervals containing zero, so those
differences were not resolved here; **that is not equivalence and not a finding that learner choice is
immaterial**, and the residual MLP is measurably worse in the same cohort. On external cohorts the
network-versus-tree contrast contains zero except NA1 16.13 `T`, where LightGBM is better. This is a bounded
comparison; it does not establish a general ranking of neural networks, and FT-Transformer, TabNet, SAINT and the
representation adapters remain unexecuted (§16.1).

### 11.3 Endpoint-cap sensitivity (completed; previously planned)

Configurations were frozen at the h90 winners and only calibration was re-selected; row masks are identical, so
only labels change.

| Cohort | Model | h60 | h90 | h120 |
|---|---|---:|---:|---:|
| T | LightGBM | 0.22781 | 0.22839 | 0.22825 |
| T | Logistic | 0.22789 | 0.22840 | 0.22831 |
| T | Plain MLP | 0.22775 | 0.22846 | 0.22845 |
| T | Residual MLP | 0.22847 | 0.22900 | 0.22891 |
| T | PT baseline | 0.22903 | 0.22952 | 0.22938 |
| N | LightGBM | 0.23898 | 0.23869 | 0.23880 |

LightGBM − PT remains favourable with intervals excluding zero at both alternative caps
(−0.00122 at h60, −0.00113 at h120); tree-versus-linear and plain-MLP-versus-tree contain zero; the residual MLP
remains worse. Those particular comparisons are retained, but learner **ordering is not invariant**: the exact
point-estimate ranking moves, and the plain MLP has the lowest `T` value at h60 while trailing LightGBM at h90.
Absolute Brier levels are **not** comparable across caps because each cap induces a different label with its own
prevalence on the same rows; the lower `T` values at h60 therefore do **not** show that the 60 s label is an
easier task, which would need a difficulty analysis on a common target that we did not run. No horizon was
selected or justified from TEST results in this follow-up, and h90 is not optimised from the TEST ranking.

### 11.4 Role representation (earlier study, unchanged)

| Finding | Completed result | Interpretation |
|---|---|---|
| Weak-label role agreement | ≈0.979 out-of-fold | Agreement with weak labels, not true-role accuracy |
| External raw `teamPosition` agreement | ≈92–95% | A different imperfect reference; transfer limits remain |
| T role LightGBM | AUC ≈0.66946, Brier 0.228240 | No clear improvement over the T specialist |
| N participant → draft → role LightGBM | AUC 0.61451 → 0.61590 → 0.61804 | Representation gains within the same learner |
| Corresponding N Brier | 0.239885 → 0.239710 → 0.239173 | A tree-versus-linear difference must not be attributed to role information |

Role-block removal in the selected T ridge analysis shows deterioration for jungle and mid information with
limited consistency; redundant information remains elsewhere. We claim no universal lane-importance ranking, and
§6.3's positional caveat applies to any role wording.

---

## 12. Explanations

### 12.1 Earlier role-model explanations (unchanged)

Up to 256 hash-selected TEST cases per cohort with 128 TRAIN background cases; five role groups plus global
context and role uncertainty give seven groups and 128 coalitions for exact group Shapley over that finite
background; the explained quantity is the final selected calibrated probability, with additive reconstruction
checked. `T` explanations show substantial global-context contribution and `N` also shows bottom/jungle
contributions. These describe particular fitted models and sampled cases, not causally necessary actions.

### 12.2 Balanced-state explanations (completed; previously "not yet executed")

Seven groups (prior win probability 1, time 2, economy 60, combat 68, objectives 176, structures 24,
health/other 21), 128 coalitions, background 128 cohort TRAIN rows, 256 hash-selected rows per cell, additivity
error <4e-16, and `Σφ + base` reproducing the frozen parent prediction to <4e-16.

| Group (T, selected LightGBM) | Mean \|φ\| all T | Share | Mean \|φ\| B40 | Share |
|---|---:|---:|---:|---:|
| Prior win probability | 0.0971 | 61.7% | 0.0216 | 27.7% |
| Economy and experience | 0.0189 | 12.0% | 0.0190 | 24.4% |
| Objectives | 0.0151 | 9.6% | 0.0135 | 17.3% |
| Combat and survival | 0.0127 | 8.1% | 0.0108 | 13.9% |
| Structures | 0.0075 | 4.8% | 0.0066 | 8.4% |
| Health/mana/other | 0.0061 | 3.9% | 0.0064 | 8.2% |
| Time and observation age | 0.0000 | 0.0% | 0.0000 | 0.0% |

The absolute contributions of non-prior groups are essentially unchanged between cells; what falls is the prior's
own contribution, partly by construction because the cell is defined by restricting `p_pre`. A plain MLP, whose
Brier difference from the tree this protocol did not resolve (+0.00007 [−0.00046, +0.00057] on `T`), distributes
attribution very differently (objectives 21.0%, combat 20.7%, structures 20.0%, economy 17.2%, prior 14.8%), so
these shares are properties of fitted models, not of the game. Masked `p_pre` is not recomputed from the masked
state and derived time relations are not restored, so masked inputs can be off-manifold; the intervals here are a
**descriptive row bootstrap over the 256 explained rows**, not the match bootstrap used for the metric contrasts
of §11, and full-versus-B40 is an unpaired comparison of different rows.

### 12.3 What the label itself weights (completed; new)

Because the selected `V` is logistic with raw calibration, `Y = 1[βᵀ(z(S_e) − z(S_pre)) > 0]` exactly. The
identity was verified on all 366,746 stored valid main-TEST and VALIDATION rows: maximum block-sum deviation
3.042e-14, maximum logit-difference deviation 6.231e-12, and `1[Δlogit > 0] = Y` on every row. The frozen `V`
preprocessing expands 361 input columns into 1,733 transformed columns, of which the `champions__` block holds
1,382 (`outputs/v_mechanism_20260916/results.json` → `group_sizes/champion_identity`; grouping rule in
`scripts/vd20260916_common.py:column_group`), and 351 numeric columns plus that block sum to 1,733. Those champion
columns contribute exactly 0.

| Block (main TEST T) | Mean \|contribution\| | Share |
|---|---:|---:|
| Combat and survival | 0.451 | 35.5% |
| Economy and experience | 0.359 | 28.2% |
| Objectives | 0.240 | 18.9% |
| Structures | 0.123 | 9.6% |
| Health/mana/other | 0.091 | 7.2% |
| Time only | 0.008 | 0.7% |
| Champion one-hot | 0 | 0 |

Time-related columns including `×time` interactions account for 19.9% on a separate accounting that overlaps the
blocks, so the two figures must not be summed. In the 2,224 of 32,981 `T` rows (6.7%) whose **pre and post states
read the same observation frame**, the frame-based blocks contribute 0 and the label is carried by event columns
together with elapsed-time terms (time-only 0.5%, time-related 25.0%) — not by events alone. That stratum is
strictly narrower than the 35.27% of `T` rows with no new frame after the last kill (§13.2), which still permits a
frame change between the pre-state and the last kill. The structures share in the small-`|ΔV|` stratum (≤0.5 pp)
is 19.3%, against 9.6% over all `T` rows and 7.9% in the `|ΔV|` > 2 pp stratum. Static champion cancellation does not
mean champions are irrelevant: they still affect `p_pre` and therefore `|ΔV|` through the sigmoid, and can affect
the realised state change. `V(S(L))` was not stored, so the pre → last-kill → endpoint split has not been
computed. This is an arithmetic decomposition of a fitted linear score, not a causal decomposition, and the
individual ridge coefficients are collinear and are not interpreted alone: the largest **mean absolute
contributions** are `blue_kills_x_time` 0.110 and `red_kills_x_time` 0.102, whereas their fitted coefficients are
−1.0564 and +0.9901 — contribution magnitudes are not coefficients, and the kill terms differ in sign by side.
Exact decomposability is not by itself what distinguishes this label from a hand-weighted exchange sum, which can
also be decomposed exactly; the distinctions are that the weights are fitted against match outcome and that the
state and window entering the difference are defined explicitly.

---

## 13. Label sensitivity and observation diagnostics (earlier studies, unchanged)

### 13.1 Changing the value model

Two alternate value models were refitted on exactly the same 424,160 TRAIN queries and 74,168 matches, each with
final and five out-of-fold adapters. `B_reg` changes `C` from 0.01 to 0.1; `B_econ` retains economy, growth,
champion and time information but removes several event and state channels (it is not an objective-only ablation).

| Value model | Main TEST `W` AUC | `W` log loss | T h90 label disagreement with primary `V` |
|---|---:|---:|---:|
| Primary `V` | 0.8586 | 0.4567 | reference |
| `B_reg` | 0.8572 | 0.4586 | 1.09% |
| `B_econ` | 0.8300 | 0.4968 | 20.09% |

T match-bootstrap intervals for those row proportions are 0.973–1.198% and 19.666–20.505%. `E` disagreement is
1.28% and 26.74%; `N` is 1.32% and 28.42%. Sensitivity concentrates near zero: within `T`, rows with primary
`|ΔV| ≤ 0.5 pp` disagree with `B_reg` in 8.15% of cases versus 0.028% above 2 pp — a reason for caution about
small changes, not a new exclusion threshold. The frozen T predictor evaluated against primary, `B_reg` and
`B_econ` labels gives AUC 0.67019, 0.66821 and 0.63338; `q` was not retrained, so this measures dependence of an
existing predictor's evaluation on the target. Strictly increasing calibration applied identically at both
endpoints preserves the sign of the change.

### 13.2 Observation gaps

In main TEST `T`, 35.27% of cases have no new frame after the last kill, and the narrower same-pre-and-post-frame
condition occurs in 2,224 cases (6.74%); event features can still change. For `B_econ`, all non-time inputs are
identical in those same-frame cases, so its labels there are driven only by time, with a match-weighted
Blue-improvement rate of ≈97.98% versus 50.46% for the primary `V` and ≈49.86% disagreement — a limitation of that
reduced comparator, not evidence that primary labels are ground truth. Across all main TEST `T`, 15.85% have
primary `|ΔV| ≤ 1 pp`. An anonymised 120-case human-review packet was prepared; **no human judgements were
collected**, and human review is outside the current agreed scope.

---

## 14. Controlled objective-feature ablation (earlier study, unchanged)

Removing only the 176 explicitly named objective columns from the 361-column `V` input (Baron 38, Elder 38, dragon
32, soul 32, Herald 12, grubs 12, Atakhan 12, including related time interactions and specified histories), with
model family, `C`, TRAIN queries, weighting, preprocessing roles, folds and raw-probability treatment matched, and
`q` not refitted:

| Main TEST metric | Full `V` | Without explicit objective features |
|---|---:|---:|
| AUC | 0.858621 | 0.852913 |
| Brier | 0.153045 | 0.156129 |
| Log loss | 0.456742 | 0.465612 |

Paired match bootstrap: Brier +0.003084 [+0.002842, +0.003310]; log loss +0.008870 [+0.008223, +0.009499], where
positive means deterioration after removal. Label disagreement is 5.089% for `E`, 5.358% for `T` (equal-match
5.239%, interval 4.982–5.480%) and 5.022% for `N`; T row disagreement is 4.903% at h60 and 5.564% at h120, and no
horizon was selected from this comparison. Gold, experience, CS, structures, time and the mixed unknown-ownership
diagnostic can retain indirect objective information, so this removes an explicit bundle, not all information
about objectives. On the small KR 16.14 pilot the direction reverses, so we do not claim that objective features
improve every patch. Single-team acquisition subsets (Baron 2,195 cases, 6.287% disagreement; ordinary dragon
7,662, 8.079%; Elder 106, 5.660%; Herald 1,523, 4.859%; grubs 461, 9.111%; Atakhan 2,722, 8.303%; confirmed soul
677, 14.328%) overlap, were removed together, and do not support a per-objective importance ranking or a causal
price.

---

## 15. Validation and evidence strength

Completed verification includes exact row-key and split checks, parent-file preservation, fold membership, label
arithmetic, feature removal, preprocessing provenance, serialisation/reload consistency and independent metric
reconstruction. Post-run check counts for the current studies: incremental baseline 117/117 (independently
re-verified and accepted with 2,961 independent metric comparisons and an unchanged results hash), matched MLP 115
(run 2; run 1's four failures were check-code errors, receipt preserved), horizon 91, champion class 91 (third
attempt; earlier failures were parent-reproduction tolerances), definition sensitivity 53 (third attempt; earlier
failures were a census invariant), balanced SHAP 36, value-mechanism decomposition 24. The earlier readiness audit
recalculated endpoint formulas and labels across seven sets and three caps, covering 1,863,414 stored row-by-cap
combinations.

Passing these checks establishes arithmetic, implementation and provenance consistency **within the checked
scope**. It does not substitute for independent measurement validation or causal identification.

---

## 16. Status of the previously planned work

### 16.1 Model coverage

| Model | Provenance | Status on the Delta V target |
|---|---|---|
| Regularised logistic (352 inputs) | New matched baseline | **Completed** (§11.1) |
| LightGBM (352 inputs) | CoG 308 family | **Completed** (§11.1) |
| Plain matched-input MLP | CoG 308 family | **Completed** (§11.2) |
| Residual MLP | New matched baseline | **Completed** (§11.2) |
| PT reinforced baseline | New | **Completed** (§11.1) |
| Bi-GRU, Transformer (sequence) | CoG 308 | Planned; adapter not closed, not fitted |
| GraphSAGE, ST-GNN (graph) | CoG 308 | Planned; adapter not closed, not fitted |
| Cross-Attn, Layered Fusion | CoG 308 | Planned; not fitted |
| Layered plus Logit | 118-era diagnostic | Not part of this study; would need its own out-of-fold base predictions |
| FT-Transformer, TabNet, SAINT | Requested by reviewer 2 of 118 | **Not executed on this target** |

The common-information track compares logistic, LightGBM, plain MLP and residual MLP on the same 352 columns and
separates algorithm effects from the old economic-only tree limitation. It does not claim that sequences, graphs
and event tokens supply identical information merely because they share a telemetry source. Future positions,
post-game role labels and previously excluded information must not be reintroduced silently, and old runners that
fit on train plus validation, use old labels, or write into old result folders cannot be executed unchanged.

### 16.2 Balanced and uncertain situations — now executed

The primary balanced subset is `p_pre ∈ [0.40, 0.60]` (B40) with `[0.45, 0.55]` (B45) as a nested sensitivity.
These are operational choices, defined from pre-engagement `V` only, never from realised `ΔV` or final outcomes,
and they are not values prescribed by any cited paper. An estimated balanced match is different from a `q`
prediction near 0.5: the former concerns eventual match prospects, the latter model confidence about improvement
direction, and neither is a ground-truth measure of intrinsic difficulty. The full benchmark is retained; no model
was retrained only on balanced cases, no case was reweighted, and no model was re-selected after inspecting
balanced TEST scores.

**Completed:** balanced-cell evaluation with counts, Brier, log loss, AUC, calibration and paired intervals, plus
time-window strata (§11.1); balanced-state group explanations with separately hash-selected cases, the same TRAIN
background and retained group definitions with additive checks (§12.2). `V` calibration is evaluated against `W`
and `q` calibration against `Y`; the two references are not interchangeable. **Remaining:** external confirmation
of the balanced results, and the representation track of §16.1.

### 16.3 Definition independence — now executed

Constants re-estimated from TRAIN patch 15.14 alone (13.96 s / 4,285 u, rounded to 14.0 s / 4,285 u) were used to
re-detect and re-label all 210,000 main and 21,190 external matches with the frozen `V` and out-of-fold adapters.

| Set | Matches changed | Rows removed | Rows added |
|---|---:|---:|---:|
| TRAIN | 3.30% | 1.85% | 1.15% |
| VALIDATION | 3.47% | 1.91% | 1.19% |
| TEST | 3.54% | 1.95% | 1.23% |
| KR 16.13 / NA1 16.13 | 4.05% / 4.01% | ≈2.4% | ≈1.5% |

The metric consequences are cell-specific and are not summarised by one bound. On the main-TEST teamfight cell the
frozen LightGBM Brier moves 0.228391 → 0.22837 between the old and dev populations — a particular small
descriptive change across two different row sets — and the main-TEST `N` frozen change is of the same order, but
external frozen cells move further (KR 16.15 `N` 0.24062 → 0.23996, about 0.0007; KR 16.13 `T` 0.23510 → 0.23476).
These are Brier comparisons across different row populations, not a bound on all metrics: main-TEST AUC
changes reach about 0.00075 in the recorded frozen-model cells. The predeclared contrast, refit minus
frozen LightGBM on `T`, is −0.00007 [−0.00023, +0.00010] (inconclusive). On `N` the refit is worse
(+0.00023 [+0.00011, +0.00034]) and on KR 16.13 `T` it is worse (+0.00052 [+0.00010, +0.00099]), both with
intervals excluding zero. Rebuilt rows —
engagements merged or split by the new gap — are harder (T Brier 0.232 versus 0.228) but are only ≈5% of `T`.

**What this does and does not settle.** It measures the consequence of one alternative constant pair; it does not
remove the fact that the primary definition was estimated with the test patch included. `D = 4,285 u` lies outside
the pooled interval [4,256.0475, 4,273.9677], although `G = 14.0 s` lies inside the temporal plateau. Inconclusive
is not equivalence, only one refit per arm exists, a 15.14+15.15 estimate was not produced, and other detector
switches were not varied.

### 16.4 Champion composition — new study, no detected benefit in the tested setup

A user proposal to embed Riot champion classes and learn composition structure (for example "a fed assassin
against a fed marksman", jungle–mid pairs, bottom duos, 5v5 structures) was tested in tabular form: slot classes
(120 columns), class-conditional team state (126), slot-pair and cross-team matchup combinations (396) and
champion identity one-hot (1,710) stacked on the frozen 352 columns, plus two draft-only arms without state.

The predeclared primary comparison, `T` LightGBM class-pairs minus base, is +0.00013 [−0.00009, +0.00035]; the
`N` counterpart is +0.00027 [+0.00004, +0.00047] (slightly worse); identity adds nothing detectable; draft-only
arms reach AUC 0.50–0.52; the logistic arms degrade monotonically with width under a **fixed** regularisation
constant. Externally the `T` contrasts contain zero and two `N` cells are worse.

**How far this goes.** In this setup, with these learners and budgets, no benefit is detected. That is not
equivalence and not evidence that no learnable composition information exists. Graph representations, finer class
taxonomies, neural arms on the class features and a re-searched regularisation constant were **not** tested. The
features are slot-index pairs under a positional proxy assumption (§6.3), so no role-matchup conclusion follows
either way. A numerical-reproduction caveat applies: the base LightGBM arm reproduces the parent exactly, while
the base logistic arm differs by ≈6e-7 to 3.4e-5 raw and up to 0.0118 after `N` isotonic calibration.

### 16.5 Relation to the suggested calibration paper

Kim, Lee and Chung study input-dependent uncertainty and calibration for LoL match-win prediction [R6]. That
supports attention to probability quality, which matters because our valuation uses probabilities. We have **not**
implemented their uncertainty-aware loss and claim no reproduction. Our raw/sigmoid/isotonic comparisons and
subgroup diagnostics are distinct procedures. The balanced-fight emphasis comes from the collaborator's e-mail,
not from that paper. The collaborator recommended reading the paper; the further idea of **comparing** their
method with simpler calibration methods in balanced states is our own proposed extension, stated in our outgoing
reply draft, and has not been requested or executed.

---

## 17. Remaining limitations and journal work

The most important unresolved issues are measurement validity, patch independence of the definition (now bounded,
not removed), observation granularity, and completion of the representation comparisons. Human semantic review
remains unperformed by scope choice, so our wording retains a model-defined outcome rather than an independently
validated fight-win truth; this is an open validity gap, not a newly imposed prerequisite.

The observed global `ΔV` may include farming, actions elsewhere, time progression and multiple conversions;
without a counterfactual design it is not the isolated effect of the focal fight. `V` was trained against `W`, so
correlation between a `V`-defined label and `W` is not independent proof that `q` contributes new information
about match victory; a `q` feature is a deterministic summary of its input, and any benefit to a finite downstream
model must be evaluated against a model with comparable pre-engagement information.

Existing external tests provide transfer evidence with calibration deterioration, unseen champions, missing
objective channels, sparse subsets and prior access histories. They do not establish universal cross-patch
operation without adaptation. Genuinely new confirmation data remain a priority.

| Reviewer or collaborator concern | Completed response | Remaining work |
|---|---|---|
| Arbitrary engagement constants | Data-estimated boundaries, stability analysis, and a completed development-patch-only re-detection | Untouched confirmation data; other detector switches; a 15.14+15.15 estimate |
| Subjective exchange weights | Learned `V`, out-of-fold labels, sensitivity, objective ablation, and an exact decomposition of what the label weights | Independent semantic criterion; alternative-label variants on the new target |
| Strong, fair model comparison | Matched-input comparison of five tabular families with one protocol; the particular comparisons named in §11.2 are retained at three endpoint caps, while exact rankings move and several teamfight contrasts stay unresolved | Modern tabular transformers; sequence, graph, cross-attention and fusion adapters |
| Importance of balanced games | Balanced-cell evaluation and balanced-state explanations executed | External confirmation; the teamfight answer is unresolved rather than negative |
| Probability calibration | `V` and `q` calibration evaluation in separate validation roles | Kim et al.'s loss not implemented; subgroup calibration study |
| Explainability | Group SHAP with additivity checks, role-block diagnostics, and the label decomposition | Corrected group definitions would require regeneration; `V(S(L))` split not computed |
| Champion composition | Tabular class, slot-pair and identity study completed with no detected benefit in the tested setup (a non-detection, not a demonstrated absence of information) | Graph and finer-class tests; the positional-semantics audit |
| Practical live use | Cutoff-respecting historical replay | Live acquisition, occurrence detection, routing, latency |
| Artefact availability | Local manifests, hashes, frozen predictions | Anonymised repository, licence, reproduction instructions, data policy |

---

## 18. Evidence and reproducibility inventory

Provenance references to retained local artefacts, not public download links.

| Evidence ID | Retained artefact | What it supports |
|---|---|---|
| E01 | `docs/TOG_END_TO_END_READINESS_AUDIT_20260915.md` and its `calculations.json` | Definitions, counts, endpoint audit, `V`/`q` results, limits |
| E02 | `outputs/full_corpus_training_20260915/` protocol and REPORT | Full designated-corpus `V`, labels and `q` |
| E03 | `outputs/cohort_role_training_20260915/` protocol and REPORT | Cohort separation, role models, earlier explanations |
| E04 | `docs/LABEL_VALIDITY_FINDINGS_20260915.md` and `label_validity_full` outputs | Alternative `V` fits, observation diagnostics |
| E05 | `docs/OBJECTIVE_CHANNEL_ABLATION_FINDINGS_20260915.md` and outputs | Explicit objective ablation and independent audits |
| E06 | `docs/DELTA_Q_FAIR_BALANCED_PROTOCOL_20260915.md` | The protocol under which §11.1–§11.3 and §12.2 were run |
| E07 | `docs/COG_MODEL_COVERAGE_DELTA_Q_20260915.md` | Planned full CoG coverage and unresolved adapters |
| E08 | CoG 308 paper text, Tables I–III | Published label, model lineup, historical results |
| E09 | Paper 118 review archive and runner | Version distinction and the Layered-plus-Logit diagnostic |
| E10 | `docs/DELTA_LABEL_LEARNING_EXAMPLES_20260915.md` | Hypothetical arithmetic and target explanation |
| E11 | `outputs/incremental_q_training_20260915/` + `outputs/claude_dispatch_incremental_q/final_acceptance.json` | Reinforced baseline comparison and its independent verification |
| E12 | `outputs/track_a_mlp_20260916/`, `docs/TRACK_A_MLP_FINDINGS_20260916.md` | Matched-input learner comparison |
| E13 | `outputs/horizon_sensitivity_20260916/`, `docs/HORIZON_SENSITIVITY_FINDINGS_20260916.md` | Endpoint-cap sensitivity |
| E14 | `outputs/definition_dev_20260916/`, `docs/DEFINITION_DEV_FINDINGS_20260916.md` | Development-patch-only definition sensitivity |
| E15 | `outputs/champion_class_20260916/`, `docs/CHAMPION_CLASS_FINDINGS_20260916.md` | Champion-class and slot-pair study |
| E16 | `outputs/v_mechanism_20260916/`, `docs/V_MECHANISM_FINDINGS_20260916.md` | Exact label decomposition |
| E17 | `outputs/balanced_shap_20260916/`, `docs/BALANCED_SHAP_FINDINGS_20260916.md` | Balanced-state explanations |
| E18 | `outputs/manuscript_integration_20260916/position_audit.md`, `citation_audit.md` | Positional-assumption correction and verified bibliography |

Interpretive statements in E12–E17 that this specification deliberately narrows are listed in
[claim_evidence_ledger.md](claim_evidence_ledger.md) §3; their numerical results are unchanged.

---

## 19. References and their scope

Only sources verified in the 16 September 2026 citation audit are cited in the current draft; they appear with
full metadata and explicit support limits in [references.bib](references.bib). The list below preserves the
earlier specification's scope annotations, including sources that are **not** in the current bibliography because
they were not re-verified; they must be re-verified before being cited in a submission.

R1. Halfaker et al. 2015, user session identification. Supports deriving activity boundaries from timing
distributions. Our KDE-valley estimator and the 13.7 s value are adaptations, not that paper's validated boundary.
*(Verified 2026-09-16; in `references.bib`.)*

R2. Jacobs and Wallach, measurement and fairness. Supports separating the construct of strategic success from its
operational measurement. It does not validate our labels. *(Not re-verified; not cited in the current draft.)*

R3. OpenDota `processTeamfights` implementation. An implementation precedent for retrospective fight
construction, not peer-reviewed evidence for LoL-specific constants. *(Not re-verified.)*

R4. Austin, Lee and Fine 2016, competing risks. Supports cumulative-incidence analysis with competing events. It
does not establish that every subsequent kill is a new fight or that 90 s is optimal. *(Not re-verified.)*

R5. Lundberg and Lee 2017, SHAP. Supports additive attribution. Our groups, background budget and sample
selection are operational choices, and the attribution is not causal here. *(Verified 2026-09-16; in
`references.bib`.)*

R6. Kim, Lee and Chung 2020, confidence-calibrated MOBA winner prediction. Supports uncertainty-aware match-win
calibration as a motivation. Its loss is not implemented in our pipeline. *(Verified 2026-09-16; in
`references.bib`.)*

R7. Maymin 2021, smart kills and worthless deaths. Supports the motivation for valuing actions relative to a
changing win estimate. Its data and metrics differ from our public-timeline design; it prescribes neither our
endpoint nor our labels. *(Verified 2026-09-16; in `references.bib`.)*

R8. Gorishniy et al. 2021, revisiting deep learning for tabular data. Motivates strong residual tabular baselines
and consistent comparison protocols. Our bounded settings are not a reproduction or a proof of general deep-model
superiority. *(Not re-verified; not cited in the current draft, although the residual-MLP arm was motivated by this
line of work.)*

R9. Ke et al. 2017, LightGBM. Establishes the learner family, not its expected ranking on the Delta V task.
*(Not re-verified.)*

R10. Lee and Ramler 2017, non-meta strategies. Supports role/composition inference as a direction. The original
work uses end-of-match information, which our pre-engagement contract forbids. *(Not re-verified.)*

R11. Hodge et al. 2021, live professional match prediction. A precedent for time-dependent esports match
prediction. It does not establish live availability of our chosen features. *(Verified 2026-09-16; in
`references.bib`.)*

R12. Schubert, Drachen and Mahlmann 2016, encounter detection. Supports spatio-temporal encounter segmentation as
a precedent and documents a large kill-free encounter population in their data (81.1%). It does not justify our
constants or our kill-conditioned population. *(Verified 2026-09-16; in `references.bib`.)*

Across these sources we distinguish literature-supported methods, game-rule or implementation precedents, values
estimated from our data, and our own operational decisions. No citation is used to imply that a threshold or a
label interpretation is externally established.
