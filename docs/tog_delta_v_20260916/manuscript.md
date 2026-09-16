# Predicting the Direction of Model-Estimated Win-Probability Change Around League of Legends Engagements

**Working draft, 16 September 2026. Not a submission.**

This document is an integrated working manuscript for a journal extension. It is not submission-ready, has not
been compiled to PDF, and reports no new experiment: every number below comes from experiment outputs that were
frozen and evaluated before this draft was written. The author block, affiliations, acknowledgments, funding
statement and anonymised-repository link are deliberately omitted; they are open submission tasks
(§11.2). All results in §8 are **exploratory follow-ups run after prior exposure to the main TEST patch and to
the external cohorts**, not untouched confirmatory tests (§7.5).

Companion files: [claim_evidence_ledger.md](claim_evidence_ledger.md) (every number below with its source path
and JSON key), [reviewer_response_matrix.md](reviewer_response_matrix.md) (issue-level response to the CoG 2026
submission 118 reviews), [collaborator_specification.md](collaborator_specification.md) (collaboration-facing
specification), [references.bib](references.bib), [README.md](README.md) (Korean navigation and status).

---

## Abstract

Teamfights are widely treated as decisive moments in League of Legends, but "which team won the fight" has no
external ground truth in public match data. We therefore study a target that is fully defined by our own
pipeline and can be audited end to end: the **direction in which a separately trained match-win value model
changes across a retrospectively identified, kill-conditioned engagement**. We fit a value model `V` that
estimates the probability of eventual Blue victory from the state observable at a query time, define an
engagement interval from kill events, and label each engagement by the sign of
`ΔV = V(S(endpoint)) − V(S(pre-onset))`. A predictor `q` then estimates the probability of a positive label from
information available strictly before onset. On a designated corpus of 210,000 Korean matches split by patch
(15.14 train, 15.15 validation, 15.16 test) with 566,104 valid engagement rows, `q` is measurably better than a
reinforced baseline that already knows the pre-engagement win estimate and the game clock: on the 32,981 teamfight
rows of the test patch, the matched-input LightGBM improves the Brier score by 0.00113 (95% match-bootstrap
interval [0.00065, 0.00159]) over that baseline, and by 0.00635 [0.00592, 0.00682] on the 130,595 other-engagement
rows. The margin is small in absolute terms and, on teamfights whose pre-engagement estimate lies in
[0.40, 0.60], that LightGBM contrast is inconclusive (−0.00043 [−0.00144, +0.00060]) while the full-logistic
contrast in the same cell excludes zero (−0.00157 [−0.00306, −0.00013]) as an exploratory secondary comparison.
Under a matched 352-feature input contract, the teamfight contrasts of a plain multilayer perceptron against
LightGBM and against regularised logistic regression have intervals containing zero — no equivalence test was
performed — a residual MLP is measurably worse (+0.00061 [+0.00012, +0.00110]), and LightGBM is better than both
networks on other engagements; these broad comparison patterns persist when the endpoint cap is moved to 60 s or
120 s, although the exact point-estimate ranking moves (at 60 s the plain MLP has the lowest teamfight Brier).
Re-deriving the engagement constants from the training patch alone changes 3.3–4.4% of matches; on main-test
teamfights the frozen LightGBM Brier moves 0.228391 → 0.22837 and the refit-versus-frozen contrast is inconclusive
(−0.00007 [−0.00023, +0.00010]), but the same refit contrast is worse on other engagements
(+0.00023 [+0.00011, +0.00034]) and on external KR 16.13 teamfights (+0.00052 [+0.00010, +0.00099]). Adding
champion-class, slot-pair and champion-identity features to the frozen 352 inputs produces no detected benefit for
LightGBM in this setup. An exact decomposition
shows the label is the sign of a linear score difference in the value model's transformed feature space, in which
static champion terms cancel exactly; a group-Shapley analysis shows the fitted `q` leans on the pre-engagement
estimate in general cases and distributes attribution to economy, objective and combat groups in balanced cases.
We report what these results do and do not establish: they characterise a model-defined strategic outcome, not a
validated judgement of fight victory, and not a causal effect of fighting.

---

## 1. Introduction

### 1.1 Problem

Public League of Legends telemetry exposes two streams with very different resolution: participant-state frames
at roughly one-minute intervals, and discrete events (kills, structures, objectives) at millisecond precision.
Work on esports moments must therefore decide two things before any modelling: *what counts as an engagement*,
and *what counts as winning it*. The second decision is the harder one. A fight can be won without a kill, by
forcing cooldowns, denying vision or taking map control; conversely, a favourable kill exchange can still cost a
team the following objective. Any label that scores events with fixed coefficients embeds the researcher's
judgement into the target, and a model trained on it may be learning the scoring rule rather than the game.

This draft takes a deliberately narrow route around that problem. Instead of asserting a fight-victory rule, we
make the outcome definition explicit and measurable: we train a match-win value model and ask whether a team's
**estimated** probability of eventual victory improved across a defined interval around the engagement. That
quantity is not fight victory. It is a model-defined strategic outcome whose construction can be audited,
decomposed and varied, and whose limits we state directly.

### 1.2 Research question

> Given only information observable strictly before the onset of a retrospectively identified kill-conditioned
> engagement, how well can we predict whether the model-estimated Blue match-win probability will be higher at a
> defined endpoint than immediately before onset, and how does that predictability depend on the learner family,
> the endpoint cap, the definition constants and the input representation?

### 1.3 Contributions

The following are stated as reusable outputs and audited negative or inconclusive findings rather than as a
restatement of the question.

1. **An auditable outcome definition with an exact mechanism account.** We give the endpoint rule, the label
   arithmetic, and an exact column-level decomposition of the label in the value model's transformed space over
   366,746 stored rows, including the fact that static champion terms cancel from the sign (§8.6). Others can
   reuse the construction, or replace `V` and re-derive labels, without reverse-engineering a scoring table.
2. **A reinforced baseline that later work should have to beat.** Our comparator is not "prior win probability
   alone" but a tensor-product spline of the pre-engagement estimate with game time (§6.2). Reporting increments
   against it, rather than against a constant, changes how large the remaining signal looks (§8.1).
3. **A matched-information learner comparison on a single input contract.** Logistic regression, LightGBM, a
   plain MLP and a residual MLP see exactly the same 352 columns under one selection and calibration protocol
   (§8.2), so family differences are not confounded with feature engineering. On teamfights the pairwise
   contrasts among logistic regression, LightGBM and the plain MLP all have intervals containing zero, while the
   residual MLP is measurably worse; no equivalence procedure was run, so the result bounds rather than settles
   the family question for anyone planning architecture work on this target.
4. **Sensitivity results with experiment-specific scope.** The 60/90/120 s endpoint study (§8.3) reports
   the tabular and neural comparisons, including changed point-estimate rankings. The TRAIN-only engagement
   definition study (§8.4) evaluates PT, logistic and LightGBM arms; it does not test neural-model robustness.
   Its main teamfight refit-versus-frozen contrast is inconclusive, while other engagements and some external
   cells worsen. Changed matches, removed rows and added rows are reported separately.
5. **No detected benefit from champion composition on this target in the tested setup** (§8.5), including the
   correction that the features are slot-index pairs under a positional proxy assumption rather than verified lane
   matchups. An interval containing zero here is a non-detection under one representation and budget, not a
   demonstrated absence of composition information.
6. **An explanation layer with stated boundaries** (§8.7): exact group Shapley values over seven groups with
   additivity checked within 4e-16 against the frozen predictions, reported as model description, not causal
   attribution.
7. **An explicit statement of what remains unestablished** (§10), including semantic validity of the label,
   confirmatory evaluation on untouched future data, representation tracks that were not executed, and live
   operation.

### 1.4 Lineage and what is not carried over

This draft's target and cohort differ from the authors' earlier conference work. The published CoG 308 paper used
a **manually weighted signed event exchange** score — `Y_CoG = 1[Σ_u α_u σ_u v_u > 0]` with
`α_u = exp(2 p_u)/Σ_v exp(2 p_v)`, hand-set event values `v_u`, importance priors `p_u` and separate special-event
handling — together with a fixed 18 s kill gap, 4,000 u spatial separation and a 10 s onset offset. Three label
lineages must be kept apart: that published exchange score; the intermediate `market_event` experiments in the
earlier codebase, whose implementation we do not re-verify here and which we therefore do **not** identify with
the published score; and the present `ΔV` target. None of the earlier constants, coefficients, metrics or model
rankings is carried into the present results. They appear
in this draft only where lineage must be explained (this subsection and
[collaborator_specification.md](collaborator_specification.md) §2) and in the reviewer-response mapping. The separate CoG submission 118 was rejected; its reviews are the input to
this revision and are answered in [reviewer_response_matrix.md](reviewer_response_matrix.md). Numbers from either
CoG record must not be compared with the numbers in §8 as if they measured the same thing: the label, the
population, the split and the input contract all changed.

---

## 2. Background and related work

Claims in this section are kept to what the cited work actually supports; per-source boundaries are recorded in
[citation_audit.md](../../outputs/manuscript_integration_20260916/citation_audit.md) and restated in
[claim_evidence_ledger.md](claim_evidence_ledger.md) §4.

### 2.1 Win probability as a valuation device

Maymin [Maymin2021] fits a logistic in-game win-probability model on elapsed minutes and both teams' cumulative
kills, towers and large monsters, samples one random minute per game to reduce within-game row dependence, and
then evaluates events by the change in the estimated win probability. That paper is the closest published
precedent for the device we use: a frozen state-conditioned probability estimate, differenced across an event
window. It is a methodological lineage only. It does not validate our feature schema, our patch and region
population, our engagement definition, our endpoint rule, or any causal reading of a probability change. Our `ΔV`
remains a model-derived difference over a window we define.

Hodge et al. [Hodge2021] build and deploy live professional DotA2 win-prediction models with a five-minute
sliding state window and minute-indexed models, evaluated chronologically and deployed at a tournament. We cite
it as evidence that live runtime MOBA prediction and its data-availability limits are an established research
setting, not as support for our League of Legends value model, its features, its calibration, or any of our
reported accuracies; their reported live figures are not our results.

Kim, Lee and Chung [Kim2020] study League of Legends winner prediction with an input-dependent,
uncertainty-aware calibration method and evaluate ECE, MCE and NLL alongside accuracy. We cite them for the
motivation to treat probability quality as a first-class concern, which matters here because our label is built
from probabilities. We did **not** implement their uncertainty-aware loss or their architecture; our `V` is a raw
logistic model, and our calibration work is limited to comparing raw, sigmoid and isotonic transforms in
dedicated validation roles.

### 2.2 Segmenting activity in time and space

Halfaker et al. [Halfaker2015] identify user sessions from inter-activity times by fitting within- and
between-session components on log-scaled gaps and choosing the threshold at the likelihood intersection,
reporting roughly one hour as a rule of thumb while showing that fitted intersections vary widely across logs. We
follow the *idea* of reading a temporal boundary out of an inter-event distribution. Our `G = 13.7 s` is a KDE
valley estimated on our own kill-gap data, not a value from that paper, and a KDE valley is not the same
estimator as a Gaussian-mixture intersection.

Schubert, Drachen and Mahlmann [Schubert2016] define Dota encounters from spatio-temporal links among units using
team/opponent relations and distance rules, and relate encounter summaries to outcome and win-probability
prediction. Two facts matter for us. First, it establishes encounter segmentation with distance and time gates as
a precedent. Second, its own data contain many encounters with no kill at all — 18,744 of 23,110 (81.1%) — which
is a direct warning that a kill-conditioned population like ours is a strict subset of "combat". It does not
justify our specific `G`, `D`, `B` or `R`, and it does not establish that an engagement causes a positive `ΔV`.

### 2.3 Attribution

Lundberg and Lee [Lundberg2017] introduce SHAP as a unified additive feature-attribution framework, with
attributions defined relative to an explanation mapping and summing to the represented model output under stated
assumptions; feature dependence and background data affect interpretation. We use exact group Shapley values over
seven fixed groups, one of which is the fixed pre-engagement probability, with a finite background sample. What we
explain is the fitted `q` output under that grouping and background. It is not an explanation of `V` or of `ΔV`,
and it is not causal evidence that a feature caused a fight to go well.

### 2.4 What no cited work supplies

No source in our bibliography establishes: that `G = 13.7 s` or `D = 4,264 u` are correct boundaries; that a 90 s
cap is optimal; that the sign of `ΔV` is fight victory; that champion composition should or should not be
learnable on this target; or that any of our reported differences generalise across patches and regions. Those
are our own operational choices and our own estimates, and they are justified in this draft by rationale plus
sensitivity analysis, not by citation.

---

## 3. Data, cohorts and eligibility

### 3.1 Main corpus and patch roles

| Patch role | Raw matches | Detected engagements | Valid ΔV rows | T rows | N rows |
|---|---:|---:|---:|---:|---:|
| TRAIN 15.14 | 74,673 | 199,480 | 199,358 | 39,605 | 159,753 |
| VALIDATION 15.15 | 74,748 | 203,292 | 203,170 | 41,315 | 161,855 |
| TEST 15.16 | 60,579 | 163,680 | 163,576 | 32,981 | 130,595 |
| **Total** | **210,000** | **566,452** | **566,104** | **113,901** | **452,203** |

The same valid-row masks apply at the 60, 90 and 120 s caps. The 348 excluded rows are engagements whose next
eligible onset falls at or before the current last kill. Stored match identifiers do not intersect across the
three patch roles. "Using the full corpus" means every designated match keeps its assigned patch role; it does
**not** mean fitting on all 210,000 matches.

### 3.2 Value-model eligibility, reported separately

Value-model eligibility is a different filter from engagement eligibility and is reported on its own terms.
208,603 of the 210,000 main matches are eligible; of the 1,397 excluded, 1,396 have no eligible query time and one
has missing or conflicting final-winner information. The final `V` is fitted on 424,160 state queries drawn from
74,168 eligible TRAIN matches, including matches that contain no qualifying engagement. Main-TEST value evaluation
uses 347,234 queries from 60,190 eligible matches. Value queries lie on a one-minute grid starting at minute two,
before match termination and within the observed frame range; one eligible query per five-minute bucket is
selected deterministically by hash and each match carries equal total weight. This is within-match sampling for
computational control, not a reduced-match pilot, and it does not mean the last two minutes of a match are
discarded.

### 3.3 Validation roles

Patch 15.15 is partitioned by match hash into four disjoint roles: V_CAL 18,579, V_SELECT 18,890, Q_CAL 18,710
and Q_SELECT 18,569 raw matches, with task-specific eligibility applied afterwards. Value calibration and
selection therefore use different matches from `q` calibration and selection. TRAIN engagement labels come from
five-fold value models that exclude the labelled match; the same out-of-fold adapter evaluates both endpoints of
a training engagement and supplies its pre-engagement probability. Validation, test and external rows use the
final value model.

### 3.4 External cohorts

| External cohort | Raw matches | Valid ΔV rows | T rows | N rows |
|---|---:|---:|---:|---:|
| KR API 16.13 | 10,064 | 25,430 | 5,202 | 20,228 |
| KR API 16.14 pilot | 200 | 475 | 101 | 374 |
| KR API 16.15 | 926 | 2,219 | 507 | 1,712 |
| NA1 API 16.13 | 10,000 | 26,538 | 5,312 | 21,226 |
| **Total** | **21,190** | **54,662** | **11,122** | **43,540** |

No EUW1 evaluation exists. Database public-patch identifiers and API patch identifiers are not interchangeable.
Every one of these sets has prior inspection or an incompletely established access history, so none is presented
as an untouched confirmatory cohort; the 200-match KR 16.14 set remains explicitly a pilot. External data also
contain champions unseen in training, affecting roughly 18–25% of value queries, and contain no observed Atakhan
or FEAT_UPDATE events, so those channels are untested under transfer.

---

## 4. Engagement definition

### 4.1 Construction

Engagements are identified retrospectively from kill events. Two constants are estimated from data rather than
asserted: a temporal gap `G` that separates kill clusters, and a spatial diameter `D` that separates
simultaneous but unrelated fighting.

| Symbol | Value in use | How it was obtained | What it does not establish |
|---|---|---|---|
| `G` | 13.7246 s, operationally 13.7 s | KDE valley between activity regimes in pooled kill-gap distributions; bootstrap interval 12.1601–15.6675 s; stability plateau roughly 10–18 s | Not Halfaker et al.'s mixture intersection; not a proven natural boundary in every game phase |
| `D` | 4,263.8688 u, operationally 4,264 u | Distance at which the observed champion-sharing frequency of temporally close kill pairs crosses 0.5; interval 4,256.0475–4,273.9677 u | Champion sharing is a proxy for common interaction, not an annotation of one combat; 0.5 is an operational cut; interpolation between 750 u bins means a narrow interval is not fine physical resolution |
| `R` | 1,600 u presence radius | Motivated by the champion-death experience range (a game rule) | Experience range is not proven to equal combat participation range |
| `B` | onset lead 15 s (`s = K − 15 s`) | Operational offset informed by assist-time scale and implementation precedent | Not a measured true combat start |
| `M` | at least two per team under the configured presence rule | Existing operational gate | Pre-onset presence and retrospective participation are different quantities |
| Scale cut | `T` requires recorded minimum cluster participation of at least 4 per side; `N = E \ T` | v3.3 scale convention | 4 is not a learned optimum; the older `cut = 3` convention is not used |
| `q` cutoff | `s − 1 ms` | Design choice that makes boundary ties explicit | Engagement existence and `T` membership are retrospective information |

`G` and `D` were estimated on a pooled record covering 208,141 matches, 10,417,458 temporal gaps and 5,460,008
temporally close kill pairs. Some estimation steps use seeded computational samples; we do not claim that every
bootstrap replicate refits every observed gap. Per-patch estimates are approximately 13.964 / 13.490 / 13.725 s
and 4,285.26 / 4,265.27 / 4,241.47 u. Additional detector switches — merging behaviour, duration limits,
participation extraction, interaction radius — remain as configured and are not reduced to `G` and `D`.

`R = 1,600 u` and the 15 s onset lead are **not** learned constants; they are rule-motivated and design choices
respectively.

### 4.2 Scope

`T` is a subset of `E`, not a disjoint alternative to it, so pooled-versus-specialist comparisons are always made
on identical rows within a cohort. The population is kill-conditioned by construction: encounters that produce no
kill, zoning, cooldown denial and vision control are outside it. Given that Schubert et al. [Schubert2016] report
81.1% of their Dota encounters as kill-free, this exclusion is substantial and is a scope limit, not an
oversight, and we do not claim to cover "all combat".

### 4.3 The pooled estimate includes the test patch

`G` and `D` were estimated on 15.14, 15.15 and 15.16 together. The prediction models' train/validation/test
separation is intact, but the *definition* is not independent of the test patch. §8.4 reports the measured
sensitivity to that dependence; it reduces the concern but does not remove the history.

---

## 5. Value model and label

### 5.1 The value model

`V` estimates `P(W = 1 | S(t))`, where `W` is actual final Blue match victory taken from the match-end event and
`S(t)` is the state built from information observed no later than `t`. The selected primary `V` is a logistic
model on expanded StateV2 features with `C = 0.01`, fitted with liblinear, maximum 1500 iterations, seed 7.
Imputation, standardisation and one-hot vocabularies are fitted on the relevant TRAIN partition only. Raw
probabilities and a positive-slope sigmoid calibration were compared on the dedicated V_CAL/V_SELECT roles by log
loss and then Brier; **raw** was selected.

| Main TEST metric against `W` (347,234 queries, 60,190 matches) | Primary `V` |
|---|---:|
| AUC | 0.858621 |
| Brier | 0.153045 |
| Log loss | 0.456742 |
| Calibration slope | ≈0.976 |

AUC in the 2–10 minute window is approximately 0.6741, so the overall figure must not be read as uniform across
match time. On external cohorts, value AUC is approximately 0.84–0.86 with calibration slopes approximately
0.81–0.87, i.e. probability calibration degrades under transfer. These are `W`-prediction numbers; they are not
comparable to the `q` numbers in §8, which predict a different target.

### 5.2 Interval and label

Let `K` be the first kill, `s = K − 15 s`, `t_pre = s − 1 ms`, `L` the last kill of the engagement, `J` the first
subsequent champion kill anywhere on the map, `S_next` the next eligible engagement's onset, and `T_end` match
termination. Missing candidates are treated as infinite.

```
e(h)  = min{ L + h,  J − 1 ms,  S_next − 1 ms,  T_end − 1 ms }
ΔV(h) = V(S(e(h))) − V(S(t_pre))
Y(h)  = 1[ ΔV(h) > 0 ]
h     = 90 s primary; 60 s and 120 s as sensitivity
```

Both endpoints of a row use the same value adapter and the same calibration. The administrative cap `L + h` is
inclusive; competing event timestamps are excluded by one millisecond, and events sharing an excluded timestamp
are not artificially ordered. Overlaps where `S_next ≤ L` are excluded (the 348 rows of §3.1). Exactly zero change
is coded `Y = 0`, meaning Blue non-improvement; no exact zeros occur in the retained valid labels, and because of
this convention the zero class is not strictly identical to "Red improvement".

**Objective acquisitions do not stop the interval.** Several acquisitions can be reflected before the endpoint.
Conversely, the next global kill does stop it, even when that kill is a distant skirmish or a pursuit, so the rule
is conservative and can truncate fighting that a human would still call part of the same sequence. `S_next` is
itself derived retrospectively, so this is **not** a fully observable live stopping rule.

### 5.3 What the label is and is not

`q = 0.7` means an estimated 70% probability of the positive `ΔV`-direction label under this definition. It is
not a 70% final match-win probability, not a 70-percentage-point gain, not a magnitude, not a causal effect of
fighting, and not a prediction that a fight will occur. Because the population is located retrospectively,
the estimand is conditional on an engagement having been identified; occurrence detection and live cohort routing
are separate, unbuilt tasks.

### 5.4 Endpoint behaviour in the stored labels

| Main TEST h90 quantity | Value |
|---|---:|
| Observed follow-up after last kill, mean / median | 45.493 s / 39.440 s |
| Reaching the 90 s cap | 14.286% |
| Stopped by the next global kill | 129,036 / 163,576 (≈78.9%) |
| No new observation frame after `L`, all 163,576 valid rows | 35.145% |
| \|ΔV\| ≤ 0.005 / 0.01 / 0.02 | 7.504% / 12.696% / 20.905% |
| Sign disagreement h60↔h90 / h90↔h120 / h60↔h120 | 1.545% / 0.608% / 1.909% |

Two frame-staleness diagnostics must not be merged. The 35.145% above is the share of **all** valid main-TEST h90
engagements with no new observation frame after the last kill `L` (the teamfight subset `T` is 35.27% on the same
condition). A strictly narrower condition — the pre-onset and endpoint states reading the **same** frame — holds
for 2,224 of 32,981 `T` rows (6.74%). Only the narrower condition removes frame-based changes between the two
query times; "no new frame after `L`" still permits a frame between the pre-state and `L`. Neither condition
removes elapsed-time and `×time` interaction changes (§8.6).

The 90 s cap was chosen from a pilot competing-event analysis in which cumulative first-objective incidence before
an additional kill was 11.93% at 60 s, 13.21% at 90 s and 13.75% at 120 s, i.e. 0.538 percentage points added
between 90 and 120 s, together with the sensitivity results above. An earlier delay analysis found that among
149,002 engagements with a first acquisition within 120 s, 55,734 (37.4%) had it after 60 s, so "every objective
follows within 60 s" is false; that is a conditional denominator and does not describe all engagements. No
published work fixes 90 s as optimal. No horizon was selected or justified from TEST results **in this follow-up**
(§8.3); that statement is about this study's protocol and does not certify that the whole earlier history of
horizon and definition development was test-naive — the pooled definition constants were estimated with the test
patch included (§4.3), and the main TEST patch had already been evaluated before these follow-ups (§7.5).

The small-\|ΔV\| band is a diagnostic stratum, not an error bar for `V` and not an exclusion threshold. That
roughly 12.7% of rows move by at most one percentage point is the reason we report sign stability under a changed
value model separately rather than treating labels as robust by default.

---

## 6. Inputs and predictors

### 6.1 The pre-engagement input contract

StateV2 combines the latest frame no later than the query time with events observed no later than that time:
total and current gold, level, experience, health and mana ratios, survival, lane and jungle CS, plus kill,
death, structure and objective events with elapsed-time features and selected interactions. Some
acquisition/death histories are proxies for buff state, not exact measurements.

| Schema | Columns before categorical expansion | Use |
|---|---:|---|
| StateV2 | 362 | State construction, including observation age |
| Primary `V` input | 361 | Excludes `snapshot_age`; 351 numeric plus 10 champion slots |
| Primary numeric `q` input | 352 | Excludes champion IDs and `snapshot_age`; includes `p_pre_V` |

The one-hot expanded dimension of `V` must not be confused with its 361 original columns. The 352 `q` columns map
to seven groups: prior win probability 1, time and observation age 2, economy and experience 60, combat and
survival 68, objectives 176, structures 24, health/mana/other 21. The group named "time" does not reintroduce the
excluded snapshot-age feature, and the mixed unknown-objective-team diagnostic sits in health/mana/other, so that
group is not purely health.

No `q` input contains the realised endpoint, the endpoint reason, the interval duration, the future participant
count, the post state, the actual `ΔV`, the final winner, or future-interpolated coordinates. Position is not a
`q` input: an earlier limited-sample comparison gave an AUC difference of +0.00312 with interval
[−0.00287, +0.00933], and decay/Kalman extrapolation did not beat last-observation carry-forward. That is a pilot
result, not a claim that position is unimportant, and retrospective position interpolation used for *detection*
is a separate matter from prediction inputs.

**Participant ordering.** StateV2 orders participants by `(team_id, participant_id)` and replaces any incoming
role-slot metadata with that order. Slot indices are therefore a deterministic identifier order. Where this draft
names slots `TOP/JG/MID/BOT/SUP`, that naming is a **frozen mapping assumption**, not a verified per-match lane
assignment (§8.5, and
[position_audit.md](../../outputs/manuscript_integration_20260916/position_audit.md)).

### 6.2 Predictor families

All families below read the identical 352 columns except the baseline, which is deliberately restricted.

- **PT (reinforced baseline).** A tensor-product spline of `p_pre_V` with game time (quantile knots, 48 design
  columns), regularised logistic, six `C` values and three calibrations selected on Q_SELECT. It is **not**
  `p_pre` alone: it can express time-varying and interaction structure in the pre-engagement estimate. Every
  "increment beyond initial advantage" statement in this draft is against PT.
- **Full logistic.** Regularised logistic on all 352 columns, six `C` values from 1e-4 to 10, tol 1e-8.
- **Full LightGBM.** Leaves 15/31/63 by minimum child samples 50/100, internal early stopping on a held-out
  match-hash subset of TRAIN using match-weighted Brier, seeds 7/42/123 averaged.
- **Plain MLP** and **residual MLP.** Widths 128/256/512 by dropout 0.1/0.3, AdamW (lr 1e-3, weight decay 1e-4),
  batch 512, at most 100 epochs, patience 10, refit on all eligible cohort TRAIN rows at the selected epoch
  count, seeds 7/42/123 averaged.

These five families constitute the matched-input track. They are **not** the eight-model CoG lineup:
FT-Transformer, TabNet, SAINT and the sequence/graph/event adapters were not executed on this target (§10.4).

---

## 7. Protocol

### 7.1 Fitting, calibration and selection

Preprocessing and base models are fitted on TRAIN; calibrators are fitted on Q_CAL; candidate selection uses
Q_SELECT by match-weighted Brier, then log loss, then candidate name. Raw, sigmoid and isotonic calibrations are
candidates for `q`; the `q` sigmoid is an unconstrained logistic calibration on clipped logits and must not be
confused with the positive-slope value-model calibrator.

### 7.2 Freezing and sealed evaluation

Each experiment writes a frozen manifest before any TEST or external prediction is read, and access to sealed
predictions is logged. Selections were never revised after seeing test results, and no contrast was promoted to
"primary" because it looked better on TEST.

### 7.3 Weighting and uncertainty

Within any reported cell, each represented match carries equal total weight, so a match contributing many
engagements does not dominate. Not every interval in this draft comes from the same resampling scheme, and the
three kinds must not be read interchangeably.

1. **Metric and contrast intervals (§8.1–§8.5).** 95% percentile intervals from paired bootstrap resampling of
   **matches** (1,000 replicates, seed 20260915), holding the fitted models fixed. They describe evaluation-sample
   variability for fixed models; they are not training-variability intervals and carry no multiple-comparison
   adjustment. Cells with fewer than 30 matches are marked sparse and carry no generalisation claim.
2. **Attribution intervals (§8.7).** Descriptive percentile intervals from resampling the 256 explained **rows**
   of a cell. They are not match-clustered, not population inference, and not comparable with the contrast
   intervals above.
3. **Definition-constant intervals (§4.1).** Bootstrap intervals of the boundary estimators `G` and `D` over the
   pooled kill-gap and kill-pair records, some steps using seeded computational samples. They quantify estimator
   variability for the constants, not predictive performance.

Intervals quoted from cited literature belong to those authors' designs and populations and are not comparable
with any of the three.

### 7.4 Verification performed by the experiment runs

Each run recorded contract tests before fitting and post-run checks afterwards: 117 checks with no failure for
the incremental-`q` run (independently re-verified and accepted, with 2,961 independent metric comparisons and
the results file hash unchanged), 115 for the matched-MLP run, 91 for horizon sensitivity, 91 for champion class,
53 for definition sensitivity, 36 for balanced SHAP, 24 for the value-mechanism decomposition. Failed
intermediate check runs and their corrections are preserved in each run's `audit_followup/`. These checks
establish arithmetic, provenance and implementation consistency inside their scope. They do not validate the
meaning of the label.

### 7.5 Exploratory status

Main TEST and the external cohorts had already been evaluated before these follow-ups were specified. Freezing
within a follow-up run controls selection inside that run; it does not restore confirmatory status to the study.
Every result in §8 is exploratory.

---

## 8. Results

All cells below are main TEST patch 15.16, primary cap h90, equal total weight per match, unless stated. `ΔBrier`
is `a − b`; negative favours `a`. Intervals are the paired match bootstrap of §7.3 unless a subsection says
otherwise (§8.7 uses row resampling).

"Predeclared primary" below means the contrast was fixed in this exploratory follow-up's own protocol and frozen
manifest before sealed predictions were read. **No external preregistration record exists**, and prior exposure to
the main TEST patch preceded these follow-ups (§7.5), so this is protocol-level predeclaration inside an
exploratory study, not a registered confirmatory test.

### 8.1 Increment beyond the reinforced baseline

**Teamfights `T` (32,981 rows, 24,020 matches).**

| Model | Brier | AUC |
|---|---:|---:|
| Constant | 0.249996 | 0.500000 |
| PT baseline | 0.229518 | 0.664795 |
| Full logistic | 0.228405 | 0.669415 |
| Full LightGBM | 0.228391 | 0.669125 |
| Prior cohort specialist (different budget/history) | 0.228171 | 0.670188 |

| Contrast (T, all rows) | ΔBrier [95%] | Reading |
|---|---|---|
| LightGBM − PT (predeclared primary) | −0.00113 [−0.00159, −0.00065] | LightGBM better |
| Logistic − PT | −0.00111 [−0.00172, −0.00051] | Logistic better |
| LightGBM − logistic | −0.00001 [−0.00049, +0.00045] | Interval contains 0 |
| New LightGBM − prior specialist | +0.00022 [−0.00031, +0.00076] | Interval contains 0 |

**Balanced teamfights `B40`** (`p_pre ∈ [0.40, 0.60]`; 4,949 rows, 4,570 matches — roughly 15% of T rows):

| Contrast (T, B40) | ΔBrier [95%] | Reading |
|---|---|---|
| LightGBM − PT (predeclared primary) | −0.00043 [−0.00144, +0.00060] | Inconclusive |
| Logistic − PT | −0.00157 [−0.00306, −0.00013] | Favours logistic in an exploratory secondary contrast |
| LightGBM − logistic | +0.00113 [−0.00002, +0.00245] | Interval contains 0 |
| New LightGBM − prior specialist | +0.00173 [+0.00036, +0.00315] | Prior specialist better in this cell |

**Other engagements `N` (130,595 rows, 54,182 matches).** PT 0.245040 (AUC 0.577865), logistic 0.242690
(0.598270), LightGBM 0.238688 (0.621140). LightGBM − PT is −0.00635 [−0.00682, −0.00592]; logistic − PT is
−0.00235 [−0.00280, −0.00193]; LightGBM − logistic is −0.00400 [−0.00444, −0.00353]. In `N` B40 (37,675 rows,
28,268 matches) LightGBM − PT is −0.00630 [−0.00703, −0.00555].

**Reading.** On the full teamfight population the full-input models carry information beyond a flexible function
of the pre-engagement estimate and the clock, but the margin is on the order of one thousandth of a Brier score.
In the balanced subset the predeclared primary contrast is inconclusive; the logistic contrast in the same
cell favours the full-input logistic, and we report it as one of several planned secondary contrasts without
multiple-comparison adjustment. **The B40 LightGBM result must not be generalised to "full-input models do not
help in balanced states"**: a different full-input family in the same cell points the other way. The prior
cohort specialist has a slightly better point estimate on full T; it comes from a different candidate pool,
budget and selection history and is not the winner of the matched Track A selection.

### 8.2 Matched-input learner families

Same 352 columns, same split, weighting, calibration and selection rules.

| Cohort / cell | Contrast | ΔBrier [95%] | Reading |
|---|---|---|---|
| T all | residual MLP − LightGBM (primary) | +0.00061 [+0.00012, +0.00110] | LightGBM better |
| T all | plain MLP − LightGBM | +0.00007 [−0.00046, +0.00057] | Inconclusive |
| T all | plain MLP − logistic | +0.00006 [−0.00032, +0.00043] | Inconclusive |
| T all | residual − plain MLP | +0.00054 [+0.00017, +0.00089] | Plain better |
| T all | plain MLP − PT | −0.00106 [−0.00173, −0.00038] | MLP better |
| T all | residual MLP − PT | −0.00052 [−0.00110, +0.00005] | Inconclusive |
| T B40 | plain MLP − LightGBM | −0.00100 [−0.00241, +0.00026] | Inconclusive (ΔAUC +0.016 [+0.002, +0.033]) |
| N all | residual MLP − LightGBM (primary) | +0.00110 [+0.00079, +0.00145] | LightGBM better |
| N all | plain MLP − LightGBM | +0.00109 [+0.00075, +0.00143] | LightGBM better |
| N all | plain MLP − logistic | −0.00291 [−0.00328, −0.00249] | MLP better |
| N B40 | residual MLP − LightGBM | +0.00185 [+0.00123, +0.00241] | LightGBM better |

Absolute T Brier point estimates, in order: prior specialist 0.228171, LightGBM 0.228391, logistic 0.228405, plain
MLP 0.228461, residual MLP 0.228999, PT 0.229518, constant 0.249996. The differences among the middle three are
smaller than their contrast intervals, so this ordering of point estimates is not itself a ranking result.
Validation-stage (Q_SELECT) point-estimate ordering agrees.
On the external cohorts the residual-MLP-versus-LightGBM contrast contains zero everywhere except NA1 16.13 `T`,
where LightGBM is better (+0.00285); the networks show no confirmed external teamfight gain.

The selected configurations used width 128 in all four cases. The selected residual MLP reached its best epoch at
epoch 1 for all three teamfight seeds, and its seed-to-seed raw Brier spread (0.2295–0.2300) is worse than the
plain MLP's (0.2287–0.2295). An early best epoch under a fixed learning rate, batch size and patience is
**consistent with** rapid overfitting but is not by itself proof of it; a wider optimisation search is a
different experiment with a different budget.

**Reading.** With information held identical, the teamfight contrasts among logistic regression, LightGBM and the
plain MLP each have intervals containing zero; the residual MLP is measurably worse than both LightGBM and the
plain MLP; and LightGBM is measurably better than both networks on the other-engagement cohort. An interval
containing zero means the specified contrast did not resolve a difference at this sample size and protocol:
**equivalence was not tested**, and these families are not shown to be interchangeable. This is a bounded
comparison over five tabular families, each with six candidate configurations (six `C` values for the logistic;
three widths × two dropout rates for the networks; three leaf counts × two minimum-child sizes for LightGBM) under
one optimiser setting, with three-seed averaging for the stochastic LightGBM and MLP families — the regularised
logistic is a single deterministic fit per candidate, not a three-seed ensemble. It does not establish that neural
networks are generally inferior on this target, and it does not cover the sequence, graph, event-token or modern
tabular-transformer families (§10.4).

### 8.3 Endpoint-cap sensitivity

The five h90-selected configurations were refit on h60 and h120 labels with hyperparameters fixed at the h90
winners and calibration re-selected on Q_SELECT. Row masks are identical across caps, so only the label changes.

| Cohort | Model | h60 | h90 | h120 |
|---|---|---:|---:|---:|
| T | LightGBM | 0.22781 | 0.22839 | 0.22825 |
| T | Logistic | 0.22789 | 0.22840 | 0.22831 |
| T | Plain MLP | 0.22775 | 0.22846 | 0.22845 |
| T | Residual MLP | 0.22847 | 0.22900 | 0.22891 |
| T | PT baseline | 0.22903 | 0.22952 | 0.22938 |
| N | LightGBM | 0.23898 | 0.23869 | 0.23880 |
| N | Plain MLP | 0.23982 | 0.23978 | 0.23942 |
| N | Residual MLP | 0.23990 | 0.23979 | 0.23985 |
| N | Logistic | 0.24308 | 0.24269 | 0.24255 |
| N | PT baseline | 0.24499 | 0.24504 | 0.24493 |

At both alternative caps: LightGBM − PT stays negative with intervals excluding zero on T
(−0.00122 [−0.00167, −0.00076] at h60; −0.00113 [−0.00163, −0.00063] at h120); LightGBM − logistic and plain MLP −
LightGBM contain zero; residual MLP − LightGBM stays positive with intervals excluding zero
(+0.00066 and +0.00067); and on N LightGBM stays ahead of both networks and of the baseline. In B40 at both caps,
LightGBM − PT contains zero on Brier while logistic − PT again favours logistic (−0.0020 at h60, −0.0017 at h120).

**Reading.** The particular comparisons listed above are retained at all three caps: LightGBM beats PT on `T`
with intervals excluding zero; the residual MLP stays behind LightGBM with intervals excluding zero; the
LightGBM-versus-logistic and plain-MLP-versus-LightGBM intervals keep containing zero; LightGBM stays ahead of
both networks and of PT on `N`. That is **not** a claim of unchanged learner ordering: the exact point-estimate
ranking moves, and at h60 the plain MLP has the lowest `T` point estimate (0.22775) while it is behind LightGBM at
h90. Three further constraints. The *labels* differ across caps — each cap induces a different target with its own
positive-class prevalence on the same rows — so absolute Brier levels are not comparable across columns, and the
lower `T` values at h60 do **not** establish that the 60 s label is an easier task; that would require a
difficulty analysis on a common target, which we did not run. The configurations were frozen at the h90 winners,
so this is not a search for the best configuration at each cap. And no horizon was selected or justified from TEST
results in this follow-up, so no claim of cross-horizon Brier superiority is made and h90 is not optimised from
the TEST ranking.

### 8.4 Definition sensitivity: constants from the training patch only

The pooled constants (13.7 s / 4,264 u) were replaced by estimates from TRAIN patch 15.14 alone (13.96 s / 4,285 u,
rounded under the same rule to 14.0 s / 4,285 u), and all 210,000 main matches plus the 21,190 external matches
were re-detected, re-labelled with the frozen `V` and out-of-fold adapters, and re-evaluated.

| Set | Matches changed | Exposure rows removed | Exposure rows added | Valid T rows old → dev (rebuilt) |
|---|---:|---:|---:|---|
| TRAIN | 3.30% (2,467 / 74,673) | 1.85% (3,690) | 1.15% (2,303) | 39,605 → 39,586 (1,972 rebuilt) |
| VALIDATION | 3.47% (2,593 / 74,748) | 1.91% (3,892) | 1.19% (2,419) | 41,315 → 41,299 (2,082) |
| TEST | 3.54% (2,143 / 60,579) | 1.95% (3,188) | 1.23% (2,013) | 32,981 → 32,968 (1,751) |
| KR 16.13 / NA1 16.13 | 4.05% / 4.01% | 2.4% | 1.5% | 5,202 → 5,186 / 5,312 → 5,307 |

Because `G` grew by about 0.3 s, adjacent clusters merge and the engagement count falls slightly (−0.7%).

| Cohort | Model | Old population Brier / AUC | Dev population Brier / AUC |
|---|---|---|---|
| T | Frozen LightGBM | 0.22839 / 0.6691 | 0.22837 / 0.6691 |
| T | Frozen logistic | 0.22840 / 0.6694 | 0.22834 / 0.6695 |
| T | Frozen PT | 0.22952 / 0.6648 | 0.22950 / 0.6648 |
| T | Refit LightGBM | — | 0.22830 / 0.6692 |
| N | Frozen LightGBM | 0.23869 / 0.6211 | 0.23868 / 0.6212 |
| N | Refit LightGBM | — | 0.23890 / 0.6199 |

Paired contrasts on the dev population: T refit − frozen LightGBM **−0.00007 [−0.00023, +0.00010]** (inconclusive);
T refit − frozen logistic −0.00001 [−0.00005, +0.00004]; N refit − frozen **+0.00023 [+0.00011, +0.00034]**
(refit slightly worse); KR 16.13 T refit − frozen +0.00052 [+0.00010, +0.00099] (refit worse); other external
cells contain zero. The frozen LightGBM − PT contrast is preserved (−0.00113 [−0.00155, −0.00071] on the dev
population). Rebuilt rows — engagements that merged or split — are harder than common rows (T Brier 0.232 versus
0.228, AUC 0.655 versus 0.669) but are only about 5% of T.

**Reading.** The size of the definition's test-patch dependence is now measured rather than assumed: a
training-patch-only constant pair differs by 0.3 s and 21 u and changes 3.3–4.4% of matches. The metric
consequences must be stated cell by cell rather than as one global bound. On the main TEST teamfight cell the
frozen LightGBM Brier moves 0.228391 → 0.22837 between the old and dev populations, a particular small descriptive
change across two different row populations; the corresponding main-TEST `N` frozen change is of the same order.
External frozen cells move further: KR 16.15 `N` 0.24062 → 0.23996 (≈0.0007) and KR 16.13 `T` 0.23510 → 0.23476.
These are Brier comparisons across different row populations, not a bound on every metric: even main-TEST
AUC changes reach about 0.00075 in the recorded frozen-model cells. For the paired refit-versus-frozen
contrasts on the dev population, `T` is inconclusive (−0.00007 [−0.00023, +0.00010]) while `N` (+0.00023) and
KR 16.13 `T` (+0.00052) are **worse** with intervals excluding zero. Four qualifications are required. First, this
does **not** remove the fact that the primary definition was estimated with the test patch included; that
dependence is retained as a limitation (§10.5), and a single alternative constant pair bounds its consequence in
the cells measured here rather than eliminating it.
Second, `D = 4,285 u` lies **outside** the pooled 95% interval [4,256.0475, 4,273.9677], although `G = 14.0 s`
lies inside the temporal stability plateau and the `G` interval. Third, the inconclusive T contrast is not
equivalence, and the small worsening on N and on one external cell is not dismissed as refit noise: only one refit
per arm was run, so refit-to-refit variability is not estimated. Fourth, the two populations' absolute metrics are
a descriptive comparison of different row sets and are not the "pure effect" of changing the definition. A
15.14+15.15 pooled estimate was not produced, and other detector switches were not varied.

### 8.5 Champion class, slot pairs and identity

Four feature blocks were stacked on the frozen 352 columns: Riot tag-derived slot classes (120 columns),
class-conditional team state aggregates (126), slot-pair and cross-team class matchup combinations (396), and
champion-identity one-hot (1,710). Two draft-only arms used composition without state.

| Arm | Width | T LightGBM Brier | Δ vs base [95%] | N LightGBM Brier | Δ vs base [95%] |
|---|---:|---:|---|---:|---|
| base (352) | 352 | 0.22839 | — | 0.23869 | — |
| + tags | 472 | 0.22842 | +0.00003 [−0.00009, +0.00016] | 0.23867 | −0.00002 [−0.00019, +0.00014] |
| + class state | 598 | 0.22872 | +0.00033 [+0.00004, +0.00064] | 0.23875 | +0.00006 [−0.00015, +0.00028] |
| **+ class pairs (predeclared primary)** | 994 | 0.22852 | **+0.00013 [−0.00009, +0.00035]** | 0.23895 | **+0.00027 [+0.00004, +0.00047]** |
| + identity | 2,062 | 0.22841 | +0.00002 [−0.00014, +0.00018] | 0.23884 | +0.00015 [−0.00004, +0.00034] |
| + class pairs + identity | 2,704 | 0.22848 | +0.00009 [−0.00013, +0.00033] | 0.23894 | +0.00026 [+0.00001, +0.00049] |
| draft class only (no state) | 534 | 0.24998 (AUC 0.508) | +0.0216 | 0.24994 (AUC 0.510) | +0.0112 |
| draft identity only (no state) | 1,710 | 0.25000 (AUC 0.503) | +0.0216 | 0.24960 (AUC 0.523) | +0.0109 |

Under logistic regression every added block is significantly worse, monotonically in width (T class pairs
+0.00259, identity +0.00675, both +0.00760). The logistic `C` was fixed at the 352-column winner and not
re-searched, so this is a design limitation of the arm rather than an independent finding about class
information. On external cohorts the primary contrast contains zero on T everywhere, and is worse on N for KR
16.13 (+0.00125 [+0.00072, +0.00178]) and NA1 16.13 (+0.00063 [+0.00003, +0.00120]).

**Positional caveat (mandatory reading).** These pair features are built from ten fixed slot columns in
participant order. No `role_slots`, `teamPosition` or per-match role join is read anywhere in the arm. They are
therefore **slot-index pairs under a positional proxy assumption**, and a feature named `cc_matchup_TOP` is not a
verified same-role matchup. In the stored main corpus, the cached role-slot ordering disagrees with the StateV2
participant order in 1.69% of matches (3,548 of 210,000: TRAIN 1,255 / 1.68%, VALIDATION 1,251 / 1.67%, TEST
1,042 / 1.72%); agreement in the remaining 98.3% does not establish that slot *k* is the corresponding lane role,
because the cached map itself is weak supervision derived from `teamPosition`/`individualPosition` with
participant-ID fills. Reported class frequencies are frequencies **by participant-order slot**.

**Reading.** In this setup, with these learners and budgets, we detect no benefit from class, slot-pair or
identity features for LightGBM, and a small worsening in two `N` cells. That is not a demonstration of
equivalence, and it is not evidence that no learnable composition information exists. Specifically: draft-only
AUC near 0.50 is a null result under the tested representation and learner budget, not proof of zero information;
the logistic degradation is confounded by a fixed regularisation constant as width grows; the LightGBM budgets
were limited (one N arm reached the 1,000-iteration cap); overfitting is one possible explanation for the flat
tree results but was not isolated; and graph representations (Track B), finer class taxonomies and other learners
were **not** tested and are not ruled out by these results. Tag, identity and class-state findings stand only
within their representation limits; true role-matchup conclusions remain unestablished.

**Numerical reproduction caveat.** The base LightGBM arm reproduces the parent incremental-`q` winner exactly
(maximum difference 0). The base logistic arm does not: raw predictions differ by about 6e-7 (T) to 3.4e-5 (N)
because of lbfgs path differences under a different design-matrix memory order, and after `N` isotonic
calibration the maximum prediction difference reaches 0.0118 at step boundaries. All arms inside this run were fit
on one code path, so within-run contrasts are internally consistent; but same-run comparisons are not
automatically immune to numerical uncertainty, and small logistic differences elsewhere should be read with this
in mind.

### 8.6 What the label mechanism actually is

Because the selected `V` is a logistic model with an identity (raw) calibration, the label is exactly the sign of
a linear score difference in the model's transformed feature space:

```
Δlogit = βᵀ( z(S(e)) − z(S(t_pre)) ),   Y = 1[ Δlogit > 0 ]
```

where `z` is the actual preprocessing output, including engineered interactions. This identity was verified on
every stored valid row of main TEST (163,576) and VALIDATION (203,170): parent `p_pre` and `p_post` reproduced
exactly, with a maximum block-sum deviation from `Δlogit` of 3.042e-14 (bound 4e-14), a maximum logit-difference
deviation of 6.231e-12 (bound 7e-12), and `1[Δlogit > 0] = Y` on 366,746 of 366,746 rows. The frozen `V`
preprocessing expands its 361 input columns to 1,733 transformed columns, of which the `champions__` group holds
1,382 (`outputs/v_mechanism_20260916/results.json` → `group_sizes/champion_identity`; grouping rule in
`scripts/vd20260916_common.py:column_group`); 351 numeric columns plus that block sum to 1,733. Those champion
columns are identical at both endpoints and contribute exactly 0.

| Block (main TEST T, 32,981 rows; mean \|Δlogit\| 0.961, mean \|ΔV\| 0.113) | Mean \|contribution\| | Share of Σ\|contribution\| |
|---|---:|---:|
| Combat and survival | 0.451 | 35.5% |
| Economy and experience | 0.359 | 28.2% |
| Objectives | 0.240 | 18.9% |
| Structures | 0.123 | 9.6% |
| Health/mana/other | 0.091 | 7.2% |
| Time only (`time_minutes`, squared) | 0.008 | 0.7% |
| Champion one-hot | 0 | 0 |

Time-related columns including `×time` interactions account for 19.9% on a separate accounting (blocks and
time-relatedness overlap, so these two figures must not be summed). The largest block agrees in sign with
`Δlogit` in 95.9% of rows; in 27.7% a single block exceeds `|Δlogit|` and is partly cancelled by others. For `N`
(130,595 rows) combat and survival dominates more strongly at 45.2%.

Strata: in the 2,224 of 32,981 T rows (6.7%) whose **pre and post states read the same observation frame**, the
frame-based blocks contribute 0 and the label is carried by event columns together with elapsed-time terms
(combat 72.2%, objectives 13.6%, structures 13.7%, time-only 0.5%, and 25.0% on the overlapping time-related
accounting). Event columns alone therefore do **not** determine the label even in this stratum. This stratum is
strictly narrower than the "no new frame after the last kill" diagnostic of §5.4, which does not remove
frame-based change between the pre-state and `L`. In the small-\|ΔV\| stratum (≤0.5 pp, 3,251 rows) the structures
share rises to 19.3%, against 9.6% over all T rows and 7.9% in the \|ΔV\| > 2 pp stratum. After 30 minutes,
objectives reach 21.9% and time-related columns 27.5%.

**Reading.** The label is a learned linear exchange score rather than a hand-weighted one: the coefficients come
from fitting match outcome, not from a design table. That is a statement about where the weights come from and
about the explicit state and window definition, not about decomposability as such — a hand-weighted event sum can
also be decomposed exactly into its terms. Four boundaries. (i) Static additive champion terms cancel from the
sign and from `Δlogit`, but champions can still affect `p_pre`, and therefore the magnitude of `ΔV` through the
sigmoid, and can affect the realised state change dynamically; cancellation is not "champions do not matter".
(ii) The largest **mean absolute column contributions** over the collected rows are `blue_kills_x_time` 0.110 and
`red_kills_x_time` 0.102, followed by per-participant `totalGold_norm`, `level_norm` and `laneCS_norm` at
0.04–0.11. These are contribution magnitudes, **not** coefficients: the fitted `β` for those two columns are
−1.0564 (blue) and +0.9901 (red), and the main kill terms likewise differ in sign by side. Under `C = 0.01` ridge
regularisation the columns are collinear, so we do not read individual signs or coefficients as directional
statements about kills. (iii) This is an arithmetic decomposition of a fitted linear score, not a causal decomposition.
(iv) `V(S(L))` was not stored, so the pre→last-kill→endpoint split of `ΔV` has **not** been computed; only the
pre/post decomposition is verified.

### 8.7 Explaining the selected predictor, full versus balanced states

Exact group Shapley values were computed for the frozen selected `q` over seven groups (128 coalitions), with a
cohort TRAIN background of 128 rows and 256 hash-selected rows per cell. Additivity holds within 4e-16, and
`Σφ + base` reproduces the parent sealed prediction for the same row to <4e-16.

| Group (T, selected LightGBM) | Mean \|φ\|, all T | Share | Mean \|φ\|, B40 | Share |
|---|---:|---:|---:|---:|
| Prior win probability `p_pre` | 0.0971 [0.090, 0.105] | 61.7% | 0.0216 [0.020, 0.023] | 27.7% |
| Economy and experience | 0.0189 [0.017, 0.021] | 12.0% | 0.0190 [0.017, 0.021] | 24.4% |
| Objectives | 0.0151 [0.014, 0.016] | 9.6% | 0.0135 [0.012, 0.015] | 17.3% |
| Combat and survival | 0.0127 [0.011, 0.014] | 8.1% | 0.0108 [0.010, 0.012] | 13.9% |
| Structures | 0.0075 | 4.8% | 0.0066 | 8.4% |
| Health/mana/other | 0.0061 | 3.9% | 0.0064 | 8.2% |
| Time and observation age | 0.0000 | 0.0% | 0.0000 | 0.0% |

The two cells overlap in 2 rows; mean `p_pre` is 0.518 and 0.503 respectively. The **absolute** contribution of
the non-`p_pre` groups is essentially unchanged between cells (economy 0.0189 versus 0.0190); what changes is the
absolute contribution of `p_pre` (0.0971 → 0.0216). The relative rise of objective and economic shares in
balanced states therefore mostly reflects a **diminished `p_pre` contribution**, not an increased absolute or
causal effect — and part of that decrease is definitional, because the B40 cell is defined by restricting
`p_pre`, which reduces its variation.

The plain MLP, whose T Brier differs from the tree's by an amount this comparison did not resolve
(+0.00007 [−0.00046, +0.00057]), distributes attribution very
differently (all-T: objectives 21.0%, combat 20.7%, structures 20.0%, economy 17.2%, `p_pre` 14.8%; B40 `p_pre`
4.1%). Two models whose accuracy difference this protocol did not resolve therefore give different attribution
profiles, which is a direct caution against reading these shares as properties of the game. For `N` the selected tree moves the same
way (`p_pre` 27.7% → 6.5%, economy 21.8% → 31.4%, combat 19.4% → 22.3%, health/mana/other 14.9% → 20.3%).

**Boundaries.** The explanation target is the frozen calibrated `q` output, not `V` and not `ΔV`. The
interventional group masking replaces columns with background values without recomputing `p_pre` from the masked
state and without restoring derived `×time` relationships, so masked inputs can be off-manifold and correlated
groups are not disentangled. The exact coalition sum is exact **conditional on the sampled rows and background**;
it is not full-population inference. The bootstrap intervals resample the 256 explained rows descriptively and
are not match-clustered population intervals. Full-versus-B40 is a comparison of different rows with no paired
interval. The zero for the time group means the selected tree rarely split on the two time columns, not that time
information is absent — `×time` interactions live inside the other groups. None of this identifies causally
necessary actions, player skill, or what a team should have done.

---

## 9. Interpretation

**There is predictable structure before a fight beyond knowing who is ahead — and it is small on teamfights.**
Against a baseline that already sees the pre-engagement win estimate and its interaction with game time, the
full-state models gain about 0.001 Brier on teamfights and about 0.006 on other engagements. The `N` gain is the
larger one and its interval is further from zero. We do not present the `T` gain as practically significant; we
present it as measurable under this protocol. We also do not explain the `T`/`N` difference: the hypothesis that
smaller engagements are less determined by global match position and more by local state is untested here — the
two cohorts differ in scale definition, row count, prevalence and baseline level simultaneously, and no mechanism
comparison was run.

**Adding composition features to the state gave no detected benefit; the state-versus-composition question is
not settled.** The selected predictor leans on the pre-engagement estimate, then on economy, objectives and combat
state (§8.7); adding class, slot-pair or identity features on top of the state produces no detected benefit for
LightGBM in this setup, with the primary interval containing zero and two `N` cells slightly worse (§8.5). This is
a non-detection under one representation, learner set and budget, not a demonstration that composition carries no
information on this target. The hypothesis that state features already carry whatever composition contributes is
*consistent with* these results but is not established by them; it competes with overfitting, budget and
representation explanations that we did not separate, and graph representations and finer taxonomies were not
tested.

**Under a fixed input contract, the tested family differences on teamfights were mostly unresolved — with one
exception.** On teamfights the logistic, LightGBM and plain-MLP contrasts have intervals containing zero, while
the residual MLP is measurably worse than both, and on other engagements LightGBM is measurably better than both
networks (§8.2); the same particular comparisons are retained at 60 and 120 s caps, though the exact point-estimate
ranking moves (§8.3). Unresolved contrasts are not evidence that learner choice is immaterial, and no equivalence
test was performed. Any future claim that a new architecture helps on this target should be made against the same
input contract, the same reinforced baseline and the same match-bootstrap protocol.

**The label is legible.** This label decomposes exactly into per-column terms of a fitted linear score (§8.6):
combat and survival columns carry the largest mean absolute contribution, economy and experience follow, then
objectives. A hand-weighted exchange score can also be decomposed exactly into its terms, so exact decomposability
is not what distinguishes the two; what distinguishes them is that the weights here are fitted against match
outcome rather than set by a design table, and that the state and window entering the difference are defined
explicitly. In the 6.7% of teamfight rows whose pre and post states read the same frame, frame-based columns
contribute nothing and the sign is carried by event columns **together with** elapsed-time terms — not by events
alone, and this stratum is narrower than the "no new frame after the last kill" diagnostic (35.27% of teamfight
rows, 35.145% of all valid rows; §5.4).
Legibility is the main methodological argument for the design. It is not an argument that the label is *correct*:
a legible measurement can still be a poor measurement of the intended construct.

**Balanced states shift the attribution profile; the incremental question there is partly unresolved.** In
`p_pre ∈ [0.40, 0.60]` teamfights the predeclared LightGBM-versus-PT contrast is inconclusive
(−0.00043 [−0.00144, +0.00060]) on 4,949 rows, but the full-logistic-versus-PT contrast in the same cell excludes
zero (−0.00157 [−0.00306, −0.00013]) as an exploratory secondary comparison without multiple-comparison
adjustment, so the remaining state signal in balanced fights is **not** shown to be unresolvable — one full-input
family did resolve a difference there. What is well supported is the attribution change: the prior estimate's
absolute contribution falls while the other groups' absolute contributions stay roughly constant, part of which
follows from the cell being defined by restricting `p_pre` (§8.7).

---

## 10. Limitations

### 10.1 Measurement validity

`Y` is defined by `V`. A high `q` score therefore demonstrates that pre-state information predicts a
model-defined quantity; it does not demonstrate agreement with any expert judgement of fight victory. No
independent outcome criterion has been applied: an anonymised 120-case review packet was prepared but no human
judgements were collected, and human review is outside the agreed scope, so this remains an open validity gap
rather than a blocking requirement. Automated checks and AI-assisted review are not reported as expert semantic
validation. Alternative-valuation evidence exists and is reported where supported — refitting `V` with `C = 0.1`
changes 1.09% of T labels while a reduced-input economic `V` changes 20.09%, and removing the 176 explicit
objective columns changes 5.358% of T labels — but every one of those alternatives is still a learned model that
induces its own model-defined label.

### 10.2 Attribution to the fight

`ΔV` is measured on the global match state. It includes farming elsewhere, map-wide developments, time
progression and multiple conversions inside the window. Without a comparable non-engagement interval or a causal
design, no part of this draft establishes that the engagement caused the observed change. We use "associated
with" wording throughout and do not report causal exchange prices for objectives.

### 10.3 Retrospective population and live operation

Engagement existence, the endpoint's competing events, `S_next` and the `T`/`N` scale label are all retrospective.
The evaluation replays historical information under a cutoff contract; it is not a demonstration of live
operation. Live ingestion, onset detection, latency and automatic cohort routing are unbuilt. Frame-based
resources can be stale between one-minute snapshots even when event counters update: 35.27% of main-TEST `T` rows
have no new frame after the last kill (35.145% over all valid main-TEST h90 rows), and in the narrower 6.74% of
`T` rows whose pre and post states read the same frame, no frame-based column can change at all between the two
query times. The two diagnostics are different populations and are used for different statements (§5.4, §8.6).

### 10.4 Unexecuted comparisons

The matched track covers five tabular families. FT-Transformer, TabNet and SAINT were not run on this target, and
the sequence, graph, cross-attention and layered-fusion adapters from the earlier lineup were not closed,
validated or fitted. The champion-class study did not test graph representations, finer class taxonomies, neural
arms on the class features, or a re-searched regularisation constant for the widened logistic arms. No claim in
this draft rules any of them out.

### 10.5 Definition and generalisation

The pooled constants were estimated with the test patch included; §8.4 bounds but does not remove this, and the
TEST-dependence of the definition is retained as a limitation rather than resolved. A training-patch-only
re-derivation moves `D` outside the pooled interval; the main-TEST T frozen LightGBM Brier changes from
0.228391 to 0.22837, while the external KR 16.15 N frozen Brier changes by roughly 0.0007. These Brier examples
do not bound AUC changes. The refit-versus-frozen
contrast is worse on `N` (+0.00023) and on KR 16.13 `T` (+0.00052), and only one refit per arm was run. Other detector switches were not varied, and a 15.14+15.15 pooled
estimate was not produced. External cohorts show calibration degradation, unseen champions, missing objective
channels, small subsets and prior access history; they do not establish cross-patch operation without adaptation.
Genuinely untouched confirmation data remain the main outstanding requirement.

### 10.6 Statistical reading

The metric and contrast intervals of §8.1–§8.5 resample matches with fitted models held fixed, with 1,000
replicates, no multiple-comparison adjustment and no training-variability component. The attribution intervals of
§8.7 are a different object: they resample the 256 explained rows of a cell descriptively and are neither
match-clustered nor population inference, so they must not be compared with the contrast intervals. The `G`/`D`
intervals of §4.1 are boundary-estimator bootstraps over the pooled kill records, and intervals reported in cited
work belong to those authors' designs (§7.3). Several planned secondary contrasts are reported; readers should
treat individual marginal results (for example logistic − PT in B40) accordingly. Intervals that contain zero are
reported as inconclusive and never as equivalence; no equivalence or non-inferiority procedure was run anywhere in
this draft, so no "no difference" conclusion is available for any contrast. Numerical reproduction differences
(§8.5) mean small logistic-path differences are not always attributable to modelling choices.

### 10.7 Scope of the population

The population is kill-conditioned. Kill-free encounters, zoning and vision control are excluded by construction,
and the meta-review question about kill-free teamfights is answered here by scope limitation rather than by
coverage. The corpus is a single high-skill Korean bracket for the main patches plus a small NA1 cohort; no
player-level, tier-level or region-general claim is made.

---

## 11. Conclusion and remaining work

### 11.1 What this draft establishes

On a 210,000-match patch-split corpus with 566,104 valid engagement rows, pre-engagement state predicts the
direction of change in a frozen match-win value model beyond a reinforced baseline of the pre-engagement estimate
and game time, by about 0.001 Brier on teamfights and 0.006 on other engagements. Under a matched 352-column
input contract, the teamfight contrasts among logistic regression, LightGBM and the plain MLP are inconclusive
(intervals contain zero; equivalence was not tested), the residual MLP is measurably worse, and trees are
measurably better on other engagements; those particular comparisons are retained at 60, 90 and 120 s caps, while
the exact point-estimate ranking moves. Re-deriving the engagement constants from the training patch alone changes
3.3–4.4% of matches; the main-TEST frozen teamfight Brier moves 0.228391 → 0.22837 and the refit-versus-frozen
teamfight contrast is inconclusive, but the same contrast is worse on other engagements and on external KR 16.13
teamfights, and the definition's original test-patch dependence remains. Champion class, slot-pair and identity
features show no detected benefit on this target in this setup, which is a non-detection rather than a
demonstrated absence of composition information. The label reduces exactly to the sign of a linear score
difference in which static champion terms cancel, and the selected predictor's attribution shifts from the prior
estimate toward economy, objective and combat groups in balanced states.

### 11.2 What remains open

| Open item | Status |
|---|---|
| Independent semantic validation of the label | Not performed; outside current scope by user decision, not a blocking prerequisite imposed here |
| Confirmatory evaluation on untouched future data | Not available; all current follow-ups are exploratory |
| Representation track (sequence, graph, cross-attention, layered fusion) and modern tabular transformers | Specified in part, not executed |
| `V(S(L))` intermediate decomposition of `ΔV` | Not computed; `V(S(L))` was not stored |
| Live onset detection, routing, latency | Not built |
| Definition estimation from 15.14+15.15; other detector switches | Not produced |
| Anonymised repository, licence, reproduction instructions, data policy | Not prepared |
| Author block, affiliations, acknowledgments, ethics/permission statements | Deliberately omitted from this draft |
| Manuscript build (LaTeX/PDF), figures, journal formatting | Not attempted in this draft |

This draft is therefore an **integrated working draft awaiting the remaining scientific and submission tasks**,
not a submission-ready manuscript.

---

## References

Full entries with verified metadata and per-source support limits are in [references.bib](references.bib); the
boundaries are restated in [claim_evidence_ledger.md](claim_evidence_ledger.md) §4 and derive from
[citation_audit.md](../../outputs/manuscript_integration_20260916/citation_audit.md).

- **[Halfaker2015]** A. Halfaker, O. Keyes, D. Kluver, J. Thebault-Spieker, T. Nguyen, K. Shores, A. Uduwage and
  M. Warncke-Wang, "User session identification based on strong regularities in inter-activity time," *WWW 2015*,
  pp. 410–418. DOI [10.1145/2736277.2741117](https://doi.org/10.1145/2736277.2741117). *Supports: reading an
  activity boundary out of an inter-event-time distribution. Does not supply `G = 13.7 s` or our KDE estimator.*
- **[Hodge2021]** V. J. Hodge, S. Devlin, N. Sephton, F. Block, P. I. Cowling and A. Drachen, "Win prediction in
  multiplayer esports: live professional match prediction," *IEEE Transactions on Games*, 13(4), 2021,
  pp. 368–379. DOI [10.1109/TG.2019.2948469](https://doi.org/10.1109/TG.2019.2948469). *Supports: live MOBA
  win prediction as an established setting with data-availability limits. Different game and protocol; its
  accuracies are not ours.*
- **[Kim2020]** D.-H. Kim, C. Lee and K.-S. Chung, "A confidence-calibrated MOBA game winner predictor,"
  *IEEE CoG 2020*, pp. 622–625. DOI
  [10.1109/CoG47356.2020.9231878](https://doi.org/10.1109/CoG47356.2020.9231878). *Supports: motivation to
  assess probability calibration and uncertainty in LoL win prediction. Its uncertainty-aware loss and
  architecture are not implemented here.*
- **[Lundberg2017]** S. M. Lundberg and S.-I. Lee, "A unified approach to interpreting model predictions,"
  *NeurIPS 2017*, pp. 4765–4774. *Supports: additive model-output attribution principles. Our grouping,
  background and masking are operational choices; the result is not causal.*
- **[Maymin2021]** P. Z. Maymin, "Smart kills and worthless deaths: eSports analytics for League of Legends,"
  *Journal of Quantitative Analysis in Sports*, 17(1), 2021, pp. 11–27. DOI
  [10.1515/jqas-2019-0096](https://doi.org/10.1515/jqas-2019-0096). *Supports: the time/state logistic
  win-probability lineage and event valuation by estimated-probability change. Does not validate our schema,
  label, endpoint or any causal reading.*
- **[Schubert2016]** M. Schubert, A. Drachen and T. Mahlmann, "Esports analytics through encounter detection,"
  *MIT Sloan Sports Analytics Conference*, 2016, paper 1458. *Supports: spatio-temporal encounter segmentation
  and outcome modelling as a precedent, and the size of the kill-free encounter population (18,744 / 23,110 =
  81.1%). Does not justify our `G`, `D`, `B`, `R` or our kill-conditioned population.*
