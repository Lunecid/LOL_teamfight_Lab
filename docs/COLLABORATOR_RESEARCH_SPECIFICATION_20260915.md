# League of Legends Engagement Prediction Research Specification

Methods results and planned journal extension

Prepared for academic collaboration by Seongeun and the research team

Research status as of 15 September 2026

## 1 Purpose and current conclusion

We study whether the global game state observed before an engagement can predict which team will improve its estimated probability of winning the match during a defined engagement and follow-up interval. Our current system connects engagement construction, strictly pre-engagement state extraction, a separately trained match-win value model, outcome-label generation, engagement prediction, and model explanation.

The full pipeline has been executed on a designated corpus of 210,000 matches with patch-based splits, followed by external patch and region evaluations. Teamfight and other-engagement models, role-based representations, SHAP explanations, valuation-model sensitivity analyses, and an explicit objective-feature ablation have been completed. The main test teamfight predictor achieves AUC 0.6702 for the direction of model-estimated win-probability change. The separate match-win value model achieves AUC 0.8586 for eventual match victory. These evaluate different targets and must not be compared as two estimates of the same performance.

Our evidence supports predictive modeling of an explicitly defined strategic outcome. It does not establish that the generated label is an independent ground-truth judgment of fight victory, that the engagement caused the entire measured change, or that the pipeline has been deployed live. The complete CoG model lineup has not yet been retrained on the new target. Balanced-state comparisons motivated by the collaboration email are specified but not executed. [E01–E07]

This document separates completed results from plans, explains why each design was adopted, and states the limits of each supporting source. All numerical examples identified as hypothetical are illustrations, not observations. Performance numbers are rounded from retained reports; exact predictions, manifests, and validation records are retained locally. The accompanying feature dictionary provides the full ordered input schema without match records or account identifiers.

## 2 Research lineage and changes since CoG

The authoritative CoG 308 paper reported a binary positive-exchange outcome derived from a richer manually constructed event score. The earlier paper 118 and an extended technical report contain additional analyses and diagnostic models. They are separate versions. We use the published 308 comparison set when describing the CoG benchmark and identify 118-specific diagnostics separately. The extended report's list of more than 25 supported architectures is not a list of 25 published comparison results. [E08–E09]

### 2 1 The original exchange label

Each post-onset event received a sign indicating the benefiting team, an event value, and an importance prior. A softmax with beta equal to 2 converted the priors into within-window weights. The weighted signed sum was thresholded at zero to obtain a binary label. The event-value and prior coefficients were fixed, domain-informed choices rather than learned exchange rates. Thus the target was information-rich in its construction, but the predictor still performed binary classification. [E08]

$$Y_i^{\mathrm{CoG}}=\mathbf{1}\!\left[\sum_{u\in i}\alpha_u\sigma_u v_u>0\right],\qquad \alpha_u=\frac{\exp(2p_u)}{\sum_{v\in i}\exp(2p_v)}.$$

| Descriptor in CoG Table I | Event value coefficient | Importance prior coefficient |
|---|---:|---:|
| Kill indicator | 1.00 | 0.25 |
| Normalized shutdown | 1.60 | 0.30 |
| Normalized streak | 0.35 | 0.15 |
| Normalized assists | 0.20 | 0.10 |
| Normalized bounty | 0.30 | 0.20 |
| Objective tier | 1.10 | 0.35 |
| Lane priority | 0.25 | 0.15 |
| Special event bonus | s(u) | s(u) |

Descriptor normalization and special-event handling belong to the original label implementation. Special kill markers contribute their bonus rather than a second full kill. The table documents the old design; these coefficients are not used in the current Delta V label. The concern raised in the collaboration email therefore led us to change the valuation formulation, rather than simply tune the old coefficients for a better test score.

The original paper clips descriptors to [0,1], log-scales shutdown/bounty, divides streak by 10 and assists by 4, and specifies objective tiers including plate 0.35, dragon 0.75, inhibitor/Atakhan 0.85, and Baron/Elder/soul/Nexus 1.0. Its special bonuses include first blood 0.20 and multi-kill 0.25, with ace handling specified separately. Near-zero ties below 1e-8 use a deterministic coin flip in that older rule. This differs from the current exact-zero non-improvement convention.

### 2 2 What changed and what was retained

| Component | CoG 308 | Current Delta V study |
|---|---|---|
| Outcome valuation | Manually weighted signed event exchange | Change in a learned state-based match-win estimate |
| Prediction target | Sign of exchange score | Sign of Delta V; continuous change retained for analysis |
| Temporal kill gap | Fixed 18 seconds | Estimated gap about 13.7 seconds |
| Spatial separation | Fixed 4000 game units | Estimated boundary about 4264 units |
| Onset offset | First kill minus 10 seconds | First kill minus 15 seconds |
| Presence radius | 1800 units in published definition | 1600 units under the current operational definition |
| Outcome endpoint | Original 30–60 second onset-based rule with ace truncation | Event-based stopping with a 90 second cap after the last kill |
| Predictive information | Multiple pre-onset representations | Current primary q uses 352 numeric pre-state features |
| Population and split | 206442 retained matches under original filtering | All 210000 designated matches retain patch roles; task-specific eligibility is reported |

These changes alter the cohort, target, and information representation. Differences between old and new AUC values cannot be interpreted as a measured improvement from one isolated intervention.

### 2 3 Published CoG performance

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

These are the original paper-reported three-seed figures on patch 15.16 under the old exchange label. The released-code correction record reports approximately 0.669 for LightGBM after localization corrections. Published and corrected-release results are distinct records. The old MLP comparison used the same approximately 2980-dimensional engineered representation as LightGBM; a mistakenly routed macro-sequence diagnostic was corrected in the released code. The original pipeline also used per-model subsampling in recorded runs, so its execution budget is not evidence that the new full-corpus comparison has already been performed. [E08–E09]

## 3 Quantities and the research estimand

| Symbol | Definition | Role |
|---|---|---|
| W | Actual final Blue match victory | Supervision and evaluation for V |
| S(t) | State constructed from information observed no later than t | Input to the value model |
| V theta | Estimated probability of eventual Blue match victory | Valuation function |
| E | Retrospectively identified eligible kill-conditioned engagements | Overall study population |
| T | Engagements with recorded cluster participation of at least four per team | Primary teamfight subgroup |
| N | Known-scale engagements in E outside T | Other-engagement subgroup |
| Delta V | Value at the defined endpoint minus value before onset | Continuous operational outcome |
| Y | Indicator that Delta V is positive | Current q training label |
| q phi | Probability that Y is one given pre-engagement information | Engagement prediction |

$$p_t=V_\theta(S(t))\approx P(W=1\mid S(t)),\qquad q_\phi(X_i^{\mathrm{pre}})\approx P(Y_i=1\mid X_i^{\mathrm{pre}}).$$

The current q target is improvement direction, not the size of the improvement. A q output of 0.70 means an estimated 70 percent probability of a Blue-improvement label under this definition. It is neither a 70 percent final match-win estimate nor a gain of 70 percentage points. Because the population is located retrospectively, our present estimand is conditional on an engagement having been identified; occurrence detection and live routing are separate tasks.

## 4 Data coverage and validation separation

### 4 1 Main corpus

| Patch role | Raw matches | Detected engagements | Valid Delta V rows | T rows | N rows |
|---|---:|---:|---:|---:|---:|
| TRAIN 15.14 | 74673 | 199480 | 199358 | 39605 | 159753 |
| VALIDATION 15.15 | 74748 | 203292 | 203170 | 41315 | 161855 |
| TEST 15.16 | 60579 | 163680 | 163576 | 32981 | 130595 |
| Total | 210000 | 566452 | 566104 | 113901 | 452203 |

The same valid-row masks are used for 60, 90, and 120 second caps. The 348 excluded engagement rows have overlapping next-engagement starts at or before the current last kill. Stored match identifiers have no intersection between the three main split roles. Using the full corpus means retaining its designated training, validation, and test roles; it does not mean fitting a model on all 210,000 matches. [E01]

The value-model eligibility rule retains 208,603 main matches. Of the 1,397 excluded matches, 1,396 lack eligible query times and one has missing or conflicting final-winner information. The final value model is fitted on 424,160 state queries from 74,168 eligible TRAIN matches, including matches without qualifying engagements.

Value-model queries are drawn from a one-minute grid beginning at minute two, before match termination and within the observed frame range. One eligible query per five-minute bucket is selected deterministically by hash, and each match receives equal total weight. This samples states within matches for computational control; it is not a small-match pilot. It does not mean omitting the final two minutes of every match.

### 4 2 Validation roles and cross fitting

Patch 15.15 is partitioned by match hash into disjoint V_CAL, V_SELECT, Q_CAL, and Q_SELECT roles containing 18,579, 18,890, 18,710, and 18,569 raw matches, respectively. Task-specific eligibility and cohort masks are applied afterward. V calibration and selection therefore use different matches from q calibration and selection.

TRAIN engagement labels use five-fold value models that exclude the corresponding match. The same out-of-fold adapter evaluates both endpoints of a training engagement and supplies its pre-engagement value. Validation, test, and external labels use the final value model trained on eligible TRAIN matches. q preprocessing and base-model fitting use TRAIN; q calibrators use Q_CAL; model and calibration choices use Q_SELECT.

For evaluation within a reported cell, each represented match has equal total weight. A match contributing many engagements does not automatically dominate that cell. Primary uncertainty intervals resample matches rather than treating all engagements or all state queries as independent.

### 4 3 External cohorts

| External cohort | Raw matches | Valid Delta V rows | T rows | N rows |
|---|---:|---:|---:|---:|
| KR API 16.13 | 10064 | 25430 | 5202 | 20228 |
| KR API 16.14 pilot | 200 | 475 | 101 | 374 |
| KR API 16.15 | 926 | 2219 | 507 | 1712 |
| NA1 API 16.13 | 10000 | 26538 | 5312 | 21226 |
| Total | 21190 | 54662 | 11122 | 43540 |

No EUW1 evaluation is available. Public season labels and API patch identifiers are not interchangeable. These external sets have prior inspection or incompletely established access histories; none is presented as an untouched confirmatory cohort. The 200-match KR 16.14 set remains explicitly a pilot.

## 5 Deriving and applying the engagement definition

The first pipeline estimates useful temporal and spatial boundaries from event relationships. It does not discover a unique natural definition of a fight. The operational population still depends on researcher-selected gates, event coverage, and game-specific observation semantics. [E01; R1–R3]

The temporal estimator examines gaps between kill events. The selected boundary is a density valley between activity regimes, using distribution modeling and KDE rather than manually retaining the original 18 seconds. The spatial estimator examines temporally close kill pairs and the distance at which observed champion-sharing frequency crosses 0.5. Champion sharing is a proxy for common interaction, not a ground-truth annotation of combat.

| Definition | Current value or rule | Evidence and qualification |
|---|---|---|
| Temporal gap G | 13.7246 seconds, operationally about 13.7 | Pooled event-gap estimation; bootstrap interval 12.1601–15.6675 |
| Temporal stability | Approximately 10–18 seconds | Stability analysis, not a guarantee for every game phase |
| Spatial boundary D | 4263.8688 units, operationally 4264 | Champion-sharing crossing; interval 4256.05–4273.97 |
| Presence radius R | 1600 units | Game-rule motivation from experience range; not proof of true participation |
| Provisional onset s | First kill minus 15 seconds | Operational offset informed by assist-time scale and implementation precedent |
| Presence gate | At least two per team under the configured presence rule | Presence and retrospective participation are different quantities |
| Teamfight subgroup | Known minimum team cluster count at least four | Current v3.3 scale convention, not an estimated optimal cut |
| q cutoff | s minus one millisecond | Makes the boundary convention explicit |

The pooled estimation record uses 208,141 matches, 10,417,458 temporal gaps, and 5,460,008 temporally close kill pairs. Some estimation steps use seeded computational samples; we do not claim every bootstrap refits every observed gap. Spatial crossing uses interpolation between distance bins of width 750 units, so a narrow uncertainty interval does not establish equally fine physical classification resolution. Patch-specific gap estimates are approximately 13.964, 13.490, and 13.725 seconds; distance estimates are approximately 4285.26, 4265.27, and 4241.47 units. [E01]

Additional detector rules, merging behavior, participation extraction, and duration handling remain in the retained detector configuration. Participation includes recorded killers, victims, assists, and related event actors; it is not an exhaustive annotation of everyone physically fighting. The current distinction between T and N uses these recorded counts. Unknown-scale rows would stay outside T and N; none occurred in the retained current evaluation.

An important limitation is that pooled G and D estimation included patches 15.14, 15.15, and 15.16. Prediction-model train/test separation is intact, but definition estimation is not independent of the test patch. A development-patch-only definition analysis or untouched future confirmation is still required for a stronger end-to-end generalization claim. Kill-free pressure, zoning, and objective contests outside the detector remain outside the present target population.

## 6 Observable state and predictive inputs

StateV2 combines the latest frame no later than the query time with events observed no later than that time. Available frame variables include total/current gold, level, experience, health and mana ratios, survival, lane CS, and jungle CS. Event-derived variables include kills, deaths, structures, plates, objective ownership or acquisitions, elapsed-time features, and selected interactions. Some acquisition/death histories are proxies, not exact live buff-state measurements.

| Schema | Size before categorical expansion | Use |
|---|---:|---|
| StateV2 | 362 columns | State construction, including observation age |
| Primary V input | 361 columns | Excludes snapshot_age; 351 numeric plus 10 champion slots |
| Primary numeric q input | 352 columns | Excludes champion IDs and snapshot_age; includes p_pre_V |
| Economic q comparator | 149 columns | Restricted representation; not an equal-information tree comparison |

The transformed dimension of V after one-hot encoding must not be confused with its 361 original columns. The full ordered names, units, exclusions, and q grouping appear in the accompanying JSON dictionary. Base participant order is team then participantId rather than a postgame lane assignment.

The 352 q features map to seven retained groups: time and observation age 2, health/mana/other 21, economy/experience 60, combat/survival 68, objectives 176, structures 24, and prior win probability 1. The time-group name does not reintroduce the excluded snapshot-age feature. The mixed unknown-objective-team diagnostic remains in health/mana/other, so that group is not purely health.

No q input contains the realized endpoint, endpoint reason, engagement duration, future participant count, post-state, actual Delta V, final match winner, or future-interpolated coordinates. The primary q does not currently include position. A previous limited-sample position comparison showed an AUC difference of +0.00312 with an interval from -0.00287 to +0.00933; extrapolating positions with decay or a Kalman filter did not establish an improvement over last-observation carry-forward. These are pilot findings, not a universal claim that position is unimportant. Retrospective position interpolation for detection is separate from prediction input construction. [E01]

## 7 The changing match win estimate

The primary value model is a logistic model on expanded StateV2 with regularization parameter C equal to 0.01, predeclared from exploratory work rather than selected by a new hyperparameter search. Numeric median imputation, standardization, and one-hot vocabularies are fitted only on the relevant TRAIN partition. The recorded base solver is liblinear, maximum 1500 iterations, seed 7, with convergence diagnostics retained. Raw probabilities and a positive-slope sigmoid calibration were compared in dedicated validation roles by log loss, then Brier; the primary selected value model uses raw probabilities. [E01–E02]

The motivation for valuing outcomes through changing match-win estimates follows the direction of Maymin's analysis [R7]. Our use of public timeline states, a simple value model, and an engagement-specific endpoint is a separate implementation. The value model is a supporting component of the present engagement study, not a claim to have solved game-independent strategic valuation.

The fitted parameters remain fixed during a match. As observations change, the state is reconstructed and the same model is evaluated again. The time features can also alter its output between recorded events. This is a changing state-conditioned estimate, not online parameter retraining and not a guarantee of continuously observed physical state.

The current system evaluates stored data as chronological, cutoff-respecting replays. Live data ingestion, operational latency, onset detection, and automatic routing to the correct retrospective cohort have not been validated. Frame-based resources may remain stale between one-minute snapshots even when event counters update. We do not set the terminal value to the actual winner's zero or one when computing an engagement endpoint; we use the observed pre-terminal state and the value model.

### 7 1 Objective representation

The state includes Baron, six elemental dragon types, ownership-confirmed dragon soul, Elder Dragon, Rift Herald, Void Grubs, and Atakhan through counts, histories, elapsed times, and interactions. The value model learns associations between these state features and final victory. There is no current main-label formula that adds a separate invariant economic score to a separately weighted strategic score.

An unassigned soul event is not credited to an arbitrary team. Unknown ownership is handled diagnostically. FEAT_UPDATE and OBJECTIVE_BOUNTY are not explicit state channels. Atakhan and FEAT_UPDATE events were not observed in the evaluated external data, so those external tests do not establish transfer for those channels. Unseen champion identities affected approximately 18–25 percent of external value queries; this is another transfer limitation. [E01]

## 8 Endpoint and label construction

Let K be the first kill, s equal K minus 15 seconds, L the last kill, J the first subsequent champion kill anywhere on the map, S_next the next eligible engagement's onset, and T_end match termination. Missing subsequent-event candidates are treated as infinity. The primary endpoint is the earliest allowed candidate, with h equal to 90 seconds; 60 and 120 seconds are sensitivity caps.

$$t_i^{\mathrm{pre}}=K_i-15\,\mathrm{s}-1\,\mathrm{ms},\qquad e_i(h)=\min\{L_i+h,\ J_i-1\,\mathrm{ms},\ S_i^{\mathrm{next}}-1\,\mathrm{ms},\ T_i^{\mathrm{end}}-1\,\mathrm{ms}\}.$$

$$\Delta V_i(h)=V_{\theta_i}(S(e_i(h)))-V_{\theta_i}(S(t_i^{\mathrm{pre}})),\qquad Y_i(h)=\mathbf{1}[\Delta V_i(h)>0].$$

The same model and calibration are used at both endpoints of a row. The administrative cap L+h is inclusive; the competing event timestamps are excluded by one millisecond. Events sharing an excluded timestamp are not artificially ordered. Overlaps with S_next at or before L are excluded under the retained mask.

Acquiring an objective does not stop the window. Several acquisitions can be reflected before the endpoint. Conversely, the next global kill stops the window even if it is a distant kill or pursuit, so this rule is conservative and does not identify a natural combat boundary perfectly. S_next itself is retrospectively derived. Returning to base or regrouping is not an endpoint because a reliable observation rule has not been established.

Exactly zero change is coded as Y=0, meaning Blue non-improvement, and is tracked separately from negative changes. No exact zeros occurred in the currently retained valid labels. A negative Delta V corresponds to improvement for Red within a complementary two-team win-probability model, but the zero convention means the complete zero class is not strictly identical to Red improvement.

### 8 1 Why the primary cap is 90 seconds

The data do not support saying that every objective follows a fight within 60 seconds. In the earlier delay analysis, among 149,002 engagements with a first acquisition within 120 seconds, 55,734, or 37.4 percent, had that acquisition after 60 seconds. This is a conditional denominator and does not describe every engagement.

In a pilot competing-event analysis using the event-based endpoint, cumulative first-objective incidence before an additional kill was 11.93 percent at 60 seconds, 13.21 percent at 90 seconds, and 13.75 percent at 120 seconds. The extra incidence from 90 to 120 seconds was approximately 0.538 percentage points. Together with sensitivity results, this motivated a practical 90 second main cap. Competing-risk methodology motivates the analysis, not the numerical choice of 90 seconds. [E01; R4]

In the full stored main TEST h90 labels, actual follow-up after the last kill averages 45.493 seconds with median 39.440 seconds. Only 14.286 percent reach the 90 second cap; 129,036 of 163,576 rows stop at the next kill. The cap therefore does not mean every engagement observes 90 seconds of additional play. Main-test E label disagreement is 1.545 percent for h60 versus h90, 0.608 percent for h90 versus h120, and 1.909 percent for h60 versus h120. These endpoint sensitivities are different experiments from changing V.

### 8 2 A hypothetical worked example

| Time | Event and role |
|---|---|
| 19 minutes 44.999 seconds | Input cutoff; V estimates Blue match-win probability as 35 percent |
| 19 minutes 45 seconds | Defined engagement onset |
| 20 minutes | First kill |
| 20 minutes 20 seconds | Last kill of this engagement |
| 20 minutes 50 seconds | Blue acquires Baron |
| 21 minutes 15 seconds | Next eligible engagement begins |
| 21 minutes 30 seconds | Next engagement's first kill |
| 21 minutes 50 seconds | Original last-kill-plus-90-second cap |

Assuming no earlier competing event, the endpoint is 21 minutes 14.999 seconds. Baron is included, and the next engagement is excluded. If the value model gives 47 percent at this endpoint, Delta V is 0.47 minus 0.35, or +0.12. This is +12 percentage points and gives Y=1. Blue remains below a 50 percent final-win estimate, but its prospects improved relative to the start. The entire 12-point change cannot be attributed to Baron alone.

| Hypothetical pre probability | Post probability | Change | Label | Interpretation |
|---:|---:|---:|---:|---|
| 35 percent | 47 percent | +12 points | 1 | Still behind but improved |
| 65 percent | 59 percent | -6 points | 0 | Still ahead but deteriorated |
| 49 percent | 50 percent | +1 point | 1 | Small improvement |
| 49 percent | 69 percent | +20 points | 1 | Larger improvement with the same binary label |

The training pair for the first example is the pre-state and Y=1. The later Baron event, post probability, and Delta V are not predictors. A q output of 0.62 means predicted probability 0.62 of the positive label. If Blue loses the match much later, this engagement label is not changed retrospectively to a negative one.

## 9 Training the engagement predictor and preventing leakage

The q task is supervised binary probability estimation. Its typical objective is a weighted binary cross-entropy plus model-specific regularization. Match weights equalize each match's total contribution within the fitting subset; they are not proportional to the realized magnitude of Delta V.

$$\mathcal{L}_q=-\frac{1}{\sum_i w_i}\sum_i w_i\{Y_i\log q_i+(1-Y_i)\log(1-q_i)\}.$$

Consequently, examples with +1 and +20 percentage-point changes have the same class label. With equal weights and predictions they contribute equally to this binary loss. Magnitude regression, a neutral band, or magnitude-weighted loss would change the target or learning objective and have not been adopted as the main task.

The completed baseline pool includes a TRAIN-prior constant, p_pre-only logistic and spline models, a regularized full-numeric logistic model referred to as ridge in artifact names, and an economic-feature LightGBM. The economic tree uses 250 trees, 15 leaves, learning rate 0.04, minimum child samples 100, L2 regularization 1, and column sampling 0.9, with seeds 7, 42, and 123. The cohort/role study adds a full-feature tree control and additional representations under an explicitly separate comparison. [E02–E03]

Raw, sigmoid, and isotonic q calibrations are fitted or compared using Q_CAL and Q_SELECT. q selection uses match-weighted Brier, then log loss, then candidate name. The current T specialist selects ridge_raw; N selects ridge_isotonic. The q sigmoid implementation is an unconstrained logistic calibration on clipped logits; it must not be confused with the positive-slope value-model calibrator.

Cross fitting prevents direct use of a training match's own final outcome in the V used to label that match. It does not make the label independent of V, remove retrospective case selection, or eliminate every source of statistical dependence. Since p_pre is itself part of the label construction, the p_pre-only baseline is especially important. Its performance measures how much predictability is already captured by initial estimated advantage.

## 10 Completed value model results

The main TEST evaluation of V contains 347,234 state queries from 60,190 eligible matches, with equal total match weighting. [E01; E05]

| Metric for actual final match victory W | Primary V |
|---|---:|
| AUC | 0.858621 |
| Brier score | 0.153045 |
| Log loss | 0.456742 |
| Calibration slope | Approximately 0.976 |

AUC in the early 2–10 minute window is approximately 0.6741. The overall AUC therefore must not be presented as equally strong at every time point. Across external sets, value-model AUC is approximately 0.84–0.86 and calibration slopes approximately 0.81–0.87, indicating probability-calibration degradation under transfer. A nine-timepoint SimpleRNN value comparator has main-test log loss approximately 0.4745. That limited comparison concerns V, not the general quality of deep engagement predictors.

## 11 Completed engagement and role results

### 11 1 Same cohort comparisons

The table uses identical main TEST h90 rows within each cohort and equal match weighting. T has 32,981 rows; N has 130,595. [E01; E03]

| Predictor | T AUC | T Brier | T log loss | N AUC | N Brier |
|---|---:|---:|---:|---:|---:|
| Constant | 0.500000 | 0.249996 | 0.693139 | 0.500000 | 0.250019 |
| p_pre logistic | 0.664191 | 0.230303 | 0.652781 | 0.554907 | 0.247363 |
| p_pre spline | 0.664191 | 0.229786 | 0.651570 | 0.560168 | 0.246337 |
| Pooled q | 0.664066 | 0.229835 | 0.652761 | 0.596790 | 0.243097 |
| Cohort specialist | 0.670188 | 0.228171 | 0.648109 | 0.598404 | 0.242675 |

On the same T rows, specialist minus pooled AUC is approximately +0.0061 and Brier approximately -0.00166. The recorded paired match-bootstrap Brier interval is [-0.00235, -0.00104]. Comparing the pooled model's overall E AUC to specialist T AUC would confound a population change with model specialization.

A post hoc paired comparison of T specialist against p_pre spline gives Brier difference -0.001615 with a 95 percent interval [-0.002276, -0.000945], and log-loss difference -0.003461 with interval [-0.004905, -0.002087]. This resamples 24,020 represented matches 1,000 times. It is exploratory and conditional on fixed models, with no multiple-comparison adjustment. The added predictive value beyond initial advantage is measurable here but modest.

### 11 2 Role representation

Role inference uses champion identity and summoner spells available at the start, with weak TRAIN role supervision and out-of-fold predictions for TRAIN examples. A constrained assignment considers all 120 one-to-one allocations of five players to five roles; marginal role probabilities weight pre-state features. The role weights describe uncertainty about role assignment, not hand-set lane importance.

The implementation compares participant, draft-information, and role representations. Draft controls distinguish the additional champion/spell information from reorganizing information into roles. The role representation includes team-role summaries, Blue-minus-Red differences, selected time interactions, and assignment uncertainty. No future fight-participant list or postgame role is supplied directly to q.

| Finding | Completed result | Interpretation |
|---|---|---|
| Weak-label role agreement | Approximately 0.979 OOF | Agreement with weak labels, not true-role accuracy |
| External raw teamPosition agreement | Approximately 92–95 percent | Different imperfect reference; transfer limits remain |
| T role LightGBM | AUC approximately 0.66946; Brier 0.228240 | No clear improvement over the T specialist |
| N participant to draft to role LightGBM | AUC 0.61451 to 0.61590 to 0.61804 | Representation gains within the same learner |
| Corresponding N Brier | 0.239885 to 0.239710 to 0.239173 | Do not attribute a tree-versus-linear difference to role information |

Role-block removal in the selected T ridge analysis shows deterioration for jungle and mid information, with limited consistency across roles and patches. Redundant information remains in other features. We do not claim a universal lane-importance ranking.

## 12 SHAP explanations

The completed role explanation analysis uses up to 256 hash-selected TEST cases per cohort and 128 TRAIN background cases. Five role groups, global context, and role uncertainty form seven groups, giving 128 coalitions for exact group Shapley evaluation over that finite background. The explained quantity is the final selected calibrated probability, and the additive reconstruction of prediction from baseline plus contributions is checked. This is group-level explanation, not exact feature-level attribution over every original column. [E03; R5]

The T explanations have substantial global-context contribution; N also shows bottom/jungle contributions. These are descriptions of particular fitted models and sampled cases. Correlations, role uncertainty, engineered interactions, and background replacement affect the attribution. They do not identify causally necessary actions, mechanical skill, or what a player should have done. The explanatory sample is smaller than the evaluation corpus and is reported as such.

The proposed balanced-state SHAP comparison has not yet been executed. Filtering the old 256 explained cases after the fact is not treated as a representative balanced-state explanation study.

## 13 Completed label sensitivity and observation diagnostics

### 13 1 Changing the value model

Two alternate value models were refitted on exactly the same 424,160 TRAIN queries and 74,168 matches, each with final and five out-of-fold adapters. B_reg changes C from 0.01 to 0.1 while retaining inputs. B_econ retains economy/growth/champion/time information but removes several event and state channels. It is not an objective-only ablation. [E04]

| Value model | Main TEST W AUC | W log loss | T h90 label disagreement with primary V |
|---|---:|---:|---:|
| Primary V | 0.8586 | 0.4567 | Reference |
| B_reg | 0.8572 | 0.4586 | 1.09 percent |
| B_econ | 0.8300 | 0.4968 | 20.09 percent |

These disagreement rates are row-based. For T, the match-bootstrap intervals for the row proportions are 0.973–1.198 percent and 19.666–20.505 percent, respectively. E disagreement is 1.28 and 26.74 percent; N disagreement is 1.32 and 28.42 percent. The existing primary V and labels remain frozen.

Regulation sensitivity concentrates near zero: within T, primary absolute Delta V at most 0.5 percentage points has 8.15 percent disagreement with B_reg; above 2 points it is 0.028 percent. This motivates cautious interpretation of small changes. It is not a newly selected exclusion threshold or an estimated label-confidence interval.

The frozen T predictor evaluated against primary, B_reg, and B_econ labels has AUC 0.67019, 0.66821, and 0.63338, respectively. q was not retrained on alternate labels. This assesses dependence of an existing predictor's evaluation on the target, not the best attainable prediction quality for each alternate target.

Strictly increasing calibration applied identically at both endpoints preserves the sign of the change. B_reg label changes therefore cannot be attributed solely to its positive-slope sigmoid calibration. Different fitted coefficients, inputs, or non-strict transformations are different issues. [E04]

### 13 2 Observation gaps

In main TEST T, 35.27 percent of cases have no new frame after the last kill. This is not equivalent to having no frame change anywhere between pre and endpoint. The narrower same-pre-and-post-frame condition occurs in 2,224 T cases, or 6.74 percent. Event features can still change despite the same frame.

For B_econ, all non-time inputs are identical in those same-frame cases. Its labels are therefore driven only by time, and its match-weighted Blue-improvement rate is approximately 97.98 percent, versus 50.46 percent for the primary V. Its disagreement with primary V is approximately 49.86 percent there. This explains a limitation of that reduced comparator; it does not establish that primary labels are ground truth.

Across all main TEST T, 15.85 percent have an absolute primary Delta V at most one percentage point. Small changes are more prevalent when no post-last-kill frame is available. These cases were not removed after inspecting TEST results. The study retained an annotated audit trail of this post hoc mechanism analysis.

An anonymized 120-case human-review packet was prepared but no human judgments were collected. Human review is outside the current agreed scope. Automated checks and AI-assisted review are not reported as expert semantic validation.

## 14 Controlled objective feature ablation

This experiment removes only the 176 explicitly named objective-related input columns from the original 361-column V input, keeping the other 185. Removal covers Baron 38, Elder 38, dragon 32, soul 32, Herald 12, grubs 12, and Atakhan 12 columns, including related time interactions and specified acquisition/death histories. Model family, C=0.01, TRAIN queries, match weighting, preprocessing roles, folds, and primary raw-probability treatment remain matched. q is not refitted. [E05]

Gold, experience, CS, structures, time, and the mixed unknown-ownership diagnostic can retain indirect objective information. The diagnostic is positive in 0.245 percent of TRAIN value queries. The experiment therefore removes an explicit feature bundle, not all conceivable information about objectives.

### 14 1 Value prediction

| Main TEST metric | Full V | Without explicit objective features |
|---|---:|---:|
| AUC | 0.858621 | 0.852913 |
| Brier | 0.153045 | 0.156129 |
| Log loss | 0.456742 | 0.465612 |

Paired match bootstrap with 1,000 replicates gives Brier difference +0.003084, interval [+0.002842, +0.003310], and log-loss difference +0.008870, interval [+0.008223, +0.009499], where a positive difference indicates deterioration after removal. Intervals describe evaluation-match uncertainty for fixed fitted models, not all training uncertainty.

| External cohort | Full V log loss | Removal V log loss |
|---|---:|---:|
| KR 16.13 | 0.470089 | 0.478690 |
| KR 16.14 pilot | 0.489228 | 0.487042 |
| KR 16.15 | 0.466855 | 0.475290 |
| NA1 16.13 | 0.484217 | 0.494309 |

The small KR 16.14 pilot has the opposite direction. We therefore do not claim that objective features improve every patch without exception.

### 14 2 Label dependence and objective subsets

| Main TEST h90 cohort | Rows | Row disagreement | Equal-match disagreement |
|---|---:|---:|---:|
| E | 163576 | 5.089 percent | 4.917 percent |
| T | 32981 | 5.358 percent | 5.239 percent |
| N | 130595 | 5.022 percent | 4.875 percent |

The T equal-match disagreement interval is 4.982–5.480 percent. T row disagreement is 4.903 percent at h60 and 5.564 percent at h120. No horizon was selected from this comparison.

The following T subsets contain acquisitions by only one team during the pre-to-endpoint interval. Subsets can overlap. All objective features were removed together; these are not separate per-objective ablations.

| Acquisition subset | T cases | Row label disagreement |
|---|---:|---:|
| Baron | 2195 | 6.287 percent |
| Ordinary dragon | 7662 | 8.079 percent |
| Elder | 106 | 5.660 percent |
| Herald | 1523 | 4.859 percent |
| Grubs | 461 | 9.111 percent |
| Atakhan | 2722 | 8.303 percent |
| Ownership-confirmed soul | 677 | 14.328 percent |

Elemental dragon types, both-team acquisitions, unknown ownership, multiple acquisitions, and the last-kill-to-endpoint subinterval are retained in detailed aggregates. Sparse Elder cases do not support a general importance ranking. Acquisition-oriented differences between the two value models include the whole observed exchange and changes to all refitted coefficients. They are not a fixed causal price for one objective.

The frozen T q has AUC 0.670188 against the primary label and 0.679085 against the removal-model label. Its predictions have not changed, so this is not a q improvement. Choosing the easier label would change the research target and is not our selection rule. The primary objective-inclusive V remains the main valuation model.

## 15 Validation and evidence strength

Completed work includes exact row-key and split checks, parent-file preservation, fold membership, label arithmetic, feature removal, preprocessing provenance, serialization/reload consistency, and independent metric reconstruction. The label-sensitivity study records 17 synthetic checks, a TRAIN raw-event comparison of 40 cases, and a final 17 of 17 execution check result. The objective study records eight synthetic preflight checks and 18 of 18 final execution-contract checks. Separate independent calculations reproduce its main metrics, disagreement rates, paired intervals, and objective-subset aggregates. [E04–E05]

The broader readiness audit recalculated endpoint formulas and labels across seven sets and three caps, covering 1,863,414 stored row-by-cap combinations. This audit did not re-extract all raw matches or formally prove every detector code path. Passing these checks establishes arithmetic, implementation, and provenance consistency within the checked scope. It does not substitute for independent measurement validation or causal identification.

## 16 Planned fair model comparisons and balanced states

This section describes plans, not completed results. Exact implementation contracts are retained in E06 and E07. The fixed V, Delta V labels, h90 primary rule, patch roles, and full-data eligibility are preserved.

### 16 1 Model coverage

| Model | CoG provenance | Planned new comparison |
|---|---|---|
| LightGBM | CoG 308 | Numeric snapshot and representation controls |
| Plain matched-input MLP | CoG 308 | Same tabular input as tree; not replaced by residual MLP |
| Bi-GRU | CoG 308 | Pre-state sequence |
| Transformer | CoG 308 | Same sequence information contract |
| GraphSAGE | CoG 308 | Player graph with audited pre-state inputs |
| ST-GNN | CoG 308 | Temporal graph representation |
| Cross-Attn | CoG 308 | Pre-event and player representations |
| Layered Fusion | CoG 308 | Multiple audited pre-state branches |
| Layered plus Logit | Earlier 118 diagnostic and runner | Separate stacking diagnostic with match-wise out-of-fold logits |
| Logistic and residual MLP | New comparison baselines | Common 352 numeric inputs |

The common-information track compares logistic, LightGBM, plain MLP, and residual MLP on the same 352 columns. It separates algorithm effects from the old economic-only tree limitation. The representation track re-evaluates the CoG pairings under new labels and exact common evaluation keys. It does not claim that sequences, graphs, and event tokens supply identical information merely because they come from the same public telemetry.

For the common track, the proposed budget is six configurations per family; stochastic learners use seeds 7, 42, and 123, with their mean raw probability as the candidate prediction. Logistic uses six C values from 0.0001 through 10. LightGBM varies leaves 15/31/63 and minimum child samples 50/100. Both MLP forms vary widths 128/256/512 and dropout 0.1/0.3. This is a bounded comparison, not evidence of universally optimal hyperparameters or equal wall-clock computation. [E06; R8–R9]

A match-hash 10 percent subset inside TRAIN controls stopping; final candidates are refitted on all eligible cohort TRAIN rows with the selected iteration or epoch count. Q_CAL fits calibrators and Q_SELECT selects candidates by Brier, log loss, and name. All models and choices are frozen before new TEST evaluation. Existing TEST exposure still makes the study exploratory.

CoG sequence/graph/event adapters remain to be closed and validated before their full fits. Future positions, postgame role labels, or previously excluded information must not be silently reintroduced. A modified graph construction must be documented as an adaptation. Logit fusion requires its own out-of-fold base predictions; value-label cross fitting alone does not make stacking safe. Old runners that fit on train plus validation, use old labels, or write to old result folders cannot be executed unchanged.

### 16 2 Balanced and uncertain situations

The collaboration email proposed emphasizing more even game states. Our primary planned balanced subset is p_pre between 0.40 and 0.60 inclusive; 0.45–0.55 is a nested sensitivity subset. These thresholds are operational choices, not values prescribed by Kim et al. They are defined from pre-engagement V only, not from realized Delta V or final outcomes.

An estimated balanced match is different from a q prediction near 0.5. The former concerns eventual match prospects; the latter concerns model confidence about engagement-improvement direction. Neither is automatically a ground-truth measure of intrinsic difficulty. We retain the full benchmark and do not retrain only on balanced cases, reweight difficult cases, or select a different model after inspecting balanced TEST scores.

Planned evaluation compares the same selected models with the p_pre-only baseline in full T and balanced T, reports Brier, log loss, AUC, calibration, counts, and paired match-bootstrap intervals, and stratifies by pre-state time windows 0–10, 10–20, 20–30, and 30 minutes onward. V calibration is evaluated against W, while q calibration is evaluated against Y. The two references are not interchangeable.

Balanced-state SHAP will use separately hash-selected cases, the same TRAIN background, retained group definitions, and additive checks. Role models have extra draft information and therefore appear in a separate information-representation comparison, not the identical-352-input learner ranking.

### 16 3 Relation to the suggested calibration paper

Kim, Lee, and Chung study input-dependent uncertainty and calibration for match-win prediction in LoL. This supports attention to probability quality, particularly because our valuation uses probabilities. We have not implemented their uncertainty-aware loss and do not claim reproduction. The balanced-fight emphasis comes from the collaborator's email. Our current raw/sigmoid/isotonic comparisons and subgroup diagnostics are distinct procedures. [R6]

## 17 Remaining limitations and journal work

The most important unresolved issues are measurement validity, patch independence of definition estimation, observation granularity, and completing comparisons on the new target. Human semantic review remains unperformed by scope choice. Our wording therefore retains the model-defined outcome rather than calling it an independently validated fight-win truth.

The observed global Delta V may include farming, actions elsewhere, time progression, and multiple conversions. Without a counterfactual design it is not the isolated effect of the focal fight. Furthermore, V was trained against W: correlation between a V-defined label and W alone is not independent proof that q contributes new information about match victory. A q feature is a deterministic summary of its input; any additional benefit to a finite downstream model must be evaluated against a model with comparable pre-information.

Existing external tests provide transfer evidence with calibration deterioration, unseen champions, missing objective channels, sparse subsets, and prior access histories. They do not establish universal cross-patch operation with no adaptation. Development-only boundary estimation and genuinely new confirmation data remain priorities.

The journal manuscript still needs consistent Methods and Results for the Delta V target, updated reviewer-response mapping, clear separation from the old exchange score, model-comparison completion, final explanations and limitations, and reproducibility/publication materials. An earlier manuscript build or a completed experiment does not establish that the updated journal submission is ready.

| Reviewer or collaborator concern | Completed response | Remaining work |
|---|---|---|
| Arbitrary engagement constants | Data-estimated boundaries and stability analysis | Development-only estimation and full constant provenance |
| Subjective exchange weights | Learned V, OOF labels, sensitivity and objective ablation | Independent semantic truth not established |
| Strong fair model comparison | New-target baselines and cohort/role controls | Full CoG lineup and common-input deep comparison |
| Importance of balanced games | Research question and evaluation protocol | Execute balanced analysis and confirm externally |
| Probability calibration | V and q calibration evaluation | Planned subgroup study; Kim loss not implemented |
| Explainability | Group SHAP and role-block diagnostics | Balanced-state explanations and manuscript integration |
| Practical live use | Cutoff-respecting historical replay | Live acquisition, occurrence detection, routing, latency |

## 18 Evidence and reproducibility inventory

The following identifiers refer to retained project artifacts. They are provenance references, not public download links. The recipient-facing specification and feature dictionary contain no raw timelines, player account identifiers, credentials, or machine-specific absolute paths. Full data and code redistribution have not been packaged as part of this document.

| Evidence ID | Retained artifact | What it supports |
|---|---|---|
| E01 | TOG_END_TO_END_READINESS_AUDIT_20260915.md and its calculations.json | Definitions, counts, endpoint audit, V/q results, limits |
| E02 | full_corpus_training_20260915 protocol and REPORT | Full designated-corpus V, labels and q |
| E03 | cohort_role_training_20260915 protocol and REPORT | Cohort separation, role models, explanations |
| E04 | LABEL_VALIDITY_FINDINGS_20260915.md and label_validity_full outputs | Alternative V fits, observation diagnostics, 17/17 checks |
| E05 | OBJECTIVE_CHANNEL_ABLATION_FINDINGS_20260915.md and associated outputs | Explicit objective ablation, independent audits, 18/18 checks |
| E06 | DELTA_Q_FAIR_BALANCED_PROTOCOL_20260915.md | Planned common-input and balanced-state study |
| E07 | COG_MODEL_COVERAGE_DELTA_Q_20260915.md | Planned complete CoG coverage and unresolved adapters |
| E08 | CoG 308 paper text, Tables I–III | Published label, model lineup and historical results |
| E09 | Paper 118 review archive, AUDIT.md and run_paper_full.py | Version distinction and Layered plus Logit diagnostic |
| E10 | DELTA_LABEL_LEARNING_EXAMPLES_20260915.md | Hypothetical arithmetic and target explanation |

The document build retains a local source-hash manifest and content/layout verification record. These make this specification traceable to the files read during preparation. They do not certify experiments beyond the explicitly checked scopes above.

## 19 References and their scope

R1. Halfaker et al. 2015. User Session Identification Based on Strong Regularities in Inter-activity Time. [Author PDF](https://ironholds.org/resources/papers/sessions.pdf). Supports deriving activity boundaries from timing distributions. Our KDE valley estimator and 13.7-second value are adaptations, not that paper's validated LoL boundary.

R2. Jacobs and Wallach. Measurement and Fairness. [Paper](https://arxiv.org/abs/1912.05511). Supports separating the construct of strategic success from the operational measurement. It does not validate our labels.

R3. OpenDota processTeamfights implementation. [Pinned source](https://github.com/odota/core/blob/e2e44328032e5ce81879f2a6ec3ee73e9120e736/processors/processTeamfights.js). An implementation precedent for retrospective fight construction, not peer-reviewed evidence for LoL-specific constants.

R4. Austin, Lee, and Fine. 2016. Introduction to the Analysis of Survival Data in the Presence of Competing Risks. [Article](https://doi.org/10.1161/CIRCULATIONAHA.115.017719). Supports cumulative-incidence analysis with competing events. It does not establish that every subsequent kill is a new fight or that 90 seconds is universally optimal.

R5. Lundberg and Lee. 2017. A Unified Approach to Interpreting Model Predictions. [Paper](https://arxiv.org/abs/1705.07874). Supports additive feature attribution. Our groups, background budget and sample selection are operational choices, and SHAP is not a causal attribution here.

R6. Kim, Lee, and Chung. 2020. A Confidence-Calibrated MOBA Game Winner Predictor. IEEE CoG. [Original paper](https://ieee-cog.org/2020/papers/paper_221.pdf). Supports uncertainty-aware match-win calibration. Its loss is not currently implemented in our pipeline.

R7. Maymin. 2021. Smart kills and worthless deaths eSports analytics for League of Legends. Journal of Quantitative Analysis in Sports 17(1), 11–27. [Full article](https://d-nb.info/1367424143/34). Supports the motivation for valuing actions relative to changing win estimates. Its richer data and specific metrics differ from our public-timeline design; it does not prescribe our endpoint or validate our engagement labels.

R8. Gorishniy et al. 2021. Revisiting Deep Learning Models for Tabular Data. NeurIPS. [Paper](https://arxiv.org/abs/2106.11959). Motivates strong residual tabular baselines and consistent model-comparison protocols. Our bounded architecture/search settings are not an exact reproduction or proof of general deep-model superiority.

R9. Ke et al. 2017. LightGBM A Highly Efficient Gradient Boosting Decision Tree. NeurIPS. [Paper](https://proceedings.neurips.cc/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html). Establishes the learner family, not its expected ranking on the new Delta V task.

R10. Lee and Ramler. 2017. Identifying and Evaluating Successful Non-meta Strategies in League of Legends. FDG. [Author PDF](https://myslu.stlawu.edu/~clee/docs/lol.team.composition.fdg.2017.pdf). Supports role/composition inference as a research direction. The original work uses information including end-of-match items, which is not allowed in our pre-engagement role predictor.

R11. Hodge et al. 2021. Win prediction in multiplayer esports live professional match prediction. IEEE Transactions on Games 13(4), 368–379. [Article](https://doi.org/10.1109/TG.2019.2948469). A precedent for time-dependent esports match prediction. It does not establish live availability of our chosen LoL features.

Across these sources, we distinguish literature-supported methods, game-rule or implementation precedents, values estimated from our data, and our own operational decisions. No citation is used to imply that every threshold or label interpretation is externally established.
