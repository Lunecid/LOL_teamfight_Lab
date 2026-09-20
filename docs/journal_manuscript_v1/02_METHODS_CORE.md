# Methods — core definitions (draft)

*Prose draft for the journal freeze paper. Numbers and procedures cite locked artifacts. Epistemic status is stated here in draft 1, not deferred to a late note.*

---

## 0. Epistemic status (required)

The primary held-out contrasts that flex \(p,t\) baselines and report CORP / external dual-stage were produced in a **review-response** setting after exposure to the 15.16 test patch in earlier analyses. We therefore label the corresponding tables **exploratory** (design tag: `EXPLORATORY_REVIEW_RESPONSE_PRIOR_TEST_EXPOSURE`): they answer the scientific questions with locked weights and documented procedures, but they are **not** confirmatory preregistered tests.

Where execution of the flexible-baseline protocol differed from the written design (uniform spline knots rather than weighted quantiles; fitting weights not mean-normalized; two-stage hyperparameter then calibrator selection), we follow the execution addendum rather than rewriting history [RR1 addendum; RR0 manifest].

The skirmish cohort **S** (§1, §5) was added on 2026-09-21 under a predeclared contract (`SCALE_SPLIT_EXPERIMENT_CONTRACT_20260920.md`; tag `EXPLORATORY_SCALE_SPLIT_PRIOR_TEST_EXPOSURE`). Its 15.16 rows had been scored in the 2026-09-15 lineage as part of the non-teamfight cohort, so S carries the same prior-exposure status as T. Its contrasts use the contract-literal **identity** calibrator; the RR12 two-stage selection, which chose a positive-slope sigmoid for every S model, is reported only as a sensitivity variant (§4.2, Results §3.1).

---

## 1. Engagement unit

An **engagement** is a kill-anchored cluster in Riot Match-V5 timeline telemetry, built by the v3.3 detector with the rules below. Constants are the values injected into the detector; derivations, intervals and the full constant table are in the definition lineage (`docs/tog_manuscript/sec_definition.tex`, cited by subsection).

- **Kill episodes and the temporal gap \(G\).** Consecutive champion kills of a match are chained into one episode when their gap is at most \(G\). \(G\) is the antimode of the log inter-kill-interval distribution (KDE valley between the within-fight and between-fight modes; a two-component Gaussian mixture decides the sides): \(\hat G = 13.7246\) s over 208 141 matches, run at \(G = 13.700\) s; match-bootstrap 95% CI 12.2–15.7 s; the episode partition has ARI ≥ 0.9 for every gap from 10 to 18 s. [sec_definition §Kill episodes and the temporal boundary \(G\)]
- **Spatial split and the diameter \(D\).** An episode is split so that no two of its kills lie more than \(D\) apart (time-ordered first-fit grouping). \(D\) is the distance at which the rate of consecutive kill pairs sharing a champion (as killer, victim or assister) falls to 0.5: \(\hat D = 4\,263.87\) u over 5 460 008 pairs, run at \(D = 4\,264.0\) u; match-bootstrap 95% CI 4 256–4 274 u, which reflects sampling only because the estimate interpolates between bin centres 750 u apart. [§The spatial boundary \(D\)]
- **Presence gate (\(R\), \(B\), \(M\)).** The cutoff is \(\tau = t_{\text{first kill}} - B\) with \(B = 15\) s. An episode becomes an engagement only if, at \(\tau\), at least \(M = 2\) alive champions of **each** team stand within \(R = 1\,600\) u of the first kill's position (the anchor). Alive status is read from the last one-minute frame before \(\tau\); positions come from a 5 s grid interpolated between one-minute frames, with each kill participant's path drawn towards its kill position. \(R\) and \(B\) are rule anchors (champion-death experience-sharing radius; assist window) held fixed by assumption and checked against patch notes 25.14–25.16; \(M\) is a convention. [§Presence gate, lead and anchor]
- **Gold-label attribution window.** The CoG-continuity label `market_event` reads events in \([\tau,\ \max(t_{\text{last kill}}+1\,\text{ms},\ \tau+H))\) with \(H = 35\) s, cut short at an ace, and only inside a disc of radius \(D\) centred on the anchor. The SVI endpoint is defined separately (§1b). [§Which events the label may read]
- **Assembly guards and remaining constants.** The table below lists every constant that shapes the engagement set, with the source class used in the definition lineage (data-derived / rule anchor / convention / engineering); values are those injected into the v3.3 detector. [`tab:constants`, sec_definition L451–L641]

| Constant | Value | Source class | Basis (measured impact where it exists) |
|---|---|---|---|
| Kill gap \(G\) | 13.7 s (estimate 13.7246 s; run at 13 700 ms) | Data-derived | KDE antimode of log inter-kill intervals; CI 12.2–15.7 s; ARI ≥ 0.9 for 10–18 s |
| Diameter \(D\) | 4 264 u (estimate 4 263.87 u) | Data-derived | 0.5 crossing of champion-sharing rate over 5 460 008 pairs; CI 4 256–4 274 u (sampling only) |
| Executions (kills with `killerId` 0) | included | Engineering | 0.41% of kills in a pilot; effect on engagements not measured |
| Presence radius \(R\) | 1 600 u | Rule anchor | champion-death experience-sharing radius; patch notes 25.14–25.16 list no change |
| Lead \(B\) | 15 s | Rule anchor | assist window; sets \(\tau\) |
| Minimum per team \(M\) | 2 (both teams, within \(R\), alive) | Convention | after Ke et al.; at \(M=3\) the corpus shrinks to 76 290 engagements |
| Position grid | 5 s; frame and kill-trajectory interpolation on | Engineering | median position error 1 149 u in a pilot; effect on the gate not measured |
| Re-engagement merge, gap, radius | on; 15 s; 2 000 u | Engineering | merges not counted |
| Duration cap | 60 s after \(\tau\) | Engineering | not measured |
| Start and end exclusion | first 2 min; \(\tau+35\) s before last frame | Engineering | not measured |
| Context guard, observation window | 30 s in 5 s bins up to \(\tau\); gap 0 | Convention (conference version) | window sweep reported in the definition lineage |
| Label horizon \(H\) (gold label only) | 35 s; ace truncation on | Engineering; convention | applies to `market_event`, not to the SVI endpoint (§1b) |
| Overlap resolution | location radius \(=D\); participation priority | Engineering | — |
| Tail buffer, minimum start gap | 0, 0 (disabled) | Engineering | — |
| Post-fight window | 30 s (diagnostic only, unused by label or features) | Engineering (unused) | no effect on the corpus |
| Interaction radius; shop events | 3 000 u; included | Engineering | participation counts (scale classes) only |
| Scale cuts | pick ≤ 1 / skirmish 2–3 / teamfight ≥ 4 | Convention | pick − teamfight AUC sign follows the cut (definition lineage) |
| Dead zone \(\varepsilon\) (gold label) | 300 g | Engineering (rule-motivated) | one base kill bounty; label-family variants in the lineage |
| Attribution radius (gold label) | \(D\) (disc diameter \(2D\)) | Engineering (tied to \(D\)) | — |
| Draws (gold label) | dropped | Convention | 33 905 of 566 452 |

- **Per-patch re-estimation.** \(G\) and \(D\) re-estimated per patch (15.14 / 15.15 / 15.16: \(G\) = 14.0 / 13.5 / 13.7 s; \(D\) = 4 285 / 4 265 / 4 241 u) all lie inside the pooled plateau and within 10% of the pooled \(D\), so one pooled definition serves the corpus. [§Re-estimation per patch]
- **Scale classes and the teamfight cohort T.** Participation per side counts that side's champions among the killers, victims and assisters of the engagement's kills plus the actors of other timeline events within 3 000 u of the anchor between \(\tau\) and the last kill; classes use the smaller side's count \(n_{\min}\): *pick* \(n_{\min}\le 1\), *skirmish* \(2\le n_{\min}\le 3\), **teamfight \(n_{\min}\ge 4\)**. The prediction sample of this paper consists of **two cohorts reported side by side**: the teamfight class **T** and the skirmish class **S** (\(2\le n_{\min}\le 3\); flag `cohort == 0 & fine == 1` in the same build). Picks (\(n_{\min}\le 1\)) are excluded a priori \pending{pick-exclusion-wording: author sentence for the reason; the scale-split contract excluded picks and no fit85-lineage result exists for them}. Each cohort has its own fitted \(q\) and baselines and its own primary contrast; the cohorts are never pooled into one table row and their absolute scores are not compared (Discussion §5). In the frozen pipeline T membership is the flag `cohort == 1` written by the 2026-09-15 cohort-role build, computed by `scale_classes` as \(\min(\texttt{cluster\_blue},\texttt{cluster\_red})\ge 4\) on the stored v3.3 participation counts (kill participants plus interaction actors within 3 000 u; not the pre-cutoff presence count), with no other filter; it coincides exactly with the v3.3 teamfight class (`fine == 2`) in every set (`docs/lineage_20260915/cr20260915_common.py` L37–60; `docs/lineage_20260915/cohort_manifest.json` `rules`). Rows additionally require `valid_h90 == 1` and finite pre/post scores (§5). Fold and validation roles are match-level: TRAIN fold = sha256('full-v-oof-20260915:' + match)[:8] mod 5; VALIDATION role = sha256('full-val-20260915:' + match)[:8] mod 4 → V_CAL, V_SELECT, Q_CAL, Q_SELECT (`docs/lineage_20260915/README.md`). The corpus counts of the definition lineage (532 547 v3.3 engagements from 208 141 matches, of which 109 829 teamfights) come from a different corpus and filter than the pooled T of this paper (113 901 from 210 000 matches) and are not tabulated together. [§Scale classes]

Sensitivity of secondary constants (anchor placement, shop-event exclusion, label horizon, execution kills, position-grid error, the \(G\times D\) sweep) was measured on conference-era or pilot corpora and is documented in the definition lineage; it has not been re-run under the current corpus. Kill-less proximity exchanges lie outside this detector's instances and are not the prediction target of this paper.

We do **not** treat the gold engagement-winner label `market_event` as the freeze paper’s primary outcome. That label remains relevant for CoG continuity and for secondary correspondence checks, but the prediction target below is defined from the frozen win-probability evaluator.

### 1b. Prediction time and outcome time

- **Prediction time.** The feature cutoff is \(\tau = t_{\text{first kill}} - B\) (\(B = 15\) s). \(q\) and the PT baselines read only the pre-state snapshot \(S_{\mathrm{pre}}\) taken at or just before \(\tau\); the actual endpoint, the actual interval length, post-engagement participants and post states are never inputs. [`V_DYNAMIC_FRAME_CONTRACT_20260919.md`; `REVIEW_RESPONSE_EXPERIMENT_DESIGN_20260920.md` §2.1]
- **Outcome time.** Let \(K\) be the first kill, \(s = K - 15\) s, \(q_{\mathrm{pre}} = s - 1\) ms, \(L\) the last kill of the engagement, \(K_{\mathrm{next}}\) the first champion kill anywhere on the map strictly after \(L\), \(S_{\mathrm{next}}\) the onset of the next eligible engagement (itself 15 s before its first kill), and \(T_{\mathrm{end}}\) the match end; absent events are \(+\infty\). The endpoint under horizon \(h\) is

\[
e_h=\min\bigl(L+1000h,\ K_{\mathrm{next}}-1,\ S_{\mathrm{next}}-1,\ T_{\mathrm{end}}-1\bigr)\ \text{[ms]},\qquad h=90\ \text{(primary)},\ 60,\ 120\ \text{(sensitivity)},
\]

  and \(\Delta\widehat V=\widehat V(S(e_h))-\widehat V(S(q_{\mathrm{pre}}))\). Objective acquisitions and aces are not terminal events under this rule. The state at a query time is the last observed one-minute frame at or before it plus every event up to it (closed inclusion; no interpolation; the frame age is stored). A row is labeled only if \(e_h\ge L\), \(e_h>q_{\mathrm{pre}}\), \(e_h\le\) last frame, \(q_{\mathrm{pre}}\) lies inside the observed frames, \(S_{\mathrm{next}}>L\), and the state builder succeeds at both queries (`valid_h90 == 1`); failures are excluded with a recorded reason, never dropped silently, and an early match end truncates \(e_h\) rather than invalidating the row or replacing \(\widehat V\) by the final outcome. Of 566 452 detected engagements, 566 104 are valid at h60, h90 and h120 alike; all 348 invalid rows are overlap cases (\(S_{\mathrm{next}}\le L\)) and none is a teamfight. Because the cap is anchored at the last kill, "h90" is not a 90 s window: on the 15.16 timeline the observed follow-up after \(L\) averages 45.493 s (median 39.440 s), 14.286% of rows reach the cap and ≈78.9% stop at the next kill; the interval length used for quiet matching (RR3), \(L_i=e_h-q_{\mathrm{pre}}\), also contains the 15 s lead and the kill cluster itself. [`docs/lineage_20260915/engagement_labels_v3_rules.py` `endpoint_rule`, `endpoint_validity`; `docs/lineage_20260915/LABEL_ENDPOINT_RULE_AND_EXAMPLES_20260914.md`; `docs/lineage_20260915/README.md` excerpts; design §2.1 L44, L177]

---

## 2. Frozen win-probability evaluator \(\widehat V\)

Let \(W\in\{0,1\}\) be the match winner (blue = 1). The evaluator

\[
\widehat V(x_t)=\widehat{\Pr}(W=1\mid x_t)
\]

is a multilayer perceptron on an expanded tabular state representation, fit under the locked **fit85** match-holdout protocol (`A_MLP_expanded`, bundle sha16 `ac459cc4397630a9`). The weights of this bundle are **frozen** for every evaluation-role computation in this paper (Q_CAL, Q_SELECT, 15.16 TEST, and the external cohorts). The 15.14 TRAIN rows used to fit \(q\) are labeled differently: five match-fold evaluators \(\widehat V^{(-k)}\) with the same architecture, preprocessing, early-stopping rule and V_CAL calibration step are refit without fold \(k\)'s matches, and \(p_{\mathrm{pre}}\), \(\Delta\widehat V\), \(Y_{\mathrm{SVI}}\) and B40 for a TRAIN engagement all come from the same \(\widehat V^{(-k)}\) (§5). The frozen bundle never labels its own training matches. Changing \(\widehat V\) would redefine \(p_{\mathrm{pre}}\), \(\Delta\widehat V\), \(Y_{\mathrm{SVI}}\), and balance slices (including B40); we therefore do not retune \(V\) to improve direction AUC.

**Inputs and architecture.** The state representation is *Expanded361*: 351 numeric features and 10 champion-ID slots. The MLP maps each champion ID to an 8-dimensional embedding (vocabulary 172, index 0 = unknown) and concatenates it with the standardized numeric block, giving a \(351 + 10\times 8 = 431\)-dimensional input; the network is Linear → LayerNorm → GELU → Dropout(0.1) for hidden widths (256, 128), then Linear(1). [`A_MLP_expanded_evaluator_meta_20260919.json` `preproc_dims`; `scripts/rr20260919_v_redesign_fit_wave4_corrected.py` `fit_mlp_emb`]

**Fitting (fit85).** Trained with AdamW (learning rate \(10^{-3}\), weight decay \(10^{-4}\)) on match-weighted binary cross-entropy, batch 1 024, seed 7, at most 100 epochs with early stopping (patience 10) on the Brier score of a 15% match-level holdout carved inside 15.14 TRAIN; the best-epoch state is restored. No full-TRAIN refit was performed (`fit_scope: train_fit85_match_holdout_frac0.15_seed7_NO_full_train_refit`). The preprocessing bundle is fit on the same TRAIN portion. [same script; meta JSON `fit_scope`]

**Calibration and selection.** A positive-slope sigmoid \(g\) is fit on 15.15 V_CAL (coefficient 0.9076, intercept 0.0640) and applied to the network logit, so \(\widehat V = g\circ f\circ T\). The candidate was selected on 15.15 V_SELECT by the predeclared time-balanced loss \(L_{\mathrm{time}} = \sum_b \alpha_b\,\mathrm{Brier}_b\), \(\alpha_b = 1/4\), over the bands [2,10), [10,20), [20,30), [30,∞) minutes (rows before 2 min excluded). Competing candidates (a GRU history model lies within ≈0.0002 of the winner on V_SELECT) are single-seed point estimates; no claim of architectural superiority is made. [meta JSON `calibration`; `V_NEXT_RUN_EXECUTION_CONTRACT_20260919.md` §4; `BAND_LEDGER_WAVE4_CORRECTED_20260919.md`]

**Role of \(V\):** a **measurement** model of match win probability at a snapshot. Its quality is assessed by proper scores and discrimination of \(W\) (\(V\to W\)), separately from whether pre-state predicts the **sign** of engagement-interval change (\(q\to\mathrm{SVI}\)).

---

## 3. Interval change and strategic value improvement (SVI)

For each engagement with pre-cutoff state \(x_{\mathrm{pre}}\) and defined post-endpoint state \(x_{\mathrm{post}}\),

\[
\Delta\widehat V
=
\widehat V(x_{\mathrm{post}})
-
\widehat V(x_{\mathrm{pre}}),
\qquad
Y_{\mathrm{SVI}}
=
\mathbf{1}[\Delta\widehat V>0].
\]

\(Y_{\mathrm{SVI}}=0\) therefore covers both a decrease and an exactly unchanged estimate (\(\Delta\widehat V=0\)). Engagements whose pre- or post-state score is not finite are excluded before labeling rather than coded 0; the 15.14 out-of-fold training labels contain one exact zero.

Interpretation (locked): \(Y_{\mathrm{SVI}}=1\) means the **estimated** blue win probability rose over the engagement interval under the frozen evaluator — not that blue necessarily became the match favorite, and not a causal attribution that “the fight alone” produced exactly the observed change (\(100\,\Delta\widehat V\) percentage points; e.g. \(0.70\to0.60\) is \(\Delta\widehat V=-0.10\), i.e. \(-10\) percentage points) (other concurrent state updates may enter the same window).

Primary prediction target: **direction** \(Y_{\mathrm{SVI}}\). Magnitude regression on \(\Delta\widehat V\) is out of scope for this journal version.

**B40:** engagements with \(p_{\mathrm{pre}}=\widehat V(x_{\mathrm{pre}})\in[0.40,0.60]\) — a 20-percentage-point balanced band centred on 0.5, not a ±0.40 band — recomputed from the frozen fit85 \(p_{\mathrm{pre}}\); on 15.16 TEST, n=5 423 engagements from 4 945 matches. Reported as a **key conditional** evaluation alongside all-T, not as a retuned primary.

---

## 4. Direction predictor \(q\) and baselines

### 4.1 Frozen \(q\)

\[
q(x_{\mathrm{pre}})
=
\widehat{\Pr}\!\left(Y_{\mathrm{SVI}}=1\mid x_{\mathrm{pre}}\right)
\]

is the locked **logistic / `logit_state`** model trained on out-of-fold SVI labels under the sealed feature order (MAIN feature-order sha16 `103f0b92094847bc` for external score-only application). Inputs use **pre-state only**. For the reported RR contrasts on T, the selected probability mapping on Q_SELECT was **identity** (no additional increasing calibrator on \(q\)).

Inputs are the 351 numeric pre-state features plus \(p_{\mathrm{pre}} = \widehat V(x_{\mathrm{pre}})\) (352 inputs), standardized and fit with L2-regularized logistic regression; the 10 raw champion-ID slots are excluded. The LightGBM candidate `lgbm_state` used the same numeric block plus the 10 champion IDs as categorical features (362 inputs), so the two learners were not compared on identical inputs. [`scripts/rr20260920_q_newv_primary_fit.py` L185–L225]

**Cohort-specific \(q\).** For S, \(q_S\) is fit with the identical specification (same 352 inputs, same L2 logistic learner fixed a priori, no learner re-selection) on S TRAIN out-of-fold labels. Two secondary arms exist only for Results §3.4: the frozen T model applied unchanged to S rows (\(q_{T\to S}\)), and a pooled fit on T ∪ S TRAIN with match weights recomputed inside the union (\(q_{TS}\)). [`scripts/rr20260920_q_newv_primary_fit.py` `--cohort-tag`, `--fixed-learner`; `scripts/ss20260920_union_labels.py`]

### 4.2 Baselines in \(p_{\mathrm{pre}}\) and time

- **PT_linear:** \(\sigma(a + b\,p_{\mathrm{pre}} + c\,t)\).  
- **PT_flex:** flexible spline expansion in \((p_{\mathrm{pre}}, t)\) with locked knot/C configuration from Q_SELECT (`n_knots_p=4`, `n_knots_t=4`, `C=0.01`, degree 3); selected calibrator **identity**.

PT_flex in detail: cubic B-spline bases (`SplineTransformer`, uniform knots, `include_bias=False`, constant extrapolation) on \(p_{\mathrm{pre}}\) (`n_knots_p`) and on time in minutes (`n_knots_t` = 4, fixed), plus the full tensor product of the two bases; standardized and fit with L2 logistic regression (lbfgs) under weights \(1/n_m\). The search grid was `n_knots_p` ∈ {4, 6} × \(C\) ∈ {0.01, 0.1, 1}; the winner by Q_SELECT match-weighted Brier was (4, 4, 0.01). The \(p\)-only spline baseline uses \(p_{\mathrm{pre}}\) alone (`n_knots` 6, \(C\) 0.01); PT_linear is standardized logistic regression on \([p_{\mathrm{pre}}, t]\). As stated in §0, knots are uniform rather than weighted-quantile, fitting weights were not mean-normalized, and hyperparameters were chosen before the identity/sigmoid calibrator (two-stage). [`scripts/rr20260920_review_response_rr12.py` `PTFlexFeatures`, `fit_pt_flex`; RR12 JSON `PT_flex`, `b_spline`]

**PT_flex_S and the S calibrator policy.** For S the same grid was searched on S Q_SELECT and the same winner (4, 4, 0.01) was selected; PT_linear_S and \(b(p)_S\) are fit likewise on S TRAIN. The two-stage rule of RR12, applied to S, chose the positive-slope sigmoid for every model (\(q_S\), PT_flex_S, PT_linear_S, \(b(p)_S\)). Because the scale-split contract fixed the identity calibrator a priori and defines \(q_{T\to S}\) as the frozen T model applied as-is, all S results in this paper use the **identity** calibrator; the selection variant is reported as a sensitivity row and moves the S primary contrast from −0.00335 to −0.00346 (Results §3.1). [`scripts/rr20260920_review_response_rr12.py` `--calibrator`; `docs/SCALE_SPLIT_TvsS_RESULTS_20260920.json` `table2a_identity`, `table2b_sensitivity`]

**Selection and calibration of \(q\).** Candidates and baselines were fit on 15.14 TRAIN (out-of-fold labels), selected on 15.15 Q_SELECT by match-weighted Brier over all T, and then frozen. Q_CAL was used only for LightGBM early stopping; a positive-slope sigmoid \(g_q\) was fit on Q_CAL for every model and compared with the identity on Q_SELECT, and the identity was retained for \(q\) and PT_flex. No separate probability calibrator is applied to the reported \(q\). The same steps were run separately on the S roles (learner fixed to `logit_state` a priori; PT_flex knots / \(C\) selected on S Q_SELECT; identity calibrator by contract). [`Q_PREDICTION_DESIGN_CONTRACT_20260920.md` §3; RR12 JSON `calibrator_choice`; `docs/SCALE_SPLIT_EXPERIMENT_CONTRACT_20260920.md` §4]

**Primary contrasts for this paper (one per cohort):** ΔBrier and related scores of frozen \(q\) versus **PT_flex** on the same T rows, and of \(q_S\) versus **PT_flex_S** on the same S rows. PT_linear is retained for continuity with earlier tables. Constant and \(p\)-only splines are reported only as context in the RR12 table, not as a model zoo search.

We do not expand Transformer / GNN / TabM candidates for this version.

---

## 5. Data splits and cohorts

**Table 5a — evaluator (time-state) roles.** [`A_MLP_expanded_evaluator_meta_20260919.json` `meta.census`; `TEST_BAND_LEDGER_MLP_FIT85_20260920.md`]

| Role | Patch | Time-state rows | Matches | Use |
|---|---|---:|---:|---|
| TRAIN fit | 15.14 | 360 479 | — | weights |
| TRAIN stop | 15.14 | 63 681 | — | early stopping (15% match holdout, seed 7) |
| V_CAL | 15.15 | 106 580 | — | positive-slope sigmoid \(g\) |
| V_SELECT | 15.15 | 108 558 | — | \(L_{\mathrm{time}}\) selection |
| TEST | 15.16 | 347 234 | 60 190 | sealed ledger |

TRAIN total 424 160 = 360 479 + 63 681. Match counts not recorded in the census are shown as "—".

**Table 5b — direction-predictor (engagement) roles.** [`PAPER_COHORT_CONTRACT_20260919.md` §2, §4; `EXPERIMENT_INVENTORY_COHORT_20260919.md` §E; RR12 JSON `census`; `Q_NEWV_FIT85_TRANSFER_16X_20260920.md`; `REVIEW_RESPONSE_RRX_EXTERNAL_20260920.md`]

| Role | Patch / region | Engagements | Matches | Label source | Use |
|---|---|---:|---:|---|---|
| TRAIN | 15.14 | 39 605 | 28 989 (Σ 5 folds) | \(\widehat V^{(-k)}\) out-of-fold | fit \(q\), PT_flex, \(b(p)\) |
| Q_CAL | 15.15 | 10 390 | 7 550 | fit85 | LightGBM early stop; \(g_q\) candidates |
| Q_SELECT | 15.15 | 10 195 | 7 469 | fit85 | select knots / \(C\) / calibrator |
| V_CAL / V_SELECT | 15.15 | 10 191 / 10 539 | 7 448 / 7 632 | — | evaluator calibration / selection matches only; never used for \(q\) (15.15 VAL total 41 315) |
| TEST | 15.16 | 32 981 | 24 020 | fit85 | primary |
| B40 ⊂ TEST | 15.16 | 5 423 | 4 945 | fit85 | conditional |
| EXT KR 16.13 | KR | 5 202 | 3 859 (10 064 collected) | fit85 | score-only |
| EXT NA1 16.13 | NA1 | 5 312 | 3 955 (10 000 collected) | fit85 | score-only |
| EXT KR 16.15 | KR | 507 | 377 (926 collected) | fit85 | reported, not pooled |
| EXT KR 16.14 pilot | KR | 101 | 80 (200 collected) | fit85 | reported, not pooled |
| TRAIN (S) | 15.14 | 123 049 | 60 820 | \(\widehat V^{(-k)}\) out-of-fold, same fold evaluators as T | fit \(q_S\), PT_flex_S, \(b(p)_S\) |
| Q_CAL (S) | 15.15 | 31 302 | 15 254 | fit85 | LightGBM early stop (diagnostic only) |
| Q_SELECT (S) | 15.15 | 31 059 | 15 282 | fit85 | select knots / \(C\) |
| TEST (S) | 15.16 | 101 205 | 49 730 | fit85 | primary (S block) |
| B40 ⊂ TEST (S) | 15.16 | 31 675 | 25 039 | fit85 | conditional (within-S) |
| EXT KR 16.13 (S) | KR | 15 641 | — | fit85 | score-only |
| EXT NA1 16.13 (S) | NA1 | 16 100 | — | fit85 | score-only |
| EXT KR 16.15 (S) | KR | 1 307 | — | fit85 | reported, not pooled |
| EXT KR 16.14 pilot (S) | KR | 285 | — | fit85 | reported, not pooled |

Pooled T (15.14 + 15.15 + 15.16) = 113 901 engagements, used for measurement tables only; summing the recorded per-role match counts gives 83 108 matches (a sum of `cohort_manifest.json` fields, not itself a recorded figure). Every T row is valid at h90; the 348 invalid rows of the corpus are all non-teamfight engagements. Sample flow: source matches (210 000 KR, patches 15.14–15.16) → detected engagements → teamfight class T → pre-state available and `valid_h90` → finite pre/post scores → role assignment as above; counts for the intermediate stages are not recorded in the freeze documents and are not estimated here.

S rows follow the same flow with the skirmish class in place of T and are labeled by the same two evaluator paths (T008 manifest: S∩T key intersection 0; every S and T TRAIN match lies inside its fold's held-out match set, 14 628–15 017 matches per fold). In 15.16, 101 242 skirmish engagements are detected and 101 205 are valid at h90. S match counts for the external sets are not recorded. For \(q_{TS}\) the T and S label sets are concatenated (TRAIN 162 654 rows from 65 465 matches; key overlap 0) and weights are recomputed inside the union; the union TEST is not an evaluation cell. [`docs/SCALE_SPLIT_RR0_MANIFEST_20260920.json` `row_counts`, `integrity`; `docs/lineage_20260915/cohort_manifest.json` `sets.MAIN_TEST.skirmish`; `.ai/reports/logs/T009_union.txt`]

**Two evaluator paths.**

| Rows | Evaluator defining \(p_{\mathrm{pre}}\), \(\Delta\widehat V\), \(Y_{\mathrm{SVI}}\), B40 | Rule |
|---|---|---|
| 15.14 TRAIN (fitting \(q\) and the PT baselines) | \(\widehat V^{(-k)}\), \(k=1,\dots,5\), match-level folds | No information from fold \(k\)'s matches enters that evaluator's preprocessing, weights, early-stop holdout, or calibrator |
| 15.15 Q_CAL / Q_SELECT, 15.16 TEST, 16.x external | Frozen fit85 bundle | Same bundle, preprocessing and calibration for every row |

External application is **score-only**: no refit of \(V\) or \(q\) weights; fail-closed if keys or feature order mismatch. Small external cohorts are not averaged into a transfer-success claim.

---

## 6. Scores and uncertainty

- **Proper scores:** Brier score and log loss (lower better).  
- **Cells:** every evaluation cell is cohort-specific (T or S). No pooled T ∪ S cell is scored, and no statistic compares the two cohorts' absolute scores; cross-cohort statements are limited to the predeclared transfer and pooling contrasts on identical rows (Results §3.4).  
- **Ranking:** AUC (reported; not the sole success criterion).  
- **Paired contrast:** ΔBrier(\(q\) − baseline) with **match-cluster bootstrap** 95% intervals. We do **not** rename these intervals as Diebold–Mariano or Giacomini–White tests.  
- **Weighting:** in every evaluation cell, each engagement (or time-state query) carries weight \(w_i = 1/n_m\), where \(n_m\) is the number of that match's rows inside the cell; weights are recomputed per cell, and Brier, log loss, AUC, positive rate, ECE and the CORP components are all weighted averages under \(w\). Exceptions are labeled where they occur: RR3 medians and quantiles are row-level; RR5a disagreement shares are unweighted; RR5b next-objective counts are unweighted. [`scripts/rr20260920_review_response_rr12.py` `match_weights`; design §4.2]  
- **Bootstrap:** paired match-cluster percentile bootstrap — matches are resampled with replacement, per-match weighted squared-error sums are re-aggregated, and the 2.5th and 97.5th percentiles of ΔBrier over 2 000 draws (seed 7) form the interval. The B40 slices of RR4 use seed 8 and its absolute-cutoff block seeds 11/12; the historical PT_linear continuity table (`Q_NEWV_FIT85_PRIMARY_20260920.md`) used 800 draws. `bootstrap_fraction_positive` is the share of draws with ΔBrier > 0, not a \(p\)-value. Intervals describe match-sampling variation under fixed evaluator, labels and selection; they do not include model-fitting or label-model uncertainty. [`bootstrap_delta_brier`; `scripts/rr20260920_review_response_rr4_margin.py`; `scripts/rr20260920_q_newv_primary_fit.py`; design §4.2]  
- **CORP decomposition** (Dimitriadis–Gneiting–Jordan): with isotonic recalibration \(p^*=\mathrm{iso}(p)\) fit on the **evaluation sample for diagnosis only**,

\[
\mathrm{MCB}=\mathrm{BS}-\mathrm{BS}_{\mathrm{iso}},
\quad
\mathrm{DSC}=\mathrm{UNC}-\mathrm{BS}_{\mathrm{iso}},
\quad
\mathrm{BS}=\mathrm{MCB}-\mathrm{DSC}+\mathrm{UNC}.
\]

CORP is a **score decomposition** on that sample. No bootstrap intervals were computed for CORP components, for the \(V\to W\) tables, for the horizon and next-objective tables, or for the external cohorts; intervals are reported only for the paired ΔBrier contrasts (RR12, RR4) and the matched quiet contrast (RR3). It is not a fitted production calibrator and not a causal mechanism of gameplay. Absolute MCB can look small relative to UNC≈0.25 while still mattering at the \(O(10^{-3})\) scale of our ΔBrier.

---

## 7. Verification analyses (not new estimands)

| Analysis | Question | Guardrail |
|---|---|---|
| Quiet vs fight \(\lvert\Delta V\rvert\) (matched) | Is fight-interval change larger than quiet matched windows? | Coverage≈28%; not ATT |
| Triad (direction / mean / scale) | Do the three objects disagree across \(p_{\mathrm{pre}}\)? | Not proof of predictability |
| λ·\(s_Q\) small-change filters | Does all-T lift vanish if rows with \(\lvert\Delta\widehat V\rvert<\lambda\,s_Q\) are excluded (subsample evaluation; excluded counts reported)? | Post-hoc; \(s_Q\) often p-only (64.9%) |
| Material / next-objective tables | Does SVI sign correspond to observable nets / upcoming elite objectives? | Correspondence ≠ \(q\) accuracy |
| Horizon h60/90/120 | How often does SVI flip; how often endpoints coincide? | Shared endpoints inflate agree |

---

## 8. What Methods deliberately omit in this version

- Retuning \(\widehat V\) for higher match AUC.  
- Expanding the \(q\) architecture zoo.  
- External probability adapters (deferred thesis/v2).  
- Completing prior deep-learner `\pending` comparisons from the CoG-extension draft.

---

## 9. Evidence trace

Every number reported in Results resolves to one of the following artifacts at commit `21391b2` (T rows) or `69130a4` (S rows, added 2026-09-21).

| Manuscript number | Source file | Field / row |
|---|---|---|
| \(V\to W\) TEST: Brier 0.1552, AUC 0.8542, n 347 234, matches 60 190 | `docs/TEST_BAND_LEDGER_MLP_FIT85_20260920.md` | L5 "Overall" |
| \(V\to W\) CORP: MCB 0.0002, DSC 0.0948 | `docs/REVIEW_RESPONSE_RR6A_CORP_20260920.md` | row `V_to_W_timeline \| overall` |
| Early band [2,10): n 120 186, Brier 0.2283, AUC 0.6642 | same | row `V_to_W_timeline \| t_2_10` |
| \(V_{\mathrm{pre}}\) / \(V_{\mathrm{post}}\): Brier 0.1442 / 0.1181, AUC 0.8750 / 0.9149 | same | rows `V_to_W_eng_pre`, `V_to_W_eng_post` (overall) |
| \(q\) TEST: Brier 0.2355, log loss 0.6637, AUC 0.6403, n 32 981 / 24 020 | `docs/REVIEW_RESPONSE_RR12_RESULTS_20260920.json` | `TEST_all.q_base` |
| ΔBrier \(q\)−PT_flex −0.00373 [−0.00461, −0.00279] | same | `bootstrap.all_q_minus_PT_flex` |
| ΔBrier \(q\)−PT_linear −0.00422 [−0.00515, −0.00323] | same | `bootstrap.all_q_minus_PT_linear` |
| B40 ΔBrier −0.00214 [−0.00422, −0.00003]; n 5 423 / 4 945 | same | `bootstrap.B40_q_minus_PT_flex`; `census.test_B40`; `TEST_B40.q_base.n_matches` |
| \(H\) 0.00181 [−0.00060, 0.00422] | same | `bootstrap.H_B40_minus_outside` |
| \(q\) / PT_flex CORP: MCB 0.00086 / 0.00053, DSC 0.0154 / 0.0113 | `docs/REVIEW_RESPONSE_RR6A_CORP_20260920.md` | rows `q_to_SVI` |
| RR3 quiet contrast 0.0633 [0.0612, 0.0654], n 9 140, coverage 0.277 | `docs/REVIEW_RESPONSE_RR3_CLOSEOUT_20260920.md`, `docs/REVIEW_RESPONSE_RR3_QUIET_20260920.json` | TEST primary contrast |
| RR5a kill-axis agreement 0.904 (all-T) / 0.928 (B40) | `docs/REVIEW_RESPONSE_RR5_RR6B_20260920.md` | RR5a crosstab rows |
| RR5b 6 004/9 934, 4 072/10 277, 8 112 game_ended_in_window | same | L35–L38 |
| RR6b flip 0.015 / 0.007 / 0.019; same-endpoint share 0.594 / 0.810 / 0.594 | same | horizon table |
| RR4 λ rows (all-T ΔBrier ≈ −0.004; n 32 981 → 26 297 at λ = 1) | `docs/REVIEW_RESPONSE_RR4_MARGIN_20260920.md` | λ table |
| EXT KR / NA1 16.13: n 5 202 / 5 312, \(V_{\mathrm{pre}}\) Brier 0.1511 / 0.1514, ΔBrier +0.0026 / +0.0040, ΔMCB 0.0045 / 0.0063, ΔDSC 0.0019 / 0.0023 | `docs/REVIEW_RESPONSE_RRX_EXTERNAL_20260920.md` | summary table |
| Evaluator census (Table 5a) | `docs/A_MLP_expanded_evaluator_meta_20260919.json` | `meta.census` |
| Engagement roles (Table 5b) | `docs/PAPER_COHORT_CONTRACT_20260919.md` §2; `docs/EXPERIMENT_INVENTORY_COHORT_20260919.md` §E | tables |
| Out-of-fold folds (5; per-fold n) | `docs/Q_NEWV_FIT85_OOF_META_SLIM_20260920.json` | per-fold entries |
| Cohort rule `cohort == 1` ⟺ min participation ≥ 4; `fine` classes | `docs/lineage_20260915/cr20260915_common.py`; `docs/lineage_20260915/cohort_manifest.json` | `scale_classes` L37–60; `rules` |
| Per-role T match counts (28 989 Σ folds; 7 550; 7 469; 7 448; 7 632; 24 020; EXT 3 859 / 3 955 / 377 / 80) | `docs/lineage_20260915/cohort_manifest.json` | `sets.<SET>.by_sub_role.<ROLE>.h90.T_matches` |
| Endpoint rule, validity flags | `docs/lineage_20260915/engagement_labels_v3_rules.py` | `endpoint_rule`, `endpoint_validity` |
| Valid rows 566 104 / 566 452; 348 overlap exclusions; realised follow-up 45.493 s / 39.440 s; cap reached 14.286%; next-kill stop ≈78.9% | `docs/lineage_20260915/README.md` (excerpts of `e0ec3d0:docs/tog_delta_v_20260916/manuscript.md` L200–208, L351–358) | tables |
| Detector constants table | `docs/tog_manuscript/sec_definition.tex` | `tab:constants` L451–L641 |
| S TEST \(q_S\): Brier 0.2458, log loss 0.6849, AUC 0.5767; PT_flex_S 0.2492 / 0.6915 / 0.5360; PT_linear_S 0.2495 / 0.6921 / 0.5301; n 101 205 / 49 730; positive rate 0.511 | `docs/SCALE_SPLIT_TvsS_RESULTS_20260920.json` | `table1_S_TEST_identity.{q_S,PT_flex_S,PT_linear_S}`; `.q_S.p_pos` |
| ΔBrier \(q_S\)−PT_flex_S −0.00335 [−0.00379, −0.00288] | same | `table2a_identity.q_S_minus_PT_flex_S` |
| Sensitivity (RR12 calibrator selection) −0.00346 [−0.00382, −0.00307] | same | `table2b_sensitivity.q_S_minus_PT_flex_S` |
| ΔBrier \(q_S\)−PT_linear_S −0.00368 [−0.00412, −0.00321] | `docs/REVIEW_RESPONSE_RR12_RESULTS_20260920_S_qS_id.json` | `bootstrap.all_q_minus_PT_linear` |
| \(q_{T\to S}\)−PT_flex_S −0.00141 [−0.00186, −0.00094] | `docs/REVIEW_RESPONSE_RR12_RESULTS_20260920_S_qT_id.json` | `bootstrap.all_q_minus_PT_flex` |
| \(q_S\)−\(q_{T\to S}\) −0.00194 [−0.00236, −0.00153]; \(q_S\)−\(q_{TS}\) −0.00024 [−0.00039, −0.00009]; \(q_{TS}\)−\(q\) (T rows) +0.00071 [+0.00018, +0.00126] | `docs/SCALE_SPLIT_TvsS_RESULTS_20260920.json` | `table2a_identity.{q_S_minus_q_T_to_S,q_S_minus_q_TS,q_TS_minus_q_T}` |
| S∩B40 ΔBrier −0.00210 [−0.00266, −0.00149]; n 31 675 / 25 039; \(H_S\) +0.00178 [+0.00096, +0.00259]; B40 Brier \(q_S\) 0.2481 / PT_flex_S 0.2502, AUC 0.5530 / 0.4999 | same; `..._S_qS_id.json` | `table2a_identity.q_S_minus_PT_flex_S_B40`; `bootstrap.H_B40_minus_outside`; `TEST_B40.{q_base,PT_flex}` |
| S CORP: \(q_S\) MCB 0.0008 / DSC 0.0048; PT_flex_S 0.0007 / 0.0014 | `docs/SCALE_SPLIT_TvsS_RESULTS_20260920.json` | `table3_CORP_identity` |
| EXT S: n 15 641 / 16 100 / 1 307 / 285; \(V_{\mathrm{pre}}\) Brier 0.1872 / 0.1871 / 0.1829 / 0.1787; ΔBrier −0.0018 / −0.0020 / −0.0033 / +0.0035; ΔMCB 0.0011 / 0.0013; ΔDSC 0.0029 / 0.0033; \(q_S\) MCB 0.0028 / 0.0038; DSC 0.0034 / 0.0035 | same; `docs/REVIEW_RESPONSE_RRX_EXTERNAL_20260920_S.md` | `table4_EXT_S`; summary table |
| S roles (Table 5b): 123 049 / 31 302 / 31 059 / 101 205 rows; 60 820 / 15 254 / 15 282 / 49 730 matches; skirmish detected 101 242; union TRAIN 162 654 / 65 465 | `docs/SCALE_SPLIT_RR0_MANIFEST_20260920.json`; `docs/lineage_20260915/cohort_manifest.json`; `.ai/reports/logs/T009_union.txt` | `row_counts`, `integrity.value_checks.<ROLE>.n_matches`; `sets.MAIN_TEST.skirmish`; union log |
| T positive rate 0.498 (match-weighted) | `docs/REVIEW_RESPONSE_RR12_RESULTS_20260920.json` | `TEST_all.constant.p_pos` |

---

## Source anchors

- V freeze: `V_EVALUATOR_FREEZE_CLOSE_20260920.md`  
- Target roles: `V_Q_TARGET_LOCK_20260920.md`  
- RR0–RRX matrix: `RESPONSE_EVIDENCE_MATRIX_20260920.md`  
- CORP / lit framing: `LIT_RESULT_BRIDGE_RR46_20260920.md`  
- Finish rules: `JOURNAL_FINISH_LOCK_20260920.md`  
- Engagement definition lineage: `docs/tog_manuscript/sec_definition.tex` (read-only)  
- Procedure code: `scripts/rr20260919_v_redesign_fit_wave4_corrected.py`, `scripts/rr20260920_q_train_oof_mlp_folds.py`, `scripts/rr20260920_q_build_newv_labels.py`, `scripts/rr20260920_q_newv_primary_fit.py`, `scripts/rr20260920_review_response_rr12.py`, `scripts/forecast_diagnostics.py`
