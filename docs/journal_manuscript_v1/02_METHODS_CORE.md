# Methods — core definitions (draft)

*Prose draft for the journal freeze paper. Numbers and procedures cite locked artifacts. Epistemic status is stated here in draft 1, not deferred to a late note.*

---

## 0. Epistemic status (required)

The primary held-out contrasts that flex \(p,t\) baselines and report CORP / external dual-stage were produced in a **review-response** setting after exposure to the 15.16 test patch in earlier analyses. We therefore label the corresponding tables **exploratory** (design tag: `EXPLORATORY_REVIEW_RESPONSE_PRIOR_TEST_EXPOSURE`): they answer the scientific questions with locked weights and documented procedures, but they are **not** confirmatory preregistered tests.

Where execution of the flexible-baseline protocol differed from the written design (uniform spline knots rather than weighted quantiles; fitting weights not mean-normalized; two-stage hyperparameter then calibrator selection), we follow the execution addendum rather than rewriting history [RR1 addendum; RR0 manifest].

---

## 1. Engagement unit

An **engagement** is a kill-anchored cluster in Riot Match-V5 timeline telemetry, built by the v3.3 detector with the rules below. Constants are the values injected into the detector; derivations, intervals and the full constant table are in the definition lineage (`docs/tog_manuscript/sec_definition.tex`, cited by subsection).

- **Kill episodes and the temporal gap \(G\).** Consecutive champion kills of a match are chained into one episode when their gap is at most \(G\). \(G\) is the antimode of the log inter-kill-interval distribution (KDE valley between the within-fight and between-fight modes; a two-component Gaussian mixture decides the sides): \(\hat G = 13.7246\) s over 208 141 matches, run at \(G = 13.700\) s; match-bootstrap 95% CI 12.2–15.7 s; the episode partition has ARI ≥ 0.9 for every gap from 10 to 18 s. [sec_definition §Kill episodes and the temporal boundary \(G\)]
- **Spatial split and the diameter \(D\).** An episode is split so that no two of its kills lie more than \(D\) apart (time-ordered first-fit grouping). \(D\) is the distance at which the rate of consecutive kill pairs sharing a champion (as killer, victim or assister) falls to 0.5: \(\hat D = 4\,263.87\) u over 5 460 008 pairs, run at \(D = 4\,264.0\) u; match-bootstrap 95% CI 4 256–4 274 u, which reflects sampling only because the estimate interpolates between bin centres 750 u apart. [§The spatial boundary \(D\)]
- **Presence gate (\(R\), \(B\), \(M\)).** The cutoff is \(\tau = t_{\text{first kill}} - B\) with \(B = 15\) s. An episode becomes an engagement only if, at \(\tau\), at least \(M = 2\) alive champions of **each** team stand within \(R = 1\,600\) u of the first kill's position (the anchor). Alive status is read from the last one-minute frame before \(\tau\); positions come from a 5 s grid interpolated between one-minute frames, with each kill participant's path drawn towards its kill position. \(R\) and \(B\) are rule anchors (champion-death experience-sharing radius; assist window) held fixed by assumption and checked against patch notes 25.14–25.16; \(M\) is a convention. [§Presence gate, lead and anchor]
- **Gold-label attribution window.** The CoG-continuity label `market_event` reads events in \([\tau,\ \max(t_{\text{last kill}}+1\,\text{ms},\ \tau+H))\) with \(H = 35\) s, cut short at an ace, and only inside a disc of radius \(D\) centred on the anchor. The SVI endpoint is defined separately (§1b). [§Which events the label may read]
- **Assembly guards (engineering constants).** Re-engagement merge on, maximum gap 15 s, merge radius 2 000 u; merged-duration cap 60 s; cutoffs in the first 2 min, or less than 35 s before the last frame, rejected; a 30 s observation window in 5 s bins up to \(\tau\); overlap between engagements resolved by participation priority with location radius \(D\); interaction radius 3 000 u for participation counts (shop events included); engagements on which the gold label finds no winner dropped. [`tab:constants`]
- **Per-patch re-estimation.** \(G\) and \(D\) re-estimated per patch (15.14 / 15.15 / 15.16: \(G\) = 14.0 / 13.5 / 13.7 s; \(D\) = 4 285 / 4 265 / 4 241 u) all lie inside the pooled plateau and within 10% of the pooled \(D\), so one pooled definition serves the corpus. [§Re-estimation per patch]
- **Scale classes and the teamfight cohort T.** Participation per side counts that side's champions among the killers, victims and assisters of the engagement's kills plus the actors of other timeline events within 3 000 u of the anchor between \(\tau\) and the last kill; classes use the smaller side's count \(n_{\min}\): *pick* \(n_{\min}\le 1\), *skirmish* \(2\le n_{\min}\le 3\), **teamfight \(n_{\min}\ge 4\)**. The prediction sample of this paper, **T**, is the teamfight class. In the frozen pipeline its membership is carried by the `cohort == 1` flag of the 2026-09-15 cohort-role build together with `valid_h90 == 1` and finite pre/post scores (§5); the code of that cohort build is not in this branch (\pending{T-cohort-build-provenance}). The corpus counts of the definition lineage (532 547 v3.3 engagements from 208 141 matches, of which 109 829 teamfights) come from a different corpus and filter than the pooled T of this paper (113 901 from 210 000 matches) and are not tabulated together. [§Scale classes]

Sensitivity of secondary constants (anchor placement, shop-event exclusion, label horizon, execution kills, position-grid error, the \(G\times D\) sweep) was measured on conference-era or pilot corpora and is documented in the definition lineage; it has not been re-run under the current corpus. Kill-less proximity exchanges lie outside this detector's instances and are not the prediction target of this paper.

We do **not** treat the gold engagement-winner label `market_event` as the freeze paper’s primary outcome. That label remains relevant for CoG continuity and for secondary correspondence checks, but the prediction target below is defined from the frozen win-probability evaluator.

### 1b. Prediction time and outcome time

- **Prediction time.** The feature cutoff is \(\tau = t_{\text{first kill}} - B\) (\(B = 15\) s). \(q\) and the PT baselines read only the pre-state snapshot \(S_{\mathrm{pre}}\) taken at or just before \(\tau\); the actual endpoint, the actual interval length, post-engagement participants and post states are never inputs. [`V_DYNAMIC_FRAME_CONTRACT_20260919.md`; `REVIEW_RESPONSE_EXPERIMENT_DESIGN_20260920.md` §2.1]
- **Outcome time.** The post state \(S_{\mathrm{end}}\) is the snapshot at the label endpoint under horizon \(h\), primary **h90**, and \(\Delta\widehat V = \widehat V(S_{\mathrm{end}}) - \widehat V(S_{\mathrm{pre}})\). "h90" does not mean that every interval is 90 s long: each engagement has its own length \(L_i = \text{endpoint}_i - \text{pre\_query}_i\), and analyses that need a length (quiet matching, RR3) use \(L_i\). Only engagements with `valid_h90 == 1` (an endpoint state exists under the frozen state pipeline) are labeled; h60 and h120 are sensitivity horizons only. The exact endpoint construction (early match end, truncation) is carried by the 2026-09-15 label build and is not restated in this branch's documents (\pending{h90-endpoint-rule}). [design §2.1 L44, L177; `Q_PREDICTION_DESIGN_CONTRACT_20260920.md` L119]

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

is the locked **logistic / `logit_state`** model trained on out-of-fold SVI labels under the sealed feature order (MAIN feature-order sha16 `103f0b92094847bc` for external score-only application). Inputs use **pre-state only**. For the reported RR contrasts, the selected probability mapping on Q_SELECT was **identity** (no additional increasing calibrator on \(q\)).

Inputs are the 351 numeric pre-state features plus \(p_{\mathrm{pre}} = \widehat V(x_{\mathrm{pre}})\) (352 inputs), standardized and fit with L2-regularized logistic regression; the 10 raw champion-ID slots are excluded. The LightGBM candidate `lgbm_state` used the same numeric block plus the 10 champion IDs as categorical features (362 inputs), so the two learners were not compared on identical inputs. [`scripts/rr20260920_q_newv_primary_fit.py` L185–L225]

### 4.2 Baselines in \(p_{\mathrm{pre}}\) and time

- **PT_linear:** \(\sigma(a + b\,p_{\mathrm{pre}} + c\,t)\).  
- **PT_flex:** flexible spline expansion in \((p_{\mathrm{pre}}, t)\) with locked knot/C configuration from Q_SELECT (`n_knots_p=4`, `n_knots_t=4`, `C=0.01`, degree 3); selected calibrator **identity**.

PT_flex in detail: cubic B-spline bases (`SplineTransformer`, uniform knots, `include_bias=False`, constant extrapolation) on \(p_{\mathrm{pre}}\) (`n_knots_p`) and on time in minutes (`n_knots_t` = 4, fixed), plus the full tensor product of the two bases; standardized and fit with L2 logistic regression (lbfgs) under weights \(1/n_m\). The search grid was `n_knots_p` ∈ {4, 6} × \(C\) ∈ {0.01, 0.1, 1}; the winner by Q_SELECT match-weighted Brier was (4, 4, 0.01). The \(p\)-only spline baseline uses \(p_{\mathrm{pre}}\) alone (`n_knots` 6, \(C\) 0.01); PT_linear is standardized logistic regression on \([p_{\mathrm{pre}}, t]\). As stated in §0, knots are uniform rather than weighted-quantile, fitting weights were not mean-normalized, and hyperparameters were chosen before the identity/sigmoid calibrator (two-stage). [`scripts/rr20260920_review_response_rr12.py` `PTFlexFeatures`, `fit_pt_flex`; RR12 JSON `PT_flex`, `b_spline`]

**Selection and calibration of \(q\).** Candidates and baselines were fit on 15.14 TRAIN (out-of-fold labels), selected on 15.15 Q_SELECT by match-weighted Brier over all T, and then frozen. Q_CAL was used only for LightGBM early stopping; a positive-slope sigmoid \(g_q\) was fit on Q_CAL for every model and compared with the identity on Q_SELECT, and the identity was retained for \(q\) and PT_flex. No separate probability calibrator is applied to the reported \(q\). [`Q_PREDICTION_DESIGN_CONTRACT_20260920.md` §3; RR12 JSON `calibrator_choice`]

**Primary contrast for this paper:** ΔBrier and related scores of frozen \(q\) versus **PT_flex** on the same rows. PT_linear is retained for continuity with earlier tables. Constant and \(p\)-only splines are reported only as context in the RR12 table, not as a model zoo search.

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
| TRAIN | 15.14 | 39 605 | — | \(\widehat V^{(-k)}\) out-of-fold | fit \(q\), PT_flex, \(b(p)\) |
| Q_CAL | 15.15 | 10 390 | — | fit85 | LightGBM early stop; \(g_q\) candidates |
| Q_SELECT | 15.15 | 10 195 | — | fit85 | select knots / \(C\) / calibrator |
| 15.15 remainder | 15.15 | 20 730 | — | — | not assigned (15.15 VAL total 41 315) |
| TEST | 15.16 | 32 981 | 24 020 | fit85 | primary |
| B40 ⊂ TEST | 15.16 | 5 423 | 4 945 | fit85 | conditional |
| EXT KR 16.13 | KR | 5 202 | 10 064 collected | fit85 | score-only |
| EXT NA1 16.13 | NA1 | 5 312 | 10 000 collected | fit85 | score-only |
| EXT KR 16.15 | KR | 507 | 926 collected | fit85 | reported, not pooled |
| EXT KR 16.14 pilot | KR | 101 | 200 collected | fit85 | reported, not pooled |

Pooled T (15.14 + 15.15 + 15.16) = 113 901 engagements, used for measurement tables only; its match count is not recorded in the freeze documents (\pending{pooled-T-matches}). Sample flow: source matches (210 000 KR, patches 15.14–15.16) → detected engagements → teamfight class T → pre-state available and `valid_h90` → finite pre/post scores → role assignment as above; counts for the intermediate stages are not recorded in the freeze documents and are not estimated here.

**Two evaluator paths.**

| Rows | Evaluator defining \(p_{\mathrm{pre}}\), \(\Delta\widehat V\), \(Y_{\mathrm{SVI}}\), B40 | Rule |
|---|---|---|
| 15.14 TRAIN (fitting \(q\) and the PT baselines) | \(\widehat V^{(-k)}\), \(k=1,\dots,5\), match-level folds | No information from fold \(k\)'s matches enters that evaluator's preprocessing, weights, early-stop holdout, or calibrator |
| 15.15 Q_CAL / Q_SELECT, 15.16 TEST, 16.x external | Frozen fit85 bundle | Same bundle, preprocessing and calibration for every row |

External application is **score-only**: no refit of \(V\) or \(q\) weights; fail-closed if keys or feature order mismatch. Small external cohorts are not averaged into a transfer-success claim.

---

## 6. Scores and uncertainty

- **Proper scores:** Brier score and log loss (lower better).  
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

Every number reported in Results resolves to one of the following artifacts at commit `21391b2`.

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

---

## Source anchors

- V freeze: `V_EVALUATOR_FREEZE_CLOSE_20260920.md`  
- Target roles: `V_Q_TARGET_LOCK_20260920.md`  
- RR0–RRX matrix: `RESPONSE_EVIDENCE_MATRIX_20260920.md`  
- CORP / lit framing: `LIT_RESULT_BRIDGE_RR46_20260920.md`  
- Finish rules: `JOURNAL_FINISH_LOCK_20260920.md`  
- Engagement definition lineage: `docs/tog_manuscript/sec_definition.tex` (read-only)  
- Procedure code: `scripts/rr20260919_v_redesign_fit_wave4_corrected.py`, `scripts/rr20260920_q_train_oof_mlp_folds.py`, `scripts/rr20260920_q_build_newv_labels.py`, `scripts/rr20260920_q_newv_primary_fit.py`, `scripts/rr20260920_review_response_rr12.py`, `scripts/forecast_diagnostics.py`
