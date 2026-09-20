# Methods — core definitions (draft)

*Prose draft for the journal freeze paper. Numbers and procedures cite locked artifacts. Epistemic status is stated here in draft 1, not deferred to a late note.*

---

## 0. Epistemic status (required)

The primary held-out contrasts that flex \(p,t\) baselines and report CORP / external dual-stage were produced in a **review-response** setting after exposure to the 15.16 test patch in earlier analyses. We therefore label the corresponding tables **exploratory** (design tag: `EXPLORATORY_REVIEW_RESPONSE_PRIOR_TEST_EXPOSURE`): they answer the scientific questions with locked weights and documented procedures, but they are **not** confirmatory preregistered tests.

Where execution of the flexible-baseline protocol differed from the written design (uniform spline knots rather than weighted quantiles; fitting weights not mean-normalized; two-stage hyperparameter then calibrator selection), we follow the execution addendum rather than rewriting history [RR1 addendum; RR0 manifest].

---

## 1. Engagement unit

An **engagement** is a kill-anchored cluster in Riot Match-V5 timeline telemetry, using the project’s locked geometric and temporal constants (gap, diameter, presence gate, pre-kill cutoff). Construction details and sensitivity to those constants are documented in the engagement-definition lineage (`docs/tog_manuscript/sec_definition.tex` and associated artifacts). Kill-less proximity exchanges are acknowledged as outside this detector’s instances; they are not the primary prediction target of this paper.

We do **not** treat the gold engagement-winner label `market_event` as the freeze paper’s primary outcome. That label remains relevant for CoG continuity and for secondary correspondence checks, but the prediction target below is defined from the frozen win-probability evaluator.

---

## 2. Frozen win-probability evaluator \(\widehat V\)

Let \(W\in\{0,1\}\) be the match winner (blue = 1). The evaluator

\[
\widehat V(x_t)=\widehat{\Pr}(W=1\mid x_t)
\]

is a multilayer perceptron on an expanded tabular state representation, fit under the locked **fit85** match-holdout protocol (`A_MLP_expanded`, bundle sha16 `ac459cc4397630a9`). Weights are **frozen** for all ΔV / SVI / \(q\) analyses in this paper. Changing \(\widehat V\) would redefine \(p_{\mathrm{pre}}\), \(\Delta\widehat V\), \(Y_{\mathrm{SVI}}\), and balance slices (including B40); we therefore do not retune \(V\) to improve direction AUC.

**Role of \(V\):** a **measurement** model of match win probability at a snapshot. Its quality is assessed by proper scores and discrimination of \(W\) (\(V\to W\)), separately from whether pre-state predicts the **sign** of engagement-interval change (\(q\to\mathrm{SVI}\)).

---

## 3. Interval change and signed value indicator (SVI)

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

Interpretation (locked): \(Y_{\mathrm{SVI}}=1\) means the **estimated** blue win probability rose over the engagement interval under the frozen evaluator — not that blue necessarily became the match favorite, and not a causal attribution that “the fight alone” produced exactly \(\Delta\widehat V\) percentage points (other concurrent state updates may enter the same window).

Primary prediction target: **direction** \(Y_{\mathrm{SVI}}\). Magnitude regression on \(\Delta\widehat V\) is out of scope for this journal version.

**B40:** engagements with \(p_{\mathrm{pre}}=\widehat V(x_{\mathrm{pre}})\) in a balanced neighborhood used in the locked tables (40-point-style balance slice as implemented in RR12; n=5423 on 15.16 TEST). Reported as a **key conditional** evaluation alongside all-T, not as a retuned primary.

---

## 4. Direction predictor \(q\) and baselines

### 4.1 Frozen \(q\)

\[
q(x_{\mathrm{pre}})
=
\widehat{\Pr}\!\left(Y_{\mathrm{SVI}}=1\mid x_{\mathrm{pre}}\right)
\]

is the locked **logistic / `logit_state`** model trained on out-of-fold SVI labels under the sealed feature order (MAIN feature-order sha16 `103f0b92094847bc` for external score-only application). Inputs use **pre-state only**. For the reported RR contrasts, the selected probability mapping on Q_SELECT was **identity** (no additional increasing calibrator on \(q\)).

### 4.2 Baselines in \(p_{\mathrm{pre}}\) and time

- **PT_linear:** \(\sigma(a + b\,p_{\mathrm{pre}} + c\,t)\).  
- **PT_flex:** flexible spline expansion in \((p_{\mathrm{pre}}, t)\) with locked knot/C configuration from Q_SELECT (`n_knots_p=4`, `n_knots_t=4`, `C=0.01`, degree 3); selected calibrator **identity**.

**Primary contrast for this paper:** ΔBrier and related scores of frozen \(q\) versus **PT_flex** on the same rows. PT_linear is retained for continuity with earlier tables. Constant and \(p\)-only splines are reported only as context in the RR12 table, not as a model zoo search.

We do not expand Transformer / GNN / TabM candidates for this version.

---

## 5. Data splits and cohorts

| Role | Patch / region | Use |
|---|---|---|
| Training / selection lineage | Earlier patches under sealed OOF / Q_SELECT protocols | Fit \(V\) (fit85), fit \(q\), select PT_flex / calibrators |
| Primary held-out evaluation | **15.16** TEST engagements | n=32 981 rows, 24 020 matches (all-T tables) |
| External score-only | **KR 16.13**, **NA1 16.13** (main); smaller 16.14/16.15 cohorts scoped | Dual-stage \(V\to W\) and \(q\to\mathrm{SVI}\) on **common-valid** rows |

External application is **score-only**: no refit of \(V\) or \(q\) weights; fail-closed if keys or feature order mismatch. Small external cohorts are not averaged into a transfer-success claim.

---

## 6. Scores and uncertainty

- **Proper scores:** Brier score and log loss (lower better).  
- **Ranking:** AUC (reported; not the sole success criterion).  
- **Paired contrast:** ΔBrier(\(q\) − baseline) with **match-cluster bootstrap** 95% intervals. We do **not** rename these intervals as Diebold–Mariano or Giacomini–White tests.  
- **CORP decomposition** (Dimitriadis–Gneiting–Jordan): with isotonic recalibration \(p^*=\mathrm{iso}(p)\) fit on the **evaluation sample for diagnosis only**,

\[
\mathrm{MCB}=\mathrm{BS}-\mathrm{BS}_{\mathrm{iso}},
\quad
\mathrm{DSC}=\mathrm{UNC}-\mathrm{BS}_{\mathrm{iso}},
\quad
\mathrm{BS}=\mathrm{MCB}-\mathrm{DSC}+\mathrm{UNC}.
\]

CORP is a **score decomposition** on that sample. It is not a fitted production calibrator and not a causal mechanism of gameplay. Absolute MCB can look small relative to UNC≈0.25 while still mattering at the \(O(10^{-3})\) scale of our ΔBrier.

---

## 7. Verification analyses (not new estimands)

| Analysis | Question | Guardrail |
|---|---|---|
| Quiet vs fight \(\lvert\Delta V\rvert\) (matched) | Is fight-interval change larger than quiet matched windows? | Coverage≈28%; not ATT |
| Triad (direction / mean / scale) | Do the three objects disagree across \(p_{\mathrm{pre}}\)? | Not proof of predictability |
| λ·\(s_Q\) small-change filters | Does all-T lift vanish if tiny ΔV rows are down-weighted? | Post-hoc; \(s_Q\) often p-only (64.9%) |
| Material / next-objective tables | Does SVI sign correspond to observable nets / upcoming elite objectives? | Correspondence ≠ \(q\) accuracy |
| Horizon h60/90/120 | How often does SVI flip; how often endpoints coincide? | Shared endpoints inflate agree |

---

## 8. What Methods deliberately omit in this version

- Retuning \(\widehat V\) for higher match AUC.  
- Expanding the \(q\) architecture zoo.  
- External probability adapters (deferred thesis/v2).  
- Completing prior deep-learner `\pending` comparisons from the CoG-extension draft.

---

## Source anchors

- V freeze: `V_EVALUATOR_FREEZE_CLOSE_20260920.md`  
- Target roles: `V_Q_TARGET_LOCK_20260920.md`  
- RR0–RRX matrix: `RESPONSE_EVIDENCE_MATRIX_20260920.md`  
- CORP / lit framing: `LIT_RESULT_BRIDGE_RR46_20260920.md`  
- Finish rules: `JOURNAL_FINISH_LOCK_20260920.md`
