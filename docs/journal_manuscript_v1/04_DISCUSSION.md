# Discussion (draft)

---

## 1. What this paper establishes

The conference line of work asked how much **public pre-fight information** can support engagement outcome prediction. This journal version keeps that purpose but changes the **measurement object**: instead of predicting a gold engagement winner alone, we define interval **direction** of estimated match win probability under a frozen evaluator \(\widehat V\), and ask whether pre-state predicts that direction **beyond** flexible summaries of initial edge and time.

On held-out patch 15.16, a frozen tabular \(q\) improves Brier score relative to the tested PT_flex baseline by about **0.0037** (AUC 0.6403 vs 0.6200). The gain is real under the locked protocol and small in absolute terms. In the balanced B40 slice the gain is smaller still and should be read cautiously. On the main external 16.13 cohorts the Brier advantage **disappears**, even though discrimination components remain favorable to \(q\) in the score decomposition.

Together, these results answer the research questions as **scoped predictability plus transfer limits**, not as a claim of a strong, portable fight tip model.

---

## 2. CoG continuity and contribution framing

Relative to the CoG study, the contribution is not “a higher winner AUC” and not “a new deep architecture.” It is:

1. a **match-linked** definition of post-engagement value change (ΔV / SVI) under a frozen WP evaluator;  
2. a **verification stack** that separates measurement quality (\(V\to W\)) from direction predictability (\(q\to\mathrm{SVI}\));  
3. an explicit test of **whether lift survives** flexed \(p,t\) baselines, small-change filters, and later environments.
---

## 3. Dependence on the evaluator

Because \(Y_{\mathrm{SVI}}\) is the sign of \(\Delta\widehat V\), every predictability claim is conditional on the frozen fit85 evaluator and engagement endpoints. Correspondence with kill and objective nets supports that SVI is not arbitrary noise, but shared inputs and common causes prevent reading those tables as causal fight effects or as \(q\) accuracy.

We deliberately did **not** retune \(V\) to raise direction AUC: doing so would change the estimand. Residual disagreement across peer evaluators (~9% sign disagreement in reused tables) remains a measurement limitation, not a license to keep searching architectures inside this paper’s version boundary.

---

## 4. Reading the score decomposition (not a game “mechanism”)

On 15.16, the Brier gap between \(q\) and PT_flex co-moves with a larger discrimination component and a somewhat larger miscalibration component for \(q\). On KR/NA1 16.13, discrimination still favors \(q\) while miscalibration dominates the net loss. These statements describe **how proper scores decompose on the evaluation samples**. They do not identify a causal mechanism inside the game client, and they do not prove that a low-dimensional recalibrator would restore transfer. An adapter experiment is deferred to a separate thesis/v2 line under freeze rules.

---

## 5. What we do not claim (Forbidden list, explicit)

- We do **not** claim that public telemetry cannot exceed AUC 0.64 under other designs.  
- We do **not** claim that the modest B40 lift proves that unobserved combat execution is the missing factor.  
- We do **not** claim that external failure is “only calibration” or that recalibration will fix transfer.  
- We do **not** claim an exhaustive model ranking or an optimal architecture.  
- We do **not** equate material/next-objective correspondence with \(q\) accuracy or with causal ATT from quiet contrasts.

---

## 6. Limitations (results-facing)

- **Exploratory status:** flexed-baseline and CORP/EXT panels follow prior TEST exposure; intervals are match-bootstrap contrasts, not renamed formal forecast tests.  
- **Estimated labels:** SVI is model-defined; absolute fight value is not observed.  
- **Quiet match coverage** and **\(s_Q\) coarseness** limit how strongly small-ΔV analyses can speak.  
- **External ΔV** is not validated merely by \(V_{\mathrm{pre}}\) Brier ≈0.15.  
- **Prior CoG-extension learner comparisons** remain incomplete and are out of scope for this freeze.

Deferred follow-up work is listed in the working notes, not in this paper.
