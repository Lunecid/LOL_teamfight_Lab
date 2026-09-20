# Review-response results × econometrics literature

**Date:** 2026-09-20  
**Status:** **CLOSED** as interpretation lock at commit `948b36a` (CORP score-gap MCB; V/q weights unchanged)  
**Prior note:** An earlier draft used \(\langle(p-p^*)^2\rangle\) as “MCB”; that is **not** CORP. Tables below are from the corrected `forecast_diagnostics.py` (exact BS = MCB − DSC + UNC).  
**Lit pack:** [literature/LOL_ECONOMETRICS_LITERATURE_20260920/](literature/LOL_ECONOMETRICS_LITERATURE_20260920/)  
**Application map:** [ECONOMETRICS_LIT_APPLICATION_LOCK_20260920.md](ECONOMETRICS_LIT_APPLICATION_LOCK_20260920.md)  
**Evidence matrix:** [RESPONSE_EVIDENCE_MATRIX_20260920.md](RESPONSE_EVIDENCE_MATRIX_20260920.md)

**Close-out (collaborator-confirmed):** CORP formula, lit→result scope, and RR4 (all-T + B40 λ, \(s_Q\) fallback disclosure) are frozen for manuscript wording. Do not reopen for redesign. Next scheduled packs: **RR5 · RR6b · RRX** (external CORP not inferred from 15.16).

Framing: **direction-of-change forecasting and conditional predictive ability on estimated WP**, not stock prediction on LoL. \(\widehat V\) is learned probability, not a market price.

---

## 0. Thesis paragraph (manuscript-facing)

> This study separates **direction**, **mean change**, and **absolute scale** of estimated post-fight win-probability moves. Direction predictability is assessed by holdout comparisons of frozen \(q\) against flexible \(p,t\) baselines (RR1/RR2); whether that gain depends only on tiny \(\lvert\Delta V\rvert\) cases is checked with **post-hoc** λ·\(s_Q\) sensitivity (RR4), not by treating large scale as proof of predictability. Probability quality of the WP evaluator \(V\to W\) and of the direction predictor \(q\to\mathrm{SVI}\) is diagnosed **separately** with proper scores and CORP (RR6a). Fight vs quiet \(\lvert\Delta V\rvert\) contrasts are background-change references, not identified causal ATT. Martingale path filters and forecast combinations remain deferred separate versions.

---

## 1. Literature roles (kept)

| Strand | Role here | Guardrail |
|---|---|---|
| Christoffersen–Diebold (2006) | Direction ≠ mean ≠ scale as **objects** | Not “volatility ⇒ sign predictable”; not proof of our \(q\) |
| Gneiting–Raftery (2007) | Brier/log loss for both stages | Proper score ≠ substantive SVI validity |
| Dimitriadis et al. CORP (2021) | MCB/DSC/UNC via **score gaps** | No TEST isotonic as frozen-model upgrade |
| Giacomini–White (2006) | Motivate condition-wise contrasts | Do not rename match-bootstrap ΔBrier as GW |
| Clements–Harvey (2010) | Motivate later \(q_\lambda\) version | Do not relabel RR1 as encompassing |
| Brown–Warner (1985) | Event-window **scale** vs signed mean | Not causal ATT; not ΔV = return |
| Maymin (2021) | LoL WP-change prior | Not “first LoL WP-change study” |
| Foster–Stine (2021) | Deferred `V_v2` path diagnostics | No martingale forced on fight-selected windows |

---

## 2. CORP definition (corrected)

With match weights and isotonic \(p^*=\mathrm{iso}(p)\) on the evaluation sample (diagnosis only):

\[
\mathrm{MCB}=\mathrm{BS}-\mathrm{BS}_{\mathrm{iso}},\qquad
\mathrm{DSC}=\mathrm{UNC}-\mathrm{BS}_{\mathrm{iso}},\qquad
\mathrm{BS}=\mathrm{MCB}-\mathrm{DSC}+\mathrm{UNC}
\]

(exact identity; asserted in code). Using \(\langle(p-p^*)^2\rangle\) as MCB **drops the cross term** and must not be reported as CORP.

---

## 3. RR6a — corrected CORP on 15.16 MAIN

**Artifacts:** [REVIEW_RESPONSE_RR6A_CORP_20260920.md](REVIEW_RESPONSE_RR6A_CORP_20260920.md)

### \(V\to W\) (timeline overall)

| Brier | MCB | DSC | UNC | AUC |
|---:|---:|---:|---:|---:|
| 0.1552 | 0.00016 | 0.0948 | 0.2499 | 0.854 |

Most of the gap below UNC is discrimination; early \(t\in[2,10)\) remains weaker (time-conditional quality, not a license to retune per-band \(V\) on TEST).

### \(q\to\mathrm{SVI}\) vs PT_flex (same rows → same UNC)

| Cell | Model | Brier | MCB | DSC | UNC |
|---|---|---:|---:|---:|---:|
| all-T | q | 0.2355 | **0.00086** | **0.0154** | 0.2500 |
| all-T | PT_flex | 0.2392 | 0.00053 | 0.0113 | 0.2500 |
| B40 | q | 0.2474 | **0.00178** | **0.00436** | 0.2500 |
| B40 | PT_flex | 0.2495 | 0.00058 | 0.00102 | 0.2500 |

Because UNC is shared, \(\Delta\mathrm{BS}=\Delta\mathrm{MCB}-\Delta\mathrm{DSC}\):

| Cell | ΔBrier(\(q\)−PT) | ΔMCB | ΔDSC |
|---|---:|---:|---:|
| all-T | −0.00373 | +0.00033 | +0.00406 |
| B40 | −0.00214 | +0.00120 | +0.00334 |

**Supported reading:** \(q\) improves net Brier because **extra discrimination outweighs a somewhat larger miscalibration component** relative to PT_flex. Absolute MCB values are small vs UNC≈0.25, but they are **not** negligible vs our \(O(10^{-3})\) ΔBrier — do **not** write “calibration is not the problem.”

**Scope:** These CORP numbers are **15.16 MAIN** diagnostics. They do **not** identify why 16.x external transfer fails; that requires RRX dual-stage \(V\to W\) and \(q\to\mathrm{SVI}\) on external cohorts.

---

## 4. RR4 — triad as object separation; λ as post-hoc sensitivity

**Artifacts:** [REVIEW_RESPONSE_RR4_MARGIN_20260920.md](REVIEW_RESPONSE_RR4_MARGIN_20260920.md)

### Triad (objects differ — not proof of predictability)

Near \(p_{\mathrm{pre}}\approx0.5\), \(E[\Delta V]\approx0\) while \(E[\lvert\Delta V\rvert]\) is largest; at \(p\in[0.9,1.0]\), \(P(\Delta V>0)\approx0.63\) while \(E[\Delta V]\approx-0.015\). That shows **direction and mean can disagree**. Large scale alone does **not** imply \(P(\Delta V>0\mid X)\neq0.5\) (fair coin ±0.18 counterexample). Binary SVI stays primary because **that is the research question**; holdout RR1/RR2 show direction predictability.

### \(s_Q\) support (disclose coarseness)

| Fallback level | n | share |
|---|---:|---:|
| `p` only | 21401 | 64.9% |
| `p,t` | 7018 | 21.3% |
| `p,t,L` | 4562 | 13.8% |

L>120s: 3031 rows (9.2%) — outside RR3 equal-L match support; \(s_Q\) there is a coarser reference, not same-length quiet.

### λ · \(s_Q\) (post-hoc; do not pick best TEST λ)

| λ | all-T cov | ΔBrier all-T [CI95] | ΔBrier B40 [CI95] |
|---:|---:|---|---|
| 0 | 1.00 | −0.0037 [−0.0046, −0.0028] | −0.0021 [−0.0042, −0.00007] |
| 0.25 | 0.94 | −0.0042 [−0.0051, −0.0032] | −0.0023 [−0.0044, −0.0003] |
| 0.5 | 0.89 | −0.0041 [−0.0050, −0.0031] | −0.0028 [−0.0049, −0.0006] |
| 1.0 | 0.80 | −0.0042 [−0.0051, −0.0032] | −0.0029 [−0.0052, −0.0007] |

**Supported claim:** Under the tested small-|ΔV| exclusion rules, **all-T (and, separately, B40-within-slice) q−PT_flex improvement does not disappear.** Not: “small-ΔV labels are fine,” and not: “pre-select large moves to deploy.”

---

## 5. RR3 / RR1–2 (unchanged roles)

- **RR3:** matched fight vs quiet \(\lvert\Delta V\rvert\) scale contrast (Brown–Warner spirit; not ATT).  
- **RR1/RR2:** holdout ΔBrier and condition-wise \(H\) (encompassing / CPA **motivation** only).

---

## 6. Deferred (unchanged)

Foster–Stine path filter, Patton–Timmermann multi-horizon, Brill–Yurko–Wyner ΔV uncertainty, Clements–Harvey \(q_\lambda\), RR5 / RR6b / RRX.
