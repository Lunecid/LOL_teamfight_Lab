# Review-response results × econometrics literature

**Date:** 2026-09-20  
**Status:** RESULT LOCK (interpretation only; V/q weights unchanged)  
**Lit pack:** [literature/LOL_ECONOMETRICS_LITERATURE_20260920/](literature/LOL_ECONOMETRICS_LITERATURE_20260920/)  
**Application map:** [ECONOMETRICS_LIT_APPLICATION_LOCK_20260920.md](ECONOMETRICS_LIT_APPLICATION_LOCK_20260920.md)  
**Evidence matrix:** [RESPONSE_EVIDENCE_MATRIX_20260920.md](RESPONSE_EVIDENCE_MATRIX_20260920.md)

Framing: this project is **direction-of-change forecasting and conditional predictive ability on estimated win probabilities**, not “stock prediction applied to LoL.” \(\widehat V\) is a learned probability, not a market price; ΔV is a percentage-point change in that probability.

---

## 1. What each literature strand licenses (and does not)

| Strand | Paper | What we may claim | What we must not claim |
|---|---|---|---|
| Direction vs mean vs vol | Christoffersen & Diebold (2006) | Binary SVI \(Y=1[\Delta V>0]\) is a distinct target from \(E[\Delta V\mid X]\); report the triad together | That their finance conditions transfer to LoL, or that our q is thereby “proven” |
| Event scale vs signed mean | Brown & Warner (1985) | Matched fights can show larger \(\lvert\Delta V\rvert\) than quiet even if signed means cancel | Causal ATT of fights; ΔV = abnormal return |
| Proper scores | Gneiting & Raftery (2007) | Brier / log loss are theoretically appropriate for \(V\to W\) and \(q\to\mathrm{SVI}\) | That a proper score validates the substantive meaning of SVI |
| CORP reliability | Dimitriadis, Gneiting & Jordan (2021) | Decompose Brier into MCB / DSC / UNC on each stage separately | That TEST-fit isotonic improves frozen \(V\)/\(q\) generalization |
| Conditional predictive ability | Giacomini & White (2006) | Motivate all-T vs B40 vs time-band contrasts (\(H\), bins) | Rename match-bootstrap ΔBrier as a GW CPA test |
| Encompassing | Clements & Harvey (2010) | Motivate later \(q_\lambda\) combinations as a **separate version** | Relabel RR1 ΔBrier as an encompassing test |
| LoL WP-change prior | Maymin (2021) | Cite as direct domain prior for event valuation via WP | “First LoL study to use win-probability change” |
| Path / martingale diagnostics | Foster & Stine (2021) | Deferred: full-path \(Q_m\) / filter as `V_v2` | Force martingale on kill-selected fight windows only |

---

## 2. RR6a ↔ CORP + proper scoring

**Artifacts:** [REVIEW_RESPONSE_RR6A_CORP_20260920.md](REVIEW_RESPONSE_RR6A_CORP_20260920.md) · `outputs/review_response_rr6a_corp_20260920/`

Dimitriadis et al. (2021) CORP writes Brier ≈ **MCB − DSC + UNC**. Gneiting & Raftery (2007) justify using Brier for both stages. We run **two separate diagnostics** (design §10 / M2):

### \(V\to W\) (fit85 timeline & engagement)

| Cell | Brier | MCB | DSC | UNC | AUC |
|---|---:|---:|---:|---:|---:|
| Timeline overall | 0.1552 | 0.0001 | 0.0948 | 0.2499 | 0.854 |
| Eng pre overall | 0.1442 | 0.0003 | 0.1065 | 0.2500 | 0.875 |
| Eng post overall | 0.1181 | 0.0002 | 0.1324 | 0.2500 | 0.915 |

**Reading with the literature.** MCB near zero means miscalibration is not the story for \(V\to W\): almost all of the Brier gap below UNC comes from **DSC** (discrimination of the isotonic recalibration of \(V\)). Early band \(t\in[2,10)\) still has weaker AUC/DSC (timeline 0.66 / 0.022), which is a **time-conditional quality** finding in the spirit of Giacomini–White’s “when does the forecast help?”, not a license to build per-band \(V\) on TEST.

### \(q\to\mathrm{SVI}\) (frozen RR12 preds)

| Cell | Brier | MCB | DSC | UNC | AUC |
|---|---:|---:|---:|---:|---:|
| q all-T | 0.2355 | 0.0004 | 0.0154 | 0.2500 | 0.640 |
| PT_flex all-T | 0.2392 | 0.0002 | 0.0113 | 0.2500 | 0.620 |
| q B40 | 0.2474 | 0.0007 | 0.0044 | 0.2500 | 0.567 |
| PT_flex B40 | 0.2495 | 0.0003 | 0.0010 | 0.2500 | 0.525 |

**Reading.** For \(q\), MCB is again small; the useful contrast vs PT_flex is **higher DSC** (0.0154 vs 0.0113 all-T; 0.0044 vs 0.0010 in B40). That matches the review question CORP was meant to answer: external/hard regimes are not primarily “AUC gone because probabilities are wildly misleveled,” but **limited discrimination** relative to UNC≈0.25. CORP here is **diagnostic only** — we do not apply the TEST isotonic map as a frozen-model fix (application lock).

---

## 3. RR4 ↔ Christoffersen–Diebold triad + small-change sensitivity

**Artifacts:** [REVIEW_RESPONSE_RR4_MARGIN_20260920.md](REVIEW_RESPONSE_RR4_MARGIN_20260920.md) · `outputs/review_response_rr4_margin_20260920/`

Christoffersen & Diebold separate:

\[
P(\Delta V>0\mid p),\qquad E[\Delta V\mid p],\qquad \text{scale }E[\lvert\Delta V\rvert\mid p].
\]

### Triad on TEST (match-weighted means; \(n=32981\))

| \(p_{\mathrm{pre}}\) | \(P(\Delta V>0)\) | \(E[\Delta V]\) | \(E[\lvert\Delta V\rvert]\) |
|---|---:|---:|---:|
| [0.0,0.1] | 0.37 | +0.012 | 0.038 |
| [0.4,0.5] | 0.49 | +0.005 | 0.183 |
| [0.5,0.6] | 0.51 | −0.002 | 0.184 |
| [0.9,1.0] | 0.63 | −0.015 | 0.042 |

**Reading.** Near 0.5, mean ΔV is near zero while absolute scale is **largest** — exactly the pattern that makes a **direction** target informative when a mean-change target looks flat. At the extremes, scale shrinks and the positive-rate tilts with \(p_{\mathrm{pre}}\). This is the empirical reason we keep binary SVI as primary (design §8) while still reporting mean and scale (RR3/RR4), following Christoffersen–Diebold’s separation rather than equating “martingale-like mean” with “sign unpredictable.”

### λ · \(s_Q\) post-hoc filter (DEV quiet \(s_Q\) from RR3 fallback)

| λ | coverage | \(E[\lvert\Delta V\rvert]\) | ΔBrier(\(q\) − PT_flex) [CI95] |
|---:|---:|---:|---|
| 0 | 1.00 | 0.112 | −0.0037 [−0.0046, −0.0028] |
| 0.25 | 0.94 | 0.120 | −0.0042 [−0.0051, −0.0032] |
| 0.5 | 0.89 | 0.127 | −0.0041 [−0.0050, −0.0031] |
| 1.0 | 0.80 | 0.139 | −0.0042 [−0.0051, −0.0032] |

**Reading.** Dropping small \(\lvert\Delta V\rvert\) relative to quiet scale \(s_Q(z)\) shifts composition toward larger moves but **does not flip** the all-T advantage of frozen \(q\) over PT_flex. Per design and lit lock: this filter uses post information, is **not** a deployable abstention rule, and we **do not** promote the λ with the best TEST ΔBrier. Absolute cutoffs {0.005, 0.01, 0.02} are a **different** auxiliary definition (not state-volatility adjusted).

---

## 4. RR3 (already closed) ↔ Brown–Warner event-scale reading

**Artifacts:** [REVIEW_RESPONSE_RR3_QUIET_20260920.md](REVIEW_RESPONSE_RR3_QUIET_20260920.md) · [close-out](REVIEW_RESPONSE_RR3_CLOSEOUT_20260920.md)

Matched no-kill quiet under fit85: TEST \(E[\lvert\Delta V\rvert]\) fight ≈0.080 vs quiet ≈0.017; paired excess ≈0.063 with CI excluding 0 (coverage ≈28% under proposal tolerances). Brown & Warner motivate looking at **event-window volatility / scale**, not only mean signed change. We use that as a **background-change contrast**, not as identified causal fight effects (fights are not random; \(\widehat V\) is not a price).

---

## 5. RR1/RR2 (already closed) ↔ encompassing / conditional ability (motivational)

RR1 ΔBrier(\(q\) − PT_flex) and RR2 cell-wise \(H\) sit next to Clements–Harvey (extra information vs a baseline) and Giacomini–White (condition matters). **Naming lock:** the implemented objects remain match-bootstrap ΔBrier and design \(H\); they are **not** renamed Diebold–Mariano or Giacomini–White tests. A true encompassing / \(q_\lambda\) combination remains a **future version**, selected only on DEV.

---

## 6. Manuscript-facing one-paragraph position

> We evaluate post-fight win-probability **direction** from pre-fight information, using a frozen estimated WP level \(\widehat V\) only as a measurement device. Econometric forecast literature supports treating direction, conditional mean, and scale as distinct objects (Christoffersen–Diebold), scoring both \(V\to W\) and \(q\to\mathrm{SVI}\) with proper scores and CORP decompositions (Gneiting–Raftery; Dimitriadis et al.), and comparing event windows to quieter background change without claiming market-style causal ATT (Brown–Warner spirit; Maymin as LoL prior). Empirically, CORP shows both stages are roughly well leveled (small MCB); \(q\)’s edge over flexible \(p,t\) baselines is discrimination, and small-|ΔV| sensitivity does not overturn that edge. We do not claim martingale WP paths imply unpredictable signs, nor that TEST CORP recalibration upgrades the frozen models.

---

## 7. Still deferred (do not expand this commit)

- Foster–Stine path filter / \(Q_m\) → `V_v2` only  
- Patton–Timmermann multi-horizon rationality on **full** timelines  
- Brill–Yurko–Wyner match-resampled ΔV uncertainty  
- Clements–Harvey style \(q_\lambda\) combination as a new scored version  
- RR5 / RR6b / RRX dual-stage external

---

## Artifact index

| Deliverable | Path |
|---|---|
| Lit BibTeX / papers.json | `docs/literature/LOL_ECONOMETRICS_LITERATURE_20260920/` |
| Application lock | `docs/ECONOMETRICS_LIT_APPLICATION_LOCK_20260920.md` |
| Shared CORP/triad | `scripts/forecast_diagnostics.py` |
| RR6a runner + report | `scripts/rr20260920_review_response_rr6a_corp.py`, `docs/REVIEW_RESPONSE_RR6A_CORP_20260920.md` |
| RR4 runner + report | `scripts/rr20260920_review_response_rr4_margin.py`, `docs/REVIEW_RESPONSE_RR4_MARGIN_20260920.md` |
| RR3 close-out | `docs/REVIEW_RESPONSE_RR3_CLOSEOUT_20260920.md` |
