# RR1 execution addendum — design proposal vs shipped run

**Date:** 2026-09-20  
**Status:** EXECUTION LOCK (closes design–code gaps for RR1 without reopening V/q)  
**Shipped results:** [REVIEW_RESPONSE_RR12_RESULTS_20260920.md](REVIEW_RESPONSE_RR12_RESULTS_20260920.md)  
**RR0:** [REVIEW_RESPONSE_RR0_MANIFEST_20260920.json](REVIEW_RESPONSE_RR0_MANIFEST_20260920.json)  
**Parent design:** [REVIEW_RESPONSE_EXPERIMENT_DESIGN_20260920.md](REVIEW_RESPONSE_EXPERIMENT_DESIGN_20260920.md)

This addendum does **not** invalidate the RR1/RR2 numbers. It states what was actually run so manuscript wording cannot claim an unexecuted procedure.

---

## Allowed claim (unchanged substance)

> On the 15.16 holdout, frozen \(q\) had lower probability-forecast loss than the **tested** flexible \(p_{\mathrm{pre}}+\)time baseline (`PT_flex`). The all-T lift is clearer than B40; B40 shows a small exploratory improvement whose CI upper bound is near 0 and does **not** clear \(\tau=0.001\) across the whole interval.

Not claimed: complete removal of initial-edge effects; tactical mechanism discovery; identical lift in every narrow bin.

---

## Design proposal vs shipped RR1

| Item | Design proposal text | Shipped RR1 (`f8b2753` / RR12 artifacts) |
|---|---|---|
| Spline knots | TRAIN OOF **match-weighted quantile** knots | `SplineTransformer(knots='uniform')` default — **uniform spacing** |
| Fit weights | Record **mean-1** normalization of \(1/n_m\) | Raw \(w_i=1/n_m\) passed to Logistic (evaluation Brier invariant to scale; **fit effective \(C\)** is not) |
| Hyperparam × calibrator | Select baseline settings **and** identity/sigmoid on Q_SELECT | **Two-stage:** (1) pick knots/\(C\) on **raw** Q_SELECT Brier; (2) then identity vs sigmoid for that winner |
| \(q\) weights | Frozen | Frozen `logit_state` (confirmed) |
| \(g_q\) | Optional increasing sigmoid | Fit on Q_CAL; **identity chosen** for \(q\) and `PT_flex` |
| B40 in selection | Forbidden | Not used for selection |

**Manuscript rule:** say “uniform-knot cubic spline + tensor interaction Logistic (`PT_flex`) under the two-stage Q_SELECT procedure,” not “weighted-quantile knots with joint config×calibrator search.”

Optional follow-up (not required to keep current claim): re-fit the **same** low-dim candidate grid with quantile knots + mean-1 weights + joint identity/sigmoid as `PT_flex_quantile` **alongside** the shipped uniform run — do not overwrite.

---

## Reporting vocabulary

| Field / phrase | Meaning |
|---|---|
| `p_gt0` in JSON | **`bootstrap_fraction_positive`**: share of match-bootstrap draws with ΔBrier\((q-\mathrm{baseline})>0\). **Not** a classical hypothesis-test \(p\)-value and not \(P(q\text{ worse})\). |
| B40 CI ≈ touches 0 from below | Small exploratory support; do **not** write “clear improvement of at least \(\tau=0.001\).” |
| Narrow bins | Diagnostic; do not cherry-pick 0.50–0.55 or declare “no information” where CI covers 0. |
| \(H\) CI covers 0 | Do **not** claim B40 is significantly harder than outside. |

---

## What this closes / what remains

| Closed now | Still open |
|---|---|
| RR0 digests + TEST `(match,s)` parity with RR12 predictions | RR3 new-V quiet |
| Honest RR1 execution description | RR4 (after RR3), RR5 polish, RR6 calib, RRX dual-stage |
| B40 wording tempered vs all-T | — |
