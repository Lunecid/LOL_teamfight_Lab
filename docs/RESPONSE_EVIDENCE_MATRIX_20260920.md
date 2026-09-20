# Review-response evidence matrix (RR1–RR6)

**Date:** 2026-09-20  
**Status:** PLAN LOCKED · **RR0–RR4 + RR6a CLOSED** at `948b36a` · next: RR5 / RR6b / RRX  
**Design:** [REVIEW_RESPONSE_EXPERIMENT_DESIGN_20260920.md](REVIEW_RESPONSE_EXPERIMENT_DESIGN_20260920.md)  
**Lit application:** [ECONOMETRICS_LIT_APPLICATION_LOCK_20260920.md](ECONOMETRICS_LIT_APPLICATION_LOCK_20260920.md)  
**Lit × results bridge (CLOSED):** [LIT_RESULT_BRIDGE_RR46_20260920.md](LIT_RESULT_BRIDGE_RR46_20260920.md)  
**RR1 execution honesty:** [REVIEW_RESPONSE_RR1_EXECUTION_ADDENDUM_20260920.md](REVIEW_RESPONSE_RR1_EXECUTION_ADDENDUM_20260920.md)  
**Interpretation lock commit:** `948b36a` · V / SVI / q **unchanged**

Purpose chain stays: **predict post-fight WP direction from pre-info** → improve measurement via \(V\) → predict with \(q\) → verify (A) beyond initial edge, (B) what ΔV measures.

Framing (lit): **direction-of-change + conditional predictive ability on estimated WP**, not stock-prediction-on-LoL. \(\widehat V\) is learned probability, not a market price.

| ID | Review ask | Reuse | New work | Status |
|---|---|---|---|---|
| **RR0** | Version / keys / weights | Labels, V sha, q weights | `response_manifest` | **done** — [RR0 manifest](REVIEW_RESPONSE_RR0_MANIFEST_20260920.json) |
| **RR1** | Flex \(p,t\) baselines + optional \(g_q\) | OOF labels, frozen `logit_state`, `PT_linear` | `b_spline`, `PT_flex`, calibrators | **done** + [execution addendum](REVIEW_RESPONSE_RR1_EXECUTION_ADDENDUM_20260920.md) |
| **RR2** | All-T ∥ B40 ∥ narrow bins; \(H\) | RR1 frozen preds | Bin tables + paired CI | **done** — [RR12 results](REVIEW_RESPONSE_RR12_RESULTS_20260920.md) |
| **RR3** | Quiet vs fight ΔV (new V) | — | Matched quiet under fit85 | **done** + [close-out](REVIEW_RESPONSE_RR3_CLOSEOUT_20260920.md) |
| **RR4** | Small-ΔV sensitivity | RR12 preds; RR3 \(s_Q\) | λ·\(s_Q\) + abs cutoffs; triad; B40-within-λ | **CLOSED** — [RR4 margin](REVIEW_RESPONSE_RR4_MARGIN_20260920.md) |
| RR5a | Material correspondence | NEWV verify table | Denominators / crosstabs | **next** |
| RR5b | Next objective after endpoint | — | 180s window events | **next** |
| **RR6a** | \(V\to W\) / \(q\to\mathrm{SVI}\) CORP | fit85 bundle; RR12 preds | CORP MCB=BS−BS_iso (score-gap) | **CLOSED** — [RR6a CORP](REVIEW_RESPONSE_RR6A_CORP_20260920.md) · lock `948b36a` |
| RR6b | Horizon / V-spec stability | h60/90/120, peer flips | Common-support flips | **next** |
| RRX | External dual stage | Transfer 16.x tables | \(V\to W\) + \(q\to\mathrm{SVI}\) split (do not infer from 15.16 CORP) | **next** |

### Lit → RR map (this pass)

| Lit | Applied in |
|---|---|
| Dimitriadis–Gneiting–Jordan CORP + Gneiting–Raftery | RR6a (`forecast_diagnostics.py`) |
| Christoffersen–Diebold (dir vs mean vs vol) | RR4 triad + binary-SVI retention |
| Brown–Warner (event scale) | RR3/RR4 interpretation (not causal ATT) |
| Giacomini–White / Clements–Harvey | Motivation only — do not rename match-bootstrap ΔBrier |

## Allowed headline (RR1–RR4 / RR6a)

15.16: frozen \(q\) beats tested `PT_flex` on all-T (B40 exploratory). Matched quiet under fit85: fight \(\lvert\Delta V\rvert\) ≫ quiet on matchable subset (not ATT). Corrected CORP: \(q\)’s net Brier gain = higher DSC minus somewhat higher MCB vs PT_flex (ΔBS=ΔMCB−ΔDSC). λ·\(s_Q\) post-hoc: all-T (and B40-within-slice) ΔBrier sign preserved; most \(s_Q\) rows are p-only fallback. 15.16 CORP does not explain 16.x transfer.

## Forbidden

- Reopen V architecture or q model zoo
- Retune B40 thresholds on TEST
- Promote RR sensitivity slices / best-λ to primary without stating post-hoc
- Reuse old_V quiet ratios for RR3
- Apply TEST-fit CORP isotonic as frozen-model “fix”
- Equate ΔV with financial returns or claim martingale ⇒ sign unpredictable
- Claim large \(E[|\Delta V|]\) alone implies direction predictability
- Claim “calibration OK / external fail = discrimination only” from MAIN CORP
- Use \(\langle(p-p^*)^2\rangle\) as CORP MCB
- Rename match-bootstrap ΔBrier as Diebold–Mariano / Giacomini–White