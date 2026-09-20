# Review-response evidence matrix (RR1–RR6)

**Date:** 2026-09-20  
**Status:** PLAN LOCKED · **RR0–RR3 closed** · RR4 next (then RR5–6 / RRX)  
**Design:** [REVIEW_RESPONSE_EXPERIMENT_DESIGN_20260920.md](REVIEW_RESPONSE_EXPERIMENT_DESIGN_20260920.md)  
**RR1 execution honesty:** [REVIEW_RESPONSE_RR1_EXECUTION_ADDENDUM_20260920.md](REVIEW_RESPONSE_RR1_EXECUTION_ADDENDUM_20260920.md)  
**Base commit:** `f8b2753` (+ local RR0/RR3 docs) · V / SVI / q **unchanged**

Purpose chain stays: **predict post-fight WP direction from pre-info** → improve measurement via \(V\) → predict with \(q\) → verify (A) beyond initial edge, (B) what ΔV measures.

| ID | Review ask | Reuse | New work | Status |
|---|---|---|---|---|
| **RR0** | Version / keys / weights | Labels, V sha, q weights | `response_manifest` | **done** — [RR0 manifest](REVIEW_RESPONSE_RR0_MANIFEST_20260920.json) |
| **RR1** | Flex \(p,t\) baselines + optional \(g_q\) | OOF labels, frozen `logit_state`, `PT_linear` | `b_spline`, `PT_flex`, calibrators | **done** + [execution addendum](REVIEW_RESPONSE_RR1_EXECUTION_ADDENDUM_20260920.md) |
| **RR2** | All-T ∥ B40 ∥ narrow bins; \(H\) | RR1 frozen preds | Bin tables + paired CI | **done** — [RR12 results](REVIEW_RESPONSE_RR12_RESULTS_20260920.md) |
| **RR3** | Quiet vs fight ΔV (new V) | — | Matched quiet under fit85 | **done** — [RR3 quiet](REVIEW_RESPONSE_RR3_QUIET_20260920.md) |
| RR4 | Small-ΔV sensitivity | RR1 preds; RR3 \(s_Q\) | λ grid post-hoc | **next** |
| RR5a | Material correspondence | NEWV verify table | Denominators / crosstabs | reuse+polish |
| RR5b | Next objective after endpoint | — | 180s window events | scheduled |
| RR6a | \(V\to W\) time-band calibration | Timeband AUCs | Reliability / slope | scheduled (parallel OK) |
| RR6b | Horizon / V-spec stability | h60/90/120, peer flips | Common-support flips | reuse+polish |
| RRX | External dual stage | Transfer 16.x tables | \(V\to W\) + \(q\to\mathrm{SVI}\) split | scheduled |

## Allowed headline (RR1–RR3)

15.16: frozen \(q\) beats tested `PT_flex` on all-T (B40 exploratory). Matched no-kill windows under fit85 show **much smaller** \|ΔV\| than fights (TEST E[|ΔV|] 0.080 vs 0.017; paired abs excess ≈0.063, CI excludes 0). Not causal; coverage ≈28% under proposal match tolerances.

## Forbidden

- Reopen V architecture or q model zoo
- Retune B40 thresholds on TEST
- Promote RR sensitivity slices to primary without stating post-hoc
- Reuse old_V quiet ratios for RR3
