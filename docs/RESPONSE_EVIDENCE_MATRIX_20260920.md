# Review-response evidence matrix (RR1–RR6)

**Date:** 2026-09-20  
**Status:** PLAN LOCKED · **RR1/RR2 complete** · RR3–RR6 / RRX next  
**Design:** [REVIEW_RESPONSE_EXPERIMENT_DESIGN_20260920.md](REVIEW_RESPONSE_EXPERIMENT_DESIGN_20260920.md)  
**Base commit:** `b6d0c769` · V / SVI / q **unchanged** (no redesign)

Purpose chain stays: **predict post-fight WP direction from pre-info** → improve measurement via \(V\) → predict with \(q\) → verify (A) beyond initial edge, (B) what ΔV measures.

| ID | Review ask | Reuse | New work | Status |
|---|---|---|---|---|
| RR0 | Version / keys / weights | Labels, V sha, q weights | `response_manifest` | pending |
| **RR1** | Flex \(p,t\) baselines + optional \(g_q\) | OOF labels, frozen `logit_state`, `PT_linear` | `b_spline`, `PT_flex`, calibrators | **done** — see RR12 results |
| **RR2** | All-T ∥ B40 ∥ narrow bins; \(H\) | RR1 frozen preds | Bin tables + paired CI | **done** — see RR12 results |
| RR3 | Quiet vs fight ΔV (new V) | — | Matched quiet under fit85 | Deferred → scheduled |
| RR4 | Small-ΔV sensitivity | RR1 preds; needs RR3 \(s_Q\) if state-vol | λ grid post-hoc | after RR3 |
| RR5a | Material correspondence | NEWV verify table | Denominators / crosstabs | reuse+polish |
| RR5b | Next objective after endpoint | — | 180s window events | scheduled |
| RR6a | \(V\to W\) time-band calibration | Timeband AUCs | Reliability / slope | scheduled |
| RR6b | Horizon / V-spec stability | h60/90/120, peer flips | Common-support flips | reuse+polish |
| RRX | External dual stage | Transfer 16.x tables | \(V\to W\) + \(q\to\mathrm{SVI}\) split | scheduled |

## Allowed headline (unchanged)

15.16: \(q\) beats **`PT_linear`**. Review response asks whether that survives **`PT_flex`** and balanced cells — answer may be yes or no; either is a valid close.

## Forbidden

- Reopen V architecture or q model zoo
- Retune B40 thresholds on TEST
- Promote RR sensitivity slices to primary without stating post-hoc
- Reuse old_V quiet ratios for RR3
