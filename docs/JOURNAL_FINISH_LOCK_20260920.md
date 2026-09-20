# Journal freeze — finish verification, not performance racing

**Date:** 2026-09-20  
**Status:** **JOURNAL FREEZE** (commit baseline: `3654d4e` + this lock)  
**Branch:** `feature/fight-boundary-pipeline`

## Decision

**Stop additional V/q performance competition for the current journal version.**  
Finish the paper as a **verification study**: what change is predicted, how much information remains beyond initial edge/time, and where that lift fails to transfer — not as “AUC reached X, stop.”

This is **not** “current models are optimal.” It is “the research question has enough answered evidence to write the manuscript; further improvement is a *new* question.”

## What the current evidence already answers

| Question | Answer (locked artifacts) |
|---|---|
| Can pre-fight public state predict post-fight ΔV **direction**? | Yes, limited: 15.16 q AUC ≈0.640; beats flexed PT |
| Beyond \(p,t\)? | All-T ΔBrier(q−PT_flex) ≈ −0.00373, CI excludes 0 |
| In balanced states? | B40 ≈ −0.00214 (exploratory; upper CI near 0 — do not oversell) |
| Transfer? | KR/NA1 16.13: lift lost; CORP shows q DSC still > PT but ΔMCB dominates |

Sources: RR12, RR4, RR6a (`948b36a` CORP), RRX (`3654d4e` common-valid rows).

## Manuscript thesis (use this wording)

> We define fight-interval win-probability **direction** from a frozen match-outcome evaluator \(\widehat V\), and ask how much of that direction is predictable from pre-fight public state beyond initial win probability and time. On held-out 15.16, a frozen \(q\) shows limited proper-score lift over flexed \(p,t\) baselines; small-|ΔV| sensitivity, material/next-objective correspondence, and horizon endpoint identity support the measurement story. On main 16.13 external cohorts the Brier lift does not hold: \(q\) still shows higher discrimination than PT_flex, but larger miscalibration dominates. Results are scoped to the frozen evaluator, engagement definition, and evaluated procedures — not a claim that public telemetry cannot exceed these numbers under other designs.

## Forbidden manuscript claims

- “Public info cannot exceed AUC 0.64”
- “B40 weakness proves unobserved combat execution is the cause”
- “External failure is only calibration” / “recalibration will fix transfer” (EXT CORP is diagnostic; no EXT adapter was validated)
- “We found the best architecture” / exhaustive model ranking
- Equating material/next-objective correspondence with \(q\) accuracy or causal ATT

## Allowed / required manuscript moves

1. **CoG → this work:** same purpose (recoverable pre-information), upgraded measurement (WP direction) + verification stack.  
2. **Separate stages:** \(V\to W\) measurement quality ≠ \(q\to\mathrm{SVI}\) predictive lift (RRX dual-stage).  
3. **Report limits honestly:** B40 near-zero CI; EXT lift loss; \(s_Q\) fallback coarseness; RR5b incomplete-window = mostly game-end, not missing data.  
4. **CLOSED packs mean:** each review concern has an analysis + residual limitation — not “all concerns vanished.”

## What not to reopen for this journal

| Temptation | Ruling |
|---|---|
| Retune \(V\) for higher match AUC | **No** — changes \(p_{\mathrm{pre}}\), ΔV, SVI, B40; not same prediction problem |
| Expand \(q\) zoo (Transformer/GNN/TabM…) for max AUC | **No** — selection overfitting risk; not failure-driven |
| Re-run same Q_CAL sigmoid story | **No** — already identity-selected on Q_SELECT |
| Block manuscript on new performance | **No** |

Quality fixes already done (do not re-open as “search”): CORP score-gap MCB; RRX common-valid rows + feature-order; RR5b game-end vs censored split (`3654d4e`).

## Deferred — thesis / v2 branch only

**Targeted** follow-up (not journal completion criterion):

> Can a **low-dimensional probability adapter** on small new-environment data reduce Brier while preserving \(q\)’s discrimination — vs the same adapter on PT — without refitting \(V\) or \(q\) weights?

Principles if run later: freeze V/SVI/q weights; separate adapt vs eval matches (time-order if possible); evaluate Brier/log loss/calibration (not AUC-only); keep score-only EXT results as the unrepaired baseline lineage.

Also deferred: Foster–Stine path filter, multi-horizon rationality, Brill–Yurko–Wyner ΔV uncertainty, \(q_\lambda\) combinations.

## Journal vs thesis split

| Venue | Role |
|---|---|
| **Journal (now)** | Predictability + verification under frozen measurement |
| **Thesis deepen** | Why EXT lift vanishes; limited adaptation experiments |

Do not structure the project so the journal cannot finish without the thesis deepen.

## Immediate next work (writing, not modeling)

1. Journal Results + Discussion around the thesis paragraph above.  
2. Review-response narrative keyed to [RESPONSE_EVIDENCE_MATRIX_20260920.md](RESPONSE_EVIDENCE_MATRIX_20260920.md).  
3. Related Work: Maymin prior; econometrics as object-separation language ([LIT_RESULT_BRIDGE_RR46_20260920.md](LIT_RESULT_BRIDGE_RR46_20260920.md)) — not “finance applied to LoL.”
