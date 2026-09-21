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
| *(2026-09-21, S cohort added to the main text — SCALE_SPLIT §8 option C; identity calibrator, contract-literal)* | |
| S: pre-state predicts SVI direction? | Yes, limited: 15.16 S q_S AUC ≈0.577; beats PT_flex_S on Brier |
| S: beyond \(p,t\)? | All-S ΔBrier(q_S−PT_flex_S) −0.00335 [−0.00379, −0.00288]; selection-variant sensitivity −0.00346 |
| S: balanced states? | S∩B40 −0.00210 [−0.00266, −0.00149] (within-S only; not placed beside T B40) |
| S: transfer? | KR/NA1 16.13 S rows: ΔBrier −0.0018 / −0.0020 (score-only, no interval); T and S external readings are listed side by side as observed orderings, not contrasted |
| Pooling T∪S? | q_TS costs T +0.00071 [+0.00018, +0.00126] and gives S −0.00024 [−0.00039, −0.00009]; cohort-specific q kept |

Sources: RR12, RR4, RR6a (`948b36a` CORP), RRX (`3654d4e` common-valid rows).

## Manuscript thesis (use this wording)

> We define fight-interval win-probability **direction** from a frozen match-outcome evaluator \(\widehat V\), and ask how much of that direction is predictable from pre-fight public state beyond initial win probability and time. On held-out 15.16, a frozen \(q\) shows limited proper-score lift over flexed \(p,t\) baselines; small-|ΔV| sensitivity, material/next-objective correspondence, and horizon endpoint identity support the measurement story. On main 16.13 external cohorts the Brier lift does not hold: \(q\) still shows higher discrimination than PT_flex, but larger miscalibration dominates. Results are scoped to the frozen evaluator, engagement definition, and evaluated procedures — not a claim that public telemetry cannot exceed these numbers under other designs.

## Forbidden manuscript claims

- “Public info cannot exceed AUC 0.64”
- “B40 weakness proves unobserved combat execution is the cause”
- “External failure is only calibration” / “recalibration will fix transfer” (EXT CORP is diagnostic; no EXT adapter was validated)
- “We found the best architecture” / exhaustive model ranking
- Equating material/next-objective correspondence with \(q\) accuracy or causal ATT
- *(added 2026-09-21)* “Teamfights (or skirmishes) are more predictable” / any scale gradient — the cut-dependent sign reversal (`sec_prediction.tex` §pred-cut) still applies
- *(added 2026-09-21)* Any mechanism story for a T/S difference (claim ledger X-31 stays withdrawn)
- *(added 2026-09-21)* Pooled T∪S rows, pick results, or small external cohorts substituted for a cohort's own contrast

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
| Add the S (skirmish) cohort to the main text *(2026-09-21)* | **Yes, as scope extension** — predeclared contract (SCALE_SPLIT, 2026-09-20), same frozen \(V\), same learner spec, identity calibrator; not a V/q performance search |

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

1. **Draft spine (started):** [journal_manuscript_v1/](journal_manuscript_v1/README.md) — outline, claim map, Methods, Results, Discussion.  
2. Next writing: Introduction, Abstract, Conclusion, Related Work — only after Methods/Results claims stay stable.  
3. Preserve `docs/tog_manuscript/` as CoG-extension lineage; do not complete `sec_learners` pendings for this version.  
4. Review-response narrative keyed to [RESPONSE_EVIDENCE_MATRIX_20260920.md](RESPONSE_EVIDENCE_MATRIX_20260920.md).
