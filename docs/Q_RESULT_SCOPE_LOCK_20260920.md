# Result scope lock — new V measurement + OOF q + 16.x transfer

**Date:** 2026-09-20  
**Status:** INTERPRETATION LOCK (post collaborator review of commit `b6d0c769`)  
**Superseding note (journal finish):** After RR12, the **manuscript-primary** flexed baseline is **`PT_flex`**. PT_linear remains the historical continuity contrast. Full RR stack: [RESPONSE_EVIDENCE_MATRIX_20260920.md](RESPONSE_EVIDENCE_MATRIX_20260920.md). Draft: [journal_manuscript_v1/](journal_manuscript_v1/README.md).  
**Not:** architecture reopen · external refit · “overall good prediction” headline

Depends on: [Q_NEWV_FIT85_PRIMARY_20260920.md](Q_NEWV_FIT85_PRIMARY_20260920.md), [Q_NEWV_FIT85_TRANSFER_16X_20260920.md](Q_NEWV_FIT85_TRANSFER_16X_20260920.md), [NEWV_ENGAGEMENT_VALUE_VERIFY_20260920.md](NEWV_ENGAGEMENT_VALUE_VERIFY_20260920.md), [Q_PREDICTION_DESIGN_CONTRACT_20260920.md](Q_PREDICTION_DESIGN_CONTRACT_20260920.md)

---

## One-paragraph claim (allowed; journal wording)

> A frozen win-probability evaluator defines engagement ΔV direction. On **15.16**, a pre-state Logistic \(q\) beats **PT_flex** (and, for continuity, PT_linear) on Brier for all-T; B40 shows a smaller exploratory lift. That extra lift **does not** hold on the main **16.13 KR/NA1** transfer cohorts under score-only freeze.

Historical one-liner (pre-RR12, PT_linear-only) is retained below for lineage only.

> *(Lineage)* … beats a **linear** \(p_{\mathrm{pre}}+\)time baseline …

---

## Three questions (keep separate)

| Question | Current position |
|---|---|
| What does new ΔV / SVI measure? | Distribution, material concordance, horizon sensitivity + **RR3 matched quiet** under fit85 (fight \|ΔV\| ≫ quiet). |
| Can direction be predicted from pre-info? | Yes vs **`PT_flex`** on 15.16 T (ΔBrier −0.00373, CI excludes 0); B40 −0.00214 exploratory. Continuity vs **`PT_linear`** −0.00422. |
| Does the lift travel? | **No** on main 16.13 KR/NA1 (ΔBrier positive). CORP: DSC still favors \(q\), MCB dominates — diagnostic only. |

---

## Execution reality vs contract wording

| Contract phrase | This shipped run |
|---|---|
| Q_CAL calibration of \(q\) | Q_SELECT chose **identity** for \(q\) / PT_flex (RR12). Do not say “sigmoid calibration completed” for primary \(q\). |
| PT as main baseline | **Manuscript primary = `PT_flex`.** `PT_linear` = continuity. |
| logit vs LGBM | Different direct inputs (352 vs 362). Say “winner under this feature mix” only if comparing those runs. |

---

## Next work — manuscript (experiments CLOSED)

Tracker: [RESPONSE_EVIDENCE_MATRIX_20260920.md](RESPONSE_EVIDENCE_MATRIX_20260920.md) · Draft: [journal_manuscript_v1/](journal_manuscript_v1/README.md)

| Priority | Deliverable |
|---|---|
| 1 | Done: RR packs CLOSED |
| 2 | **In progress:** Methods / Results / Discussion freeze spine |
| 3 | Next: Intro / Abstract / Conclusion / Related Work from actual Results |
| 4 | No architecture reopen |

---

## Forbidden slides from this snapshot

- Treating 210k matches as the prediction \(n\)
- Comparing AUC to old-V \(q\) as “improvement”
- Averaging small 16.15/pilot with 16.13 to claim transfer success
- Saying “Q_CAL calibration completed” for this primary table
- Equating kill concordance with independent fight-winner accuracy
