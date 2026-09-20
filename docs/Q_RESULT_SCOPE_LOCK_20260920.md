# Result scope lock — new V measurement + OOF q + 16.x transfer

**Date:** 2026-09-20  
**Status:** INTERPRETATION LOCK (post collaborator review of commit `b6d0c769`)  
**Not:** architecture reopen · external refit · “overall good prediction” headline

Depends on: [Q_NEWV_FIT85_PRIMARY_20260920.md](Q_NEWV_FIT85_PRIMARY_20260920.md), [Q_NEWV_FIT85_TRANSFER_16X_20260920.md](Q_NEWV_FIT85_TRANSFER_16X_20260920.md), [NEWV_ENGAGEMENT_VALUE_VERIFY_20260920.md](NEWV_ENGAGEMENT_VALUE_VERIFY_20260920.md), [Q_PREDICTION_DESIGN_CONTRACT_20260920.md](Q_PREDICTION_DESIGN_CONTRACT_20260920.md)

---

## One-paragraph claim (allowed)

> A frozen win-probability evaluator defines engagement ΔV direction. A pre-state Logistic \(q\) beats a **linear** \(p_{\mathrm{pre}}+\)time baseline on the **15.16** holdout (overall and B40 point+CI). That extra lift **does not** hold on the main **16.13 KR/NA1** transfer cohorts under score-only freeze.

---

## Three questions (keep separate)

| Question | Current position |
|---|---|
| What does new ΔV / SVI measure? | Distribution, material concordance, horizon sensitivity documented under fit85 MLP. Quiet still **Deferred**. |
| Can direction be predicted from pre-info? | Yes vs **`PT_linear`** on 15.16 T (ΔBrier −0.00422, CI excludes 0) and B40 (ΔBrier −0.00225, CI excludes 0). |
| Does the lift travel? | **No** on main 16.13 KR/NA1 (ΔBrier positive). Cause split open. |

---

## Execution reality vs contract wording

| Contract phrase | This shipped run |
|---|---|
| Q_CAL calibration of \(q\) | **Not done.** Q_CAL = LGBM early-stop only. Report as **pre-calibrator** primary. |
| PT as main baseline | Implemented as **`PT_linear`** (`σ(a+bp+ct)`). Flexible PT = **follow-up**, new OOF/\(p_{\mathrm{pre}}\) only. |
| logit vs LGBM | Different direct inputs (352 vs 362). Say “winner under this feature mix.” |

---

## Next work — review-response pack (design locked)

Full plan: [REVIEW_RESPONSE_EXPERIMENT_DESIGN_20260920.md](REVIEW_RESPONSE_EXPERIMENT_DESIGN_20260920.md)  
Tracker: [RESPONSE_EVIDENCE_MATRIX_20260920.md](RESPONSE_EVIDENCE_MATRIX_20260920.md)

| Priority | Deliverable |
|---|---|
| 1 | Done: **B40 CI** + public JSON digests |
| 2 | **RR1+RR2:** `PT_flex` / `b_spline` + optional \(g_q\); all-T ∥ B40 ∥ narrow bins (do first) |
| 3 | **RR3:** new-V quiet matched intervals (not old_V ratios) |
| 4 | **RR6** calib/stability polish ∥ **RR5** material + next-objective |
| 5 | **RRX:** frozen \(V\to W\) and \(q\to\mathrm{SVI}\) on 16.x |
| 6 | Results section from this scope — no architecture reopen |

---

## Forbidden slides from this snapshot

- Treating 210k matches as the prediction \(n\)
- Comparing AUC to old-V \(q\) as “improvement”
- Averaging small 16.15/pilot with 16.13 to claim transfer success
- Saying “Q_CAL calibration completed” for this primary table
- Equating kill concordance with independent fight-winner accuracy
