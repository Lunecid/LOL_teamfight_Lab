# Outline and claim–evidence map

Working title (provisional):  
**Predicting post-engagement win-probability direction from public pre-state in League of Legends**

Thesis (locked):  
We define fight-interval win-probability **direction** from a frozen match-outcome evaluator \(\widehat V\), and ask how much of that direction is predictable from pre-fight public state beyond initial win probability and time. On held-out 15.16, a frozen \(q\) shows limited proper-score lift over flexed \(p,t\) baselines; measurement checks and external transfer bound how far that lift travels.

---

## Research questions (editorial proposal — pending author confirmation)

| ID | Question (locked wording) | Answer location |
|---|---|---|
| **J-RQ1** | What post-engagement state change does match-linked SVI reflect, and how stable is it under reasonable changes of interval / evaluator peers? | Methods §measurement + Results §ΔV/SVI |
| **J-RQ2** | How much of SVI direction is predictable from pre-engagement public state beyond initial WP and time? | Results §\(q\) vs PT |
| **Transfer / scope** | Does that lift hold on later external cohorts under score-only freeze? | Results §EXT |

*Status note.* The locked journal plan (`docs/JOURNAL_RESEARCH_PLAN_20260919.md` §6) lists three questions, J-RQ1 / J-RQ2 / J-RQ3. The table above merges the first clause of J-RQ3 (dependence on initial edge) into J-RQ2 and leaves transfer unnumbered. The final structure awaits the author's decision; no wording in the Question column has been changed here.

Purpose continuity with CoG: recover how much **pre-information** is available before a fight — here the object is **direction of estimated WP change**, not gold engagement winner.

---

## Section outline (reader order)

1. **Introduction** *(later)* — purpose → measurement upgrade → predictability beyond \(p,t\) → transfer limits.  
2. **Related Work** *(later)* — Maymin WP-change; encounter detection; proper scores / direction vs mean as **object separation**.  
3. **Methods** — engagement definition; frozen \(\widehat V\); ΔV/SVI; \(q\) and PT baselines; splits; scores; CORP as diagnostic decomposition; epistemic status.  
4. **Results**  
   4.1 \(V\to W\) quality (timeline + engagement pre/post)  
   4.2 ΔV / SVI characteristics (triad; quiet contrast; material / next-objective; horizons)  
   4.3 Direction prediction: \(q\) vs PT_flex (all-T + B40)  
   4.4 Sensitivity (λ·\(s_Q\); post-hoc)  
   4.5 External dual-stage (KR/NA1 16.13)  
5. **Discussion** — what the lift means; dependence on \(\widehat V\); where lift fails; Forbidden denials; deferred work.  
6. **Limitations / Availability / Conclusion** *(later)*

**Do not order Results by RR number.** RR IDs are for evidence tracing only.

---

## Claim–evidence map

Status codes: **supported** | **exploratory** | **forbidden if overstated** | **deferred**

Uncertainty column = bootstrap interval only (match-cluster percentile, 2 000 draws, seed 7 unless noted). CORP components are score decomposition and belong under Interpretation limit.

| ID | Claim (manuscript-facing) | Cohort / n | Point estimate | Uncertainty | Artifact | Status | Interpretation limit |
|---|---|---|---|---|---|---|---|
| C1 | Frozen fit85 MLP \(\widehat V\) ranks match outcomes on 15.16 timeline | timeline overall n=347234 | Brier 0.1552; AUC 0.8542 | — (no interval) | RR6a | supported | Measurement quality ≠ SVI validity alone; CORP components MCB 0.0002 / DSC 0.0948 are a score decomposition, not uncertainty |
| C2 | Early band weaker than late | t∈[2,10) | Brier 0.2283; AUC 0.6642 | — | RR6a | supported | Time-conditional; not a license to retune \(V\) on TEST |
| C3 | Engagement-time \(V_{\mathrm{pre}}\) / \(V_{\mathrm{post}}\) improve vs early timeline | eng pre/post n=32981 | Brier 0.1442 / 0.1181; AUC 0.8750 / 0.9149 | — | RR6a | supported | Same frozen evaluator |
| C4 | Direction, signed mean, and scale of ΔV are distinct objects | TEST by \(p_{\mathrm{pre}}\) bins | e.g. near 0.5: \(E[\Delta V]\approx0\), large \(E[\lvert\Delta V\rvert]\); high \(p\): \(P(\Delta V>0)\approx0.63\) with negative \(E[\Delta V]\) | — | RR4 triad | supported | Does **not** prove predictability |
| C5 | On matchable quiet subset, fight \(\lvert\Delta V\rvert\) ≫ quiet | matched n=9140; coverage≈28% | E[|ΔV|_f−|ΔV|_q]=0.0633 | CI95 [0.0612, 0.0654] | RR3 | exploratory scope | Not ATT; not all fights |
| C6 | Kill-axis sign agrees with SVI ~0.90 when decided | all_T decided | agree 0.904 | — | RR5a | supported as correspondence | Material nets overlap \(V\) inputs; ≠ \(q\) accuracy |
| C7 | Among decided elite objectives in 180s, SVI+ vs SVI− Blue rates differ | decided | Blue-credited share among decided: 0.604 (6 004/9 934) vs 0.396 (4 072/10 277); unweighted counts | — | RR5b | correspondence | Incomplete windows mostly game_ended_in_window (8112), not censored |
| C8 | Horizon SVI flips low; much agree is shared endpoints | h60/90/120 | flip 0.007–0.019; same-ep share 0.59–0.81 | — | RR6b | supported | Stability ≠ independent horizon rationality |
| C9 | On 15.16 all-T, frozen \(q\) beats tested PT_flex on Brier | n=32981; matches=24020 | ΔBrier −0.00373; q AUC 0.6403; PT_flex AUC 0.6200 | CI95 [−0.00461, −0.00279] | RR12 | **exploratory** (prior TEST exposure) | Beyond *tested* \(p,t\) summaries; the logistic \(q\) uses 351 numeric features + \(p_{\mathrm{pre}}\) while the LGBM candidate adds 10 champion IDs — not a same-input learner comparison |
| C10 | Continuity vs PT_linear | same | ΔBrier −0.00422 | CI95 [−0.00515, −0.00323] | RR12 | exploratory | Historical primary; PT_flex is flexed stress |
| C11 | B40: small additional lift | n=5423 | ΔBrier −0.00214 | CI95 [−0.00422, −0.00003] | RR12 | exploratory | Upper CI near 0; do not call clear ≥0.001 gain |
| C12 | Heterogeneity H=D_B40−D_outside not clearly nonzero | — | H=0.00181 | CI95 [−0.00060, 0.00422] | RR12 | exploratory | Do not claim balanced states significantly harder |
| C13 | On 15.16, q’s Brier gain vs PT_flex co-moves with higher DSC and somewhat higher MCB | all-T | ΔMCB +0.00033; ΔDSC +0.00406 | — (no interval) | RR6a | supported as **decomposition** | Not a causal “mechanism” of the game |
| C14 | Excluding small \|ΔV\| under tested λ·\(s_Q\) rules does not erase all-T lift | λ∈{0,0.25,0.5,1} | ΔBrier stays ≈−0.004 | CIs exclude 0 | RR4 | post-hoc exploratory | \(s_Q\) 64.9% p-only; do not pick λ on TEST |
| C15 | Main KR/NA1 16.13: q Brier **worse** than PT_flex | n=5202 / 5312 | ΔBrier +0.0026 / +0.0040 | — (no interval; point-estimate ordering) | RRX | supported | Lift does not transfer under score-only freeze; ΔMCB > ΔDSC is the decomposition reading |
| C16 | On those cohorts q DSC still > PT but MCB larger | KR/NA1 | q MCB 0.0065/0.0100; DSC 0.0082/0.0064 | — | RRX | diagnostic | Not proof that recalibration will fix transfer |
| C17 | \(V_{\mathrm{pre}}\) Brier ~0.15 on EXT | KR/NA1 | 0.1511 / 0.1514 | — | RRX | supported | Does **not** alone validate EXT ΔV labels |
| F1 | Public telemetry cannot exceed AUC 0.64 | — | — | — | — | **forbidden** | |
| F2 | B40 weakness proves missing combat skill | — | — | — | — | **forbidden** | |
| F3 | Recalibration will fix EXT | — | — | — | — | **forbidden** / deferred adapter | |
| F4 | Best architecture / exhaustive ranking | — | — | — | — | **forbidden** | |
| F5 | Material correspondence = q accuracy or ATT | — | — | — | — | **forbidden** | |

---

## Reporting checklist note (selective)

Venue instructions (IEEE ToG) govern structure and originality. For prediction reporting hygiene only, optionally cross-check: prediction-time feature availability; train/select/test separation; discrimination and calibration; external evaluation; uncertainty. **TRIPOD+AI is a clinical prediction-model reporting guideline** — use relevant items selectively; do not treat this study as a clinical TRIPOD primary application.
