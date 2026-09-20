# Self-review of the journal_manuscript_v1 spine

**Date:** 2026-09-21 · **Base:** commit `42bb2c6` (T012) · **Scope:** read-only review of files 01–04 against the finish lock, the scale-split contract and results, the RR1 execution addendum and `.ai/CONTEXT.md`.
**Method:** paper-self-review checklist (structure, logic, claim audit, figures, writing, verdict); scientific-critical-thinking for evidence grading; statistical-analysis used only to write the inference-methods note. No number below is new; all are quoted from the drafts or cited `docs/` files. The two `\pending{}` markers are known author items, not defects.

---

## 1. Structure review

| Section | Status | Gap |
|---|---|---|
| Outline (01) | present / needs work | RQ table is an editorial proposal awaiting author decision; no row assigns C18–C22 or the transfer/pooling question to an RQ. Claim map C1–C22, F1–F8 complete. |
| Methods §0 | complete | — |
| Methods §1, §1b | complete (one `\pending`) | The T-membership paragraph carries four messages (rule, code path, fold hashing, corpus counts). Split. |
| Methods §2 | complete | Single-seed candidates are acknowledged; add that the frozen V̂ itself is one seed. |
| Methods §3 | complete | — |
| Methods §4 | complete | §4.2 "S calibrator policy" and "Selection and calibration of q" state the S identity-vs-sigmoid rule twice. Merge. |
| Methods §5 | complete | S EXT match counts are "—"; intermediate sample-flow counts unrecorded (stated honestly). |
| Methods §6 | complete / needs one fix | "intervals are reported only for … (RR12, RR4) and … (RR3)" omits the SS intervals of Results §3.1–§3.4. |
| Methods §7, §8 | complete | — |
| Methods §9 | needs work | No trace row for: [20,30) band (0.0891 / 0.9496; ledger `t_20_30`), PT_flex / PT_linear T rows (0.2392 / 0.6713 / 0.6200; 0.2397 / 0.6723 / 0.6164), T B40 cells (0.2474 / 0.5669; 0.2495 / 0.5248), peer agreement 0.915 / 0.914 (RR5_RR6B), EXT KR 16.15 and pilot rows (0.1533 / 0.1560; +0.0006 / −0.0085), pick share 18% (`cohort_manifest.json` `MAIN_TEST.pick` 29 457 of `rows` 163 680, a derived ratio), B40 `bootstrap_fraction_positive` 0.0240. |
| Results §1 | complete | — |
| Results §2.1 | needs work | Triad is prose only ("near zero", "≈0.63"); print the RR4 bin table with n per bin. |
| Results §2.2–§2.4 | complete | §2.4 peer figure "about 0.91" untraced (see §9). |
| Results §3.1 | complete | — |
| Results §3.2 | complete | All-T prints fraction-positive 0 but B40 does not; print 0.0240 or drop the field. |
| Results §3.3, §3.4 | complete | — |
| Results §4 | present / needs work | No table; per-λ estimates and excluded counts (32 981 → 26 297 at λ = 1) are traced but not printed. T only — see §2(d). |
| Results §5 | complete (one `\pending`) | — |
| Results §6 | complete | "do not oversell" is a note to self, not manuscript prose. |
| Discussion §1 | complete | S paragraph omits the S external ordering (pending author). |
| Discussion §2 | present / thin | Contribution list cannot be judged until Related Work exists. |
| Discussion §3, §4 | complete | §4 is T-only; after the `\pending` decision it must either add the S decomposition or say so. |
| Discussion §5 | complete | — |
| Discussion §6 | complete | Add: no multiplicity adjustment; single seed for V̂ and q; one test patch and training region. |

**Not yet written:** Abstract, Introduction, Related Work, Conclusion / Limitations / Availability, LaTeX port (README, lock §"Immediate next work"). Untracked draft and brief files (05–08, reviewer and venue notes) exist in the working tree at review time; they are outside the committed package and were not reviewed.

---

## 2. Logic consistency

Mapping:

| Question | Methods | Results | Discussion |
|---|---|---|---|
| J-RQ1 (what SVI measures; stability) | §2, §3, §7 | §1, §2.1–§2.4 | §3 |
| J-RQ2 (predictable beyond p, t) | §4, §6 | §3.1, §3.3 (T and S blocks) | §1, §4 |
| Transfer / scope (unnumbered) | §5 EXT rows | §3.2 B40, §4 λ, §5 EXT | §1, §4, §6 |
| (no RQ) cross-cohort transfer, pooling | §4.1 | §3.4, §6 | §1, §5 |

Flags:

(a) **RQ count.** The outline proposes two questions plus unnumbered transfer; the three lock documents hold J-RQ1/2/3 (pending, WORKING_NOTES §D). Whichever option is chosen, J-RQ3's scope list must add engagement scale (T, S), and the outline must say which question C18–C22 answer.

(b) **J-RQ2 as locked is partly unanswerable.** Its second clause asks how learner choice and information-set differences on the same information affect quality. Methods §4.1 states that `logit_state` (352 inputs) and `lgbm_state` (362 inputs) "were not compared on identical inputs", and the freeze has no information-set ablation. The author must revise the wording (option C) or declare the clause out of scope.

(c) **Thesis paragraph is T-only.** The lock and outline say "on main 16.13 cohorts the Brier lift does not hold". With S in the main text this is true for T rows and, on point ordering, not for S rows (−0.0018 / −0.0020). Qualify "for the teamfight cohort" or add an S clause once `\pending{ext-TS-sentence}` is decided.

(d) **Scope note overstates.** Outline: S is reported "in a second block of every prediction table". Results §4 (λ·s_Q) is T-only and contract §5 lists no λ analysis for S. Change to "every primary prediction table (§3.1–§3.3, §5)".

(e) **Claim-map status vs Results wording.** C15 and C17 carry "supported" while Results §5 calls the same numbers "an observed ordering of point estimates". Use "supported as observed ordering (no interval)", as C22 does.

(f) **Methods §6 interval inventory** omits SS. Discussion §6 first bullet implies intervals for "CORP/EXT panels"; they have none.

(g) **Derived ratios.** B40 shares 16.4% / 31.3% and the pick share 18% are ratios of recorded counts; mark them derived in §9, as the 83 108 match sum already is.

(h) **Results §6 vs Discussion.** The summary answers "S: does lift transfer externally?" but Discussion §1 and §4 never mention S external rows; consistent only until the pending sentence is resolved.

---

## 3. Claim audit

**C1.** Claim: frozen V̂ ranks match outcomes on the 15.16 timeline (Brier 0.1552, AUC 0.8542). Verdict: keep. Evidence used: TEST_BAND_LEDGER L5; RR6A `overall`. Missing evidence: no interval (n = 347 234, so precision is not the issue). Overclaim risk: low; only if read as validating ΔV. Suggested wording: "Match-weighted over 347 234 queries from 60 190 matches, V̂ attains Brier 0.1552 and AUC 0.8542 (no interval computed)."

**C2.** Claim: early band [2,10) weaker. Verdict: keep. Evidence: RR6A `t_2_10`. Missing: none. Risk: low. Wording: keep; keep the ledger-name note (`t_0_10` = same cell).

**C3.** Claim: V_pre / V_post "improve vs early timeline". Verdict: revise. Evidence: RR6A `eng_pre`, `eng_post`. Missing: the comparison mixes populations with different time composition. Risk: moderate — reads as "the evaluator is better at fights". Wording: "On the 32 981 engagement snapshots V_pre / V_post reach Brier 0.1442 / 0.1181 and AUC 0.8750 / 0.9149; these describe the evaluator on the rows that define ΔV, not a comparison with the timeline table."

**C4.** Claim: direction, signed mean and scale are distinct objects. Verdict: keep. Evidence: RR4 triad. Missing: bin-level n and values. Risk: low while phrased "need not move together". Wording: keep; add the bin table.

**C5.** Claim: fight |ΔV| ≫ quiet on the matchable subset. Verdict: weaken ("≫"). Evidence: RR3 0.0633 [0.0612, 0.0654]. Missing: coverage ≈ 28%; long engagements unmatchable; selection into the matched set. Risk: moderate if generalized to all fights. Wording: "On the matched subset (≈ 28% of TEST fights), fight |ΔV| exceeds quiet |ΔV| by 0.0633 [0.0612, 0.0654]."

**C6.** Claim: kill-axis sign agrees with SVI 0.904 when decided. Verdict: keep as correspondence. Evidence: RR5a. Missing: no interval; kill features are V̂ inputs. Risk: moderate if read as label validation. Wording: "Where the kill differential is decided, its sign matches SVI in 0.904 of match-weighted cases; kills are inputs to V̂, so this is internal consistency, not validation."

**C7.** Claim: Blue-credited elite objective share 0.604 after SVI+ vs 0.396 after SVI−. Verdict: keep as correspondence. Evidence: RR5b counts. Missing: unweighted; conditional on a decided objective within 180 s; 8 112 game-ended cases outside the denominator. Risk: moderate. Wording: add "unweighted counts, conditional on a decided objective".

**C8.** Claim: horizon flips low; agreement largely shared endpoints. Verdict: keep. Evidence: RR6b (0.015 / 0.007 / 0.019; 0.594 / 0.810 / 0.594). Missing: none. Risk: low. Wording: keep.

**C9.** Claim: q beats PT_flex on all-T. Verdict: keep, exploratory. Evidence: −0.00373 [−0.00461, −0.00279]; fraction positive 0. Missing: model-fitting uncertainty; one test patch; prior exposure. Risk: moderate if "beyond p, t" is read as beyond any function of (p, t) — the tested family is `n_knots_p` ∈ {4, 6} × C ∈ {0.01, 0.1, 1}, uniform knots. Wording: "q's Brier is 0.00373 below the tested PT_flex (match-bootstrap 95% [−0.00461, −0.00279]); 'beyond p, t' means beyond this tested spline family."

**C10.** Claim: continuity vs PT_linear. Verdict: keep. Evidence: −0.00422 [−0.00515, −0.00323] (2 000 draws). Missing: none. Risk: low. Wording: keep as continuity, not a second headline.

**C11.** Claim: B40 small additional lift. Verdict: weaken. Evidence: −0.00214 [−0.00422, −0.00003]; fraction positive 0.0240; RR4 re-draw upper bound −0.0001. Missing: the upper bound is within Monte Carlo error of zero across seeds. Risk: high if the word "lift" survives. Wording: "In B40 the point estimate favours q (−0.00214) but the interval reaches −0.00003; the slice is consistent with a gain and with none."

**C12.** Claim: H not clearly nonzero. Verdict: keep. Evidence: 0.00181 [−0.00060, 0.00422]. Missing: none. Risk: low, unless inverted into "no heterogeneity". Wording: "the data neither establish nor exclude a smaller lift in B40".

**C13.** Claim: Brier gain co-moves with higher DSC and somewhat higher MCB. Verdict: keep as decomposition. Evidence: ΔMCB +0.00033, ΔDSC +0.00406. Missing: no interval; isotonic fit in-sample. Risk: moderate if "co-moves" is read causally. Wording: "The 0.0037 gap decomposes into ΔDSC +0.00406 and ΔMCB +0.00033 (point values)."

**C14.** Claim: λ·s_Q exclusions do not erase the lift. Verdict: keep, post-hoc. Evidence: RR4 λ table (≈ −0.004, CIs exclude 0; n 32 981 → 26 297). Missing: per-λ values in the text; 64.9% p-only fallback. Risk: moderate if used as proof of label robustness. Wording: print the per-λ rows; keep "post-hoc".

**C15.** Claim: on KR/NA1 16.13 q's Brier is worse than PT_flex. Verdict: revise status label. Evidence: +0.0026 / +0.0040 (n 5 202 / 5 312). Missing: no interval; these cells have 3 859 / 3 955 matches against 24 020 for the all-T interval. Risk: moderate — "does not transfer" is stated as established. Wording: "On KR and NA1 16.13, q's Brier exceeds PT_flex's by 0.0026 and 0.0040 (point estimates, no interval); the 15.16 advantage is not observed there."

**C16.** Claim: q DSC still > PT but MCB larger on EXT. Verdict: keep as diagnostic. Evidence: q MCB 0.0065 / 0.0100, DSC 0.0082 / 0.0064. Missing: no interval. Risk: moderate (F3). Wording: add "not evidence that recalibration would restore the gap".

**C17.** Claim: V_pre Brier ≈ 0.15 on EXT. Verdict: keep. Evidence: 0.1511 / 0.1514. Missing: none. Risk: low with the existing "does not validate labels" clause. Wording: keep.

**C18.** Claim: q_S beats PT_flex_S on all-S. Verdict: keep, exploratory. Evidence: −0.00335 [−0.00379, −0.00288]; sensitivity −0.00346 [−0.00382, −0.00307]. Missing: same as C9; S was run after T's TEST result was known. Risk: moderate. Wording: "q_S's Brier is 0.00335 below PT_flex_S [−0.00379, −0.00288] under the contract-literal identity calibrator; the selection variant gives −0.00346."

**C19.** Claim: frozen T model transfers part of its signal to S. Verdict: weaken. Evidence: q_T→S − PT_flex_S −0.00141 [−0.00186, −0.00094]; q_S − q_T→S −0.00194 [−0.00236, −0.00153]. Missing: one of five secondary intervals; under the sigmoid variant the second contrast is −0.00138. Risk: moderate — "signal" implies a feature story (F7). Wording: "Applied unchanged to S rows, the T model has lower Brier than PT_flex_S and higher Brier than q_S; both intervals exclude zero."

**C20.** Claim: pooling helps neither cohort. Verdict: weaken. Evidence: −0.00024 [−0.00039, −0.00009]; +0.00071 [+0.00018, +0.00126]. Missing: under the sensitivity variant q_S − q_TS is −0.00012 [−0.00024, 0.00001], covering zero. Risk: moderate. Wording: "Pooling costs T 0.00071 [0.00018, 0.00126]; for S the difference is −0.00024 (identity) or −0.00012 with an interval covering zero (sigmoid variant)."

**C21.** Claim: S∩B40 lift clearly nonzero and smaller than outside. Verdict: keep, drop "clearly". Evidence: −0.00210 [−0.00266, −0.00149]; H_S +0.00178 [+0.00096, +0.00259]. Missing: none within S. Risk: low–moderate (invites a T comparison). Wording: "Within S the B40 interval excludes zero and H_S is positive."

**C22.** Claim: q_S below PT_flex_S on KR/NA1 16.13 S rows. Verdict: keep as observed ordering (author pending). Evidence: −0.0018 / −0.0020 (n 15 641 / 16 100). Missing: no interval; the 285-row pilot reverses (+0.0035). Risk: moderate — beside the T reversal it invites a mechanism (F7). Wording: "On the S rows of KR and NA1 16.13, q_S's Brier is 0.0018 and 0.0020 below PT_flex_S's (point estimates, no interval)."

---

## 4. Evidence grading

GRADE-style, adapted: an exploratory held-out evaluation starts at *moderate*; downgraded for risk of bias (prior TEST exposure), indirectness (single patch, single evaluator), imprecision (no interval, small cells) and multiplicity; upgraded for consistent effects.

| Headline | Grade | Reasons |
|---|---|---|
| T primary (C9) | moderate | Interval excludes zero with margin; direction repeats vs PT_linear and across λ rows. Downgraded: prior TEST exposure; one test patch, one training region; label from one single-seed evaluator (peer disagreement ≈ 9%); fitting uncertainty excluded. |
| S primary (C18) | moderate (lower end) | Larger n, tighter interval, predeclared contract. Downgraded: run after T's TEST result; calibrator choice moves the estimate (−0.00335 vs −0.00346); same exposure and single-evaluator limits. |
| Transfer partial (C19) | low | Secondary; one of five interval contrasts, no adjustment; magnitude calibrator-dependent; interpretable only as a Brier ordering on identical rows. |
| Pooling cost (C20) | low | Small effects; T cost interval excludes zero but the S side covers zero in the sensitivity variant; secondary. |
| EXT loss on T (C15) | low | No interval; cells of 3 859 / 3 955 matches; two concordant cohorts and a reversal larger than the in-patch gain keep it above very low. |
| EXT retention on S (C22) | low | No interval; larger cells than T EXT; the smallest cohort reverses; not yet author-approved and must not be contrasted with the T loss. |

---

## 5. Inference methods note

**What the interval is.** Rows carry weight 1/n_m, so each match contributes one unit; matches are resampled with replacement (2 000 draws, seed 7), per-match weighted squared-error sums for q and baseline are re-aggregated on the same rows, and the 2.5th / 97.5th percentiles of ΔBrier form the interval. This is a paired cluster bootstrap at the right unit: engagements within a match share state.

**What it covers.** Match-sampling variation within the test patch, given fixed evaluator weights, labels, q and baseline weights, and hyperparameters.

**What it does not cover.** (i) Model-fitting uncertainty: q, PT_flex and V̂ are single fits with seed 7. (ii) Label-model uncertainty: Y_SVI is the sign of a model quantity; peer evaluators agree on the sign in 0.915 / 0.914 of rows, and that disagreement lies outside the interval. (iii) Selection on Q_SELECT: knots, C and the calibrator were chosen on 15.15 and are treated as fixed. (iv) Population: one patch and one region for TEST. Methods §6 states (i) and (ii); add (iii) and (iv) in one sentence.

**Percentile vs BCa.** ΔBrier is a weighted mean of per-match differences; with 24 020 (T) or 49 730 (S) clusters its bootstrap distribution is close to symmetric, so percentile, basic and BCa intervals would agree to the printed precision. The exception is the T B40 slice (4 945 matches): the upper bound is −0.00003 under seed 7, −0.0001 under the RR4 seed and −0.00007 in the LIT bridge, so Monte Carlo error at 2 000 draws is already of the order of the distance to zero. No recomputation is needed; the wording must not say "excludes zero" for this slice.

**Multiplicity.** The drafts print fourteen explicit intervals plus the λ rows; one contrast per cohort is primary by contract, the rest secondary and "reported regardless of sign". No adjustment is applied, and none is required for a predeclared primary, but a reviewer will discount the narrowest secondary margins (q_S − q_TS at −0.00009; T B40). State: "No multiplicity adjustment was applied; secondary intervals are descriptive."

**Likely reviewer questions and pre-emptive wording.** Why not Diebold–Mariano / Giacomini–White: keep Methods §6. Is 0.0037 meaningful: relate to UNC ≈ 0.25 and the contract's τ = 0.001 interpretation threshold. Repeated seeds: "single seed; not repeated". Calibration beyond CORP: "no reliability diagram is reported". `bootstrap_fraction_positive = 0`: write "no draw exceeded zero", never "p = 0". AUC intervals: "none computed".

---

## 6. Figure/table check

No table is numbered or captioned; every Results table is introduced by prose only. Weighting (w = 1/n_m) appears in Methods §6 and Results §1 but in no table.

| Table | Caption | n | matches | weights stated |
|---|---|---|---|---|
| Results §3.1 T, S | none | yes | yes | no |
| Results §3.2 T, S | none | yes | yes | no |
| Results §3.3 T, S (CORP) | none | no | no | no |
| Results §3.4 contrasts | none | rows only | no | in lead sentence |
| Results §5 EXT T, S | none | yes | no (T counts 3 859 / 3 955 exist in Table 5b; S "—") | no |
| Results §6 summary | none | n/a | n/a | n/a |

Methods tables (constants, 5a, 5b, evaluator paths, §7, §9) carry labels at most; 5b has "—" cells for unrecorded match counts.

The paper has **no figures**. Two can be drawn from numbers already in the drafts: a forest-style plot of the fourteen ΔBrier intervals grouped by cohort and role, and the RR4 triad by p_pre bin. A reliability diagram would need new computation and should not be promised.

---

## 7. Writing clarity

Rule: one paragraph, one message; no filler connectives; no adjectives the numbers do not carry.

1. Results §3.1 — "Thus, under the frozen evaluator and engagement definition, pre-state q improves proper-score loss relative to the tested flexible functions of (p_pre, t)." → "Under the frozen evaluator and engagement definition, q has lower Brier and log loss than the tested spline baseline in (p_pre, t)."
2. Results §3.1 — "The absolute lift is small relative to UNC≈0.25; we do not present q as a strong operational tip engine." → "The gain of 0.0037 is small against an uncertainty term of about 0.25; q is not proposed as a decision tool."
3. Results §2.1 — "This motivates keeping binary SVI as the primary target without treating large scale as evidence that direction is predictable from pre-state." → "We therefore keep binary SVI as the target; a large |ΔV| does not imply that its sign is predictable."
4. Results §3.2 (T) — "This is a small exploratory improvement: the interval's upper end sits near zero, and we do not claim a clear gain of size ≥0.001." → "The interval reaches −0.00003; the B40 estimate is consistent with a gain and with no gain."
5. Results §3.4 — "The frozen teamfight model applied unchanged to skirmish rows still beats PT_flex_S, so part of its signal transfers; the skirmish-specific model adds a further, interval-excluding improvement." → "Applied unchanged to skirmish rows, the frozen T model has lower Brier than PT_flex_S; q_S is lower still. Both intervals exclude zero."
6. Results §5 — the "Main reading" paragraph carries six messages. Split: ordering and no interval; decomposition and no adapter; companion-baseline sentence to a footnote.
7. Results §6 — "Small exploratory lift; do not oversell." → "Point estimate favours q; interval touches zero."
8. Discussion §1 — "The gain is real under the locked protocol and small in absolute terms." → "The interval excludes zero under the locked protocol; the gain is small."
9. Discussion §1 — "Together, these results answer the research questions as scoped predictability plus transfer limits, not as a claim of a strong, portable fight tip model." → "The results show limited in-patch predictability of direction beyond (p_pre, t) for both cohorts, and its loss on the external teamfight rows."
10. Discussion §3 — "Residual disagreement across peer evaluators (~9% sign disagreement in reused tables) remains a measurement limitation, not a license to keep searching architectures inside this paper's version boundary." → "Peer evaluators agree with fit85 on the SVI sign in 0.915 / 0.914 of rows; the remainder is a limitation of the label, not a reason to search evaluators."

Also replace "interval-excluding" (§3.4, Discussion §1) with "interval excludes zero", and "does not hold" (Results §5) with the point-estimate sentence in C15.

---

## 8. Final checklist and readiness verdict

Checklist:

- [ ] Abstract states problem, method, result, contribution — **not written**
- [ ] Introduction states gap and questions — **not written**; RQ structure undecided
- [x] Methods reproducible from cited artifacts (constants, roles, seeds, bootstrap stated)
- [x] Results numbers resolve to `docs/` artifacts — except the seven untraced items listed in §1 (Methods §9)
- [x] Uncertainty reported where it exists; absence stated where it does not
- [ ] Over-strong claims weakened — C3, C5, C11, C15/C17 label, C19, C20, C21 pending (§3)
- [x] Discussion names limitations and Forbidden denials
- [ ] Tables captioned and numbered — none
- [ ] Figures — none
- [ ] Citations — no reference list exists yet; Related Work not written

**Verdict: not ready for submission** (the manuscript is a spine, not a paper). For the spine itself: *needs moderate revision*.

Blocking (author decisions or missing text):
1. RQ structure (2 + transfer vs 3) and the unanswerable second clause of locked J-RQ2; the S contrasts need an RQ home.
2. The two `\pending{}` items (pick-exclusion reason; EXT T/S side-by-side sentence) and the resulting T-only thesis paragraph in the lock and outline.
3. Abstract, Introduction, Related Work, Conclusion, reference list, and a numbered table set.

Polish (cosmetic, no author decision needed):
1. Methods §9 trace rows for the untraced numbers; Methods §6 add SS to the interval inventory; mark derived ratios (16.4%, 31.3%, 18%).
2. Claim wording per §3 (C11, C19, C20 first) and the ten sentence fixes in §7.
3. Print or drop `bootstrap_fraction_positive` consistently; unify C15/C17 status with Results §5 wording; correct the "every prediction table" scope note.

Missing evidence to be stated as absent (no new computation): intervals for EXT, CORP, AUC and the V→W tables; repeated-seed fits; reliability diagrams. Figure/table risk: with no figures and uncaptioned tables a reviewer cannot locate the primary contrast at a glance; the forest plot in §6 is the cheapest remedy.
