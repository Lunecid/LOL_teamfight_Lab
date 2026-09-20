# Results (draft)

*Reader order: evaluator quality → change objects → direction predictability → sensitivity → external transfer. RR IDs in parentheses are evidence pointers only. All primary prediction contrasts are **exploratory** (prior TEST exposure; Methods §0).*

---

## 1. Match win-probability quality of the frozen evaluator (\(V\to W\))

On the 15.16 timeline evaluation (n=347 234 bucket-spaced time-state queries from 60 190 matches; match-weighted scores; no bootstrap interval for this table), frozen fit85 \(\widehat V\) attains Brier **0.1552** and AUC **0.8542**. CORP decomposition attributes most of the gap below UNC≈0.25 to discrimination (DSC **0.0948**) with a small miscalibration component (MCB **0.0002**) [RR6a].

Performance is **time-conditional**: the early band \(t\in[2,10)\) is weaker (Brier **0.2283**, AUC **0.6642**) than later bands (e.g. \(t\in[20,30)\): Brier **0.0891**, AUC **0.9496**). We report this as measurement context, not as a reason to retune \(V\) on the test patch.

Restricted to engagement pre- and post-snapshots on the prediction sample (n=32 981), \(V_{\mathrm{pre}}\) and \(V_{\mathrm{post}}\) reach Brier **0.1442** / **0.1181** and AUC **0.8750** / **0.9149** [RR6a]. These figures characterize the evaluator used to **define** ΔV; they are not themselves the direction-prediction result.

---

## 2. What ΔV and SVI look like (measurement story)

### 2.1 Direction, mean, and scale are different objects

Binning engagements by \(p_{\mathrm{pre}}\) shows that signed mean change, absolute scale, and positive-direction rate need not move together [RR4 triad]. Near balanced \(p_{\mathrm{pre}}\), \(E[\Delta V]\) is near zero while \(E[\lvert\Delta V\rvert]\) is largest; at high \(p_{\mathrm{pre}}\) (e.g. [0.9,1.0]), \(P(\Delta V>0)\approx0.63\) while \(E[\Delta V]\) remains slightly negative. This motivates keeping **binary SVI** as the primary target without treating large scale as evidence that direction is predictable from pre-state.

### 2.2 Fight versus quiet magnitude (matched subset)

On the matchable quiet subset (same match, close \(p\) and \(t\), equal length; coverage ≈**28%** of TEST fights), fight absolute change exceeds quiet:  
\(E[\lvert\Delta V\rvert_{\mathrm{fight}}-\lvert\Delta V\rvert_{\mathrm{quiet}}]=\mathbf{0.0633}\), 95% CI **[0.0612, 0.0654]** [RR3]. Long engagements (\(L>120\)s, 9.2% of TEST) are structurally unmatchable under the equal-length quiet rule. We treat this as a **scale contrast**, not an identified causal fight effect.

### 2.3 Correspondence with material nets and next objectives

When kill differentials are decided, match-weighted sign agreement with SVI is **0.904** (all-T) and **0.928** (B40) [RR5a]. Other axes (epic / structure / objective / alive) show lower agreement and higher tie shares. Material features overlap inputs to \(\widehat V\); these tables are **correspondence**, not independent fight-winner accuracy and not \(q\) accuracy.

In a 180 s window after the engagement endpoint, among cases in which a first elite objective (Baron, dragon, Herald, Horde, Atakhan) is credited to one team, the share credited to Blue is **0.604** after SVI+ (6 004 of 9 934 decided cases) and **0.396** after SVI− (4 072 of 10 277) [RR5b]. These are unweighted case counts; they are neither match win rates nor \(q\) accuracy. Incomplete follow-up is dominated by `game_ended_in_window_no_objective` (8112 of 32 981), not observation censoring (0 in this table).

### 2.4 Horizon endpoints

Comparing SVI at horizons 60 / 90 / 120 s, flip rates are low (0.007–0.019) and sign agreement is high, but **same-endpoint share** is 0.59–0.81 depending on the pair; flips concentrate on differing endpoints [RR6b]. Horizon agreement is therefore not interpreted as fully independent multi-horizon rationality.

Peer evaluators agree with fit85 SVI sign at about **0.91** on reused verify tables [RR6b] — stability across nearby \(V\) specs, not proof of a unique true fight value.

---

## 3. Direction prediction beyond initial edge and time (\(q\to\mathrm{SVI}\))

### 3.1 Primary held-out panel (15.16 TEST)

| Model | n | Matches | Brier ↓ | Log loss ↓ | AUC ↑ |
|---|---:|---:|---:|---:|---:|
| \(q\) (identity) | 32981 | 24020 | 0.2355 | 0.6637 | 0.6403 |
| PT_flex | 32981 | 24020 | 0.2392 | 0.6713 | 0.6200 |
| PT_linear | 32981 | 24020 | 0.2397 | 0.6723 | 0.6164 |

**Primary contrast** ΔBrier(\(q\) − PT_flex), all-T:  
**−0.00373**, match-bootstrap 95% CI **[−0.00461, −0.00279]** (fraction of draws with ΔBrier>0 = 0) [RR12].

Continuity vs PT_linear: ΔBrier **−0.00422**, CI **[−0.00515, −0.00323]**.

Thus, under the frozen evaluator and engagement definition, pre-state \(q\) improves proper-score loss relative to the **tested** flexible functions of \((p_{\mathrm{pre}},t)\). The absolute lift is small relative to UNC≈0.25; we do not present \(q\) as a strong operational tip engine.

### 3.2 Balanced slice (B40)

| Model | n | Matches | Brier | AUC |
|---|---:|---:|---:|---:|
| \(q\) | 5423 | 4945 | 0.2474 | 0.5669 |
| PT_flex | 5423 | 4945 | 0.2495 | 0.5248 |

ΔBrier(\(q\) − PT_flex) = **−0.00214**, CI **[−0.00422, −0.00003]** [RR12, field `bootstrap.B40_q_minus_PT_flex`; the RR4 re-draw of the same contrast with a different bootstrap seed prints the upper bound as −0.0001 with an identical point estimate].  
This is a **small exploratory** improvement: the interval’s upper end sits near zero, and we do **not** claim a clear gain of size ≥0.001. Heterogeneity \(H=D_{\mathrm{B40}}-D_{\mathrm{outside}}\) has CI covering zero (**[−0.00060, 0.00422]**); we do **not** conclude that balanced states are significantly harder than unbalanced ones.

### 3.3 Score decomposition on the same 15.16 rows

On all-T, the CORP decomposition gives the following point values (no bootstrap intervals) [RR6a]:

| Model | Brier | MCB | DSC |
|---|---:|---:|---:|
| \(q\) | 0.2355 | 0.00086 | 0.0154 |
| PT_flex | 0.2392 | 0.00053 | 0.0113 |

So \(\Delta\mathrm{Brier}\approx\Delta\mathrm{MCB}-\Delta\mathrm{DSC}\) with **ΔMCB ≈ +0.00033** and **ΔDSC ≈ +0.00406**: on this sample, \(q\)’s net Brier improvement coincides with **higher discrimination and a somewhat larger miscalibration component** than PT_flex. We report this as a **decomposition of the score gap**, not as a causal mechanism of League of Legends or as evidence that “calibration is fine.”

---

## 4. Sensitivity to small absolute changes (post-hoc)

Using quiet-scale references \(s_Q\) with fallback (64.9% of TEST rows use **p-only** fallback), excluding rows with \(\lvert\Delta V\rvert<\lambda s_Q\) for λ∈{0.25,0.5,1.0} does **not** remove the all-T Brier advantage of \(q\) over PT_flex (point estimates remain about −0.004 with CIs excluding 0) [RR4]. Absolute \(\lvert\Delta V\rvert\) cutoffs yield the same qualitative pattern.

These analyses are **post-hoc** and do not authorize selecting λ on TEST or claiming that tiny-ΔV labels are problem-free.

---

## 5. External score-only evaluation (dual stage)

On common-valid rows with sealed feature order [RRX]:

| Cohort | n | \(V_{\mathrm{pre}}\) Brier | ΔBrier(\(q\)−PT_flex) | ΔMCB | ΔDSC | q MCB | q DSC |
|---|---:|---:|---:|---:|---:|---:|---:|
| KR 16.13 | 5202 | 0.1511 | **+0.0026** | 0.0045 | 0.0019 | 0.0065 | 0.0082 |
| NA1 16.13 | 5312 | 0.1514 | **+0.0040** | 0.0063 | 0.0023 | 0.0100 | 0.0064 |
| KR 16.15 | 507 | 0.1533 | +0.0006 | — | — | — | — |
| KR 16.14 pilot | 101 | 0.1560 | −0.0085 | — | — | — | — |

**Main reading (KR/NA1 16.13):** the 15.16 Brier lift of \(q\) over PT_flex **does not hold**. Discrimination for \(q\) remains higher than for PT_flex on these cohorts, but a **larger miscalibration component** dominates the net Brier gap. \(V_{\mathrm{pre}}\) Brier ≈0.15 does not by itself validate external ΔV labels. CORP here is **diagnostic**; no external adapter was fitted or validated in this version. Smaller cohorts are reported for completeness and are not pooled into a success claim. No bootstrap intervals were computed for the external cohorts, so "does not hold" is an observed ordering of point estimates, not a tested difference. The companion transfer table against PT_linear (+0.00274 / +0.00342 for KR / NA1 16.13) uses a different baseline and is not the quantity shown here.

---

## 6. Summary of empirical answers

| Question | Empirical answer on locked procedures |
|---|---|
| Can pre-state predict SVI direction? | Yes, limited (15.16 AUC 0.6403; beats PT_flex on Brier). |
| Beyond \(p,t\)? | All-T ΔBrier −0.00373 (CI excludes 0). |
| In balanced states? | Small exploratory lift; do not oversell. |
| Does lift transfer? | Not on main KR/NA1 16.13 under score-only freeze (point-estimate ordering; no interval). |

Artifacts: `outputs/review_response_rr12_20260920/`, `..._rr6a_corp_.../`, `..._rrx_external_.../`, and companion RR3–RR5 docs linked from `RESPONSE_EVIDENCE_MATRIX_20260920.md`.
