# Related literature — forecast value + win-probability redesign (use bounds)

**Status:** 2026-09-19 — companion to [STRATEGIC_VALUE_LABEL_REDESIGN_20260919.md](STRATEGIC_VALUE_LABEL_REDESIGN_20260919.md) and [V_REDESIGN_CONTRACT_20260919.md](V_REDESIGN_CONTRACT_20260919.md)  
**Index entry:** [WINPROB_V_DESIGN_PACK_20260919.md](WINPROB_V_DESIGN_PACK_20260919.md) §3  
**Role:** measurement / WP-design **frame**, not proof that B40 “must be hard.”

---

## A. Win-probability redesign (read first for \(\widehat{V}\))

| Work | Use as | Do not claim |
|---|---|---|
| **Hodge et al. (2021)** *IEEE ToG* — live esports WP | Time-wise eval; history; **learners = Logistic + Random Forest + LightGBM**; per-minute as Choice B spirit | Must ship one model per minute for \(\Delta V\); ignore LR/RF |
| **Kim, Lee & Chung (2020)** *IEEE CoG* — calibrated LoL WP | Prob. quality + calibration compare (ECE ≠ all scores) | DU loss is mandatory day one |
| **Maymin (2021)** *JQAS* — smart kills / WP change | Match WP → event value via \(\Delta\widehat{V}\); **shared** model over time | Sparse logistic features are our ceiling |
| **Ke et al. (2022)** *IEEE CoG* | Optional: past fights into match WP | Same as “predict next fight winner” |
| **Hitar-García et al. (2023)** *IEEE ToG* | Optional pre-game / synergy features | In-game time-band WP paper |

**Read order for V work:** Hodge → Kim → Maymin.  
Our sealed band curves vs CoG/ToG: [TIMEBAND_COG_TOG_SVI_COMPARE_20260919.md](TIMEBAND_COG_TOG_SVI_COMPARE_20260919.md).

---

## B. Cite for \(q\) / incremental framing (OK)

| Work | Use as | Do not claim |
|---|---|---|
| **Chong & Hendry (1986)**; **Fair & Shiller (1989/90)** | Background: incremental info / forecast comparison | Our ΔBrier alone = encompassing proved |
| **Clements & Harvey (2010)** | Probability-forecast encompassing under quadratic/log scores — closest to Brier/logloss SVI | We ran their exact test unless we did |
| **Giacomini & White (2006)** | Motivation: average vs **conditional / state-dependent** predictive ability | We ran their time-series CPA test on engagements |
| **Diebold & Mariano (1995)** | Loss-differential comparison family | Our match bootstrap = DM statistic |
| **Gneiting & Raftery (2007)** | Proper scoring rules for probabilistic forecasts (prefer over Fair 1996 elections) | — |
| **Clark & McCracken (2001)** | Caution: nested linear 1-step asymptotics | PT vs LightGBM automatically satisfy their nested conditions |
| **Lock & Nettleton (2014)**; **iWinRNFL** | Sports WP: prior + state; complexity ≠ lift | Direct proof of our B40 SVI result |
| **WPA / LI (Mills; Tango; MLB glossary)** | Genealogy for \(\widehat{\Delta V}\) / importance ≠ predictability | B40 := high leverage |
| **Brill–Yurko–Wyner (2024)** | WP estimation uncertainty / dependence | Their WP CI = our ΔBrier CI |
| **Serie A closing-price preprint (2026)** | Optional aside: zero pool weight vs market, other uses of calibrated probs | Core warrant for our contribution |

**Omit as central cite:** Fair (1996) *Econometrics and Presidential Elections* — vote-share model; do not attribute “50% ⇒ closeness is the criterion” as a principle from that paper without a verified quote.

---

## Mapping

- **\(\widehat{V}\) redesign** ↔ Hodge / Kim / Maymin (+ Gneiting–Raftery for selection).  
- **Axis A (predictability of SVI)** ↔ incremental accuracy (after new V freezes).  
- **Axis B (contested difficulty)** ↔ state-dependent value (Giacomini–White **motivation**).  
- WPA ↔ continuous \(\widehat{\Delta V}\) genealogy; SVI = direction only; not causal fight WPA.

---

## One-sentence paper stance

Literature tells us **how to build and score time-aware match WP** and **how to ask** whether features add information beyond a baseline; it does **not** prove that balanced prior-win summaries are intrinsically unpredictable or that large \(|\widehat{\Delta V}|\) implies a clean label.
