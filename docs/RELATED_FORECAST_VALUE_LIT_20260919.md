# Related literature — incremental / state-dependent forecast value (use bounds)

**Status:** 2026-09-19 — companion to [STRATEGIC_VALUE_LABEL_REDESIGN_20260919.md](STRATEGIC_VALUE_LABEL_REDESIGN_20260919.md)  
**Role:** measurement and interpretation **frame**, not justification that B40 “must be hard.”

---

## Cite for framing (OK)

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

## Mapping to our RQs (two axes)

- **Axis A (predictability)** ↔ incremental accuracy (DM-style loss differentials; Brier as proper score).  
- **Axis B (contested difficulty)** ↔ state-dependent value (Giacomini–White **motivation**; estimate \(H\) with our match bootstrap).  
- Combination \(p_\lambda\) ↔ optional Clements–Harvey-style probability combination (auxiliary).  
- WPA ↔ continuous \(\widehat{\Delta V}\) genealogy; SVI = direction only; not causal fight WPA.

---

## One-sentence paper stance

Literature tells us **how to ask** whether features add information beyond a baseline and whether that increment varies by state; it does **not** prove that balanced prior-win summaries are intrinsically unpredictable or that large \(|\widehat{\Delta V}|\) implies a clean label.
