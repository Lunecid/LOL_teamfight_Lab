# Econometrics literature → current review-response code map

**Date:** 2026-09-20  
**Status:** APPLICATION LOCK · lit→RR map frozen with CORP score-gap MCB at `948b36a` (does not reopen V/q architecture)  
**Pack:** [literature/LOL_ECONOMETRICS_LITERATURE_20260920/](literature/LOL_ECONOMETRICS_LITERATURE_20260920/)  
**Companion:** [RELATED_FORECAST_VALUE_LIT_20260919.md](RELATED_FORECAST_VALUE_LIT_20260919.md)  
**Result bridge (CLOSED):** [LIT_RESULT_BRIDGE_RR46_20260920.md](LIT_RESULT_BRIDGE_RR46_20260920.md)

Framing kept: this is **direction-of-change forecasting + conditional predictive ability on estimated WP levels**, not “stock prediction applied to LoL.” \(\widehat{V}\) is a learned probability, not a market price.

## Immediate code (this pass)

| Lit anchor | RR task | Code deliverable |
|---|---|---|
| Dimitriadis–Gneiting–Jordan (2021) CORP + Gneiting–Raftery (2007) | **RR6a** | `forecast_diagnostics.py` CORP with **MCB=BS−BS_iso**, **DSC=UNC−BS_iso** (exact identity); V→W and q→SVI separately; diagnostic only |
| Christoffersen–Diebold (2006) + Brown–Warner (1985) spirit | **RR4** (+ RR3 report axes) | Triad \(P(\Delta V>0)\), \(E[\Delta V]\), scale; λ-grid small-\|ΔV\| sensitivity vs Q_CAL \(s_Q\) fallback |
| Giacomini–White (2006) motivation | already RR2 \(H\) | Keep match-bootstrap; do **not** rename as GW CPA test |
| Clements–Harvey (2010) | later optional | \(q_\lambda\) combination = separate version, not rename of RR1 ΔBrier |

## Deferred (v2 / thesis deepen)

| Lit | Use |
|---|---|
| Foster–Stine (2021) | Full-path \(Q_m=\sum(\Delta V)^2\) diagnostics; martingale filter only as `V_v2` candidate |
| Patton–Timmermann (2012) | Multi-horizon rationality on full timelines (not fight-selected only) |
| Brill–Yurko–Wyner (2026) | Match-resampled V uncertainty for ΔV (not copy their ESS numbers) |
| Abadie–Imbens (2008) | Caution when discussing RR3 matching CIs (our estimator ≠ their ATT matching) |

## Forbidden slides from lit

- Equating ΔV with financial returns / market efficiency
- “Martingale ⇒ sign unpredictable” (Christoffersen–Diebold separates mean vs direction)
- “Large \(E[|\Delta V|]\) ⇒ direction is predictable” (scale ≠ conditional sign probability)
- “Small absolute MCB ⇒ calibration is not the problem” when ΔBrier is \(O(10^{-3})\)
- Inferring 16.x external failure modes from 15.16 MAIN CORP alone
- Calling match-bootstrap ΔBrier a Diebold–Mariano or Giacomini–White test
- Applying TEST-fit CORP recalibration as improved frozen \(q\)/\(V\) generalization
- Claiming first LoL WP-change evaluation (Maymin 2021 is direct prior)
- Using \(\langle(p-p^*)^2\rangle\) as CORP MCB (must be BS − BS_iso)