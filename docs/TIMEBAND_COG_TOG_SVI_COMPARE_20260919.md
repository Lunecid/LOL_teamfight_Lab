# Time-band model performance — CoG / local ToG pipeline / SVI comparison

**Date:** 2026-09-19  
**Question:** Do past lines already show clock-dependent skill, and how do sealed curves line up?

**Naming:** Rows labeled **“ToG (local pipeline)”** are from this repo’s `temporal_winprob_v3` / ToG manuscript work — **not** numerical claims copied from **Hodge et al. (2021) IEEE ToG** (that paper is a *design* precedent only; see [RELATED_FORECAST_VALUE_LIT_20260919.md](RELATED_FORECAST_VALUE_LIT_20260919.md)).

**New redesign V:** Do **not** treat tables below as `shared_lgbm` performance. Use [BAND_LEDGER_SHARED_LGBM_20260919.md](BAND_LEDGER_SHARED_LGBM_20260919.md).

**Epistemic:** figures from locked manuscripts / frozen metrics; SVI strata are exploratory re-scores after TEST exposure.

---

## 1. What each paper measures (do not mix labels)

| Line | Target | Typical clock cut | Primary metric |
|---|---|---|---|
| **CoG** | Pre-engagement → **fight outcome** (Eq.3 / kill-window win) | early / mid / late (code: ≤14 / ≤25 / >25 min) | AUC |
| **ToG** | Timeline \(\widehat{V}\to\) **match win** \(W\) (and separately engagement prediction) | **2–10 / 10–20 / 20–30 / 30+** min | AUC, ECE |
| **SVI (now)** | (i) \(q\to\mathrm{SVI}\); (ii) \(\widehat{V}\to W\) | **[0,10) / [10,20) / [20,30) / [30,∞)** on recorded `s_ms` / `query_ms` | ΔBrier \(q-\mathrm{PT}\); AUC |

Same qualitative claim (“later frames are easier”) can hold while **absolute AUC levels are not comparable** across rows.

---

## 2. Numbers side by side

### 2.1 CoG — fight-outcome LightGBM by phase

Source: [CoG2026_Paper.md](CoG2026_Paper.md) §7.1 (“Conditional predictability”); phase cuts in `analysis/phase_stratified_shap.py` (`≤14` / `≤25` / else).

| Phase | LightGBM AUC (paper) |
|---|---:|
| early | **0.616** |
| mid | **0.712** |
| late | **0.807** |

Cross-check (presence-gate retrain, DEFINITION_EVIDENCE §15): early / mid / late **0.616 / 0.705 / 0.798** — same monotone rise.

Corpus-v3 CoG protocol (DEFINITION_EVIDENCE §19): **0.611 / 0.696 / 0.795**.

**Reading:** CoG already treated minute-wise / phase-wise AUC as a required report (“Minute-wise Performance” in EXPERIMENT.md). Skill rises sharply with phase.

### 2.2 ToG — match-win \(\widehat{V}\) by 10-minute bands

Source: `문서/LOL_Teamfight/outputs/temporal_winprob_v3/metrics/test_metrics.json` → `test_grid_time_bands` (candidate **A**); also manuscript `sec_limitations.tex` (late AUC ≈ 0.844, late ECE worse).

| Band | n rows | AUC | Brier | ECE |
|---|---:|---:|---:|---:|
| 2–10 | 39,770 | **0.674** | 0.226 | 0.029 |
| 10–20 | 47,006 | **0.855** | 0.155 | 0.007 |
| 20–30 | 30,289 | **0.945** | 0.094 | 0.005 |
| 30+ | 6,079 | **0.930** | 0.106 | 0.018 |
| overall (grid) | — | **0.845** | — | — |

**Reading:** ToG already cut \(\widehat{V}\) on the **same 10-minute clock family** we use now. Early band ≈ 0.67; mid/late ≈ 0.85–0.95. Late ECE can degrade even when AUC stays high — calibration ≠ ranking.

### 2.3 SVI — current sealed reports (recorded `s_ms`)

Source: `outputs/svi_time_banded_models_20260919/`, `outputs/svi_v_time_strata_20260919/`.

**A. \(q\to\mathrm{SVI}\) (15.16 T) — LightGBM**

| Band | n | AUC | ΔBrier vs PT | 95% CI |
|---|---:|---:|---:|---|
| [0,10) | 2,265 | 0.536 | +0.00021 | ∋ 0 |
| [10,20) | 11,858 | 0.612 | −0.00105 | excludes 0 |
| [20,30) | 15,446 | 0.697 | −0.00117 | excludes 0 |
| [30,∞) | 3,412 | 0.676 | −0.00316 | excludes 0 |

**B. \(\widehat{V}\to W\) — MAIN_TEST timeline queries**

| Band | AUC | Brier |
|---|---:|---:|
| [0,10) | 0.689 | 0.222 |
| [10,20) | 0.862 | 0.152 |
| [20,30) | 0.948 | 0.090 |
| [30,∞) | 0.945 | 0.093 |
| all | 0.854 | 0.155 |

**C. \(\widehat{V}\to W\) — 15.16 T engagement anchors (pre / post h90)**

| Band | pre AUC | post AUC |
|---|---:|---:|
| [0,10) | 0.720 | 0.764 |
| [10,20) | 0.843 | 0.877 |
| [20,30) | 0.904 | 0.937 |
| [30,∞) | 0.861 | 0.933 |

---

## 3. Comparison (same story, different estimands)

| Claim | CoG | ToG \(\widehat{V}\) | SVI \(\widehat{V}\) | SVI \(q\) |
|---|---|---|---|---|
| Skill **changes by clock** | Yes (phase) | Yes (10-min) | Yes (10-min) | Yes (10-min) |
| Early hardest | Yes (~0.61) | Yes (~0.67) | Yes (~0.69) | Yes (~0.54 AUC; lift CI∋0) |
| Mid/late much easier | Yes → ~0.80 | Yes → ~0.93–0.95 | Yes → ~0.95 | Milder AUC rise; **Brier lift peaks late** |
| Band table = primary? | Planned (“minute-wise”) | In metrics JSON / limitations | **Locked** in V_DYNAMIC + time-banded report | Same |

**Closest like-for-like:** ToG `test_grid_time_bands` vs SVI MAIN_TEST `v_MAIN_TEST` time strata — both \(\widehat{V}\to W\) on public frames. Curves match closely (early ~0.67–0.69 → late ~0.93–0.95). SVI is a re-report on the 210k sealed pipeline, not a new phenomenon.

**Not like-for-like:** CoG early/mid/late fight-outcome AUC vs SVI \(q\to\mathrm{SVI}\) AUC. Different \(Y\), different cuts. Use only for the **qualitative** succession claim: “performance must be shown by game time.”

---

## 4. What this buys the succession paper

1. **Precedent:** CoG and ToG already documented clock-dependent skill; SVI is continuing that reporting duty under a new estimand.  
2. **Design lock:** 10-minute bands + recorded ms are ToG-compatible; early/mid/late remains an optional CoG-style appendix cut.  
3. **Honest gap:** SVI’s **incremental** story (\(q-\mathrm{PT}\)) is *not* “AUC rises like CoG’s late 0.80” — overall lift is small; **late bands carry ΔBrier**, early does not clear 0.  
4. **I2 warrant:** SVI \(\widehat{V}\) band curve ≈ ToG → reading \(\Delta\widehat{V}\) on fight frames is consistent with the prior win-prob model’s known time profile.

---

## 5. Sources (cite paths)

| ID | Path |
|---|---|
| CoG text | `docs/CoG2026_Paper.md` §7.1 |
| CoG phase code | `analysis/phase_stratified_shap.py` |
| CoG phase cross-check | `docs/DEFINITION_EVIDENCE.md` §15, §19 |
| ToG V bands | `…/temporal_winprob_v3/metrics/test_metrics.json` → `test_grid_time_bands` |
| ToG limitations | `docs/tog_manuscript/sec_limitations.tex` (late AUC / ECE) |
| SVI band models | `outputs/svi_time_banded_models_20260919/` |
| SVI V strata | `outputs/svi_v_time_strata_20260919/` |
| Canvas | `canvases/svi-cog-tog-timeband-compare.canvas.tsx` |
