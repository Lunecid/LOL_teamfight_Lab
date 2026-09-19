# Experiment inventory — locked cohort (2026-09-19)

**Purpose:** gather every run and artifact under the paper cohort contract **before** locking Intro RQs.  
**Authority:** [PAPER_COHORT_CONTRACT_20260919.md](PAPER_COHORT_CONTRACT_20260919.md).  
**Epistemic (all sealed prediction claims):** exploratory follow-up after prior TEST/external exposure — not confirmatory.

**Weighting fix (same day, post-review):** B40 / \(H\) / stratum ΔBrier now use **cell-internal match-equal** weights for both point estimates and bootstrap; bootstrap `estimate` = observed Δ (not mean of replicates). Recompute cites: `outputs/svi_primary_table_20260919/`, `svi_state_dependent_20260919/`, `svi_lift_localization_20260919/`. Treat pre-fix REPORT numbers as superseded.

**Lineage note (same day, V redesign):** All `svi_*_20260919` prediction tables below are under the **previous** frozen \(\widehat{V}\). New match-WP work follows [V_REDESIGN_CONTRACT_20260919.md](V_REDESIGN_CONTRACT_20260919.md); do not mix lineages in one claim.

---

## Cohort contract (reminder)

| Layer | Definition | Use |
|---|---|---|
| **Study corpus** | 210k KR matches, patches **15.14 + 15.15 + 15.16** | Fit / select / measurement |
| **Primary prediction sample** | **15.16 teamfight T**, \(n=32{,}981\) / \(24{,}020\) matches | Every main-table / ΔBrier claim |
| **Measurement (pooled T)** | 15.14–15.16 T, \(n=113{,}901\) / \(83{,}108\) matches | Concordance, quiet, patch strata |
| **Transfer** | 2026 API `16.x` (KR/NA1) | Score-only; never selection |
| **Primary contrast** | \(q\) − **PT** (match-weighted Brier); \(b(p)\) secondary | Negative ⇒ \(q\) better |
| **Primary \(q\)** | LightGBM (`lgbm_winner`) | Frozen from Q_SELECT on 210k |

Scripts live in repo `scripts/rr20260919_*.py`; large tensors under data-root `문서/LOL_Teamfight/outputs/`.

---

## Catalog (chronological-ish)

| # | Output dir | Script / role | Sample | What it answers |
|---|---|---|---|---|
| 0a | `reviewer_response_phase1_20260919` | shortcut audit | 15.16 T | Does \(q\) beat \(p_{\mathrm{pre}}\) / \(b(p)\) / PT? B40/B45 cells |
| 0b | `reviewer_response_phase2_20260919` | material + quiet smoke | 15.16 T (+ quiet smoke) | Kill/obj/alive vs SVI; quiet \(\|\Delta V\|\) |
| 0c | `reviewer_response_all_patches_20260919` | patch strata | 15.14/15/16 T | Same checks by patch (15.14 TRAIN in-sample LGBM — not headline) |
| 1 | `svi_validation_suite_20260919` | disagreement + quiet type-A | patch strata | Kill×SVI disagree taxonomy; proximity quiet |
| 2 | `svi_cohort_aligned_20260919` | aligned measurement | pooled 210k T | Concordance + quiet A/B on **one** population |
| 3 | `svi_reselection_20260919` | model slate freeze | Q_SELECT → 15.16 T | Classical + Track-A; **winner = LightGBM** |
| 4 | `svi_material_quiet_20260919` | material retrain + quiet | 15.16 T / pooled quiet | Kill/obj models vs SVI; quiet ratios |
| 5 | `svi_transfer_2026_20260919` | transfer score-only | EXT 16.x T (+N) | Frozen \(q\) vs PT on 2026 |
| 6 | `svi_lean_tabm_20260919` | TabM reselection | Q_SELECT → 15.16 T | TabM ≈ LGBM; does not replace winner |
| 7 | `svi_overnight_20260919` | pipeline stages | pooled + 15.16 + EXT | Concordance, quiet matched, cross-target, transfer, Tier B |
| 8 | `svi_primary_table_20260919` | **headline prediction table** | identical 15.16 T | All models same rows; bootstrap \(q-\mathrm{PT}\) |
| 9 | `svi_lift_localization_20260919` | strata of LGBM−PT | 15.16 T + EXT cells | \(p_{\mathrm{pre}}\) / time / B40×time |
| 10 | `svi_state_dependent_20260919` | \(H\), flex PT, ablation, ΔV | 15.16 T | State contrast + B40 info sets + continuous ΔV |
| 11 | `svi_v_time_strata_20260919` | \(\widehat{V}\to W\) by time band | MAIN_TEST queries + 15.16 T anchors | Frame-aligned WP quality (C03); pre/post h90 |
| 12 | `svi_horizon_sensitivity_20260919` | h60/h90/h120 flip | 15.16 T intersection | C04 label sensitivity; primary stays h90 |
| 13 | `svi_time_banded_models_20260919` | **q + V̂ by `s_ms` bands** | 15.16 T + EXT | Primary model report cut on recorded clock (**old V**) |
| 14 | `v_redesign_20260919` | V redesign wave-1 fit | 210k bucket queries | Shared LGBM vs per-band vs legacy; TEST band ledger |

Each dir typically has `REPORT.md` + `results.json` (overnight also has `stage_*/results.json`).

---

## A. Primary prediction (15.16 T, identical rows)

**Authority:** `outputs/svi_primary_table_20260919/`  
**Cross-check:** phase1; reselection; lean TabM; overnight Tier B.

### A1. Overall model table (match-weighted Brier)

| Model | Brier | AUC | ΔBrier vs PT | Role |
|---|---:|---:|---:|---|
| \(p_{\mathrm{pre}}\) raw | 0.27055 | 0.6642 | +0.04103 | diagnostic |
| \(b(p)\) ≈ old logistic | 0.23030 | 0.6642 | +0.00079 | prior-only baseline |
| **PT** | **0.22952** | 0.6648 | 0 | **primary comparator** |
| logistic | 0.22840 | 0.6694 | −0.00111 | \(q\) |
| **LightGBM** | **0.22839** | 0.6691 | **−0.00113** | **\(q\) winner** |
| MLP | 0.22846 | 0.6694 | −0.00106 | \(q\) |
| residual MLP | 0.22900 | 0.6668 | −0.00052 | \(q\) |
| TabM | 0.22846 | 0.6693 | −0.00106 | \(q\) (lean reselection) |
| TabNet (overnight) | 0.22844 | 0.6689 | −0.00108 | Tier B ≈ LGBM |
| FT-Transformer | 0.25013 | 0.5006 | +0.02061 | collapsed |

### A2. Bootstrap (match-clustered, models fixed)

| Contrast | ΔBrier | 95% CI | Notes |
|---|---:|---|---|
| LGBM − PT | −0.00113 | [−0.00156, −0.00069] | \(P(\Delta<0)=1\) |
| LGBM − \(b(p)\) | −0.00192 | [−0.00245, −0.00141] | |
| TabM − PT | −0.00105 | [−0.00163, −0.00047] | does not beat LGBM |
| B40 LGBM − PT | −0.00041 | [−0.00146, +0.00066] | **CI includes 0** |

### A3. Reselection / deep slate

| Artifact | Finding |
|---|---|
| `svi_reselection_20260919` | Q_SELECT freezes LightGBM; sealed LGBM Brier 0.22839 |
| `svi_lean_tabm_20260919` | Chosen `tabm_K32_W128_D0.1__raw`; sealed TabM−LGBM ≈ +0.00007 |
| overnight Tier B | TabNet ≈ LGBM; FT collapsed; SAINT deferred |

### A4. Phase-1 shortcut (same 15.16 T; earlier report)

Same sealed numbers: LGBM−PT ΔBrier −0.0011 [−0.0016, −0.0007]; LGBM−\(b(p)\) −0.0019; also ΔAUC vs PT ≈ +0.0043.

---

## B. Strata / state dependence (15.16 T)

**Authority:** `outputs/svi_lift_localization_20260919/`, `outputs/svi_state_dependent_20260919/`.

### B1. \(p_{\mathrm{pre}}\) bands (LGBM − PT ΔBrier)

| Band | n | ΔBrier | 95% CI | \(P(\Delta<0)\) |
|---|---:|---:|---|---:|
| all | 32981 | −0.00113 | [−0.00156, −0.00069] | 1.000 |
| skew low | 13742 | −0.00097 | [−0.00162, −0.00034] | 0.998 |
| **B40** | 4949 | **−0.00044** | **[−0.00146, +0.00066]** | 0.782 |
| B45 | 2459 | −0.00084 | [−0.00220, +0.00050] | 0.880 |
| skew high | 14290 | −0.00181 | [−0.00254, −0.00114] | 1.000 |
| B20 | 8799 | −0.00123 | [−0.00206, −0.00032] | 0.999 |
| B60 | 5163 | −0.00073 | [−0.00165, +0.00022] | 0.930 |
| B80 | 9127 | −0.00243 | [−0.00339, −0.00157] | 1.000 |

### B2. Time bands

| Band | n | ΔBrier | 95% CI |
|---|---:|---:|---|
| \(t<10\) | 2265 | +0.00021 | [−0.00158, +0.00217] |
| 10–20 | 11858 | −0.00105 | [−0.00171, −0.00043] |
| 20–30 | 15446 | −0.00117 | [−0.00179, −0.00057] |
| \(t\ge30\) | 3412 | −0.00316 | [−0.00486, −0.00150] |

### B3. B40 × time (all CI include 0 or wide)

| Cell | n | ΔBrier | 95% CI |
|---|---:|---:|---|
| B40 × \(t<10\) | 833 | −0.00022 | [−0.00241, +0.00200] |
| B40 × 10–20 | 1940 | +0.00038 | [−0.00097, +0.00180] |
| B40 × 20–30 | 1709 | −0.00094 | [−0.00270, +0.00076] |
| B40 × \(t\ge30\) | 467 | −0.00325 | [−0.00808, +0.00143] |

### B4. Direct \(H\) contrast (state-dependent)

\(d_i=(Y-q)^2-(Y-\mathrm{PT})^2\), \(H=E[d\mid B40]-E[d\mid B40^{c}]\).

| Quantity | Value |
|---|---|
| \(E[d\mid B40]\) | −0.000435 |
| \(E[d\mid B40^{c}]\) | −0.001305 |
| **\(H\)** | **+0.000870** CI [−0.000422, +0.001954]; \(P(H<0)=0.095\) |
| τ (a priori) | 0.001 |

### B5. Flexible PT (Q_SELECT → freeze → 15.16)

| | all | B40 |
|---|---:|---:|
| Chosen | `flex_df7_ix` (Q_SELECT Brier 0.23039) | |
| \(q\) − sealed PT | −0.00113 | −0.00044 |
| \(q\) − flex PT | −0.00136 | −0.00012 |
| flex Brier | 0.22975 | 0.24998 |
| sealed PT Brier | 0.22952 | 0.25030 |

### B6. B40 information-set ablation (fit full TRAIN T; eval B40 TEST)

| Set | n_feat | Brier | vs sealed PT |
|---|---:|---:|---:|
| sealed PT (ref) | — | 0.25030 | 0 |
| sealed LGBM (ref) | — | 0.24986 | −0.00044 |
| S0 \(p_{\mathrm{pre}}\)+time | 2 | 0.25413 | +0.00383 |
| S1 economic | 149 | 0.25313 | +0.00284 |
| S2 + combat | 203 | 0.25034 | +0.00004 |
| S3 full ridge | 352 | 0.25181 | +0.00151 |

Order-dependent; not unique group contributions. Lean LGBM hyperparameters ≠ sealed winner config.

### B7. Continuous \(\widehat{\Delta V}\) (appendix protocol)

| Sample | n | mean\(\|\Delta V\|\) | MAE zero | MAE const | MAE ridge(\(p,t\)) |
|---|---:|---:|---:|---:|---:|
| all | 32981 | 0.1096 | 0.1096 | 0.1096 | 0.1121 |
| B40 | 4949 | **0.1890** | 0.1890 | 0.1890 | 0.1890 |

---

## C. Validation / measurement (pooled 210k T — not prediction headline)

**Authority:** overnight stages + `svi_cohort_aligned_20260919` + validation suite + phase2/all_patches.

### C1. Material concordance (pooled T)

| Axis | Decided n | Tie share | Agree with SVI | Disagree |
|---|---:|---:|---:|---:|
| Kill | 99091 | 0.130 | **0.937** | 0.063 |
| Objective | 75389 | 0.338 | 0.845 | 0.155 |
| Alive | 50064 | 0.560 | 0.819 | 0.181 |

Kill–SVI disagree \(n=6395\): among them obj⇄SVI ≈ 0.36; obj-tie ≈ 0.41.

Patch strata (15.14/15/16) show nearly identical agree rates (all_patches / aligned).

### C2. Quiet references

| Design | Key number | Source |
|---|---|---|
| Matched \(p_{\mathrm{pre}}\)×time×duration | mean \(\|\Delta V\|\) ratio **2.67** (median 2.88); pairs=2361; signed fight−quiet +0.0056 | overnight quiet |
| Type-B unmatched smoke | ratio ≈ 2.3–2.5 by patch | all_patches / aligned |
| Type-A proximity | ratio ≈ 1.4–1.6 | validation suite / aligned |

### C3. Cross-target (352 LightGBM; 15.16 T)

| Target | n | Brier | AUC |
|---|---:|---:|---:|
| SVI (native) | 32981 | 0.2289 | 0.667 |
| Kill | 28711 | 0.2025 | 0.752 |
| Obj | 21779 | 0.1556 | 0.855 |

Kill→SVI substitute (best cal=sigmoid): ΔBrier vs native **+0.00043** (worse) → SVI not reducible to kills.

### C4. Material retrain (`svi_material_quiet_20260919`)

Within-label lifts vs \(p_{\mathrm{pre}}\) on 15.16 T: SVI −0.040; kill −0.016; obj −0.003 (different labels — compare within row only).

---

## D. Transfer (2026, freeze-score)

**Authority:** `svi_transfer_2026_20260919` + overnight `stage_transfer`.

### D1. \(q \to \mathrm{SVI}\) (teamfight T)

| Cohort | n | LGBM−PT ΔBrier | Headline? |
|---|---:|---:|---|
| KR 16.13 | 5202 | −0.00155 | yes |
| NA1 16.13 | 5312 | −0.00196 | yes |
| KR 16.15 | 507 | −0.00225 | yes |
| KR 16.14 pilot | 101 | **+0.00288** | no |

Engagement **N** (non-T): LGBM−PT ≈ −0.0047 to −0.0056 (appendix).

### D2. \(\widehat{V} \to W\) (match win)

**Authority (time bands + engagement anchors):** `outputs/svi_v_time_strata_20260919/`  
Contract: [V_DYNAMIC_FRAME_CONTRACT_20260919.md](V_DYNAMIC_FRAME_CONTRACT_20260919.md).

**MAIN_TEST timeline queries** (match-weighted; \(n=1{,}488{,}325\)):

| Band | Brier | AUC |
|---|---:|---:|
| all | 0.1555 | 0.8545 |
| \(t<10\) | 0.2220 | 0.6894 |
| 10–20 | 0.1519 | 0.8617 |
| 20–30 | 0.0904 | 0.9482 |
| \(t\ge30\) | 0.0934 | 0.9448 |

**15.16 T engagement anchors** (\(n=32{,}981\); same \(W\)):

| Point | Brier | AUC |
|---|---:|---:|
| Pre \(p_{\mathrm{pre}}\) | 0.1418 | 0.8792 |
| Post \(p_{\mathrm{post,h90}}\) | 0.1164 | 0.9173 |

Post improves on pre in every time band (see REPORT) — supports reading \(\Delta\widehat{V}\) on fight frames.

**External overall** (same script; overnight cross-check):

| Cohort | AUC | Brier |
|---|---:|---:|
| KR 16.13 | 0.848 | 0.159 |
| NA1 16.13 | 0.840 | 0.164 |
| KR 16.15 | 0.853 | 0.157 |

EUW1: 0 complete pairs — omitted. Lift localization also reports transfer stratum cells (all / B40 / \(t_{20-30}\)).

---

## E. Patch diagnostics (same corpus; careful use)

**Authority:** `reviewer_response_all_patches_20260919`.

| Patch | Role | n | \(p_{\mathrm{pre}}\) AUC | LGBM AUC | Note |
|---|---|---:|---:|---:|---|
| 15.14 | TRAIN | 39605 | 0.653 | 0.711 | **in-sample \(q\)** — not generalization |
| 15.15 | VAL | 41315 | 0.659 | 0.662 | selection/cal strata |
| 15.16 | TEST | 32981 | 0.664 | 0.669 | **headline** |

Material/quiet patterns stable across patches (agree ~0.94 kill; \(\|\Delta V\|\) ratio ~2.2–2.5).

---

## F. Artifact index (paths)

```
outputs/
  reviewer_response_phase1_20260919/{REPORT.md,results.json}
  reviewer_response_phase2_20260919/{REPORT.md,results.json}
  reviewer_response_all_patches_20260919/{REPORT.md,results.json}
  svi_validation_suite_20260919/{REPORT.md,results.json}
  svi_cohort_aligned_20260919/{REPORT.md,results.json}
  svi_reselection_20260919/{REPORT.md,results.json,CITE_SHEET.md,winner_manifest.json}
  svi_material_quiet_20260919/{REPORT.md,results.json}
  svi_transfer_2026_20260919/{REPORT.md,results.json}
  svi_lean_tabm_20260919/{REPORT.md,results.json,winner_manifest.json,eval/,smoke_train_only/}
  svi_overnight_20260919/{REPORT.md,results.json,stage_*/}
  svi_primary_table_20260919/{REPORT.md,results.json}
  svi_lift_localization_20260919/{REPORT.md,results.json}
  svi_state_dependent_20260919/{REPORT.md,results.json}
  svi_v_time_strata_20260919/{REPORT.md,results.json}
  svi_horizon_sensitivity_20260919/{REPORT.md,results.json}
  svi_time_banded_models_20260919/{REPORT.md,results.json}
```

Contracts / redesign (docs, not numbers):  
`PAPER_COHORT_CONTRACT_20260919.md`, `STRATEGIC_VALUE_LABEL_REDESIGN_20260919.md`,  
`SVI_MODEL_RESELECTION_TRANSFER_20260919.md`, `RELATED_FORECAST_VALUE_LIT_20260919.md`,  
`COLLABORATOR_REPLY_EVAL_CONTRACT_20260919.md`, `V_DYNAMIC_FRAME_CONTRACT_20260919.md`,  
`COG_SUCCESSION_LOCK_20260919.md`, `SHARED_EXPERIMENT_MATRIX_20260919.md`.

---

## G. One-page facts (no RQ framing yet)

1. **Small overall lift is real on 15.16 T:** LGBM−PT ≈ −0.00113, CI excludes 0; several tabular models agree; TabNet ≈; FT failed lean budget.  
2. **B40 lift is uncertain:** CI includes 0; \(H\) CI includes 0; ablation does not clearly beat sealed PT on B40.  
3. **Lift is uneven:** late game and high-skew bands carry more point improvement than B40 / early.  
4. **Label is not kill:** kill agree ~0.94 but 6% disagree; kill→SVI substitute worse; quiet \(\|\Delta V\|\) smaller.  
5. **Transfer directionally holds** on headline 16.13/16.15; pilot 16.14 does not.  
6. **B40 has larger mean \(\|\Delta V\|\)** than overall — magnitude and sign predictability diverge.  
7. **\(\widehat{V}\) is usable on fight frames:** post-h90 beats pre on 15.16 T (AUC 0.917 vs 0.879); timeline AUC rises with clock — report by band, not pooled only.  
8. **Horizon is stable:** on intersection, SVI agree h60↔h90 ≈ 0.986, h90↔h120 ≈ 0.993; ΔV Pearson ≥ 0.99 — h90 primary is not fragile to ±30 s.

Next step (when ready): map these facts → Intro RQs (not forced to two axes only).
