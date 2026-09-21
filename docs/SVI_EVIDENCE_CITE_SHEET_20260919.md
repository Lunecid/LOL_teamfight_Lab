# SVI evidence cite sheet (2026-09-19)

**Epistemic:** exploratory follow-up after prior TEST/external exposure — not confirmatory.  
**Weighting rule (locked):** within each reported cell, match-equal total weight; bootstrap `estimate` = observed Δ (not mean of replicates).

Use these paths/numbers in Methods/Results; do not mix with pre-recompute REPORT figures.

---

## Evidence IDs → artifacts

| evidence_id | Role (I*) | Artifact | Cite for |
|---|---|---|---|
| E-PRIMARY | I3 | `outputs/svi_primary_table_20260919/` | Main \(q\) table; LGBM−PT |
| E-LIFT | I4 | `outputs/svi_lift_localization_20260919/` | Time / \(p_{\mathrm{pre}}\) / B40×time strata |
| E-STATE | I4 | `outputs/svi_state_dependent_20260919/` | \(H\); flex PT; B40 ablation; ΔV appendix |
| E-V-TIME | I2/C03 | `outputs/svi_v_time_strata_20260919/` | Frame-aligned \(\widehat{V}\to W\) by band; eng pre/post |
| E-HORIZON | I2/C04 | `outputs/svi_horizon_sensitivity_20260919/` | h60/h90/h120 SVI flip + ΔV corr |
| E-TIME-MODELS | I3/I4 | `outputs/svi_time_banded_models_20260919/` | **Primary** q table + V̂ + EXT cut on recorded `s_ms` |
| E-TRANSFER | I4 | `outputs/svi_transfer_2026_20260919/` | Freeze-score EXT \(q\to\mathrm{SVI}\) |
| E-RESELECT | I3 | `outputs/svi_reselection_20260919/` + lean TabM | Q_SELECT winner = LightGBM |
| E-LABEL | I2 | overnight / cohort_aligned / validation_suite | Concordance + quiet |
| E-CONTRACT | — | `docs/PAPER_COHORT_CONTRACT_20260919.md` | Corpus vs 15.16 T sample |
| E-V-FRAME | I2 | `docs/V_DYNAMIC_FRAME_CONTRACT_20260919.md` | What “dynamic \(\widehat{V}\)” means |

Inventory summary: [EXPERIMENT_INVENTORY_COHORT_20260919.md](EXPERIMENT_INVENTORY_COHORT_20260919.md).  
Matrix placement: [SHARED_EXPERIMENT_MATRIX_20260919.md](SHARED_EXPERIMENT_MATRIX_20260919.md).

---

## Locked headline numbers (post weight-fix)

### Prediction — 15.16 T (\(n=32{,}981\))

| Claim | Number | Source |
|---|---|---|
| LGBM − PT ΔBrier | **−0.00113** CI [−0.00156, −0.00069] | E-PRIMARY (pooled summary) |
| **LGBM − PT by `s_ms` band** | \(t{<}10\): +0.00021 CI∋0; 10–20: −0.00105; 20–30: −0.00117; \(t{\ge}30\): **−0.00316** | **E-TIME-MODELS (primary report)** |
| B40 LGBM − PT | **−0.00044** CI [−0.00146, +0.00066] (∋ 0) | E-PRIMARY / E-STATE |
| \(H\) (B40 − outside) | **+0.00087** CI [−0.00026, +0.00205] (∋ 0) | E-STATE |

### \(\widehat{V}\to W\) — frame-aligned (E-V-TIME)

| Claim | Number |
|---|---|
| MAIN_TEST overall AUC / Brier | 0.8545 / 0.1555 |
| MAIN_TEST \(t<10\) → \(t_{20-30}\) AUC | 0.689 → 0.948 |
| 15.16 T pre \(p_{\mathrm{pre}}\) AUC / Brier | 0.8792 / 0.1418 |
| 15.16 T post \(p_{\mathrm{post,h90}}\) AUC / Brier | 0.9173 / 0.1164 |
| EXT KR/NA1 16.13 AUC | 0.848 / 0.840 |

**Safe I2 sentence:** On sealed 15.16 T fight anchors, post-horizon \(\widehat{V}\) improves on pre (AUC 0.917 vs 0.879); timeline skill rises with clock — report by band. This warrants reading \(\Delta\widehat{V}\), not claiming fight-win truth.

**Reporting rule:** time-band tables are primary; pooled AUC is summary only (same for EXT).

### Horizon sensitivity — C04 (E-HORIZON)

| Pair | SVI agree | ΔV Pearson |
|---|---:|---:|
| h60 ↔ h90 | 0.986 | 0.993 |
| h90 ↔ h120 | 0.993 | 0.996 |
| h60 ↔ h120 | 0.982 | 0.990 |

Primary remains **h90**; ±30 s endpoint does not rewrite the label at scale.

### Label / transfer (short)

| Claim | Number | Source |
|---|---|---|
| Kill×SVI agree | ~0.94; substitute kill→SVI worse | E-LABEL |
| EXT KR/NA1 16.13 LGBM−PT | −0.00155 / −0.00196 | E-TRANSFER |

---

## Safe vs unsafe claims (copy into drafts)

| Safe | Unsafe |
|---|---|
| Small overall \(q\) lift over PT on 15.16 T; CI excludes 0 | “Material / decisive lift” without τ discussion |
| B40 incremental gain **not clearly confirmed** (CI∋0; \(H\)∋0) | “No information in B40” or “B40 fails so pipeline fails” |
| Lift uneven (late / high-skew stronger) | Infer heterogeneity from sig-outside vs non-sig-B40 alone |
| \(\widehat{V}\) usable on fight frames; post > pre | Higher \(\widehat{V}\) AUC ⇒ SVI = fight win |
| Transfer directionally holds on headline 16.x | Selection / refit on EXT |

---

## Still open (do not cite as done)

| ID | Status |
|---|---|
| C04 horizon / V-spec sensitivity (flip rates) | **Done** for horizons (`svi_horizon_sensitivity`); alternate V̂ specs still deferred |
| M01 information-set slate (master) | Not run |
| M02 case trace | Not run |
| I1 definition sensitivity full lock | Partly prior docs |
