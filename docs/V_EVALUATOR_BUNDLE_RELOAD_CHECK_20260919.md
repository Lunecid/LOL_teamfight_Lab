# Evaluator bundle reload check — A_MLP_expanded

Generated: 2026-09-20T00:10:58+09:00

**Fit scope (LOCKED for continuity):** `train_fit85_match_holdout_frac0.15_seed7_NO_full_train_refit`

Collaborator freeze-prep: preproc + model + calibration must travel together. This check loads the joblib in-process twice and compares calibrated probs.

## Bundle path

- `outputs/v_redesign_wave4_corrected_20260919/evaluators/A_MLP_expanded_evaluator.joblib`
- Meta: `outputs/v_redesign_wave4_corrected_20260919/evaluators/A_MLP_expanded_evaluator_meta.json`

## Checks

| Check | Result |
|---|---|
| Reload bit-match (atol=1e-06) | **True** (max abs diff=0.000e+00) |
| Single vs batch (n≤64) | **True** |
| Finite raw/calib | **True** |
| **PASS** | **True** |

## Contract notes

- Continuity / SVI labels for this lineage must use **this** evaluator, not a later full-TRAIN refit, unless a new bundle version is cut.
- Full-TRAIN refit remains optional `corrected_v2` and requires recorded `best_epoch` before re-calib + continuity.
- History density caveat for GRU/H5 unchanged — see wave-4 ledger.
