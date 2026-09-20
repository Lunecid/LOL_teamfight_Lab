# Evaluator bundle reproducibility — A_MLP_expanded

Updated: 2026-09-20T00:32:08+09:00

**Fit scope (LOCKED):** `train_fit85_match_holdout_frac0.15_seed7_NO_full_train_refit`

$$\widehat{V}_{\mathrm{fit85}}=g_{\mathrm{V\_CAL}}\circ f_{\mathrm{MLP,fit85}}\circ T_{\mathrm{fit85}}$$

## Scope of checks

| Check | Meaning | Status |
|---|---|---|
| Same-process reload | Load bundle twice in one process | done earlier (diff=0) |
| **Bundle vs wave-4 live path** | Rebuild TRAIN fit85 `ProfileBundle` + `A_MLP_expanded.joblib` + calib vs bundle | **True** (max abs=0.000e+00) |
| **Fresh OS subprocess** | Worker gets **only** bundle + fixed `X.npy` (no TRAIN, no refit) | **True** (max abs=0.000e+00) |

**PASS all:** **True** (n=4096, atol=1e-06)

## Artifacts

- Bundle: `outputs/v_redesign_wave4_corrected_20260919/evaluators/A_MLP_expanded_evaluator.joblib`
- JSON: `docs/V_EVALUATOR_REPRO_SUBPROCESS_PARITY_20260920.json`
- Worker: `scripts/rr20260919_v_score_bundle_worker.py`

## Reading

- Earlier ‘reload PASS’ was **same-process only**; this update closes **subprocess + live-path parity**.
- Continuity / TEST band tables must score through this bundle (or an identical hash).
