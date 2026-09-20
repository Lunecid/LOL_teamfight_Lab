# V 개발 단계 마감 — `A_MLP_expanded` fit85

**Date:** 2026-09-20  
**Status:** **평가기 계보 PROVISIONAL FREEZE for ΔV/SVI measurement.**  
전체 TRAIN 재적합(`corrected_v2`)은 별도 버전. 모델 탐색 재개 금지.

---

## Frozen object

\[
\widehat{V}_{\mathrm{new}}
=
g_{\mathrm{V\_CAL}}
\circ f_{\mathrm{MLP,fit85}}
\circ T_{\mathrm{fit85}}
\]

| Field | Value |
|---|---|
| Bundle | `outputs/v_redesign_wave4_corrected_20260919/evaluators/A_MLP_expanded_evaluator.joblib` |
| Fit scope | `train_fit85_match_holdout_frac0.15_seed7_NO_full_train_refit` |
| SHA256 | `AC459CC4397630A953D672730CAA687B50FC1EDCD5A029928FAD236FC5B204E3` |
| Selection rule | Predeclared \(L_{\mathrm{time}}\) on V_SELECT (wave-4 corrected) |

---

## Close checklist (no re-run if already PASS)

| Item | Evidence | Status |
|---|---|---|
| Subprocess reload + live-path parity | [V_EVALUATOR_BUNDLE_RELOAD_CHECK_20260919.md](V_EVALUATOR_BUNDLE_RELOAD_CHECK_20260919.md), `docs/V_EVALUATOR_REPRO_SUBPROCESS_PARITY_20260920.json` | **PASS** (max abs=0) |
| Time-band / per-minute WP quality | [TEST_BAND_LEDGER_MLP_FIT85_20260920.md](TEST_BAND_LEDGER_MLP_FIT85_20260920.md), `docs/figures/mlp_fit85_test_per_minute_*.png` | **linked** |
| Continuity diagnostics | [CONTINUITY_LEDGER_MLP_FIT85_20260920.md](CONTINUITY_LEDGER_MLP_FIT85_20260920.md) | **linked** (diagnostic, not selection) |
| RQ placement | [J_RQ1_SCOPE_LOCK_20260920.md](J_RQ1_SCOPE_LOCK_20260920.md) | J-RQ1 part ① closed enough to measure; ② = next |

**Exit conditions NOT required:** beating every peer in every minute; higher AUC; full-TRAIN refit.

---

## What this freeze authorizes

- Build **new-V engagement ΔV/SVI result tables** with fixed T / onset·cutoff / h90 (engagement definition unchanged).
- Recalculate **B40 from new \(p_{\mathrm{pre}}\)** — do not reuse old-label B40.
- Run RQ1 part ② analyses (distribution, material correspondence, stability).

## What this freeze does **not** authorize

- Claiming “MLP is globally best” or “SVI = true fight winner”.
- Jumping to **q** model comparison before the new engagement value table + RQ1 analyses.
- Using fit85 SVI on TRAIN matches as q labels without OOF / leave-match V (deferred to RQ2 prep).
