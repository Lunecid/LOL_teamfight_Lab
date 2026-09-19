# V next-run execution contract — accept collaborator input redesign

**Date:** 2026-09-19  
**Accepts:** [V_MODEL_INPUT_DESIGN_20260919/](V_MODEL_INPUT_DESIGN_20260919/) (commit-reviewed against `154caf97`)  
**Status:** LOCKED for next execution order. Remote was not changed by the collaborator; this contract is our local adoption.

---

## 0. Epistemic narrowing (mandatory)

| Old claim (too strong) | Corrected stance |
|---|---|
| “History does not help” | **Invalid.** H5 vs A0 confounds learner + categorical handling + history definition. Compare **same LightGBM** current vs history after input fix. |
| “RNN / TCN unsuitable” | **Invalid as model verdict.** Right-aligned stacks + `pack_padded_sequence` prefix lengths **dropped real observations** (synthetic check reproduced). Re-fit after left-align / `pack_sequence`. |
| “A0 is the final freeze” | **Too early.** A0 remains **baseline + provisional candidate**; final freeze waits for corrected-input Phase A→B and full V-4. |

Preserve pre-fix wave-2/3 numbers as **`INPUT_IMPL_v0`** artifacts. Do not cite RNN/TCN v0 scores as learner evidence.

---

## 1. Confirmed bugs (local reproduction)

| ID | Finding | Action |
|---|---|---|
| PACK-1 | Right-align + `pack_padded_sequence(lengths)` packs leading pads | Left-align valid prefix; assert packed values == observations |
| TCN-1 | `mask.sum()-1` on right-align does not index last observation correctly for dilated stack | Left-align + gather last valid index; check conv length growth |
| HIST-1 | History = last K **bucket supervised rows**, not guaranteed minute frames | Separate supervised query sample from full as-of observation store |
| ENG-1 | H5 continuity scores last buckets, not `X_pre`/`X_post` as current token | Shared history builder with as-of current token |
| CAT-1 | Champion IDs: LGBM category vs float in RF/MLP/H/S | Typed adapters (one-hot / native cat / embedding) |
| CAL-1 | V_CAL used for early stop **and** calibration | TRAIN-inner stop split; V_CAL = calibration only after refit |

Synthetic record: `docs/V_MODEL_INPUT_DESIGN_20260919/synthetic_checks.json` (also re-run in-repo).

---

## 2. Feature profiles (not “just make 361 bigger”)

| Profile | Definition | Role |
|---|---|---|
| **Expanded361** | 351 numeric + 10 champion categorical | **Primary** comparison |
| **Core267** | drop 94 team×time interactions → 257+10 | Ablation only |

Reference column reconstruction: `STATEV2_REFERENCE_FEATURE_SET.json` — **must match runtime manifest** before fitting. Runtime dump: `outputs/v_redesign_feature_manifest_20260919.json` (generated).

---

## 3. Execution order (do not skip)

| Step | Work | Done when |
|---:|---|---|
| **1** | Runtime feature manifest (names, dtypes, order, source hash) | Matches / diffs vs reference logged | **DONE** — order match True |
| **2** | Fix pack/mask/categorical/current-token builders + synthetic tests | PACK-1 / TCN-1 / CAT-1 unit checks green | **DONE** — wave4 adapters + left-align |
| **3** | Phase A: current-frame Logistic / LGBM / MLP on Expanded361 (+ Core267) | V_SELECT \(L_{\mathrm{time}}\) table under fixed \(T\) | **DONE** — see [BAND_LEDGER_WAVE4_CORRECTED_20260919.md](BAND_LEDGER_WAVE4_CORRECTED_20260919.md) |
| **4** | Phase B: same LGBM ± H3/H5; one fixed GRU + optional Transformer | History value isolated | **DONE** — H5 hurts LGBM; GRU competitive |
| **5** | Optional team-set encoder | Only if A–B leave room | pending |
| **6** | Freeze \(V=g\circ f\circ T\) after DEV continuity (\(D_{\mathrm{switch}}\), quiet) | freeze_manifest | next: continuity on MLP / A0 / GRU |

---

## 4. Split / weight contract (next fits)

| Split | Role |
|---|---|
| TRAIN inner fit | Candidate training |
| TRAIN inner match-level stop | iteration / epoch |
| Full TRAIN | refit at chosen iteration/epoch |
| V_CAL | calibrator \(g\) only |
| V_SELECT | \(L_{\mathrm{time}}\) selection |
| TEST | sealed ledger + continuity diagnostic |

Match-equal weights; **mean-one normalize** and keep that scale across Logistic / LGBM / NN. Record `subsample_freq=1` when using LGBM row bagging.

---

## 5. Adoption progress

1. Collaborator pack imported under `docs/V_MODEL_INPUT_DESIGN_20260919/`.
2. PACK-1 left-align + synthetic check.
3. Runtime 361-feature manifest (order match).
4. Wave-4 corrected Phase A/B re-fit complete — [BAND_LEDGER_WAVE4_CORRECTED_20260919.md](BAND_LEDGER_WAVE4_CORRECTED_20260919.md).

**Remaining before freeze:** continuity on `A_MLP_expanded` (+ A0, GRU); optional full-TRAIN refit at stopped iteration; \(D_{\mathrm{switch}}\) / quiet.
