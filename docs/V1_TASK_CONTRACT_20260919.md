# V-1 task contract — fields to freeze before fitting new \(\widehat{V}\)

**Status:** **V-1 complete**; V-2 wave-1 reported; V-3 **provisional** = Choice A `shared_lgbm` (a priori); V-4 **partial**.  
**Parent:** [V_REDESIGN_CONTRACT_20260919.md](V_REDESIGN_CONTRACT_20260919.md) · pack: [WINPROB_V_DESIGN_PACK_20260919.md](WINPROB_V_DESIGN_PACK_20260919.md)

Closed: pre-2 excluded; \(\alpha_b=1/4\); 361-d expanded StateV2 − `snapshot_age_s`; Choice A default.  
**Not claimed:** “V-4 proved model-switch artifact / rejected per-band.” Boundary excess on TEST = diagnostic warning only.

---

## 1. Estimand (LOCK)

| Field | Value |
|---|---|
| Label | \(W\in\{0,1\}\): Blue wins the **match** |
| Final evaluator | \(\widehat{V}_{\mathrm{final}}=g\circ f\circ T\) (same map for select, ledger, ΔV, labels) |
| Default architecture | **Choice A** — shared time-conditional \(\theta\) (**a priori**) |
| Comparator | Choice B — per-band LightGBM (ablation) |

---

## 2. Population & queries (LOCK)

| Field | Value |
|---|---|
| Study corpus | 210k KR, patches **15.14 + 15.15 + 15.16** |
| Training units | timed states \((m,t,X_{\le t},W)\), not engagement-only |
| Design grid | Match-V5 minute frames from **120 s** |
| **Wave-1 fit/eval sample** | **`bucket_only=True`** (`v_is_bucket_sample`) — subset of the minute grid; **not** every frame |
| Application diagnostics | Engagement `s` / endpoint \(h\) (h90) |

---

## 3. Split roles (LOCK)

| Slice | Role |
|---|---|
| 15.14 | Fit (+ OOF for TRAIN engagement labels) |
| 15.15 V_CAL | Fit \(g\) |
| 15.15 V_SELECT | **Only** place that selects among candidates (\(L_{\mathrm{time}}\)) |
| 15.16 TEST | Sealed ledger + exploratory continuity **diagnostics** — **never** the freeze selection rule |

If a future run used TEST continuity to *choose* A vs B, that must be labeled selection-on-TEST.  
Current freeze: **A priori Choice A**; TEST continuity did not select.

---

## 4. Inputs (LOCK)

| Block | Include? |
|---|---|
| Clock \(t\) | yes |
| Gold/XP/level/CS / objectives / alive / kills | yes |
| Champion IDs (basic) | yes |
| History K∈{0,3,5} | wave-2 |
| CoG 30 s fight window as V history | no (default) |

---

## 5. Weights & selection (LOCK)

| Field | Value |
|---|---|
| Within-band | Match-equal |
| Primary | \(L_{\mathrm{time}}=\sum_b\alpha_b\mathrm{Brier}_b\), \(\alpha_b=1/4\) on V_SELECT |
| **Empty / tiny band** | Candidate **ineligible** if any required band has \(n<50\) — do **not** drop that \(\alpha_b\) and renormalize silently |
| Tie-break (wave-1) | **Overall** match-weighted logloss on V_SELECT (same rows as select_overall). *Not* time-balanced logloss — if both needed later, add explicitly |
| Diagnostics | Band AUC; ECE/reliability (**not yet in wave-1 script**); engagement-time \(V\to W\) |
| Uncertainty | Match-clustered bootstrap on sealed eval (**deferred** for V redesign ledger) |

---

## 6. Calibration (LOCK)

| Policy | Rule |
|---|---|
| Default | One \(g\) on V_CAL after \(f\) fit; **all** ΔV / labels / primary continuity use \(g\circ f\) |
| Per-band \(g_b\) | Ablation only; re-check signs/magnitudes |
| Raw \(f\) | Diagnostic companion tables only |

---

## 7. Continuity suite (V-4)

| Check | Wave-1 script |
|---|---|
| Consecutive bucket \|Δp\| interior vs boundary | **yes** (warning; not event-matched) |
| Spec sign agree | **yes** |
| \(D_{\mathrm{switch}}(x)\) same-state band swap | **no** |
| Quiet / frame-refresh | **no** |

---

## 8. Artifact naming (LOCK)

| Item | Path |
|---|---|
| Run dir | `outputs/v_redesign_20260919/` |
| Selection + TEST ledger JSON | `results.json` (+ git extract [BAND_LEDGER_SHARED_LGBM_20260919.md](BAND_LEDGER_SHARED_LGBM_20260919.md)) |
| **Final evaluator freeze** | `freeze_manifest.json` (\(T,f,g\), sha pointers) |
| Short pointer | `winner_manifest.json` (mechanical V_SELECT winner vs provisional Choice A) |
| Legacy SVI/\(q\) | `outputs/svi_*_20260919/` — **old_V** only |

(`v_winner_manifest.json` is **not** used — prefer `freeze_manifest.json`.)

---

## 9. Closed vs open

**Closed:** pre-2; \(\alpha_b\); feature def; Choice A default; bucket_only documented; eval map \(g\circ f\).  
**Open before final freeze:** DEV \(D_{\mathrm{switch}}\); quiet/event strata; ECE; match-bootstrap CIs; optional history K; OOF \(p_{\mathrm{pre}}\) quality vs final \(g\).
