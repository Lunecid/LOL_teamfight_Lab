# V-1 task contract — fields to freeze before fitting new \(\widehat{V}\)

**Status:** **V-1 complete**; wave-1 fit + V-4 done; **V-3 provisional freeze = `shared_lgbm`** → `outputs/v_redesign_20260919/freeze_manifest.json`  
**Parent:** [V_REDESIGN_CONTRACT_20260919.md](V_REDESIGN_CONTRACT_20260919.md)

Closed defaults: pre-2 **excluded** (120s grid); \(\alpha_b=1/4\); feature = expanded StateV2 minus `snapshot_age_s` (361 cols); Choice A freeze; per-band rejected on boundary |Δp| excess.  
Deferred: history K=3/5; TRAIN OOF adapters before q targets; optional shared logistic re-fit (legacy remains baseline compare).

---

## 1. Estimand (LOCK)

| Field | Value |
|---|---|
| Label | \(W\in\{0,1\}\): Blue wins the **match** |
| Predictor | \(\widehat{V}_\theta(X_{\le t})=\widehat{P}(W=1\mid X_{\le t})\) |
| Not the target | CoG exchange sign; SVI; “fight win” |
| Default architecture | **Choice A** — single shared time-conditional \(\theta\) |
| Comparator | Choice B — per-band LightGBM (ablation) |

---

## 2. Population & queries (LOCK)

| Field | Value |
|---|---|
| Study corpus | 210k KR matches, patches **15.14 + 15.15 + 15.16** (same as paper cohort) |
| Training units | \((m,t,X_{m,\le t},W_m)\) on **all eligible timed states**, not engagement-only |
| Base query grid | Match-V5 **minute frames** (`frames[].timestamp` / pipeline `query_ms`) |
| Application diagnostics | Engagement `s` / endpoint \(h\) (primary **h90** later) + off-grid reconstruction |
| Exclude pre-2 min? | **Yes** — corpus V grid starts at **120 s** (same as ToG / `fc20260915`) |
| Unfinished matches | Natural: band \(b\) only includes matches still live at those queries; always report \(n\), \(n_{\mathrm{match}}\) |

---

## 3. Split roles (LOCK — mirror cohort contract)

| Slice | Role for \(\widehat{V}\) |
|---|---|
| 15.14 | Fit candidates (and OOF for that patch’s engagement labels later) |
| 15.15 | Calibrate + **select** (\(\alpha_b\), model family, history arm) — never 15.16 for selection |
| 15.16 | Sealed band ledger + application-time diagnostics only |
| EXT 16.x | Score-only transfer **after** freeze (optional V-3+) |

OOF for TRAIN engagement SVI (later): same fold never sees that match’s \(W\) when scoring pre/end.

---

## 4. Inputs \(X_{\le t}\) (LOCK definition, then freeze for horse-race)

| Block | Include? | Notes |
|---|---|---|
| Clock \(t\) (minutes / spline / soft phase) | **yes** | Required for Choice A |
| Gold / XP / level / CS diffs | **yes** | Public frame state |
| Objectives / structures cumulative | **yes** | |
| Alive / kill tallies | **yes** | |
| Champion / draft IDs | **yes (basic)** | Pre-game known; Hitar-García-style skill/synergy = optional later |
| Soft phase encoding (CoG-style) | optional | Can duplicate raw \(t\) — ablate |
| Last **K∈{0,3,5}** minute frames | V-2 wave 2 | Mask early; do not drop rows silently |
| CoG 30 s fight window as V history | **no** (default) | Near-static; not the history axis |
| \(q\)’s 352 ridge as hard cap | **no** | Redefine for \(W\); freeze one def within a compare |

**Staleness:** between frames, event-updated fields vs last-snapshot resources — document feature policy in the feature manifest.

---

## 5. Weights & selection (LOCK)

| Field | Value |
|---|---|
| Within-band row weight | **Match-equal**: each match total weight 1 inside the evaluation cell |
| Across-band selection | \(L_{\mathrm{time}}=\sum_b \alpha_b \mathrm{Brier}_b\) |
| \(\alpha_b\) (proposal) | **Locked:** equal four bands \(\alpha_b=1/4\) |
| Primary | Time-balanced **Brier** on 15.15 (dev) |
| Tie-break | Log loss (same weights) |
| Diagnostics (not selection alone) | Band AUC, ECE / reliability, minute curves |
| Uncertainty | Match-clustered bootstrap on sealed eval; never treat rows as i.i.d. |

Absolute thresholds (AUC≥…, ECE≤…) — **not set** in V-1.

---

## 6. Calibration policy (LOCK intent)

| Policy | Rule |
|---|---|
| Default | Fit **one** calibration map \(g\) on 15.15 (or shared isotonic/Platt on pooled dev queries) after model fit |
| Per-band \(g_b\) | Allowed only as ablation; must re-check \(\mathrm{sign}(\Delta\widehat{V})\) / magnitudes on boundary pairs |
| MLP | Report **pre/post** calib; temperature scaling first; DU loss optional controlled arm (Kim) |

---

## 7. Continuity suite (V-4 checklist — plan now)

| Check | Pass idea (qualitative) |
|---|---|
| Band boundary, no new events | \(\|\Delta V\|\) distribution vs interior minute steps |
| Frame refresh only | Separate event-driven vs snapshot-age updates |
| Quiet windows | Matched time / \(V_{\mathrm{pre}}\) / duration — reference drift |
| Spec swap | Sign agree rate across top-2 candidates |
| Stochastic forward | If applicable: sign flip rate under inference noise |

---

## 8. Artifact naming (LOCK)

| Item | Path pattern |
|---|---|
| New V training runs | `outputs/v_redesign_20260919/` (or dated successor) |
| Band ledger | `.../BAND_LEDGER.md` + `results.json` |
| Freeze manifest | `.../v_winner_manifest.json` (sha, features, calib, \(\alpha_b\), git) |
| Legacy SVI/\(q\) | Keep `outputs/svi_*_20260919/` — label **old_V** in any cite |

---

## 9. Open TODOs before V-2

1. Confirm include vs exclude minutes \(<2\).  
2. Freeze feature manifest v0 (column list + staleness rules).  
3. Confirm \(\alpha_b=1/4\) vs match-count-weighted alternative (**a priori only**).  
4. Point to concrete extract tensors under data-root for minute-grid \(X\) (reuse `temporal_winprob_v3` / full_corpus paths — document which).  
5. Legacy \(\widehat{V}\) path for same-query baseline compare (current sealed `v_MAIN_*.npz`).

When these five are closed, mark V-1 **complete** and open V-2 scripts.
