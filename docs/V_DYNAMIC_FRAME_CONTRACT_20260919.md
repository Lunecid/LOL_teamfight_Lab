# \(\widehat{V}\) dynamic / frame-aligned contract

**Status:** locked 2026-09-19 (frame / band **reporting**); **superseded for development order by** [V_REDESIGN_CONTRACT_20260919.md](V_REDESIGN_CONTRACT_20260919.md)  
**Role:** I2 — how to score and report a frozen \(\widehat{V}\) on the recorded clock. Full redesign (shared vs per-band, selection, ΔV checks) lives in the redesign contract.

Companions: [V_REDESIGN_CONTRACT_20260919.md](V_REDESIGN_CONTRACT_20260919.md), [COG_SUCCESSION_LOCK_20260919.md](COG_SUCCESSION_LOCK_20260919.md), [EXPERIMENT_DESIGN_MAP_20260919.md](EXPERIMENT_DESIGN_MAP_20260919.md), [PAPER_COHORT_CONTRACT_20260919.md](PAPER_COHORT_CONTRACT_20260919.md).

---

## 1. What “dynamic” means here

Not a product live-server requirement. Not continual re-fitting of weights.

\[
\widehat{V}(S_t)=\widehat{P}(W=1\mid S_t)
\]

- **Parameters frozen**; **inputs \(S_t\) change** as the match evolves.  
- When an engagement ends and the public state updates (kills, gold, objectives, …), evaluating \(\widehat{V}\) on the new state yields a new win probability.  
- Engagement advantage is the **frame-aligned difference**

\[
\Delta\widehat{V}=\widehat{V}(S_{\mathrm{end}})-\widehat{V}(S_{\mathrm{pre}}),\qquad
\mathrm{SVI}=1[\Delta\widehat{V}>0].
\]

---

## 2. Frame grid (design to the data)

Public Match-V5 timelines already record time on every usable state:

| Source field | What it is |
|---|---|
| Timeline `frames[].timestamp` (ms) | Raw game clock on each public frame (~`frameInterval`) |
| Pipeline `query_ms` | Same clock when \(\widehat{V}\) is scored on the state grid |
| Engagement `s` / `L` / `endpoint_h*` | Onset / cutoff / label endpoint on that same ms clock |

We do **not** invent a separate “live product” clock. Bands are just bins of these **recorded** times:

`[0,10)`, `[10,20)`, `[20,30)`, `[30,∞)` minutes from `query_ms` or engagement onset `s`.

| Rule | Contract |
|---|---|
| Evaluation times | Only times where a usable snapshot exists under the frozen state pipeline |
| \(S_{\mathrm{pre}}\) | Snapshot at/just before engagement onset (existing `pre_snapshot` / `q_pre` rules) |
| \(S_{\mathrm{end}}\) | Snapshot at label endpoint under horizon \(h\) (primary **h90**) |
| Forbidden | Pretending \(\widehat{V}\) is scored at arbitrary sub-frame times without a snapshot; pooling away the recorded clock |

Performance reporting for \(\widehat{V}\to W\) must use the **same recorded clock bands** as \(q\) lift localization.

**Rule:** band-wise tables (cut on raw ms → minutes) are the **primary** performance report. A single pooled AUC/Brier is allowed only as a one-line summary. Early and late frames are different information regimes — skill **must** be shown to change across bands (or the absence of change must be argued). Same rule for MAIN_TEST and EXT.

---

## 3. Required \(\widehat{V}\) performance tables (C03)

### 3.1 Bucket / timeline queries (state grid)

Source: `full_corpus_training_20260915/eval/predictions/v_MAIN_TEST.npz` (and EXT `v_*.npz`).

| Table | Rows | Metrics |
|---|---|---|
| Overall | all scored queries (match-weighted) | Brier, AUC vs `winner_blue` |
| By time band | same bands as above on `query_ms` | Brier, AUC, \(n\) |
| Optional | `is_bucket_sample` subset | same |

### 3.2 Engagement anchors (SVI path)

On **15.16 T**, \(n=32{,}981\) sealed engagement rows:

| Point | Score | Against |
|---|---|---|
| Pre | `p_pre` = \(\widehat{V}(S_{\mathrm{pre}})\) | match \(W\) |
| Post (h90) | `p_post_h90` = \(\widehat{V}(S_{\mathrm{end}})\) | match \(W\) |
| By pre-fight time band | both | Brier, AUC |

This shows whether WP quality at **fight-relevant frames** (not only uniform buckets) is adequate for reading \(\Delta\widehat{V}\).

### 3.3 What good looks like (interpretation, not a gate to stop the paper)

- Later bands usually easier (more information) — report, do not require monotonic perfection.  
- Pre vs post: post should not be *worse* than pre in a way that makes \(\Delta\widehat{V}\) untrustworthy without explanation.  
- **Do not** treat higher \(\widehat{V}\) AUC as automatic license to claim SVI is fight-win truth.

---

## 4. Sensitivity (C04) — still secondary

Horizon \(h\in\{60,90,120\}\), alternate \(\widehat{V}\) specs: report **label flip rates** and whether **headline \(q-\mathrm{PT}\)** moves — only after 3.1–3.2 are frozen.

**Horizon flip (done):** `outputs/svi_horizon_sensitivity_20260919/` — SVI agree h60↔h90 ≈ 0.986 on 15.16 T intersection; primary stays h90. Alternate \(\widehat{V}\) architecture specs remain deferred.

---

## 5. Relation to \(q\)

| Object | Estimand | Dynamic? |
|---|---|---|
| \(\widehat{V}(S_t)\) | \(P(W=1\mid S_t)\) | Yes — via changing \(S_t\) on the frame grid |
| \(q(X_{\mathrm{pre}})\) | \(P(\mathrm{SVI}=1\mid X_{\mathrm{pre}})\) | Pre-fight forecast of the sign of the WP jump |

Improving \(\widehat{V}\) is an **I2 warrant** task. It does **not** replace I3/I4 \(q\) evaluation, and chasing \(\widehat{V}\) AUC is not the research motive.

---

## 6. Artifact

Script: `scripts/rr20260919_svi_v_time_strata.py`  
Output: `outputs/svi_v_time_strata_20260919/{results.json,REPORT.md}`  
**Status:** C03 tables generated 2026-09-19 (cite sheet: [SVI_EVIDENCE_CITE_SHEET_20260919.md](SVI_EVIDENCE_CITE_SHEET_20260919.md)).
