# \(\widehat{V}\) redesign contract — match-win WP before SVI / \(q\)

**Status:** locked 2026-09-19; **V-3 provisional = Choice A `shared_lgbm` is exploratory only**.  
**V-2 wave-1 is insufficient** — full candidate matrix: [V2_CANDIDATE_MATRIX_20260919.md](V2_CANDIDATE_MATRIX_20260919.md) (WP lit + CoG/ToG learners). Final freeze waits at least on **Tier 0**.  
**Priority:** finish the V horse-race before further \(q\) search. Old `svi_*` = previous \(\widehat{V}\) only.

**Final evaluator:** \(\widehat{V}_{\mathrm{final}}=g\circ f\circ T\).

Companions: [WINPROB_V_DESIGN_PACK_20260919.md](WINPROB_V_DESIGN_PACK_20260919.md), [V2_CANDIDATE_MATRIX_20260919.md](V2_CANDIDATE_MATRIX_20260919.md), [BAND_LEDGER_SHARED_LGBM_20260919.md](BAND_LEDGER_SHARED_LGBM_20260919.md), [MODELS.md](MODELS.md) (CoG zoo), [RELATED_FORECAST_VALUE_LIT_20260919.md](RELATED_FORECAST_VALUE_LIT_20260919.md).

---

## 0. Why this comes first

SVI and \(\Delta\widehat{V}\) treat \(\widehat{V}\)’s output **change** as engagement value. Therefore \(\widehat{V}\to W\) quality is not a side check — it is the **measurement foundation**.

**Order (locked):**

1. Define the win-probability task and evaluation criteria  
2. Develop / select / validate \(\widehat{V}\) **with time-band records**  
3. Freeze \(\widehat{V}\) (params, preprocess, calibration lineage)  
4. Check \(\Delta\widehat{V}\) continuity / stability (boundary, refresh, quiet)  
5. Build SVI from frozen \(\widehat{V}\)  
6. Only then fit / evaluate engagement predictor \(q\)

Pause additional \(q\) architecture search until step 5.

---

## 1. Estimand (this stage)

At time \(t\), with public information available up to \(t\):

\[
\widehat{V}_\theta(X_{\le t})=\widehat{P}(W=1\mid X_{\le t}),
\]

where \(W=1\) iff **Blue wins the match**. Ground truth for this stage is **match outcome**, not CoG’s hand-weighted exchange sign and not SVI.

After freezing \(\theta\):

\[
\Delta\widehat{V}
=\widehat{V}_\theta(X_{\le t_{\mathrm{end}}})
-\widehat{V}_\theta(X_{\le t_{\mathrm{pre}}}).
\]

**Separate:**

| Concept | Example |
|---|---|
| Who is ahead **now** | \(V=0.80\) still favors Blue |
| Who the **interval helped** | \(0.80\to 0.65\) is adverse for Blue |

**Forbidden:** crowning CoG’s best fight-outcome learner as the default \(\widehat{V}\) winner. Different target.

---

## 2. What “time-band performance” means (and does not)

| Means | Does **not** mean |
|---|---|
| Fix **evaluation criteria and selection protocol** that use time bands | Force equal AUC in every band |
| Select model + calibration on **development** data under that protocol, then freeze | Pick a different winner per TEST minute after looking at TEST |
| Keep band-wise scores as a **validated performance ledger** | Claim those numbers stay true on future patches without re-check |

Bands for reporting (recorded Match-V5 ms → minutes), same family as ToG / current strata:

`[0,10)`, `[10,20)`, `[20,30)`, `[30,∞)`  
(ToG often starts at 2–10; document if excluding pre-2.)

Always report **\(n\) / matches per band** — late bands are survivorship-selected unfinished matches.

---

## 3. Architecture choice (locked default)

### Choice A — shared time-conditional model (**default**)

\[
\widehat{V}_\theta(S_t,t)
\]

One frozen \(\theta\) takes state and clock. Time-band tables are **evaluations** of that single map. Preferred starting point because \(\Delta\widehat{V}\) compares two outputs of the **same** evaluator.

### Choice B — per-band / per-minute models (**comparator only until proven**)

\[
\widehat{V}_t(S_t)
\]

Precedent: Hodge et al. (live esports WP, separate models by minute).  
**Extra requirement for us:** cross-time **comparability**. At a band boundary, state-fixed jumps from switching \(\widehat{V}_t\) must be measured; good per-band AUC does **not** auto-approve \(\Delta\widehat{V}\).

**Decision rule:** build A first; keep B as ablation. Adopt B only if (i) development scoring clearly better **and** (ii) boundary / continuity checks pass.

Shared models can still jump from binning, refresh lag, or calibration switches — run the same continuity checks on A.

Do **not** smooth away real event-driven WP moves; suppress only **spurious** jumps from model switch / implementation / observation refresh.

---

## 4. Learning units (separate from engagements)

Training instances:

\[
(m,\,t,\,X_{m,\le t},\,W_m)
\]

on **eligible matches’ timed states**, **not** only times that later host an engagement. Goal is general match WP, not engagement-conditional WP.

- Base queries: minute grid (Match-V5 frames).  
- Diagnostics: off-grid times (engagement pre/end) with the **same** state reconstruction rules.  
- Between minutes: events update; gold/position may stay on last frame — encode that honestly (staleness / age features as needed).  
- Early history shortfall: explicit mask / burn-in rule — **do not** silently drop early rows or band coverage shifts.

**Input set for \(\widehat{V}\)** is redefined for \(W\); it is **not** permanently capped at the 352-d \(q\) ridge. Within any model horse-race, freeze one shared input definition.

---

## 5. Candidate slate (first round)

| Arm | Role | Question |
|---|---|---|
| Time-conditional **logistic** | Interpretable floor | Where is the linear baseline? |
| Time-conditional **LightGBM** | Shared nonlinear | Does nonlinearity help on the same inputs? |
| Time-conditional **MLP** ± calibration | Prob. quality | Neural probs + Kim-style calibration compare |
| **Per-band LightGBM** | Choice B comparator | Does specialization beat shared \(\theta\)? |
| History stack (3–5 real minute frames) / GRU | Second wave | Does true history beat current state alone? |

Do **not** require CoG’s 30 s fight window as \(\widehat{V}\) history (CoG already noted near-static snapshots there). Prefer **current + last 3 or 5 minute frames**.

First lock **representation + time handling** (rows 1–4); history expansion after.

Logistic remains a **reference**, not a mandatory production \(\widehat{V}\).

---

## 6. Selection metrics (not AUC-only)

\(\Delta\widehat{V}\) uses probability **levels**, so prefer **proper scoring rules** (Gneiting–Raftery):

| Layer | Report |
|---|---|
| Minute / band curves | AUC, **Brier**, log loss, \(n\), matches |
| Predeclared bands | Calibration (ECE / reliability), effect sizes + uncertainty |
| Overall | Mean under natural query measure **and** time-balanced summary |
| Application times | Engagement pre / end + off-grid diagnostics |

**Proposed primary selection (dev):** time-balanced Brier

\[
L_{\mathrm{time}}=\sum_{b=1}^{B}\alpha_b L_b
\]

with \(\alpha_b\) fixed **a priori**; log loss as tie-break; band ECE + AUC as diagnostics.

Within a band, use **match-equal** total weight; bootstrap / CI at **match** cluster level (Brill et al. motivate dependence — do not copy their error sizes).

**No arbitrary absolute gates yet** (e.g. “AUC≥0.8 at 10 min”). First: same-query compare to legacy \(\widehat{V}\), band calibration, and where new candidates win/lose.

---

## 7. After WP quality: \(\Delta\widehat{V}\) checks (before SVI freeze)

1. **Same frozen lineage** at both times (preprocess + calib). If stochastic nets: check whether inference noise flips small \(\Delta\widehat{V}\) signs.  
2. **Decompose jumps:** band boundary (no new events) vs frame refresh lag vs real events; quiet matched windows as reference (not causal).  
3. **Calibration across time:** one strictly increasing \(g\) preserves \(\mathrm{sign}(\Delta)\); **per-band different** \(g_b\) does **not**. Lower ECE alone ≠ license for \(\Delta\widehat{V}\).  
4. **Spec stability:** alternate reasonable \(\widehat{V}\) — how often does \(\mathrm{sign}(\Delta\widehat{V})\) flip?

Do **not** choose \(\widehat{V}\) to make \(q\) easier.

**OOF rule preserved:** TRAIN engagement labels from \(\widehat{V}\) that never saw that match’s \(W\) (joint pre/end under same OOF fold).

---

## 8. Literature (V redesign map)

| Work | Use | Do not |
|---|---|---|
| **Hodge et al. (2021)** ToG — live WP, history features, time-wise eval / per-minute models | Time-band eval + history axes; Choice B precedent | Auto-adopt per-minute models for \(\Delta V\) |
| **Kim, Lee & Chung (2020)** CoG — calibrated LoL WP | Prob. quality + calib compare; MLP±calib | Mandate DU loss as day-one default |
| **Maymin (2021)** JQAS — WP change for kill value (**outside** CoG/ToG but central for us) | Match WP → event value via \(\Delta\widehat{V}\); **one** shared model | Copy sparse logistic features as our ceiling |
| **Ke et al. (2022)** | Optional: past fights into match WP | Confuse with predicting the next fight winner |
| **Hitar-García et al. (2023)** | Optional pre-game / roster features | Treat as in-game time-band WP paper |
| **Gneiting–Raftery (2007)** | Proper scores for selection | — |
| **Brill–Yurko–Wyner (2024)** | Match-level dependence / WP uncertainty framing | Paste their CI widths onto us |

**Read order:** Hodge → Kim → Maymin.

Detail compare of *our* sealed curves vs CoG/ToG: [TIMEBAND_COG_TOG_SVI_COMPARE_20260919.md](TIMEBAND_COG_TOG_SVI_COMPARE_20260919.md).

---

## 9. Work packages

| ID | Freeze |
|---|---|
| **V-1** | Task contract |
| **V-2 wave 1** | shared/per-band LGBM + legacy — **too thin; not final** |
| **V-2 matrix** | [V2_CANDIDATE_MATRIX_20260919.md](V2_CANDIDATE_MATRIX_20260919.md) Tier 0–3 (+ Tier 4 when inputs exist) |
| **V-3** | Freeze only after Tier 0 (+ expected Tier 1) under \(L_{\mathrm{time}}\) |
| **V-4 partial→full** | Calibrated continuity + \(D_{\mathrm{switch}}\) / quiet |
| **Then** | New SVI / \(q\) |

---

## 10. One-sentence lock

> Race the **WP-literature and CoG-adapted learners** for shared time-conditional \(\widehat{V}=g\circ f\circ T\), select on time-aware proper scores, check \(\Delta\widehat{V}\) sensitivity — **then** rebuild SVI and \(q\).
