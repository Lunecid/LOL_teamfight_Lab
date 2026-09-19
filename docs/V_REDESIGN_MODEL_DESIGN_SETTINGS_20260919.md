# V redesign — full model design & settings pack

**Date:** 2026-09-19  
**Branch intent:** document every horse-race arm’s **input vector**, **architecture**, **hyperparameters**, **calibration**, and **selection protocol** so IDs (`A0`, `H5`, …) are auditable.  
**Epistemic:** `EXPLORATORY_V_REDESIGN_SAME_COHORT_PRIOR_TEST_EXPOSURE`  
**Related:** [V_REDESIGN_ID_GLOSSARY_20260919.md](V_REDESIGN_ID_GLOSSARY_20260919.md) · [V2_CANDIDATE_MATRIX_20260919.md](V2_CANDIDATE_MATRIX_20260919.md) · [V_REDESIGN_CONTRACT_20260919.md](V_REDESIGN_CONTRACT_20260919.md)

---

## 0. Task & evaluation (shared by all arms)

| Item | Setting |
|---|---|
| Estimand | \(\widehat{V}(X_{\le t})=\widehat{P}(W=1\mid X_{\le t})\) (blue win) |
| Label \(W\) | Match outcome from MAIN roles |
| Rows | `bucket_only=True` StateV2 query rows |
| TRAIN | `fold0`…`fold4` (match-equal sample weights \(1/\#rows\_in\_match\)) |
| V_CAL | fit monotone calibrator \(g\) only |
| V_SELECT | select by \(L_{\mathrm{time}}=\sum_{b}\alpha_b\mathrm{Brier}_b\), \(\alpha_b=1/4\) |
| Time bands \(b\) | `[0,10)`, `[10,20)`, `[20,30)`, `[30,∞)` minutes |
| TEST | sealed ledger + continuity **diagnostic** (not selection) |
| Evaluator map | \(V_{\mathrm{final}}=g\circ f\circ T\); \(g=\) PosSlopeSigmoid (Platt on logit of \(f\)); reject if slope ≤ 0 |
| ΔV policy | **Choice A** a priori (one shared \(f\) for all times) |
| Scripts | `scripts/rr20260919_v_redesign_fit_wave2.py`, `…_fit_wave3_tier23.py`, `…_continuity_wave2.py` |

**Choice A vs B (policy, not a learner):**

- **Choice A:** one \(f\) for all \(t\) (time may enter as features / via history).  
- **Choice B:** separate \(f_b\) per time band (arm **B0**).

---

## 1. Shared input definition \(T(X)\)

### 1.1 Source matrix

- Layout: MAIN corpus via `fc20260915_data.load_v_rows(..., bucket_only=True)`.
- Raw row: StateV2 feature vector + `match`, `query_ms`.
- Dropped column for redesign arms: **`snapshot_age_s`**.
- Per-frame dimension after drop: **\(d=361\)** (verified on fitted A0/A2b/A3 packs).

### 1.2 Column handling by learner family

| Family | Matrix construction | Categoricals |
|---|---|---|
| A0 logistic | `feature_matrix(X, names, "expanded")` → `keep` indices into raw \(X\) (\(|keep|=361\)) | pipeline inside `train.state_value_experiment.logistic` |
| A1 RF / A3 MLP | `expanded_X`: drop `snapshot_age_s` → float matrix \(N\times 361\) | none special-cased |
| A2 / A2b / B0 LGBM | same \(N\times 361\); `*_champion_id` cast to pandas `category` | yes |
| H3 / H5 | stack last \(K\) frames of the 361-d vector + \(K\) mask bits | LGBM on flat float (no category) |
| S\* sequence | tensor \((N,K,361)\) + mask \((N,K)\); StandardScaler fit on **last** frame only | — |
| legacy | sealed `v_final_raw.joblib` + `predict_matrix` (compare-only; not redesign \(T\)) | sealed |

### 1.3 History / sequence geometry

For match-sorted bucket queries, at row \(t\) take frames \(\{t-K+1,\ldots,t\}\) (right-aligned); earlier slots zero + mask \(0\).

| Arm | \(K\) | Model input shape |
|---|---:|---|
| A\* / B0 / legacy | 1 | \(N\times 361\) (current frame) |
| **H3** | 3 | flat \(N\times (3\cdot 361 + 3)=N\times 1086\) |
| **H5** / **S\*** | 5 | H5 flat \(N\times (5\cdot 361 + 5)=N\times 1810\); S\* seq \(N\times 5\times 361\) + mask |

**Same feature vocabulary, different temporal assembly.** No player-graph / event stream in Tier 0–3 (Tier 4 blocked).

---

## 2. Model catalogue (design + settings)

### 2.1 Tier 0–1 — current-frame tabular (wave-2)

Artifact root: `outputs/v_redesign_wave2_20260919/`  
Script: `scripts/rr20260919_v_redesign_fit_wave2.py`

#### A0 — Shared logistic (Choice A) — **provisional freeze candidate**

| Field | Value |
|---|---|
| Precedent | Maymin (2021); Hodge LR |
| Input | current-frame expanded StateV2, \(d=361\) |
| Model | `logistic(cols, C=V_C)` with `V_C=0.01` (repo default) |
| Weights | match-equal |
| Calib \(g\) | PosSlopeSigmoid on V_CAL (`coef≈1.008`, `intercept≈-0.0235`) |
| V_SELECT \(L_{\mathrm{time}}\) | **0.137638** (mech. winner among eligible redesign) |
| Pack | `models/A0_shared_logistic.joblib` |

#### A1 — Shared random forest

| Field | Value |
|---|---|
| Precedent | Hodge RF |
| Input | \(N\times 361\) |
| Hypers | `n_estimators=300`, `max_depth=20`, `min_samples_leaf=50`, `max_features=sqrt`, `max_samples=0.5`, `n_jobs=4`, `random_state=7` |
| \(L_{\mathrm{time}}\) | 0.143363 (weak) |

#### A2 — Shared LightGBM (wave-1 hypers)

| Field | Value |
|---|---|
| Precedent | Hodge LGBM; wave-1 baseline |
| Input | \(N\times 361\) + champion categoricals |
| Hypers | `n_estimators=400`, `lr=0.05`, `num_leaves=31`, `subsample=0.8`, `colsample_bytree=0.8`, `reg_lambda=1.0`, `seed=7`, `n_jobs=4` |
| \(L_{\mathrm{time}}\) | 0.139299 |

#### A2b — Shared LightGBM (CoG paper hypers)

| Field | Value |
|---|---|
| Precedent | CoG §4.2.1 / MODELS.md |
| Input | same as A2 |
| Hypers | `n_estimators=5000`, `lr=0.03`, `max_depth=6`, `num_leaves=31`, `subsample=0.7`, `colsample_bytree=0.7`, `reg_alpha=1.0`, `reg_lambda=5.0`, `min_child_samples=200`, early_stopping=200 on V_CAL |
| \(L_{\mathrm{time}}\) | 0.138907 |

#### A3 — Shared MLP

| Field | Value |
|---|---|
| Precedent | Kim CoG WP; CoG matched MLP |
| Input | \(N\times 361\), StandardScaler on TRAIN |
| Net | Linear→LayerNorm→GELU→Dropout(0.1) × hidden `(256,128)` → Linear(1) |
| Opt | AdamW `lr=1e-3`, `weight_decay=1e-4` |
| Train | `max_epochs=40`, `batch=8192`, BCE-with-logits match-weighted, early stop patience 6 on V_CAL Brier |
| \(L_{\mathrm{time}}\) | 0.138532 |

#### B0 — Per-band LightGBM (Choice B)

| Field | Value |
|---|---|
| Precedent | Hodge per-minute spirit |
| Input | same 361-d; **four separate** LGBM fits by band |
| Hypers | wave-1 LGBM settings (as A2) per band |
| \(L_{\mathrm{time}}\) | 0.139016 |

#### legacy — Old sealed logistic (ineligible)

| Field | Value |
|---|---|
| Path | `…/models/v/v_final_raw.joblib` |
| Role | compare-only WP / continuity reference |
| \(L_{\mathrm{time}}\) | 0.137208 (slightly better than A0; **not** redesign freeze) |

---

### 2.2 Tier 2–3 — history & sequence (wave-3)

Artifact root: `outputs/v_redesign_wave3_tier23_20260919/`  
Script: `scripts/rr20260919_v_redesign_fit_wave3_tier23.py`

#### H3 / H5 — Flat history LightGBM

| Field | H3 | H5 |
|---|---|---|
| Precedent | Hodge history | same |
| Input | flat \(1086\)-d | flat \(1810\)-d |
| Hypers | same as A2 wave-1 LGBM | same |
| \(L_{\mathrm{time}}\) | 0.138609 | **0.138488** (best Tier 2–3; still behind A0) |

#### S\* — Sequence nets on \(K=5\)

Common train settings:

- StandardScaler fit on **last-frame** features; applied to all \(K\) steps  
- AdamW `lr=1e-3`, `weight_decay=1e-4`  
- BCE-with-logits, match weights  
- `max_epochs=25`, `batch=1024`, patience 5 on V_CAL Brier  
- Head: Linear→GELU→Dropout(0.1)→Linear(1)  
- Persist: `state_dict` + scaler (not live Module)  
- Seed: 7  

| ID | Encoder | Key sizes | \(L_{\mathrm{time}}\) |
|---|---|---|---:|
| **S_bigru_K5** | BiGRU | 2 layers, hidden 128, bidir, dropout 0.1 | 0.157962 |
| **S_bilstm_K5** | BiLSTM | same | 0.158649 |
| **S_transformer_K5** | TransformerEncoder | proj 361→64; 2 layers; nhead=4; FFN 128; dropout 0.1; masked mean pool | 0.139787 |
| **S_tcn_K5** | Dilated Conv1d stack | channels 64; k=3; dilations 1/2/4; last real timestep | 0.157817 |
| S_mamba | — | **skipped** (`mamba_ssm` unavailable) | — |

---

### 2.3 Tier 4 — GNN / event / fusion

**Blocked:** no query-time player-node / adjacency / event stream in current StateV2-only rows. Do not claim Tier 4 ran.

---

## 3. Continuity protocol (partial V-4)

Script: `scripts/rr20260919_v_redesign_continuity_wave2.py --with-h5`  
Ledger: [CONTINUITY_LEDGER_WAVE2_20260919.md](CONTINUITY_LEDGER_WAVE2_20260919.md)

| Check | Definition | Arms |
|---|---|---|
| Consecutive \|Δp\| | TEST bucket queries; interior vs band-boundary minute crossings | A0, A3, A2b, H5, legacy |
| Sign(ΔV) agree | MAIN_TEST engagements, h90 endpoint, nonzero signs | pairwise among above |
| Not yet | \(D_{\mathrm{switch}}(x)\), quiet/event strata, frame-refresh-only | — |

Headline: A0 CAL excess **0.0112** (= legacy); A0↔legacy sign agree **0.9815**.

---

## 4. Freeze status (as of this pack)

| Decision | Status |
|---|---|
| Mechanical WP winner (eligible) | **A0 shared logistic** |
| History/sequence beat A0? | **No** |
| Partial V-4 continuity overturn A0? | **No** |
| Provisional freeze | **Choice A + A0** (still need full V-4 before final freeze_manifest) |
| Do not freeze | H\*, S\*, B0, A1; legacy ineligible |

---

## 5. Artifact index

| Path | Contents |
|---|---|
| `docs/V_REDESIGN_ID_GLOSSARY_20260919.md` | Plain-language ID map |
| `docs/BAND_LEDGER_WAVE2_TIER01_20260919.md` | Tier 0–1 WP table |
| `docs/BAND_LEDGER_WAVE3_TIER23_20260919.md` | Tier 2–3 WP table |
| `docs/CONTINUITY_LEDGER_WAVE2_20260919.md` | Continuity summary |
| `docs/V2_CANDIDATE_MATRIX_20260919.md` | Execution matrix / status board |
| `outputs/v_redesign_wave2_20260919/` | models + `results.json` + continuity reports (local) |
| `outputs/v_redesign_wave3_tier23_20260919/` | history/seq models + `results.json` (local) |

---

## 6. One-sentence lock

> Redesign horse-race freezes a **shared input vocabulary** (StateV2 expanded, \(d=361\), no `snapshot_age_s`); **A\*/B0** score the current frame, **H\*/S\*** score the same vocabulary over last \(K\) frames; selection is \(L_{\mathrm{time}}\) on V_SELECT with Choice-A ΔV default; current best eligible arm is **A0 shared logistic**.
