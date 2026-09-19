# V-2 candidate matrix — WP literature + CoG / local ToG learners

**Status:** 2026-09-19 — Tier 0–3 complete on StateV2; **A0 logistic still leads**; Tier 4 GNN blocked on inputs.  
**Parent:** [V_REDESIGN_CONTRACT_20260919.md](V_REDESIGN_CONTRACT_20260919.md) · [WINPROB_V_DESIGN_PACK_20260919.md](WINPROB_V_DESIGN_PACK_20260919.md)  
**CoG architecture catalog:** [MODELS.md](MODELS.md) · [CoG2026_Paper.md](CoG2026_Paper.md) §4.2

**Task (unchanged):** \(\widehat{V}(X_{\le t})=\widehat{P}(W=1\mid X_{\le t})\) on public match state.  
**Selection (unchanged):** \(L_{\mathrm{time}}\) on V_SELECT; Choice A default until ΔV checks; evaluator \(g\circ f\circ T\).

---

## 0. Why wave-1 is insufficient

Wave-1 only raced **shared LGBM (400 trees)** vs **per-band LGBM** vs **legacy logistic**.  
That skips the WP papers’ own learners and almost the entire CoG / local ToG model zoo.

| Gap | Why it matters |
|---|---|
| No **Maymin-style logistic** re-fit on our \(X\) | Shared-model ΔV precedent not in the horse-race |
| No **Hodge LR / RF / LGBM** slate | Live WP literature’s actual algorithms |
| No **Kim MLP ± calib / DU** | CoG WP calibration line missing |
| No **CoG matched MLP** | Separates representation vs learner (CoG’s own lesson) |
| No **history / sequence** arms | Hodge history + CoG BiGRU/Transformer precedent |
| No **CoG GNN / fusion** | Requires node/event views — separate input track |

---

## 1. Source papers → learners (what they actually used)

### 1.1 Match / live win-probability literature

| Source | Estimand | Learners used | Map onto our V |
|---|---|---|---|
| **Maymin (2021)** JQAS | Match WP → kill value via ΔWP | **Logistic** (sparse features + time) | Tier 0: shared logistic on StateV2 |
| **Hodge et al. (2021)** IEEE ToG (DotA2 live WP) | Live match win | **Logistic, Random Forest, LightGBM** | Tier 0: same three on our \(X\); time-band / history as Choice B / Tier 2 |
| **Kim, Lee & Chung (2020)** IEEE CoG | LoL match winner + **calibration** | Neural net + **temp / Platt / DU-style calib** | Tier 0–1: MLP + calib arms |
| **Ke et al. (2022)** CoG | Match WP with past fights | (architecture secondary) | Optional later: fight-history features |
| **Hitar-García et al. (2023)** ToG | Pre-game / synergy | (pre-game) | Optional draft features only |

### 1.2 Our CoG paper (fight-outcome — adapt carefully)

CoG predicts **engagement exchange sign**, not \(W\). Learners still belong in the V horse-race **when the input view exists**.

| Family | Models ([MODELS.md](MODELS.md)) | V track |
|---|---|---|
| Tabular | **LightGBM**, **matched MLP** | Tier 1 (same StateV2 / expanded tab) |
| Sequential | BiGRU, BiLSTM, Transformer, TCN, Mamba (+ hybrid h0) | Tier 2–3 on **K-frame history** of StateV2 |
| GNN | GCN, GraphSAGE, GATv2, MPNN | Tier 4 — needs player-node / adjacency at query time |
| Spatio-temporal | ST-GNN, ST-GCN, ST-Mamba, ms_dyngraph, EventXAttn | Tier 4 |
| Fusion / stack | Layered Fusion, gated GNN–BiGRU, OOF/factorial stacking | Tier 4+ after base V views exist |

### 1.3 Local ToG / temporal_winprob pipeline (match WP on this corpus)

| Learner (manuscript) | Role for V |
|---|---|
| Scoreboard LightGBM (8 lead diffs) | Ablation: weak feature ceiling |
| Linear / logistic on expanded columns | Tier 0 |
| Paper LightGBM config | Tier 1 (vs our 400-tree wave-1) |
| MLP (deep tabular) | Tier 1 |
| BiGRU / Transformer on macro sequence | Tier 2–3 |

---

## 2. Execution tiers (required order)

### Tier 0 — WP literature must-run (same \(T\), Choice A)

| ID | Model | Precedent | Notes |
|---|---|---|---|
| A0 | Shared **logistic** (expanded StateV2) | Maymin; Hodge LR | Re-fit in redesign folder (not only legacy joblib) |
| A1 | Shared **Random Forest** | Hodge RF | Match-equal weights; cap trees / depth for 400k rows |
| A2 | Shared **LightGBM** (wave-1 + **CoG paper hyperparams**) | Hodge LGBM; CoG tab | Two hyper arms, not one rough setting |
| A3 | Shared **MLP** + **temperature / Platt** | Kim; CoG matched MLP | Report pre/post calib; DU loss = optional controlled arm |

All use \(L_{\mathrm{time}}\) on V_SELECT; TEST ledger; \(g\circ f\) continuity primary.

### Tier 1 — CoG tabular fidelity

| ID | Model | Precedent |
|---|---|---|
| A2b | LightGBM with CoG published settings (5k / lr 0.03 / depth 6 / α,λ,…) | CoG §4.2.1 / MODELS.md |
| A3b | Matched-input MLP (same \(X\) as LGBM) | CoG LightGBM–MLP gap lesson |

### Tier 2 — Time handling (Choice A history + Choice B)

| ID | Model | Precedent |
|---|---|---|
| B0 | Per-band LGBM / logistic / MLP (already started for LGBM) | Hodge per-minute spirit |
| H3 / H5 | Shared LGBM or MLP on **last K∈{3,5} minute frames** (masked early) | Hodge history; CoG sequence |

### Tier 3 — CoG sequential on history stack (StateV2 sequence)

| ID | Model | Precedent |
|---|---|---|
| S1 | BiGRU | CoG / local ToG |
| S2 | BiLSTM | CoG |
| S3 | Transformer | CoG / local ToG |
| S4 | TCN | CoG |
| S5 | Mamba | CoG |
| S1h | Hybrid h0 BiGRU | CoG |

### Tier 4 — CoG graph / event / fusion (**blocked until node+event extract at query_ms**)

Do **not** pretend StateV2-only rows can host GATv2. Build query-time player graphs + events, then:

GCN, GraphSAGE, GATv2, MPNN, ST-*, EventXAttn, Layered Fusion, stacking.

---

## 3. What is **not** required for “fair WP horse-race”

- Copying CoG’s **30 s fight window** as V history (contract: no).  
- Claiming Tier 4 ran when only tabular \(X\) exists.  
- Using fight-outcome AUC from CoG as V→\(W\) evidence.  
- Freezing on a single 400-tree LGBM because it was convenient.

---

## 4. Status board (update as runs finish)

| ID | Status | Artifact |
|---|---|---|
| A2 wave-1 (LGBM 400) | done | `v_redesign_20260919/models/shared_lgbm.joblib` |
| B0 per-band LGBM | done | `…/per_band_lgbm.joblib` |
| Legacy logistic | compare only | sealed `v_final_raw.joblib` |
| A0 shared logistic re-fit | **done** (v0 baseline) | `v_redesign_wave2_20260919/models/A0_*.joblib` |
| Wave-4 corrected Phase A/B | **done** | `v_redesign_wave4_corrected_20260919/` — **MLP expanded leads** |
| A1 RF | **done** (weak) | `…/A1_shared_rf.joblib` |
| A2b CoG-hyper LGBM | **done** | `…/A2b_shared_lgbm_cog.joblib` |
| A3 / A3b MLP ± calib | **done** (Platt-style \(g\)) | `…/A3_shared_mlp.joblib` |
| H3/H5 | **done** (behind A0) | `v_redesign_wave3_tier23_20260919/models/H{3,5}_lgbm.joblib` |
| S1–S4 (BiGRU/BiLSTM/Transformer/TCN) | **done** (Transformer≈H3; RNNs weak) | `…/S_*_K5.joblib` |
| S5 Mamba | **skipped** (no `mamba_ssm`) | — |
| Tier 4 GNN/fusion | **BLOCKED** (inputs) | — |

**Ledgers:** [BAND_LEDGER_WAVE2_TIER01_20260919.md](BAND_LEDGER_WAVE2_TIER01_20260919.md) · [BAND_LEDGER_WAVE3_TIER23_20260919.md](BAND_LEDGER_WAVE3_TIER23_20260919.md) · [BAND_LEDGER_WAVE4_CORRECTED_20260919.md](BAND_LEDGER_WAVE4_CORRECTED_20260919.md) · [CONTINUITY_LEDGER_WAVE2_20260919.md](CONTINUITY_LEDGER_WAVE2_20260919.md) · [V_REDESIGN_ID_GLOSSARY_20260919.md](V_REDESIGN_ID_GLOSSARY_20260919.md) · [V_REDESIGN_MODEL_DESIGN_SETTINGS_20260919.md](V_REDESIGN_MODEL_DESIGN_SETTINGS_20260919.md) · [V_NEXT_RUN_EXECUTION_CONTRACT_20260919.md](V_NEXT_RUN_EXECUTION_CONTRACT_20260919.md) · [V_MODEL_INPUT_DESIGN_20260919/](V_MODEL_INPUT_DESIGN_20260919/)

**Freeze rule:** INPUT_IMPL_v0 history/RNN scores superseded by wave-4 corrected re-comparison. **Corrected-input mechanical leader = embedding MLP (Expanded361)**; GRU history nearly ties; flat H5 LGBM does not beat same-learner current. Continuity on MLP (+ A0 / GRU) before final freeze. Tier 4 still input-blocked.

---

## 5. One-sentence lock

> Before locking \(\widehat{V}\), race the **win-probability papers’ learners** (logistic, RF, LightGBM, MLP±calib) and the **CoG tabular/sequence families adapted to match-state inputs**; treat CoG GNN/fusion as a separate input-complete track — not a reason to freeze a single rough LGBM.
