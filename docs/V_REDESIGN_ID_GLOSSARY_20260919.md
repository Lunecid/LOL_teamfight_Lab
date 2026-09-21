# V redesign ID glossary (A / B / H / S / Choice)

Short IDs in ledgers are **run labels**, not paper claims. Always read with the plain-language name.

**Full settings pack (inputs, vector shapes, hypers, calib):** [V_REDESIGN_MODEL_DESIGN_SETTINGS_20260919.md](V_REDESIGN_MODEL_DESIGN_SETTINGS_20260919.md)

---

## Policy choices (not models)

| ID | Meaning |
|---|---|
| **Choice A** | One shared model for all match times (time can be a feature). **A priori default** for ΔV / SVI because the same \(f\) scores pre and post. |
| **Choice B** | Separate models per time band (`[0,10)`, `[10,20)`, `[20,30)`, `[30,∞)`). Ablation unless continuity clearly favors it. |

---

## Family prefixes

| Prefix | Family | What it is |
|---|---|---|
| **A\*** | Shared tabular | Current-frame StateV2 → win prob. Choice A. |
| **B\*** | Per-band tabular | Same features, but a different \(f\) in each time band. Choice B. |
| **H\*** | History (flat) | Last **K** minute frames concatenated → LightGBM (pads early). |
| **S\*** | Sequence | Same K-frame stack, but sequential net (BiGRU, Transformer, …). |
| **REF_** | Reference | Already-selected arm carried into a later wave for comparison. |
| **legacy_** | Old sealed V | Pre-redesign logistic; compare-only, not eligible as redesign freeze. |

---

## Concrete arms (what we actually ran)

### Tier 0–1 — shared / per-band (wave-2)

| ID | Plain name | Learner | Input |
|---|---|---|---|
| **A0** | Shared logistic | Logistic regression | Current frame only |
| **A1** | Shared random forest | RF | Current frame |
| **A2** | Shared LightGBM (wave-1 hypers) | LGBM ~400 trees | Current frame |
| **A2b** | Shared LightGBM (CoG hypers) | LGBM CoG paper settings | Current frame |
| **A3** | Shared MLP | Torch MLP + calib \(g\) | Current frame |
| **B0** | Per-band LightGBM | One LGBM per time band | Current frame in-band |

### Tier 2–3 — history / sequence (wave-3)

| ID | Plain name | Learner | Input |
|---|---|---|---|
| **H3** | History-3 LGBM | LightGBM on flat stack | Last **3** frames |
| **H5** | History-5 LGBM | LightGBM on flat stack | Last **5** frames |
| **S_bigru_K5** | BiGRU sequence | BiGRU | K=5 sequence |
| **S_bilstm_K5** | BiLSTM sequence | BiLSTM | K=5 |
| **S_transformer_K5** | Transformer sequence | Transformer encoder | K=5 |
| **S_tcn_K5** | TCN sequence | Temporal conv | K=5 |

### Other labels

| ID | Plain name |
|---|---|
| **REF_A0_logistic_K1** | Same as A0, cited as reference in the Tier 2–3 horse-race |
| **legacy_logistic_raw** | Old sealed match-WP logistic (ineligible for redesign freeze) |

---

## How to say freeze status in one sentence

**Choice A + A0 (shared logistic on the current frame)** is the provisional redesign freeze candidate: best eligible \(L_{\mathrm{time}}\), continuity excess ≈ legacy, sign(ΔV) ≈98% with legacy. History (**H\***) and sequence (**S\***) did not beat it on WP; they are recorded ablations.
