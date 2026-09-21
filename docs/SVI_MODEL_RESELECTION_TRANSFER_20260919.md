# SVI model reselection + 2026 transfer — protocol (revised)

**Status:** revised 2026-09-19 after collaborator review — **evaluation contract before model expansion**  
**Companions:** [PAPER_COHORT_CONTRACT_20260919.md](PAPER_COHORT_CONTRACT_20260919.md),  
[STRATEGIC_VALUE_LABEL_REDESIGN_20260919.md](STRATEGIC_VALUE_LABEL_REDESIGN_20260919.md)

---

## 0. Priority order (do this before growing the model zoo)

1. Lock **same-row** 15.16 T prediction table; primary contrast **\(q-\mathrm{PT}\)** (+ B40).  
2. Clarify existing sealed lift + uncertainty; research question = “how much beyond PT?”, not “must find positive lift.”  
3. Fix material denominators / ties / weights; upgrade Quiet to **state·time·duration** matching + signed drift.  
4. Then run a **lean** reselection budget.  
5. Transfer with **frozen** \(\widehat{V}\), \(q\), PT, \(b(p)\), preproc, calibrators; report \(\widehat{V}{\to}W\) and \(q{\to}\mathrm{SVI}\).

Core claim is **not** “which architecture wins AUC,” but **whether pre-engagement features add stable information about SVI beyond prior×time, and how far that can be interpreted strategically.**

---

## 1. Goal chain (revised)

```
Fix eval contract → Select (lean) → Concordance + Quiet (label meaning)
                 → Cross-target retrain (separate) → Transfer (ˆV and q)
```

| Step | What | What it answers |
|---|---|---|
| Concordance | SVI vs kill/obj/alive signs | Correspondence with observed outcomes (not “external truth”) |
| Cross-target retrain | Same family on kill/obj labels | Predictability of each target |
| Kill→SVI substitute | Kill scores → SVI probs (recalibrated) | Substitutability |
| Quiet | Matched no-kill vs fight | Drift reference |
| Transfer | Frozen stack on 16.x | \(\widehat{V}\) and \(q\) generalisation |

Retrain ≠ revalidate the SVI label.

---

## 2. Model slate (narrowed)

### Tier A — run for primary comparison (352 inputs)

| Family | Role |
|---|---|
| \(p_{\mathrm{pre}}\) | Diagnostic ordering only |
| \(b(p)\), **PT** | SVI-probability baselines (**PT = primary comparator**) |
| Logistic, **LightGBM**, **MLP** | Classical + CoG neural |
| **TabM** | Efficient MLP ensemble (meta check) |

### Tier B — optional / reviewer-facing

| Family | Role |
|---|---|
| EmbedMLP | Ablation: specify encoding (piecewise-linear vs periodic, etc.) |
| FT-Transformer, TabNet, SAINT | Only if kept for R2/R4 response; SAINT needs full row-attention inference contract (not only train-time match-disjoint batches) |
| residual MLP | Ablation only |

### Out of this 352-input round

GraphSAGE / ST-GNN / Cross-Attn / Layered / sequence models: **outside the matched-352 comparison**, not “invalid for the research question.”

**Selection:** Q_SELECT Brier → log loss → name; seeds `{7,42,123}` for stochastic families.  
**Sealed report:** 15.16 T only. Never select on TEST/external.  
**Conclusions:** limited to the declared search budget (equal #configs ≠ equal compute).

Record in every manifest: fixed vs searched hyperparameters (lr, wd, batch, embedding definition, …).

---

## 3. Transfer cohorts (unchanged inventory)

| Cohort | Matches | Headline? |
|---|---:|---|
| KR 16.13 | 10,064 | yes |
| NA1 16.13 | 10,000 | yes |
| KR 16.15 | 926 | yes |
| KR 16.14 pilot | 200 | no |

Per set: score-only; freeze full stack (§0); unseen champions noted for **both** \(\widehat{V}\) and \(q\).

---

## 4. Material + Quiet (split)

### 4.1 Concordance (label meaning)

- Decided-set definition; weights; \(n\); agree+disagree=1.  
- Language: correspondence with material outcomes.  
- Disagreement taxonomy (obj-with-SVI / tie / with-kill / other); stability under \(\widehat{V}\) variants as follow-up.

### 4.2 Cross-target retrain (not validity)

- Same family + 352 inputs on decided kill/obj rows; eval on **same** decided 15.16 rows.  
- Do not compare AUCs across different targets/samples as “SVI is a better target.”

### 4.3 Substitute check

- Map kill scores → SVI probabilities via calibrator fit on **non-TRAIN, non-TEST** rows (e.g. Q_CAL), then score on 15.16.

### 4.4 Quiet

- Match: match-id **and** \(p_{\mathrm{pre}}\) band, clock band, realized duration; frame-refresh diagnostic.  
- Report: mean \(|\widehat{\Delta V}|\) ratio (named), median ratio, mean signed \(\widehat{\Delta V}\), \(P(+)\).  
- Not “pure fight contribution.”

---

## 5. Outputs

| Path | Content |
|---|---|
| `outputs/svi_primary_table_20260919/` | **Paper prediction table** (same 15.16 rows; Brier + boot CIs) |
| `outputs/svi_lean_tabm_20260919/` | TabM lean fit + sealed score |
| `outputs/svi_overnight_20260919/` | concordance, quiet match, cross-target, transfer, Tier B |
| `outputs/svi_reselection_20260919/` | earlier aggregate freeze (superseded by primary table) |

**Locked winner:** LightGBM. TabNet ≈ LGBM; TabM close; FT collapsed under lean budget.

---

## 6. Code entry points

- `scripts/rr20260919_svi_reselection.py`  
- `scripts/rr20260919_svi_material_quiet_models.py`  
- `scripts/rr20260919_svi_transfer_2026.py`  
- `scripts/rr20260919_svi_cohort_aligned.py`  
- `train/svi_tabular_meta.py`  
- `scripts/run_deep_tabular_baselines.py` (FT/TabNet/SAINT if Tier B executed)

---

## 7. Epistemic banner (every REPORT)

Exploratory follow-up after prior TEST/external exposure; this run’s candidates/rules frozen before sealed reopen. Not confirmatory. Bootstrap = eval-sample variability under frozen artifacts unless re-fit is stated.
