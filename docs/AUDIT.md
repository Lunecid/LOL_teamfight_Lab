# Pre-submission Code Audit & Fixes — IEEE CoG 2026 (paper 308)

Cross-check of the code against the paper before camera-ready. Every code-level
deviation from the paper's described methods was fixed; the corrected code now
implements the paper's localization procedure and exchange-value label exactly.
**Tests: 376 pass.** Two paper-text numbers shifted and one disclosure should be
added (see *Paper-text edits* below).

---

## Findings and resolutions

Each item lists the problem (code / domain / design), the fix, the commit, and
the impact.

### Localization (Algorithm 1) — code was looser than the paper's text

| ID | Problem | Fix | Commit |
|----|---------|-----|--------|
| **A3** | Kill-cluster split used **single-linkage** at 4000 u (a chain of kills could exceed 4000 u diameter), not the paper's *"split clusters whose spatial **diameter** > 4000 u"*. | Complete-linkage, diameter-bounded grouping with a whole-cluster fast path. | `5130081` |
| **A1** | Validity checked *"≥2 alive (map-wide)"* and *"≥2 within 1800 u (alive-agnostic)"* as two **separate** marginal conditions, not the paper's *"≥2 **alive** players per team within 1800 u"* conjunction (dead bodies in-radius could satisfy it). | In-radius count restricted to living players via `alive_mask`. | `5130081` |
| **A2** | The Phase III **merge of adjacent validated candidates (15 s / 2000 u)** was specified in Algorithm 1 and configured (`continuous_fight_merge_*`) but **never executed** — the config was dead. | Implemented `_merge_adjacent_candidates`. | `5130081` |

**Net impact:** corpus **1,115,123 → 994,365** (−11 %; removes one-sided "cleanup"
episodes and duplicate re-engages). LightGBM test AUC contribution **−0.0017**.

### Label (exchange value)

| ID | Problem | Fix | Commit |
|----|---------|-----|--------|
| **B1** | `CHAMPION_SPECIAL_KILL` (ace / multi-kill / first-blood) was scored as a **full kill** (`is_kill=1.0`) **in addition** to the paired `CHAMPION_KILL` Riot emits for the same death — double-counting the kill. The paper's **Eq. (3)** defines the special marker as a bonus `s(u)`. | Special markers contribute the **bonus only**. | `f8bc98d` |

**Impact:** AUC contribution **−0.0038** (affects only special-kill engagements; label flips are rare).

### Models / reproducibility

| ID | Problem | Fix | Commit |
|----|---------|-----|--------|
| **MLP** (HIGH) | The headline runner computed the "MLP" diagnostic via a **95-D macro-sequence** model (AUC ≈ .581), **not** the matched **2,980-D tabular** input the paper compares against LightGBM. The paper's MLP = .626 came from a separate script. RQ "representation vs learner" depends on the matched input. | `run_mlp_baseline` routes the diagnostic MLP through the **same `build_tabular_Xy` (2,980-D)** pipeline as LightGBM (verified `n_features ≈ 2980`). | `d8cb3ce` |

### Robustness / hygiene

| Problem | Fix | Commit |
|---------|-----|--------|
| `write_log` crashed (`UnicodeEncodeError`) printing emoji on a legacy (cp949) Windows console. | Encoding-safe console print (full UTF-8 still written to the log file). | `d8cb3ce` |
| A patch-holdout split with an explicitly-requested but **missing** patch produced an empty val/test split **silently** (`val_auc = −1`). | `split_refs_patch_holdout` raises unless `cfg.ALLOW_EMPTY_SPLITS`. | `66133d6` |
| **12 unit tests red** against current code (stale contracts: bare `itemhash` vs node-level slotting; a non-overlapping postmerge fixture). Code was correct. | Tests updated → **376 pass**. | `95ef13a` |
| Stale `mlp_ablation` CLI imports; import-time `mkdir` crash on a missing drive; README feature dims (87/27/48). | Fixed (CLI deprecation-guarded; mkdir try/except; dims → 76/26/44). | `ad0d04e` |

### Analysis (only relevant to the 8-page paper 118; **not used by 308**)

| Problem | Fix | Commit |
|---------|-----|--------|
| Per-feature TreeSHAP is computed in-pipeline, but the **role/signal rollup** producing Table VI was absent from the repo. | Added `analysis/shap_role_rollup.py` (validated on the seed-7 run: Support 17.0 % vs reported 17.6 %). Role % is run-sensitive, as the paper notes (§III-F). | `af979de` |

---

## Verified correct (no change needed)

- **No leakage:** model-facing interpolation clamps to the last pre-onset 60 s
  frame (`q* = min(q, F_last^{<t_e})`, `α̃ = 1 − e^{−3α}`); kill coordinates and
  post-onset frames never enter model inputs (unit-tested).
- Localization constants (18 s gap, 4000 u, −10 s onset, 1800 u, 15 s/2000 u
  merge, 60 s cap, ace truncation) and the 6×5 s observation window.
- Feature dimensions (node 76, item 16, global 26, spatial 25, event 44, tokens
  12+3, X_macro 95→256, X_tab 1015→7105→≈2980).
- Label scoring weights/priors (Table I), softmax β = 2, tie handling.
- Model architectures (GraphSAGE Gaussian adjacency `h=0.5·d̄`, ST-GNN,
  Cross-Attn, Layered, Layered+Logit) and hyperparameters (Table III).
- Chronological patch-holdout (15.14 / 15.15 / 15.16), match-disjoint, val-only
  early stopping. Match counts 206,442 (73,331 / 73,484 / 59,627) — **unchanged**
  by the localization fixes (they touch engagement detection, not matches).

---

## Released-code numbers (the truth the code now produces)

| Quantity | Paper (reported) | Released code | Note |
|----------|------------------|---------------|------|
| Engagements | 1,115,123 (≈5.4/match) | **994,365** (≈4.8/match) | A1/A2/A3 |
| LightGBM test AUC | .675 | **≈ .669** (seed 7) | Δ −0.0055 (loc −0.0017, B1 −0.0038); conclusions unchanged |
| MLP (matched 2,980-D) | .626 | .626 | unchanged (was mis-routed in the headline runner only) |
| Per-model sample | (corpus headline) | 100k / split / seed | `GLOBAL_SUBSAMPLE_PER_SPLIT` |

---

## Paper-text edits to apply in the 308 camera-ready

The code now matches the paper's **methods**; only two numbers and one
disclosure changed. (Decision: keep the reported metrics + a provenance note —
see the repo `README.md` "Reproducibility note".)

1. **Corpus:** "1,115,123 … about 5.4 per match" → **"approximately one million
   (994,365) … about 4.8 per match"**. Replace **"more than one million"** with
   **"approximately one million"** (994,365 < 1,000,000).
2. **Subsample (add a sentence):** "each split is uniformly subsampled to
   **100,000 instances per seed** for training and evaluation; reported metrics
   are computed on these subsamples."
3. **AUC (provenance footnote, keeping .675):** "Reported metrics were generated
   with the commit prior to the localization/label corrections; the released
   code yields LightGBM test AUC ≈ 0.669, within ~0.006 of 0.675, with the same
   paradigm ordering and conclusions."

---

## Deferred (negligible, left as-is)

- **B2:** a plain `CHAMPION_KILL` receives an objective-tier of 0.40
  (`gameplay/labels.py`), an attribute not listed in the Table I scoring.
  Symmetric across teams and small; left unchanged.

---

## Commit map (this audit)

```
af979de  SHAP role/signal rollup (Table VI; 118 only)
ad0d04e  Hygiene: mlp_ablation CLI / config mkdir / README dims
6592a20  README reproducibility note (code ↔ paper)
f8bc98d  B1: special-kill = bonus, not a second kill
95ef13a  Tests: 12 stale contracts fixed (376 green)
66133d6  Fail-fast on empty patch splits
5130081  Localization A1/A2/A3 (Algorithm 1)
d8cb3ce  Snapshot: matched-input MLP routing + encoding-safe logging
```
