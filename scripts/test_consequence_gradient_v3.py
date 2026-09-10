"""Is the "predictability falls as consequence rises" gradient real, or a collider artefact?

Four independent reviews raised the same objection: |delta-V| is computed FROM the outcome, so
stratifying AUC on it conditions on a function of the label.  In a lopsided state a large |delta-V|
means the trailing team won, and a model that keys on the lead is then wrong by construction.  The
gradient could be mechanical.  Five tests, in order of how decisive they are:

  T1 SIMULATION NULL.  Draw y_sim ~ Bernoulli(p) from the model's own probabilities.  The predictor
     is then perfectly calibrated by construction and carries no relationship to |delta-V| beyond
     what the distribution of p in each stratum forces.  The gradient this produces IS the artefact.
     Compare it with the observed gradient.
  T2 PRE-FIGHT STAKES.  Re-stratify on stakes PREDICTED before the fight (a regressor for |delta-V|
     trained on pre-cutoff X only).  No outcome enters the stratifier, so no collider.
  T3 HEADROOM-CROSSED.  Does the |delta-V| gradient survive inside a fixed headroom band?
  T4 LEAD-ONLY BASELINES.  How much of AUC 0.6649 is just "the team that is ahead wins"?  Fit
     V_pre-only, gold-difference-only and a small lead+phase model on the same rows.
  T5 INTERVALS.  Match-level bootstrap CIs on every stratum AUC, so no single number stands bare.

Read-only with respect to frozen artefacts; fits only small diagnostic models.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from lightgbm import LGBMRegressor
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from train.state_value_experiment import learner, match_weights, metrics
from scripts.train_engagement_predictor_v3 import load_split

N_BOOT = 300


def wauc(y, p, g):
    y = np.asarray(y)
    if len(set(y.tolist())) < 2:
        return None
    return float(roc_auc_score(y, p, sample_weight=match_weights(g)))


def auc_ci(y, p, g, n_boot=N_BOOT, seed=7):
    """Match-level bootstrap CI for a single AUC."""
    y = np.asarray(y)
    if len(set(y.tolist())) < 2:
        return {"auc": None, "reason": "single class"}
    uniq, inverse = np.unique(g, return_inverse=True)
    rows = [np.flatnonzero(inverse == i) for i in range(len(uniq))]
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        pick = rng.integers(len(uniq), size=len(uniq))
        idx = np.concatenate([rows[i] for i in pick])
        w = np.concatenate([np.full(len(rows[i]), 1.0 / len(rows[i])) for i in pick])
        if len(set(y[idx].tolist())) < 2:
            continue
        draws.append(float(roc_auc_score(y[idx], p[idx], sample_weight=w / w.mean())))
    a = np.asarray(draws)
    return {"auc": wauc(y, p, g), "lo": float(np.percentile(a, 2.5)), "hi": float(np.percentile(a, 97.5)),
            "replicates": int(len(a))}


def quartile_masks(values, labels=("Q1", "Q2", "Q3", "Q4")):
    edges = np.quantile(values, [0, .25, .5, .75, 1.])
    out = []
    for q, name in enumerate(labels):
        lo, hi = edges[q], edges[q + 1]
        mask = (values >= lo) & (values <= hi if q == 3 else values < hi)
        out.append((f"{name} [{lo:.3g},{hi:.3g}]", mask))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--v3-dir", type=Path, default=ROOT / "outputs/temporal_winprob_v3_buckets")
    ap.add_argument("--dataset", type=Path, default=ROOT / "outputs/state_value_main_50k")
    ap.add_argument("--pred-dir", type=Path, default=ROOT / "outputs/engagement_predictor_v3")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs/engagement_predictor_v3/consequence_gradient.json")
    ap.add_argument("--trees", type=int, default=250)
    a = ap.parse_args()
    started = time.time()

    tr = load_split(a.dataset, a.v3_dir / "engagement_changes_predict_train.npz")
    te = load_split(a.dataset, a.v3_dir / "engagement_changes.npz")
    with np.load(a.pred_dir / "predictions.npz", allow_pickle=False) as z:
        stored = {k: z[k] for k in z.files}
    assert list(stored["test_id"].astype(str)) == list(te["id"])
    p_market = stored["test_p_market"]
    x_names = json.loads((a.dataset / "schema.json").read_text(encoding="utf-8"))["input_names"]

    def vpre(split, path):
        with np.load(path, allow_pickle=False) as z:
            order = {e: i for i, e in enumerate(z["id"].astype(str))}
            return z["expanded"][[order[e] for e in split["id"]], 0]
    v_tr, v_te = vpre(tr, a.v3_dir / "engagement_changes_predict_train.npz"), vpre(te, a.v3_dir / "engagement_changes.npz")
    h_tr, h_te = np.minimum(v_tr, 1 - v_tr), np.minimum(v_te, 1 - v_te)
    abs_tr, abs_te = np.abs(tr["delta"]) * 100, np.abs(te["delta"]) * 100
    lab = te["market"] >= 0
    y, g = te["market"], te["match"]
    print(f"[gradient] loaded, {lab.sum()} labelled test rows ({time.time()-started:.0f}s)", flush=True)

    out = {"rows_labelled": int(lab.sum()), "matches": int(len(set(g[lab].tolist())))}

    # ---- T1 simulation null -----------------------------------------------------------
    rng = np.random.default_rng(7)
    sims = []
    for rep in range(20):
        y_sim = (rng.random(len(p_market)) < p_market).astype(int)
        sims.append([wauc(y_sim[m & lab], p_market[m & lab], g[m & lab]) for _, m in quartile_masks(abs_te)])
    sims = np.array([[np.nan if x is None else x for x in row] for row in sims], dtype=float)
    observed = [wauc(y[m & lab], p_market[m & lab], g[m & lab]) for _, m in quartile_masks(abs_te)]
    out["T1_simulation_null"] = {
        "design": "y_sim ~ Bernoulli(p_market); the predictor is calibrated by construction and "
                  "independent of |delta-V| given p, so any gradient here is the stratification artefact",
        "quartiles": [n for n, _ in quartile_masks(abs_te)],
        "observed_auc": observed,
        "null_auc_mean": np.nanmean(sims, axis=0).tolist(),
        "null_auc_p2.5": np.nanpercentile(sims, 2.5, axis=0).tolist(),
        "null_auc_p97.5": np.nanpercentile(sims, 97.5, axis=0).tolist(),
        "observed_minus_null": (np.array(observed) - np.nanmean(sims, axis=0)).tolist(),
        "observed_spread": float(observed[0] - observed[-1]),
        "null_spread_mean": float(np.nanmean(sims[:, 0] - sims[:, -1]))}
    print(f"[gradient] T1 observed spread {out['T1_simulation_null']['observed_spread']:.4f} vs "
          f"null {out['T1_simulation_null']['null_spread_mean']:.4f} ({time.time()-started:.0f}s)", flush=True)

    # ---- T2 pre-fight stakes (no collider) --------------------------------------------
    reg = LGBMRegressor(n_estimators=a.trees, num_leaves=15, learning_rate=.04, min_child_samples=40,
                        reg_lambda=1., colsample_bytree=.9, random_state=7, n_jobs=4, verbosity=-1)
    reg.fit(tr["X"], abs_tr, sample_weight=match_weights(tr["match"]))
    stakes_hat = reg.predict(te["X"])
    out["T2_predicted_stakes"] = {
        "design": "stratify on |delta-V| PREDICTED from pre-cutoff X only; the outcome never enters the stratifier",
        "spearman_pred_vs_true_stakes": float(spearmanr(stakes_hat, abs_te).correlation),
        "by_quartile": {n: {**auc_ci(y[m & lab], p_market[m & lab], g[m & lab]),
                            "n": int((m & lab).sum()),
                            "mean_true_stakes_pp": float(abs_te[m & lab].mean()),
                            "mean_headroom": float(h_te[m & lab].mean())}
                        for n, m in quartile_masks(stakes_hat)}}
    print("[gradient] T2 " + " ".join(f"{v['auc']:.3f}" for v in out["T2_predicted_stakes"]["by_quartile"].values())
          + f" ({time.time()-started:.0f}s)", flush=True)

    # ---- T3 headroom-crossed ----------------------------------------------------------
    crossed = {}
    for hn, hm in quartile_masks(h_te, ("H1", "H2", "H3", "H4")):
        sub = h_te[hm]
        for an, am in quartile_masks(abs_te[hm]):
            cell = hm.copy()
            cell[np.flatnonzero(hm)] = am
            mm = cell & lab
            crossed[f"{hn.split()[0]} x {an.split()[0]}"] = {**auc_ci(y[mm], p_market[mm], g[mm]), "n": int(mm.sum())}
    out["T3_headroom_crossed"] = {
        "design": "|delta-V| quartiles computed WITHIN each headroom quartile; if the gradient is only "
                  "headroom it should flatten inside a band",
        "cells": crossed}
    for hn in ("H1", "H2", "H3", "H4"):
        row = [crossed[k]["auc"] for k in crossed if k.startswith(hn + " ")]
        out["T3_headroom_crossed"].setdefault("spread_within_band", {})[hn] = (
            None if any(v is None for v in row) else float(row[0] - row[-1]))
    print(f"[gradient] T3 within-band spreads {out['T3_headroom_crossed']['spread_within_band']} ({time.time()-started:.0f}s)", flush=True)

    # ---- T4 lead-only baselines -------------------------------------------------------
    gd = x_names.index("goldDiff__last") if "goldDiff__last" in x_names else None
    xd = x_names.index("xpDiff__last") if "xpDiff__last" in x_names else None
    baselines = {"V_pre_only": (v_tr.reshape(-1, 1), v_te.reshape(-1, 1))}
    if gd is not None:
        baselines["goldDiff_only"] = (tr["X"][:, [gd]], te["X"][:, [gd]])
        cols = [c for c in (gd, xd) if c is not None]
        baselines["lead_plus_phase"] = (np.column_stack([tr["X"][:, cols], v_tr, h_tr, tr["minute"]]),
                                        np.column_stack([te["X"][:, cols], v_te, h_te, te["minute"]]))
    out["T4_lead_baselines"] = {"full_X": auc_ci(y[lab], p_market[lab], g[lab])}
    for name, (A, B) in baselines.items():
        m = learner(trees=a.trees).fit(A[tr["market"] >= 0], tr["market"][tr["market"] >= 0],
                                       sample_weight=match_weights(tr["match"][tr["market"] >= 0]))
        p = m.predict_proba(B)[:, 1]
        cell = auc_ci(y[lab], p[lab], g[lab])
        cell["by_predicted_stakes_quartile"] = {n: wauc(y[mm & lab], p[mm & lab], g[mm & lab])
                                                for n, mm in quartile_masks(stakes_hat)}
        out["T4_lead_baselines"][name] = cell
        print(f"[gradient] T4 {name:18s} AUC {cell['auc']:.4f} ({time.time()-started:.0f}s)", flush=True)
    full = out["T4_lead_baselines"]["full_X"]["auc"]
    best_lead = max(v["auc"] for k, v in out["T4_lead_baselines"].items() if k != "full_X")
    out["T4_lead_baselines"]["headline"] = {
        "full_X": full, "best_lead_only": best_lead, "gap": full - best_lead,
        "reading": "the gap is what the 7,106 features add beyond reading the lead"}

    # ---- T5 intervals on the original table -------------------------------------------
    out["T5_original_table_with_ci"] = {
        n: {**auc_ci(y[m & lab], p_market[m & lab], g[m & lab]), "n": int((m & lab).sum())}
        for n, m in quartile_masks(abs_te)}
    out["elapsed_seconds"] = round(time.time() - started, 2)
    a.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({"T1_observed_vs_null_spread": [out["T1_simulation_null"]["observed_spread"],
                                                     out["T1_simulation_null"]["null_spread_mean"]],
                      "T2_predicted_stakes": {k: round(v["auc"], 4) for k, v in out["T2_predicted_stakes"]["by_quartile"].items()},
                      "T4": out["T4_lead_baselines"]["headline"],
                      "T5_Q4": out["T5_original_table_with_ci"][list(out["T5_original_table_with_ci"])[-1]]},
                     indent=2), flush=True)


if __name__ == "__main__":
    main()
