"""Why does a predicted engagement outcome not improve match prediction, and what would?

Three questions, one script, no retraining of anything frozen:

  1. WHERE does the realized outcome's information sit?  Decompose the A -> C gain by |delta-V|
     quartile and by headroom, and put the predictor's own accuracy on the same axis.  If C's gain
     lives in the quartiles where the predictor is at chance, the null in B is explained.

  2. WHAT WOULD IT BUY to be able to predict one quartile?  A partial oracle: realized label inside
     quartile q, the actual prediction everywhere else.  The gap to A is the prize for that quartile.

  3. HOW GOOD would a predictor have to be?  Sweep a synthetic predictor of controlled quality -
     score = y + N(0, sigma), which has AUC = Phi(1/(sigma*sqrt(2))) - across AUC 0.60 .. 1.00,
     refit the stacking each time, and find where the gain first excludes zero.  The synthetic
     predictor is an oracle-with-noise, so this is an upper bound for a real predictor of that AUC.

Also refits the ceiling with market_event as the engagement label, since the contract now names
market_event the outcome axis while the original A/B/C used the delta-V sign.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import norm
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from train.state_value_experiment import learner, match_weights, metrics, paired_bootstrap
from scripts.train_engagement_predictor_v3 import load_split

SWEEP_AUC = (0.60, 0.665, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00)


def wauc(y, p, g):
    return float(roc_auc_score(y, p, sample_weight=match_weights(g))) if len(set(y.tolist())) == 2 else None


def synthetic(y, target_auc, rng):
    """A predictor of y with the requested AUC: y + gaussian noise, sigma from the normal model."""
    if target_auc >= 1.0:
        return y.astype(float)
    sigma = 1.0 / (np.sqrt(2.0) * norm.ppf(target_auc))
    return y.astype(float) + rng.normal(0.0, sigma, size=len(y))


def quartiles(values):
    edges = np.quantile(values, [0, .25, .5, .75, 1.])
    out = []
    for q in range(4):
        lo, hi = edges[q], edges[q + 1]
        mask = (values >= lo) & (values <= hi if q == 3 else values < hi)
        out.append((f"Q{q+1} [{lo:.1f},{hi:.1f}]", mask))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--v3-dir", type=Path, default=ROOT / "outputs/temporal_winprob_v3_buckets")
    ap.add_argument("--dataset", type=Path, default=ROOT / "outputs/state_value_main_50k")
    ap.add_argument("--pred-dir", type=Path, default=ROOT / "outputs/engagement_predictor_v3")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs/engagement_predictor_v3/stacking_ceiling.json")
    ap.add_argument("--trees", type=int, default=250)
    a = ap.parse_args()
    started = time.time()
    rng = np.random.default_rng(7)

    tr = load_split(a.dataset, a.v3_dir / "engagement_changes_predict_train.npz")
    te = load_split(a.dataset, a.v3_dir / "engagement_changes.npz")
    print(f"[ceiling] loaded ({time.time()-started:.0f}s)", flush=True)
    with np.load(a.pred_dir / "predictions.npz", allow_pickle=False) as z:
        stored = {k: z[k] for k in z.files}
    assert list(stored["test_id"].astype(str)) == list(te["id"]), "prediction rows do not match the split"
    pA = stored["A_pre_information"]
    pB = stored["B_plus_predicted_engagement"]
    pC = stored["C_plus_realized_label_RETROSPECTIVE"]
    oof, p_eng = stored["train_oof"], stored["test_p_deltaV_weighted"]
    y_match_tr, y_match_te = tr["winner"], te["winner"]
    gtr, gte = tr["match"], te["match"]
    wtr = match_weights(gtr)
    abs_pp = np.abs(te["delta"]) * 100.0
    with np.load(a.v3_dir / "engagement_changes.npz", allow_pickle=False) as z:
        order = {e: i for i, e in enumerate(z["id"].astype(str))}
        v_pre = z["expanded"][[order[e] for e in te["id"]], 0]
    headroom = np.minimum(v_pre, 1.0 - v_pre)

    out = {"note": "A/B/C reuse the stored predictions; the engagement label there is sign(delta-V)",
           "stored_check": {"A": wauc(y_match_te, pA, gte), "B": wauc(y_match_te, pB, gte), "C": wauc(y_match_te, pC, gte)},
           "by_abs_deltaV_quartile": {}, "by_headroom_quartile": {}, "partial_oracle": {}, "sweep": []}

    # ---- 1. where the realized outcome's information sits -------------------------------
    for axis, values, key in (("|delta-V| pp", abs_pp, "by_abs_deltaV_quartile"),
                              ("headroom", headroom, "by_headroom_quartile")):
        for name, mask in quartiles(values):
            cell = {"n": int(mask.sum()), "matches": int(len(set(gte[mask].tolist()))),
                    "A": wauc(y_match_te[mask], pA[mask], gte[mask]),
                    "B": wauc(y_match_te[mask], pB[mask], gte[mask]),
                    "C": wauc(y_match_te[mask], pC[mask], gte[mask]),
                    "engagement_predictor_auc": wauc(te["y"][mask], p_eng[mask], gte[mask]),
                    "engagement_predictor_auc_market": (
                        wauc(te["market"][mask & (te["market"] >= 0)], stored["test_p_market"][mask & (te["market"] >= 0)],
                             gte[mask & (te["market"] >= 0)]))}
            cell["C_minus_A"] = None if cell["C"] is None else cell["C"] - cell["A"]
            cell["C_minus_A_ci"] = paired_bootstrap(y_match_te[mask], pA[mask], pC[mask], gte[mask])
            out[key][name] = cell
        print(f"[ceiling] {axis} decomposed ({time.time()-started:.0f}s)", flush=True)

    # ---- 2. partial oracle: realized label inside one quartile only ---------------------
    tr_abs = np.abs(tr["delta"]) * 100.0
    tr_edges = np.quantile(tr_abs, [0, .25, .5, .75, 1.])
    te_edges = np.quantile(abs_pp, [0, .25, .5, .75, 1.])
    for q in range(4):
        f_tr, f_te = oof.copy(), p_eng.copy()
        m_tr = (tr_abs >= tr_edges[q]) & (tr_abs <= te_edges[4] if q == 3 else tr_abs < tr_edges[q + 1])
        m_te = (abs_pp >= te_edges[q]) & (abs_pp <= te_edges[4] if q == 3 else abs_pp < te_edges[q + 1])
        f_tr[m_tr], f_te[m_te] = tr["y"][m_tr], te["y"][m_te]
        m = learner(trees=a.trees).fit(np.column_stack([tr["X"], f_tr]), y_match_tr, sample_weight=wtr)
        p = m.predict_proba(np.column_stack([te["X"], f_te]))[:, 1]
        out["partial_oracle"][f"Q{q+1}"] = {
            "oracle_rows_test": int(m_te.sum()), "auc": wauc(y_match_te, p, gte),
            "minus_B": paired_bootstrap(y_match_te, pB, p, gte), "minus_A": paired_bootstrap(y_match_te, pA, p, gte)}
        print(f"[ceiling] partial oracle Q{q+1} AUC {out['partial_oracle'][f'Q{q+1}']['auc']:.6f} ({time.time()-started:.0f}s)", flush=True)

    # ---- 3. how good would a predictor have to be? -------------------------------------
    for target in SWEEP_AUC:
        s_tr, s_te = synthetic(tr["y"], target, rng), synthetic(te["y"], target, rng)
        realised = wauc(te["y"], s_te, gte)
        m = learner(trees=a.trees).fit(np.column_stack([tr["X"], s_tr]), y_match_tr, sample_weight=wtr)
        p = m.predict_proba(np.column_stack([te["X"], s_te]))[:, 1]
        boot = paired_bootstrap(y_match_te, pA, p, gte)
        row = {"target_engagement_auc": target, "realised_engagement_auc": realised,
               "match_auc": wauc(y_match_te, p, gte), "minus_A": boot,
               "excludes_zero": bool(boot["auc_improvement_ci95"][0] > 0)}
        out["sweep"].append(row)
        print(f"[ceiling] sweep {target:.3f} (realised {realised:.4f}) -> match AUC {row['match_auc']:.6f} "
              f"CI {np.round(boot['auc_improvement_ci95'], 5).tolist()} ({time.time()-started:.0f}s)", flush=True)

    # ---- ceiling under the contract's own label ----------------------------------------
    m = learner(trees=a.trees).fit(np.column_stack([tr["X"], tr["market"]]), y_match_tr, sample_weight=wtr)
    p = m.predict_proba(np.column_stack([te["X"], te["market"]]))[:, 1]
    out["ceiling_with_market_event_label"] = {"auc": wauc(y_match_te, p, gte),
                                              "minus_A": paired_bootstrap(y_match_te, pA, p, gte)}
    first = next((r for r in out["sweep"] if r["excludes_zero"]), None)
    out["threshold_engagement_auc_for_a_detectable_gain"] = first["target_engagement_auc"] if first else None
    out["elapsed_seconds"] = round(time.time() - started, 2)
    a.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({"threshold": out["threshold_engagement_auc_for_a_detectable_gain"],
                      "ceiling_market": out["ceiling_with_market_event_label"]["auc"],
                      "C_minus_A_by_quartile": {k: (None if v["C_minus_A"] is None else round(v["C_minus_A"], 5))
                                                for k, v in out["by_abs_deltaV_quartile"].items()}}, indent=2), flush=True)


if __name__ == "__main__":
    main()
