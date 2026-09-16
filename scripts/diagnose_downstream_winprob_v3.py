"""Does predicting the engagement help predict the win-probability change it produces?

The A/B/C stack asked whether a predicted engagement improves prediction of the FINAL match winner,
about thirty minutes away, and found nothing.  This asks the proximate question instead: does it
help predict delta-V - the win-probability swing the fight itself produces - and how does that
decay as the horizon moves from the last kill to +30 s and +60 s.

For each horizon the same three layers as before, now as regressions on a continuous target:

Every number is reported BY GAME-TIME BAND as well as pooled: a win-probability swing at minute 8
and one at minute 32 are different quantities (late swings sit against less headroom), so a pooled
figure hides the thing being asked.  Bands are the cutoff minute: 2-10, 10-20, 20-30, 30+.

  A'  X only                      (7,106 pre-cutoff features)
  B'  X + PREDICTED outcome       (out-of-fold on train, model prediction on test)
  C'  X + REALIZED outcome        (retrospective ceiling)
  R   realized outcome ALONE      (one feature) - how much of the swing is just who won

Same caveat as before, stated up front: the predicted outcome is a function of X, so a tree on X
can in principle represent it and B' - A' is a representational question, not a claim about the
game.  C' - A' is the informative quantity, and R alone says how much of the swing the outcome
accounts for on its own.
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
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from train.state_value_experiment import learner, match_weights
from scripts.train_engagement_predictor_v3 import load_split

HORIZONS = (("end", 1), ("plus_30s", 2), ("plus_60s", 3))
TIME_BANDS = ((2, 10), (10, 20), (20, 30), (30, 1000))
SEED = 7


def regressor(trees, jobs):
    return LGBMRegressor(n_estimators=trees, num_leaves=31, learning_rate=.05, min_child_samples=40,
                         reg_lambda=1., colsample_bytree=.9, subsample=.9, random_state=SEED,
                         n_jobs=jobs, verbosity=-1)


def score(y_true, y_pred, groups):
    w = match_weights(groups)
    return {"spearman": float(spearmanr(y_true, y_pred).correlation),
            "r2": float(r2_score(y_true, y_pred, sample_weight=w)),
            "rmse_pp": float(np.sqrt(np.average((y_true - y_pred) ** 2, weights=w))),
            "sign_accuracy": float(np.average((y_pred > 0) == (y_true > 0), weights=w)),
            "n": int(len(y_true))}


def paired_delta(y, a, b, groups, n_boot=300, seed=SEED):
    """Match-level paired bootstrap for the difference in Spearman and R^2 (b minus a)."""
    uniq, inv = np.unique(groups, return_inverse=True)
    rows = [np.flatnonzero(inv == i) for i in range(len(uniq))]
    rng = np.random.default_rng(seed)
    sp, r2 = [], []
    for _ in range(n_boot):
        idx = np.concatenate([rows[i] for i in rng.integers(len(uniq), size=len(uniq))])
        sp.append(spearmanr(y[idx], b[idx]).correlation - spearmanr(y[idx], a[idx]).correlation)
        r2.append(r2_score(y[idx], b[idx]) - r2_score(y[idx], a[idx]))
    f = lambda v: {"mean": float(np.mean(v)), "lo": float(np.percentile(v, 2.5)),
                   "hi": float(np.percentile(v, 97.5)),
                   "excludes_zero": bool(np.percentile(v, 2.5) > 0 or np.percentile(v, 97.5) < 0)}
    return {"spearman": f(sp), "r2": f(r2), "n_boot": n_boot}


def market_oof(X, y, groups, folds, trees, jobs, cache: Path):
    if cache.exists():
        with np.load(cache, allow_pickle=False) as z:
            if len(z["oof"]) == len(y):
                print(f"[downstream] reusing cached market OOF ({cache.name})", flush=True)
                return z["oof"]
    oof = np.full(len(y), np.nan)
    for k, (tr, va) in enumerate(GroupKFold(folds).split(X, y, groups), 1):
        m = learner(trees=trees).fit(X[tr], y[tr], sample_weight=match_weights(groups[tr]))
        oof[va] = m.predict_proba(X[va])[:, 1]
        print(f"[downstream] market OOF fold {k}/{folds} auc {roc_auc_score(y[va], oof[va]):.4f}", flush=True)
    np.savez_compressed(cache, oof=oof)
    return oof


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--v3-dir", type=Path, default=ROOT / "outputs/temporal_winprob_v3_buckets")
    ap.add_argument("--dataset", type=Path, default=ROOT / "outputs/state_value_main_50k")
    ap.add_argument("--pred-dir", type=Path, default=ROOT / "outputs/engagement_predictor_v3")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs/engagement_predictor_v3/downstream_winprob.json")
    ap.add_argument("--trees", type=int, default=250)
    ap.add_argument("--jobs", type=int, default=6)
    a = ap.parse_args()
    started = time.time()

    tr = load_split(a.dataset, a.v3_dir / "engagement_changes_predict_train.npz")
    te = load_split(a.dataset, a.v3_dir / "engagement_changes.npz")

    def vseries(split, path):
        with np.load(path, allow_pickle=False) as z:
            order = {e: i for i, e in enumerate(z["id"].astype(str))}
            rows = [order[e] for e in split["id"]]
            return z["expanded"][rows]                       # (n, 4): pre, end, +30s, +60s
    V_tr = vseries(tr, a.v3_dir / "engagement_changes_predict_train.npz")
    V_te = vseries(te, a.v3_dir / "engagement_changes.npz")
    print(f"[downstream] loaded train {len(tr['y'])} / test {len(te['y'])} ({time.time()-started:.0f}s)", flush=True)

    ktr, kte = tr["market"] >= 0, te["market"] >= 0
    oof = market_oof(tr["X"][ktr], tr["market"][ktr], tr["match"][ktr], 5, a.trees, a.jobs,
                     a.pred_dir / "market_oof_predict_train.npz")
    # scatter the labelled-row OOF back to full length so any horizon mask can index it directly
    oof_full = np.full(len(tr["y"]), np.nan)
    oof_full[ktr] = oof
    full = learner(trees=a.trees).fit(tr["X"][ktr], tr["market"][ktr],
                                      sample_weight=match_weights(tr["match"][ktr]))
    p_te = full.predict_proba(te["X"])[:, 1]
    print(f"[downstream] market predictor test AUC {roc_auc_score(te['market'][kte], p_te[kte]):.4f} "
          f"({time.time()-started:.0f}s)", flush=True)

    out = {"note": "targets are win-probability changes in percentage points; the predicted outcome "
                   "is a function of X, so B'-A' is a representational question, not a game finding",
           "market_predictor_test_auc": float(roc_auc_score(te["market"][kte], p_te[kte])),
           "horizons": {}}

    for hname, hcol in HORIZONS:
        d_tr = (V_tr[:, hcol] - V_tr[:, 0]) * 100.0
        d_te = (V_te[:, hcol] - V_te[:, 0]) * 100.0
        m_tr = ktr & np.isfinite(d_tr)
        m_te = kte & np.isfinite(d_te)
        Xtr, ytr, gtr = tr["X"][m_tr], d_tr[m_tr], tr["match"][m_tr]
        Xte, yte, gte = te["X"][m_te], d_te[m_te], te["match"][m_te]
        o_tr, o_te = oof_full[m_tr], p_te[m_te]
        assert np.isfinite(o_tr).all(), "out-of-fold predictions missing for some training rows"
        r_tr, r_te = tr["market"][m_tr].astype(float), te["market"][m_te].astype(float)
        wtr = match_weights(gtr)
        cell = {"n_train": int(m_tr.sum()), "n_test": int(m_te.sum()),
                "target_sd_pp": float(yte.std()), "target_mean_pp": float(yte.mean())}
        preds = {}
        for name, add_tr, add_te in (("A_X_only", None, None),
                                     ("B_plus_predicted", o_tr, o_te),
                                     ("C_plus_realized", r_tr, r_te)):
            A = Xtr if add_tr is None else np.column_stack([Xtr, add_tr])
            B = Xte if add_te is None else np.column_stack([Xte, add_te])
            m = regressor(a.trees, a.jobs).fit(A, ytr, sample_weight=wtr)
            preds[name] = m.predict(B)
            cell[name] = score(yte, preds[name], gte)
            print(f"[downstream] {hname:9s} {name:18s} spearman {cell[name]['spearman']:+.4f} "
                  f"r2 {cell[name]['r2']:+.4f} rmse {cell[name]['rmse_pp']:.2f}pp ({time.time()-started:.0f}s)", flush=True)
        m = regressor(a.trees, a.jobs).fit(r_tr.reshape(-1, 1), ytr, sample_weight=wtr)
        preds["R_realized_only"] = m.predict(r_te.reshape(-1, 1))
        cell["R_realized_outcome_alone"] = score(yte, preds["R_realized_only"], gte)
        # the decomposition the question actually asks for: by game phase, per layer
        minute = te["minute"][m_te]
        cell["by_game_time"] = {}
        for lo, hi in TIME_BANDS:
            bm = (minute >= lo) & (minute < hi)
            if bm.sum() < 200:
                continue
            band = {"n": int(bm.sum()), "matches": int(len(set(gte[bm].tolist()))),
                    "target_sd_pp": float(yte[bm].std()), "target_mean_pp": float(yte[bm].mean()),
                    "mean_abs_target_pp": float(np.abs(yte[bm]).mean())}
            for key, pv in preds.items():
                band[key] = score(yte[bm], pv[bm], gte[bm])
            band["C_minus_A"] = paired_delta(yte[bm], preds["A_X_only"], preds["C_plus_realized"], gte[bm])
            band["B_minus_A"] = paired_delta(yte[bm], preds["A_X_only"], preds["B_plus_predicted"], gte[bm])
            cell["by_game_time"][f"{lo}-{hi}"] = band
            print(f"[downstream] {hname:9s} band {lo}-{hi:<4d} n={bm.sum():>6d} "
                  f"A r2 {band['A_X_only']['r2']:+.4f} | C r2 {band['C_plus_realized']['r2']:+.4f} "
                  f"| R-alone r2 {band['R_realized_only']['r2']:+.4f}", flush=True)
        cell["paired"] = {
            "B_minus_A": paired_delta(yte, preds["A_X_only"], preds["B_plus_predicted"], gte),
            "C_minus_A": paired_delta(yte, preds["A_X_only"], preds["C_plus_realized"], gte),
            "C_minus_B": paired_delta(yte, preds["B_plus_predicted"], preds["C_plus_realized"], gte)}
        out["horizons"][hname] = cell
        a.out.write_text(json.dumps(out, indent=2), encoding="utf-8")

    out["elapsed_seconds"] = round(time.time() - started, 2)
    a.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({h: {k: round(v["spearman"], 4) for k, v in c.items()
                          if isinstance(v, dict) and "spearman" in v}
                      for h, c in out["horizons"].items()}, indent=2), flush=True)


if __name__ == "__main__":
    main()
