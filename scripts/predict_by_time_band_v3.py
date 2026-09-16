"""Does training a separate outcome predictor per game phase beat one model that sees time?

The single market_event predictor was only EVALUATED per band (2-10 min 0.630, 10-20 0.651,
20-30 0.720, 30+ 0.664).  This trains one LightGBM per cutoff-time band on predict_train and
scores it on the same band of predict_test, with a match-level paired bootstrap against the
global model on identical rows.  It also fills the transfer matrix (an early-game model scored
on late-game rows and vice versa) and lists the top gain features per band by name, which is
the material for 'what decides an engagement early versus late'.

Same learner, same splits, same label (market_event, ties dropped), match weights only.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from train.state_value_experiment import learner, match_weights, metrics, paired_bootstrap
from scripts.train_engagement_predictor_v3 import load_split

BANDS = (("2-10", 2, 10), ("10-20", 10, 20), ("20-30", 20, 30), ("30+", 30, 1000))


def band_mask(minute, lo, hi):
    return (minute >= lo) & (minute < hi)


def top_features(model, names, k=12):
    gain = model.booster_.feature_importance(importance_type="gain")
    order = np.argsort(gain)[::-1][:k]
    total = gain.sum()
    return [{"feature": names[i], "gain_share": float(gain[i] / total)} for i in order]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--v3-dir", type=Path, default=ROOT / "outputs/temporal_winprob_v3_buckets")
    ap.add_argument("--dataset", type=Path, default=ROOT / "outputs/state_value_main_50k")
    ap.add_argument("--pred-dir", type=Path, default=ROOT / "outputs/engagement_predictor_v3")
    ap.add_argument("--trees", type=int, default=250)
    a = ap.parse_args()
    started = time.time()

    names = json.loads((a.dataset / "schema.json").read_text(encoding="utf-8"))["input_names"]
    tr = load_split(a.dataset, a.v3_dir / "engagement_changes_predict_train.npz")
    te = load_split(a.dataset, a.v3_dir / "engagement_changes.npz")
    ktr, kte = tr["market"] >= 0, te["market"] >= 0
    print(f"[bands] loaded train {ktr.sum()} / test {kte.sum()} labelled rows ({time.time()-started:.0f}s)", flush=True)

    global_model = joblib.load(a.pred_dir / "engagement_predictor_market_event.joblib")
    p_global = global_model.predict_proba(te["X"])[:, 1]

    out = {"label": "market_event (ties dropped)", "learner": {"trees": a.trees, "num_leaves": 15, "lr": .04, "seed": 7},
           "bands": {}, "transfer_auc": {}, "global_top_features": top_features(global_model, names)}
    band_models, band_pred = {}, np.full(len(te["y"]), np.nan)
    for name, lo, hi in BANDS:
        mtr = ktr & band_mask(tr["minute"], lo, hi)
        mte = kte & band_mask(te["minute"], lo, hi)
        m = learner(trees=a.trees).fit(tr["X"][mtr], tr["market"][mtr], sample_weight=match_weights(tr["match"][mtr]))
        band_models[name] = m
        p_band = m.predict_proba(te["X"][mte])[:, 1]
        band_pred[mte] = p_band
        y, g = te["market"][mte], te["match"][mte]
        cell = {"train_rows": int(mtr.sum()), "train_matches": int(len(set(tr["match"][mtr].tolist()))),
                "global_model_on_band": metrics(y, p_global[mte], g),
                "band_model_on_band": metrics(y, p_band, g),
                "band_minus_global": paired_bootstrap(y, p_global[mte], p_band, g),
                "top_features": top_features(m, names)}
        cell["auc_gain"] = cell["band_model_on_band"]["auc"] - cell["global_model_on_band"]["auc"]
        out["bands"][name] = cell
        print(f"[bands] {name:6s} global {cell['global_model_on_band']['auc']:.4f}  band {cell['band_model_on_band']['auc']:.4f}  "
              f"diff {cell['auc_gain']:+.4f}  CI {cell['band_minus_global']['auc_improvement_ci95']} ({time.time()-started:.0f}s)", flush=True)

    # stitched: every test row scored by its own band's model, against the global model on all rows
    out["stitched_all_rows"] = {"global": metrics(te["market"][kte], p_global[kte], te["match"][kte]),
                                "per_band_stitched": metrics(te["market"][kte], band_pred[kte], te["match"][kte]),
                                "stitched_minus_global": paired_bootstrap(te["market"][kte], p_global[kte], band_pred[kte], te["match"][kte])}
    for src, _, _ in BANDS:
        out["transfer_auc"][src] = {}
        for dst, lo, hi in BANDS:
            mte = kte & band_mask(te["minute"], lo, hi)
            c = metrics(te["market"][mte], band_models[src].predict_proba(te["X"][mte])[:, 1], te["match"][mte])
            out["transfer_auc"][src][dst] = c["auc"]
    out["elapsed_seconds"] = round(time.time() - started, 2)
    (a.pred_dir / "by_time_band.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    for name in band_models:
        joblib.dump(band_models[name], a.pred_dir / f"engagement_predictor_market_event_band_{name.replace('+','plus')}.joblib")
    s = out["stitched_all_rows"]
    print(f"[bands] stitched {s['per_band_stitched']['auc']:.4f} vs global {s['global']['auc']:.4f}  "
          f"CI {s['stitched_minus_global']['auc_improvement_ci95']}", flush=True)


if __name__ == "__main__":
    main()
