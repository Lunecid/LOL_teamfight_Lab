"""Pilot: does evidence-corrected state at the cutoff improve engagement-outcome prediction?

Three models, same learner (LightGBM 250 / 15 leaves / lr .04 / seed 7), same splits, same label
(market_event, ties dropped), match weights only:

  X            the frozen 7,106 pre-cutoff features (held frames + aggregations)
  X + hold     X plus the 306-column state block read from the last frame - a control for
               'just adding these columns'
  X + evidence X plus the same block reconstructed from events (levels, items, deaths, buffs,
               causal extrapolation, calibrated position fusion)

The gap between the last two is the value of the evidence.  A block-only model shows how much
signal the block carries on its own.  Differences are given a match-level paired bootstrap and
a per-phase breakdown, since the early game is where held positions were the deciding feature.
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

TIME_BANDS = ((2, 10), (10, 20), (20, 30), (30, 1000))


def join_block(split, block_path):
    with np.load(block_path, allow_pickle=False) as z:
        ids = z["id"].astype(str)
        order = {e: i for i, e in enumerate(ids)}
        ev, hold, names = z["evidence"], z["hold"], z["names"].astype(str).tolist()
    rows = np.array([order.get(e, -1) for e in split["id"]])
    keep = rows >= 0
    return keep, ev[rows[keep]], hold[rows[keep]], names


def top_features(model, names, k=15):
    gain = model.booster_.feature_importance(importance_type="gain")
    order = np.argsort(gain)[::-1][:k]
    return [{"feature": names[i], "gain_share": float(gain[i] / gain.sum())} for i in order]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--v3-dir", type=Path, default=ROOT / "outputs/temporal_winprob_v3_buckets")
    ap.add_argument("--dataset", type=Path, default=ROOT / "outputs/state_value_main_50k")
    ap.add_argument("--block-dir", type=Path, default=ROOT / "outputs/evidence_state")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs/evidence_state/pilot_results.json")
    ap.add_argument("--trees", type=int, default=250)
    a = ap.parse_args()
    started = time.time()
    x_names = json.loads((a.dataset / "schema.json").read_text(encoding="utf-8"))["input_names"]

    tr = load_split(a.dataset, a.v3_dir / "engagement_changes_predict_train.npz")
    te = load_split(a.dataset, a.v3_dir / "engagement_changes.npz")
    ktr, btr_ev, btr_hold, bnames = join_block(tr, a.block_dir / "block_predict_train.npz")
    kte, bte_ev, bte_hold, _ = join_block(te, a.block_dir / "block_predict_test.npz")
    ktr &= tr["market"] >= 0
    kte &= te["market"] >= 0
    # the block rows were selected with keep masks in join_block; re-align to the label mask
    def align(keep_all, keep_lab, block):
        idx = np.flatnonzero(keep_all)
        return block[np.isin(idx, np.flatnonzero(keep_lab))]
    keep_tr_all, _, _, _ = join_block(tr, a.block_dir / "block_predict_train.npz")
    keep_te_all, _, _, _ = join_block(te, a.block_dir / "block_predict_test.npz")
    btr_ev, btr_hold = align(keep_tr_all, ktr, btr_ev), align(keep_tr_all, ktr, btr_hold)
    bte_ev, bte_hold = align(keep_te_all, kte, bte_ev), align(keep_te_all, kte, bte_hold)
    Xtr, ytr, gtr = tr["X"][ktr], tr["market"][ktr], tr["match"][ktr]
    Xte, yte, gte, minute = te["X"][kte], te["market"][kte], te["match"][kte], te["minute"][kte]
    assert len(btr_ev) == len(Xtr) and len(bte_ev) == len(Xte)
    wtr = match_weights(gtr)
    print(f"[pilot] train {len(ytr)} / test {len(yte)} labelled rows, block {btr_ev.shape[1]} cols ({time.time()-started:.0f}s)", flush=True)

    variants = {
        "X": (Xtr, Xte, x_names),
        "X_plus_hold": (np.hstack([Xtr, btr_hold]), np.hstack([Xte, bte_hold]), x_names + [f"hold_{n}" for n in bnames]),
        "X_plus_evidence": (np.hstack([Xtr, btr_ev]), np.hstack([Xte, bte_ev]), x_names + [f"ev_{n}" for n in bnames]),
        "evidence_block_only": (btr_ev, bte_ev, [f"ev_{n}" for n in bnames]),
        "hold_block_only": (btr_hold, bte_hold, [f"hold_{n}" for n in bnames]),
    }
    preds, out = {}, {"label": "market_event (ties dropped)", "rows": {"train": int(len(ytr)), "test": int(len(yte))},
                      "block_columns": int(btr_ev.shape[1]), "variants": {}}
    for name, (A, B, names) in variants.items():
        m = learner(trees=a.trees).fit(A, ytr, sample_weight=wtr)
        p = m.predict_proba(B)[:, 1]
        preds[name] = p
        cell = {"test": metrics(yte, p, gte), "by_game_time": {}, "top_features": top_features(m, names)}
        for lo, hi in TIME_BANDS:
            mask = (minute >= lo) & (minute < hi)
            if mask.sum() > 50:
                cell["by_game_time"][f"{lo}-{hi}"] = metrics(yte[mask], p[mask], gte[mask])
        out["variants"][name] = cell
        joblib.dump(m, a.out.parent / f"pilot_model_{name}.joblib")
        print(f"[pilot] {name:20s} AUC {cell['test']['auc']:.4f}  " +
              "  ".join(f"{b}:{c['auc']:.3f}" for b, c in cell["by_game_time"].items()) +
              f"  ({time.time()-started:.0f}s)", flush=True)
    out["paired"] = {
        "evidence_minus_hold": paired_bootstrap(yte, preds["X_plus_hold"], preds["X_plus_evidence"], gte),
        "evidence_minus_X": paired_bootstrap(yte, preds["X"], preds["X_plus_evidence"], gte),
        "hold_minus_X": paired_bootstrap(yte, preds["X"], preds["X_plus_hold"], gte),
        "block_only_evidence_minus_hold": paired_bootstrap(yte, preds["hold_block_only"], preds["evidence_block_only"], gte)}
    for lo, hi in TIME_BANDS:
        mask = (minute >= lo) & (minute < hi)
        if mask.sum() > 50:
            out["paired"][f"evidence_minus_hold_{lo}-{hi}"] = paired_bootstrap(
                yte[mask], preds["X_plus_hold"][mask], preds["X_plus_evidence"][mask], gte[mask])
    ev_used = [f for f in out["variants"]["X_plus_evidence"]["top_features"] if f["feature"].startswith("ev_")]
    out["evidence_columns_in_top15_of_augmented_model"] = ev_used
    out["elapsed_seconds"] = round(time.time() - started, 2)
    a.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    np.savez_compressed(a.out.parent / "pilot_predictions.npz", id=te["id"][kte], match=gte, y=yte, minute=minute,
                        **{f"p_{k}": v for k, v in preds.items()})
    print(json.dumps({k: {"auc_ci95": v["auc_improvement_ci95"]} for k, v in out["paired"].items()}, indent=2), flush=True)


if __name__ == "__main__":
    main()
