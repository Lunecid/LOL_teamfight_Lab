"""Does a predicted engagement OUTCOME improve match-outcome prediction?  The market_event re-run.

scripts/train_engagement_predictor_v3.py answered "does a predicted engagement help predict the final
winner?" with an A / B / C stack, but the column it added in B was the |delta-V|-weighted
sign(delta-V) predictor (test AUC 0.583), not the market_event outcome predictor the paper reports
(test AUC 0.665).  Every sentence that quotes B - A for "the engagement predictor" was therefore about
the weaker model.  This script re-runs B with the market_event predictor and changes nothing else:

  rows      predict_train / predict_test from load_split (same X_cutoff, 7,106 columns)
  learner   train.state_value_experiment.learner(seed=7, trees) with trees and folds read from the
            published results.json
  weights   match weights
  CI        train.state_value_experiment.paired_bootstrap (match-clustered, 300 draws, seed 7)

Layers (all predict the final match winner on predict_test):
  A                  X only                                              (must reproduce the published A)
  B_deltaV           X + the published deltaV-weighted predictor         (must reproduce the published B)
  B_market           X + market_event predictor: out-of-fold on ALL predict_train rows (models fit on the
                     labelled rows of the training folds, draws predicted too), full-train model on test
  B_market_nan       X + the stored labelled-row OOF (outputs/engagement_predictor_v3/
                     market_oof_predict_train.npz), draws left NaN, same test predictions
  C_market           X + the REALISED market_event label (draws NaN)     retrospective ceiling

The market test predictions must equal the stored test_p_market (same learner, rows and weights).
Reproduction checks are recorded, not asserted, so a mismatch is visible in the output.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from train.state_value_experiment import assert_disjoint, learner, match_weights, metrics, paired_bootstrap
from scripts.train_engagement_predictor_v3 import load_split


def git_state():
    def run(*a):
        return subprocess.run(["git", "-C", str(ROOT), *a], capture_output=True, text=True).stdout.strip()
    return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain", "--", "scripts", "train"))}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--v3-dir", type=Path, default=ROOT / "outputs/temporal_winprob_v3_buckets")
    ap.add_argument("--dataset", type=Path, default=ROOT / "outputs/state_value_main_50k")
    ap.add_argument("--pred-dir", type=Path, default=ROOT / "outputs/engagement_predictor_v3")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    started = time.time()

    published = json.loads((a.pred_dir / "results.json").read_text(encoding="utf-8"))
    trees, folds = int(published["learner"]["trees"]), int(published["learner"]["folds"])
    tr = load_split(a.dataset, a.v3_dir / "engagement_changes_predict_train.npz")
    te = load_split(a.dataset, a.v3_dir / "engagement_changes.npz")
    assert_disjoint(tr["match"], te["match"])
    print(f"[stack] train {len(tr['id'])} / test {len(te['id'])} rows, trees {trees}, folds {folds} "
          f"({time.time() - started:.0f}s)", flush=True)

    with np.load(a.pred_dir / "predictions.npz", allow_pickle=False) as z:
        stored = {k: z[k] for k in z.files}
    if not (np.array_equal(stored["train_id"].astype(str), tr["id"]) and np.array_equal(stored["test_id"].astype(str), te["id"])):
        raise SystemExit("row order differs from the published predictions.npz; refusing to pair")

    # ---- market_event predictor: OOF over every train row, full model for test ---------------
    lab_tr, lab_te = tr["market"] >= 0, te["market"] >= 0
    oof_all = np.full(len(tr["id"]), np.nan)
    fold_auc = []
    for k, (f_tr, f_va) in enumerate(GroupKFold(folds).split(tr["X"], tr["market"], tr["match"]), 1):
        assert_disjoint(tr["match"][f_tr], tr["match"][f_va])
        fit_rows = f_tr[lab_tr[f_tr]]
        m = learner(trees=trees).fit(tr["X"][fit_rows], tr["market"][fit_rows], sample_weight=match_weights(tr["match"][fit_rows]))
        oof_all[f_va] = m.predict_proba(tr["X"][f_va])[:, 1]
        va_lab = f_va[lab_tr[f_va]]
        fold_auc.append(float(roc_auc_score(tr["market"][va_lab], oof_all[va_lab])))
        print(f"[stack] market OOF fold {k}/{folds} auc {fold_auc[-1]:.4f} ({time.time() - started:.0f}s)", flush=True)
    full = learner(trees=trees).fit(tr["X"][lab_tr], tr["market"][lab_tr], sample_weight=match_weights(tr["match"][lab_tr]))
    p_market = full.predict_proba(te["X"])[:, 1]
    reproduction = {
        "test_p_market_max_abs_diff": float(np.max(np.abs(p_market - stored["test_p_market"]))),
        "market_test_auc_now": metrics(te["market"][lab_te], p_market[lab_te], te["match"][lab_te])["auc"],
        "market_test_auc_published": published["market_event"]["test"]["auc"],
        "market_oof_train_auc_labelled": metrics(tr["market"][lab_tr], oof_all[lab_tr], tr["match"][lab_tr])["auc"],
        "market_oof_fold_auc": fold_auc,
    }

    stored_oof = np.load(a.pred_dir / "market_oof_predict_train.npz")["oof"]
    oof_nan = np.full(len(tr["id"]), np.nan)
    if len(stored_oof) == int(lab_tr.sum()):
        oof_nan[lab_tr] = stored_oof
    else:
        raise SystemExit(f"stored market OOF has {len(stored_oof)} rows, expected {int(lab_tr.sum())}")
    market_real_tr = np.where(lab_tr, tr["market"], np.nan).astype(float)
    market_real_te = np.where(lab_te, te["market"], np.nan).astype(float)

    layers = (("A", None, None),
              ("B_deltaV", stored["train_oof"], stored["test_p_deltaV_weighted"]),
              ("B_market", oof_all, p_market),
              ("B_market_nan", oof_nan, p_market),
              ("C_market", market_real_tr, market_real_te))
    preds, out = {}, {"what": __doc__.split("\n\n")[0], "trees": trees, "folds": folds,
                      "rows": {"train": int(len(tr["id"])), "test": int(len(te["id"])),
                               "test_matches": int(len(set(te["match"].tolist()))),
                               "train_market_labelled": int(lab_tr.sum()), "test_market_labelled": int(lab_te.sum())},
                      "reproduction": reproduction, "layers": {}, "paired_vs_A": {}}
    for name, add_tr, add_te in layers:
        Xa = tr["X"] if add_tr is None else np.column_stack([tr["X"], add_tr])
        Xb = te["X"] if add_te is None else np.column_stack([te["X"], add_te])
        m = learner(trees=trees).fit(Xa, tr["winner"], sample_weight=match_weights(tr["match"]))
        preds[name] = m.predict_proba(Xb)[:, 1]
        met = metrics(te["winner"], preds[name], te["match"])
        out["layers"][name] = {k: met[k] for k in ("n", "matches", "auc", "brier", "log_loss")}
        print(f"[stack] {name:13s} match-outcome AUC {met['auc']:.6f} ({time.time() - started:.0f}s)", flush=True)
    for name in ("B_deltaV", "B_market", "B_market_nan", "C_market"):
        out["paired_vs_A"][name] = paired_bootstrap(te["winner"], preds["A"], preds[name], te["match"])
    reproduction["A_auc_now"] = out["layers"]["A"]["auc"]
    reproduction["A_auc_published"] = published["A_pre_information"]["auc"]
    reproduction["B_deltaV_auc_now"] = out["layers"]["B_deltaV"]["auc"]
    reproduction["B_deltaV_auc_published"] = published["B_plus_predicted_engagement"]["auc"]
    reproduction["A_reproduced"] = abs(reproduction["A_auc_now"] - reproduction["A_auc_published"]) < 1e-9
    reproduction["B_deltaV_reproduced"] = abs(reproduction["B_deltaV_auc_now"] - reproduction["B_deltaV_auc_published"]) < 1e-9
    reproduction["test_p_market_reproduced"] = reproduction["test_p_market_max_abs_diff"] < 1e-9

    out["provenance"] = {**git_state(), "python": sys.version.split()[0], "elapsed_s": round(time.time() - started, 1),
                         "inputs": {"v3_dir": str(a.v3_dir), "dataset": str(a.dataset), "pred_dir": str(a.pred_dir)}}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    np.savez_compressed(a.out.with_suffix(".preds.npz"), test_id=te["id"], test_match=te["match"], test_winner=te["winner"],
                        train_market_oof_all=oof_all, test_p_market=p_market, **preds)
    print(json.dumps({"reproduction": reproduction,
                      "auc": {k: round(v["auc"], 6) for k, v in out["layers"].items()},
                      "B_market_minus_A_auc_ci95": out["paired_vs_A"]["B_market"]["auc_improvement_ci95"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
