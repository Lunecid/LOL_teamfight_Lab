"""Telemetry vs telemetry+vision fusion experiment on shared engagements.

Joins the released pipeline's tabular baseline (X, y, keys) with the
formation-structure features extracted from captured pre-fight replay
windows, then compares LightGBM models under match-grouped 5-fold CV:

    telemetry-only   (the CoG representation, recomputed on this corpus)
    vision-only      (panel + minimap formation features)
    telemetry+vision (concatenation)

for each observation window W in {10, 20, 30} s.  Reported effect is the
paired out-of-fold AUC difference with a match-cluster bootstrap CI; the
same folds and rows are used for every model, so modality is the only
variable.

Example:
    python scripts/run_fusion_experiment.py ^
        --telemetry D:/LOL_Project/fusion_2615/features/telemetry_baseline.npz ^
        --vision D:/LOL_Project/fusion_2615/features/engagement_features.jsonl ^
        --output D:/LOL_Project/fusion_2615/features/fusion_results.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

WINDOWS = ("w10", "w20", "w30")
SEED = 7


def load_vision(path: Path) -> dict[tuple[str, int], dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {
        (row["match_id"], int(round(row["prediction_cutoff_s"] * 1000))): row
        for row in rows
    }


def vision_matrix(rows: list[dict], window: str) -> tuple[np.ndarray, list[str]]:
    names = sorted({k for row in rows for k in row["features"][window]})
    out = np.full((len(rows), len(names)), np.nan)
    for i, row in enumerate(rows):
        feats = row["features"][window]
        for j, name in enumerate(names):
            value = feats.get(name)
            if value is not None:
                out[i, j] = float(value)
    return out, names


# One shared regularization grid for every modality: at ~420 training rows a
# 7,105-D representation memorizes noise with light regularization, so each
# outer fold selects its config by inner grouped CV -- identically for
# telemetry, vision, and fusion, keeping the comparison fair.
PARAM_GRID = (
    {"n_estimators": 100, "num_leaves": 7, "max_depth": 3,
     "colsample_bytree": 0.2, "min_child_samples": 30},
    {"n_estimators": 200, "num_leaves": 15, "max_depth": 4,
     "colsample_bytree": 0.5, "min_child_samples": 20},
    {"n_estimators": 400, "num_leaves": 31, "max_depth": -1,
     "colsample_bytree": 0.9, "min_child_samples": 20},
)


def _fit(params, X, y):
    model = LGBMClassifier(
        learning_rate=0.05, subsample=0.9, random_state=SEED, verbose=-1, **params
    )
    model.fit(X, y)
    return model


def oof_predictions(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    oof = np.full(len(y), np.nan)
    for train, test in GroupKFold(n_splits=5).split(X, y, groups):
        best_params, best_auc = None, -1.0
        for params in PARAM_GRID:
            inner_oof = np.full(len(train), np.nan)
            inner = GroupKFold(n_splits=3).split(X[train], y[train], groups[train])
            for fit_idx, val_idx in inner:
                model = _fit(params, X[train][fit_idx], y[train][fit_idx])
                inner_oof[val_idx] = model.predict_proba(X[train][val_idx])[:, 1]
            auc = roc_auc_score(y[train], inner_oof)
            if auc > best_auc:
                best_params, best_auc = params, auc
        model = _fit(best_params, X[train], y[train])
        oof[test] = model.predict_proba(X[test])[:, 1]
    assert not np.isnan(oof).any()
    return oof


def cluster_bootstrap_delta(y, pred_a, pred_b, groups, n_boot=2000):
    rng = np.random.default_rng(SEED)
    unique = np.unique(groups)
    index_by_group = {g: np.where(groups == g)[0] for g in unique}
    deltas = []
    for _ in range(n_boot):
        sample = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([index_by_group[g] for g in sample])
        if len(np.unique(y[idx])) < 2:
            continue
        deltas.append(
            roc_auc_score(y[idx], pred_b[idx]) - roc_auc_score(y[idx], pred_a[idx])
        )
    deltas = np.array(deltas)
    return {
        "delta_mean": float(deltas.mean()),
        "ci_2.5": float(np.percentile(deltas, 2.5)),
        "ci_97.5": float(np.percentile(deltas, 97.5)),
        "p_leq_0": float((deltas <= 0).mean()),
        "n_boot": int(len(deltas)),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--telemetry", required=True, type=Path)
    parser.add_argument("--vision", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)

    blob = np.load(args.telemetry)
    keys = json.loads(args.telemetry.with_suffix(".keys.json").read_text(encoding="utf-8"))["keys"]
    vision = load_vision(args.vision)

    joined = [
        (i, vision[(k["match_id"], k["engage_ts_ms"])])
        for i, k in enumerate(keys)
        if (k["match_id"], k["engage_ts_ms"]) in vision
    ]
    tele_idx = np.array([i for i, _ in joined])
    vision_rows = [row for _, row in joined]
    X_tel = blob["X"][tele_idx]
    y = blob["y"][tele_idx]
    groups = np.array([keys[i]["match_id"] for i in tele_idx])
    print(
        f"telemetry engagements: {len(keys)} | vision windows: {len(vision)}"
        f" | joined: {len(joined)} | positives: {y.mean():.3f}"
        f" | matches: {len(np.unique(groups))}"
    )

    results: dict[str, object] = {
        "joined": len(joined),
        "positive_rate": float(y.mean()),
        "matches": int(len(np.unique(groups))),
    }
    pred_tel = oof_predictions(X_tel, y, groups)
    auc_tel = roc_auc_score(y, pred_tel)
    results["telemetry_auc"] = float(auc_tel)
    print(f"telemetry-only AUC: {auc_tel:.4f}")

    for window in WINDOWS:
        X_vis, names = vision_matrix(vision_rows, window)
        pred_vis = oof_predictions(X_vis, y, groups)
        X_fus = np.hstack([X_tel, X_vis])
        pred_fus = oof_predictions(X_fus, y, groups)
        # late fusion: rank-average of the two modality predictions
        from scipy.stats import rankdata

        pred_late = (rankdata(pred_tel) + rankdata(pred_vis)) / (2.0 * len(y))
        stats = cluster_bootstrap_delta(y, pred_tel, pred_fus, groups)
        stats_late = cluster_bootstrap_delta(y, pred_tel, pred_late, groups)
        results[window] = {
            "n_vision_features": len(names),
            "vision_auc": float(roc_auc_score(y, pred_vis)),
            "fusion_auc": float(roc_auc_score(y, pred_fus)),
            "late_fusion_auc": float(roc_auc_score(y, pred_late)),
            "delta_fusion_minus_telemetry": stats,
            "delta_late_fusion_minus_telemetry": stats_late,
        }
        print(
            f"{window}: vision AUC={results[window]['vision_auc']:.4f}"
            f" fusion AUC={results[window]['fusion_auc']:.4f}"
            f" dAUC={stats['delta_mean']:+.4f}"
            f" CI[{stats['ci_2.5']:+.4f},{stats['ci_97.5']:+.4f}]"
            f" p(<=0)={stats['p_leq_0']:.3f}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("wrote", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
