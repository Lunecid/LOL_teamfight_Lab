"""What information does the vision modality actually carry?

Three questions, answered on the joined engagements:

1. Is vision a proxy for telemetry?  Rank-correlate the two frozen
   out-of-fold prediction vectors, and rank-correlate the one quantity both
   modalities measure directly -- the team HP-sum difference -- as vision
   reads it at the prediction cutoff versus as telemetry reads it at the
   last minute boundary.
2. Which channel carries the signal?  Refit the vision model restricted to
   the HUD panel block (per-player HP / alive / resource) and to the minimap
   block (spread, component count, mass, team distance, closing speed,
   distance to the fight), under the same folds and grid as the main
   experiment, and score them inside the |goldDiff| strata.
3. How stale is telemetry at the cutoff?  Report the distribution of
   time since the last minute boundary.

Sub-channel AUCs come from refits, so their stratum split inherits the
time/gold entanglement documented by run_time_confound_check.py; the
comparison of interest is the *pattern* across strata, not the absolute
level.

Example:
    python scripts/run_vision_ablation.py ^
        --preds D:/LOL_Project/fusion_2615/features/stratified_results_v2_fixed.preds.npz ^
        --telemetry D:/LOL_Project/fusion_2615/features/telemetry_baseline_v2.npz ^
        --vision D:/LOL_Project/fusion_2615/features/engagement_features_v2.jsonl ^
        --output D:/LOL_Project/fusion_2615/features/vision_ablation.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROLES = ("TOP", "JNG", "MID", "BOT", "SUP")
# minimap-derived feature name fragments; everything else in the row is HUD panel
MINIMAP_TOKENS = ("spread", "components", "mass", "dist_to_fight",
                  "team_distance", "closing_speed")


def load_fusion_module():
    spec = importlib.util.spec_from_file_location(
        "rfe", PROJECT_ROOT / "scripts" / "run_fusion_experiment.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def single_auc(y: np.ndarray, values: np.ndarray) -> float:
    ok = ~np.isnan(values)
    ranks = rankdata(values[ok])
    pos = y[ok] == 1
    n1, n0 = int(pos.sum()), int((~pos).sum())
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preds", required=True, type=Path)
    parser.add_argument("--telemetry", required=True, type=Path)
    parser.add_argument("--vision", required=True, type=Path)
    parser.add_argument("--window", default="w30")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)

    rfe = load_fusion_module()
    blob = np.load(args.preds, allow_pickle=True)
    y, gold, groups = blob["y"], blob["gold"], blob["groups"]
    pred_tel, pred_vis = blob["pred_telemetry"], blob["pred_vision"]

    meta = json.loads(args.telemetry.with_suffix(".keys.json").read_text(encoding="utf-8"))
    stored = meta["feature_names"]
    telemetry_X = np.load(args.telemetry)["X"]
    vision_rows = {
        (r["match_id"], int(round(r["prediction_cutoff_s"] * 1000))): r
        for r in (json.loads(line) for line in
                  args.vision.read_text(encoding="utf-8").splitlines() if line.strip())
    }
    selected = [(i, vision_rows[(k["match_id"], k["engage_ts_ms"])])
                for i, k in enumerate(meta["keys"])
                if (k["match_id"], k["engage_ts_ms"]) in vision_rows]
    idx = np.array([i for i, _ in selected])
    rows = [row for _, row in selected]
    if len(idx) != len(y):
        raise SystemExit(f"join mismatch: {len(idx)} rows vs {len(y)} predictions")

    seconds = np.array([meta["keys"][i]["engage_ts_ms"] / 1000.0 for i in idx])
    staleness = seconds % 60.0

    def telemetry_column(name: str) -> np.ndarray:
        return telemetry_X[idx, stored.index(name)]

    hp_vision = np.array([
        row["features"][args.window].get("diff_hp_sum_last", np.nan) or np.nan
        for row in rows
    ], dtype=float)
    hp_telemetry = (sum(telemetry_column(f"b{r}_hp_pct__last") for r in ROLES)
                    - sum(telemetry_column(f"r{r}_hp_pct__last") for r in ROLES))
    both = ~np.isnan(hp_vision)

    results: dict[str, object] = {
        "n": int(len(y)),
        "telemetry_staleness_seconds": {
            "median": float(np.median(staleness)),
            "p10": float(np.percentile(staleness, 10)),
            "p90": float(np.percentile(staleness, 90)),
        },
        "prediction_rank_correlation": float(spearmanr(pred_tel, pred_vis).statistic),
        "team_hp_sum_difference": {
            "rank_correlation_vision_vs_telemetry": float(
                spearmanr(hp_vision[both], hp_telemetry[both]).statistic
            ),
            "vision_auc": single_auc(y, hp_vision),
            "telemetry_auc": single_auc(y, hp_telemetry),
        },
    }
    print(f"n={len(y)} | telemetry staleness at cutoff:"
          f" median={results['telemetry_staleness_seconds']['median']:.1f}s")
    print(f"rank corr(telemetry pred, vision pred) = "
          f"{results['prediction_rank_correlation']:+.3f}")
    hp = results["team_hp_sum_difference"]
    print(f"team HP-sum difference: rank corr(vision@cutoff, telemetry@boundary) = "
          f"{hp['rank_correlation_vision_vs_telemetry']:.3f}"
          f" | AUC vision={hp['vision_auc']:.3f} telemetry={hp['telemetry_auc']:.3f}")

    X, names = rfe.vision_matrix(rows, args.window)
    minimap = [j for j, n in enumerate(names) if any(t in n for t in MINIMAP_TOKENS)]
    panel = [j for j, n in enumerate(names) if j not in minimap and n != "frames_used"]
    blocks = {
        "panel_hud": panel,
        "minimap_formation": minimap,
        "hp_only": [j for j, n in enumerate(names) if "hp" in n],
        "alive_only": [j for j, n in enumerate(names) if "alive" in n],
    }
    imbalance = np.abs(gold)
    cut = float(np.median(imbalance))
    close, one_sided = imbalance <= cut, imbalance > cut

    def scored(pred) -> dict:
        return {
            "overall": float(roc_auc_score(y, pred)),
            "close": float(roc_auc_score(y[close], pred[close])),
            "one_sided": float(roc_auc_score(y[one_sided], pred[one_sided])),
        }

    table = {"telemetry": scored(pred_tel), "vision_all": scored(pred_vis)}
    for name, columns in blocks.items():
        table[name] = scored(rfe.oof_predictions(X[:, columns], y, groups))
        table[name]["n_features"] = len(columns)
    results["ablation"] = table
    print(f"{'representation':22s} {'overall':>8s} {'close':>8s} {'one-sided':>10s}")
    for name, entry in table.items():
        print(f"{name:22s} {entry['overall']:8.3f} {entry['close']:8.3f}"
              f" {entry['one_sided']:10.3f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("wrote", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
