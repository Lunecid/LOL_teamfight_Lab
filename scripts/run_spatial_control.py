"""Is the vision advantage staleness, or representation?

The minimap block carries the close-stratum advantage, but telemetry also
knows where everyone is -- once a minute.  This control gives telemetry
the *same summary statistics* the minimap block uses (team centroid, RMS
spread, cluster count, inter-team distance, distance to the fight
location), computed from the participantFrames positions at the last
minute boundary at or before the prediction cutoff, and asks whether the
gap closes.

    gap closes  -> vision's edge was recency: the same representation,
                   measured at the right moment
    gap remains -> vision's edge is the measurement itself, not its age

Also rank-correlates each telemetry-derived statistic against the minimap
statistic it mirrors, which separates the two explanations directly.

Example:
    python scripts/run_spatial_control.py ^
        --preds D:/LOL_Project/fusion_2615/features/stratified_results_v2_fixed.preds.npz ^
        --telemetry D:/LOL_Project/fusion_2615/features/telemetry_baseline_v2.npz ^
        --vision D:/LOL_Project/fusion_2615/features/engagement_features_v2.jsonl ^
        --timelines D:/LOL_Project/fusion_2615/raw/kr/timeline ^
        --windows D:/LOL_Project/fusion_2615/vision_windows ^
        --output D:/LOL_Project/fusion_2615/features/spatial_control.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUFFIXES = ("last", "mean", "std", "min", "max", "delta", "slope")
BLUE = tuple(str(i) for i in range(1, 6))
RED = tuple(str(i) for i in range(6, 11))
# single-linkage threshold for "these players are in the same group", in world
# units; the map is 15,000 across, so this is roughly a screen's width.
CLUSTER_LINK_UNITS = 1500.0


def load_fusion_module():
    spec = importlib.util.spec_from_file_location(
        "rfe", PROJECT_ROOT / "scripts" / "run_fusion_experiment.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def resolve_names(stored_names: list[str]) -> tuple[list[str], list[str]]:
    k = len(stored_names) // len(SUFFIXES)
    if {n.rsplit("__", 1)[1] for n in stored_names[:k]} == {SUFFIXES[0]}:
        return list(stored_names), [n.rsplit("__", 1)[0] for n in stored_names[:k]]
    base = [n.rsplit("__", 1)[0] for n in stored_names[:: len(SUFFIXES)]]
    return [f"{n}__{s}" for s in SUFFIXES for n in base], base


def cluster_count(points: np.ndarray, link: float = CLUSTER_LINK_UNITS) -> int:
    """Single-linkage connected components under a distance threshold."""
    n = len(points)
    if n == 0:
        return 0
    seen, groups = set(), 0
    for start in range(n):
        if start in seen:
            continue
        groups += 1
        stack = [start]
        seen.add(start)
        while stack:
            current = stack.pop()
            for other in range(n):
                if other in seen:
                    continue
                if np.hypot(*(points[current] - points[other])) <= link:
                    seen.add(other)
                    stack.append(other)
    return groups


def team_stats(points: np.ndarray, fight: np.ndarray, prefix: str) -> dict:
    out: dict[str, float] = {}
    if len(points) == 0:
        return {f"{prefix}_{k}": np.nan
                for k in ("centroid_x", "centroid_y", "spread", "components",
                          "alive", "dist_to_fight")}
    centroid = points.mean(axis=0)
    out[f"{prefix}_centroid_x"], out[f"{prefix}_centroid_y"] = centroid
    out[f"{prefix}_spread"] = float(np.sqrt(((points - centroid) ** 2).sum(axis=1).mean()))
    out[f"{prefix}_components"] = float(cluster_count(points))
    out[f"{prefix}_alive"] = float(len(points))
    out[f"{prefix}_dist_to_fight"] = float(np.hypot(*(centroid - fight)))
    return out


def boundary_formation(frames: list[dict], cutoff_s: float, fight: np.ndarray) -> dict | None:
    """Formation summary from the last minute boundary at or before the cutoff."""
    index = int(cutoff_s // 60)
    if index >= len(frames):
        return None
    participants = frames[index].get("participantFrames", {})
    teams = {}
    for name, ids in (("blue", BLUE), ("red", RED)):
        points = []
        for pid in ids:
            row = participants.get(pid)
            if not row:
                continue
            position = row.get("position")
            health = row.get("championStats", {}).get("health")
            if position is None or (health is not None and health <= 0):
                continue
            points.append([float(position["x"]), float(position["y"])])
        teams[name] = np.array(points, dtype=float).reshape(-1, 2)
    out = {}
    out.update(team_stats(teams["blue"], fight, "tel_blue"))
    out.update(team_stats(teams["red"], fight, "tel_red"))
    if len(teams["blue"]) and len(teams["red"]):
        out["tel_team_distance"] = float(np.hypot(
            *(teams["blue"].mean(axis=0) - teams["red"].mean(axis=0))))
    else:
        out["tel_team_distance"] = np.nan
    for key in ("spread", "components", "alive", "dist_to_fight"):
        out[f"tel_diff_{key}"] = out[f"tel_blue_{key}"] - out[f"tel_red_{key}"]
    return out


def drop_non_causal(names: list[str], matrix: np.ndarray) -> tuple[list[str], np.ndarray]:
    """Distance-to-fight reads the first-kill position, which is after the cutoff."""
    keep = [j for j, n in enumerate(names) if "dist_to_fight" not in n]
    return [names[j] for j in keep], matrix[:, keep]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preds", required=True, type=Path)
    parser.add_argument("--telemetry", required=True, type=Path)
    parser.add_argument("--vision", required=True, type=Path)
    parser.add_argument("--timelines", required=True, type=Path)
    parser.add_argument("--windows", required=True, type=Path)
    parser.add_argument("--window", default="w30")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)

    rfe = load_fusion_module()
    blob = np.load(args.preds, allow_pickle=True)
    y, gold, groups = blob["y"], blob["gold"], blob["groups"]
    pred_tel, pred_vis = blob["pred_telemetry"], blob["pred_vision"]

    meta = json.loads(args.telemetry.with_suffix(".keys.json").read_text(encoding="utf-8"))
    names, base = resolve_names(meta["feature_names"])
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

    # fight centroids come from the capture manifests (world coordinates)
    centroids: dict[tuple[str, int], tuple[float, float]] = {}
    for path in args.windows.glob("*/engagement_*/window.json"):
        window = json.loads(path.read_text(encoding="utf-8"))
        key = (window["match_id"], int(round(window["prediction_cutoff_s"] * 1000)))
        centroids[key] = tuple(window["centroid"])

    timelines: dict[str, list[dict]] = {}
    spatial_rows, missing = [], 0
    for position, i in enumerate(idx):
        key = (meta["keys"][i]["match_id"], meta["keys"][i]["engage_ts_ms"])
        match_id = key[0]
        if match_id not in timelines:
            path = args.timelines / f"{match_id}.json"
            timelines[match_id] = (
                json.loads(path.read_text(encoding="utf-8"))["info"]["frames"]
                if path.exists() else []
            )
        fight = np.array(centroids.get(key, (7500.0, 7500.0)), dtype=float)
        stats = boundary_formation(timelines[match_id], key[1] / 1000.0, fight)
        if stats is None:
            stats = {}
            missing += 1
        spatial_rows.append(stats)

    feature_names = sorted({k for r in spatial_rows for k in r})
    X_spatial = np.full((len(spatial_rows), len(feature_names)), np.nan)
    for i, row in enumerate(spatial_rows):
        for j, name in enumerate(feature_names):
            value = row.get(name)
            if value is not None:
                X_spatial[i, j] = value
    print(f"telemetry spatial features: {len(feature_names)} | rows missing a boundary: {missing}")

    # does telemetry's boundary geometry measure what the minimap measures?
    # keep the non-causal columns here: the correlations are a measurement
    # diagnostic, not a model input
    X_vis, vision_names = rfe.vision_matrix(rows, args.window, include_non_causal=True)
    pairs = {
        "team_distance": ("tel_team_distance", "team_distance_last"),
        "blue_spread": ("tel_blue_spread", "blue_spread_last"),
        "red_spread": ("tel_red_spread", "red_spread_last"),
        "blue_dist_to_fight": ("tel_blue_dist_to_fight", "blue_dist_to_fight_last"),
        "red_dist_to_fight": ("tel_red_dist_to_fight", "red_dist_to_fight_last"),
        "blue_components": ("tel_blue_components", "blue_components_last"),
    }
    correlations = {}
    for label, (tel_name, vis_name) in pairs.items():
        if tel_name not in feature_names or vis_name not in vision_names:
            continue
        a = X_spatial[:, feature_names.index(tel_name)]
        b = X_vis[:, vision_names.index(vis_name)]
        ok = ~np.isnan(a) & ~np.isnan(b)
        correlations[label] = float(spearmanr(a[ok], b[ok]).statistic)
    print("rank corr(telemetry boundary geometry, minimap geometry at cutoff):")
    for label, value in correlations.items():
        print(f"  {label:20s} {value:+.3f}")

    macro = [b for b in base if ("Diff" in b) or b.startswith("dist_") or b == "time_norm"]
    compact = [names.index(f"{b}__{s}") for b in macro for s in ("last", "delta", "slope")]
    X_tel = telemetry_X[idx][:, compact]
    causal_names, X_vis_causal = drop_non_causal(vision_names, X_vis)
    spatial_names, X_spatial_causal = drop_non_causal(feature_names, X_spatial)
    minimap_tokens = ("spread", "components", "mass", "team_distance", "closing_speed")
    minimap = [j for j, n in enumerate(causal_names) if any(t in n for t in minimap_tokens)]

    imbalance = np.abs(gold)
    cut = float(np.median(imbalance))
    close, one_sided = imbalance <= cut, imbalance > cut

    def scored(pred) -> dict:
        return {
            "overall": float(roc_auc_score(y, pred)),
            "close": float(roc_auc_score(y[close], pred[close])),
            "one_sided": float(roc_auc_score(y[one_sided], pred[one_sided])),
        }

    table = {"telemetry_compact": scored(pred_tel)}
    for label, matrix in (
        ("vision_all_causal", X_vis_causal),
        ("telemetry_spatial_only_causal", X_spatial_causal),
        ("telemetry_compact_plus_spatial_causal", np.hstack([X_tel, X_spatial_causal])),
        ("vision_minimap_only_causal", X_vis_causal[:, minimap]),
    ):
        table[label] = scored(rfe.oof_predictions(matrix, y, groups))
        table[label]["n_features"] = matrix.shape[1]

    print(f"{'representation':34s} {'overall':>8s} {'close':>8s} {'one-sided':>10s}")
    for label, entry in table.items():
        print(f"{label:34s} {entry['overall']:8.3f} {entry['close']:8.3f}"
              f" {entry['one_sided']:10.3f}")

    results = {
        "n": int(len(y)),
        "rows_missing_boundary": missing,
        "spatial_feature_names": feature_names,
        "geometry_rank_correlations": correlations,
        "ablation": table,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("wrote", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
