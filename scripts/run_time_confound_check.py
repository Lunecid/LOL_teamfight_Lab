"""Is the close-gold stratum just the early game?

|goldDiff| grows as a match progresses, so a median split on resource
imbalance is partly a split on match time.  Any claim that the vision
advantage concentrates in *close* engagements has to survive conditioning
on when the engagement happened.

This reuses the frozen out-of-fold predictions written by
``run_fusion_stratified.py`` (``*.preds.npz``), so no model is refit and
the numbers are exactly the ones the stratified experiment reports.
Reports the time/gold rank correlation, both 1-D stratifications, the 2x2
cells, and the gold interaction bootstrapped *within* each time half.

Example:
    python scripts/run_time_confound_check.py ^
        --preds D:/LOL_Project/fusion_2615/features/stratified_results_v2_fixed.preds.npz ^
        --telemetry D:/LOL_Project/fusion_2615/features/telemetry_baseline_v2.npz ^
        --vision D:/LOL_Project/fusion_2615/features/engagement_features_v2.jsonl ^
        --output D:/LOL_Project/fusion_2615/features/time_confound_check.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

SEED = 7


def engage_minutes(telemetry: Path, vision: Path, n_expected: int) -> np.ndarray:
    """Engage times for the joined rows, in the join order the experiment used."""
    meta = json.loads(telemetry.with_suffix(".keys.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in vision.read_text(encoding="utf-8").splitlines() if line.strip()]
    present = {(r["match_id"], int(round(r["prediction_cutoff_s"] * 1000))) for r in rows}
    minutes = np.array([
        k["engage_ts_ms"] / 60000.0
        for k in meta["keys"]
        if (k["match_id"], k["engage_ts_ms"]) in present
    ])
    if len(minutes) != n_expected:
        raise SystemExit(f"join mismatch: {len(minutes)} times vs {n_expected} predictions")
    return minutes


def auc(y, pred, mask, min_n=40):
    if mask.sum() < min_n or len(np.unique(y[mask])) < 2:
        return None
    return float(roc_auc_score(y[mask], pred[mask]))


def bootstrap_interaction(y, pred_a, pred_b, groups, pool, masks, n_boot=2000):
    """Bootstrap (delta in mask[0]) - (delta in mask[1]) over matches inside `pool`."""
    rng = np.random.default_rng(SEED)
    unique = np.unique(groups[pool])
    index_by_group = {g: np.where((groups == g) & pool)[0] for g in unique}
    values = []
    for _ in range(n_boot):
        sample = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([index_by_group[g] for g in sample])
        deltas = []
        for mask in masks:
            sub = idx[mask[idx]]
            if len(sub) < 20 or len(np.unique(y[sub])) < 2:
                deltas = None
                break
            deltas.append(
                roc_auc_score(y[sub], pred_b[sub]) - roc_auc_score(y[sub], pred_a[sub])
            )
        if deltas:
            values.append(deltas[0] - deltas[1])
    arr = np.array(values)
    return {
        "interaction_mean": float(arr.mean()),
        "ci_2.5": float(np.percentile(arr, 2.5)),
        "ci_97.5": float(np.percentile(arr, 97.5)),
        "p_leq_0": float((arr <= 0).mean()),
        "n_boot": int(len(arr)),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preds", required=True, type=Path)
    parser.add_argument("--telemetry", required=True, type=Path)
    parser.add_argument("--vision", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)

    blob = np.load(args.preds, allow_pickle=True)
    y, gold, groups = blob["y"], blob["gold"], blob["groups"]
    tel, vis, late = blob["pred_telemetry"], blob["pred_vision"], blob["pred_late"]
    minutes = engage_minutes(args.telemetry, args.vision, len(y))

    imbalance = np.abs(gold)
    rho = float(spearmanr(minutes, imbalance).statistic)
    t_cut, g_cut = float(np.median(minutes)), float(np.median(imbalance))
    early, late_t = minutes <= t_cut, minutes > t_cut
    close, one_sided = imbalance <= g_cut, imbalance > g_cut

    results: dict[str, object] = {
        "n": int(len(y)),
        "spearman_time_vs_abs_gold": rho,
        "time_median_min": t_cut,
        "abs_gold_median": g_cut,
        "splits": {},
        "cells": {},
    }
    print(f"n={len(y)} | Spearman(time, |goldDiff|)={rho:.3f}"
          f" | time median={t_cut:.1f}min | |gold| median={g_cut:.3f}")

    for name, mask in (("early", early), ("late", late_t),
                       ("close", close), ("one_sided", one_sided)):
        entry = {
            "n": int(mask.sum()),
            "telemetry_auc": auc(y, tel, mask),
            "vision_auc": auc(y, vis, mask),
            "late_fusion_auc": auc(y, late, mask),
        }
        entry["delta_vision_minus_telemetry"] = entry["vision_auc"] - entry["telemetry_auc"]
        results["splits"][name] = entry
        print(f"  {name:10s} n={entry['n']:4d} tel={entry['telemetry_auc']:.3f}"
              f" vis={entry['vision_auc']:.3f} d={entry['delta_vision_minus_telemetry']:+.3f}")

    for t_name, t_mask in (("early", early), ("late", late_t)):
        for g_name, g_mask in (("close", close), ("one_sided", one_sided)):
            mask = t_mask & g_mask
            entry = {
                "n": int(mask.sum()),
                "median_minute": float(np.median(minutes[mask])),
                "telemetry_auc": auc(y, tel, mask),
                "vision_auc": auc(y, vis, mask),
            }
            entry["delta_vision_minus_telemetry"] = (
                entry["vision_auc"] - entry["telemetry_auc"]
                if entry["vision_auc"] is not None and entry["telemetry_auc"] is not None
                else None
            )
            results["cells"][f"{t_name}__{g_name}"] = entry
            print(f"  {t_name:5s} x {g_name:9s} n={entry['n']:4d}"
                  f" medT={entry['median_minute']:4.1f}"
                  f" d={entry['delta_vision_minus_telemetry']:+.3f}")

    everything = np.ones(len(y), dtype=bool)
    results["gold_interaction_within_time"] = {
        name: bootstrap_interaction(y, tel, vis, groups, pool, (close, one_sided))
        for name, pool in (("early", early), ("late", late_t), ("all", everything))
    }
    results["time_interaction_within_gold"] = {
        name: bootstrap_interaction(y, tel, vis, groups, pool, (early, late_t))
        for name, pool in (("close", close), ("one_sided", one_sided), ("all", everything))
    }
    for key in ("gold_interaction_within_time", "time_interaction_within_gold"):
        print(key)
        for name, stats in results[key].items():
            print(f"  {name:10s} {stats['interaction_mean']:+.3f}"
                  f" CI[{stats['ci_2.5']:+.3f},{stats['ci_97.5']:+.3f}]"
                  f" p(<=0)={stats['p_leq_0']:.3f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("wrote", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
