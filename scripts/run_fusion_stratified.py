"""Stratified test of the CoG pre-registered hypothesis.

CoG 2026 reports that predictive power concentrates in one-sided resource
states and that close-gold engagements stay hard.  If replay video carries
the sub-minute execution state that minute-resolution telemetry cannot see,
the vision advantage should be *larger* in the close-gold stratum.

Method: fit each modality once under match-grouped CV (identical folds and
rows), then evaluate the frozen out-of-fold predictions inside strata
defined by |goldDiff| at the prediction cutoff.  Models are never refit per
stratum, so stratum sample size cannot confound the comparison.  The
interaction (delta_close - delta_one_sided) gets its own match-cluster
bootstrap.

Example:
    python scripts/run_fusion_stratified.py ^
        --telemetry D:/LOL_Project/fusion_2615/features/telemetry_baseline.npz ^
        --vision D:/LOL_Project/fusion_2615/features/engagement_features.jsonl ^
        --output D:/LOL_Project/fusion_2615/features/stratified_results.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUFFIXES = ("last", "mean", "std", "min", "max", "delta", "slope")
SEED = 7


def load_fusion_module():
    spec = importlib.util.spec_from_file_location(
        "rfe", PROJECT_ROOT / "scripts" / "run_fusion_experiment.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def resolve_names(stored_names: list[str]) -> tuple[list[str], list[str]]:
    """Return (names, bases) in the suffix-major order the values use.

    Exports made before the feature-contract fix stored feature-major names
    against suffix-major values; later exports store suffix-major names
    directly.  Detect which case this file is: in suffix-major order the
    first ``len/7`` names all share the ``__last`` suffix.
    """
    k = len(stored_names) // len(SUFFIXES)
    first_block = {n.rsplit("__", 1)[1] for n in stored_names[:k]}
    if first_block == {SUFFIXES[0]}:  # already suffix-major: use as stored
        return list(stored_names), [n.rsplit("__", 1)[0] for n in stored_names[:k]]
    base = [n.rsplit("__", 1)[0] for n in stored_names[:: len(SUFFIXES)]]
    return [f"{n}__{s}" for s in SUFFIXES for n in base], base


def stratum_auc(y, pred, mask):
    if mask.sum() < 20 or len(np.unique(y[mask])) < 2:
        return None
    return float(roc_auc_score(y[mask], pred[mask]))


def cluster_bootstrap_stratified(y, pred_a, pred_b, groups, masks, n_boot=2000):
    """Bootstrap deltas per stratum and their difference, resampling matches."""
    rng = np.random.default_rng(SEED)
    unique = np.unique(groups)
    index_by_group = {g: np.where(groups == g)[0] for g in unique}
    collected = {name: [] for name in masks}
    interaction = []
    for _ in range(n_boot):
        sample = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([index_by_group[g] for g in sample])
        deltas = {}
        for name, mask in masks.items():
            sub = idx[mask[idx]]
            if len(sub) < 20 or len(np.unique(y[sub])) < 2:
                deltas[name] = None
                continue
            delta = roc_auc_score(y[sub], pred_b[sub]) - roc_auc_score(y[sub], pred_a[sub])
            deltas[name] = delta
            collected[name].append(delta)
        if deltas.get("close") is not None and deltas.get("one_sided") is not None:
            interaction.append(deltas["close"] - deltas["one_sided"])
    def summary(values):
        arr = np.array(values)
        return {
            "delta_mean": float(arr.mean()),
            "ci_2.5": float(np.percentile(arr, 2.5)),
            "ci_97.5": float(np.percentile(arr, 97.5)),
            "p_leq_0": float((arr <= 0).mean()),
            "n_boot": int(len(arr)),
        }
    out = {name: summary(vals) for name, vals in collected.items() if vals}
    if interaction:
        out["interaction_close_minus_one_sided"] = summary(interaction)
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--telemetry", required=True, type=Path)
    parser.add_argument("--vision", required=True, type=Path)
    parser.add_argument("--window", default="w30")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)

    rfe = load_fusion_module()
    blob = np.load(args.telemetry)
    meta = json.loads(args.telemetry.with_suffix(".keys.json").read_text(encoding="utf-8"))
    names, base = resolve_names(meta["feature_names"])
    keys = meta["keys"]
    vision = rfe.load_vision(args.vision)

    joined = [
        (i, vision[(k["match_id"], k["engage_ts_ms"])])
        for i, k in enumerate(keys)
        if (k["match_id"], k["engage_ts_ms"]) in vision
    ]
    idx = np.array([i for i, _ in joined])
    rows = [row for _, row in joined]
    y = blob["y"][idx]
    groups = np.array([keys[i]["match_id"] for i in idx])
    X_all = blob["X"][idx]

    macro_bases = [b for b in base if ("Diff" in b) or b.startswith("dist_") or b == "time_norm"]
    compact_idx = [names.index(f"{b}__{s}") for b in macro_bases for s in ("last", "delta", "slope")]
    X_tel = X_all[:, compact_idx]
    X_vis, _ = rfe.vision_matrix(rows, args.window)

    gold = X_all[:, names.index("goldDiff__last")]
    magnitude = np.abs(gold)
    cut = np.median(magnitude)
    masks = {"close": magnitude <= cut, "one_sided": magnitude > cut}
    print(
        f"n={len(y)} matches={len(np.unique(groups))} | close={int(masks['close'].sum())}"
        f" one_sided={int(masks['one_sided'].sum())} | |goldDiff| median={cut:.4f}"
    )

    pred_tel = rfe.oof_predictions(X_tel, y, groups)
    pred_vis = rfe.oof_predictions(X_vis, y, groups)
    pred_fus = rfe.oof_predictions(np.hstack([X_tel, X_vis]), y, groups)

    results: dict[str, object] = {
        "window": args.window,
        "n": int(len(y)),
        "matches": int(len(np.unique(groups))),
        "gold_split_value": float(cut),
        "overall": {
            "telemetry_auc": float(roc_auc_score(y, pred_tel)),
            "vision_auc": float(roc_auc_score(y, pred_vis)),
            "fusion_auc": float(roc_auc_score(y, pred_fus)),
        },
    }
    for name, mask in masks.items():
        results[name] = {
            "n": int(mask.sum()),
            "positive_rate": float(y[mask].mean()),
            "telemetry_auc": stratum_auc(y, pred_tel, mask),
            "vision_auc": stratum_auc(y, pred_vis, mask),
            "fusion_auc": stratum_auc(y, pred_fus, mask),
        }
    results["bootstrap_vision_minus_telemetry"] = cluster_bootstrap_stratified(
        y, pred_tel, pred_vis, groups, masks
    )
    results["bootstrap_fusion_minus_telemetry"] = cluster_bootstrap_stratified(
        y, pred_tel, pred_fus, groups, masks
    )

    print(json.dumps({k: results[k] for k in ("overall", "close", "one_sided")}, indent=2))
    print("vision-telemetry:", json.dumps(results["bootstrap_vision_minus_telemetry"], indent=2))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("wrote", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
