"""Matched-size telemetry control: same protocol, different corpus.

Answers why the recomputed telemetry baseline sits near chance on the 2026
fusion corpus while the CoG paper reports 0.675: sample size, corpus shift,
or the early-fight skew of the captured engagements.  Runs the identical
representation (``build_tabular_Xy``) and the identical nested-CV LightGBM
protocol (imported from ``run_fusion_experiment``) on a seeded sample of
matches from whatever cache ``LOL_OUTPUT_ROOT`` points at, then reports:

    full-matrix OOF AUC        (7,105-D, the number comparable to 0.675)
    compact macro OOF AUC      (team-diff bases x last/delta/slope)
    single-feature AUCs        (goldDiff, killDiff_cum, ... at the cutoff)
    |goldDiff| median split    of the frozen full-model OOF predictions

Cell A (original paper corpus, matched n):
    LOL_OUTPUT_ROOT=D:/LOL_Project python scripts/run_corpus_control.py ^
        --n-matches 553 --output <...>/corpus_control_original.json
Cell B (2026 fusion corpus, all engagements, from the exported npz):
    python scripts/run_corpus_control.py ^
        --telemetry-npz D:/LOL_Project/fusion_2615/features/telemetry_baseline_v2.npz ^
        --output <...>/corpus_control_2026.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SUFFIXES = ("last", "mean", "std", "min", "max", "delta", "slope")
SINGLES = (
    "goldDiff__last", "xpDiff__last", "killDiff_cum__last",
    "towerDiff_cum__last", "aliveDiff__last", "avgLevelDiff__last",
    "time_norm__last",
)


def load_fusion_module():
    spec = importlib.util.spec_from_file_location(
        "rfe", PROJECT_ROOT / "scripts" / "run_fusion_experiment.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def resolve_names(stored_names: list[str]) -> tuple[list[str], list[str]]:
    """Suffix-major names for suffix-major values (order auto-detected)."""
    k = len(stored_names) // len(SUFFIXES)
    first_block = {n.rsplit("__", 1)[1] for n in stored_names[:k]}
    if first_block == {SUFFIXES[0]}:
        return list(stored_names), [n.rsplit("__", 1)[0] for n in stored_names[:k]]
    base = [n.rsplit("__", 1)[0] for n in stored_names[:: len(SUFFIXES)]]
    return [f"{n}__{s}" for s in SUFFIXES for n in base], base


def single_auc(y: np.ndarray, values: np.ndarray) -> float:
    pos = y == 1
    r = rankdata(values)
    n1, n0 = int(pos.sum()), int((~pos).sum())
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def build_from_cache(n_matches: int, seed: int):
    from core.config import CACHE_DIR  # noqa: PLC0415 -- env-dependent import
    from data.index_split import build_fight_index  # noqa: PLC0415
    from train.baseline import build_tabular_Xy  # noqa: PLC0415

    metas = sorted(CACHE_DIR.glob("*.meta.json"))
    mids = [p.stem.replace(".meta", "") for p in metas]
    print(f"cache: {CACHE_DIR} | matches available: {len(mids)}")
    if n_matches and n_matches < len(mids):
        mids = sorted(random.Random(seed).sample(mids, n_matches))
    print(f"sampled matches: {len(mids)}")
    refs = build_fight_index(cache_match_ids=mids)
    print(f"fight refs: {len(refs)}")
    X, y, feat_names, used = build_tabular_Xy(refs, feature_set="full")
    groups = np.array([r.match_id for r in used])
    engage_min = np.array([r.t_start_ts for r in used], dtype=np.float64) / 60000.0
    return X, y, groups, engage_min, list(feat_names)


def load_from_npz(path: Path):
    blob = np.load(path)
    meta = json.loads(path.with_suffix(".keys.json").read_text(encoding="utf-8"))
    groups = np.array([k["match_id"] for k in meta["keys"]])
    engage_min = np.array([k["engage_ts_ms"] for k in meta["keys"]], dtype=np.float64) / 60000.0
    return blob["X"], blob["y"], groups, engage_min, meta["feature_names"]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--telemetry-npz", type=Path, default=None,
                        help="load an exported baseline instead of rebuilding from cache")
    parser.add_argument("--n-matches", type=int, default=553)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)

    if args.telemetry_npz is not None:
        X, y, groups, engage_min, stored = load_from_npz(args.telemetry_npz)
    else:
        X, y, groups, engage_min, stored = build_from_cache(args.n_matches, args.seed)
    names, base = resolve_names(stored)
    print(f"X: {X.shape} | positives: {y.mean():.3f} | matches: {len(np.unique(groups))}")

    rfe = load_fusion_module()
    gold = X[:, names.index("goldDiff__last")]
    results: dict[str, object] = {
        "n": int(len(y)),
        "matches": int(len(np.unique(groups))),
        "positive_rate": float(y.mean()),
        "engage_min_median": float(np.median(engage_min)),
        "abs_gold_diff_median": float(np.median(np.abs(gold))),
        "single_feature_auc": {
            nm: single_auc(y, X[:, names.index(nm)]) for nm in SINGLES if nm in names
        },
    }
    print("single-feature AUC:", json.dumps(results["single_feature_auc"], indent=1))

    pred_full = rfe.oof_predictions(X, y, groups)
    results["full_matrix_auc"] = float(roc_auc_score(y, pred_full))
    print(f"full-matrix ({X.shape[1]}-D) OOF AUC: {results['full_matrix_auc']:.4f}")

    macro = [b for b in base if ("Diff" in b) or b.startswith("dist_") or b == "time_norm"]
    compact_idx = [names.index(f"{b}__{s}") for b in macro for s in ("last", "delta", "slope")]
    pred_compact = rfe.oof_predictions(X[:, compact_idx], y, groups)
    results["compact_dims"] = len(compact_idx)
    results["compact_auc"] = float(roc_auc_score(y, pred_compact))
    print(f"compact ({len(compact_idx)}-D) OOF AUC: {results['compact_auc']:.4f}")

    cut = float(np.median(np.abs(gold)))
    for name, mask in (("close", np.abs(gold) <= cut), ("one_sided", np.abs(gold) > cut)):
        results[name] = {
            "n": int(mask.sum()),
            "full_matrix_auc": float(roc_auc_score(y[mask], pred_full[mask])),
            "compact_auc": float(roc_auc_score(y[mask], pred_compact[mask])),
        }
        print(f"{name}: n={mask.sum()} full={results[name]['full_matrix_auc']:.4f}"
              f" compact={results[name]['compact_auc']:.4f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("wrote", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
