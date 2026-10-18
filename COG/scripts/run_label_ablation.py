"""Is the model learning the game, or the labelling heuristic?

CoG 2026 review R2's main concern: the Eq.3 engagement label aggregates
post-onset events with hand-set weights, so a model fit to it might be
learning the weighting scheme rather than the underlying dynamics.  This
ablation holds the representation fixed and swaps the label:

    attention_value_win   Eq.3 as submitted (cfg default)
    micro_win             raw kill advantage inside the cluster window
    kill_survival         kills + survivor margin
    weighted              gold/objective-weighted composite

For every scheme the same matches, the same engagements, the same tabular
matrix, and the same match-grouped folds are used; only y changes.  Rows
are restricted to engagements where every scheme yields a label (tie-drop
intersection) so AUCs are computed on identical rows.  Reported:

    pairwise label agreement between schemes
    per-scheme nested-CV LightGBM AUC (identical folds)
    cross-label AUC: model trained/evaluated under Eq.3 folds, scored
    against each alternative y (does the *ranking* transfer?)

If agreements are high and AUC is stable across schemes, the prediction
target is the engagement's material outcome, not the coefficient choice.

Run against whatever corpus LOL_OUTPUT_ROOT points at, e.g. the original
paper cache with the seeded 553-match pilot sample:

    LOL_OUTPUT_ROOT=D:/LOL_Project python scripts/run_label_ablation.py ^
        --n-matches 553 --seed 7 ^
        --output D:/LOL_Project/fusion_2615/features/label_ablation_pilot.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SCHEMES = ("attention_value_win", "micro_win", "kill_survival", "weighted")


def load_fusion_module():
    spec = importlib.util.spec_from_file_location(
        "rfe", PROJECT_ROOT / "scripts" / "run_fusion_experiment.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-matches", type=int, default=553)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--feature-set", default="full")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)

    from core.config import CACHE_DIR, cfg
    from data.index_split import build_fight_index
    from train.baseline import build_tabular_Xy

    mids = [p.stem.replace(".meta", "") for p in CACHE_DIR.glob("*.meta.json")]
    print(f"cache: {CACHE_DIR} | matches available: {len(mids)}")
    if args.n_matches and args.n_matches < len(mids):
        mids = sorted(random.Random(args.seed).sample(mids, args.n_matches))
    refs = build_fight_index(cache_match_ids=mids)
    print(f"matches: {len(mids)} | fight refs: {len(refs)}")

    default_scheme = str(getattr(cfg, "LABEL_TYPE"))
    per_scheme: dict[str, dict] = {}
    key_of = lambda r: (r.match_id, int(r.t_start_ts))  # noqa: E731
    try:
        for scheme in SCHEMES:
            cfg.LABEL_TYPE = scheme
            X, y, names, used = build_tabular_Xy(refs, feature_set=args.feature_set)
            per_scheme[scheme] = {
                "X": X, "y": y,
                "rows": {key_of(r): i for i, r in enumerate(used)},
            }
            print(f"[{scheme}] rows={len(used)} positives={y.mean():.3f}")
    finally:
        cfg.LABEL_TYPE = default_scheme

    shared = set.intersection(*(set(s["rows"]) for s in per_scheme.values()))
    shared = sorted(shared)
    print(f"rows labelled by every scheme: {len(shared)}")
    base = per_scheme[SCHEMES[0]]
    base_idx = np.array([base["rows"][k] for k in shared])
    X = base["X"][base_idx]
    groups = np.array([k[0] for k in shared])
    Y = {
        scheme: blob["y"][np.array([blob["rows"][k] for k in shared])]
        for scheme, blob in per_scheme.items()
    }

    results: dict[str, object] = {
        "n_matches": len(mids),
        "n_refs": len(refs),
        "n_shared_rows": len(shared),
        "positive_rate": {s: float(v.mean()) for s, v in Y.items()},
        "pairwise_agreement": {},
        "auc_same_scheme": {},
        "auc_eq3_model_on_alt_labels": {},
    }
    for i, a in enumerate(SCHEMES):
        for b in SCHEMES[i + 1:]:
            agreement = float((Y[a] == Y[b]).mean())
            results["pairwise_agreement"][f"{a}__vs__{b}"] = agreement
            print(f"agreement {a} vs {b}: {agreement:.3f}")

    rfe = load_fusion_module()
    preds = {}
    for scheme in SCHEMES:
        preds[scheme] = rfe.oof_predictions(X, Y[scheme], groups)
        auc = float(roc_auc_score(Y[scheme], preds[scheme]))
        results["auc_same_scheme"][scheme] = auc
        print(f"AUC trained+scored on {scheme}: {auc:.4f}")

    eq3 = preds["attention_value_win"]
    for scheme in SCHEMES:
        auc = float(roc_auc_score(Y[scheme], eq3))
        results["auc_eq3_model_on_alt_labels"][scheme] = auc
        print(f"Eq.3-trained model scored against {scheme} labels: {auc:.4f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    np.savez_compressed(
        args.output.with_suffix(".preds.npz"),
        groups=groups,
        **{f"y_{s}": Y[s] for s in SCHEMES},
        **{f"pred_{s}": preds[s] for s in SCHEMES},
    )
    print("wrote", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
