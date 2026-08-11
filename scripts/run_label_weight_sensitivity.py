"""How sensitive are the results to Eq.3's hand-set coefficients?

Completes the answer to CoG review R2's label concern: perturb one Eq.3
coefficient at a time (x0.5 / x2), recompute labels on the same detected
engagements, and report how many labels flip and how far the same
representation's nested-CV AUC moves.  Small flip rates and a flat AUC
mean the coefficient choice is not what the model learns.

All value weights and the softmax temperature were already runtime
config; the attention-prior coefficients were promoted to config fields
(defaults identical to the paper) for this sweep.

    LOL_OUTPUT_ROOT=D:/LOL_Project python scripts/run_label_weight_sensitivity.py ^
        --n-matches 553 --seed 7 ^
        --output D:/LOL_Project/fusion_2615/features/label_weight_sensitivity.json
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

# one-at-a-time perturbations around the paper values
SETTINGS: list[tuple[str, dict[str, float]]] = [
    ("default", {}),
    ("w_kill_x0.5", {"LABEL_ATTN_W_KILL": 0.5}),
    ("w_kill_x2", {"LABEL_ATTN_W_KILL": 2.0}),
    ("w_shutdown_x0.5", {"LABEL_ATTN_W_SHUTDOWN": 0.8}),
    ("w_shutdown_x2", {"LABEL_ATTN_W_SHUTDOWN": 3.2}),
    ("w_objective_x0.5", {"LABEL_ATTN_W_OBJECTIVE": 0.55}),
    ("w_objective_x2", {"LABEL_ATTN_W_OBJECTIVE": 2.2}),
    ("beta_x0.5", {"LABEL_ATTN_BETA": 1.0}),
    ("beta_x2", {"LABEL_ATTN_BETA": 4.0}),
    ("prior_kill_x0.5", {"LABEL_ATTN_PRIOR_W_KILL": 0.125}),
    ("prior_kill_x2", {"LABEL_ATTN_PRIOR_W_KILL": 0.5}),
]


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
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)

    from core.config import CACHE_DIR, cfg
    from data.index_split import build_fight_index
    from train.baseline import build_tabular_Xy

    mids = [p.stem.replace(".meta", "") for p in CACHE_DIR.glob("*.meta.json")]
    if args.n_matches and args.n_matches < len(mids):
        mids = sorted(random.Random(args.seed).sample(mids, args.n_matches))
    refs = build_fight_index(cache_match_ids=mids)
    print(f"matches={len(mids)} refs={len(refs)}")

    rfe = load_fusion_module()
    key_of = lambda r: (r.match_id, int(r.t_start_ts))  # noqa: E731

    # snapshot every touched key once; each setting starts from the snapshot.
    # keys absent from cfg fall back to the getattr defaults in labels.py,
    # which equal the paper values, so restoring the snapshot restores those.
    touched = sorted({k for _, o in SETTINGS for k in o})
    saved = {k: getattr(cfg, k, None) for k in touched}

    def set_cfg(overrides: dict[str, float]) -> None:
        for k in touched:
            if saved[k] is None and hasattr(cfg, k):
                try:
                    delattr(cfg, k)
                except AttributeError:
                    setattr(cfg, k, saved[k])
            elif saved[k] is not None:
                setattr(cfg, k, saved[k])
        for k, v in overrides.items():
            setattr(cfg, k, v)

    results: dict[str, object] = {"n_matches": len(mids), "n_refs": len(refs), "settings": {}}
    baseline: dict[tuple[str, int], int] = {}
    for name, overrides in SETTINGS:
        set_cfg(overrides)
        X, y, _, used = build_tabular_Xy(refs, feature_set="full")
        rows = {key_of(r): int(y[i]) for i, r in enumerate(used)}
        groups = np.array([r.match_id for r in used])
        pred = rfe.oof_predictions(X, y, groups)
        entry = {
            "overrides": overrides,
            "n_rows": int(len(y)),
            "positive_rate": float(y.mean()),
            "auc": float(roc_auc_score(y, pred)),
        }
        if name == "default":
            baseline = rows
        else:
            shared = [k for k in rows if k in baseline]
            flips = sum(1 for k in shared if rows[k] != baseline[k])
            entry["shared_rows"] = len(shared)
            entry["label_flip_rate"] = flips / max(1, len(shared))
        results["settings"][name] = entry
        flip = entry.get("label_flip_rate")
        print(f"[{name}] rows={entry['n_rows']} pos={entry['positive_rate']:.3f}"
              f" auc={entry['auc']:.4f}"
              + (f" flips={flip:.3f}" if flip is not None else ""))
    set_cfg({})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("wrote", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
