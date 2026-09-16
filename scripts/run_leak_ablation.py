"""How much of the AUC came from the two input leaks? Same engagements, same label, flags toggled.

Configurations (label fixed to the v3.3 market_event, engagement-attributed, priced):
  clean          TIME_NORM_ABSOLUTE=True,  ANCHORS_CAUSAL=True   (v3.3)
  time_leak      TIME_NORM_ABSOLUTE=False, ANCHORS_CAUSAL=True   (time_norm = t / match length)
  anchor_leak    TIME_NORM_ABSOLUTE=True,  ANCHORS_CAUSAL=False  (anchors = every tower that fell)
  both_leaks     TIME_NORM_ABSOLUTE=False, ANCHORS_CAUSAL=False  (v3 feature path)

Rows are built once per configuration on the same fight refs and aligned by (match, cutoff);
the learner and folds are the ToG protocol (match-grouped 5-fold OOF LightGBM).  Reports the
overall AUC and the AUC by participation scale (teamfight = smaller side >= 4).

    LOL_OUTPUT_ROOT=D:/LOL_Project python scripts/run_leak_ablation.py --n-matches 5000 \\
        --output D:/LOL_Project/fusion_2615/features/leak_ablation_v33.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

V3 = {"TF2_KILL_CLUSTER_GAP_MS": 13700, "CLUSTER_MAX_DIAMETER": 4264.0, "TF2_VALIDITY_RADIUS": 1600.0,
      "TF2_ENGAGE_PRE_KILL_MS": 15000, "FIGHT_HORIZON_SEC": 35, "FIGHT_CONTEXT_SEC": 30,
      "LABEL_TYPE": "market_event", "LABEL_TIE_POLICY": "drop", "LABEL_EVENT_ATTRIBUTION": "engagement",
      "FIGHT_INDEX_NUM_WORKERS": 1, "FIGHT_INDEX_CACHE_ENABLED": False, "DUMP_FIGHTS": False}
CONFIGS = {"clean": (True, True), "time_leak": (False, True), "anchor_leak": (True, False), "both_leaks": (False, False)}


def load_module(name):
    spec = importlib.util.spec_from_file_location(name, PROJECT_ROOT / "scripts" / f"{name}.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def scale_class(refs):
    out = []
    for r in refs:
        s = min(int(r.det_cluster_blue), int(r.det_cluster_red))
        out.append("pick" if s <= 1 else ("skirmish" if s <= 3 else "teamfight"))
    return np.array(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-matches", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--configs", default=",".join(CONFIGS))
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args(argv)
    from core.config import CACHE_DIR, cfg
    for k, v in V3.items():
        setattr(cfg, k, v)
    from data.index_split import build_fight_index
    from train.baseline import build_tabular_Xy
    sd = load_module("run_scale_decomposition")

    mids = [p.stem.replace(".meta", "") for p in CACHE_DIR.glob("*.meta.json")]
    if args.n_matches and args.n_matches < len(mids):
        mids = sorted(random.Random(args.seed).sample(mids, args.n_matches))
    t0 = time.time()
    refs = build_fight_index(cache_match_ids=mids)
    print(f"{len(mids)} matches, {len(refs)} refs ({time.time() - t0:.0f}s)", flush=True)
    results = {"n_matches": len(mids), "n_refs": len(refs), "configs": {}}
    base_rows = None
    for name in [c.strip() for c in args.configs.split(",") if c.strip()]:
        t_abs, anc = CONFIGS[name]
        cfg.TIME_NORM_ABSOLUTE, cfg.ANCHORS_CAUSAL = t_abs, anc
        t1 = time.time()
        X, y, names, used = build_tabular_Xy(refs, feature_set="full")
        keys = [(r.match_id, int(r.t_start_ts)) for r in used]
        if base_rows is None:
            base_rows = keys
        assert keys == base_rows, "row set changed between configurations"
        groups = np.array([k[0] for k in keys]); y = np.asarray(y).astype(int)
        cls = scale_class(used)
        pred = sd.oof_predictions(X, y, groups)
        rec = {"n_rows": int(len(y)), "n_features": int(X.shape[1]), "auc": float(roc_auc_score(y, pred)),
               "by_class": {c: {"n": int((cls == c).sum()), "auc": float(roc_auc_score(y[cls == c], pred[cls == c]))}
                            for c in ("pick", "skirmish", "teamfight") if (cls == c).sum() > 50 and len(set(y[cls == c])) > 1},
               "build_s": round(time.time() - t1)}
        rec["pick_minus_teamfight"] = (rec["by_class"]["pick"]["auc"] - rec["by_class"]["teamfight"]["auc"]
                                       if "pick" in rec["by_class"] and "teamfight" in rec["by_class"] else None)
        results["configs"][name] = rec
        print(f"[{name:12s}] rows {len(y):,} AUC {rec['auc']:.4f} | " +
              " ".join(f"{c} {v['auc']:.4f}" for c, v in rec["by_class"].items()) + f" ({rec['build_s']}s)", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=1), encoding="utf-8")
    print("wrote", args.output, f"{time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
