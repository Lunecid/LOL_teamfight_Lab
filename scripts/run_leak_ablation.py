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

--save-predictions DIR (off by default; without it the run and its JSON are unchanged) also writes,
per configuration, DIR/pred_<config>.npz with the out-of-fold predictions and the row keys, and
DIR/manifest.json with the drawn match ids, the cfg values, library versions, git state and file
hashes.  scripts/leak_ablation_paired_cis_v33.py turns those files into match-clustered paired
bootstrap intervals for every difference the JSON reports.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import random
import subprocess
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


# ---------------------------------------------------------------- optional prediction dump (--save-predictions)
# Nothing below runs unless --save-predictions is given, and nothing below feeds back into the rows,
# the learner or the JSON: it only reads what main() already computed.
N_SPLITS = 5          # sd.oof_predictions default; the dump re-derives the same GroupKFold folds
PRED_SCHEMA = "leak_ablation_predictions/v1"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_state() -> dict:
    def run(*cmd):
        return subprocess.run(["git", *cmd], cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=60).stdout
    try:
        dirty = [ln for ln in run("status", "--porcelain", "--untracked-files=no").splitlines() if ln.strip()]
        return {"commit": run("rev-parse", "HEAD").strip(), "tracked_changes": dirty}
    except Exception as e:  # provenance only; never fail the run on it
        return {"error": repr(e)}


def _fold_ids(y, groups) -> np.ndarray:
    """Fold of every row under sd.oof_predictions' GroupKFold(N_SPLITS) (GroupKFold ignores X)."""
    from sklearn.model_selection import GroupKFold
    fold = np.full(len(y), -1, dtype=np.int8)
    for k, (_, test) in enumerate(GroupKFold(n_splits=N_SPLITS).split(np.zeros((len(y), 1)), y, groups)):
        fold[test] = k
    assert (fold >= 0).all()
    return fold


def save_predictions(out_dir: Path, name: str, used, refs, y, pred, groups, cls) -> dict:
    """pred_<name>.npz: one row per engagement, in the order main() scored them (no pickled objects)."""
    ref_pos = {id(r): i for i, r in enumerate(refs)}
    fold = _fold_ids(y, groups)
    arrays = {
        "y": np.asarray(y, dtype=np.int8), "pred": np.asarray(pred, dtype=np.float64),
        "groups": np.asarray(groups).astype(str), "match_id": np.array([r.match_id for r in used]).astype(str),
        "t_start_ts": np.array([int(r.t_start_ts) for r in used], dtype=np.int64),
        "first_kill_ts": np.array([int(r.first_kill_ts) for r in used], dtype=np.int64),
        "last_kill_ts": np.array([int(r.last_kill_ts) for r in used], dtype=np.int64),
        "label_end_ts": np.array([int(r.label_end_ts) for r in used], dtype=np.int64),
        "patch": np.array([str(r.patch) for r in used]).astype(str),
        "det_cluster_blue": np.array([int(r.det_cluster_blue) for r in used], dtype=np.int16),
        "det_cluster_red": np.array([int(r.det_cluster_red) for r in used], dtype=np.int16),
        "scale_class": np.asarray(cls).astype(str), "fold": fold,
        "ref_index": np.array([ref_pos[id(r)] for r in used], dtype=np.int64),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"pred_{name}.npz"
    tmp = out_dir / f"pred_{name}.tmp.npz"
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)
    return {"file": path.name, "sha256": _sha256(path), "n_rows": int(len(y)),
            "fold_sizes": [int((fold == k).sum()) for k in range(N_SPLITS)]}


def write_manifest(out_dir: Path, args, mids, refs, files: dict, started: float) -> None:
    from core.config import CACHE_DIR, cfg
    import lightgbm
    import sklearn
    cfg_keys = sorted(set(V3) | {"TIME_NORM_ABSOLUTE", "ANCHORS_CAUSAL", "TAB_FRAME_AGE_FEATURE", "USE_MOMENTUM_FEATURES",
                                 "LABEL_TIE_SEED", "CACHE_DIRNAME", "FEATURE_VERSION"})
    manifest = {
        "schema": PRED_SCHEMA, "script": "scripts/run_leak_ablation.py",
        "argv": sys.argv, "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "output_json": str(args.output), "configs_flags": {k: {"TIME_NORM_ABSOLUTE": v[0], "ANCHORS_CAUSAL": v[1]}
                                                          for k, v in CONFIGS.items()},
        "v3_overrides": V3, "cfg_after_run": {k: getattr(cfg, k, None) for k in cfg_keys},
        "cache_dir": str(CACHE_DIR), "n_matches": len(mids), "n_refs": len(refs), "match_ids": list(mids),
        "match_ids_sha256": hashlib.sha256("\n".join(mids).encode("utf-8")).hexdigest(),
        "n_splits": N_SPLITS, "learner": "run_scale_decomposition.oof_predictions (published LightGBM configuration)",
        "files": files,
        "env": {k: os.environ.get(k) for k in ("LOL_OUTPUT_ROOT", "LOL_CFG_PRESET", "LOL_CFG_OVERRIDES", "LOKY_MAX_CPU_COUNT",
                                                "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
        "versions": {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
                     "numpy": np.__version__, "sklearn": sklearn.__version__, "lightgbm": lightgbm.__version__},
        "git": _git_state(), "started": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(started)),
        "finished": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "row_keys": "every pred_<config>.npz holds the same rows in the same order (asserted on (match_id, t_start_ts))",
    }
    tmp = out_dir / "manifest.tmp.json"
    tmp.write_text(json.dumps(manifest, indent=1, default=str), encoding="utf-8")
    os.replace(tmp, out_dir / "manifest.json")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-matches", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--configs", default=",".join(CONFIGS))
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--save-predictions", type=Path, default=None, metavar="DIR",
                    help="also write DIR/pred_<config>.npz (y, pred, groups, row keys, folds) and DIR/manifest.json; "
                         "off by default")
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
    saved = {}
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
        if args.save_predictions is not None:
            saved[name] = save_predictions(args.save_predictions, name, used, refs, y, pred, groups, cls)
            write_manifest(args.save_predictions, args, mids, refs, saved, t0)   # rewritten after every config
            print(f"  saved {args.save_predictions / saved[name]['file']}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=1), encoding="utf-8")
    print("wrote", args.output, f"{time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
