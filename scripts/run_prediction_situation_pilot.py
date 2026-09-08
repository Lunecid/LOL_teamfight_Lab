"""Pilot for the prediction-situation checks listed in docs/DEFINITION_EVIDENCE.md section 18.1.

Everything runs on the seeded 553-match pilot sample under the v3 definition
(G 13.7 s, D 4,264 u, R 1,600 u, B 15 s, market_lex label with draws dropped,
horizon 35 s, context 30 s) with the paper's tabular representation and the
ToG protocol learner (match-grouped 5-fold OOF LightGBM):

  ctx        context window 15 / 30 / 60 s (L = ctx / 5 s bins): AUC per length
  horizon    label horizon 35 / 45 / 60 s: label agreement with 35 s on shared
             engagements, AUC per horizon, and the 35 s model scored on the
             other horizons' labels (does the ranking transfer?)
  label      market_event (kill gold actually paid, ms-exact) vs market_lex
             (minute-frame team gold, linear): agreement, AUC, cross-scoring
  freshness  the age of the last frame before the cutoff appended as a feature

The fight index is built in-process (one worker, cache off) so that in-process
cfg changes reach the detector.

    LOL_OUTPUT_ROOT=D:/LOL_Project python scripts/run_prediction_situation_pilot.py \\
        --output D:/LOL_Project/fusion_2615/features/prediction_situation_pilot.json
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
      "LABEL_TYPE": "market_lex", "LABEL_TIE_POLICY": "drop",
      "FIGHT_INDEX_NUM_WORKERS": 1, "FIGHT_INDEX_CACHE_ENABLED": False, "DUMP_FIGHTS": False}


def load_module(name: str):
    spec = importlib.util.spec_from_file_location(name, PROJECT_ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def key_of(r):
    return (r.match_id, int(r.t_start_ts))


def build(refs, feature_set: str):
    from train.baseline import build_tabular_Xy
    X, y, names, used = build_tabular_Xy(refs, feature_set=feature_set)
    return {"X": X, "y": np.asarray(y).astype(int), "names": names, "used": used,
            "rows": {key_of(r): i for i, r in enumerate(used)}}


def align(a: dict, b: dict):
    common = sorted(set(a["rows"]) & set(b["rows"]))
    ia = np.array([a["rows"][k] for k in common], dtype=int)
    ib = np.array([b["rows"][k] for k in common], dtype=int)
    return common, ia, ib


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-matches", type=int, default=553)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--feature-set", default="full")
    ap.add_argument("--ctx", default="15,30,60")
    ap.add_argument("--horizons", default="35,45,60")
    ap.add_argument("--checks", default="ctx,horizon,label,freshness")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args(argv)
    checks = {c.strip() for c in args.checks.split(",") if c.strip()}

    from core.config import CACHE_DIR, cfg
    for k, v in V3.items():
        setattr(cfg, k, v)
    from data.index_split import build_fight_index
    from data.cache_io import load_match_cache
    sd = load_module("run_scale_decomposition")
    oof = sd.oof_predictions

    mids = [p.stem.replace(".meta", "") for p in CACHE_DIR.glob("*.meta.json")]
    if args.n_matches and args.n_matches < len(mids):
        mids = sorted(random.Random(args.seed).sample(mids, args.n_matches))
    t0 = time.time()
    refs = build_fight_index(cache_match_ids=mids)
    base = build(refs, args.feature_set)
    groups = np.array([r.match_id for r in base["used"]])
    print(f"pilot: {len(mids)} matches, {len(refs)} refs, {len(base['used'])} labelled rows, "
          f"{base['X'].shape[1]} features, positives {base['y'].mean():.3f} ({time.time() - t0:.0f}s)", flush=True)
    base_pred = oof(base["X"], base["y"], groups)
    base_auc = float(roc_auc_score(base["y"], base_pred))
    out = {"n_matches": len(mids), "n_refs": len(refs), "n_rows": int(len(base["y"])), "n_features": int(base["X"].shape[1]),
           "positive_rate": float(base["y"].mean()), "v3": V3, "base_auc": base_auc, "checks": {}}
    print(f"base AUC (ctx 30, H 35, market_lex): {base_auc:.4f}", flush=True)

    if "ctx" in checks:
        res = {}
        for ctx in [int(x) for x in args.ctx.split(",")]:
            cfg.FIGHT_CONTEXT_SEC = ctx
            b = build(refs, args.feature_set)
            common, ia, ib = align(base, b)
            g = np.array([k[0] for k in common])
            pred = oof(b["X"][ib], b["y"][ib], g)
            res[str(ctx)] = {"n_rows": int(len(ib)), "n_features": int(b["X"].shape[1]),
                             "auc": float(roc_auc_score(b["y"][ib], pred))}
            print(f"[ctx {ctx} s] features {b['X'].shape[1]} AUC {res[str(ctx)]['auc']:.4f}", flush=True)
        cfg.FIGHT_CONTEXT_SEC = V3["FIGHT_CONTEXT_SEC"]
        out["checks"]["ctx"] = res

    if "horizon" in checks:
        res = {}
        for H in [int(x) for x in args.horizons.split(",")]:
            cfg.FIGHT_HORIZON_SEC = H
            refs_h = build_fight_index(cache_match_ids=mids)
            b = build(refs_h, args.feature_set)
            common, ia, ib = align(base, b)
            g = np.array([k[0] for k in common])
            pred_h = oof(b["X"][ib], b["y"][ib], g)
            res[str(H)] = {"n_refs": len(refs_h), "n_rows": int(len(b["y"])), "n_common_with_35": int(len(common)),
                           "label_agreement_with_35": float(np.mean(base["y"][ia] == b["y"][ib])),
                           "positive_rate": float(b["y"].mean()),
                           "auc_own": float(roc_auc_score(b["y"][ib], pred_h)),
                           "auc_35_model_on_this_label": float(roc_auc_score(b["y"][ib], base_pred[ia]))}
            print(f"[H {H} s] rows {len(b['y'])} agree {res[str(H)]['label_agreement_with_35']:.3f} "
                  f"AUC own {res[str(H)]['auc_own']:.4f} | 35-model {res[str(H)]['auc_35_model_on_this_label']:.4f}", flush=True)
        cfg.FIGHT_HORIZON_SEC = V3["FIGHT_HORIZON_SEC"]
        out["checks"]["horizon"] = res

    if "label" in checks:
        res = {}
        for lab in ("market_event", "attention_value_win"):
            cfg.LABEL_TYPE = lab
            b = build(refs, args.feature_set)
            common, ia, ib = align(base, b)
            g = np.array([k[0] for k in common])
            pred_l = oof(b["X"][ib], b["y"][ib], g)
            res[lab] = {"n_rows": int(len(b["y"])), "n_common": int(len(common)),
                        "agreement_with_market_lex": float(np.mean(base["y"][ia] == b["y"][ib])),
                        "positive_rate": float(b["y"].mean()),
                        "auc_own": float(roc_auc_score(b["y"][ib], pred_l)),
                        "auc_market_lex_model_on_this_label": float(roc_auc_score(b["y"][ib], base_pred[ia])),
                        "auc_this_model_on_market_lex_label": float(roc_auc_score(base["y"][ia], pred_l))}
            print(f"[label {lab}] rows {len(b['y'])} agree {res[lab]['agreement_with_market_lex']:.3f} "
                  f"AUC own {res[lab]['auc_own']:.4f} | lex-model {res[lab]['auc_market_lex_model_on_this_label']:.4f}", flush=True)
        cfg.LABEL_TYPE = V3["LABEL_TYPE"]
        out["checks"]["label"] = res

    if "freshness" in checks:
        cache_ts = {}
        ages = np.zeros(len(base["used"]), dtype=np.float32)
        for i, r in enumerate(base["used"]):
            if r.match_id not in cache_ts:
                pack = load_match_cache(r.match_id)
                cache_ts[r.match_id] = np.asarray(pack["minute_ts"], dtype=np.int64) if pack else np.array([], dtype=np.int64)
            ts = cache_ts[r.match_id]
            j = int(np.searchsorted(ts, int(r.t_start_ts), side="right") - 1)
            ages[i] = (int(r.t_start_ts) - int(ts[j])) / 1000.0 if j >= 0 else np.nan
        ages = np.nan_to_num(ages, nan=30.0)
        X_aug = np.hstack([base["X"], ages.reshape(-1, 1)])
        pred_aug = oof(X_aug, base["y"], groups)
        pred_age = oof(ages.reshape(-1, 1), base["y"], groups)
        out["checks"]["freshness"] = {"age_s_quantiles": {f"p{p}": float(np.percentile(ages, p)) for p in (10, 50, 90)},
                                      "auc_base": base_auc, "auc_with_age": float(roc_auc_score(base["y"], pred_aug)),
                                      "auc_age_only": float(roc_auc_score(base["y"], pred_age))}
        print(f"[freshness] age median {np.median(ages):.1f}s AUC base {base_auc:.4f} +age {out['checks']['freshness']['auc_with_age']:.4f} "
              f"age-only {out['checks']['freshness']['auc_age_only']:.4f}", flush=True)

    out["elapsed_s"] = round(time.time() - t0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("wrote", args.output, f"in {out['elapsed_s']}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
