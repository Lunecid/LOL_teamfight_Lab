"""TreeSHAP attribution, re-run after the name-order fix, under market_lex.

The pre-fix pipeline assigned suffix-major values to feature-major names,
so every published attribution named the wrong feature. Names in the
shards are already suffix-major; LightGBM's native pred_contrib supplies
exact TreeSHAP values without the shap package.

    python scripts/run_shap_attribution.py --shards D:/LOL_Project/fusion_2615/corpus_shards_mlex ^
        --n-matches 40000 --output D:/LOL_Project/fusion_2615/features/shap_mlex.json
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--shards", required=True, type=Path)
    p.add_argument("--n-matches", type=int, default=40000)
    p.add_argument("--shap-rows", type=int, default=20000)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args(argv)

    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "dtb", Path(__file__).resolve().parent / "run_deep_tabular_baselines.py")
    dtb = importlib.util.module_from_spec(spec); spec.loader.exec_module(dtb)
    data = dtb.load_subsample(a.shards, a.n_matches, dtb.SEED)
    X, y, groups = data["X"], data["y"], data["groups"]
    mp = data["min_participants"]
    names = json.loads((a.shards / "feature_names.json").read_text(encoding="utf-8"))["names"]
    tr, va, te = dtb.split_by_match(groups, dtb.SEED)
    print(f"rows={len(y)} features={X.shape[1]}")

    from lightgbm import LGBMClassifier, early_stopping, log_evaluation
    m = LGBMClassifier(n_estimators=2000, learning_rate=0.05, num_leaves=31,
                       subsample=0.9, colsample_bytree=0.9, random_state=dtb.SEED,
                       n_jobs=-1, verbose=-1)
    m.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], eval_metric="auc",
          callbacks=[early_stopping(50, verbose=False), log_evaluation(0)])

    rng = np.random.default_rng(dtb.SEED)
    ti = np.flatnonzero(te)
    pick_rows = rng.choice(ti, size=min(a.shap_rows, len(ti)), replace=False)
    contrib = m.predict(X[pick_rows], pred_contrib=True)[:, :-1]
    mean_abs = np.abs(contrib).mean(axis=0)

    def top(mask, k=25):
        ma = np.abs(contrib[mask]).mean(axis=0)
        idx = np.argsort(ma)[::-1][:k]
        return [{"feature": names[j], "mean_abs_shap": float(ma[j])} for j in idx]

    cls = np.where(mp[pick_rows] <= 1, "pick",
                   np.where(mp[pick_rows] == 2, "skirmish", "teamfight"))
    out = {"n_shap_rows": int(len(pick_rows)),
           "overall_top": top(np.ones(len(pick_rows), bool)),
           "by_scale": {c: top(cls == c) for c in ("pick", "skirmish", "teamfight")}}
    base = {}
    for j, nm in enumerate(names):
        base.setdefault(nm.rsplit("__", 1)[0], 0.0)
        base[nm.rsplit("__", 1)[0]] += float(mean_abs[j])
    out["base_top"] = sorted(base.items(), key=lambda kv: -kv[1])[:25]
    for r in out["overall_top"][:12]:
        print(f"  {r['feature']:34s} {r['mean_abs_shap']:.4f}")
    a.output.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("wrote", a.output)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
