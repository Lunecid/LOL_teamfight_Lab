"""TreeSHAP attribution, re-run after the name-order fix, under market_lex.

The pre-fix pipeline assigned suffix-major values to feature-major names,
so every published attribution named the wrong feature. Names in the
shards are already suffix-major; LightGBM's native pred_contrib supplies
exact TreeSHAP values without the shap package.

    python scripts/run_shap_attribution.py --shards D:/LOL_Project/fusion_2615/corpus_shards_mlex ^
        --n-matches 40000 --output D:/LOL_Project/fusion_2615/features/shap_mlex.json

Corpus v3.3 flags (defaults reproduce the run above; the JSON only gains a "provenance" block):
  --y-key y_market_event   label column; rows with label < 0 (draws under the drop policy) are removed.
                           The default "y" is the shard's row label, a coin flip on ties in v3.3.
  --split patch            train 15.14 / val 15.15 / test 15.16 via run_deep_tabular_baselines.split_by_patch
  --teamfight-min 4        v3 participation cut for by_scale (default 3 = the published run)
  --csv PATH               full mean |SHAP| vector, one row per column (feature,mean_abs_shap: the
                           analysis/shap_role_rollup.py input), plus <PATH stem>.base.csv with the sum over each
                           base's summary statistics (base,mean_abs_shap,n_columns)
  --n-boot 1000            provenance.test_auc: test-split AUC with a match-clustered percentile bootstrap CI
                           (run_deep_tabular_baselines.cluster_bootstrap_auc); 0 = point estimate only

    LOL_OUTPUT_ROOT=D:/LOL_Project .venv/Scripts/python.exe scripts/run_shap_attribution.py \\
        --shards D:/LOL_Project/fusion_2615/corpus_shards_v33 \\
        --n-matches 40000 --y-key y_market_event --split patch --teamfight-min 4 \\
        --csv D:/LOL_Project/fusion_2615/features/tog_revision/A7-shap-forensics/shap_v33_mean_abs.csv \\
        --output D:/LOL_Project/fusion_2615/features/tog_revision/A7-shap-forensics/shap_v33.json

Reference: Lundberg et al., "From local explanations to global understanding with explainable AI for
trees", Nature Machine Intelligence 2020 -- path-dependent Tree SHAP (what pred_contrib=True computes,
on the raw log-odds margin) and mean |SHAP| as global importance.  For the clean-vs-leaky forensics on
identical engagements see scripts/run_shap_leak_forensics_v33.py.
"""
from __future__ import annotations
import argparse, csv, json, subprocess, time
from pathlib import Path
import numpy as np

def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--shards", required=True, type=Path)
    p.add_argument("--n-matches", type=int, default=40000)
    p.add_argument("--shap-rows", type=int, default=20000)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--y-key", default="y",
                   help="label column in the shards; rows with label < 0 (draws under the drop policy) are removed")
    p.add_argument("--split", choices=("match", "patch"), default="match",
                   help="match: seeded 70/15/15 match split (published); patch: whole-patch holdout")
    p.add_argument("--train-patch", default="15.14")
    p.add_argument("--val-patch", default="15.15")
    p.add_argument("--test-patch", default="15.16")
    p.add_argument("--teamfight-min", type=int, default=3,
                   help="smaller side's participation at which by_scale calls it a teamfight (3 = published, 4 = v3)")
    p.add_argument("--n-jobs", type=int, default=-1, help="LightGBM threads")
    p.add_argument("--csv", type=Path, default=None,
                   help="also write the full mean |SHAP| vector and <stem>.base.csv (sum over each base's statistics)")
    p.add_argument("--n-boot", type=int, default=1000,
                   help="match-clustered bootstrap replicates for the test AUC CI in provenance (0 = no CI)")
    a = p.parse_args(argv)
    started = time.time()

    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "dtb", Path(__file__).resolve().parent / "run_deep_tabular_baselines.py")
    dtb = importlib.util.module_from_spec(spec); spec.loader.exec_module(dtb)
    data = dtb.load_subsample(a.shards, a.n_matches, dtb.SEED, y_key=a.y_key)
    keep = data["y"] >= 0
    if not keep.all():
        print(f"label {a.y_key}: dropping {int((~keep).sum())} rows without a label (draws)")
        for k in ("X", "y", "groups", "patch", "min_participants"):
            data[k] = data[k][keep]
    X, y, groups = data["X"], data["y"], data["groups"]
    mp = data["min_participants"]
    names = json.loads((a.shards / "feature_names.json").read_text(encoding="utf-8"))["names"]
    if a.split == "patch":
        tr, va, te = dtb.split_by_patch(data["patch"], a.train_patch, a.val_patch, a.test_patch)
    else:
        tr, va, te = dtb.split_by_match(groups, dtb.SEED)
    print(f"rows={len(y)} features={X.shape[1]}")

    from lightgbm import LGBMClassifier, early_stopping, log_evaluation
    m = LGBMClassifier(n_estimators=2000, learning_rate=0.05, num_leaves=31,
                       subsample=0.9, colsample_bytree=0.9, random_state=dtb.SEED,
                       n_jobs=a.n_jobs, verbose=-1)
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
                   np.where(mp[pick_rows] < a.teamfight_min, "skirmish", "teamfight"))
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
    if a.csv is not None:
        a.csv.parent.mkdir(parents=True, exist_ok=True)
        with open(a.csv, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f); w.writerow(["feature", "mean_abs_shap"])
            w.writerows([nm, repr(float(v))] for nm, v in zip(names, mean_abs))
        n_cols = {}
        for nm in names:
            n_cols[nm.rsplit("__", 1)[0]] = n_cols.get(nm.rsplit("__", 1)[0], 0) + 1
        base_csv = a.csv.with_name(a.csv.stem + ".base.csv")
        with open(base_csv, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f); w.writerow(["base", "mean_abs_shap", "n_columns"])
            w.writerows([b, repr(v), n_cols[b]] for b, v in sorted(base.items(), key=lambda kv: -kv[1]))
        print("wrote", a.csv, "and", base_csv)
    manifest = a.shards / "manifest.json"
    mf = json.loads(manifest.read_text(encoding="utf-8")) if manifest.exists() else {}
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(Path(__file__).resolve().parents[1]),
                                capture_output=True, text=True).stdout.strip()
    except Exception:
        commit = ""
    from sklearn.metrics import roc_auc_score
    p_te = m.predict_proba(X[te])[:, 1]
    test_auc = {"auc": float(roc_auc_score(y[te], p_te)), "n_rows": int(te.sum()),
                "n_matches": int(len(np.unique(groups[te]))), "positive_rate": float(y[te].mean())}
    if a.n_boot > 0:
        boot = dtb.cluster_bootstrap_auc(y[te], {"lightgbm": p_te}, groups[te], n_boot=a.n_boot, seed=dtb.SEED)
        test_auc.update({"ci95": boot["auc"]["lightgbm"]["ci95"], "n_boot": boot["n_boot"], "method": boot["method"]})
    deviations = [
        "Tree SHAP is LightGBM's pred_contrib (path-dependent conditional expectation, raw log-odds margin), the "
        "algorithm of Lundberg et al. 2020; the interventional variant is not used.",
        "Global importance is mean |SHAP| over a seeded sample of test rows (--shap-rows), one model, one seed.",
        "LightGBM is fitted on raw features; the learner comparison (run_deep_tabular_baselines) standardises on "
        "train rows first.",
        "The base rollup sums mean |SHAP| over a base's summary-statistic columns (<base>__<statistic>).",
    ]
    if a.split == "match":
        deviations.append("split_by_match: a seeded 70/15/15 match split, not the patch holdout of the learner comparison.")
    if a.y_key == "y":
        deviations.append("label 'y' is the shard row label; in corpus v3.3 it breaks draws by a coin flip "
                          "(use --y-key y_market_event for the drop policy).")
    out["provenance"] = {
        "git_commit": commit, "shards": str(a.shards),
        "corpus_manifest": {k: mf.get(k) for k in ("run_id", "git_commit", "preset", "feature_names_sha1")},
        "y_key": a.y_key, "n_rows": int(len(y)), "n_matches": int(len(np.unique(groups))), "seed": int(dtb.SEED),
        "split": {"kind": a.split, "train": int(tr.sum()), "val": int(va.sum()), "test": int(te.sum()),
                  **({"train_patch": a.train_patch, "val_patch": a.val_patch, "test_patch": a.test_patch}
                     if a.split == "patch" else {})},
        "teamfight_min": a.teamfight_min, "best_iteration": int(m.best_iteration_ or m.n_estimators),
        "diagnostic_val_auc": float(roc_auc_score(y[va], m.predict_proba(X[va])[:, 1])),
        "test_auc": test_auc,
        "shap_rows_by_scale": {c: int((cls == c).sum()) for c in ("pick", "skirmish", "teamfight")},
        "reference": "Lundberg et al., Nature Machine Intelligence 2, 56-67 (2020): Tree SHAP, mean |SHAP| importance",
        "deviations": deviations,
        "n_jobs": a.n_jobs, "wall_s": round(time.time() - started, 1),
    }
    a.output.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("wrote", a.output)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
