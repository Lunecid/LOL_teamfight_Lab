"""Match-clustered paired bootstrap intervals for the patch-holdout learner comparison (v3.3).

The learner table ranks nine learners on one test patch (15.16, 153,816 engagements).  A ranking
is only a finding where the gap survives the test set's own sampling variability, and engagements
from one match share its state, so the resampling unit is the match, not the row.  Every learner
is scored on the SAME resampled matches in each replicate and differences are taken inside the
replicate, so the interval carries the correlation between learners.

Method and references
  resampling   Davison & Hinkley (1997), Bootstrap Methods and their Application, section 3.8
               (hierarchical data): resample the top-level clusters (matches) with replacement and
               keep every row of a drawn cluster.  This is what paired() in
               run_model_comparison_v33.py does; the same generator, seed and draw order are used,
               so the first 300 replicates reproduce its published vs_lgbm_paper numbers exactly
               (checked below and written to the JSON).
  interval     Efron & Tibshirani (1993), An Introduction to the Bootstrap, chapter 13 (percentile
               interval), 95%.
  p value      Monte Carlo p, (1 + #{replicates on the far side of 0}) / (R + 1), doubled for two
               sides (Davison & Hinkley 1997, chapter 4); Holm (1979) step-down adjustment across
               the distinct comparisons in the file.
  AUC          Mann-Whitney statistic with ties scored 1/2 (Hanley & McNeil 1982), which is what
               sklearn.metrics.roc_auc_score computes.

Computation: a replicate draws n_matches match indices with replacement and a row's weight is the
number of times its match was drawn.  The weighted Mann-Whitney AUC equals roc_auc_score on the
concatenated resample exactly (checked on the first replicates; the gap is written out), and it
lets one sort per learner serve every replicate.

Nothing here touches training, selection or calibration: it reads frozen test-patch predictions.

    python scripts/paired_learner_cis_v33.py ^
        --out D:/LOL_Project/fusion_2615/features/tog_revision/A8-input-audit-and-cis/paired_learner_cis_v33.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

SEED = 7
ROOT = Path(__file__).resolve().parents[1]
FEATURES = Path("D:/LOL_Project/fusion_2615/features")
DEEP_NAMES = {"lightgbm": "lgbm_7106", "mlp": "mlp", "tabnet": "tabnet",
              "ft_transformer": "ft_transformer", "saint": "saint"}
DEEP_MODELS = ("tabnet", "ft_transformer", "saint")
HEADLINE = ("mlp->ft_transformer", "mlp->saint", "ft_transformer->saint", "mlp->tabnet",
            "lgbm_paper->ft_transformer", "lgbm_paper->mlp", "lgbm_paper->lgbm_deep",
            "lgbm_paper->lgbm_7106", "lgbm_paper->linear", "lead_only->lgbm_paper")
PRIOR_VERIFIER = {"mlp->ft_transformer": [0.0009, -0.0006, 0.0023], "mlp->saint": [-0.0006, -0.0024, 0.0008],
                  "lgbm_paper->ft_transformer": [-0.0105, -0.0117, -0.0092]}
DEVIATIONS = [
    "rows are weighted by match multiplicity instead of concatenating the resample; the AUC is "
    "identical (max gap on the checked replicates is written under validation)",
    "percentile interval, not BCa or studentised",
    "matches are the only clustering level; the same player or account across matches is not clustered",
    "each learner is one fitted model (one training seed): the intervals cover test-set sampling only, "
    "not training-seed variability",
    "intervals are per comparison and unadjusted; the Holm-adjusted p is the family-wise statement, with the "
    "family being every distinct learner pair in this file (the union of the five planned comparison sets, a "
    "pair listed in two sets counted once)",
    "p values are Monte Carlo from the same replicates, so the smallest attainable two-sided p is 2 / (R + 1)",
]
CAVEATS = [
    "tabnet: the published predictions subtract TabNet's sparsity term (loss - lambda_sparse * mask entropy; main "
    "repo scripts/run_deep_tabular_baselines.py line 430 at commit 60945ed and line 418 at 314ee78, the committed "
    "versions either side of the run), where Arik & Pfister 2021 (section 3, feature selection) add it; "
    "comparisons involving tabnet describe those predictions and will move after the corrected re-run",
    "the deep models received no hyperparameter search.  Parameter counts from the builders at commit 60945ed "
    "with the configurations in the JSONs: MLP 4,167,681; TabNet 4,075,989; FT-Transformer 285,857 (d_token 32, "
    "3 layers, 8 heads, 1,016 tokens); SAINT 459,713 (supervised only, no contrastive pre-training).  A tie "
    "between the MLP and an attention model is a statement about these configurations",
    "columns: lead_only 8; linear, lgbm_paper and lgbm_deep 6,164; lgbm_7106, mlp, tabnet, ft_transformer and "
    "saint 7,106.  The 942 extra columns are constant across the corpus (model_comparison_input_audit_extended.json)",
    "lgbm_7106 is the deep pipeline's LightGBM (2,000-tree cap, early-stopping patience 50; 60945ed lines 111-117) "
    "and lgbm_paper the published configuration (400-tree cap, patience 100; worktree "
    "scripts/run_model_comparison_v33.py lines 124-131); both learning rate 0.05 and 31 leaves",
    "the tree run's and the deep runs' learners were fitted in different processes; the audit shows their test "
    "rows, labels, matches and row order are identical, so the pairing is exact",
]


def sha1_of(path, chunk=1 << 22):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def file_info(path):
    st = path.stat()
    return {"sha1": sha1_of(path), "bytes": st.st_size,
            "mtime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(st.st_mtime))}


def git_state():
    def run(*cmd):
        try:
            return subprocess.run(["git", "-C", str(ROOT), *cmd], capture_output=True, text=True,
                                  timeout=30).stdout.strip()
        except Exception as exc:  # provenance must never abort the run
            return f"unavailable: {exc}"
    return {"repo": str(ROOT), "commit": run("rev-parse", "HEAD"),
            "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty_scripts": [ln[3:] for ln in run("status", "--porcelain").splitlines()
                              if ln[3:].startswith("scripts/")]}


def prepare(score, y):
    """Sort once per learner; the tie groups do not depend on the bootstrap weights."""
    order = np.argsort(score, kind="mergesort")
    s = score[order]
    starts = np.flatnonzero(np.r_[True, s[1:] != s[:-1]])
    return order, starts, y[order] == 1


def weighted_auc(prep, w):
    """Mann-Whitney AUC over weighted rows, ties scored 1/2."""
    order, starts, pos = prep
    ws = w[order]
    gp = np.add.reduceat(np.where(pos, ws, 0.0), starts)
    gn = np.add.reduceat(np.where(pos, 0.0, ws), starts)
    denom = gp.sum() * gn.sum()
    if denom == 0:
        return float("nan")
    return float((gp * (np.cumsum(gn) - 0.5 * gn)).sum() / denom)


def load_predictions(tree_json, tree_preds, deep_jsons, log):
    tree = json.loads(tree_json.read_text(encoding="utf-8"))
    with np.load(tree_preds, allow_pickle=False) as z:
        y, groups = z["y"].astype(np.int64), z["groups"]
        preds = {k: z[k].astype(np.float64) for k in z.files if k not in ("y", "groups", "classes", "present")}
    sources = {k: {"file": str(tree_preds), "run": "run_model_comparison_v33.py", "model_key": k,
                   "n_features": tree["models"][k]["n_features"],
                   "reported_auc": tree["models"][k]["overall_auc"]} for k in preds}
    label_keys, files = {tree["y_key"]}, {str(tree_json): file_info(tree_json), str(tree_preds): file_info(tree_preds)}
    split = (tree["split"]["train"], tree["split"]["val"], tree["split"]["test"],
             tree["split"]["rows"]["train"], tree["split"]["rows"]["val"], tree["split"]["rows"]["test"])
    for jp in deep_jsons:
        run = json.loads(jp.read_text(encoding="utf-8"))
        sp = run["split"]
        if (sp["train_patch"], sp["val_patch"], sp["test_patch"], sp["train"], sp["val"], sp["test"]) != split:
            raise SystemExit(f"{jp}: split {sp} differs from the tree run {split}")
        pp = jp.with_suffix(".preds.npz")
        with np.load(pp, allow_pickle=False) as z:
            if not (np.array_equal(z["y_test"].astype(np.int64), y) and np.array_equal(z["groups_test"], groups)):
                raise SystemExit(f"{pp}: y_test / groups_test differ from {tree_preds}; rows are not paired")
            for k in z.files:
                if not k.startswith("pred_"):
                    continue
                base = k[len("pred_"):]
                name = DEEP_NAMES.get(base, base)
                if name in preds:
                    raise SystemExit(f"{name} appears in two prediction files")
                preds[name] = z[k].astype(np.float64)
                sources[name] = {"file": str(pp), "run": "run_deep_tabular_baselines.py (--split patch)",
                                 "model_key": base, "n_features": run["n_features"],
                                 "reported_auc": run["models"][base]["test_auc"]}
        label_keys.add(sp["y_key"])
        files[str(jp)] = file_info(jp)
        files[str(pp)] = file_info(pp)
        log(f"  {pp.name}: rows paired with the tree run")
    if len(label_keys) != 1:
        raise SystemExit(f"label keys differ across runs: {label_keys}")
    return tree, y, groups, preds, sources, label_keys.pop(), files


def planned_pairs(names):
    """{(a, b): families}; the reported difference is AUC(b) - AUC(a)."""
    out = {}

    def add(a, b, family):
        if a in names and b in names and a != b:
            out.setdefault((a, b), []).append(family)
    for k in names:
        add("lgbm_paper", k, "every learner vs lgbm_paper")
    for k in DEEP_MODELS:
        add("mlp", k, "each deep model vs MLP")
    add("ft_transformer", "saint", "FT-Transformer vs SAINT")
    add("lgbm_paper", "lgbm_deep", "lgbm_deep vs lgbm_paper")
    for k in names:
        add("lead_only", k, "lead_only vs every learner")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tree-results", type=Path, default=FEATURES / "model_comparison_v33_patch.json")
    ap.add_argument("--tree-preds", type=Path, default=FEATURES / "model_comparison_v33_patch.preds.npz")
    ap.add_argument("--deep-results", default=",".join(str(FEATURES / f"deep_tabular_v33_patch_{t}.json")
                                                       for t in ("full", "ft", "saint")),
                    help="comma-separated deep JSONs; predictions are read from <json>.preds.npz")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--validate-reps", type=int, default=3,
                    help="replicates re-scored by concatenation + roc_auc_score")
    ap.add_argument("--out", type=Path,
                    default=FEATURES / "tog_revision" / "A8-input-audit-and-cis" / "paired_learner_cis_v33.json")
    a = ap.parse_args()
    log = lambda *s: print(*s, flush=True)
    started = time.time()
    if a.n_boot < 1000:
        raise SystemExit("--n-boot must be >= 1000 for the published intervals")

    deep_jsons = [Path(s.strip()) for s in a.deep_results.split(",") if s.strip()]
    tree, y, groups, preds, sources, y_key, files = load_predictions(a.tree_results, a.tree_preds, deep_jsons, log)
    names = list(preds)
    uniq, inv = np.unique(groups, return_inverse=True)
    log(f"rows {len(y):,}  matches {len(uniq):,}  positive {y.mean():.4f}  learners {names}")

    prep = {k: prepare(preds[k], y) for k in names}
    point = {k: float(roc_auc_score(y, preds[k])) for k in names}
    gap_point = max(abs(weighted_auc(prep[k], np.ones(len(y))) - point[k]) for k in names)
    gap_reported = max(abs(point[k] - sources[k]["reported_auc"]) for k in names)
    log(f"point AUCs reproduce the JSONs (max gap {gap_reported:.1e}); unit-weight AUC gap {gap_point:.1e}")

    rng = np.random.default_rng(a.seed)
    boot = np.empty((a.n_boot, len(names)))
    for b in range(a.n_boot):
        w = np.bincount(rng.integers(len(uniq), size=len(uniq)), minlength=len(uniq)).astype(np.float64)[inv]
        for j, k in enumerate(names):
            boot[b, j] = weighted_auc(prep[k], w)
        if (b + 1) % 250 == 0:
            log(f"  replicate {b + 1}/{a.n_boot} ({time.time() - started:.0f}s)")
    usable = ~np.isnan(boot).any(axis=1)
    col = {k: j for j, k in enumerate(names)}

    # the weighted computation must equal the concatenated resample paired() scores
    rng_v = np.random.default_rng(a.seed)
    rows = np.split(np.argsort(inv, kind="stable"), np.cumsum(np.bincount(inv))[:-1])
    concat_gap = 0.0
    for b in range(min(a.validate_reps, a.n_boot)):
        idx = np.concatenate([rows[i] for i in rng_v.integers(len(uniq), size=len(uniq))])
        concat_gap = max(concat_gap, max(abs(roc_auc_score(y[idx], preds[k][idx]) - boot[b, col[k]])
                                         for k in names))
    published = {}
    for k, v in (tree.get("vs_lgbm_paper") or {}).items():
        if k in col and v["n_boot"] <= a.n_boot and usable[:v["n_boot"]].all():
            d = boot[:v["n_boot"], col[k]] - boot[:v["n_boot"], col["lgbm_paper"]]
            published[k] = {"published": [v["mean"], v["lo"], v["hi"]],
                            "recomputed": [float(d.mean()), *map(float, np.percentile(d, [2.5, 97.5]))]}
    repro_gap = max((abs(p - q) for v in published.values() for p, q in zip(v["published"], v["recomputed"])),
                    default=float("nan"))
    log(f"concatenation gap {concat_gap:.1e}; published vs_lgbm_paper reproduced to {repro_gap:.1e}")

    B = boot[usable]
    comparisons = {}
    for (ka, kb), families in planned_pairs(names).items():
        d = B[:, col[kb]] - B[:, col[ka]]
        lo, hi = (float(x) for x in np.percentile(d, [2.5, 97.5]))
        far = min(int((d <= 0).sum()), int((d >= 0).sum()))
        comparisons[f"{ka}->{kb}"] = {
            "a": ka, "b": kb, "families": families, "difference": "AUC(b) - AUC(a)",
            "diff": point[kb] - point[ka], "boot_mean": float(d.mean()), "boot_se": float(d.std(ddof=1)),
            "ci95": [lo, hi], "excludes_zero": bool(lo > 0 or hi < 0),
            "p_two_sided": min(1.0, 2 * (1 + far) / (len(d) + 1)),
            "reading": "tie: the 95% interval contains 0" if lo <= 0 <= hi else f"{kb if lo > 0 else ka} higher",
            "n_matches": int(len(uniq)), "n_rows": int(len(y))}
    by_pair = {}
    for key, c in comparisons.items():
        by_pair.setdefault(frozenset((c["a"], c["b"])), []).append(key)
    ordered = sorted(by_pair.items(), key=lambda kv: comparisons[kv[1][0]]["p_two_sided"])
    running = 0.0
    for i, (_, keys) in enumerate(ordered):
        running = max(running, min(1.0, (len(ordered) - i) * comparisons[keys[0]]["p_two_sided"]))
        for key in keys:
            comparisons[key]["p_holm"] = running

    learners = {k: {"test_auc": point[k], "ci95": [float(x) for x in np.percentile(B[:, col[k]], [2.5, 97.5])],
                    **sources[k]} for k in sorted(names, key=lambda k: -point[k])}
    headline = {k: {"diff": round(comparisons[k]["diff"], 4),
                    "ci95": [round(x, 4) for x in comparisons[k]["ci95"]],
                    "reading": comparisons[k]["reading"]} for k in HEADLINE if k in comparisons}
    manifest = tree.get("corpus_manifest", {})
    payload = {
        "item": "A8-input-audit-and-cis",
        "method": "match-clustered paired percentile bootstrap (Davison & Hinkley 1997 s3.8; "
                  "Efron & Tibshirani 1993 ch13); Holm 1979 across distinct comparisons",
        "headline": headline, "learners": learners, "comparisons": comparisons,
        "validation": {"max_gap_point_auc_vs_reported_json": gap_reported,
                       "max_gap_unit_weight_auc_vs_sklearn": gap_point,
                       "max_gap_weighted_vs_concatenated_resample": concat_gap,
                       "replicates_checked_by_concatenation": int(min(a.validate_reps, a.n_boot)),
                       "published_vs_lgbm_paper_reproduction": published,
                       "max_gap_published_vs_lgbm_paper": repro_gap,
                       "prior_verifier_values_diff_lo_hi": PRIOR_VERIFIER},
        "deviations": DEVIATIONS, "caveats": CAVEATS,
        "provenance": {"script": str(Path(__file__).resolve()), "git": git_state(), "argv": sys.argv[1:],
                       "preset": os.environ.get("LOL_CFG_PRESET", "unset (corpus manifest below)"),
                       "corpus_run_id": manifest.get("run_id"), "corpus_git_commit": manifest.get("git_commit"),
                       "detector": manifest.get("detector"), "label_key": y_key,
                       "split": {"kind": "patch holdout", "train": tree["split"]["train"],
                                 "val": tree["split"]["val"], "test": tree["split"]["test"],
                                 "rows": tree["split"]["rows"]},
                       "n_rows_test": int(len(y)), "n_matches_test": int(len(uniq)),
                       "positive_rate_test": float(y.mean()), "seed": a.seed,
                       "n_boot": a.n_boot, "n_boot_usable": int(usable.sum()), "input_files": files,
                       "script_sha1": sha1_of(Path(__file__).resolve()),
                       "input_audit": "model_comparison_input_audit_extended.json (same folder)",
                       "wall_clock_s": round(time.time() - started, 1),
                       "finished": time.strftime("%Y-%m-%d %H:%M:%S")},
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log("\n" + "\n".join(f"  {k:28s} {v['diff']:+.4f} [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}]  {v['reading']}"
                         for k, v in headline.items()))
    log(f"wrote {a.out} ({time.time() - started:.0f}s)")


if __name__ == "__main__":
    main()
