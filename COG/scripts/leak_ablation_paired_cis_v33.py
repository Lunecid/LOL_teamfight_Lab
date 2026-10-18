"""Paired match-clustered bootstrap intervals for the input-leak ablation (v3.3).

Input: the per-configuration out-of-fold predictions written by

    python scripts/run_leak_ablation.py ... --output <rerun.json> --save-predictions <DIR>

(DIR/pred_<config>.npz + DIR/manifest.json).  The four configurations score the same engagements with
the same labels and the same match-grouped folds, so every difference between them is a paired
difference: one bootstrap replicate draws matches with replacement (a match keeps all its engagements;
engagements of one match share the game state and are not independent) and every statistic of every
configuration is recomputed on that one draw.  An interval for a difference is the 2.5th-97.5th
percentile of the replicate differences (percentile bootstrap; Efron and Tibshirani 1993).

Statistics, per configuration (the values leak_ablation_v33.json reports):
  auc_all, auc_pick, auc_skirmish, auc_teamfight   out-of-fold AUC overall and by participation scale
                                                    (smaller side <= 1 pick, 2-3 skirmish, >= 4 teamfight)
  pick_minus_teamfight                             auc_pick - auc_teamfight
Differences:
  every pair of configurations, later minus earlier in the order clean, time_leak, anchor_leak,
  both_leaks, for all five statistics; the pairs X - clean are the
  numbers in docs/tog_manuscript/sec_prediction.tex (tab:pred-leak and the text below it), and the
  pick_minus_teamfight row of a pair is the change of the class gap (a difference of differences);
  the two-switch interaction (both_leaks - anchor_leak) - (time_leak - clean) for all five statistics.

Checks written to the output (the script aborts on the first three):
  rows      the four prediction files hold identical row keys, labels, groups, classes and folds;
  points    AUCs recomputed from the predictions equal the run's own JSON (--ablation-json);
  weights   the weighted AUC of replicate 0 equals roc_auc_score on the explicitly resampled rows;
  reproduction  the re-run's point values against the published leak_ablation_v33.json (--published) and
            the re-run's per-fold AUCs against the published log (--published-log, 4 decimals).  A
            mismatch is reported, not fatal: the intervals then describe the re-run.

    python scripts/leak_ablation_paired_cis_v33.py \\
        --pred-dir D:/LOL_Project/fusion_2615/features/tog_revision/C3-leak-ablation/predictions \\
        --ablation-json D:/LOL_Project/fusion_2615/features/tog_revision/C3-leak-ablation/leak_ablation_v33_rerun.json \\
        --published D:/LOL_Project/fusion_2615/features/leak_ablation_v33.json \\
        --published-log D:/LOL_Project/fusion_2615/features/leak_ablation_v33.log \\
        --output D:/LOL_Project/fusion_2615/features/tog_revision/C3-leak-ablation/leak_ablation_paired_cis_v33.json
"""
from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):   # numpy only; one thread is plenty
    os.environ.setdefault(_k, "1")

import argparse
import hashlib
import itertools
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

CONFIGS = ("clean", "time_leak", "anchor_leak", "both_leaks")      # run_leak_ablation.CONFIGS order
CLASSES = ("pick", "skirmish", "teamfight")                        # run_leak_ablation.scale_class
MIN_CLASS_N = 50               # run_leak_ablation: a class is scored when n > 50 and both labels occur
STATS = ("auc_all", "auc_pick", "auc_skirmish", "auc_teamfight", "pick_minus_teamfight")
KEY_ARRAYS = ("match_id", "t_start_ts", "groups", "y", "scale_class", "fold")
PRIMARY_PAIRS = (("time_leak", "clean"), ("anchor_leak", "clean"), ("both_leaks", "clean"))
SWITCHES = {   # the same leak switched on with the other input held fixed
    "time_switch_at_causal_anchors": ("time_leak", "clean"),
    "time_switch_at_leaky_anchors": ("both_leaks", "anchor_leak"),
    "anchor_switch_at_absolute_time": ("anchor_leak", "clean"),
    "anchor_switch_at_leaky_time": ("both_leaks", "time_leak"),
}
POINT_TOL = 1e-9               # recomputed vs the run's own JSON (same predictions)
REPRO_EXACT_TOL = 1e-9         # re-run vs published JSON
LOG_TOL = 5e-5 + 1e-9          # the log prints fold AUCs with 4 decimals


# ------------------------------------------------------------------ weighted AUC (ties count one half)

class AucPlan:
    """Sort once; AUC under any non-negative row weights is then O(n) (Mann-Whitney with tie blocks)."""

    def __init__(self, y: np.ndarray, score: np.ndarray, rows: np.ndarray):
        self.rows = rows
        s = np.asarray(score, dtype=np.float64)[rows]
        order = np.argsort(s, kind="mergesort")
        ss = s[order]
        new = np.ones(len(ss), dtype=bool)
        new[1:] = ss[1:] != ss[:-1]
        self.block = np.cumsum(new) - 1
        self.n_blocks = int(self.block[-1]) + 1 if len(ss) else 0
        self.order = rows[order]
        self.pos = np.asarray(y)[self.order] == 1

    def auc(self, w_rows: np.ndarray | None) -> float:
        w = np.ones(len(self.order)) if w_rows is None else w_rows[self.order]
        wp = np.bincount(self.block, weights=w * self.pos, minlength=self.n_blocks)
        wn = np.bincount(self.block, weights=w * ~self.pos, minlength=self.n_blocks)
        P, N = float(wp.sum()), float(wn.sum())
        if P <= 0.0 or N <= 0.0:
            return float("nan")
        below = np.cumsum(wn) - wn
        return float(np.dot(wp, below + 0.5 * wn) / (P * N))


def stats_from(aucs: dict) -> dict:
    """aucs: config -> {all, pick, skirmish, teamfight} (value or None) -> config -> STATS."""
    out = {}
    for c, a in aucs.items():
        rec = {"auc_all": a.get("all")}
        for k in CLASSES:
            rec[f"auc_{k}"] = a.get(k)
        p, t = a.get("pick"), a.get("teamfight")
        rec["pick_minus_teamfight"] = (p - t) if (p is not None and t is not None) else None
        out[c] = rec
    return out


def contrasts(stats: dict, configs) -> dict:
    """Every pair (later minus earlier in CONFIGS order) and the interaction, per statistic (None when a side is None)."""
    def sub(x, y):
        return None if (x is None or y is None) else x - y
    out = {}
    for b, a in itertools.combinations(configs, 2):
        out[f"{a} - {b}"] = {s: sub(stats[a][s], stats[b][s]) for s in STATS}
    if all(c in configs for c in CONFIGS):
        out["interaction"] = {s: sub(sub(stats["both_leaks"][s], stats["anchor_leak"][s]),
                                     sub(stats["time_leak"][s], stats["clean"][s])) for s in STATS}
    return out


def summarize(values: list, point, alpha: float) -> dict:
    arr = np.asarray([v for v in values if v is not None and v == v], dtype=np.float64)
    rec = {"point": point}
    if not arr.size or point is None:
        return {**rec, "ci_lo": None, "ci_hi": None, "n_boot": int(arr.size)}
    lo, hi = np.percentile(arr, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    rec.update(ci_lo=float(lo), ci_hi=float(hi), boot_mean=float(arr.mean()), boot_se=float(arr.std(ddof=1)) if arr.size > 1 else None,
               share_leq_0=float((arr <= 0).mean()), share_geq_0=float((arr >= 0).mean()),
               excludes_zero=bool(lo > 0 or hi < 0), n_boot=int(arr.size))
    return rec


# ------------------------------------------------------------------ inputs and checks

def load_predictions(pred_dir: Path, configs) -> tuple[dict, dict, dict]:
    manifest_path = pred_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    data, hashes = {}, {}
    for c in configs:
        path = pred_dir / f"pred_{c}.npz"
        if not path.exists():
            raise FileNotFoundError(f"{path} missing (run scripts/run_leak_ablation.py with --save-predictions)")
        h = hashlib.sha256(path.read_bytes()).hexdigest()
        want = ((manifest.get("files") or {}).get(c) or {}).get("sha256")
        if want and want != h:
            raise ValueError(f"{path}: sha256 {h} differs from manifest {want}")
        with np.load(path, allow_pickle=False) as z:
            data[c] = {k: z[k] for k in z.files}
        hashes[c] = h
    return data, manifest, hashes


def check_rows(data: dict, configs) -> dict:
    ref = data[configs[0]]
    for c in configs:
        for k in KEY_ARRAYS:
            if not np.array_equal(data[c][k], ref[k]):
                raise AssertionError(f"pred_{c}.npz: '{k}' differs from pred_{configs[0]}.npz; rows are not paired")
        if not np.isfinite(data[c]["pred"]).all():
            raise AssertionError(f"pred_{c}.npz: non-finite predictions")
    y = ref["y"]
    if not set(np.unique(y).tolist()) <= {0, 1}:
        raise AssertionError(f"labels outside {{0, 1}}: {np.unique(y)}")
    if not np.array_equal(ref["groups"], ref["match_id"]):
        raise AssertionError("groups are not the match ids")
    # fold purity: every match lies in exactly one fold (min fold == max fold per match)
    _, inv = np.unique(ref["groups"], return_inverse=True)
    fmin = np.full(inv.max() + 1, 127, dtype=np.int16); fmax = np.full(inv.max() + 1, -1, dtype=np.int16)
    np.minimum.at(fmin, inv, ref["fold"].astype(np.int16)); np.maximum.at(fmax, inv, ref["fold"].astype(np.int16))
    if not np.array_equal(fmin, fmax):
        raise AssertionError("a match spans more than one fold")
    return {"status": "pass", "n_rows": int(len(y)), "n_matches": int(inv.max() + 1), "positive_share": float(y.mean()),
            "class_n": {k: int((ref["scale_class"] == k).sum()) for k in CLASSES},
            "fold_sizes": [int((ref["fold"] == k).sum()) for k in range(int(ref["fold"].max()) + 1)],
            "identical_across_configs": list(KEY_ARRAYS), "matches_within_one_fold": True}


def class_rows(data: dict, configs) -> dict:
    ref = data[configs[0]]
    y, cls = ref["y"], ref["scale_class"]
    rows = {"all": np.arange(len(y))}
    for k in CLASSES:
        idx = np.flatnonzero(cls == k)
        if len(idx) > MIN_CLASS_N and len(np.unique(y[idx])) > 1:      # run_leak_ablation's rule
            rows[k] = idx
    return rows


def point_aucs(data: dict, rows: dict, configs) -> dict:
    y = data[configs[0]]["y"]
    return {c: {k: float(roc_auc_score(y[idx], data[c]["pred"][idx])) for k, idx in rows.items()} for c in configs}


def json_stats(doc: dict, configs) -> dict:
    """leak_ablation JSON -> config -> STATS (None where the JSON has no value)."""
    out = {}
    for c in configs:
        rec = (doc.get("configs") or {}).get(c)
        if rec is None:
            continue
        a = {"all": rec.get("auc")}
        for k in CLASSES:
            if k in (rec.get("by_class") or {}):
                a[k] = rec["by_class"][k]["auc"]
        out[c] = stats_from({c: a})[c]
        out[c]["_pick_minus_teamfight_stored"] = rec.get("pick_minus_teamfight")
        out[c]["_n_rows"] = rec.get("n_rows")
        out[c]["_class_n"] = {k: v.get("n") for k, v in (rec.get("by_class") or {}).items()}
    return out


def compare(ours: dict, theirs: dict, configs, tol: float) -> dict:
    diffs, rows = [], {}
    for c in configs:
        if c not in theirs:
            rows[c] = "absent"
            continue
        rows[c] = {}
        for s in STATS:
            a, b = ours[c][s], theirs[c][s]
            if a is None or b is None:
                rows[c][s] = {"ours": a, "theirs": b, "abs_diff": None}
                continue
            d = abs(a - b)
            diffs.append(d)
            rows[c][s] = {"ours": a, "theirs": b, "abs_diff": d}
    mx = max(diffs) if diffs else None
    return {"max_abs_diff": mx, "tolerance": tol, "status": ("absent" if mx is None else ("identical" if mx <= tol else "differs")),
            "by_config": rows}


def parse_log_folds(path: Path) -> dict:
    """run_leak_ablation log -> config -> [fold AUCs]: fold lines precede their '[config] rows ...' line."""
    pending, out = [], {}
    fold_re = re.compile(r"fold (\d+)/(\d+): train=(\d+) test=(\d+) auc=([0-9.]+)")
    cfg_re = re.compile(r"^\[(\w+)\s*\] rows")
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = fold_re.search(line)
        if m:
            pending.append({"fold": int(m.group(1)), "train": int(m.group(3)), "test": int(m.group(4)), "auc": float(m.group(5))})
            continue
        m = cfg_re.match(line.strip())
        if m:
            out[m.group(1)] = pending
            pending = []
    return out


def fold_check(data: dict, configs, log_folds: dict) -> dict:
    y, fold = data[configs[0]]["y"], data[configs[0]]["fold"]
    n_folds = int(fold.max()) + 1
    res, worst, ok = {}, 0.0, True
    for c in configs:
        ours = [float(roc_auc_score(y[fold == k], data[c]["pred"][fold == k])) for k in range(n_folds)]
        theirs = log_folds.get(c)
        rec = {"rerun_fold_auc": ours, "rerun_fold_test_n": [int((fold == k).sum()) for k in range(n_folds)]}
        if theirs and len(theirs) == n_folds:
            # GroupKFold yields fold k as its k-th split; the log prints them in that order
            d = [abs(a - t["auc"]) for a, t in zip(ours, theirs)]
            sizes_ok = [t["test"] for t in theirs] == rec["rerun_fold_test_n"]
            rec.update(published_fold_auc=[t["auc"] for t in theirs], published_fold_test_n=[t["test"] for t in theirs],
                       max_abs_diff=max(d), fold_sizes_equal=sizes_ok, match=bool(max(d) <= LOG_TOL and sizes_ok))
            worst = max(worst, max(d)); ok = ok and rec["match"]
        else:
            rec["published_fold_auc"] = None; ok = False
        res[c] = rec
    return {"tolerance": LOG_TOL, "all_match": ok, "max_abs_diff": worst, "by_config": res}


# ------------------------------------------------------------------ bootstrap

def bootstrap(data: dict, rows: dict, configs, n_boot: int, seed: int, alpha: float, log) -> dict:
    y = data[configs[0]]["y"]
    groups = data[configs[0]]["groups"]
    _, inverse = np.unique(groups, return_inverse=True)
    n_groups = int(inverse.max()) + 1
    plans = {c: {k: AucPlan(y, data[c]["pred"], idx) for k, idx in rows.items()} for c in configs}

    point = stats_from({c: {k: plans[c][k].auc(None) for k in rows} for c in configs})
    point_diff = contrasts(point, configs)

    rng = np.random.default_rng(seed)
    draws = {c: {s: [] for s in STATS} for c in configs}
    ddraws = {name: {s: [] for s in STATS} for name in point_diff}
    skipped, weight_check = 0, None
    t0 = time.time()
    for b in range(int(n_boot)):
        counts = np.bincount(rng.integers(0, n_groups, size=n_groups), minlength=n_groups)
        w = counts[inverse].astype(np.float64)
        rep = {c: {k: plans[c][k].auc(w) for k in rows} for c in configs}
        if b == 0:   # the weighted AUC equals roc_auc_score on the explicitly resampled rows (overall and one class)
            weight_check = []
            for c, k in ((configs[0], "all"), (configs[-1], [x for x in rows if x != "all"][-1] if len(rows) > 1 else "all")):
                sub = rows[k]
                idx = np.repeat(sub, w[sub].astype(np.int64))
                ref = float(roc_auc_score(y[idx], data[c]["pred"][idx]))
                rec = {"replicate": 0, "config": c, "rows": k, "weighted_auc": rep[c][k], "roc_auc_score_resampled": ref,
                       "abs_diff": abs(ref - rep[c][k])}
                weight_check.append(rec)
                if not rec["abs_diff"] <= 1e-10:
                    raise AssertionError(f"weighted AUC check failed: {rec}")
        if any(v != v for r in rep.values() for v in r.values()):
            skipped += 1                       # a class lost one label in this draw; drop the whole replicate
            continue
        st = stats_from(rep)
        for c in configs:
            for s in STATS:
                draws[c][s].append(st[c][s])
        for name, rec in contrasts(st, configs).items():
            for s in STATS:
                ddraws[name][s].append(rec[s])
        if (b + 1) % 500 == 0:
            log(f"  bootstrap {b + 1}/{n_boot} ({time.time() - t0:.0f}s)")

    per_config = {c: {s: summarize(draws[c][s], point[c][s], alpha) for s in STATS} for c in configs}
    diffs = {name: {s: summarize(ddraws[name][s], point_diff[name][s], alpha) for s in STATS} for name in point_diff}
    return {"method": "match-clustered paired percentile bootstrap: one draw of matches (with replacement) per replicate, "
                      "shared by every configuration, class and statistic; AUC ties count one half",
            "n_boot_requested": int(n_boot), "n_boot_used": int(n_boot) - skipped, "n_boot_skipped_single_label_class": skipped,
            "seed": int(seed), "alpha": alpha, "rng": "numpy.random.default_rng(seed).integers(0, n_matches, n_matches) per replicate",
            "n_matches": n_groups, "weight_check": weight_check, "per_config": per_config, "differences": diffs,
            "bootstrap_seconds": round(time.time() - t0, 1)}


# ------------------------------------------------------------------ main

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred-dir", type=Path, required=True, help="DIR given to run_leak_ablation.py --save-predictions")
    ap.add_argument("--ablation-json", type=Path, default=None, help="the JSON written by the same run (point check)")
    ap.add_argument("--published", type=Path, default=None, help="published leak_ablation_v33.json (reproduction check)")
    ap.add_argument("--published-log", type=Path, default=None, help="published leak_ablation_v33.log (fold AUC check)")
    ap.add_argument("--configs", default=",".join(CONFIGS))
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args(argv)
    started = time.time()

    def log(msg):
        print(msg, flush=True)

    requested = [c.strip() for c in args.configs.split(",") if c.strip()]
    unknown = [c for c in requested if c not in CONFIGS]
    if unknown:
        raise SystemExit(f"unknown configs {unknown}; known {CONFIGS}")
    configs = [c for c in CONFIGS if c in requested]      # canonical order fixes the sign of every difference
    data, manifest, hashes = load_predictions(args.pred_dir, configs)
    rows_check = check_rows(data, configs)
    log(f"rows: {rows_check['n_rows']:,} engagements, {rows_check['n_matches']:,} matches, classes {rows_check['class_n']}, "
        f"folds {rows_check['fold_sizes']}")
    rows = class_rows(data, configs)

    points = stats_from(point_aucs(data, rows, configs))
    out = {"item": "C3-leak-ablation-cis", "script": "scripts/leak_ablation_paired_cis_v33.py", "argv": sys.argv,
           "inputs": {"pred_dir": str(args.pred_dir), "pred_sha256": hashes,
                      "manifest": {k: manifest.get(k) for k in ("schema", "argv", "n_matches", "n_refs", "match_ids_sha256",
                                                               "cache_dir", "versions", "git", "env", "started", "finished",
                                                               "output_json")},
                      "ablation_json": str(args.ablation_json) if args.ablation_json else None,
                      "published": str(args.published) if args.published else None,
                      "published_log": str(args.published_log) if args.published_log else None},
           "configs": configs, "classes_scored": [k for k in rows if k != "all"], "checks": {"rows": rows_check}}

    if args.ablation_json:
        own = json_stats(json.loads(args.ablation_json.read_text(encoding="utf-8")), configs)
        chk = compare(points, own, configs, POINT_TOL)
        for c in configs:   # counts and the stored gap must agree too
            if c in own:
                if own[c]["_n_rows"] != rows_check["n_rows"]:
                    raise AssertionError(f"{c}: n_rows {own[c]['_n_rows']} in JSON vs {rows_check['n_rows']} saved")
                if any(own[c]["_class_n"].get(k) != int((data[c]["scale_class"] == k).sum()) for k in own[c]["_class_n"]):
                    raise AssertionError(f"{c}: class counts differ between JSON and predictions")
                st = own[c]["_pick_minus_teamfight_stored"]
                if st is not None and abs(st - own[c]["pick_minus_teamfight"]) > POINT_TOL:
                    raise AssertionError(f"{c}: stored pick_minus_teamfight inconsistent")
        if chk["status"] != "identical":
            raise AssertionError(f"predictions do not reproduce {args.ablation_json}: max |diff| {chk['max_abs_diff']}")
        out["checks"]["points_vs_run_json"] = chk
        log(f"points vs run JSON: {chk['status']} (max |diff| {chk['max_abs_diff']:.2e})")

    if args.published:
        pub_doc = json.loads(args.published.read_text(encoding="utf-8"))
        pub = json_stats(pub_doc, configs)
        chk = compare(points, pub, configs, REPRO_EXACT_TOL)
        chk["n_rows_equal"] = all(pub.get(c, {}).get("_n_rows") == rows_check["n_rows"] for c in configs)
        chk["n_matches_equal"] = pub_doc.get("n_matches") == manifest.get("n_matches")
        chk["n_refs_equal"] = pub_doc.get("n_refs") == manifest.get("n_refs")
        chk["class_n_equal"] = all(pub.get(c, {}).get("_class_n") == {k: int((data[c]["scale_class"] == k).sum())
                                                                     for k in pub.get(c, {}).get("_class_n", {})} for c in configs)
        out["checks"]["reproduction_vs_published"] = chk
        log(f"re-run vs published JSON: {chk['status']} (max |diff| {chk['max_abs_diff']}); rows equal {chk['n_rows_equal']}, "
            f"refs equal {chk['n_refs_equal']}, class n equal {chk['class_n_equal']}")
    if args.published_log:
        chk = fold_check(data, configs, parse_log_folds(args.published_log))
        out["checks"]["folds_vs_published_log"] = chk
        log(f"fold AUCs vs published log: all match {chk['all_match']} (max |diff| {chk['max_abs_diff']:.2e})")
    repro = out["checks"].get("reproduction_vs_published", {}).get("status")
    out["interpretation"] = (
        "The intervals are computed on the re-run's predictions.  "
        + ("The re-run reproduces the published point values, so the intervals apply to the published table."
           if repro == "identical" else
           "The re-run does not reproduce the published point values exactly (see checks.reproduction_vs_published); "
           "report the re-run's point values with these intervals, or state both." if repro == "differs" else
           "No published JSON was compared."))

    log(f"bootstrap: {args.n_boot} replicates, seed {args.seed}")
    boot = bootstrap(data, rows, configs, args.n_boot, args.seed, args.alpha, log)
    for c in configs:   # the bootstrap's unweighted AUCs must equal roc_auc_score
        for s in STATS:
            a, b = boot["per_config"][c][s]["point"], points[c][s]
            if (a is None) != (b is None) or (a is not None and abs(a - b) > POINT_TOL):
                raise AssertionError(f"tie-aware AUC {a} != roc_auc_score {b} for {c}/{s}")
    out["bootstrap"] = boot
    if all(c in configs for c in CONFIGS):
        out["switch_effects"] = {name: {"pair": f"{a} - {b}", **boot["differences"][f"{a} - {b}"]} for name, (a, b) in SWITCHES.items()}
        out["switch_effects"]["interaction"] = {"pair": "(both_leaks - anchor_leak) - (time_leak - clean)",
                                                **boot["differences"]["interaction"]}
    out["manuscript_map"] = {   # docs/tog_manuscript/sec_prediction.tex, tab:pred-leak and the paragraph that reads it
        "table_rows": {c: f"bootstrap.per_config.{c}.<stat>" for c in configs},
        "leak_effects_vs_clean": {f"{a}": f"bootstrap.differences['{a} - {b}'].<stat>" for a, b in PRIMARY_PAIRS if a in configs and b in configs},
        "gap_change_vs_clean": {f"{a}": f"bootstrap.differences['{a} - {b}'].pick_minus_teamfight" for a, b in PRIMARY_PAIRS
                                if a in configs and b in configs},
        "stats": list(STATS)}
    out["seconds"] = round(time.time() - started, 1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(".tmp.json")
    tmp.write_text(json.dumps(out, indent=1), encoding="utf-8")
    os.replace(tmp, args.output)

    for a, b in PRIMARY_PAIRS:
        if a in configs and b in configs:
            d = boot["differences"][f"{a} - {b}"]
            log(f"{a} - {b}: " + "; ".join(f"{s} {d[s]['point']:+.4f} [{d[s]['ci_lo']:+.4f}, {d[s]['ci_hi']:+.4f}]"
                                           for s in STATS if d[s]["point"] is not None))
    for c in configs:
        g = boot["per_config"][c]["pick_minus_teamfight"]
        if g["point"] is not None:
            log(f"{c}: pick_minus_teamfight {g['point']:+.4f} [{g['ci_lo']:+.4f}, {g['ci_hi']:+.4f}]")
    log(f"wrote {args.output} ({out['seconds']}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
